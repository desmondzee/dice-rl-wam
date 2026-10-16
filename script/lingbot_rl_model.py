import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

STATE_DIM = 3072
HORIZON = 16
ACTION_DIM = 30
USED_DOF = 7
HIDDEN = (1024, 1024, 1024)
ENSEMBLE = 10
BETA = 100.0
EPSILON = -0.5
GAMMA = 0.99
N_STEP_CHUNKS = 3
GRADIENT_STEPS = 10
UPDATE_EVERY_CHUNKS = 4
ACTOR_EVERY = 2
TAU = 0.01
ADAM_LR = 1e-4
WEIGHT_DECAY = 1e-5
MAX_GRAD_NORM = 1.0
LR_CYCLE = 1000
LR_WARMUP = 10
LR_MIN = 1e-6
BATCH = 256
REPLAY_CAPACITY = 100_000
K_CANDIDATES = 4
PROPRIO_DIM = 8


def mlp_float(tensor):
    if not torch.is_tensor(tensor):
        tensor = torch.as_tensor(tensor)
    return tensor.float()


def mask_unused_dof(chunk):
    out = mlp_float(chunk).clone()
    out[..., USED_DOF:] = 0
    return out


def apply_residual(a_base, residual):
    return mask_unused_dof(mlp_float(a_base) + mlp_float(residual))


def cosine_restart_lr(step):
    position = step % LR_CYCLE
    if position < LR_WARMUP:
        scale = position / LR_WARMUP
    else:
        scale = 0.5 * (1.0 + math.cos(math.pi * (position - LR_WARMUP) / (LR_CYCLE - LR_WARMUP)))
    return (LR_MIN + (ADAM_LR - LR_MIN) * scale) / ADAM_LR


def _mlp(in_dim, out_dim):
    dims = [in_dim, *HIDDEN, out_dim]
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


RESIDUAL_INPUTS = {"z": ("z",), "base": ("base",), "z_base": ("z", "base")}


class ResidualActor(nn.Module):
    def __init__(self, residual_input="z", state_dim=STATE_DIM):
        super().__init__()
        self.inputs = RESIDUAL_INPUTS[residual_input]
        self.net = _mlp(state_dim + len(self.inputs) * HORIZON * ACTION_DIM, HORIZON * ACTION_DIM)
        final = self.net[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, state, noise, a_base):
        state = mlp_float(state)
        batch = state.shape[0]
        parts = [state] + [mlp_float(noise if name == "z" else a_base).reshape(batch, -1) for name in self.inputs]
        residual = self.net(torch.cat(parts, dim=-1))
        return mask_unused_dof(residual.reshape(batch, HORIZON, ACTION_DIM))


class CriticEnsemble(nn.Module):
    def __init__(self, state_dim=STATE_DIM):
        super().__init__()
        self.heads = nn.ModuleList([_mlp(state_dim + HORIZON * ACTION_DIM, 1) for _ in range(ENSEMBLE)])

    def forward(self, state, action, return_all=False):
        state = mlp_float(state)
        batch = state.shape[0]
        x = torch.cat([state, mask_unused_dof(action).reshape(batch, -1)], dim=-1)
        qs = [head(x) for head in self.heads]
        if return_all:
            return qs
        return torch.min(torch.stack(qs, dim=0), dim=0).values


class PlanCritic(nn.Module):
    def __init__(self, state_dim=STATE_DIM, plan_dim=STATE_DIM):
        super().__init__()
        self.heads = nn.ModuleList([_mlp(state_dim + plan_dim, 1) for _ in range(ENSEMBLE)])

    def forward(self, state, pool, return_all=False):
        x = torch.cat([mlp_float(state), mlp_float(pool)], dim=-1)
        vs = [head(x) for head in self.heads]
        if return_all:
            return vs
        return torch.min(torch.stack(vs, dim=0), dim=0).values


class DiceResidualModel:
    def __init__(self, device="cpu", residual_input="z", epsilon=EPSILON, plan_candidates=1, state_features="pool", proprio=False,
                 plan_target="q"):
        self.device = device
        self.epsilon = epsilon
        self.plans = int(plan_candidates)
        self.plan_target = plan_target
        self.features = state_features
        self.proprio = bool(proprio)
        self.state_dim = STATE_DIM * (2 if state_features == "struct" else 1) + (PROPRIO_DIM if self.proprio else 0)
        self.plan_dim = STATE_DIM * (8 if state_features == "struct" else 1) if self.plans > 1 else 0
        self.actor = ResidualActor(residual_input, self.state_dim).to(device)
        self.critic = CriticEnsemble(self.state_dim + self.plan_dim).to(device)
        self.target_critic = copy.deepcopy(self.critic).to(device)
        for parameter in self.target_critic.parameters():
            parameter.requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=ADAM_LR, weight_decay=WEIGHT_DECAY)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=ADAM_LR, weight_decay=WEIGHT_DECAY)
        self.actor_lr = torch.optim.lr_scheduler.LambdaLR(self.actor_opt, cosine_restart_lr)
        self.critic_lr = torch.optim.lr_scheduler.LambdaLR(self.critic_opt, cosine_restart_lr)
        self.plan_critic = self.plan_opt = self.plan_lr = None
        if self.plans > 1:
            self.plan_critic = PlanCritic(self.state_dim, self.plan_dim).to(device)
            self.plan_opt = torch.optim.Adam(self.plan_critic.parameters(), lr=ADAM_LR, weight_decay=WEIGHT_DECAY)
            self.plan_lr = torch.optim.lr_scheduler.LambdaLR(self.plan_opt, cosine_restart_lr)

    def state_input(self, pool, struct, proprio):
        if self.features == "pool":
            state = mlp_float(pool)
        else:
            struct = mlp_float(struct)
            state = struct.reshape(struct.shape[0], -1)
        if not self.proprio:
            return state
        if proprio is None:
            raise ValueError("proprio is required by this recipe")
        return torch.cat([state, mlp_float(proprio).to(state.device)], dim=-1)

    def plan_input(self, pool, struct):
        if pool is None:
            return None
        if self.features == "pool":
            return mlp_float(pool)
        struct = mlp_float(struct)
        return struct.reshape(*struct.shape[:-2], -1)

    @staticmethod
    def layout_of(payload):
        layout = payload.get("layout", {})
        out = {"state_features": layout.get("state_features", "pool"), "proprio": bool(layout.get("proprio", False))}
        if "plan_target" in layout:
            out["plan_target"] = layout["plan_target"]
        return out

    def critic_state(self, state, pool):
        if self.plans == 1:
            return mlp_float(state)
        return torch.cat([mlp_float(state), mlp_float(pool)], dim=-1)

    def rank_candidates(self, state, pools, a_base_all, z_all):
        with torch.no_grad():
            a_base_all, z_all = mlp_float(a_base_all), mlp_float(z_all)
            plans, k = a_base_all.shape[:2]
            flat_state = mlp_float(state).expand(plans * k, -1)
            flat_pool = mlp_float(pools).repeat_interleave(k, dim=0)
            flat_base = a_base_all.reshape(plans * k, HORIZON, ACTION_DIM)
            action = apply_residual(flat_base, self.actor(flat_state, z_all.repeat(plans, 1, 1), flat_base))
            return int(self.critic(self.critic_state(flat_state, flat_pool), action).reshape(-1).argmax()) // k

    def rank_plans(self, state, pools):
        with torch.no_grad():
            pools = mlp_float(pools)
            state = mlp_float(state).expand(pools.shape[0], -1)
            if self.plan_target == "rollout":
                return int(torch.stack(self.plan_critic(state, pools, return_all=True)).mean(dim=0).reshape(-1).argmax())
            return int(self.plan_critic(state, pools).reshape(-1).argmax())

    def n_step_target(self, reward, done, next_critic_state, z_next_all, a_base_next_all, n_steps):
        with torch.no_grad():
            next_critic_state = mlp_float(next_critic_state)
            z_next_all = mlp_float(z_next_all)
            a_base_next_all = mlp_float(a_base_next_all)
            batch, k = z_next_all.shape[0], z_next_all.shape[1]
            critic_k = next_critic_state.unsqueeze(1).expand(batch, k, next_critic_state.shape[-1]).reshape(batch * k, -1)
            state_k = critic_k[:, :self.state_dim]
            z_flat = z_next_all.reshape(batch * k, HORIZON, ACTION_DIM)
            base_flat = a_base_next_all.reshape(batch * k, HORIZON, ACTION_DIM)
            a_next = apply_residual(base_flat, self.actor(state_k, z_flat, base_flat))
            backup = self.target_critic(critic_k, a_next).reshape(batch, k, 1).mean(dim=1)
            return reward + (GAMMA ** n_steps) * (1.0 - done) * backup

    def update_critic(self, critic_state, action, target_q):
        preds = self.critic(critic_state, action, return_all=True)
        loss = torch.stack([F.mse_loss(pred, target_q) for pred in preds]).sum()
        self.critic_opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), MAX_GRAD_NORM)
        self.critic_opt.step()
        self.critic_lr.step()
        return {
            "critic_loss": float(loss.detach()),
            "critic_grad_norm": float(grad_norm),
            "critic_q_mean": float(torch.stack(preds).mean().detach()),
        }

    def update_actor(self, state, critic_state, action_stored, z_all, a_base_all, is_expert, mc_return, filter_active):
        state = mlp_float(state)
        critic_state = mlp_float(critic_state)
        z_all = mlp_float(z_all)
        a_base_all = mlp_float(a_base_all)
        batch, k = z_all.shape[0], z_all.shape[1]
        state_k = state.unsqueeze(1).expand(batch, k, state.shape[-1]).reshape(batch * k, -1)
        critic_k = critic_state.unsqueeze(1).expand(batch, k, critic_state.shape[-1]).reshape(batch * k, -1)
        z_flat = z_all.reshape(batch * k, HORIZON, ACTION_DIM)
        base_flat = a_base_all.reshape(batch * k, HORIZON, ACTION_DIM)
        action = apply_residual(base_flat, self.actor(state_k, z_flat, base_flat))
        q_a = self.critic(critic_k, action).reshape(batch, k)
        online = (mlp_float(is_expert) == 0).float()
        with torch.no_grad():
            q_base = self.critic(critic_k, base_flat).reshape(batch, k)
            overestimation = self.critic(critic_state, action_stored) - mlp_float(mc_return)
            better = (q_a > q_base).float()
            keep = torch.ones_like(better)
            if filter_active:
                keep = 1.0 - better * (overestimation < self.epsilon).float()
            keep = torch.maximum(keep, 1.0 - online)
            q_scale = (q_a.abs() * online).sum() / (online.sum() * k).clamp(min=1.0)
        q_term = -(q_a.mean(dim=1, keepdim=True) * online).mean()
        if q_scale > 1e-8:
            q_term = q_term / q_scale
        mse = ((action - base_flat) ** 2).sum(dim=(1, 2)).reshape(batch, k) / (HORIZON * USED_DOF)
        bc = (keep * mse).mean()
        loss = q_term + BETA * bc
        self.actor_opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), MAX_GRAD_NORM)
        self.actor_opt.step()
        self.actor_lr.step()
        return {
            "actor_loss": float(loss.detach()),
            "actor_q_loss": float(q_term.detach()),
            "actor_bc_loss": float(bc.detach()),
            "actor_grad_norm": float(grad_norm),
            "residual_rms": float(mse.detach().mean().sqrt()),
            "q_mean": float(q_a.detach().mean()),
            "q_min": float(q_a.detach().min()),
            "pretrained_q_mean": float(q_base.mean()),
            "q_advantage": float((q_a.detach() - q_base).mean()),
            "better_than_base_rate": float(better.mean()),
            "bc_filter_rate": float(keep.mean()),
            "q_overestimation": float(overestimation.mean()),
            "q_overestimation_online": float((overestimation * online).sum() / online.sum().clamp(min=1.0)),
        }

    def update_plan_critic(self, state, pools_all, z_all, plan_a_base_all, plan_labels=None, is_expert=None):
        if self.plan_target == "rollout":
            if plan_labels is None:
                raise ValueError("rollout plan target needs plan_labels")
            online = slice(None) if is_expert is None else mlp_float(is_expert).reshape(-1) == 0
            state, pools_all, z_all, plan_a_base_all, plan_labels = (
                mlp_float(part)[online] for part in (state, pools_all, z_all, plan_a_base_all, plan_labels))
        state, pools_all = mlp_float(state), mlp_float(pools_all)
        z_all, plan_a_base_all = mlp_float(z_all), mlp_float(plan_a_base_all)
        batch, plans, k = plan_a_base_all.shape[:3]
        sd, pd = state.shape[-1], pools_all.shape[-1]
        with torch.no_grad():
            flat_state = state[:, None, None].expand(batch, plans, k, sd).reshape(-1, sd)
            flat_pool = pools_all[:, :, None].expand(batch, plans, k, pd).reshape(-1, pd)
            flat_z = z_all[:, None].expand(batch, plans, k, HORIZON, ACTION_DIM).reshape(-1, HORIZON, ACTION_DIM)
            flat_base = plan_a_base_all.reshape(-1, HORIZON, ACTION_DIM)
            action = apply_residual(flat_base, self.actor(flat_state, flat_z, flat_base))
            target = self.critic(self.critic_state(flat_state, flat_pool), action).reshape(batch, plans, k).max(dim=2).values
        extra = {}
        if self.plan_target == "rollout":
            labels = plan_labels.mean(dim=-1)
            extra = {"plan_target_q_mean": float(target.mean()),
                     "plan_label_vs_q_agreement": float((labels.argmax(dim=1) == target.argmax(dim=1)).float().mean())}
            target = labels
        preds = self.plan_critic(
            state[:, None].expand(batch, plans, sd).reshape(-1, sd), pools_all.reshape(-1, pd), return_all=True)
        loss = torch.stack([F.mse_loss(pred.reshape(batch, plans), target) for pred in preds]).sum()
        self.plan_opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.plan_critic.parameters(), MAX_GRAD_NORM)
        self.plan_opt.step()
        self.plan_lr.step()
        return {"plan_critic_loss": float(loss.detach()), "v_plan_mean": float(torch.stack(preds).mean().detach()),
                "plan_target_mean": float(target.mean()), **extra}

    def polyak_update(self):
        with torch.no_grad():
            for parameter, target in zip(self.critic.parameters(), self.target_critic.parameters()):
                target.data.mul_(1.0 - TAU).add_(parameter.data, alpha=TAU)

    def inference_state_dict(self):
        payload = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
        }
        payload["layout"] = {"state_features": self.features, "proprio": self.proprio, "plan_candidates": self.plans}
        if self.plan_target != "q":
            payload["layout"]["plan_target"] = self.plan_target
        if self.plan_critic is not None:
            payload["plan_critic"] = self.plan_critic.state_dict()
        return payload

    def load_inference_state_dict(self, payload):
        layout = payload.get("layout")
        if layout is not None and (layout["state_features"], bool(layout["proprio"]), int(layout.get("plan_candidates", self.plans)),
                                   layout.get("plan_target", "q")) != (self.features, self.proprio, self.plans, self.plan_target):
            raise ValueError("residual checkpoint layout mismatch")
        self.actor.load_state_dict(payload["actor"])
        self.critic.load_state_dict(payload["critic"])
        self.target_critic.load_state_dict(payload["target_critic"])
        if self.plan_critic is not None:
            if "plan_critic" not in payload:
                raise ValueError("plan critic weights missing")
            self.plan_critic.load_state_dict(payload["plan_critic"])

    def resume_state_dict(self):
        payload = {
            **self.inference_state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "actor_lr": self.actor_lr.state_dict(),
            "critic_lr": self.critic_lr.state_dict(),
        }
        if self.plan_opt is not None:
            payload["plan_opt"] = self.plan_opt.state_dict()
            payload["plan_lr"] = self.plan_lr.state_dict()
        return payload

    def load_resume_state_dict(self, payload):
        self.load_inference_state_dict(payload)
        self.actor_opt.load_state_dict(payload["actor_opt"])
        self.critic_opt.load_state_dict(payload["critic_opt"])
        self.actor_lr.load_state_dict(payload["actor_lr"])
        self.critic_lr.load_state_dict(payload["critic_lr"])
        if self.plan_opt is not None:
            self.plan_opt.load_state_dict(payload["plan_opt"])
            self.plan_lr.load_state_dict(payload["plan_lr"])
