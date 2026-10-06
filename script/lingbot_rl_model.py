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
    def __init__(self, residual_input="z"):
        super().__init__()
        self.inputs = RESIDUAL_INPUTS[residual_input]
        self.net = _mlp(STATE_DIM + len(self.inputs) * HORIZON * ACTION_DIM, HORIZON * ACTION_DIM)
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
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList(
            [_mlp(STATE_DIM + HORIZON * ACTION_DIM, 1) for _ in range(ENSEMBLE)]
        )

    def forward(self, state, action, return_all=False):
        state = mlp_float(state)
        batch = state.shape[0]
        x = torch.cat([state, mask_unused_dof(action).reshape(batch, -1)], dim=-1)
        qs = [head(x) for head in self.heads]
        if return_all:
            return qs
        return torch.min(torch.stack(qs, dim=0), dim=0).values


class DiceResidualModel:
    def __init__(self, device="cpu", residual_input="z", epsilon=EPSILON):
        self.device = device
        self.epsilon = epsilon
        self.actor = ResidualActor(residual_input).to(device)
        self.critic = CriticEnsemble().to(device)
        self.target_critic = copy.deepcopy(self.critic).to(device)
        for parameter in self.target_critic.parameters():
            parameter.requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=ADAM_LR, weight_decay=WEIGHT_DECAY)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=ADAM_LR, weight_decay=WEIGHT_DECAY)
        self.actor_lr = torch.optim.lr_scheduler.LambdaLR(self.actor_opt, cosine_restart_lr)
        self.critic_lr = torch.optim.lr_scheduler.LambdaLR(self.critic_opt, cosine_restart_lr)

    def n_step_target(self, reward, done, next_state, z_next_all, a_base_next_all, n_steps):
        with torch.no_grad():
            next_state = mlp_float(next_state)
            z_next_all = mlp_float(z_next_all)
            a_base_next_all = mlp_float(a_base_next_all)
            batch, k = z_next_all.shape[0], z_next_all.shape[1]
            state_k = next_state.unsqueeze(1).expand(batch, k, next_state.shape[-1]).reshape(batch * k, -1)
            z_flat = z_next_all.reshape(batch * k, HORIZON, ACTION_DIM)
            base_flat = a_base_next_all.reshape(batch * k, HORIZON, ACTION_DIM)
            a_next = apply_residual(base_flat, self.actor(state_k, z_flat, base_flat))
            backup = self.target_critic(state_k, a_next).reshape(batch, k, 1).mean(dim=1)
            return reward + (GAMMA ** n_steps) * (1.0 - done) * backup

    def update_critic(self, state, action, target_q):
        preds = self.critic(state, action, return_all=True)
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

    def update_actor(self, state, action_stored, z_all, a_base_all, is_expert, mc_return, filter_active):
        state = mlp_float(state)
        z_all = mlp_float(z_all)
        a_base_all = mlp_float(a_base_all)
        batch, k = z_all.shape[0], z_all.shape[1]
        state_k = state.unsqueeze(1).expand(batch, k, state.shape[-1]).reshape(batch * k, -1)
        z_flat = z_all.reshape(batch * k, HORIZON, ACTION_DIM)
        base_flat = a_base_all.reshape(batch * k, HORIZON, ACTION_DIM)
        action = apply_residual(base_flat, self.actor(state_k, z_flat, base_flat))
        q_a = self.critic(state_k, action).reshape(batch, k)
        online = (mlp_float(is_expert) == 0).float()
        with torch.no_grad():
            q_base = self.critic(state_k, base_flat).reshape(batch, k)
            overestimation = self.critic(state, action_stored) - mlp_float(mc_return)
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

    def polyak_update(self):
        with torch.no_grad():
            for parameter, target in zip(self.critic.parameters(), self.target_critic.parameters()):
                target.data.mul_(1.0 - TAU).add_(parameter.data, alpha=TAU)

    def inference_state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
        }

    def load_inference_state_dict(self, payload):
        self.actor.load_state_dict(payload["actor"])
        self.critic.load_state_dict(payload["critic"])
        self.target_critic.load_state_dict(payload["target_critic"])

    def resume_state_dict(self):
        return {
            **self.inference_state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "actor_lr": self.actor_lr.state_dict(),
            "critic_lr": self.critic_lr.state_dict(),
        }

    def load_resume_state_dict(self, payload):
        self.load_inference_state_dict(payload)
        self.actor_opt.load_state_dict(payload["actor_opt"])
        self.critic_opt.load_state_dict(payload["critic_opt"])
        self.actor_lr.load_state_dict(payload["actor_lr"])
        self.critic_lr.load_state_dict(payload["critic_lr"])
