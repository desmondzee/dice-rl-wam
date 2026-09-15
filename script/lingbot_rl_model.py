import copy

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
UTD = 10
TAU = 0.01
ADAM_LR = 1e-4
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


def _mlp(in_dim, out_dim):
    dims = [in_dim, *HIDDEN, out_dim]
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


class ResidualActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = _mlp(STATE_DIM + HORIZON * ACTION_DIM, HORIZON * ACTION_DIM)
        final = self.net[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, state, noise):
        state = mlp_float(state)
        noise = mlp_float(noise)
        batch = state.shape[0]
        residual = self.net(torch.cat([state, noise.reshape(batch, -1)], dim=-1))
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
    def __init__(self, device="cpu"):
        self.device = device
        self.actor = ResidualActor().to(device)
        self.critic = CriticEnsemble().to(device)
        self.target_critic = copy.deepcopy(self.critic).to(device)
        for parameter in self.target_critic.parameters():
            parameter.requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=ADAM_LR)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=ADAM_LR)

    def n_step_target(self, reward, done, next_state, next_action, n_steps):
        with torch.no_grad():
            backup = self.target_critic(next_state, next_action)
            return reward + (GAMMA ** n_steps) * (1.0 - done) * backup

    def update_critic(self, state, action, target_q, is_expert):
        preds = self.critic(state, action, return_all=True)
        loss = torch.stack([F.mse_loss(pred, target_q) for pred in preds]).sum()
        self.critic_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.critic_opt.step()
        self.polyak_update()
        return {
            "critic_loss": float(loss.detach()),
            "q_mean": float(torch.stack(preds).mean().detach()),
        }

    def update_actor(self, state, noise, a_base, is_expert, target_q):
        residual = self.actor(state, noise)
        action = apply_residual(a_base, residual)
        q_a = self.critic(state, action)
        with torch.no_grad():
            q_base = self.critic(state, a_base)
            better = (q_a > q_base).float()
            underestimated = ((q_a - target_q) < EPSILON).float()
            bc_keep = 1.0 - better * underestimated
        online = (is_expert == 0).float()
        q_term = -(q_a * online).sum() / online.sum().clamp(min=1.0)
        mse = ((action - a_base) ** 2).mean(dim=(1, 2), keepdim=True)
        bc = (bc_keep * mse).mean()
        loss = q_term + BETA * bc
        self.actor_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.actor_opt.step()
        return {
            "actor_loss": float(loss.detach()),
            "residual_rms": float(((action - a_base).detach() ** 2).mean().sqrt()),
            "q_mean": float(q_a.detach().mean()),
            "q_min": float(q_a.detach().min()),
            "bc_filter_rate": float(bc_keep.mean()),
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
        }

    def load_resume_state_dict(self, payload):
        self.load_inference_state_dict(payload)
        self.actor_opt.load_state_dict(payload["actor_opt"])
        self.critic_opt.load_state_dict(payload["critic_opt"])
