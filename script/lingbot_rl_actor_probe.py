import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from script.lingbot_rl_model import (
    ACTION_DIM, ADAM_LR, BETA, HORIZON, MAX_GRAD_NORM, STATE_DIM, USED_DOF, WEIGHT_DECAY,
    DiceResidualModel, _mlp, apply_residual, mask_unused_dof, mlp_float,
)
from script.lingbot_rl_sharpen import bin_entropy, log_std

VARIANTS = {"z": ("z",), "base": ("base",), "z_base": ("z", "base")}


class ProbeActor(nn.Module):
    def __init__(self, inputs):
        super().__init__()
        self.inputs = inputs
        self.net = _mlp(STATE_DIM + len(inputs) * HORIZON * ACTION_DIM, HORIZON * ACTION_DIM)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, state, z, base):
        parts = [mlp_float(state)] + [mlp_float(z if name == "z" else base).reshape(state.shape[0], -1) for name in self.inputs]
        return mask_unused_dof(self.net(torch.cat(parts, dim=-1)).reshape(state.shape[0], HORIZON, ACTION_DIM))


def load_rows(resume_path):
    payload = torch.load(resume_path, map_location="cpu", weights_only=False)
    rows = [r for r in payload["replay"]["data"] if float(r["is_expert"]) == 0.0]
    s = torch.from_numpy(np.stack([r["s"] for r in rows])).float()
    z = torch.from_numpy(np.stack([r["z_all"] for r in rows])).float()
    base = torch.from_numpy(np.stack([r["a_base_all"] for r in rows])).float()
    return s, z, base


def flat(s, z, base):
    n, k = z.shape[:2]
    return s.unsqueeze(1).expand(n, k, -1).reshape(n * k, -1), z.reshape(n * k, HORIZON, ACTION_DIM), base.reshape(n * k, HORIZON, ACTION_DIM), n, k


def diagnostics(actor, critic, s, z, base, batch=1024):
    out = {"delta_v": [], "delta_h": [], "delta_log_std": [], "residual": [], "specific": [], "offset": []}
    with torch.no_grad():
        for start in range(0, s.shape[0], batch):
            sk, zk, bk, n, k = flat(s[start:start + batch], z[start:start + batch], base[start:start + batch])
            r = actor(sk, zk, bk)
            a = apply_residual(bk, r)
            out["delta_v"].append((critic(sk, a) - critic(sk, bk)).reshape(n, k).mean(dim=1))
            after, before = a.reshape(n, k, HORIZON, ACTION_DIM).numpy(), bk.reshape(n, k, HORIZON, ACTION_DIM).numpy()
            out["delta_h"].append(torch.from_numpy(bin_entropy(after) - bin_entropy(before)))
            out["delta_log_std"].append(torch.from_numpy(log_std(after) - log_std(before)))
            live = r.reshape(n, k, HORIZON, ACTION_DIM)[..., :USED_DOF]
            specific = live - live.mean(dim=1, keepdim=True)
            offset = bk.reshape(n, k, HORIZON, ACTION_DIM)[..., :USED_DOF]
            offset = offset - offset.mean(dim=1, keepdim=True)
            out["residual"].append(live.reshape(-1)); out["specific"].append(specific.reshape(-1)); out["offset"].append(offset.reshape(-1))
    cat = {key: torch.cat(value) for key, value in out.items()}
    specific, offset, residual = cat["specific"], cat["offset"], cat["residual"]
    return {
        "delta_v_mean": float(cat["delta_v"].mean()), "delta_h_mean": float(cat["delta_h"].mean()),
        "delta_log_std_mean": float(cat["delta_log_std"].mean()), "sharpened_fraction": float((cat["delta_log_std"] < 0).float().mean()),
        "r_delta_v_delta_h": float(np.corrcoef(cat["delta_v"].numpy(), cat["delta_h"].numpy())[0, 1]),
        "residual_rms": float(residual.pow(2).mean().sqrt()),
        "specific_energy_fraction": float(specific.pow(2).sum() / residual.pow(2).sum().clamp(min=1e-12)),
        "corr_specific_offset": float(np.corrcoef(specific.numpy(), offset.numpy())[0, 1]),
    }


def train(actor, critic, s, z, base, steps, batch, seed):
    generator = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(actor.parameters(), lr=ADAM_LR, weight_decay=WEIGHT_DECAY)
    for step in range(steps):
        pick = torch.randint(0, s.shape[0], (batch,), generator=generator)
        sk, zk, bk, n, k = flat(s[pick], z[pick], base[pick])
        a = apply_residual(bk, actor(sk, zk, bk))
        q = critic(sk, a).reshape(n, k)
        scale = q.abs().mean().detach().clamp(min=1e-8)
        bc = ((a - bk) ** 2).sum(dim=(1, 2)).reshape(n, k) / (HORIZON * USED_DOF)
        loss = -(q.mean(dim=1)).mean() / scale + BETA * bc.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), MAX_GRAD_NORM)
        opt.step()
        if (step + 1) % 500 == 0:
            print(f"step {step + 1} loss {float(loss):.4f} q {float(q.mean()):.4f} bc {float(bc.mean()):.5f}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--residual", type=Path, required=True)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    s, z, base = load_rows(args.resume)
    split = int(s.shape[0] * 0.9)
    model = DiceResidualModel()
    model.load_inference_state_dict(torch.load(args.residual, map_location="cpu", weights_only=True))
    for parameter in model.critic.parameters():
        parameter.requires_grad_(False)
    actor = ProbeActor(VARIANTS[args.variant])
    train(actor, model.critic, s[:split], z[:split], base[:split], args.steps, args.batch, args.seed)
    summary = {"variant": args.variant, "steps": args.steps, "train_rows": split, "heldout_rows": s.shape[0] - split,
               "heldout": diagnostics(actor, model.critic, s[split:], z[split:], base[split:]),
               "train": diagnostics(actor, model.critic, s[:split][:4000], z[:split][:4000], base[:split][:4000])}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    torch.save(actor.state_dict(), args.out / "actor.pt")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
