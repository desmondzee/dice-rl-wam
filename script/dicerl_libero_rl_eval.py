import argparse
import os
import re

import torch

from script.dicerl_libero_eval import add_protocol_args, make_env, protocol_inputs, run_dir_of, run_protocol, shift_totals, write_results

STATE_DIM = 8


def iteration_of(ckpt_path):
    match = re.fullmatch(r"model_step_(\d+)\.pth", os.path.basename(ckpt_path))
    if match is None:
        raise ValueError(f"{ckpt_path}: expected a model_step_<N>.pth checkpoint")
    return int(match.group(1))


def residual_cfg(ckpt_path, device):
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), ".hydra", "config.yaml"))
    cfg.device = device
    return cfg


def load_residual(ckpt_path, device):
    import hydra
    from omegaconf import OmegaConf
    cfg = residual_cfg(ckpt_path, device)
    prior = OmegaConf.load(os.path.join(run_dir_of(os.path.expandvars(cfg.base_policy_path)), ".hydra", "config.yaml"))
    cfg.obs_dim = STATE_DIM + prior.model.network.visual_feature_dim
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = hydra.utils.instantiate(cfg.model)
    model.load_state_dict(checkpoint["model_state_dict"])
    return cfg, model.to(device).eval()


def augment(model, obs, device):
    state = torch.from_numpy(obs["state"]).float().to(device)
    rgb = torch.from_numpy(obs["rgb"]).float().to(device).permute(0, 1, 4, 2, 3)
    with torch.no_grad():
        features = model.pretrained_flow_policy.network.extract_visual_features({"state": state, "rgb": rgb})
    return torch.cat([state, features.unsqueeze(1).expand(-1, state.shape[1], -1)], dim=-1)


def residual_policy(model, act_steps, device):
    def policy(obs):
        state = augment(model, obs, device)
        with torch.no_grad():
            actions, _ = model.get_exploration_action(state, num_samples=16, exploration_strategy="max_q_min", training_step=10 ** 6)
        return actions[:, :act_steps].cpu().numpy()
    return policy


def main():
    parser = argparse.ArgumentParser()
    add_protocol_args(parser)
    args = parser.parse_args()
    iteration = iteration_of(args.ckpt)
    cfg = residual_cfg(args.ckpt, args.device)
    init_states, labels, plan, shift = protocol_inputs(args, cfg.env.name)
    _, model = load_residual(args.ckpt, args.device)
    venv = make_env(cfg, args.envs)
    try:
        rows = run_protocol(residual_policy(model, cfg.act_steps, args.device), venv, init_states, plan, args.envs, cfg.act_steps, cfg.env.max_episode_steps, shift, labels)
        stats = shift_totals(venv, shift)
    finally:
        venv.close()
    write_results(args, cfg, rows, {"weights": "model_state_dict", "iteration": iteration, "shift_stats": stats})


if __name__ == "__main__":
    main()
