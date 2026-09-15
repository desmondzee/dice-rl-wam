from dataclasses import asdict, dataclass

from script.lingbot_eval_config import (
    CAMERAS, LEROBOT_REVISION, LIBERO_ASSETS_REPO, LIBERO_ASSETS_REVISION,
    TASK_IDS, validate_name,
)
from script.lingbot_sft_config import MODEL_REPO, MODEL_REVISION


@dataclass(frozen=True)
class RLConfig:
    source_run: str = "libero30-sft"
    checkpoint_step: int = 600
    seed: int = 42
    wandb_project: str = "dice-lingbot-va-rl"
    wandb_entity: str | None = None
    video_steps: int = 20
    action_steps: int = 50
    video_exec_step: int = -1
    k_candidates: int = 4
    online_env_steps: int = 100_000
    train_eval_every: int = 25_000
    train_eval_episodes_per_task: int = 1

    def validate(self):
        validate_name(self.source_run)
        if self.source_run != "libero30-sft":
            raise ValueError("RL is pinned to libero30-sft")
        if self.checkpoint_step != 600:
            raise ValueError("RL is pinned to checkpoint step 600")
        if self.video_steps != 20 or self.action_steps != 50 or self.video_exec_step != -1:
            raise ValueError("RL must use the released 20/50 full-video sampler")
        if self.k_candidates != 4:
            raise ValueError("Collection K is pinned to 4")
        if self.online_env_steps != 100_000:
            raise ValueError("Online budget is 100,000 env action steps")
        if self.train_eval_every != 25_000 or self.train_eval_episodes_per_task != 1:
            raise ValueError("Train-time eval is 1 rollout/task every 25,000 env steps")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("Invalid seed")
        if not self.wandb_project:
            raise ValueError("W&B project is required")
        return self

    @property
    def default_run_name(self):
        return f"{self.source_run}-dice-baseline"

    def rlpd_expert_ratio(self, env_steps):
        span = self.online_env_steps
        t = min(max(env_steps, 0), span) / span
        return 0.5 + (0.1 - 0.5) * t

    def protocol(self):
        self.validate()
        return {
            "version": 2,
            "suite": "libero_10",
            "task_ids": list(TASK_IDS),
            "max_policy_steps": 520,
            "settling_steps": 10,
            "control_freq": 20,
            "control_mode": "relative",
            "hard_reset": True,
            "environment_batch_size": 1,
            "camera_keys": list(CAMERAS),
            "camera_orientation": "vertical_flip_only_native_lingbot",
            "resolution": [128, 128],
            "frame_chunk_size": 4,
            "action_per_frame": 4,
            "video_steps": self.video_steps,
            "action_steps": self.action_steps,
            "video_exec_step": self.video_exec_step,
            "video_guidance": 5.0,
            "action_guidance": 1.0,
            "snr_shift": 5.0,
            "action_snr_shift": 0.05,
            "attention_window": 30,
            "action_normalization_epsilon": 1e-6,
            "sampler": "released_lingbot_libero_defaults",
            "dtype": "bfloat16",
            "attention_backend": "torch",
            "noisy_history": False,
            "text_encoder_device": "cpu",
            "image_hflip": False,
            "camera_layout": "width_concat",
            "used_action_channels": list(range(7)),
            "k_candidates": self.k_candidates,
            "online_env_steps": self.online_env_steps,
            "residual_input": "z",
            "multi_sample_candidates": self.k_candidates,
            "bc_filter_anchor": "mc_return",
            "utd_sampling": "fresh_minibatch_per_step",
            "actor_final_init": "zeros",
            "eval_best_of_n": self.k_candidates,
            "mlp_hidden": [1024, 1024, 1024],
            "critic_ensemble": 10,
            "beta": 100.0,
            "epsilon": -0.5,
            "n_step_chunks": 3,
            "gamma": 0.99,
            "utd": 10,
            "tau": 0.01,
            "adam_lr": 1e-4,
            "batch_size": 256,
            "rlpd_start": 0.5,
            "rlpd_end": 0.1,
            "replay_capacity": 100_000,
            "train_eval_every": self.train_eval_every,
            "train_eval_episodes_per_task": self.train_eval_episodes_per_task,
            "comparison_episodes_per_task": 20,
            "comparison_initial_state_offset": 1,
            "comparison_seed": 42,
            "checkpoint_step": self.checkpoint_step,
            "source_run": self.source_run,
            "lerobot_revision": LEROBOT_REVISION,
            "model_repo": MODEL_REPO,
            "model_revision": MODEL_REVISION,
            "libero_assets_repo": LIBERO_ASSETS_REPO,
            "libero_assets_revision": LIBERO_ASSETS_REVISION,
        }

    def to_dict(self):
        return {**asdict(self), "protocol": self.protocol()}


def main():
    import json
    print(json.dumps(RLConfig().validate().to_dict(), indent=2))


if __name__ == "__main__":
    main()
