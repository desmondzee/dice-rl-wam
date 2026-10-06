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
    task_ids: tuple = (0,)
    online_env_steps: int = 660_000
    checkpoint_every: int = 80_000
    train_eval_episodes: int = 10
    selection_warmup_steps: int = 64_000
    bc_filter_warmup_steps: int = 128_000
    rlpd_start: float = 0.9
    rlpd_end: float = 0.1
    rlpd_steps: int = 208_000
    residual_input: str = "z"
    epsilon: float = -0.5

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
        if not self.task_ids or any(task not in TASK_IDS for task in self.task_ids) or len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError("task_ids must be distinct LIBERO-10 task ids")
        if self.online_env_steps < 1 or self.checkpoint_every < 1 or self.train_eval_episodes < 1:
            raise ValueError("Budget, checkpoint stride, and train-eval episodes must be positive")
        if min(self.selection_warmup_steps, self.bc_filter_warmup_steps, self.rlpd_steps) < 0:
            raise ValueError("Warmup windows must be non-negative")
        if self.residual_input not in ("z", "base", "z_base"):
            raise ValueError("residual_input must be z, base, or z_base")
        if self.epsilon > 0:
            raise ValueError("epsilon must be non-positive")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("Invalid seed")
        if not self.wandb_project:
            raise ValueError("W&B project is required")
        return self

    @property
    def default_run_name(self):
        return "dice-t" + "-".join(str(task) for task in self.task_ids)

    def rlpd_expert_ratio(self, env_steps):
        t = min(max(env_steps, 0), self.rlpd_steps) / max(self.rlpd_steps, 1)
        return self.rlpd_start + (self.rlpd_end - self.rlpd_start) * t

    def checkpoint_schedule(self):
        return sorted(set(range(0, self.online_env_steps + 1, self.checkpoint_every)) | {self.online_env_steps})

    def train_eval_schedule(self):
        every = self.checkpoint_every
        return [point for point in self.checkpoint_schedule() if point == 0 or (point % every == 0 and (point // every) % 2 == 1)]

    def protocol(self):
        self.validate()
        return {
            "version": 3,
            "suite": "libero_10",
            "task_ids": list(self.task_ids),
            "train_init_states": "procedural",
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
            "residual_input": self.residual_input,
            "multi_sample_candidates": self.k_candidates,
            "selection": "max_q_min",
            "selection_warmup_steps": self.selection_warmup_steps,
            "bc_filter_anchor": "q_stored_minus_mc_return",
            "bc_filter_warmup_steps": self.bc_filter_warmup_steps,
            "expert_rows_keep_bc": True,
            "actor_q_normalization": "mean_abs_q_online",
            "actor_final_init": "zeros",
            "eval_best_of_n": self.k_candidates,
            "mlp_hidden": [1024, 1024, 1024],
            "critic_ensemble": 10,
            "beta": 100.0,
            "epsilon": self.epsilon,
            "n_step_chunks": 3,
            "gamma": 0.99,
            "gradient_steps": 10,
            "update_every_chunks": 4,
            "actor_every": 2,
            "min_online_rows": 256,
            "tau": 0.01,
            "adam_lr": 1e-4,
            "weight_decay": 1e-5,
            "max_grad_norm": 1.0,
            "lr_schedule": "cosine_restarts_1000_warmup_10_min_1e-6",
            "batch_size": 256,
            "rlpd_start": self.rlpd_start,
            "rlpd_end": self.rlpd_end,
            "rlpd_steps": self.rlpd_steps,
            "replay_capacity": 100_000,
            "checkpoint_every": self.checkpoint_every,
            "train_eval_episodes": self.train_eval_episodes,
            "train_eval_schedule": self.train_eval_schedule(),
            "checkpoint_step": self.checkpoint_step,
            "source_run": self.source_run,
            "lerobot_revision": LEROBOT_REVISION,
            "model_repo": MODEL_REPO,
            "model_revision": MODEL_REVISION,
            "libero_assets_repo": LIBERO_ASSETS_REPO,
            "libero_assets_revision": LIBERO_ASSETS_REVISION,
        }

    def to_dict(self):
        return {**asdict(self), "task_ids": list(self.task_ids), "protocol": self.protocol()}


def config_from_dict(payload):
    fields = {key: payload[key] for key in payload if key in RLConfig.__dataclass_fields__}
    if "task_ids" in fields:
        fields["task_ids"] = tuple(int(task) for task in fields["task_ids"])
    return RLConfig(**fields).validate()


def main():
    import json
    print(json.dumps(RLConfig().validate().to_dict(), indent=2))


if __name__ == "__main__":
    main()
