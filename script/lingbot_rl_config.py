from dataclasses import asdict, dataclass, field

from script.lingbot_eval_config import (
    CAMERAS, LEROBOT_REVISION, LIBERO_ASSETS_REPO, LIBERO_ASSETS_REVISION,
    TASK_IDS, validate_name, validate_sampler_steps,
)
from script.lingbot_sft_config import MODEL_REPO, MODEL_REVISION

PLUS_VARIANTS = {"robot_init": tuple(range(640, 682)), "layout": tuple(range(1982, 2021))}


def plus_split(axis):
    variants = PLUS_VARIANTS[axis]
    train = tuple(variant for position, variant in enumerate(variants) if position % 3 != 2)
    test = tuple(variant for position, variant in enumerate(variants) if position % 3 == 2)
    return train, test


def plus_train_variants():
    return {axis: plus_split(axis)[0] for axis in PLUS_VARIANTS}


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
    plan_candidates: int = 1
    plan_random_prob: float = 0.1
    state_features: str = "pool"
    proprio: bool = False
    plus_train: dict = field(default_factory=dict)
    plan_target: str = "q"
    store_tokens: bool = False
    label_workers: int = 4

    def validate(self):
        validate_name(self.source_run)
        if self.source_run != "libero30-sft":
            raise ValueError("RL is pinned to libero30-sft")
        if self.checkpoint_step != 600:
            raise ValueError("RL is pinned to checkpoint step 600")
        validate_sampler_steps(self.video_steps, self.action_steps, self.video_exec_step)
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
        if type(self.plan_candidates) is not int or not 1 <= self.plan_candidates <= 8:
            raise ValueError("plan_candidates must be an integer in [1, 8]")
        if not 0.0 <= float(self.plan_random_prob) <= 1.0:
            raise ValueError("plan_random_prob must lie in [0, 1]")
        if self.state_features not in ("pool", "struct"):
            raise ValueError("state_features must be pool or struct")
        if type(self.proprio) is not bool:
            raise ValueError("proprio must be a bool")
        if self.plus_train and set(self.plus_train) != set(PLUS_VARIANTS):
            raise ValueError("plus_train must list both robot_init and layout variants")
        for axis, variants in self.plus_train.items():
            if not variants or len(set(variants)) != len(variants) or any(v not in plus_split(axis)[0] for v in variants):
                raise ValueError("plus_train variants must be distinct task 9 training variants")
        if self.plus_train and tuple(self.task_ids) != (9,):
            raise ValueError("plus_train collection is defined for task 9 only")
        if self.plan_target not in ("q", "rollout"):
            raise ValueError("plan_target must be q or rollout")
        if self.plan_target == "rollout" and (self.plan_candidates < 2 or not self.proprio):
            raise ValueError("rollout plan labels need plan_candidates > 1 and proprio")
        if type(self.store_tokens) is not bool or (self.store_tokens and self.plan_candidates < 2):
            raise ValueError("store_tokens must be a bool and needs plan_candidates > 1")
        if type(self.label_workers) is not int or self.label_workers < 1:
            raise ValueError("label_workers must be a positive integer")
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
        protocol = {
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
            "plan_candidates": self.plan_candidates,
            "plan_random_prob": self.plan_random_prob,
            "plan_critic": "regression_max_k_min_q" if self.plan_candidates > 1 else "none",
            "plan_pool": "critic_pool_of_denoised_chunk" if self.plan_candidates > 1 else "none",
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
        if (self.state_features, self.proprio, bool(self.plus_train)) != ("pool", False, False):
            protocol.update({
                "state_features": self.state_features,
                "proprio": self.proprio,
                "plus_train": {axis: [int(v) for v in variants] for axis, variants in sorted(self.plus_train.items())},
                "collection_mix": "thirds_standard_robot_init_layout" if self.plus_train else "standard",
            })
        if self.plan_target != "q":
            protocol.update({"plan_target": self.plan_target, "plan_critic": "regression_mean_k_rollout_progress"})
        return protocol

    def to_dict(self):
        return {**asdict(self), "task_ids": list(self.task_ids), "protocol": self.protocol()}


def config_from_dict(payload):
    fields = {key: payload[key] for key in payload if key in RLConfig.__dataclass_fields__}
    if "task_ids" in fields:
        fields["task_ids"] = tuple(int(task) for task in fields["task_ids"])
    if "plus_train" in fields:
        fields["plus_train"] = {axis: tuple(int(v) for v in variants) for axis, variants in fields["plus_train"].items()}
    return RLConfig(**fields).validate()


def main():
    import json
    print(json.dumps(RLConfig().validate().to_dict(), indent=2))


if __name__ == "__main__":
    main()
