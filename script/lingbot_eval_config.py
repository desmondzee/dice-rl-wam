import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass

from script.lingbot_sft_config import MODEL_REPO, MODEL_REVISION


LEROBOT_REVISION = "3f2c29ef7e44b1ddccbcda3b6a63939e53639e9e"
LIBERO_ASSETS_REPO = "lerobot/libero-assets"
LIBERO_ASSETS_REVISION = "0b3ea86be5fe169d0fd036ae63d1070ec09e90f6"
TASK_IDS = tuple(range(10))
CAMERAS = ("observation.images.image", "observation.images.image2")
ARCHITECTURE_KEYS = (
    "patch_size", "num_attention_heads", "attention_head_dim", "in_channels",
    "out_channels", "action_dim", "text_dim", "freq_dim", "ffn_dim", "num_layers",
    "cross_attn_norm", "eps", "rope_max_seq_len",
)


def validate_name(name):
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", name):
        raise ValueError("Names must contain only letters, numbers, underscores, and hyphens")
    return name


@dataclass(frozen=True)
class EvalConfig:
    source_run: str = "libero30-sft"
    checkpoint_step: int = 600
    stage: str = "smoke"
    seed: int = 42
    wandb_project: str = "dice-lingbot-va-eval"
    wandb_entity: str | None = None
    task_ids: tuple = TASK_IDS
    shard: int = 0
    shards: int = 1

    def validate(self):
        validate_name(self.source_run)
        if self.stage not in ("smoke", "eval", "heldout"):
            raise ValueError("Evaluation stage must be smoke, eval, or heldout")
        if not self.task_ids or any(task not in TASK_IDS for task in self.task_ids) or len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError("task_ids must be distinct LIBERO-10 task ids")
        if type(self.shards) is not int or type(self.shard) is not int or not 0 <= self.shard < self.shards:
            raise ValueError("shard must lie in [0, shards)")
        if type(self.checkpoint_step) is not int or not 1 <= self.checkpoint_step <= 1000:
            raise ValueError("Invalid checkpoint step")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("Invalid evaluation seed")
        if not self.wandb_project:
            raise ValueError("W&B project is required")
        return self

    @property
    def episodes_per_task(self):
        return {"smoke": 1, "eval": 20, "heldout": 100}[self.stage]

    @property
    def initial_state_offset(self):
        return 1 if self.stage == "eval" else 0

    @property
    def default_run_name(self):
        return f"{self.source_run}-step{self.checkpoint_step:06d}-{self.stage}"

    def protocol(self):
        self.validate()
        return {
            "version": 2,
            "suite": "libero_10",
            "task_ids": list(self.task_ids),
            "shard": self.shard,
            "shards": self.shards,
            "episodes_per_task": self.episodes_per_task,
            "initial_state_offset": self.initial_state_offset,
            "base_seed": self.seed,
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
            "video_steps": 20,
            "action_steps": 50,
            "video_exec_step": -1,
            "video_guidance": 5.0,
            "action_guidance": 1.0,
            "snr_shift": 5.0,
            "action_snr_shift": 0.05,
            "attention_window": 30,
            "action_normalization_epsilon": 1e-6,
            "sampler": "released_lingbot_libero_defaults",
            "dtype": "bfloat16",
            "attention_backend": "torch",
            "lerobot_revision": LEROBOT_REVISION,
            "model_repo": MODEL_REPO,
            "model_revision": MODEL_REVISION,
            "libero_assets_repo": LIBERO_ASSETS_REPO,
            "libero_assets_revision": LIBERO_ASSETS_REVISION,
        }

    def to_dict(self):
        return {**asdict(self), "task_ids": list(self.task_ids), "protocol": self.protocol()}


def eval_config_from_dict(payload):
    fields = {key: payload[key] for key in payload if key in EvalConfig.__dataclass_fields__}
    if "task_ids" in fields:
        fields["task_ids"] = tuple(int(task) for task in fields["task_ids"])
    return EvalConfig(**fields).validate()


def episode_plan(config, task_id, initial_state_count):
    config.validate()
    if task_id not in TASK_IDS:
        raise ValueError("Invalid task ID")
    states = initial_state_count - config.initial_state_offset
    if states < 1 or states < min(config.episodes_per_task, initial_state_count):
        raise ValueError("Not enough distinct initial states for this evaluation protocol")
    plan = []
    for episode_index in range(config.episodes_per_task):
        init_state_id = config.initial_state_offset + episode_index % states
        identity = ["libero_10", config.seed, task_id, init_state_id] + [episode_index // states] * (episode_index >= states)
        seed = int.from_bytes(hashlib.sha256(json.dumps(identity).encode()).digest()[:4], "big")
        plan.append({"task_id": task_id, "episode_index": episode_index, "init_state_id": init_state_id, "seed": seed})
    return plan


def shard_plan(config, plan):
    return plan[config.shard::config.shards]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "eval", "heldout"), default="smoke")
    parser.add_argument("--checkpoint-step", type=int, default=600)
    args = parser.parse_args()
    print(json.dumps(EvalConfig(stage=args.stage, checkpoint_step=args.checkpoint_step).to_dict(), indent=2))


if __name__ == "__main__":
    main()
