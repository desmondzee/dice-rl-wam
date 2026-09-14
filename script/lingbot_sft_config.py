import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

UPSTREAM_REVISION = "7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb"
MODEL_REPO = "robbyant/lingbot-va-base"
MODEL_REVISION = "68b7bc1b35da6ddc67ea94c4ceb58d768fbb3f9c"
DATASET_REPO = "robbyant/libero-long-lerobot"
DATASET_REVISION = "8c0313b1c7cd9fa3798798479cbf59b11af8979d"
CAMERAS = ("observation.images.agentview_rgb", "observation.images.eye_in_hand_rgb")
RECIPE_VERSION = 1


@dataclass(frozen=True)
class SFTConfig:
    seed: int = 42
    demos_per_task: int = 30
    num_steps: int = 1000
    learning_rate: float = 1e-5
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.1
    warmup_steps: int = 10
    grad_clip: float = 2.0
    global_batch_size: int = 80
    batch_size: int = 1
    save_interval: int = 100
    weight_steps: tuple[int, ...] = (100, 200, 400, 600, 800, 1000)
    cfg_prob: float = 0.1
    snr_shift: float = 5.0
    action_snr_shift: float = 0.05
    load_worker: int = 2
    wandb_project: str = "dice-lingbot-va-sft"
    wandb_entity: str | None = None

    def validate(self):
        if self.demos_per_task != 30:
            raise ValueError("This experiment is restricted to 30 demonstrations per task")
        if not 1 <= self.num_steps <= 1000:
            raise ValueError("The initial SFT run must stop within 1,000 optimizer updates")
        if self.batch_size != 1 or self.global_batch_size != 80:
            raise ValueError("Use one full episode per GPU microbatch and effective batch size 80")
        if self.seed < 0 or self.save_interval < 1 or self.load_worker < 0:
            raise ValueError("Invalid seed, checkpoint interval, or worker count")
        if not math.isfinite(self.learning_rate) or not math.isfinite(self.weight_decay) or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer configuration")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("Optimizer betas must be in [0,1)")
        if self.grad_clip != 2.0:
            raise ValueError("The pinned native trainer clips gradients at 2.0")
        if not 0 <= self.cfg_prob <= 1 or self.warmup_steps < 0:
            raise ValueError("Invalid dropout or warmup configuration")
        if not all(0 < step <= 1000 for step in self.weight_steps):
            raise ValueError("Invalid weight checkpoint milestone")
        return self

    def accumulation_steps(self, world_size):
        if world_size not in (4, 8):
            raise ValueError("SFT requires 4 or 8 GPUs")
        return self.global_batch_size // (world_size * self.batch_size)

    def save_weights_at(self, step):
        return step in self.weight_steps or step == self.num_steps

    def to_dict(self):
        return asdict(self)

    def training_identity(self):
        values = self.to_dict()
        for key in ("num_steps", "save_interval", "weight_steps", "load_worker", "wandb_entity", "wandb_project"):
            values.pop(key)
        return values


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def provenance():
    return {
        "recipe_version": RECIPE_VERSION,
        "upstream_revision": UPSTREAM_REVISION,
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "suite": "libero_10",
        "camera_order": list(CAMERAS),
        "resolution_per_camera": [128, 128],
        "action_channels": list(range(7)),
        "action_dim": 30,
        "action_per_latent_frame": 4,
        "training_chunk_size_range": [1, 4],
        "training_attention_window_range": [4, 64],
        "training_video_history_noise_probability": 0.5,
        "training_sequence": "one full published latent episode per microbatch; no new packing or cropping",
        "normalization": "q01/q99 of raw actions in the selected 300 episodes only; epsilon=1e-6; clip=[-1.5,1.5]",
        "paper_references": ["https://arxiv.org/html/2601.21998v2", "https://arxiv.org/html/2603.10263v2"],
        "comparability_limits": [
            "DICE-RL Appendix A does not publish the exact 30-demo subset or pi0 SFT update budget.",
            "LingBot-VA paper reports 4K LIBERO updates and sequence length 100K; released code uses 5K and whole-episode microbatches.",
            "This run follows the released shared-backbone trainer, shortened to at most 1K updates.",
            "Published latent dataset is used as supplied; its metadata contains no independent demonstration-success flag.",
            "Full-data released normalization is deliberately not reused: statistics are fit only on the 30-demo subset.",
            "SFT loss does not establish a 40-70 percent success rate; rollout evaluation and LeRobot conversion are separate work.",
            "The paper's video cutoff/action inference steps differ from the released LIBERO inference configuration; neither is an SFT hyperparameter."
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, choices=(4, 8), default=8)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    cfg = SFTConfig(num_steps=args.steps).validate()
    output = {"training": cfg.to_dict(), "provenance": provenance(), "gpus": args.gpus,
              "gradient_accumulation_steps": cfg.accumulation_steps(args.gpus)}
    if args.manifest:
        output["dataset_manifest"] = json.loads(args.manifest.read_text())
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
