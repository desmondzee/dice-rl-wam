import argparse
import copy
import fcntl
import hashlib
import json
import os
import random
import re
import shutil
import sys
import tempfile
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict, set_state_dict

from script.lingbot_sft_config import MODEL_REVISION, SFTConfig, fingerprint, provenance
from script.lingbot_sft_patch import apply_patch


_RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}\Z")


def capture_rng():
    np_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (np_state[0], np_state[1].tolist(), np_state[2], np_state[3], np_state[4]),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }


def restore_rng(state):
    random.setstate(state["python"])
    ns = state["numpy"]
    np.random.set_state((ns[0], np.asarray(ns[1], dtype=np.uint32), ns[2], ns[3], ns[4]))
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"])


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _distributed():
    return dist.is_available() and dist.is_initialized()


def _rank():
    return dist.get_rank() if _distributed() else 0


def _world_size():
    return dist.get_world_size() if _distributed() else 1


def _sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _broadcast_object(value):
    if not _distributed():
        return value
    values = [value if _rank() == 0 else None]
    dist.broadcast_object_list(values, src=0)
    return values[0]


def _raise_status(status):
    status = _broadcast_object(status)
    if not status["ok"]:
        raise RuntimeError(status["error"])


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path, value):
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_status(run_dir, state, step, target_steps):
    path = Path(run_dir) / "status.json"
    temporary = path.with_name("status.json.tmp")
    payload = {"state": state, "step": step, "target_steps": target_steps}
    with temporary.open("w") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _native_config(spec, paths, run_dir, rank, local_rank, world_size):
    from wan_va.configs import VA_CONFIGS

    manifest = json.loads(Path(paths["manifest_path"]).read_text())
    config = copy.deepcopy(VA_CONFIGS["libero_train"])
    config.update(spec.to_dict())
    config.update(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        dataset_path=paths["dataset_path"],
        manifest_path=paths["manifest_path"],
        empty_emb_path=paths["empty_emb_path"],
        norm_stat=manifest["norm_stat"],
        wan22_pretrained_model_name_or_path=paths["model_path"],
        save_root=str(run_dir),
        enable_wandb=False,
        gradient_accumulation_steps=spec.accumulation_steps(world_size),
    )
    config.pop("resume_from", None)
    return config


def _native_sft_class():
    from wan_va.train import Trainer

    class SFTTrainer(Trainer):
        def __init__(self, config, spec, paths, run_dir, patch_hashes, run_name, resume, modal_volume=None):
            self.spec = spec
            self.paths = paths
            self.run_dir = Path(run_dir)
            self.patch_hashes = patch_hashes
            self.run_name = run_name
            self.resume_requested = resume
            self.modal_volume = modal_volume
            self.data_epoch = 0
            self.data_offset = 0
            self.train_loader_iter = None
            self.wandb_run = None
            super().__init__(config)
            if self.train_loader.sampler is not None and hasattr(self.train_loader.sampler, "seed"):
                self.train_loader.sampler.seed = spec.seed
            self.train_loader.generator = torch.Generator().manual_seed(spec.seed + config.rank)
            self.empty_emb = torch.load(paths["empty_emb_path"], map_location="cpu", weights_only=True)
            if tuple(self.empty_emb.shape) != (512, 4096) or self.empty_emb.dtype != torch.bfloat16:
                raise ValueError("Empty embedding must have shape [512,4096] and bfloat16 dtype")
            self.empty_emb = self.empty_emb.to(self.device)
            seed_everything(spec.seed + config.rank)

        def _position_data_iterator(self):
            if self.train_loader_iter is None:
                if hasattr(self.train_loader.sampler, "set_epoch"):
                    self.train_loader.sampler.set_epoch(self.data_epoch)
                self.train_loader_iter = iter(self.train_loader)
                for _ in range(self.data_offset):
                    next(self.train_loader_iter)

        def _get_next_batch(self):
            self._position_data_iterator()
            try:
                batch = next(self.train_loader_iter)
            except StopIteration:
                self.data_epoch += 1
                self.data_offset = 0
                if hasattr(self.train_loader.sampler, "set_epoch"):
                    self.train_loader.sampler.set_epoch(self.data_epoch)
                self.train_loader_iter = iter(self.train_loader)
                batch = next(self.train_loader_iter)
            self.data_offset += 1
            if torch.rand(1).item() < self.spec.cfg_prob:
                batch["text_emb"] = self.empty_emb.unsqueeze(0).clone()
            return batch

        def _rank_state(self):
            return {
                "rng": capture_rng(),
                "data_epoch": self.data_epoch,
                "data_offset": self.data_offset,
                "loader_generator": self.train_loader.generator.get_state(),
            }

        def _collect_states(self):
            options = StateDictOptions(full_state_dict=True, cpu_offload=True)
            model_state, optimizer_state = get_state_dict(self.transformer, self.optimizer, options=options)
            rank_state = self._rank_state()
            if _distributed():
                gathered = [None for _ in range(_world_size())]
                dist.all_gather_object(gathered, rank_state)
            else:
                gathered = [rank_state]
            return model_state, optimizer_state, gathered

        def _metadata(self, rank_states):
            return {
                "format_version": 1,
                "step": self.step,
                "rank_states": rank_states,
                "world_size": self.config.world_size,
                "dataset_fingerprint": self.paths["dataset_fingerprint"],
                "training_identity": self.spec.training_identity(),
                "provenance": provenance(),
                "wandb_id": fingerprint([self.run_name, self.paths["dataset_fingerprint"]])[:16],
                "wandb_name": self.run_name,
                "wandb_project": self.spec.wandb_project,
                "wandb_entity": self.spec.wandb_entity,
                "patch_hashes": self.patch_hashes,
                "run_name": self.run_name,
            }

        def _write_latest(self, model_state, optimizer_state, rank_states):
            latest_dir = self.run_dir / "resume"
            latest_dir.mkdir(parents=True, exist_ok=True)
            temporary = latest_dir / "latest.tmp"
            target = latest_dir / "latest.pt"
            payload = {
                **self._metadata(rank_states),
                "model": model_state,
                "optimizer": optimizer_state,
                "scheduler": self.lr_scheduler.state_dict(),
            }
            with temporary.open("wb") as handle:
                torch.save(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)

        def _weight_metadata(self):
            manifest = json.loads(Path(self.paths["manifest_path"]).read_text())
            return {
                "sft_config": self.spec.to_dict(),
                "norm_stats": manifest["norm_stat"],
                "dataset_manifest": manifest,
            }

        def _write_weights(self, model_state):
            checkpoint_root = self.run_dir / "checkpoints"
            checkpoint_root.mkdir(parents=True, exist_ok=True)
            final_dir = checkpoint_root / f"step_{self.step:06d}"
            temporary = Path(tempfile.mkdtemp(prefix=f".step_{self.step:06d}-", dir=checkpoint_root))
            try:
                transformer_dir = temporary / "transformer"
                transformer_dir.mkdir(parents=True)
                weights = {key: value.to(torch.bfloat16).contiguous() for key, value in model_state.items()}
                staged_weights = transformer_dir / "diffusion_pytorch_model.safetensors"
                save_file(weights, str(staged_weights))
                config = dict(self.transformer.config)
                config.pop("_name_or_path", None)
                config["attn_mode"] = "torch"
                staged_config = transformer_dir / "config.json"
                _write_json(staged_config, config)
                metadata = self._weight_metadata()
                _write_json(temporary / "sft_config.json", metadata["sft_config"])
                _write_json(temporary / "norm_stats.json", metadata["norm_stats"])
                _write_json(temporary / "dataset_manifest.json", metadata["dataset_manifest"])
                if final_dir.exists():
                    required = [
                        final_dir / "transformer" / "diffusion_pytorch_model.safetensors",
                        final_dir / "transformer" / "config.json",
                        final_dir / "sft_config.json",
                        final_dir / "norm_stats.json",
                        final_dir / "dataset_manifest.json",
                    ]
                    if not all(path.exists() for path in required):
                        raise FileExistsError(f"Incomplete checkpoint collision at {final_dir}")
                    for name, value in metadata.items():
                        if json.loads((final_dir / f"{name}.json").read_text()) != _jsonable(value):
                            raise FileExistsError(f"Refusing checkpoint collision at {final_dir}")
                    existing_weights = final_dir / "transformer" / "diffusion_pytorch_model.safetensors"
                    existing_config = final_dir / "transformer" / "config.json"
                    if _sha256(existing_weights) != _sha256(staged_weights) or _sha256(existing_config) != _sha256(staged_config):
                        raise FileExistsError(f"Refusing checkpoint collision at {final_dir}")
                    return
                os.replace(temporary, final_dir)
                temporary = None
            finally:
                if temporary is not None and temporary.exists():
                    shutil.rmtree(temporary)

        def save_checkpoint(self):
            model_state, optimizer_state, rank_states = self._collect_states()
            status = {"ok": True, "error": ""}
            if self.config.rank == 0:
                try:
                    if self.spec.save_weights_at(self.step):
                        self._write_weights(model_state)
                    self._write_latest(model_state, optimizer_state, rank_states)
                except Exception as exc:
                    status = {"ok": False, "error": f"Checkpoint save failed: {exc}"}
            _raise_status(status)
            if self.modal_volume is not None:
                status = {"ok": True, "error": ""}
                if self.config.rank == 0:
                    try:
                        self.modal_volume.commit()
                    except Exception as exc:
                        status = {"ok": False, "error": f"Modal volume commit failed: {exc}"}
                _raise_status(status)
            if _distributed():
                dist.barrier()

        def _validate_resume(self, metadata):
            if metadata["format_version"] != 1:
                raise ValueError("Unsupported checkpoint format")
            if not isinstance(metadata["step"], int) or metadata["step"] <= 0:
                raise ValueError("Checkpoint step must be positive")
            if metadata["world_size"] != self.config.world_size:
                raise ValueError("Checkpoint world size differs from the current run")
            if metadata["dataset_fingerprint"] != self.paths["dataset_fingerprint"]:
                raise ValueError("Checkpoint dataset fingerprint differs from the current run")
            if metadata["training_identity"] != self.spec.training_identity():
                raise ValueError("Checkpoint training identity differs from the current run")
            if metadata["provenance"] != provenance():
                raise ValueError("Checkpoint provenance differs from the current run")
            if metadata["patch_hashes"] != self.patch_hashes:
                raise ValueError("Checkpoint patch hashes differ from the current run")
            expected_id = fingerprint([self.run_name, self.paths["dataset_fingerprint"]])[:16]
            if metadata["run_name"] != self.run_name or metadata["wandb_id"] != expected_id:
                raise ValueError("Checkpoint run identity differs from the current run")
            if metadata["wandb_project"] != self.spec.wandb_project or metadata["wandb_entity"] != self.spec.wandb_entity:
                raise ValueError("Checkpoint W&B identity differs from the current run")
            rank_states = metadata["rank_states"]
            if not isinstance(rank_states, list) or len(rank_states) != self.config.world_size:
                raise ValueError("Checkpoint rank state count differs from world size")
            loader_length = len(self.train_loader)
            for record in rank_states:
                if not isinstance(record["data_epoch"], int) or record["data_epoch"] < 0:
                    raise ValueError("Checkpoint data epoch is invalid")
                if not isinstance(record["data_offset"], int) or not 0 <= record["data_offset"] <= loader_length:
                    raise ValueError("Checkpoint data offset is invalid")
            if metadata["step"] >= self.spec.num_steps:
                raise ValueError("Requested num_steps must exceed checkpoint.step")

        def resume_from_latest(self):
            path = self.run_dir / "resume" / "latest.pt"
            status = {"ok": True, "error": ""}
            payload = None
            if self.config.rank == 0:
                try:
                    if not path.exists():
                        raise FileNotFoundError(f"Missing full checkpoint: {path}")
                    payload = torch.load(path, map_location="cpu", weights_only=True)
                    self._validate_resume(payload)
                except Exception as exc:
                    status = {"ok": False, "error": f"Resume validation failed: {exc}"}
            _raise_status(status)
            metadata = _broadcast_object({key: value for key, value in payload.items() if key not in ("model", "optimizer", "scheduler")} if payload is not None else None)
            self._validate_resume(metadata)
            model_state = payload["model"] if self.config.rank == 0 else {}
            optimizer_state = payload["optimizer"] if self.config.rank == 0 else {}
            options = StateDictOptions(
                full_state_dict=True,
                broadcast_from_rank0=_distributed(),
                strict=True,
            )
            set_state_dict(
                self.transformer,
                self.optimizer,
                model_state_dict=model_state,
                optim_state_dict=optimizer_state,
                options=options,
            )
            scheduler_state = payload["scheduler"] if self.config.rank == 0 else None
            scheduler_state = _broadcast_object(scheduler_state)
            self.lr_scheduler.load_state_dict(scheduler_state)
            rank_states = metadata["rank_states"]
            record = rank_states[self.config.rank]
            self.step = metadata["step"]
            self.data_epoch = record["data_epoch"]
            self.data_offset = record["data_offset"]
            self.train_loader_iter = None
            self._position_data_iterator()
            self.train_loader.generator.set_state(record["loader_generator"])
            restore_rng(record["rng"])

        def _start_wandb(self):
            saved_rng = capture_rng()
            try:
                if self.config.rank != 0:
                    return
                import wandb

                run_id = fingerprint([self.run_name, self.paths["dataset_fingerprint"]])[:16]
                self.wandb_run = wandb.init(
                    project=self.spec.wandb_project,
                    entity=self.spec.wandb_entity,
                    id=run_id,
                    name=self.run_name,
                    resume="must" if self.resume_requested else "never",
                    allow_val_change=True,
                    save_code=False,
                    config={
                        "spec": self.spec.to_dict(),
                        "provenance": provenance(),
                        "paths": self.paths,
                        "patch_hashes": self.patch_hashes,
                    },
                )
                self.wandb_run.define_metric("train/optimizer_step")
                for metric in (
                    "train/video_loss",
                    "train/action_loss",
                    "train/loss",
                    "train/grad_norm",
                    "train/lr",
                    "train/step_seconds",
                    "train/episodes_per_second",
                    "train/peak_gpu_memory_bytes",
                ):
                    self.wandb_run.define_metric(metric, step_metric="train/optimizer_step")
                artifact = wandb.Artifact(f"{self.run_name}-provenance", type="sft-provenance")
                manifest = json.loads(Path(self.paths["manifest_path"]).read_text())
                artifact.metadata = {
                    "norm_stats": manifest["norm_stat"],
                    "spec": self.spec.to_dict(),
                    "provenance": provenance(),
                    "patch_hashes": self.patch_hashes,
                }
                artifact.add_file(self.paths["manifest_path"], name="dataset_manifest.json")
                run_metadata = self.run_dir / "run_metadata.json"
                if run_metadata.exists():
                    artifact.add_file(str(run_metadata), name="run_metadata.json")
                self.wandb_run.log_artifact(artifact)
            finally:
                restore_rng(saved_rng)

        def _log_step(self, totals, norm, duration):
            if self.config.rank != 0:
                return
            metrics = {
                "train/video_loss": totals[0].item(),
                "train/action_loss": totals[1].item(),
                "train/loss": totals.sum().item(),
                "train/grad_norm": norm.item(),
                "train/lr": self.lr_scheduler.get_last_lr()[0],
                "train/optimizer_step": self.step,
                "train/step_seconds": duration.item(),
                "train/episodes_per_second": 80 / duration.item(),
                "train/peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
            }
            if self.wandb_run is not None:
                self.wandb_run.log(metrics, step=self.step)

        def train(self):
            completed = False
            try:
                startup = {"ok": True, "error": ""}
                if self.config.rank == 0:
                    try:
                        self._start_wandb()
                        _write_status(self.run_dir, "running", self.step, self.spec.num_steps)
                    except Exception as exc:
                        startup = {"ok": False, "error": f"Training startup failed: {exc}"}
                _raise_status(startup)
                if self.config.rank != 0:
                    self._start_wandb()
                self.transformer.train()
                self.optimizer.zero_grad()
                while self.step < self.spec.num_steps:
                    _sync_cuda()
                    started = time.perf_counter()
                    totals = torch.zeros(2, device=self.device)
                    for microstep in range(self.gradient_accumulation_steps):
                        losses = self._train_step(self._get_next_batch(), microstep)
                        totals += torch.stack([losses["latent_loss"], losses["action_loss"]])
                    self.step += 1
                    totals /= self.config.world_size
                    if _distributed():
                        dist.all_reduce(totals)
                    _sync_cuda()
                    duration = torch.tensor(time.perf_counter() - started, device=self.device)
                    if _distributed():
                        dist.all_reduce(duration, op=dist.ReduceOp.MAX)
                    if not torch.isfinite(totals).all():
                        raise RuntimeError("Non-finite SFT loss")
                    norm = losses["total_norm"]
                    if hasattr(norm, "full_tensor"):
                        norm = norm.full_tensor()
                    self._log_step(totals, norm, duration)
                    if self.step % self.spec.save_interval == 0 or self.spec.save_weights_at(self.step):
                        self.save_checkpoint()
                completion = {"ok": True, "error": ""}
                if self.config.rank == 0:
                    try:
                        _write_status(self.run_dir, "completed", self.step, self.spec.num_steps)
                        if self.modal_volume is not None:
                            self.modal_volume.commit()
                    except Exception as exc:
                        completion = {"ok": False, "error": f"Completion publication failed: {exc}"}
                _raise_status(completion)
                completed = True
            finally:
                if self.wandb_run is not None:
                    self.wandb_run.finish(exit_code=0 if completed else 1)

    return SFTTrainer


def _load_prepared(path, spec):
    data = json.loads(Path(path).read_text())
    paths = data.get("paths", data)
    required = {"dataset_path", "manifest_path", "model_path", "empty_emb_path", "dataset_fingerprint"}
    if not required.issubset(paths):
        raise ValueError(f"Prepared paths missing keys: {sorted(required - set(paths))}")
    model_root = Path(paths["model_path"])
    transformer_root = model_root / "transformer"
    if model_root.name != MODEL_REVISION or not (transformer_root / "config.json").exists():
        raise ValueError("Prepared model path is not the pinned model snapshot")
    if not any((transformer_root / name).exists() for name in ("diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.safetensors.index.json")):
        raise ValueError("Prepared model transformer weights are missing")
    from script.lingbot_sft_data import validate_manifest

    manifest = json.loads(Path(paths["manifest_path"]).read_text())
    validate_manifest(manifest, spec)
    if manifest["fingerprint"] != paths["dataset_fingerprint"]:
        raise ValueError("Prepared dataset fingerprint does not match manifest")
    return paths


def _lock_run(run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = run_dir / ".run.lock"
    handle = lock_path.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError(f"Run directory is already locked: {run_dir}")
    return handle


def _parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--wandb-project", default=SFTConfig().wandb_project)
    parser.add_argument("--wandb-entity")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--modal-volume")
    return parser


def _run_identity(spec, paths, patch_hashes, world_size, run_name):
    return {
        "training_identity": spec.training_identity(),
        "provenance": provenance(),
        "paths": paths,
        "patch_hashes": patch_hashes,
        "world_size": world_size,
        "run_name": run_name,
        "wandb_project": spec.wandb_project,
        "wandb_entity": spec.wandb_entity,
    }


def main(argv=None):
    args = _parser().parse_args(argv)
    if not _RUN_NAME.fullmatch(args.run_name):
        raise SystemExit("run-name must match [A-Za-z0-9][A-Za-z0-9_-]{0,79}")
    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    if world_size not in (4, 8):
        raise SystemExit("torchrun must provide WORLD_SIZE=4 or WORLD_SIZE=8")
    spec = SFTConfig(num_steps=args.steps, wandb_project=args.wandb_project, wandb_entity=args.wandb_entity).validate()
    if not os.environ.get("WANDB_API_KEY"):
        raise SystemExit("WANDB_API_KEY must be set for training")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(args.upstream_root))
    sys.path.insert(0, str(args.upstream_root / "wan_va"))
    paths = _load_prepared(args.prepared, spec)
    run_dir = args.run_dir.resolve()
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    lock = None
    process_group = False
    try:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", timeout=timedelta(minutes=30))
        process_group = True
        patch_info = None
        patch_status = {"ok": True, "error": ""}
        if rank == 0:
            try:
                patch_info = apply_patch(args.upstream_root)
            except Exception as exc:
                patch_status = {"ok": False, "error": f"Upstream patch failed: {exc}"}
        _raise_status(patch_status)
        patch_info = _broadcast_object(patch_info)
        startup = {"ok": True, "error": ""}
        if rank == 0:
            try:
                existing = run_dir.exists() and any(run_dir.iterdir())
                if existing and not args.resume:
                    raise RuntimeError("Existing run requires --resume")
                if args.resume and not (run_dir / "resume" / "latest.pt").exists():
                    raise RuntimeError("--resume requires run_dir/resume/latest.pt")
                identity = _run_identity(spec, paths, patch_info["hashes"], world_size, args.run_name)
                lock = _lock_run(run_dir)
                metadata_path = run_dir / "run_metadata.json"
                if metadata_path.exists():
                    old = json.loads(metadata_path.read_text())
                    if old.get("identity") != identity:
                        raise RuntimeError("Run metadata differs from the selected immutable identity")
                else:
                    _write_json(metadata_path, {"spec": spec.to_dict(), "identity": identity})
            except Exception as exc:
                startup = {"ok": False, "error": f"Run startup failed: {exc}"}
        _raise_status(startup)
        modal_volume = None
        if args.modal_volume:
            import modal

            modal_volume = modal.Volume.from_name(args.modal_volume)
        config = _native_config(spec, paths, run_dir, rank, local_rank, world_size)
        trainer = _native_sft_class()(config, spec, paths, run_dir, patch_info["hashes"], args.run_name, args.resume, modal_volume)
        if args.resume:
            trainer.resume_from_latest()
        trainer.train()
    finally:
        if process_group:
            dist.destroy_process_group()
        if lock is not None:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            lock.close()


if __name__ == "__main__":
    main()
