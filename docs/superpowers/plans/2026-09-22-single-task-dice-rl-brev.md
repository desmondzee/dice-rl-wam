# Single-Task DICE-RL on Brev (Phase A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `script/lingbot_rl_train.py` train DICE-RL on one LIBERO-10 task from procedurally sampled initial states for 660k env steps, and ship the Brev scripts (`push`, `setup`, `train`, `sync`, `pull`) that run it on an on-demand H100 with W&B logging and a lean smoke.

**Architecture:** All changes stay inside the existing residual-RL stack (`script/lingbot_rl_*.py`) and its mocked-CPU test file. The RL protocol goes to version 3: `task_ids` selects the task(s), the training env is built with `init_states=False` so every reset samples placements from the BDDL regions, truncation no longer counts as terminal, the RLPD decay spans `rlpd_t_ratio` instead of the whole budget, the train-eval becomes 10 procedural episodes per checkpoint, a checkpoint is always written at the final step, and an optional `DICE_SYNC_CMD` hook fires after every checkpoint. The Brev side is five bash scripts that mirror the Modal image and launcher exactly (same LeRobot revision, same `uv sync` extras, same `script.lingbot_eval prepare` call), run training under tmux, and move artifacts with rsync and `modal volume put`.

**Tech Stack:** PyTorch (CPU in tests), numpy, pytest, bash, tmux, rsync, `uv`, Modal CLI (only for `volume get/put`), W&B. Local verification interpreter: `.cache/eval-venv/bin/python`. No GPU, no network, no `modal run`, no real LIBERO env in tests.

**Spec:** `docs/superpowers/specs/2026-09-22-single-task-dice-rl-design.md` (§3 decisions, §5.5 inline train-eval, §6 recipe v3, §7 Phase A, §8 gates, §12 costs). Phase B (Modal evaluation protocol v2, SFT baseline, analysis) is a separate plan and is **not** in scope here.

## Global Constraints

- Frozen prior is never invoked inside `_update_from_buffer` / `DiceResidualModel`; training must still run with the stub policy in tests.
- K stays 4: `RLConfig.k_candidates` remains pinned and validation keeps rejecting other values. Sampler pins (20/50, `video_exec_step=-1`), `source_run="libero30-sft"`, `checkpoint_step=600` remain hard errors.
- Unchanged hyperparameters: β=100, ε=−0.5, γ=0.99, n-step 3, UTD 10, τ=0.01, LR 1e-4, batch 256, ensemble 10, replay 100k, RLPD 0.5→0.1, zero-init actor, MC-return BC anchor, best-of-4 at collection.
- Do not edit `.cache/lerobot`, `script/lingbot_eval.py`, or `script/lingbot_eval_config.py` (AGENTS.md; those are Phase B). `EvalConfig` stays 1/20 episodes by stage.
- No comments or docstrings in new or modified code, including tests and bash (the user wants lean code). Shell scripts may keep a `#!/usr/bin/env bash` shebang and `set -euo pipefail` only.
- Secrets (`WANDB_API_KEY`, `HF_TOKEN`, `MODAL_TOKEN_ID/SECRET`) come only from the environment or the gitignored `brev/env.sh`; never in the repo, argv echoes, or logs.
- Verification for every Python task: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py` (72 tests pass today in ~35 s). Bash tasks: `.cache/eval-venv/bin/python -m pytest -q tests/test_brev_scripts.py`.
- Commit messages: one plain imperative sentence in repo style (e.g. "Store per-chunk candidate sets and Monte-Carlo returns in the DICE-RL replay buffer."), ending with the trailer line `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Old `resume/latest.pt` and `expert_features.pt` files from recipe v2 become intentionally unresumable/rebuildable (recipe fingerprint and expert recipe both change). No compatibility shims.

---

### Task 1: RL config v3 — task selection, 660k budget, `rlpd_t_ratio`, checkpoint cadence, validated ranges

**Files:**
- Modify: `script/lingbot_rl_config.py` (whole `RLConfig` dataclass, lines 10–133)
- Test: `tests/test_lingbot_rl.py` (lines 25–101: `test_config_pins_released_libero_sampler_and_step_600`, `test_config_rejects_sampler_and_recipe_drift`, `test_rlpd_ratio_anneals_over_env_steps`)

**Interfaces:**
- Consumes: `TASK_IDS` from `script.lingbot_eval_config` (tuple `0..9`).
- Produces: `RLConfig` fields `task_ids: tuple[int, ...]` (default all ten), `online_env_steps=660_000`, `rlpd_t_ratio=320_000`, `train_eval_every=80_000`, `train_eval_episodes_per_task=10`; `RLConfig.rlpd_expert_ratio(env_steps)` decays over `rlpd_t_ratio` then holds 0.1; `protocol()["version"] == 3`, `protocol()["task_ids"]`, `protocol()["rlpd_t_ratio"]`, `protocol()["training_init_states"] == "procedural"`, `protocol()["truncation"] == "bootstrap"`; `default_run_name` is `libero30-sft-dice-task{t}` for a single task. Later tasks read `config.task_ids` and `config.train_eval_episodes_per_task`.

- [ ] **Step 1: Update the existing config tests to the v3 recipe**

In `tests/test_lingbot_rl.py`, inside `test_config_pins_released_libero_sampler_and_step_600`, replace these four assertions:

```python
    assert cfg.online_env_steps == 100_000
```
→
```python
    assert cfg.online_env_steps == 660_000
    assert cfg.task_ids == tuple(range(10))
    assert cfg.rlpd_t_ratio == 320_000
```
and
```python
    assert proto["train_eval_every"] == 25_000
    assert proto["train_eval_episodes_per_task"] == 1
```
→
```python
    assert proto["train_eval_every"] == 80_000
    assert proto["train_eval_episodes_per_task"] == 10
    assert proto["task_ids"] == list(range(10))
    assert proto["rlpd_t_ratio"] == 320_000
    assert proto["training_init_states"] == "procedural"
    assert proto["truncation"] == "bootstrap"
```
and
```python
    assert proto["version"] == 2
```
→
```python
    assert proto["version"] == 3
```

Replace the parametrize list of `test_config_rejects_sampler_and_recipe_drift` with:

```python
@pytest.mark.parametrize("changes", [
    {"video_steps": 3},
    {"action_steps": 10},
    {"video_exec_step": 12},
    {"checkpoint_step": 400},
    {"source_run": "other-run"},
    {"k_candidates": 16},
    {"train_eval_every": 0},
    {"train_eval_every": 700_000},
    {"train_eval_episodes_per_task": 0},
    {"online_env_steps": 0},
    {"rlpd_t_ratio": 0},
    {"rlpd_t_ratio": 700_000},
    {"task_ids": ()},
    {"task_ids": (10,)},
    {"task_ids": (0, 0)},
])
```

Replace `test_rlpd_ratio_anneals_over_env_steps` with:

```python
def test_rlpd_ratio_anneals_over_t_ratio_then_holds():
    cfg = RLConfig().validate()
    assert cfg.rlpd_expert_ratio(0) == pytest.approx(0.5)
    assert cfg.rlpd_expert_ratio(160_000) == pytest.approx(0.3)
    assert cfg.rlpd_expert_ratio(320_000) == pytest.approx(0.1)
    assert cfg.rlpd_expert_ratio(660_000) == pytest.approx(0.1)


def test_config_single_task_from_json_list_and_run_name():
    cfg = RLConfig(task_ids=[4]).validate()
    assert cfg.task_ids == (4,)
    assert cfg.default_run_name == "libero30-sft-dice-task4"
    assert RLConfig().default_run_name == "libero30-sft-dice-baseline"
    assert RLConfig(**{k: v for k, v in cfg.to_dict().items() if k != "protocol"}).validate().task_ids == (4,)
```

- [ ] **Step 2: Run the config tests to verify they fail**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py -k "config or rlpd"`
Expected: FAIL — `AssertionError` on `online_env_steps`, `TypeError: __init__() got an unexpected keyword argument 'task_ids'`, and `ValueError` not raised for the new drift cases.

- [ ] **Step 3: Rewrite `RLConfig`**

Replace the dataclass body in `script/lingbot_rl_config.py` (keep the imports and `main()`):

```python
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
    task_ids: tuple = tuple(TASK_IDS)
    online_env_steps: int = 660_000
    rlpd_t_ratio: int = 320_000
    train_eval_every: int = 80_000
    train_eval_episodes_per_task: int = 10

    def __post_init__(self):
        object.__setattr__(self, "task_ids", tuple(int(task) for task in self.task_ids))

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
        if not self.task_ids or len(set(self.task_ids)) != len(self.task_ids) \
                or any(task not in TASK_IDS for task in self.task_ids):
            raise ValueError("task_ids must be distinct LIBERO-10 task ids")
        if type(self.online_env_steps) is not int or self.online_env_steps < 1:
            raise ValueError("online_env_steps must be a positive integer")
        if type(self.rlpd_t_ratio) is not int or not 1 <= self.rlpd_t_ratio <= self.online_env_steps:
            raise ValueError("rlpd_t_ratio must lie in [1, online_env_steps]")
        if type(self.train_eval_every) is not int or not 1 <= self.train_eval_every <= self.online_env_steps:
            raise ValueError("train_eval_every must lie in [1, online_env_steps]")
        if type(self.train_eval_episodes_per_task) is not int or self.train_eval_episodes_per_task < 1:
            raise ValueError("train_eval_episodes_per_task must be positive")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("Invalid seed")
        if not self.wandb_project:
            raise ValueError("W&B project is required")
        return self

    @property
    def default_run_name(self):
        if len(self.task_ids) == 1:
            return f"{self.source_run}-dice-task{self.task_ids[0]}"
        return f"{self.source_run}-dice-baseline"

    def rlpd_expert_ratio(self, env_steps):
        t = min(max(env_steps, 0), self.rlpd_t_ratio) / self.rlpd_t_ratio
        return 0.5 + (0.1 - 0.5) * t

    def protocol(self):
        self.validate()
        return {
            "version": 3,
            "suite": "libero_10",
            "task_ids": list(self.task_ids),
            "training_init_states": "procedural",
            "truncation": "bootstrap",
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
            "rlpd_t_ratio": self.rlpd_t_ratio,
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
```

- [ ] **Step 4: Run the config tests, then the full file**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py -k "config or rlpd"`
Expected: PASS (all parametrized drift cases raise `ValueError`).

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py`
Expected: PASS. (`test_train_eval_fires_after_crossing_chunk_stride` still passes because it calls `train_eval_schedule(100_000, 25_000)` directly; Task 3 changes that.)

Run: `.cache/eval-venv/bin/python -m script.lingbot_rl_config`
Expected: JSON with `"version": 3`, `"task_ids": [0, ..., 9]`, `"online_env_steps": 660000`.

- [ ] **Step 5: Commit**

```bash
git add script/lingbot_rl_config.py tests/test_lingbot_rl.py
git commit -m "$(cat <<'MSG'
Bump the DICE-RL recipe to version 3 with task selection, a 660k budget, an RLPD decay window, and an 80k checkpoint cadence.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
MSG
)"
```

---

### Task 2: Expert demos filtered to the configured tasks

**Files:**
- Modify: `script/lingbot_rl_data.py` (`EXPERT_RECIPE` line 13, `expert_fingerprint` lines 24–43, `featurize_experts` lines 126–145, `load_manifest_episodes` lines 148–196)
- Test: `tests/test_lingbot_rl.py`

**Interfaces:**
- Consumes: manifest dict with `episodes: [{"episode_index": int, "tasks": [instruction]}]` (300 entries) and `task_to_id: {instruction: task_id}`.
- Produces: `select_manifest_episodes(manifest, task_to_id, task_ids=None) -> list[dict]`; `load_manifest_episodes(dataset_root, manifest, task_to_id=None, norm=None, task_ids=None)`; `featurize_experts(policy, dataset, manifest=None, norm=None, cache_path=None, task_ids=None)`; `expert_fingerprint(manifest, norm, episodes=None, task_ids=None)`. Task 3 passes `task_ids=config.task_ids` to both loaders.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_lingbot_rl.py`:

```python
def _manifest_for_tasks(per_task=3):
    names = [f"instruction {task}" for task in range(10)]
    episodes = []
    index = 0
    for name in names:
        for _ in range(per_task):
            episodes.append({"episode_index": index, "tasks": [name]})
            index += 1
    return {"episodes": episodes, "fingerprint": "abc"}, {name: task for task, name in enumerate(names)}


def test_select_manifest_episodes_filters_by_task_ids():
    from script.lingbot_rl_data import select_manifest_episodes

    manifest, task_to_id = _manifest_for_tasks()
    all_episodes = select_manifest_episodes(manifest, task_to_id)
    assert len(all_episodes) == 30
    only_four = select_manifest_episodes(manifest, task_to_id, task_ids=(4,))
    assert [episode["episode_index"] for episode in only_four] == [12, 13, 14]
    assert select_manifest_episodes(manifest, task_to_id, task_ids=(0, 9))[-1]["episode_index"] == 29
    with pytest.raises(ValueError, match="No SFT demonstrations"):
        select_manifest_episodes({"episodes": [], "fingerprint": "x"}, task_to_id, task_ids=(4,))


def test_expert_fingerprint_depends_on_task_ids_and_recipe_v2():
    from script.lingbot_rl_data import EXPERT_RECIPE, expert_fingerprint

    norm = _eval_normalization()
    manifest = {"fingerprint": "abc"}
    assert EXPERT_RECIPE == "dice-rl-expert-v2"
    assert expert_fingerprint(manifest, norm, task_ids=(0,)) != expert_fingerprint(manifest, norm, task_ids=(4,))
    assert expert_fingerprint(manifest, norm, task_ids=(4, 0)) == expert_fingerprint(manifest, norm, task_ids=[0, 4])
    assert expert_fingerprint(manifest, norm, task_ids=None) != expert_fingerprint(manifest, norm, task_ids=(0,))


def test_featurize_experts_cache_keyed_by_task_ids(tmp_path):
    from script.lingbot_rl_data import featurize_experts

    class Policy:
        config = None

        def reset(self):
            return None

        def extract_critic_state(self, batch):
            return torch.zeros(1, STATE_DIM)

    norm = _eval_normalization()
    episode = {"actions": np.zeros((12, USED_DOF), np.float32), "task": "t", "task_id": 4, "success": True}
    cache = tmp_path / "expert_features.pt"
    rows = featurize_experts(Policy(), [episode], {"fingerprint": "abc"}, norm, cache, task_ids=(4,))
    assert len(rows) == 1
    cached = featurize_experts(Policy(), [], {"fingerprint": "abc"}, norm, cache, task_ids=(4,))
    assert len(cached) == 1
    assert np.array_equal(cached[0]["a"], rows[0]["a"])
    assert float(cached[0]["done"]) == 1.0
    assert featurize_experts(Policy(), [], {"fingerprint": "abc"}, norm, cache, task_ids=(0,)) == []
```

`_eval_normalization` already exists at line 235 of the test file and returns the `{"q01": ..., "q99": ...}` dict used by the featurization tests.

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py -k "select_manifest or fingerprint_depends or cache_keyed"`
Expected: FAIL — `ImportError: cannot import name 'select_manifest_episodes'`, `TypeError: expert_fingerprint() got an unexpected keyword argument 'task_ids'`.

- [ ] **Step 3: Implement**

In `script/lingbot_rl_data.py`:

```python
EXPERT_RECIPE = "dice-rl-expert-v2"
```

Replace `expert_fingerprint`:

```python
def expert_fingerprint(manifest, norm, episodes=None, task_ids=None):
    proto = RLConfig().protocol()
    payload = {
        "recipe": EXPERT_RECIPE,
        "video_steps": proto["video_steps"],
        "action_steps": proto["action_steps"],
        "video_exec_step": proto["video_exec_step"],
        "checkpoint_step": 600,
        "source_run": "libero30-sft",
        "q01": [float(x) for x in norm["q01"]],
        "q99": [float(x) for x in norm["q99"]],
        "task_ids": sorted(int(task) for task in task_ids) if task_ids is not None else None,
    }
    if manifest is not None:
        payload["manifest"] = manifest.get("fingerprint", manifest)
    elif episodes is not None:
        payload["episodes"] = [
            {"task": ep.get("task"), "task_id": int(ep.get("task_id", -1)), "n": int(len(ep["actions"]))}
            for ep in episodes
        ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
```

Change the `featurize_experts` signature and fingerprint call:

```python
def featurize_experts(policy, dataset, manifest=None, norm=None, cache_path=None, task_ids=None):
    if norm is None:
        raise ValueError("Checkpoint norm_stats.json is required")
    episodes = list(dataset)
    cache_path = Path(cache_path) if cache_path is not None else None
    fingerprint = expert_fingerprint(manifest, norm, episodes, task_ids)
```
(rest of the function unchanged).

Add `select_manifest_episodes` above `load_manifest_episodes` and use it:

```python
def select_manifest_episodes(manifest, task_to_id, task_ids=None):
    episodes = manifest["episodes"]
    if task_ids is None:
        return list(episodes)
    wanted = {int(task) for task in task_ids}
    selected = [episode for episode in episodes if task_to_id[episode["tasks"][0]] in wanted]
    if not selected:
        raise ValueError("No SFT demonstrations for the configured task_ids")
    return selected


def load_manifest_episodes(dataset_root, manifest, task_to_id=None, norm=None, task_ids=None):
    import pyarrow.parquet as pq

    if norm is None:
        raise ValueError("Checkpoint norm_stats.json is required")
    root = Path(dataset_root)
    info = json.loads((root / "meta" / "info.json").read_text())
    if len(manifest["episodes"]) != 300:
        raise ValueError("Expected 300 SFT demonstrations in the checkpoint manifest")
    if task_to_id is None:
        task_names = []
        for episode in manifest["episodes"]:
            name = episode["tasks"][0]
            if name not in task_names:
                task_names.append(name)
        task_to_id = {name: index for index, name in enumerate(task_names)}
    episodes = select_manifest_episodes(manifest, task_to_id, task_ids)
    loaded = []
    for episode in episodes:
```
Delete the old `episodes = manifest["episodes"]` / `if len(episodes) != 300` lines and the docstring; the loop body is unchanged.

- [ ] **Step 4: Run the tests**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py -k "select_manifest or fingerprint_depends or cache_keyed or expert or featurize"`
Expected: PASS.

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add script/lingbot_rl_data.py tests/test_lingbot_rl.py
git commit -m "$(cat <<'MSG'
Filter RLPD expert demos to the configured tasks and key the feature cache by task set.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
MSG
)"
```

---

### Task 3: Train loop — single task, procedural initial states, truncation bootstrap, final-step checkpoint, 10-episode procedural train-eval

**Files:**
- Modify: `script/lingbot_rl_train.py` (`make_env` 34–40, `train_eval_schedule` 59–62, `_train_eval` 245–265, `train` 285–496)
- Test: `tests/test_lingbot_rl.py`

**Interfaces:**
- Consumes: `RLConfig.task_ids`, `RLConfig.train_eval_episodes_per_task` (Task 1); `load_manifest_episodes(..., task_ids=)`, `featurize_experts(..., task_ids=)` (Task 2); `run_episode(policy, env, entry, instruction, norm, device)` from `script.lingbot_eval` (unchanged; it sets `env.init_state_id = entry["init_state_id"]`, which `LiberoEnv` ignores when `init_states=False`).
- Produces: `make_env(task_id, suite=None, init_states=True)`; `train_eval_plan(task_id, episodes, seed=42) -> list[{"task_id","episode_index","init_state_id","seed"}]`; `train_eval_schedule(budget, every)` always ends with `budget`; `_train_eval(policy, tasks, norm, device, suite, env_steps, output_dir, episodes)` writing `train_eval/step_XXXXXX/task_XX/episode_YYY.json` and returning `per_task_success` as rates. Module-level test helpers `_stub_policy()` and `_StubEnv` are defined here and reused by Task 4.

- [ ] **Step 1: Write the failing tests**

Update the existing `test_train_eval_fires_after_crossing_chunk_stride` (line 605): change its last line

```python
    assert train_eval_schedule(12, 25_000) == [0]
```
→
```python
    assert train_eval_schedule(12, 25_000) == [0, 12]
    assert train_eval_schedule(660_000, 80_000)[-3:] == [560_000, 640_000, 660_000]
    assert train_eval_schedule(640_000, 80_000)[-2:] == [560_000, 640_000]
```

Append module-level helpers and new tests:

```python
def _stub_policy(success_state=None):
    class StubPolicy:
        def reset(self):
            self._executed_actions = None

        def extract_critic_state(self, batch):
            return torch.zeros(1, STATE_DIM)

        def decode_candidates(self, batch, k=4, **kwargs):
            return {
                "s": torch.zeros(1, STATE_DIM),
                "z": torch.zeros(k, HORIZON, ACTION_DIM),
                "a_base": torch.zeros(k, HORIZON, ACTION_DIM),
                "video_noise": torch.zeros(1),
                "first_chunk": True,
            }

        def commit_executed(self, chunk, first_chunk=False):
            self._executed_actions = chunk

        def select_action(self, batch):
            return torch.zeros(1, 7)

        def observe_env_step(self, batch):
            return None

    return StubPolicy()


class _StubEnv:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.steps = 0
        self.init_state_id = 0
        self.seeds = []
        _StubEnv.instances.append(self)

    def _frame(self, seed):
        frame = np.full((128, 128, 3), ((seed or 0) * 7) % 251, np.uint8)
        return {"pixels": {"image": frame, "image2": frame}}

    def reset(self, seed=None):
        self.steps = 0
        self.seeds.append(seed)
        return self._frame(seed), {}

    def step(self, action):
        self.steps += 1
        return self._frame(self.seeds[-1]), 0.0, False, self.steps >= 12, {"is_success": False}

    def close(self):
        return None


def _fake_wandb(monkeypatch, logs):
    import sys

    monkeypatch.setitem(
        sys.modules,
        "wandb",
        type("W", (), {
            "init": staticmethod(lambda **k: type("R", (), {
                "id": "unit-run", "log": logs.append, "summary": {}, "finish": lambda **k: None,
            })()),
            "finish": staticmethod(lambda **k: None),
        })(),
    )


def _prepared(tmp_path, tasks=range(10)):
    prepared = {
        "tasks": [{"task_id": task, "instruction": f"task-{task}", "initial_state_count": 50} for task in tasks],
        "normalization": {"q01": [-1.0] * 7 + [0.0] * 23, "q99": [1.0] * 7 + [0.0] * 23},
        "checkpoint": None, "model_path": None, "architecture": {}, "assets_path": None,
        "source_run": "libero30-sft", "checkpoint_step": 600,
    }
    path = tmp_path / "prepared.json"
    path.write_text(__import__("json").dumps(prepared))
    return path


def test_make_env_passes_init_states_flag(monkeypatch):
    import script.lingbot_rl_train as train

    _StubEnv.instances.clear()
    monkeypatch.setattr(train, "LiberoEnv", _StubEnv)
    train.make_env(3, None)
    train.make_env(3, None, init_states=False)
    assert _StubEnv.instances[0].kwargs["init_states"] is True
    assert _StubEnv.instances[1].kwargs["init_states"] is False
    assert _StubEnv.instances[1].kwargs["hard_reset"] is True
    assert _StubEnv.instances[1].kwargs["task_id"] == 3


def test_train_eval_plan_is_fixed_across_calls_and_distinct_per_episode():
    from script.lingbot_rl_train import train_eval_plan

    plan = train_eval_plan(4, 10)
    assert [entry["episode_index"] for entry in plan] == list(range(10))
    assert all(entry["task_id"] == 4 for entry in plan)
    assert len({entry["seed"] for entry in plan}) == 10
    assert plan == train_eval_plan(4, 10)
    assert plan[0]["seed"] != train_eval_plan(0, 1)[0]["seed"]


def test_train_single_task_uses_procedural_envs_and_only_that_task(tmp_path, monkeypatch):
    import script.lingbot_rl_train as train

    _StubEnv.instances.clear()
    logs = []
    _fake_wandb(monkeypatch, logs)
    monkeypatch.setattr(train, "load_residual_policy", lambda *a, **k: _stub_policy())
    monkeypatch.setattr(train, "LiberoEnv", _StubEnv)
    config = RLConfig(task_ids=(4,))
    train.train(config=config, output_dir=tmp_path, run_name="unit", max_env_steps=24,
                prepared_path=_prepared(tmp_path))
    assert _StubEnv.instances
    assert {env.kwargs["task_id"] for env in _StubEnv.instances} == {4}
    assert all(env.kwargs["init_states"] is False for env in _StubEnv.instances)
    assert len({seed for env in _StubEnv.instances for seed in env.seeds}) == len(_StubEnv.instances)


def test_train_rejects_prepared_without_configured_task(tmp_path, monkeypatch):
    import script.lingbot_rl_train as train

    logs = []
    _fake_wandb(monkeypatch, logs)
    monkeypatch.setattr(train, "load_residual_policy", lambda *a, **k: _stub_policy())
    monkeypatch.setattr(train, "LiberoEnv", _StubEnv)
    with pytest.raises(ValueError, match="task_ids"):
        train.train(config=RLConfig(task_ids=(4,)), output_dir=tmp_path, run_name="unit",
                    max_env_steps=12, prepared_path=_prepared(tmp_path, tasks=(0, 1)))


def test_truncated_episode_rows_are_not_terminal(tmp_path, monkeypatch):
    import script.lingbot_rl_train as train

    logs = []
    _fake_wandb(monkeypatch, logs)
    monkeypatch.setattr(train, "load_residual_policy", lambda *a, **k: _stub_policy())
    monkeypatch.setattr(train, "LiberoEnv", _StubEnv)
    train.train(config=RLConfig(task_ids=(0,)), output_dir=tmp_path, run_name="unit", max_env_steps=12,
                prepared_path=_prepared(tmp_path))
    resume = torch.load(tmp_path / "resume" / "latest.pt", map_location="cpu", weights_only=False)
    rows = [row for row in resume["replay"]["data"] if float(row["is_expert"]) == 0.0]
    assert rows
    assert all(float(row["done"]) == 0.0 for row in rows)
    assert all(float(row["mc_return"]) == 0.0 for row in rows)


def test_successful_episode_row_is_terminal(tmp_path, monkeypatch):
    import script.lingbot_rl_train as train

    class WinEnv(_StubEnv):
        def step(self, action):
            self.steps += 1
            return self._frame(self.seeds[-1]), 0.0, False, False, {"is_success": self.steps >= 5}

    logs = []
    _fake_wandb(monkeypatch, logs)
    monkeypatch.setattr(train, "load_residual_policy", lambda *a, **k: _stub_policy())
    monkeypatch.setattr(train, "LiberoEnv", WinEnv)
    train.train(config=RLConfig(task_ids=(0,)), output_dir=tmp_path, run_name="unit", max_env_steps=5,
                prepared_path=_prepared(tmp_path))
    resume = torch.load(tmp_path / "resume" / "latest.pt", map_location="cpu", weights_only=False)
    rows = [row for row in resume["replay"]["data"] if float(row["is_expert"]) == 0.0]
    assert float(rows[-1]["done"]) == 1.0
    assert float(rows[-1]["reward"]) == 1.0
    assert float(rows[-1]["mc_return"]) == 1.0


def test_train_eval_runs_configured_procedural_episodes(tmp_path, monkeypatch):
    import script.lingbot_rl_train as train

    _StubEnv.instances.clear()
    monkeypatch.setattr(train, "LiberoEnv", _StubEnv)
    monkeypatch.setattr("script.lingbot_eval.run_episode",
                        lambda policy, env, entry, instruction, norm, device: {**entry, "success": entry["episode_index"] % 2 == 0})
    tasks = [{"task_id": 4, "instruction": "task-4", "initial_state_count": 50}]
    summary = train._train_eval(_stub_policy(), tasks, {}, "cpu", None, 80_000, tmp_path, 10)
    assert summary["per_task_success"] == {"4": 0.5}
    assert summary["macro_success_rate"] == 0.5
    assert len(_StubEnv.instances) == 1
    assert _StubEnv.instances[0].kwargs["init_states"] is False
    assert sorted(path.name for path in (tmp_path / "train_eval" / "step_080000" / "task_04").iterdir()) == \
        [f"episode_{index:03d}.json" for index in range(10)]
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py -k "make_env or train_eval_plan or single_task or rejects_prepared or truncated or successful_episode or configured_procedural or crossing_chunk"`
Expected: FAIL — `TypeError: make_env() got an unexpected keyword argument 'init_states'`, `ImportError: train_eval_plan`, `[0] != [0, 12]`, `done == 1.0` on truncated rows, `_train_eval() takes 7 positional arguments`.

- [ ] **Step 3: Implement**

In `script/lingbot_rl_train.py` add `import hashlib` to the imports, then:

```python
def make_env(task_id, suite=None, init_states=True):
    return _env_class()(
        task_suite=suite, task_id=task_id, task_suite_name="libero_10",
        episode_length=520, observation_height=128, observation_width=128, obs_type="pixels",
        init_states=init_states, n_envs=1, num_steps_wait=10, control_freq=20, control_mode="relative",
        hard_reset=True,
    )
```

```python
def train_eval_schedule(budget, every):
    if every < 1:
        raise ValueError("train_eval_every must be positive")
    points = set(range(0, budget + 1, every))
    points.add(int(budget))
    return sorted(points)


def train_eval_plan(task_id, episodes, seed=42):
    plan = []
    for index in range(int(episodes)):
        identity = ["libero_10_procedural", int(seed), int(task_id), index]
        digest = hashlib.sha256(json.dumps(identity).encode()).digest()[:4]
        plan.append({"task_id": int(task_id), "episode_index": index, "init_state_id": index,
                     "seed": int.from_bytes(digest, "big")})
    return plan
```

Replace `_train_eval`:

```python
def _train_eval(policy, tasks, norm, device, suite, env_steps, output_dir, episodes):
    from script.lingbot_eval import run_episode

    rows = []
    for task in tasks:
        env = make_env(task["task_id"], suite, init_states=False)
        try:
            for entry in train_eval_plan(task["task_id"], episodes):
                row = run_episode(policy, env, entry, task["instruction"], norm, device)
                rows.append(row)
                _write_json(
                    Path(output_dir) / "train_eval" / f"step_{env_steps:06d}" / f"task_{task['task_id']:02d}"
                    / f"episode_{entry['episode_index']:03d}.json",
                    row,
                )
        finally:
            env.close()
    per_task = {}
    for task in tasks:
        task_rows = [row for row in rows if int(row["task_id"]) == task["task_id"]]
        per_task[str(task["task_id"])] = float(np.mean([bool(row["success"]) for row in task_rows]))
    return {"env_steps": env_steps, "per_task_success": per_task,
            "macro_success_rate": float(np.mean([bool(row["success"]) for row in rows]))}
```

Inside `train(...)`:

1. Replace `tasks = prepared["tasks"]` with:
```python
    tasks = [task for task in prepared["tasks"] if task["task_id"] in config.task_ids]
```
2. In the expert-ingest block, pass the task filter through both loaders:
```python
            episodes = load_manifest_episodes(
                dataset_root, read_json(Path(prepared["checkpoint"]) / "dataset_manifest.json"),
                task_to_id, norm, task_ids=config.task_ids)
            _ingest_experts(buffer, featurize_experts(
                policy, episodes, read_json(Path(prepared["checkpoint"]) / "dataset_manifest.json"), norm,
                cache_path, task_ids=config.task_ids))
        elif cache_path.is_file():
            _ingest_experts(buffer, featurize_experts(
                policy, [], read_json(Path(prepared["checkpoint"]) / "dataset_manifest.json"), norm,
                cache_path, task_ids=config.task_ids))
```
3. After the `describe_suite` block, filter and check:
```python
        suite, suite_tasks = describe_suite(prepared["assets_path"])
        tasks = [task for task in suite_tasks if task["task_id"] in config.task_ids]
    if not tasks:
        raise ValueError("Prepared tasks do not cover the configured task_ids")
```
4. Change the expert-missing message:
```python
            raise RuntimeError("Expert features missing; RLPD requires the SFT demos for task_ids")
```
5. In `maybe_eval`, pass the episode count and use a procedural env for sharpening:
```python
            summary = _train_eval(policy, tasks, norm, device, suite, env_steps, output_dir,
                                  config.train_eval_episodes_per_task)
            ...
            task = tasks[0]
            env = make_env(task["task_id"], suite, init_states=False)
```
6. Replace the episode head of the collection loop:
```python
    while env_steps < budget:
        task = tasks[int(np.random.randint(len(tasks)))]
        task_id = task["task_id"]
        env = make_env(task_id, suite, init_states=False)
```
7. Replace the termination logic inside the env-action loop and after it:
```python
                chunk_reward = 0.0
                done = 0.0
                episode_over = False
                executed_n = 0
                for index in range(n_env):
                    action = decode_action(env_actions[:, index, :].reshape(1, 7), norm)
                    observation, _, terminated, truncated, info = env.step(action)
                    policy.observe_env_step(observation_batch(observation, task["instruction"], device))
                    executed_n += 1
                    episode_length += 1
                    env_steps += 1
                    if env_success(info) and not saw_success:
                        chunk_reward = 1.0
                        saw_success = True
                        episode_success = 1.0
                    if saw_success or terminated:
                        done = 1.0
                    if done or truncated or episode_length >= 520 or env_steps >= budget:
                        episode_over = True
                        break
```
and at the bottom of the chunk loop:
```python
                if episode_over:
                    break
            buffer.finalize_episode()
```

- [ ] **Step 4: Run the tests**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py -k "make_env or train_eval_plan or single_task or rejects_prepared or truncated or successful_episode or configured_procedural or crossing_chunk"`
Expected: PASS.

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py`
Expected: PASS (the two pre-existing mocked train tests still pass: their `StubEnv` accepts `**kwargs` and returns `truncated=True` at step 12, so their rows now carry `done=0`, which they do not assert on).

- [ ] **Step 5: Commit**

```bash
git add script/lingbot_rl_train.py tests/test_lingbot_rl.py
git commit -m "$(cat <<'MSG'
Train DICE-RL on the configured tasks from procedural initial states, bootstrap through truncation, and evaluate ten procedural episodes per checkpoint.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
MSG
)"
```

---

### Task 4: `DICE_SYNC_CMD` checkpoint hook and the `check-inits` operation

**Files:**
- Modify: `script/lingbot_rl_train.py` (`maybe_eval` checkpoint block, end-of-run save, `main`)
- Test: `tests/test_lingbot_rl.py`

**Interfaces:**
- Consumes: `_stub_policy()`, `_StubEnv`, `_fake_wandb`, `_prepared` from Task 3; `make_env(..., init_states=False)`.
- Produces: `run_sync_hook(output_dir, env_steps) -> int | None` (returns the hook's exit code, `None` when `DICE_SYNC_CMD` is unset; never raises); `procedural_reset_difference(env, seeds=(1, 2)) -> float`; CLI operation `check-inits --task-id T [--prepared-path P]` printing `{"task_id", "mean_abs_pixel_diff", "ok"}` and exiting 1 when `ok` is false. Task 5's `train.sh` sets `DICE_SYNC_CMD`, and `setup.sh` runs `check-inits`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_lingbot_rl.py`:

```python
def test_sync_hook_runs_after_final_checkpoint_with_step_env(tmp_path, monkeypatch):
    import script.lingbot_rl_train as train

    marker = tmp_path / "hook.txt"
    monkeypatch.setenv("DICE_SYNC_CMD", f'printf "%s %s" "$DICE_STEP" "$DICE_OUTPUT_DIR" > "{marker}"')
    logs = []
    _fake_wandb(monkeypatch, logs)
    monkeypatch.setattr(train, "load_residual_policy", lambda *a, **k: _stub_policy())
    monkeypatch.setattr(train, "LiberoEnv", _StubEnv)
    train.train(config=RLConfig(task_ids=(0,)), output_dir=tmp_path, run_name="unit", max_env_steps=12,
                prepared_path=_prepared(tmp_path))
    assert marker.read_text() == f"12 {tmp_path}"


def test_sync_hook_failure_is_reported_not_raised(tmp_path, monkeypatch, capsys):
    from script.lingbot_rl_train import run_sync_hook

    monkeypatch.delenv("DICE_SYNC_CMD", raising=False)
    assert run_sync_hook(tmp_path, 5) is None
    monkeypatch.setenv("DICE_SYNC_CMD", "exit 3")
    assert run_sync_hook(tmp_path, 5) == 3
    assert "sync hook exited 3" in capsys.readouterr().err


def test_procedural_reset_difference_detects_identical_placements():
    from script.lingbot_rl_train import procedural_reset_difference

    varied = _StubEnv()
    assert procedural_reset_difference(varied, seeds=(1, 2)) > 0.5
    assert varied.seeds == [1, 2]

    class Frozen(_StubEnv):
        def reset(self, seed=None):
            self.seeds.append(seed)
            return self._frame(7), {}

    assert procedural_reset_difference(Frozen(), seeds=(1, 2)) == 0.0


def test_check_inits_cli_exits_nonzero_on_frozen_env(tmp_path, monkeypatch, capsys):
    import script.lingbot_rl_train as train

    class Frozen(_StubEnv):
        def reset(self, seed=None):
            self.seeds.append(seed)
            return self._frame(7), {}

    monkeypatch.setattr(train, "LiberoEnv", Frozen)
    monkeypatch.setattr("sys.argv", ["prog", "check-inits", "--task-id", "4"])
    with pytest.raises(SystemExit) as exc:
        train.main()
    assert exc.value.code == 1
    payload = __import__("json").loads(capsys.readouterr().out)
    assert payload == {"task_id": 4, "mean_abs_pixel_diff": 0.0, "ok": False}
    monkeypatch.setattr(train, "LiberoEnv", _StubEnv)
    monkeypatch.setattr("sys.argv", ["prog", "check-inits", "--task-id", "4"])
    train.main()
    assert __import__("json").loads(capsys.readouterr().out)["ok"] is True
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py -k "sync_hook or procedural_reset or check_inits"`
Expected: FAIL — `ImportError: run_sync_hook`, `FileNotFoundError: hook.txt`, `argparse` error `invalid choice: 'check-inits'`.

- [ ] **Step 3: Implement**

In `script/lingbot_rl_train.py` add `import subprocess` and `import sys` to the imports. Add after `save_inference_checkpoints`:

```python
def run_sync_hook(output_dir, env_steps):
    command = os.environ.get("DICE_SYNC_CMD", "").strip()
    if not command:
        return None
    env = {**os.environ, "DICE_OUTPUT_DIR": str(output_dir), "DICE_STEP": str(int(env_steps))}
    result = subprocess.run(command, shell=True, env=env)
    if result.returncode != 0:
        print(f"sync hook exited {result.returncode}", file=sys.stderr)
    return result.returncode


def procedural_reset_difference(env, seeds=(1, 2)):
    frames = []
    for seed in seeds:
        observation, _ = env.reset(seed=seed)
        frames.append(np.asarray(observation["pixels"]["image"], dtype=np.float32))
    return float(np.mean(np.abs(frames[0] - frames[1])))
```

In `maybe_eval`, after `save_resume(...)` and before `if commit is not None:` add:
```python
            run_sync_hook(output_dir, env_steps)
```
At the end of `train`, after `_write_json(output_dir / "summary.json", ...)` add:
```python
    run_sync_hook(output_dir, env_steps)
```

In `main()`:

```python
    parser.add_argument("operation", choices=("train", "eval", "check-inits"))
    ...
    parser.add_argument("--task-id", type=int)
    args = parser.parse_args()
    if args.operation == "check-inits":
        if args.task_id is None:
            parser.error("check-inits requires --task-id")
        prepared = _read_prepared(args.prepared_path)
        suite = None
        if prepared.get("assets_path"):
            from script.lingbot_eval import describe_suite
            suite, _ = describe_suite(prepared["assets_path"])
        env = make_env(args.task_id, suite, init_states=False)
        try:
            difference = procedural_reset_difference(env)
        finally:
            env.close()
        ok = difference > 0.5
        print(json.dumps({"task_id": args.task_id, "mean_abs_pixel_diff": difference, "ok": ok}))
        if not ok:
            sys.exit(1)
        return
```
Place this block before the existing `payload = json.loads(args.config_json) ...` line so `check-inits` needs no config.

- [ ] **Step 4: Run the tests**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py -k "sync_hook or procedural_reset or check_inits"`
Expected: PASS.

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py`
Expected: PASS.

Run: `.cache/eval-venv/bin/python -m script.lingbot_rl_train --help`
Expected: usage lists `{train,eval,check-inits}` and `--task-id`.

- [ ] **Step 5: Commit**

```bash
git add script/lingbot_rl_train.py tests/test_lingbot_rl.py
git commit -m "$(cat <<'MSG'
Run an optional sync command after each DICE-RL checkpoint and add a procedural-init check operation.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
MSG
)"
```

---

### Task 5: Brev scripts — `env.example`, `push.sh`, `setup.sh`, `train.sh`, `sync.sh`, `pull.sh`

**Files:**
- Create: `brev/env.example`, `brev/push.sh`, `brev/setup.sh`, `brev/train.sh`, `brev/sync.sh`, `brev/pull.sh`
- Modify: `.gitignore` (add `brev/env.sh`)
- Test: `tests/test_brev_scripts.py` (new)

**Interfaces:**
- Consumes: `script.lingbot_eval prepare` CLI (`--config-json --cache-root --checkpoint --prepared-output`, needs `LEROBOT_SOURCE_ROOT` and `HF_TOKEN`); `script.lingbot_rl_train train --config-json --prepared-path --output-dir --run-name --dataset-root [--resume] [--max-env-steps]` and `check-inits --task-id --prepared-path` (Task 4); `DICE_SYNC_CMD` env contract `DICE_OUTPUT_DIR`, `DICE_STEP`; constants `LEROBOT_REVISION` (`script/lingbot_eval_config.py`), `DATASET_REPO`/`DATASET_REVISION` (`script/lingbot_sft_config.py`), volumes `dice-lingbot-sft-runs`, `dice-lingbot-rl-runs` (`script/lingbot_rl_modal.py`).
- Produces: on the box, `$DICE_DATA/{lerobot,cache/hub,sft/libero30-sft/checkpoints/step_000600,prepared.json,dataset_root,libero-config,runs/<run>/}`; `runs/<run>/config.json` (the `RLConfig` overrides), `runs/<run>/run.sh` (the exact training command), `runs/<run>/train.log`; tmux session `dice-<run>`. On the Mac, `result/brev/<run>/`. Every script honours `DICE_DRY_RUN=1` (print the command it would run, exit 0) so the tests can drive them without a box.

All scripts source `brev/env.sh` next to themselves when it exists (override the path with `DICE_ENV_FILE`; the tests point it at `/dev/null` so a real `env.sh` on the Mac cannot leak into them), then apply defaults.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_brev_scripts.py`:

```python
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ("push.sh", "setup.sh", "train.sh", "sync.sh", "pull.sh")


def _run(script, args, env, cwd=None):
    merged = {**os.environ, "DICE_DRY_RUN": "1", "DICE_ENV_FILE": "/dev/null", **env}
    for name in ("DICE_SYNC_CMD", "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "WANDB_ENTITY"):
        merged.pop(name, None)
    return subprocess.run(["bash", str(ROOT / "brev" / script), *args], env=merged, cwd=cwd,
                          capture_output=True, text=True)


def _box(tmp_path):
    data = tmp_path / "data"
    sft = data / "sft" / "libero30-sft" / "checkpoints" / "step_000600"
    sft.mkdir(parents=True)
    (sft / "norm_stats.json").write_text("{}")
    (data / "prepared.json").write_text("{}")
    (data / "dataset_root").write_text(str(data / "cache" / "hub" / "snap"))
    (data / "lerobot" / ".venv" / "bin").mkdir(parents=True)
    (data / "lerobot" / ".venv" / "bin" / "python").write_text("")
    return {"DICE_DATA": str(data), "DICE_REPO": str(ROOT),
            "WANDB_API_KEY": "wandb-secret-value", "HF_TOKEN": "hf-secret-value"}


@pytest.mark.parametrize("script", SCRIPTS)
def test_scripts_parse_and_are_executable(script):
    path = ROOT / "brev" / script
    assert os.access(path, os.X_OK)
    assert subprocess.run(["bash", "-n", str(path)], capture_output=True).returncode == 0


def test_env_example_lists_required_variables_without_values():
    text = (ROOT / "brev" / "env.example").read_text()
    for name in ("DICE_REPO", "DICE_DATA", "WANDB_API_KEY", "HF_TOKEN", "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"):
        assert f"export {name}=" in text
    assert "brev/env.sh" in (ROOT / ".gitignore").read_text()


def test_train_dry_run_builds_single_task_command_and_resumes_when_state_exists(tmp_path):
    env = _box(tmp_path)
    first = _run("train.sh", ["unit-t4", "4"], env)
    assert first.returncode == 0, first.stderr
    assert '"task_ids": [4]' in first.stdout
    assert "--run-name unit-t4" in first.stdout
    assert "--dataset-root" in first.stdout
    assert "--resume" not in first.stdout
    assert "tmux new-session -d -s dice-unit-t4" in first.stdout
    assert "DICE_SYNC_CMD" in first.stdout
    assert "--config-json" in first.stdout
    run_dir = Path(env["DICE_DATA"]) / "runs" / "unit-t4"
    assert (run_dir / "run.sh").is_file()
    assert '"task_ids": [4]' in (run_dir / "config.json").read_text()
    (run_dir / "resume").mkdir()
    (run_dir / "resume" / "latest.pt").write_text("")
    second = _run("train.sh", ["unit-t4", "4"], env)
    assert "--resume" in second.stdout
    smoke = _run("train.sh", ["smoke", "0", "--max-env-steps", "32"], env)
    assert "--max-env-steps 32" in smoke.stdout
    assert '"wandb_entity": null' in smoke.stdout


def test_train_requires_run_name_task_and_prepared(tmp_path):
    env = _box(tmp_path)
    assert _run("train.sh", [], env).returncode != 0
    assert _run("train.sh", ["only-name"], env).returncode != 0
    Path(env["DICE_DATA"], "prepared.json").unlink()
    missing = _run("train.sh", ["r", "0"], env)
    assert missing.returncode != 0
    assert "prepared.json" in missing.stderr


def test_sync_skips_without_modal_credentials_and_puts_step_residual(tmp_path):
    env = _box(tmp_path)
    out = Path(env["DICE_DATA"]) / "runs" / "r"
    (out / "train_eval" / "step_080000").mkdir(parents=True)
    (out / "train_eval" / "step_080000" / "residual.pt").write_text("")
    (out / "residual.pt").write_text("")
    base = {**env, "DICE_OUTPUT_DIR": str(out), "DICE_STEP": "80000"}
    skipped = _run("sync.sh", ["r"], {**base, "MODAL_TOKEN_ID": "", "MODAL_TOKEN_SECRET": ""})
    assert skipped.returncode == 0
    assert "skip" in skipped.stdout
    put = _run("sync.sh", ["r"], {**base, "MODAL_TOKEN_ID": "a", "MODAL_TOKEN_SECRET": "b"})
    assert put.returncode == 0, put.stderr
    assert "modal volume put dice-lingbot-rl-runs" in put.stdout
    assert "train_eval/step_080000/residual.pt r/train_eval/step_080000/residual.pt" in put.stdout
    final = _run("sync.sh", ["r"], {**base, "DICE_STEP": "660000", "MODAL_TOKEN_ID": "a", "MODAL_TOKEN_SECRET": "b"})
    assert "residual.pt r/residual.pt" in final.stdout


def test_pull_excludes_resume_unless_requested(tmp_path):
    env = {"DICE_DATA": "/data/dice"}
    plain = _run("pull.sh", ["box", "r"], env, cwd=tmp_path)
    assert plain.returncode == 0, plain.stderr
    assert "--exclude resume/" in plain.stdout
    assert "box:/data/dice/runs/r/" in plain.stdout
    assert str(Path("result/brev/r")) in plain.stdout
    with_resume = _run("pull.sh", ["box", "r", "--resume"], env, cwd=tmp_path)
    assert "--exclude resume/" not in with_resume.stdout


def test_push_syncs_repo_without_heavy_dirs_and_sft_on_request(tmp_path):
    env = {"DICE_DATA": "/data/dice"}
    plain = _run("push.sh", ["box"], env, cwd=ROOT)
    assert plain.returncode == 0, plain.stderr
    for excluded in (".venv", ".cache", "checkpoints", "result", ".git"):
        assert f"--exclude {excluded}" in plain.stdout
    assert "step_000600" not in plain.stdout
    sft = _run("push.sh", ["box", "--sft"], env, cwd=ROOT)
    assert "checkpoints/lingbot-sft/libero30-sft/step_000600/" in sft.stdout
    assert "box:/data/dice/sft/libero30-sft/checkpoints/step_000600/" in sft.stdout


def test_setup_dry_run_pins_lerobot_revision_and_prepare_call(tmp_path):
    env = _box(tmp_path)
    out = _run("setup.sh", [], env)
    assert out.returncode == 0, out.stderr
    revision = (ROOT / "script" / "lingbot_eval_config.py").read_text().split('LEROBOT_REVISION = "')[1].split('"')[0]
    assert f"git -C {env['DICE_DATA']}/lerobot checkout {revision}" in out.stdout
    assert "--extra lingbot_va --extra libero --extra evaluation --no-editable" in out.stdout
    assert "modal==1.1.4" in out.stdout
    assert "script.lingbot_eval prepare" in out.stdout
    assert "script.lingbot_rl_train check-inits --task-id 0" in out.stdout
    assert "hf-secret-value" not in out.stdout + out.stderr
    assert "wandb-secret-value" not in out.stdout + out.stderr
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_brev_scripts.py`
Expected: FAIL — `AssertionError` on `os.access` / `FileNotFoundError` for every script.

- [ ] **Step 3: Create the scripts**

`brev/env.example`:
```bash
export DICE_REPO=$HOME/dice-rl-wam
export DICE_DATA=/data/dice
export WANDB_API_KEY=
export WANDB_ENTITY=
export HF_TOKEN=
export MODAL_TOKEN_ID=
export MODAL_TOKEN_SECRET=
```

Append to `.gitignore`:
```
brev/env.sh
```

`brev/push.sh` (runs on the Mac):
```bash
#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${DICE_ENV_FILE:-$HERE/env.sh}"
[ -f "$ENV_FILE" ] && source "$ENV_FILE"
HOST="${1:?usage: push.sh <ssh-host> [--sft]}"
WITH_SFT="${2:-}"
: "${DICE_DATA:=/data/dice}"
: "${DICE_REMOTE_REPO:=dice-rl-wam}"
REPO="$(cd "$HERE/.." && pwd)"
run() { if [ "${DICE_DRY_RUN:-0}" = "1" ]; then echo "$*"; else "$@"; fi; }
run rsync -az --delete --exclude .venv --exclude .cache --exclude checkpoints --exclude result --exclude .git \
  --exclude '__pycache__' --exclude '*.pyc' --exclude .pytest_cache --exclude 'brev/env.sh' \
  "$REPO/" "$HOST:$DICE_REMOTE_REPO/"
if [ "$WITH_SFT" = "--sft" ]; then
  run rsync -az --partial --progress "$REPO/checkpoints/lingbot-sft/libero30-sft/step_000600/" \
    "$HOST:$DICE_DATA/sft/libero30-sft/checkpoints/step_000600/"
fi
```

`brev/setup.sh` (runs on the box):
```bash
#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${DICE_ENV_FILE:-$HERE/env.sh}"
[ -f "$ENV_FILE" ] && source "$ENV_FILE"
: "${DICE_REPO:=$(cd "$HERE/.." && pwd)}"
: "${DICE_DATA:=/data/dice}"
: "${HF_TOKEN:?HF_TOKEN is required}"
: "${WANDB_API_KEY:?WANDB_API_KEY is required}"
DRY="${DICE_DRY_RUN:-0}"
run() { if [ "$DRY" = "1" ]; then echo "$*"; else "$@"; fi; }
LEROBOT="$DICE_DATA/lerobot"
PY="$LEROBOT/.venv/bin/python"
SFT="$DICE_DATA/sft/libero30-sft/checkpoints/step_000600"
REVISION="$(grep -o 'LEROBOT_REVISION = "[0-9a-f]*"' "$DICE_REPO/script/lingbot_eval_config.py" | cut -d'"' -f2)"
DATASET_REPO="$(grep -o 'DATASET_REPO = "[^"]*"' "$DICE_REPO/script/lingbot_sft_config.py" | cut -d'"' -f2)"
DATASET_REVISION="$(grep -o 'DATASET_REVISION = "[0-9a-f]*"' "$DICE_REPO/script/lingbot_sft_config.py" | cut -d'"' -f2)"
export PATH="$HOME/.local/bin:$PATH"
export LEROBOT_SOURCE_ROOT="$LEROBOT" HF_HOME="$DICE_DATA/cache/hub" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export LIBERO_CONFIG_PATH="$DICE_DATA/libero-config" PYTHONPATH="$DICE_REPO" TOKENIZERS_PARALLELISM=false
run mkdir -p "$DICE_DATA/cache/hub" "$DICE_DATA/runs" "$DICE_DATA/sft/libero30-sft/checkpoints"
if command -v apt-get >/dev/null 2>&1; then
  run sudo apt-get update -y
  run sudo apt-get install -y git ffmpeg libgl1 libegl1 libegl1-mesa-dev libgl1-mesa-dev libglib2.0-0 libglvnd0 libgles2 build-essential cmake tmux rsync curl
fi
if ! command -v uv >/dev/null 2>&1; then
  run bash -c "curl -LsSf https://astral.sh/uv/0.8.18/install.sh | sh"
fi
if [ ! -d "$LEROBOT/.git" ]; then
  run git clone https://github.com/huggingface/lerobot.git "$LEROBOT"
fi
run git -C "$LEROBOT" checkout "$REVISION"
run uv sync --index-url https://pypi.org/simple --project "$LEROBOT" --python 3.12 --locked --no-default-groups --extra lingbot_va --extra libero --extra evaluation --no-editable
run uv export --index-url https://pypi.org/simple --project "$LEROBOT" --locked --no-default-groups --extra lingbot_va --extra libero --extra evaluation --no-emit-project --no-hashes --output-file "$DICE_DATA/lingbot-eval-deps.txt"
run uv pip install --index-url https://pypi.org/simple --python "$PY" --constraint "$DICE_DATA/lingbot-eval-deps.txt" --exclude-newer 2026-09-05T00:00:00Z modal==1.1.4
if [ -n "${MODAL_TOKEN_ID:-}" ] && [ -n "${MODAL_TOKEN_SECRET:-}" ]; then
  run "$PY" -m modal token set --token-id "$MODAL_TOKEN_ID" --token-secret "$MODAL_TOKEN_SECRET" --profile brev
  run "$PY" -m modal profile activate brev
fi
if [ ! -f "$SFT/norm_stats.json" ]; then
  if [ -n "${MODAL_TOKEN_ID:-}" ]; then
    run "$PY" -m modal volume get dice-lingbot-sft-runs libero30-sft/checkpoints/step_000600 "$DICE_DATA/sft/libero30-sft/checkpoints/"
  else
    echo "SFT checkpoint missing at $SFT; run brev/push.sh <host> --sft from the Mac" >&2
    exit 1
  fi
fi
run "$PY" -m script.lingbot_eval prepare --config-json '{"source_run":"libero30-sft","checkpoint_step":600,"stage":"eval","seed":42}' --cache-root "$DICE_DATA/cache" --checkpoint "$SFT" --prepared-output "$DICE_DATA/prepared.json"
run "$PY" -c "import os; from script.lingbot_eval import download_snapshot; download_snapshot('$DATASET_REPO', repo_type='dataset', revision='$DATASET_REVISION', cache_dir='$DICE_DATA/cache/hub', token=os.environ['HF_TOKEN'])"
run bash -c "echo '$DICE_DATA/cache/hub/datasets--${DATASET_REPO//\//--}/snapshots/$DATASET_REVISION' > '$DICE_DATA/dataset_root'"
run "$PY" -c "from lerobot.policies.lingbot_va.modeling_lingbot_va import LingBotVAPolicy; import wandb, modal"
run "$PY" -m script.lingbot_rl_config
run "$PY" -m script.lingbot_rl_train check-inits --task-id 0 --prepared-path "$(cat "$DICE_DATA/prepared.json" 2>/dev/null | "$PY" -c 'import json,sys; print(json.load(sys.stdin).get("prepared_path",""))' 2>/dev/null || true)"
echo "setup complete: $DICE_DATA"
```

In dry-run mode the `prepared.json` from `_box` is `{}` and `$PY` is an empty file, so the `--prepared-path` substitution collapses to an empty string; the test only checks the `check-inits --task-id 0` text.

`brev/train.sh` (runs on the box):
```bash
#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${DICE_ENV_FILE:-$HERE/env.sh}"
[ -f "$ENV_FILE" ] && source "$ENV_FILE"
RUN="${1:?usage: train.sh <run-name> <task-id> [--max-env-steps N] [--no-tmux]}"
TASK="${2:?usage: train.sh <run-name> <task-id> [--max-env-steps N] [--no-tmux]}"
shift 2
MAX_ENV_STEPS=""
NO_TMUX=0
while [ $# -gt 0 ]; do
  case "$1" in
    --max-env-steps) MAX_ENV_STEPS="$2"; shift 2 ;;
    --no-tmux) NO_TMUX=1; shift ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
: "${DICE_REPO:=$(cd "$HERE/.." && pwd)}"
: "${DICE_DATA:=/data/dice}"
: "${WANDB_API_KEY:?WANDB_API_KEY is required}"
DRY="${DICE_DRY_RUN:-0}"
PY="$DICE_DATA/lerobot/.venv/bin/python"
PREPARED_INDEX="$DICE_DATA/prepared.json"
[ -f "$PREPARED_INDEX" ] || { echo "missing $PREPARED_INDEX; run brev/setup.sh first" >&2; exit 1; }
PREPARED="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("prepared_path", sys.argv[1]))' "$PREPARED_INDEX")"
DATASET_ROOT="$(cat "$DICE_DATA/dataset_root")"
OUT="$DICE_DATA/runs/$RUN"
mkdir -p "$OUT"
ENTITY="${WANDB_ENTITY:-}"
python3 -c 'import json,sys; print(json.dumps({"task_ids": [int(sys.argv[1])], "wandb_entity": sys.argv[2] or None}))' "$TASK" "$ENTITY" > "$OUT/config.json"
ARGS=(--prepared-path "$PREPARED" --output-dir "$OUT" --run-name "$RUN" --dataset-root "$DATASET_ROOT")
[ -f "$OUT/resume/latest.pt" ] && ARGS+=(--resume)
[ -n "$MAX_ENV_STEPS" ] && ARGS+=(--max-env-steps "$MAX_ENV_STEPS")
{
  echo '#!/usr/bin/env bash'
  echo 'set -euo pipefail'
  echo "export PYTHONPATH='$DICE_REPO' HF_HOME='$DICE_DATA/cache/hub' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"
  echo "export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl LIBERO_CONFIG_PATH='$DICE_DATA/libero-config' TOKENIZERS_PARALLELISM=false"
  echo "export LEROBOT_SOURCE_ROOT='$DICE_DATA/lerobot' DICE_SYNC_CMD='$DICE_REPO/brev/sync.sh $RUN'"
  echo "cd '$DICE_REPO'"
  printf 'exec %q -m script.lingbot_rl_train train --config-json "$(cat %q)"' "$PY" "$OUT/config.json"
  printf ' %q' "${ARGS[@]}"
  printf ' 2>&1 | tee -a %q\n' "$OUT/train.log"
} > "$OUT/run.sh"
chmod +x "$OUT/run.sh"
if [ "$DRY" = "1" ]; then
  cat "$OUT/config.json" "$OUT/run.sh"
  echo "tmux new-session -d -s dice-$RUN bash $OUT/run.sh"
  exit 0
fi
if [ "$NO_TMUX" = "1" ]; then
  exec bash "$OUT/run.sh"
fi
if tmux has-session -t "dice-$RUN" 2>/dev/null; then
  echo "session dice-$RUN already running; attach with: tmux attach -t dice-$RUN" >&2
  exit 1
fi
tmux new-session -d -s "dice-$RUN" "bash $OUT/run.sh"
echo "started dice-$RUN; attach with: tmux attach -t dice-$RUN; log: $OUT/train.log"
```

`brev/sync.sh` (runs on the box, called by the hook):
```bash
#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${DICE_ENV_FILE:-$HERE/env.sh}"
[ -f "$ENV_FILE" ] && source "$ENV_FILE"
RUN="${1:?usage: sync.sh <run-name>}"
: "${DICE_DATA:=/data/dice}"
: "${DICE_OUTPUT_DIR:?DICE_OUTPUT_DIR is set by the training hook}"
: "${DICE_STEP:?DICE_STEP is set by the training hook}"
DRY="${DICE_DRY_RUN:-0}"
PY="$DICE_DATA/lerobot/.venv/bin/python"
run() { if [ "$DRY" = "1" ]; then echo "$*"; else "$@"; fi; }
if [ -z "${MODAL_TOKEN_ID:-}" ] || [ -z "${MODAL_TOKEN_SECRET:-}" ]; then
  echo "sync skip: no Modal credentials (step $DICE_STEP)"
  exit 0
fi
STEP_DIR="train_eval/step_$(printf '%06d' "$DICE_STEP")"
if [ -f "$DICE_OUTPUT_DIR/$STEP_DIR/residual.pt" ]; then
  run "$PY" -m modal volume put dice-lingbot-rl-runs "$DICE_OUTPUT_DIR/$STEP_DIR/residual.pt" "$RUN/$STEP_DIR/residual.pt" --force
else
  run "$PY" -m modal volume put dice-lingbot-rl-runs "$DICE_OUTPUT_DIR/residual.pt" "$RUN/residual.pt" --force
fi
[ -f "$DICE_OUTPUT_DIR/settings.json" ] && run "$PY" -m modal volume put dice-lingbot-rl-runs "$DICE_OUTPUT_DIR/settings.json" "$RUN/settings.json" --force
echo "sync done: step $DICE_STEP"
```

`brev/pull.sh` (runs on the Mac):
```bash
#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${DICE_ENV_FILE:-$HERE/env.sh}"
[ -f "$ENV_FILE" ] && source "$ENV_FILE"
HOST="${1:?usage: pull.sh <ssh-host> <run-name> [--resume]}"
RUN="${2:?usage: pull.sh <ssh-host> <run-name> [--resume]}"
WITH_RESUME="${3:-}"
: "${DICE_DATA:=/data/dice}"
DEST="result/brev/$RUN"
run() { if [ "${DICE_DRY_RUN:-0}" = "1" ]; then echo "$*"; else "$@"; fi; }
run mkdir -p "$DEST"
ARGS=(-az --partial --progress)
[ "$WITH_RESUME" = "--resume" ] || ARGS+=(--exclude resume/)
run rsync "${ARGS[@]}" "$HOST:$DICE_DATA/runs/$RUN/" "$DEST/"
echo "pulled $RUN into $DEST"
```

Then:
```bash
chmod +x brev/push.sh brev/setup.sh brev/train.sh brev/sync.sh brev/pull.sh
```

- [ ] **Step 4: Run the tests**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_brev_scripts.py`
Expected: PASS (8 tests). If `test_setup_dry_run...` fails on the `sudo` line under dry-run, the `run` wrapper is echoing rather than executing — check that every side-effecting line goes through `run`.

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py`
Expected: PASS (unchanged).

- [ ] **Step 5: Commit**

```bash
git add brev/ .gitignore tests/test_brev_scripts.py
git commit -m "$(cat <<'MSG'
Add the Brev push, setup, train, sync, and pull scripts for single-task DICE-RL runs.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
MSG
)"
```

---

### Task 6: Docs, spec correction, and the full verification pass

**Files:**
- Modify: `README.md` (new section after "## Evaluate step 600 on one H100"), `AGENTS.md` (the "## LingBot DICE-RL" narrow checks), `docs/superpowers/specs/2026-09-22-single-task-dice-rl-design.md` (§7 truncation bullet)

**Interfaces:** none; documentation only.

- [ ] **Step 1: Fix the spec's finalize claim**

In the spec, §7 `lingbot_rl_train.py` "episode end" bullet, replace the last sentence

```
`finalize_episode` runs on every episode end, not only when
    `done=1` (`:476`), so a budget-cut episode still gets MC returns.
```
with
```
`finalize_episode` already runs after every episode (`:478`), so the only
    change is which rows carry `done=1`.
```

- [ ] **Step 2: Add the Brev section to `README.md`**

Insert before "## Evaluate step 600 on one H100" (or at the end of the RL section if one exists):

```markdown
## Single-task RL on Brev

Training runs on an on-demand H100 Brev instance; evaluation stays on Modal. Copy `brev/env.example` to `brev/env.sh` (gitignored) and fill in `WANDB_API_KEY`, `HF_TOKEN`, and optionally the Modal token so checkpoints can be pushed to the `dice-lingbot-rl-runs` volume. The box needs ~30 GB on `DICE_DATA` (frozen text encoder/VAE 14 GB, SFT step 600 9.5 GB, demo latents 2.3 GB, LIBERO assets 0.4 GB).

From the Mac, with an ssh alias for the instance:

```bash
brev/push.sh <host> --sft
```

On the box:

```bash
brev/setup.sh
brev/train.sh smoke-t0 0 --max-env-steps 32 --no-tmux
brev/train.sh dice-t0 0
```

`setup.sh` mirrors the Modal image (pinned LeRobot revision, same `uv sync` extras), runs `script.lingbot_eval prepare`, downloads the demo latents, and ends with `check-inits`, which asserts two procedural resets differ. `train.sh` writes `runs/<run>/run.sh` and starts it in tmux session `dice-<run>`; rerunning the same name resumes from `runs/<run>/resume/latest.pt`. Every checkpoint (every 80k env steps and at the final step) fires `brev/sync.sh`, which pushes `residual.pt` to Modal when credentials exist and is otherwise a no-op.

Back on the Mac:

```bash
brev/pull.sh <host> dice-t0
```

pulls checkpoints, train-eval JSONs, and `train.log` into `result/brev/dice-t0/`; add `--resume` to also pull the multi-GB resume state.
```

- [ ] **Step 3: Update `AGENTS.md` narrow checks**

In the "## LingBot DICE-RL" section, add to the narrow-checks code block:

```bash
.cache/eval-venv/bin/python -m pytest -q tests/test_brev_scripts.py
bash -n brev/setup.sh brev/train.sh brev/sync.sh brev/pull.sh brev/push.sh
```

and after the block add one paragraph:

```markdown
Recipe version 3 trains the tasks in `RLConfig.task_ids` from procedural LIBERO initial states (`init_states=False`); the canonical 50 init states are reserved for evaluation. Brev scripts under `brev/` honour `DICE_DRY_RUN=1` and are tested that way; never run them for real as local verification. `brev/env.sh` holds secrets and is gitignored.
```

- [ ] **Step 4: Full verification**

Run:
```bash
.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py tests/test_brev_scripts.py
.cache/eval-venv/bin/python -m compileall -q script/lingbot_rl_*.py
.cache/eval-venv/bin/python -m script.lingbot_rl_config
.cache/eval-venv/bin/python -m script.lingbot_rl_train --help
uv run --no-project --with modal==1.1.4 --with click==8.1.8 --with typer==0.16.0 modal run -m script.lingbot_rl_modal --help
```
Expected: all tests pass, compile clean, config prints version 3, help lists `check-inits`, Modal launcher help still renders.

- [ ] **Step 5: Commit**

```bash
git add README.md AGENTS.md docs/superpowers/specs/2026-09-22-single-task-dice-rl-design.md
git commit -m "$(cat <<'MSG'
Document the Brev single-task DICE-RL workflow and recipe version 3.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
MSG
)"
```

---

## After Phase A lands (not tasks in this plan)

1. `brev/push.sh <host> --sft`, `brev/setup.sh`, then the lean smoke `brev/train.sh smoke-t0 0 --max-env-steps 32 --no-tmux` (~15 min after setup). Confirm: W&B run visible, `runs/smoke-t0/residual.pt` written, `sync done`/`sync skip` printed by the hook.
2. Start the real runs: `brev/train.sh dice-t0 0` on box A and `brev/train.sh dice-t4 4` on box B.
3. Phase B plan: eval protocol v2 (`heldout` stage, 50 states × 2 seeds), SFT baseline on Modal, Modal checkpoint evals from the synced `residual.pt`, k=1 switch, paired analysis. Must land before the runs reach 80k (~9 h).
