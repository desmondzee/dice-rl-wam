# LIBERO-Plus collection with structured features and proprioception: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Two task 9 runs (`dice-t9-plus-m4-pool`, `dice-t9-plus-m4-struct`) collected under the LIBERO-Plus robot-initial-state and layout training variants with a thirds reset mix, proprioception appended to every critic and actor input, run B replacing the image pools by per-camera / per-latent-frame means, and both buffers storing the full feature set (structured means, tokens, latents, proprioception, simulator object poses, reset kind) for the offline study.

**Architecture:** The model owns the input layout: `DiceResidualModel(state_features, proprio)` builds `state_input(pool, struct, proprio)` and `plan_input(pool, struct)` and sizes the actor, Q ensemble and plan critic from them; the policy computes pool, structured means and tokens from one transformer pass and delegates the layout to the model; the trainer draws one of three resets per episode from a dedicated per-episode generator, builds plus-variant environments, reads proprioception from `obs_type="pixels_agent_pos"` and object poses from the robosuite env, and stores every feature in the replay rows; evaluation reconstructs the layout from the residual checkpoint's `layout` record.

**Tech Stack:** Torch 2.11 (cu130 on the GB10s), the pinned LeRobot LingBot-VA port, numpy replay with bf16 torch tensors for the stored features, pytest with the mocked transformer and the `_StubPolicy` / `_smoke_train` fixtures in `tests/test_lingbot_rl.py`, LIBERO-Plus at `4976dc3` activated by `~/Documents/dice/libero-plus-config/activate.sh` on the GB10s.

**Spec:** docs/superpowers/specs/2026-10-16-ood-collection-design.md (revision f4a5594, which adds `object_state`).

## Global Constraints

- Zero comments and docstrings in `script/lingbot_rl_*.py` and tests; lean code, no bloated tests, only necessary tests.
- No secrets in logs or argv.
- The existing single-plan and M=4 recipes must still reproduce bit-for-bit when the new config fields are at their defaults (`proprio` False, `state_features` "pool", `plus_train` empty).
- The resume fingerprint must change when any new field changes.
- Evaluation must reconstruct the model's input layout from the residual checkpoint alone.
- Two runs: run A `state_features "pool"` + proprio, run B `state_features "struct"` + proprio; both store the full feature set listed in the spec; both collect with the thirds reset mix over the training variants listed in the spec; W&B config must record the new fields and the per-reset-kind metrics.
- Do not edit `.cache/lerobot`; do not edit `script/lingbot_eval.py` (its `harness_sha256` enters evaluation provenance); do not instantiate Hydra `DistillResidualRLModel`; tests must not call `modal run` or `.remote()`.
- No GPU jobs, Modal, ssh or remote machines from implementers; Task 9 is run by the controller on a GB10.
- Verify every task with `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py tests/test_lingbot_eval.py`, `.cache/eval-venv/bin/python -m script.lingbot_rl_config`, `.cache/eval-venv/bin/python -m compileall -q script/`, `bash -n brev/setup.sh brev/train.sh brev/sync.sh brev/pull.sh brev/eval.sh`.

## Decisions the spec left open (binding for every task)

- **Proprioception into the actor.** The buffer keeps `s` as the image feature the model consumes for the state (the 3072 pool in both runs, since run B reads `state_struct`); `proprio` is a separate (8,) float32 key. The model concatenates: `state_input(pool, struct, proprio)` returns `pool` (run A) or `struct.reshape(B, 6144)` (run B), with `proprio` appended when `self.proprio`; this is the actor's state and the first part of the critic and plan-critic inputs. `critic_state(state, plan)` keeps its signature; `plan_input(pool, struct)` returns the plan pool (run A) or the (8 × 3072) flattened struct (run B). Buffer keys `plan_pool`, `plan_pools_all`, `plan_pool_next` keep holding pools in both runs; run B's model reads `plan_struct`, `plan_struct_all`, `plan_struct_next`.
- **Proprio for later chunks at evaluation.** `select_action` calls `predict_action_chunk(None)` after the first chunk, so `ResidualLingBotPolicy.select_action` stashes `batch.get("proprio")` in `self._last_proprio` and `_state_input(decoded, batch)` uses the batch's proprio when a batch is given, else the stash. The trainer always passes the chunk-start batch.
- **Batches with proprio.** `script/lingbot_eval.py` is not edited. `lingbot_rl_train.rl_observation_batch(observation, instruction, device, proprio)` wraps `observation_batch` and adds `batch["proprio"]` (1, 8) float32; `lingbot_rl_train.run_rl_episode(..., proprio, video_writer=None)` delegates to `lingbot_eval.run_episode` when `proprio` is False (bit-for-bit today's path) and otherwise runs the same loop with `rl_observation_batch`.
- **Reset draw RNG.** `draw_reset(config, episode)` uses `np.random.default_rng([config.seed, episode])`, so episode `i` draws the same (kind, variant, state) in both runs regardless of episode lengths, and the global `np.random` stream is untouched (bit-for-bit for `plus_train` empty). The episode ordinal is saved in the resume file (`episodes`). For seed 42 the first twelve draws are `(0, None), (1, 673), (0, None), (2, 1989), (1, 670), (0, None), (2, 1997), (1, 661), (1, 677), (2, 2001), (2, 1995), (1, 640)`; all three kinds appear by episode 3.
- **Layout reset state.** The third value of the draw is a uniform float; `init_state_id = int(draw * len(env._init_states))` (one stored state per layout variant today, so 0).
- **Object poses.** `object_state` is recorded whenever `config.proprio` is set (both runs). At startup the trainer resets one base-task env (`init_states=False`, seed `config.seed`, RNG captured and restored around it), reads `object_layout(env._env.env)` = sorted `obj_body_id` names plus every non-free MuJoCo joint whose body lies under an object body (the articulated joints), writes it to `settings.json` as `object_layout`, and at every chunk start reads `object_state(env._env.env, layout)` = `[body_xpos (3), body_xquat (4)] per body` then the joint `qpos` values; distractor bodies of layout variants are not in the layout and are ignored; a missing name raises. Expert rows carry an empty `object_state`.
- **Plus suite.** With `plus_train` set, the trainer uses `_get_suite("libero_10")` from the active LIBERO-Plus install (as the plus driver does) instead of `describe_suite`, since the plus suite has more than ten tasks; task 9 of it must be the base task (checked in Task 9).
- **Checkpoint layout.** `inference_state_dict()` adds `"layout": {"state_features", "proprio", "plan_candidates"}`; `load_inference_state_dict` raises on a mismatch; `DiceResidualModel.layout_of(payload)` returns `{"state_features", "proprio"}` with pool / False defaults for pre-change checkpoints. `plan_candidates` and `residual_input` still come from the eval config as today.
- **Variant split (spec section 1).** Train robot-init (28): 640, 641, 643, 644, 646, 647, 649, 650, 652, 653, 655, 656, 658, 659, 661, 662, 664, 665, 667, 668, 670, 671, 673, 674, 676, 677, 679, 680. Test robot-init (14): 642, 645, 648, 651, 654, 657, 660, 663, 666, 669, 672, 675, 678, 681. Train layout (26): 1982, 1983, 1985, 1986, 1988, 1989, 1991, 1992, 1994, 1995, 1997, 1998, 2000, 2001, 2003, 2004, 2006, 2007, 2009, 2010, 2012, 2013, 2015, 2016, 2018, 2019. Test layout (13): 1984, 1987, 1990, 1993, 1996, 1999, 2002, 2005, 2008, 2011, 2014, 2017, 2020.
- **Token geometry.** The transformer flattens latents `b c (f p1) (h p2) (w p3) -> b (f h w) ...` (`.cache/lerobot/src/lerobot/policies/lingbot_va/utils.py:1056`), so with the 8 × 16 latent grid and patch (1, 2, 2) each latent frame is 4 rows × 8 columns of tokens, row-major; columns 0–3 are camera 1 (`observation.images.image`), 4–7 camera 2. `structured_means` reshapes `(b, f·4·8, C)` to `(b, f, 4, 2, 4, C)` and averages rows and columns. `plan_struct_all` is (M, 8, 3072) frame-major (index = frame × 2 + camera); `state_struct` is (2, 3072).

## Review Focus

1. A proprio checkpoint evaluated through a path whose batches lack `proprio` must raise, not silently feed zeros: Task 4 tests `state_input(..., None)` raises `ValueError`.
2. A plus-variant env missing one of the base task's objects (or a renamed body) must raise at the chunk read rather than store a shifted vector: Task 6 tests `object_state` raises `KeyError` for a missing body and ignores distractors.
3. A run resumed under different `plus_train` / `proprio` / `state_features` must be refused by the recipe fingerprint: Task 1 tests the three protocols differ (the refusal mechanism is `test_load_resume_rejects_recipe_mismatch`).
4. A pre-change residual checkpoint (no `layout`) must still evaluate as pool / no proprio, and a checkpoint whose `plan_candidates` disagrees with `--plans` must refuse: Task 7 tests both.
5. Robot-init resets must use `init_states=False` with a fresh env per episode and layout resets `init_states=True` on the plus suite, since repeated resets of one env are not reproducible (docs/libero-plus/README.md): Task 6 asserts the env kwargs per reset kind against `draw_reset`.

## File map

| file | responsibility after this plan |
|---|---|
| `script/lingbot_rl_config.py` | `state_features`, `proprio`, `plus_train`; `PLUS_VARIANTS`, `plus_split`, `plus_train_variants`; validation; fingerprint keys |
| `script/lingbot_rl_policy.py` | `structured_means`; `_features_from_latent` (pool, struct, tokens from one pass); `extract_features_from_latent`; `select_plan` / `decode_candidates` carrying struct, tokens, latents; `_state_input` / `_plan_input`; proprio stash in `select_action` |
| `script/lingbot_rl_buffer.py` | new keys with legacy defaults, bf16 storage, `*_next` keys at finalisation, store-only keys excluded from `sample` |
| `script/lingbot_rl_model.py` | `state_dim` / `plan_dim` from `state_features` and `proprio`; `state_input`, `plan_input`; `layout` in the inference state dict; `layout_of` |
| `script/lingbot_rl_data.py` | expert recipe v3: struct, tokens, latents from the published latents, proprio from the parquet |
| `script/lingbot_rl_train.py` | `make_env(plus, proprio)`, `_plus_suite`, `draw_reset`, `proprio_from_observation`, `rl_observation_batch`, `run_rl_episode`, `object_layout` / `object_state`, per-kind metrics, `episodes` in the resume, layout-aware `evaluate` |
| `brev/train.sh` | `--plus`, `--proprio`, `--state-features` |
| `docs/libero-plus/plus_eval.py` | layout from the checkpoint; `pixels_agent_pos` env and `run_rl_episode` when proprio |
| `docs/libero-plus/README.md` | the variant split |

---

### Task 1: Config fields, variant split and fingerprint

**Files:**
- Modify: `script/lingbot_rl_config.py`
- Test: `tests/test_lingbot_rl.py`

**Interfaces:**
- Produces: `RLConfig.state_features: str = "pool"`, `RLConfig.proprio: bool = False`, `RLConfig.plus_train: dict = {}` (axis → tuple of variant indices, both axes or none); `PLUS_VARIANTS`, `plus_split(axis) -> (train_tuple, test_tuple)`, `plus_train_variants() -> {"robot_init": (...), "layout": (...)}`; protocol keys `state_features`, `proprio`, `plus_train` (lists), `collection_mix` (`"standard"` | `"thirds_standard_robot_init_layout"`); `config_from_dict` restores `plus_train` tuples from JSON lists.

- [ ] **Step 1: Write the failing test** (after `test_config_plan_fields_enter_the_fingerprint`)

```python
def test_config_feature_fields_enter_the_fingerprint_for_both_runs():
    from script.lingbot_rl_config import plus_split, plus_train_variants
    base = RLConfig().protocol()
    assert (base["state_features"], base["proprio"], base["plus_train"], base["collection_mix"]) == ("pool", False, {}, "standard")
    assert plus_split("robot_init")[1] == tuple(range(642, 682, 3)) and plus_split("layout")[1] == tuple(range(1984, 2021, 3))
    train = plus_train_variants()
    assert (len(train["robot_init"]), len(train["layout"])) == (28, 26) and 642 not in train["robot_init"]
    run_a = RLConfig(task_ids=(9,), plan_candidates=4, epsilon=-0.3, residual_input="base", proprio=True, plus_train=train).protocol()
    run_b = RLConfig(task_ids=(9,), plan_candidates=4, epsilon=-0.3, residual_input="base", proprio=True, state_features="struct",
                     plus_train=train).protocol()
    assert run_a["collection_mix"] == "thirds_standard_robot_init_layout" and run_a["plus_train"]["layout"][:3] == [1982, 1983, 1985]
    assert len({json.dumps(protocol, sort_keys=True) for protocol in (base, run_a, run_b)}) == 3
    restored = config_from_dict(json.loads(json.dumps(RLConfig(task_ids=(9,), plus_train=train).to_dict())))
    assert restored.plus_train == train
    for bad in (dict(state_features="tokens"), dict(proprio=1), dict(task_ids=(9,), plus_train={"view": (1,)}),
                dict(task_ids=(9,), plus_train={"layout": (1984,)}), dict(task_ids=(9,), plus_train={"layout": (1982,)}),
                dict(plus_train=train)):
        with pytest.raises(ValueError):
            RLConfig(**bad).validate()
```

- [ ] **Step 2: Run it**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py -k feature_fields`
Expected: FAIL with `ImportError: cannot import name 'plus_split'`.

- [ ] **Step 3: Implement** in `script/lingbot_rl_config.py`

Line 1 becomes `from dataclasses import asdict, dataclass, field`. After the imports (before `@dataclass`):

```python
PLUS_VARIANTS = {"robot_init": tuple(range(640, 682)), "layout": tuple(range(1982, 2021))}


def plus_split(axis):
    variants = PLUS_VARIANTS[axis]
    train = tuple(variant for position, variant in enumerate(variants) if position % 3 != 2)
    test = tuple(variant for position, variant in enumerate(variants) if position % 3 == 2)
    return train, test


def plus_train_variants():
    return {axis: plus_split(axis)[0] for axis in PLUS_VARIANTS}
```

Fields after `plan_random_prob: float = 0.1`:

```python
    state_features: str = "pool"
    proprio: bool = False
    plus_train: dict = field(default_factory=dict)
```

In `validate()` before `return self`:

```python
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
```

In `protocol()` after `"plan_pool": ...,`:

```python
            "state_features": self.state_features,
            "proprio": self.proprio,
            "plus_train": {axis: [int(v) for v in variants] for axis, variants in sorted(self.plus_train.items())},
            "collection_mix": "thirds_standard_robot_init_layout" if self.plus_train else "standard",
```

In `config_from_dict` after the `task_ids` block:

```python
    if "plus_train" in fields:
        fields["plus_train"] = {axis: tuple(int(v) for v in variants) for axis, variants in fields["plus_train"].items()}
```

- [ ] **Step 4: Verify and commit**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py && .cache/eval-venv/bin/python -m script.lingbot_rl_config`

```bash
git add script/lingbot_rl_config.py tests/test_lingbot_rl.py
git commit -m "Add state_features, proprio and the plus_train variant split to the RL recipe fingerprint."
```

---

### Task 2: Structured pooling in the policy

**Files:**
- Modify: `script/lingbot_rl_policy.py`, `tests/test_lingbot_rl.py` (`_mock_policy`, `_mock_features`, three tests)

**Interfaces:**
- Produces:
  - `structured_means(video_tokens (b, f·rows·cols, C), frames, rows, cols) -> (b, frames, 2, C)`: camera halves of each row, text excluded.
  - `ResidualLingBotPolicy._features_from_latent(latent (b, 48, T, h, w)) -> {"pool": (b, 3072), "struct": (b, T, 2, 3072), "tokens": (b, 32·T, 3072)}` from one transformer pass; `_pool_from_latent(latent)` = its `"pool"`.
  - `extract_features_from_latent(latent, batch)` (same dict, inputs moved to the policy device/dtype); `extract_critic_state_from_latent` = its `"pool"`.
  - `decode_candidates` output gains `state_struct (1, 2, 3072)`, `state_tokens (1, 32, 3072)`, `state_latent (1, 48, 1, h, w)`; `select_plan` output gains those plus `plan_struct_all (M, 8, 3072)`, `plan_latents_all (M, 48, 4, h, w)`, `plan_struct (1, 8, 3072)` of the executed plan. Ranking still receives the raw pool and pools in this task (Task 4 switches it to the model inputs).

- [ ] **Step 1: Mock support** in `tests/test_lingbot_rl.py`

In `_mock_policy`'s `Transformer.__init__` after `self.calls = []` add `self.create_empty_cache = lambda *args, **kwargs: None`. Add `attn_window=30` to the `policy.config = SimpleNamespace(...)` arguments (`_ensure_critic_cache` reads it). After `policy.residual_model = None` add `policy._critic_cache_batches = {}`. Add a helper before `test_single_plan_infer_reproduces_baseline_bit_for_bit`:

```python
def _mock_features(pool):
    return lambda latent: {"pool": pool(latent), "struct": pool(latent)[:, None, None].expand(-1, latent.shape[2], 2, -1),
                           "tokens": pool(latent)[:, None].expand(-1, 32 * latent.shape[2], -1)}
```

Replace `policy._pool_from_latent = lambda latent: torch.zeros(1, STATE_DIM)` in `test_decode_candidates_keeps_its_keys_on_the_mocked_policy` with `policy._features_from_latent = _mock_features(lambda latent: torch.zeros(1, STATE_DIM))` and its key assertion with `assert set(decoded) == {"s", "state_struct", "state_tokens", "state_latent", "z", "a_base", "video_noise", "latents", "first_chunk"}`. In `test_select_plan_batched_matches_looped_decode_plans_and_keeps_winner_rows` keep `looped._pool_from_latent = pool` and change the other two assignments to `policy._begin_chunk, policy._features_from_latent = begin, _mock_features(pool)` and `winner._begin_chunk, winner._features_from_latent = begin, _mock_features(pool)`. `test_decode_plans_shapes_and_restores_policy_state` is unchanged (`decode_plans` keeps `_pool_from_latent`).

- [ ] **Step 2: Write the failing tests**

```python
def test_structured_means_split_cameras_and_match_the_pool_without_text():
    from script.lingbot_rl_policy import structured_means
    tokens = torch.zeros(2, 4 * 4 * 8, STATE_DIM)
    grid = tokens.reshape(2, 4, 4, 8, STATE_DIM)
    grid[:, :, :, :4] = 1.0
    grid[:, :, :, 4:] = 3.0
    grid[1, 2] += 10.0
    struct = structured_means(tokens, 4, 4, 8)
    assert struct.shape == (2, 4, 2, STATE_DIM)
    assert torch.equal(struct[0, :, 0], torch.ones(4, STATE_DIM)) and torch.equal(struct[0, :, 1], torch.full((4, STATE_DIM), 3.0))
    assert torch.equal(struct[1, 2, 0], torch.full((STATE_DIM,), 11.0)) and torch.equal(struct[1, 1, 0], torch.ones(STATE_DIM))
    torch.testing.assert_close(pool_critic_state(tokens[:1], torch.full((1, 512, STATE_DIM), 2.0))[0], struct[0].mean(dim=(0, 1)))


def test_features_from_latent_returns_pool_struct_and_tokens_from_one_pass():
    policy = _mock_policy()
    latent = torch.randn(1, 48, 4, 2, 4)
    calls = len(policy.transformer.calls)
    feats = policy._features_from_latent(latent)
    assert len(policy.transformer.calls) == calls + 1
    assert feats["pool"].shape == (1, STATE_DIM) and feats["struct"].shape == (1, 4, 2, STATE_DIM) and feats["tokens"].shape == (1, 8, STATE_DIM)
    assert torch.equal(feats["struct"][0, :, 0], feats["tokens"][0, 0::2]) and torch.equal(feats["struct"][0, :, 1], feats["tokens"][0, 1::2])
    assert torch.equal(feats["pool"], policy._pool_from_latent(latent))
```

Append to `test_select_plan_batched_matches_looped_decode_plans_and_keeps_winner_rows` right after `assert out["plan_index"] == 1 and torch.equal(out["plan_pool"], out["plan_pools"][1:2])`:

```python
    assert out["plan_struct_all"].shape == (2, 8, STATE_DIM) and torch.equal(out["plan_struct"], out["plan_struct_all"][1:2])
    assert out["state_struct"].shape == (1, 2, STATE_DIM) and out["state_tokens"].shape == (1, 32, STATE_DIM)
    assert out["plan_latents_all"].shape == (2, 48, 4, 2, 4) and out["state_latent"].shape == (1, 48, 1, 2, 4)
```

- [ ] **Step 3: Run them**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py -k "structured_means or features_from_latent or select_plan_batched or decode_candidates_keeps"`
Expected: FAIL with `ImportError` / `AttributeError: _features_from_latent`.

- [ ] **Step 4: Implement** in `script/lingbot_rl_policy.py`

After `pool_critic_state`:

```python
def structured_means(video_tokens, frames, rows, cols):
    b, width = video_tokens.shape[0], video_tokens.shape[-1]
    return video_tokens.reshape(b, frames, rows, 2, cols // 2, width).mean(dim=(2, 4))
```

Replace `extract_critic_state_from_latent` and `_pool_from_latent` (lines 190–223) with:

```python
    @torch.no_grad()
    def extract_features_from_latent(self, latent, batch):
        self.eval()
        self._ensure_frozen_modules()
        self._maybe_init_prompt(batch)
        if latent.ndim != 5:
            raise ValueError("Published latents must be shaped (B, C, T, H, W)")
        return self._features_from_latent(latent.to(device=self.config.device, dtype=self.dtype))

    @torch.no_grad()
    def extract_critic_state_from_latent(self, latent, batch):
        return self.extract_features_from_latent(latent, batch)["pool"]

    @torch.no_grad()
    def _features_from_latent(self, latent):
        b = latent.shape[0]
        name = self._ensure_critic_cache(b)
        captured = []

        def hook(module, inputs, output):
            captured.append(inputs[0])

        handle = self.transformer.proj_out.register_forward_hook(hook)
        try:
            payload = self._prepare_latent_input(latent, None, 0, 0, None, None, frame_st_id=0)["latent_res_lst"]
            payload["text_emb"] = self._prompt_embeds.to(self.dtype).expand(b, -1, -1).clone()
            payload["grid_id"] = payload["grid_id"][None].repeat(b, 1, 1)
            payload["timesteps"] = payload["timesteps"][None].repeat(b, 1)
            self.transformer(payload, update_cache=0, cache_name=name, action_mode=False)
            if not captured:
                raise RuntimeError("Failed to capture pre-proj_out video tokens")
            video_tokens = captured[0].float()
            text_tokens = self.transformer.condition_embedder.text_embedder(
                self._prompt_embeds.to(self.dtype)
            ).float()
            patch = self.config.patch_size
            latent_h, latent_w = self._latent_hw
            return {"pool": pool_critic_state(video_tokens, text_tokens.expand(b, -1, -1)),
                    "struct": structured_means(video_tokens, latent.shape[2] // patch[0], latent_h // patch[1], latent_w // patch[2]),
                    "tokens": video_tokens}
        finally:
            handle.remove()

    @torch.no_grad()
    def _pool_from_latent(self, latent):
        return self._features_from_latent(latent)["pool"]
```

In `decode_candidates` replace `state = self._pool_from_latent(critic_latent)` with `feats = self._features_from_latent(critic_latent)` and the returned dict with:

```python
        return {
            "s": feats["pool"].float(),
            "state_struct": feats["struct"][:, 0],
            "state_tokens": feats["tokens"],
            "state_latent": critic_latent,
            "z": model_to_mlp(z_model).float(),
            "a_base": model_to_mlp(actions).float(),
            "video_noise": video,
            "latents": latents,
            "first_chunk": first,
        }
```

In `select_plan` replace `state = self._pool_from_latent(critic_latent)` with `feats = self._features_from_latent(critic_latent)` and `state = feats["pool"]`; replace `pools = self._pool_from_latent(latents).float()` and the `out = {...}` line with:

```python
        plan_feats = self._features_from_latent(latents)
        pools = plan_feats["pool"].float()
        out = {"s": state.float(), "state_struct": feats["struct"][:, 0], "state_tokens": feats["tokens"], "state_latent": critic_latent,
               "z": model_to_mlp(action_noise).float(), "plan_pools": pools,
               "plan_struct_all": plan_feats["struct"].reshape(plans, -1, STATE_DIM), "plan_latents_all": latents,
               "video_noise": video_noise, "first_chunk": first}
```

and the `out.update({...})` line with:

```python
        out.update({"plan_pool": pools[index:index + 1], "plan_struct": out["plan_struct_all"][index:index + 1],
                    "plan_index": index, "latents": latents[index:index + 1]})
```

- [ ] **Step 5: Verify and commit**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py && .cache/eval-venv/bin/python -m compileall -q script/`

```bash
git add script/lingbot_rl_policy.py tests/test_lingbot_rl.py
git commit -m "Return per-camera, per-frame structured means, tokens and latents alongside the critic pool."
```

---

### Task 3: Replay keys, bf16 storage and legacy defaults

**Files:**
- Modify: `script/lingbot_rl_buffer.py`
- Test: `tests/test_lingbot_rl.py`

**Interfaces:**
- Produces: `host_bf16(value) -> cpu bf16 contiguous tensor`; row keys `proprio (8,) f32`, `proprio_next (8,) f32`, `state_struct (2, 3072) bf16`, `state_struct_next`, `state_tokens (32, 3072) bf16`, `state_latent (48, 1, 8, 16) bf16`, `plan_struct (8, 3072) bf16`, `plan_struct_all (M, 8, 3072) bf16`, `plan_struct_next (8, 3072) bf16`, `plan_latent_all (M, 48, 4, 8, 16) bf16`, `object_state (n,) f32` (n may be 0), `reset_kind int`, `variant_index int`; an input row may carry `plan_latent (48, 4, 8, 16)` instead of `plan_latent_all` (repeated over M and popped). `sample()` returns `proprio`, `proprio_next` (float32) and the five `*_struct*` keys (bf16, stacked, on `self.device`); `state_tokens`, `state_latent`, `plan_latent_all`, `object_state`, `reset_kind`, `variant_index` are store-only. `finalize_episode` sets `proprio_next`, `state_struct_next`, `plan_struct_next` from the n-step row like `s_next`. Rows loaded from an older `state_dict` get the same defaults.

- [ ] **Step 1: Write the failing test** (after `test_plan_rows_default_finalize_and_sample_with_one_shape`)

```python
def test_feature_rows_default_finalize_and_sample_in_bf16_with_store_only_keys():
    buffer = ChunkReplay(plans=2)
    first = _replay_row(0.0, 0.0)
    first.update(proprio=np.full(8, 0.5, np.float32), state_struct=np.full((2, STATE_DIM), 1.0, np.float32),
                 plan_struct=np.full((8, STATE_DIM), 1.0, np.float32), object_state=np.arange(11, dtype=np.float32),
                 reset_kind=2, variant_index=1989)
    buffer.add_online(first)
    second = _replay_row(1.0, 1.0)
    second.update(proprio=np.full(8, 2.0, np.float32), state_struct=torch.full((2, STATE_DIM), 2.0, dtype=torch.bfloat16),
                  plan_struct_all=torch.full((2, 8, STATE_DIM), 3.0, dtype=torch.bfloat16),
                  plan_latent_all=torch.zeros(2, 48, 4, 8, 16, dtype=torch.bfloat16))
    buffer.add_online(second)
    buffer.finalize_episode()
    rows = buffer.rows()
    assert rows[0]["state_struct"].dtype == torch.bfloat16 and rows[0]["plan_struct_all"].shape == (2, 8, STATE_DIM)
    assert float(rows[0]["plan_struct_all"][1, 0, 0]) == 1.0 and rows[0]["state_tokens"].shape == (32, STATE_DIM)
    assert rows[0]["plan_latent_all"].shape == (2, 48, 4, 8, 16) and rows[0]["state_latent"].shape == (48, 1, 8, 16)
    assert float(rows[0]["proprio_next"][0]) == 2.0 and float(rows[1]["proprio_next"][0]) == 2.0
    assert float(rows[0]["state_struct_next"][0, 0]) == 2.0 and float(rows[0]["plan_struct_next"][0, 0]) == 0.0
    assert rows[0]["object_state"].shape == (11,) and rows[1]["object_state"].shape == (0,)
    assert (rows[0]["reset_kind"], rows[0]["variant_index"], rows[1]["reset_kind"], rows[1]["variant_index"]) == (2, 1989, 0, 0)
    batch = buffer.sample(2, 0.0)
    assert batch["state_struct"].shape == (2, 2, STATE_DIM) and batch["state_struct"].dtype == torch.bfloat16
    assert batch["proprio"].shape == (2, 8) and batch["plan_struct_all"].shape == (2, 2, 8, STATE_DIM)
    assert batch["plan_struct"].shape == batch["plan_struct_next"].shape == (2, 8, STATE_DIM)
    assert not {"state_tokens", "state_latent", "plan_latent_all", "object_state", "reset_kind", "variant_index"} & set(batch)
    legacy = ChunkReplay(plans=2)
    payload = buffer.state_dict()
    for row in payload["data"]:
        for key in ("proprio", "proprio_next", "state_struct", "state_struct_next", "state_tokens", "state_latent", "plan_struct",
                    "plan_struct_all", "plan_struct_next", "plan_latent_all", "object_state", "reset_kind", "variant_index"):
            row.pop(key)
    legacy.load_state_dict(payload)
    assert legacy.sample(2, 0.0)["plan_struct"].shape == (2, 8, STATE_DIM) and legacy.rows()[0]["object_state"].shape == (0,)
```

- [ ] **Step 2: Run it**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py -k feature_rows`
Expected: FAIL with `KeyError: 'state_struct'` (or `AttributeError` on `.dtype`).

- [ ] **Step 3: Implement** in `script/lingbot_rl_buffer.py`

Replace the key tuples at the top (lines 6–10 and 21–25) with:

```python
PLAN_FLOAT_KEYS = ("plan_pool", "plan_pools_all", "plan_a_base_all", "plan_pool_next")
FLOAT_KEYS = (
    "s", "z", "a_base", "a", "reward", "done", "s_next", "a_next", "n_steps", "is_expert",
    "z_all", "a_base_all", "z_next_all", "a_base_next_all", "mc_return", *PLAN_FLOAT_KEYS, "proprio", "proprio_next",
)
BF16_KEYS = ("state_struct", "state_struct_next", "plan_struct", "plan_struct_all", "plan_struct_next")
STORED_BF16_KEYS = ("state_tokens", "state_latent", "plan_latent_all")
KEYS = (
    "s", "z", "a_base", "a", "reward", "done", "s_next", "a_next",
    "n_steps", "is_expert", "task_id", "n_env_actions",
    "z_all", "a_base_all", "z_next_all", "a_base_next_all", "mc_return", *PLAN_FLOAT_KEYS, "plan_index",
    "proprio", "proprio_next", *BF16_KEYS,
)
```

After `_host_float`:

```python
def host_bf16(value):
    return torch.as_tensor(value).detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
```

After `_plan_defaults`:

```python
    def _feature_defaults(self, stored):
        s = _host_float(stored["s"])
        stored.setdefault("proprio", np.zeros(8, np.float32))
        stored.setdefault("proprio_next", np.array(stored["proprio"], copy=True))
        stored.setdefault("state_struct", np.repeat(s[None], 2, axis=0))
        stored.setdefault("state_struct_next", torch.as_tensor(stored["state_struct"]).clone())
        stored.setdefault("state_tokens", np.repeat(s[None], 32, axis=0))
        stored.setdefault("state_latent", np.zeros((48, 1, 8, 16), np.float32))
        stored.setdefault("plan_struct", np.repeat(_host_float(stored["plan_pool"])[None], 8, axis=0))
        stored.setdefault("plan_struct_all", np.repeat(_host_float(stored["plan_struct"])[None], self.plans, axis=0))
        stored.setdefault("plan_struct_next", torch.as_tensor(stored["plan_struct"]).clone())
        plan_latent = stored.pop("plan_latent", None)
        if plan_latent is None:
            plan_latent = np.zeros((48, 4, 8, 16), np.float32)
        stored.setdefault("plan_latent_all", np.repeat(_host_float(plan_latent)[None], self.plans, axis=0))
        stored.setdefault("object_state", np.zeros(0, np.float32))
        stored.setdefault("reset_kind", 0)
        stored.setdefault("variant_index", int(stored["task_id"]))
        for key in (*BF16_KEYS, *STORED_BF16_KEYS):
            stored[key] = host_bf16(stored[key])
        if stored["plan_struct_all"].shape[0] != self.plans or stored["plan_latent_all"].shape[0] != self.plans:
            raise ValueError(f"Rows must carry exactly {self.plans} plans")
        stored["object_state"] = _host_float(stored["object_state"])
        stored["reset_kind"] = int(stored["reset_kind"])
        stored["variant_index"] = int(stored["variant_index"])
        return stored
```

In `_store`, after `self._plan_defaults(stored)` add `self._feature_defaults(stored)`. In `finalize_episode`, after the `plan_pool_next` line:

```python
            row["proprio_next"] = np.array(episode[nxt]["proprio"], copy=True)
            row["state_struct_next"] = episode[nxt]["state_struct"].clone()
            row["plan_struct_next"] = episode[nxt]["plan_struct"].clone()
```

In `sample`, make the loop body start with:

```python
        for key in KEYS:
            if key in BF16_KEYS:
                batch[key] = torch.stack([self._data[i][key] for i in indices]).to(self.device)
                continue
```

In `load_state_dict`, after `self._plan_defaults(row)` add `self._feature_defaults(row)`.

- [ ] **Step 4: Verify and commit**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py`

```bash
git add script/lingbot_rl_buffer.py tests/test_lingbot_rl.py
git commit -m "Store structured features, tokens, latents, proprioception, object poses and the reset kind in replay rows."
```

---

### Task 4: Model input layout and the policy / trainer call sites

**Files:**
- Modify: `script/lingbot_rl_model.py`, `script/lingbot_rl_policy.py` (`__init__`, `select_action`, `_state_input`, `_plan_input`, `select_plan` ranking, `_apply_residual_choice`, `predict_action_chunk`), `script/lingbot_rl_train.py` (`RESUME_KEYS`, `save_inference` only), `tests/test_lingbot_rl.py`

**Interfaces:**
- Produces:
  - `PROPRIO_DIM = 8`; `ResidualActor(residual_input="z", state_dim=STATE_DIM)`; `PlanCritic(state_dim=STATE_DIM, plan_dim=STATE_DIM)`.
  - `DiceResidualModel(device, residual_input, epsilon, plan_candidates=1, state_features="pool", proprio=False)` with `features`, `proprio`, `state_dim` (= 3072 × (2 if struct else 1) + 8 if proprio) and `plan_dim` (= 3072 × (8 if struct else 1) when `plans > 1`, else 0).
  - `model.state_input(pool (B, 3072), struct (B, 2, 3072) | None, proprio (B, 8) | None) -> (B, state_dim)`; raises `ValueError` when `self.proprio` and `proprio is None`.
  - `model.plan_input(pool (…, 3072) | None, struct (…, 8, 3072) | None) -> (…, plan_dim) | None`.
  - `critic_state(state, plan)`, `rank_plans(state, plans)`, `rank_candidates(state, plans, a_base_all, z_all)`, `n_step_target`, `update_actor`, `update_critic`, `update_plan_critic(state, plans_all (B, M, plan_dim), z_all, plan_a_base_all)` all take the model inputs above.
  - `inference_state_dict()["layout"] == {"state_features", "proprio", "plan_candidates"}`; `load_inference_state_dict` raises `ValueError("residual checkpoint layout mismatch")`; `DiceResidualModel.layout_of(payload) -> {"state_features", "proprio"}`.
  - Policy: `_state_input(decoded, batch | None) -> (1, state_dim)`, `_plan_input(pool, struct)`; `select_plan` passes model inputs to `rank_fn` / `rank_candidates` when `self.residual_model` is set (raw pools otherwise); `predict_action_chunk` sets `decoded["state_in"]`; `select_action` stashes `self._last_proprio`.

- [ ] **Step 1: Write the failing tests**

```python
def test_model_input_layout_follows_state_features_and_proprio():
    from script.lingbot_rl_model import PROPRIO_DIM
    struct = DiceResidualModel(device="cpu", state_features="struct", proprio=True)
    mixed = DiceResidualModel(device="cpu", plan_candidates=2, proprio=True)
    assert (struct.state_dim, struct.plan_dim) == (2 * STATE_DIM + PROPRIO_DIM, 0)
    assert (mixed.state_dim, mixed.plan_dim) == (STATE_DIM + PROPRIO_DIM, STATE_DIM)
    assert struct.actor.net[0].in_features == struct.state_dim + HORIZON * ACTION_DIM and struct.plan_critic is None
    assert struct.critic.heads[0][0].in_features == struct.state_dim + HORIZON * ACTION_DIM
    assert mixed.critic.heads[0][0].in_features == mixed.state_dim + mixed.plan_dim + HORIZON * ACTION_DIM
    assert mixed.plan_critic.heads[0][0].in_features == mixed.state_dim + mixed.plan_dim
    s, st, pr = torch.zeros(3, STATE_DIM), torch.arange(3 * 2 * STATE_DIM).reshape(3, 2, STATE_DIM).float(), torch.ones(3, PROPRIO_DIM)
    state = struct.state_input(s, st, pr)
    assert torch.equal(state[:, :2 * STATE_DIM], st.reshape(3, -1)) and torch.equal(state[:, -PROPRIO_DIM:], pr)
    assert torch.equal(mixed.state_input(s, st, pr)[:, :STATE_DIM], s) and mixed.plan_input(None, None) is None
    assert struct.plan_input(torch.zeros(3, 2, STATE_DIM), torch.ones(3, 2, 8, STATE_DIM)).shape == (3, 2, 8 * STATE_DIM)
    assert torch.equal(mixed.plan_input(torch.full((3, 2, STATE_DIM), 2.0), None), torch.full((3, 2, STATE_DIM), 2.0))
    with pytest.raises(ValueError):
        struct.state_input(s, st, None)
    payload = struct.inference_state_dict()
    assert payload["layout"] == {"state_features": "struct", "proprio": True, "plan_candidates": 1}
    assert DiceResidualModel.layout_of({"actor": {}}) == {"state_features": "pool", "proprio": False}
    with pytest.raises(ValueError, match="layout mismatch"):
        DiceResidualModel(device="cpu", proprio=True).load_inference_state_dict(payload)
    info = mixed.update_plan_critic(mixed.state_input(s, None, pr), torch.randn(3, 2, STATE_DIM), torch.randn(3, 4, HORIZON, ACTION_DIM),
                                    mask_unused_dof(torch.randn(3, 2, 4, HORIZON, ACTION_DIM)))
    assert set(info) == {"plan_critic_loss", "v_plan_mean", "plan_target_mean"}
    del struct, mixed
    torch.manual_seed(0)
    a = DiceResidualModel(device="cpu", plan_candidates=2)
    torch.manual_seed(0)
    b = DiceResidualModel(device="cpu", plan_candidates=2, state_features="pool", proprio=False)
    assert all(torch.equal(x, y) for x, y in zip(a.critic.parameters(), b.critic.parameters()))


def test_policy_feeds_proprio_from_the_batch_or_the_select_action_stash():
    from lerobot.policies.lingbot_va.modeling_lingbot_va import LingBotVAPolicy
    from script.lingbot_rl_policy import ResidualLingBotPolicy
    policy = ResidualLingBotPolicy.__new__(ResidualLingBotPolicy)
    policy.eval_candidates, policy.eval_plans, policy.eval_plan_ranker = 2, 1, "v_plan"
    policy.residual_model = DiceResidualModel(device="cpu", state_features="struct", proprio=True)
    captured = {}
    policy.decode_candidates = lambda batch, k: {"s": torch.zeros(1, STATE_DIM), "state_struct": torch.ones(1, 2, STATE_DIM),
                                                 "z": torch.zeros(2, HORIZON, ACTION_DIM), "a_base": torch.zeros(2, HORIZON, ACTION_DIM),
                                                 "first_chunk": False, "latents": torch.zeros(1)}
    policy._apply_residual_choice = lambda dec, index: captured.update(state=dec["state_in"]) or torch.zeros(1, 16, USED_DOF)
    policy.predict_action_chunk({"task": ["t"], "proprio": torch.full((1, 8), 3.0)})
    assert captured["state"].shape == (1, 2 * STATE_DIM + 8) and float(captured["state"][0, -1]) == 3.0 and float(captured["state"][0, 0]) == 1.0
    policy._last_proprio = torch.full((1, 8), 5.0)
    policy.predict_action_chunk(None)
    assert float(captured["state"][0, -1]) == 5.0
    original = LingBotVAPolicy.select_action
    LingBotVAPolicy.select_action = lambda self, batch, **kwargs: "ok"
    try:
        assert policy.select_action({"proprio": torch.full((1, 8), 7.0)}) == "ok" and float(policy._last_proprio[0, 0]) == 7.0
    finally:
        LingBotVAPolicy.select_action = original
```

Update existing assertions: in `test_train_smoke_loop_gates_updates_saves_weights_and_runs_hook` change `assert set(payload) <= {"actor", "critic", "target_critic"}` to `assert set(payload) <= {"actor", "critic", "target_critic", "layout"}`; in `test_save_inference_checkpoints_keeps_latest_and_step_copy` change both `== {"actor", "critic", "target_critic"}` to `== {"actor", "critic", "target_critic", "layout"}`; in `test_plan_critic_regresses_onto_max_candidate_min_q_and_leaves_q_and_actor_alone` change `assert set(single.inference_state_dict()) == {"actor", "critic", "target_critic"}` to include `"layout"`.

- [ ] **Step 2: Run them**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py -k "input_layout or proprio_from_the_batch or smoke_loop or keeps_latest or plan_critic_regresses"`
Expected: FAIL with `ImportError: cannot import name 'PROPRIO_DIM'` and `TypeError` on `state_features`.

- [ ] **Step 3: Implement** in `script/lingbot_rl_model.py`

After `K_CANDIDATES = 4` add `PROPRIO_DIM = 8`. `ResidualActor.__init__(self, residual_input="z", state_dim=STATE_DIM)` with `self.net = _mlp(state_dim + len(self.inputs) * HORIZON * ACTION_DIM, HORIZON * ACTION_DIM)`. `PlanCritic.__init__(self, state_dim=STATE_DIM, plan_dim=STATE_DIM)` with `_mlp(state_dim + plan_dim, 1)`.

`DiceResidualModel.__init__`:

```python
    def __init__(self, device="cpu", residual_input="z", epsilon=EPSILON, plan_candidates=1, state_features="pool", proprio=False):
        self.device = device
        self.epsilon = epsilon
        self.plans = int(plan_candidates)
        self.features = state_features
        self.proprio = bool(proprio)
        self.state_dim = STATE_DIM * (2 if state_features == "struct" else 1) + (PROPRIO_DIM if self.proprio else 0)
        self.plan_dim = STATE_DIM * (8 if state_features == "struct" else 1) if self.plans > 1 else 0
        self.actor = ResidualActor(residual_input, self.state_dim).to(device)
        self.critic = CriticEnsemble(self.state_dim + self.plan_dim).to(device)
```

(the rest unchanged, except `self.plan_critic = PlanCritic(self.state_dim, self.plan_dim).to(device)`). After `__init__`:

```python
    def state_input(self, pool, struct, proprio):
        if self.features == "pool":
            state = mlp_float(pool)
        else:
            struct = mlp_float(struct)
            state = struct.reshape(struct.shape[0], -1)
        if not self.proprio:
            return state
        if proprio is None:
            raise ValueError("proprio is required by this recipe")
        return torch.cat([state, mlp_float(proprio).to(state.device)], dim=-1)

    def plan_input(self, pool, struct):
        if pool is None:
            return None
        if self.features == "pool":
            return mlp_float(pool)
        struct = mlp_float(struct)
        return struct.reshape(*struct.shape[:-2], -1)

    @staticmethod
    def layout_of(payload):
        layout = payload.get("layout", {})
        return {"state_features": layout.get("state_features", "pool"), "proprio": bool(layout.get("proprio", False))}
```

`n_step_target`: `state_k = critic_k[:, :self.state_dim]`. `update_plan_critic`: replace the four `STATE_DIM` uses:

```python
        batch, plans, k = plan_a_base_all.shape[:3]
        sd, pd = state.shape[-1], pools_all.shape[-1]
        with torch.no_grad():
            flat_state = state[:, None, None].expand(batch, plans, k, sd).reshape(-1, sd)
            flat_pool = pools_all[:, :, None].expand(batch, plans, k, pd).reshape(-1, pd)
            ...
        preds = self.plan_critic(
            state[:, None].expand(batch, plans, sd).reshape(-1, sd), pools_all.reshape(-1, pd), return_all=True)
```

`inference_state_dict`: add `payload["layout"] = {"state_features": self.features, "proprio": self.proprio, "plan_candidates": self.plans}` before the plan-critic line. `load_inference_state_dict` starts with:

```python
        layout = payload.get("layout")
        if layout is not None and (layout["state_features"], bool(layout["proprio"]), int(layout.get("plan_candidates", self.plans))) \
                != (self.features, self.proprio, self.plans):
            raise ValueError("residual checkpoint layout mismatch")
```

- [ ] **Step 4: Policy call sites** in `script/lingbot_rl_policy.py`

In `__init__` add `self._last_proprio = None`. After `_get_t5_prompt_embeds`:

```python
    def select_action(self, batch, **kwargs):
        self._last_proprio = batch.get("proprio")
        return super().select_action(batch, **kwargs)

    def _state_input(self, decoded, batch):
        if self.residual_model is None:
            return decoded["s"].float()
        proprio = batch.get("proprio") if batch is not None else self._last_proprio
        return self.residual_model.state_input(decoded["s"], decoded.get("state_struct"), proprio)

    def _plan_input(self, pool, struct):
        if self.residual_model is None:
            return pool.float()
        return self.residual_model.plan_input(pool, struct)
```

In `select_plan` replace the ranking block (from `if random_plan:` to the `rank_fn` call) with:

```python
        state_in, pools_in = self._state_input(out, batch), self._plan_input(pools, out["plan_struct_all"])
        if random_plan:
            index = int(np.random.randint(plans))
        elif rank_candidates is not None:
            index = int(rank_candidates(state_in, pools_in, out["plan_a_base_all"], out["z"]))
        else:
            index = int(rank_fn(state_in, pools_in))
```

`_apply_residual_choice`: replace the three `state` lines with `state = decoded["state_in"][:1]`. `predict_action_chunk`:

```python
    @torch.no_grad()
    def predict_action_chunk(self, batch, **kwargs):
        k = self.eval_candidates if self.residual_model is not None else 1
        plans = self.eval_plans if self.residual_model is not None else 1
        if plans > 1:
            decoded = self.select_plan(
                batch, plans, k, rank_fn=self.residual_model.rank_plans, random_plan=self.eval_plan_ranker == "random",
                rank_candidates=self.residual_model.rank_candidates if self.eval_plan_ranker == "q" else None)
        else:
            decoded = self.decode_candidates(batch, k=k)
        decoded["state_in"] = self._state_input(decoded, batch)
        index = 0
        if self.residual_model is not None and decoded["a_base"].shape[0] > 1:
            a_base = decoded["a_base"].float()
            noise = decoded["z"].float()
            state = decoded["state_in"].expand(a_base.shape[0], -1)
            executed = apply_residual(a_base, self.residual_model.actor(state, noise, a_base))
            if plans > 1:
                state = self.residual_model.critic_state(state, self._plan_input(decoded["plan_pool"], decoded.get("plan_struct")).expand(k, -1))
            index = int(self.residual_model.critic(state, executed).reshape(-1).argmax())
        return self._apply_residual_choice(decoded, index)
```

In `script/lingbot_rl_train.py`: add `"layout"` to `RESUME_KEYS` and to the allowed set in `save_inference` (`{"actor", "critic", "target_critic", "plan_critic", "layout"}`).

- [ ] **Step 5: Verify and commit**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py tests/test_lingbot_eval.py && .cache/eval-venv/bin/python -m compileall -q script/`

```bash
git add script/lingbot_rl_model.py script/lingbot_rl_policy.py script/lingbot_rl_train.py tests/test_lingbot_rl.py
git commit -m "Size the actor and critics from state_features and proprio, record the layout in checkpoints, and feed proprioception through the policy."
```

---

### Task 5: Expert featurisation v3

**Files:**
- Modify: `script/lingbot_rl_data.py`, `tests/test_lingbot_rl.py` (two stubs)

**Interfaces:**
- Consumes: `policy.extract_features_from_latent(latent, batch) -> {"pool", "struct", "tokens"}` (Task 2).
- Produces: `EXPERT_RECIPE = "dice-rl-expert-v3"`; expert rows carry `proprio`, `proprio_next` (8,) float32 from the parquet `observation.state` at the chunk's first frame; `state_struct (2, 3072)`, `state_tokens (32, 3072)`, `state_latent (48, 1, h, w)`, `plan_struct (8, 3072)`, `plan_latent (48, 4, h, w)` as cpu bf16 tensors (published-latent branch only; the RGB branch leaves them to the buffer defaults); `load_manifest_episodes` adds `"states": (T, 8) float32` to every episode.

- [ ] **Step 1: Update the stubs and write the failing assertions**

In `test_featurize_experts_pools_published_latents` replace the stub's `extract_critic_state_from_latent` with:

```python
        def extract_features_from_latent(self, latent, batch):
            assert latent.shape[2] in (1, 4)
            value, frames = float(latent[0, 0, 0, 0, 0]), latent.shape[2]
            return {"pool": torch.full((1, 3072), value), "struct": torch.full((1, frames, 2, 3072), value),
                    "tokens": torch.full((1, 32 * frames, 3072), value)}
```

Rewrite `test_expert_rows_pool_the_true_plan_frames`:

```python
def test_expert_rows_pool_the_true_plan_frames(tmp_path):
    from script.lingbot_rl_data import featurize_experts
    calls = []

    class StubPolicy:
        config = type("C", (), {"device": "cpu"})()

        def reset(self):
            return None

        def extract_features_from_latent(self, latent, batch):
            calls.append(tuple(latent.shape))
            value, frames = float(latent[0, 0, 0, 0, 0]), latent.shape[2]
            return {"pool": torch.full((1, STATE_DIM), value), "struct": torch.full((1, frames, 2, STATE_DIM), value),
                    "tokens": torch.full((1, 32 * frames, STATE_DIM), value)}

    latents = torch.arange(10, dtype=torch.float32).reshape(1, 1, 10, 1, 1).expand(1, 48, 10, 2, 2).clone()
    episode = {"actions": np.zeros((36, USED_DOF), np.float32), "task": "t", "task_id": 9, "latents": latents, "success": True,
               "states": np.arange(36 * 8, dtype=np.float32).reshape(36, 8)}
    rows = featurize_experts(StubPolicy(), [episode], None, _eval_normalization(), tmp_path / "cache.pt")
    assert [row["s"][0] for row in rows] == [0.0, 3.0, 7.0]
    assert [row["plan_pool"][0] for row in rows] == [0.0, 4.0, 8.0]
    assert (1, 48, 4, 2, 2) in calls
    assert [float(row["proprio"][0]) for row in rows] == [0.0, 96.0, 224.0]
    assert [float(row["proprio_next"][0]) for row in rows] == [96.0, 224.0, 224.0]
    assert rows[1]["plan_struct"].shape == (8, STATE_DIM) and rows[1]["plan_struct"].dtype == torch.bfloat16 and float(rows[1]["plan_struct"][0, 0]) == 4.0
    assert rows[0]["state_struct"].shape == (2, STATE_DIM) and rows[0]["state_tokens"].shape == (32, STATE_DIM)
    assert rows[1]["state_latent"].shape == (48, 1, 2, 2) and float(rows[1]["state_latent"][0, 0, 0, 0]) == 3.0
    assert rows[1]["plan_latent"].shape == (48, 4, 2, 2) and float(rows[1]["plan_latent"][0, 0, 0, 0]) == 4.0
```

- [ ] **Step 2: Run them**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py -k "true_plan_frames or pools_published"`
Expected: FAIL with `AttributeError: 'StubPolicy' object has no attribute 'extract_critic_state_from_latent'`.

- [ ] **Step 3: Implement** in `script/lingbot_rl_data.py`

`EXPERT_RECIPE = "dice-rl-expert-v3"`; import `STATE_DIM` from `script.lingbot_rl_model`. In `_episode_rows`, add `states = episode.get("states")` before the loop and replace the body from `if episode.get("latents") is not None:` to `rows.append({...})` with:

```python
        proprio = np.asarray(states[offset], np.float32) if states is not None else np.zeros(8, np.float32)
        extras = {}
        if episode.get("latents") is not None:
            latents = episode["latents"]
            if latents.ndim != 5:
                raise ValueError("Published expert latents must be shaped (B, C, T, H, W)")
            frame_count = latents.shape[2]
            index = min(offset // 4, max(frame_count - 1, 0))
            if not hasattr(policy, "extract_features_from_latent"):
                raise TypeError("Policy must pool critic state from published latents")
            task_batch = {"task": [episode["task"]]}
            feats = policy.extract_features_from_latent(latents[:, :, index:index + 1], task_batch)
            state = _tensor_state(feats["pool"])
            start = index if first else index + 1
            plan_frames = torch.clamp(torch.arange(start, start + 4), max=max(frame_count - 1, 0))
            plan = policy.extract_features_from_latent(latents[:, :, plan_frames], task_batch)
            plan_pool = _tensor_state(plan["pool"])
            extras = {"state_struct": feats["struct"][0, 0], "state_tokens": feats["tokens"][0],
                      "state_latent": latents[0, :, index:index + 1], "plan_struct": plan["struct"][0].reshape(-1, STATE_DIM),
                      "plan_latent": latents[0, :, plan_frames]}
        else:
            state = _tensor_state(policy.extract_critic_state(_critic_batch(episode, offset, device)))
            plan_pool = state.copy()
        rows.append({
            "s": state,
            "plan_pool": plan_pool,
            "z": np.zeros((HORIZON, ACTION_DIM), dtype=np.float32),
            "a_base": chunk.copy(),
            "a": chunk.copy(),
            "reward": np.float32(0.0),
            "done": np.float32(0.0),
            "s_next": state.copy(),
            "proprio": proprio,
            "proprio_next": proprio.copy(),
            "task_id": int(episode.get("task_id", 0)),
            "n_env_actions": int(n_env),
            "is_expert": np.float32(1.0),
            **{key: value.detach().to(device="cpu", dtype=torch.bfloat16) for key, value in extras.items()},
        })
```

In the trailing loop add `rows[index]["proprio_next"] = rows[index + 1]["proprio"].copy()` beside the `s_next` line.

In `load_manifest_episodes`: read `columns=["action", "episode_index", "frame_index", "observation.state"]`; after the `actions` check:

```python
        states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        if states.shape != (len(actions), 8) or not np.isfinite(states).all():
            raise ValueError(f"Expected finite eight-dimensional observation.state in {parquet_path}")
```

and add `"states": states,` to the appended dict. The recipe string is in `expert_fingerprint`, so existing caches rebuild (same number of pooling passes as v2; 15–25 min on a GB10).

- [ ] **Step 4: Verify and commit**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py`

```bash
git add script/lingbot_rl_data.py tests/test_lingbot_rl.py
git commit -m "Expert recipe v3: structured features, tokens and latents from the published latents, proprioception from the parquet."
```

---

### Task 6: Trainer: plus environments, reset draw, proprioception, object poses, metrics, train.sh

**Files:**
- Modify: `script/lingbot_rl_train.py`, `brev/train.sh`, `tests/test_lingbot_rl.py` (`_StubPolicy`, `_sum_critic_model`, `_smoke_train`, three `load_resume` unpackings, new tests)

**Interfaces:**
- Consumes: Tasks 1–5.
- Produces:
  - `make_env(task_id, suite=None, init_states=True, plus=False, proprio=False)` (adds `is_libero_plus=plus`, `obs_type="pixels_agent_pos"` when `proprio`).
  - `_plus_suite()`; `RESET_KINDS = ("standard", "robot_init", "layout")`; `draw_reset(config, episode) -> (kind int, variant int | None, draw float)`.
  - `proprio_from_observation(observation) -> np.float32 (8,)`; `rl_observation_batch(observation, instruction, device, proprio)`; `run_rl_episode(policy, env, entry, instruction, norm, device, proprio, video_writer=None)`.
  - `object_layout(inner) -> {"bodies": [...], "joints": [...]}`; `object_state(inner, layout) -> np.float32 (7·len(bodies) + len(joints),)`.
  - `save_resume(..., wandb_id=None, episodes=0)`; `load_resume -> (env_steps, chunks, checkpointed, wandb_id, episodes)`; `RESUME_KEYS` gains `"episodes"`.
  - `_train_eval(..., episodes, proprio=False)`.
  - `settings.json` gains `"object_layout"` (None unless `config.proprio`). W&B per-episode keys `reset_kind`, `variant_index`, and with `plus_train`: `episode_success_<kind>`, `episode_length_<kind>`; update-cadence plan keys additionally as `<key>/<kind>`.
  - `brev/train.sh`: `--state-features pool|struct`, `--proprio`, `--plus`.

- [ ] **Step 1: Fixtures** in `tests/test_lingbot_rl.py`

`_StubPolicy.decode_candidates` returns, in addition: `"state_struct": torch.zeros(1, 2, STATE_DIM), "state_tokens": torch.zeros(1, 32, STATE_DIM), "state_latent": torch.zeros(1, 48, 1, 8, 16)`. Replace `_StubPolicy.select_plan` with:

```python
    def select_plan(self, batch, plans, k, rank_fn=None, random_plan=False, decode_all=False):
        type(self).plan_calls.append((plans, k, random_plan, decode_all))
        base = self.decode_candidates(batch, k)
        pools = torch.stack([torch.full((STATE_DIM,), float(m)) for m in range(plans)])
        struct_all = pools[:, None].expand(plans, 8, STATE_DIM)
        all_base = torch.stack([base["a_base"] + 10.0 * m for m in range(plans)])
        model = self.residual_model
        index = 1 if random_plan else int(rank_fn(
            model.state_input(base["s"], base["state_struct"], batch.get("proprio")), model.plan_input(pools, struct_all)))
        return {**base, "plan_pools": pools, "plan_struct_all": struct_all, "plan_latents_all": torch.zeros(plans, 48, 4, 8, 16),
                "plan_pool": pools[index:index + 1], "plan_struct": struct_all[index:index + 1], "plan_index": index,
                "plan_a_base_all": all_base, "a_base": all_base[index]}
```

`_sum_critic_model(device="cpu", residual_input="z", epsilon=-0.5, plan_candidates=1, **kwargs)` forwarding `**kwargs` to `DiceResidualModel`. In `_smoke_train`'s `StubEnv`:

```python
    class StubEnv:
        def __init__(self, **env_kwargs):
            envs.append(env_kwargs)
            self.steps = 0
            self.proprio = env_kwargs.get("obs_type") == "pixels_agent_pos"
            self._init_states = [0]
            self._env = SimpleNamespace(env=None)

        def _obs(self):
            zeros = np.zeros((128, 128, 3), np.uint8)
            obs = {"pixels": {"image": zeros, "image2": zeros}}
            if self.proprio:
                obs["robot_state"] = {"eef": {"pos": np.array([0.1, 0.2, 0.3]), "quat": np.array([0.0, 0.0, 0.0, 1.0])},
                                      "gripper": {"qpos": np.array([0.04, -0.04])}}
            return obs
```

(`from types import SimpleNamespace` at the top of the test file.) Add to the monkeypatches: `monkeypatch.setattr(train, "_plus_suite", lambda: None)`, `monkeypatch.setattr(train, "object_layout", lambda inner: {"bodies": ["mug"], "joints": ["door"]})`, `monkeypatch.setattr(train, "object_state", lambda inner, layout: np.array([7.0, 8.0], np.float32))`. Change the three `load_resume` unpackings (`test_save_load_resume_roundtrips_rng_checkpointed_and_schedulers`, `test_load_resume_restores_rng_from_non_byte_mapped_tensor`, `test_load_resume_accepts_legacy_torch_numpy_rng`) to five names, e.g. `env_steps, chunks, checkpointed, wandb_id, episodes = load_resume(...)`, and in the first assert `episodes == 0`.

- [ ] **Step 2: Write the failing tests**

```python
def test_draw_reset_is_per_episode_deterministic_and_covers_every_kind():
    from script.lingbot_rl_config import plus_train_variants
    from script.lingbot_rl_train import draw_reset
    config = RLConfig(task_ids=(9,), plus_train=plus_train_variants())
    draws = [draw_reset(config, episode) for episode in range(300)]
    assert draws == [draw_reset(config, episode) for episode in range(300)]
    assert {draw[0] for draw in draws} == {0, 1, 2} and all(0.0 <= draw[2] < 1.0 for draw in draws)
    assert all(draw[1] is None for draw in draws if draw[0] == 0)
    assert all(draw[1] in config.plus_train[("robot_init", "layout")[draw[0] - 1]] for draw in draws if draw[0])
    assert [draw[:2] for draw in draws[:4]] == [(0, None), (1, 673), (0, None), (2, 1989)]
    assert draw_reset(RLConfig(), 5) == (0, None, 0.0)


def test_proprio_from_observation_follows_the_dataset_convention():
    from script.lingbot_rl_train import proprio_from_observation
    observation = {"robot_state": {"eef": {"pos": np.array([1.0, 2.0, 3.0]), "quat": np.array([0.0, 0.0, np.sin(0.3), np.cos(0.3)])},
                                   "gripper": {"qpos": np.array([0.04, -0.04])}}}
    state = proprio_from_observation(observation)
    assert state.dtype == np.float32 and state.shape == (8,)
    np.testing.assert_allclose(state, [1.0, 2.0, 3.0, 0.0, 0.0, 0.6, 0.04, -0.04], atol=1e-6)
    observation["robot_state"]["eef"]["quat"] = np.array([0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(proprio_from_observation(observation)[3:6], 0.0)


def test_object_layout_walks_articulated_joints_under_base_bodies_and_reads_poses():
    from script.lingbot_rl_train import object_layout, object_state
    model = SimpleNamespace(njnt=3, jnt_type=[0, 3, 3], jnt_bodyid=[2, 4, 5], body_parentid=[0, 0, 0, 0, 3, 1],
                            joint_id2name=lambda joint: ["mug_free", "door_hinge", "robot_joint"][joint],
                            get_joint_qpos_addr=lambda name: {"door_hinge": 9}[name])
    data = SimpleNamespace(body_xpos=np.arange(18, dtype=np.float64).reshape(6, 3), body_xquat=np.arange(24, dtype=np.float64).reshape(6, 4),
                           qpos=np.arange(12, dtype=np.float64))
    inner = SimpleNamespace(sim=SimpleNamespace(model=model, data=data), obj_body_id={"mug": 2, "microwave": 3})
    layout = object_layout(inner)
    assert layout == {"bodies": ["microwave", "mug"], "joints": ["door_hinge"]}
    expected = np.concatenate([data.body_xpos[3], data.body_xquat[3], data.body_xpos[2], data.body_xquat[2], [9.0]]).astype(np.float32)
    np.testing.assert_array_equal(object_state(inner, layout), expected)
    inner.obj_body_id["cookies"] = 5
    np.testing.assert_array_equal(object_state(inner, layout), expected)
    with pytest.raises(KeyError):
        object_state(inner, {"bodies": ["bowl"], "joints": []})


def test_update_from_buffer_feeds_struct_and_proprio_inputs():
    from script.lingbot_rl_train import _update_from_buffer
    model = DiceResidualModel(device="cpu", state_features="struct", proprio=True)
    info = _update_from_buffer(model, _online_buffer(), 0.0, "cpu", False)
    assert {"critic_loss", "actor_loss"} <= set(info)


def test_train_under_plus_resets_with_proprio_stores_features_and_logs_per_kind(tmp_path, monkeypatch):
    from script.lingbot_rl_config import plus_train_variants
    from script.lingbot_rl_train import draw_reset
    config = RLConfig(task_ids=(9,), plan_candidates=2, plan_random_prob=0.0, selection_warmup_steps=0, residual_input="base",
                      proprio=True, plus_train=plus_train_variants())
    out = _smoke_train(tmp_path, monkeypatch, 60, model_factory=lambda **kw: DiceResidualModel(**kw), config=config)
    probe, envs = out["envs"][0], out["envs"][1:]
    assert (probe["task_id"], probe["init_states"], probe["obs_type"]) == (9, False, "pixels_agent_pos")
    draws = [draw_reset(config, episode) for episode in range(len(envs))]
    assert [draw[0] for draw in draws] == [0, 1, 0, 2, 1] and all(env["obs_type"] == "pixels_agent_pos" for env in envs)
    assert [(env["task_id"], env["init_states"], env["is_libero_plus"]) for env in envs] == \
        [(9 if variant is None else variant, kind == 2, kind > 0) for kind, variant, _ in draws]
    rows = [row for row in out["rows"] if float(row["is_expert"]) == 0.0]
    np.testing.assert_allclose(rows[0]["proprio"], [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.04, -0.04], atol=1e-6)
    assert rows[0]["state_struct"].dtype == torch.bfloat16 and rows[0]["state_struct"].shape == (2, STATE_DIM)
    assert rows[0]["plan_struct_all"].shape == (2, 8, STATE_DIM) and rows[0]["plan_latent_all"].shape == (2, 48, 4, 8, 16)
    assert rows[0]["state_tokens"].shape == (32, STATE_DIM) and rows[0]["object_state"].tolist() == [7.0, 8.0]
    assert [row["reset_kind"] for row in rows] == [0, 1, 0, 2, 1] and [row["variant_index"] for row in rows[:2]] == [9, 673]
    episode_logs = [entry for entry in out["logs"] if "reset_kind" in entry]
    assert episode_logs[1]["variant_index"] == 673 and {"episode_success_robot_init", "episode_length_robot_init"} <= set(episode_logs[1])
    updates = [entry for entry in out["logs"] if "plan_critic_loss" in entry]
    assert updates and any(key.startswith("plan_agreement/") for key in updates[0])
    settings = json.loads((tmp_path / "settings.json").read_text())
    assert settings["object_layout"] == {"bodies": ["mug"], "joints": ["door"]} and settings["config"]["plus_train"]["layout"][0] == 1982
    assert out["resume"]["episodes"] == 5 and out["resume"]["layout"] == {"state_features": "pool", "proprio": True, "plan_candidates": 2}
    assert out["wandb"][0]["config"]["proprio"] is True and out["wandb"][0]["config"]["state_features"] == "pool"
    assert not any("episode_success_standard" in entry for entry in _smoke_train(tmp_path / "plain", monkeypatch, 24)["logs"])


_GB10 = bool(os.environ.get("DICE_DATA")) and importlib.util.find_spec("libero") is not None


@pytest.mark.skipif(not _GB10, reason="needs DICE_DATA and the LIBERO install on a GB10")
def test_env_proprio_matches_the_first_demo_frame_of_a_stored_initial_state():
    import pyarrow.parquet as pq
    from pathlib import Path
    from script.lingbot_eval import describe_suite, read_json, seed_all
    from script.lingbot_rl_train import _env_class, proprio_from_observation
    data = Path(os.environ["DICE_DATA"])
    prepared = read_json(data / "prepared.json")
    prepared = read_json(prepared["prepared_path"]) if "prepared_path" in prepared else prepared
    manifest = read_json(Path(prepared["checkpoint"]) / "dataset_manifest.json")
    root = Path((data / "dataset_root").read_text().strip())
    instruction = next(task["instruction"] for task in prepared["tasks"] if task["task_id"] == 9)
    demos = []
    for episode in [episode for episode in manifest["episodes"] if episode["tasks"][0] == instruction][:3]:
        chunk = episode["episode_index"] // manifest["info"]["chunks_size"]
        path = root / manifest["info"]["data_path"].format(episode_chunk=chunk, episode_index=episode["episode_index"])
        demos.append(np.asarray(pq.read_table(path, columns=["observation.state"])["observation.state"][0].as_py(), np.float32))
    suite, _ = describe_suite(prepared["assets_path"])
    env = _env_class()(task_suite=suite, task_id=9, task_suite_name="libero_10", episode_length=520, observation_height=128,
                       observation_width=128, obs_type="pixels_agent_pos", init_states=True, n_envs=1, num_steps_wait=0,
                       control_freq=20, control_mode="relative", hard_reset=True)
    try:
        states = []
        for state_id in range(len(env._init_states)):
            env.init_state_id = state_id
            seed_all(0)
            observation, _ = env.reset(seed=0)
            states.append(proprio_from_observation(observation))
    finally:
        env.close()
    states = np.stack(states)
    assert states.shape == (50, 8)
    for demo in demos:
        gaps = np.abs(states - demo[None]).max(axis=1)
        assert gaps.min() < 2e-3, (gaps.min(), np.abs(states - demo[None])[gaps.argmin()])
```

(`import importlib.util` and `import os` at the top of the test file.) The trainer test keeps `state_features` at "pool" so the unit tests never build the struct M=2 model (about 2.5 GB of float32 parameters); the struct path through the update is covered by `test_update_from_buffer_feeds_struct_and_proprio_inputs` on a plan-free model and by run B's smoke in Task 9. The GB10 test compares every one of the 50 standard task 9 initial states with the first `observation.state` of three manifest demonstrations, so it needs no demo-to-state mapping; a failure with gaps only in entries 3–5 means the axis-angle or quaternion convention differs from the dataset's, and the launch is gated on it (spec section 7).

- [ ] **Step 3: Run them**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py -k "draw_reset or proprio_from_observation or object_layout or feeds_struct or under_plus or demo_frame"`
Expected: FAIL with `ImportError` (`draw_reset`); the GB10 test reports SKIPPED.

- [ ] **Step 4: Environment, draw, proprio, object poses, episodes** in `script/lingbot_rl_train.py`

Imports: `from script.lingbot_rl_buffer import ChunkReplay, host_bf16`. `RESUME_KEYS` gains `"episodes"`. Replace `make_env`:

```python
def make_env(task_id, suite=None, init_states=True, plus=False, proprio=False):
    return _env_class()(
        task_suite=suite, task_id=task_id, task_suite_name="libero_10",
        episode_length=520, observation_height=128, observation_width=128,
        obs_type="pixels_agent_pos" if proprio else "pixels",
        init_states=init_states, n_envs=1, num_steps_wait=10, control_freq=20, control_mode="relative",
        hard_reset=True, is_libero_plus=plus,
    )


def _plus_suite():
    from lerobot.envs.libero import _get_suite
    return _get_suite("libero_10")


RESET_KINDS = ("standard", "robot_init", "layout")


def draw_reset(config, episode):
    if not config.plus_train:
        return 0, None, 0.0
    rng = np.random.default_rng([config.seed, episode])
    kind = int(rng.integers(3))
    if kind == 0:
        return 0, None, float(rng.random())
    variants = config.plus_train[("robot_init", "layout")[kind - 1]]
    return kind, int(variants[rng.integers(len(variants))]), float(rng.random())


def proprio_from_observation(observation):
    state = observation["robot_state"]
    quat = np.asarray(state["eef"]["quat"], dtype=np.float64).reshape(4)
    w = min(max(float(quat[3]), -1.0), 1.0)
    den = float(np.sqrt(max(1.0 - w * w, 0.0)))
    axis_angle = np.zeros(3) if den <= 1e-10 else quat[:3] * (2.0 * np.arccos(w) / den)
    return np.concatenate([np.asarray(state["eef"]["pos"], dtype=np.float64).reshape(3), axis_angle,
                           np.asarray(state["gripper"]["qpos"], dtype=np.float64).reshape(2)]).astype(np.float32)


def rl_observation_batch(observation, instruction, device, proprio):
    from script.lingbot_eval import observation_batch
    batch = observation_batch(observation, instruction, device)
    if proprio:
        batch["proprio"] = torch.from_numpy(proprio_from_observation(observation))[None].to(device)
    return batch


def run_rl_episode(policy, env, entry, instruction, norm, device, proprio, video_writer=None):
    from script.lingbot_eval import decode_action, run_episode, seed_all, video_frame
    if not proprio:
        return run_episode(policy, env, entry, instruction, norm, device, video_writer=video_writer)
    started = time.perf_counter()
    seed_all(entry["seed"])
    env.init_state_id = entry["init_state_id"]
    observation, _ = env.reset(seed=entry["seed"])
    policy.reset()
    seed_all(entry["seed"])
    if video_writer is not None:
        video_writer.append_data(video_frame(observation))
    success = terminated = truncated = False
    with torch.inference_mode():
        for step in range(1, 521):
            normalized = policy.select_action(rl_observation_batch(observation, instruction, device, True))
            observation, _, terminated, truncated, info = env.step(decode_action(normalized, norm))
            success = env_success(info)
            if video_writer is not None:
                video_writer.append_data(video_frame(observation))
            if success or terminated or truncated:
                break
    truncated = bool(truncated or (step == 520 and not success and not terminated))
    return {**entry, "instruction": instruction, "success": success, "policy_steps": step, "terminated": bool(terminated),
            "truncated": truncated, "seconds": time.perf_counter() - started}


def object_layout(inner):
    model = inner.sim.model
    bodies = sorted(inner.obj_body_id)
    joints = []
    for name in bodies:
        root = inner.obj_body_id[name]
        for joint in range(model.njnt):
            if model.jnt_type[joint] == 0:
                continue
            body = model.jnt_bodyid[joint]
            while body > 0 and body != root:
                body = model.body_parentid[body]
            if body == root:
                joints.append(model.joint_id2name(joint))
    return {"bodies": bodies, "joints": joints}


def object_state(inner, layout):
    data, model = inner.sim.data, inner.sim.model
    parts = [np.concatenate([data.body_xpos[inner.obj_body_id[name]], data.body_xquat[inner.obj_body_id[name]]])
             for name in layout["bodies"]]
    parts.append(np.asarray([data.qpos[model.get_joint_qpos_addr(name)] for name in layout["joints"]], dtype=np.float64))
    return np.concatenate(parts).astype(np.float32)
```

`run_rl_episode` is the loop of `lingbot_eval.run_episode` (lines 266–297) with the batch builder swapped; `env_success` is the module's existing helper. `save_resume(path, model, buffer, env_steps, chunks, recipe, checkpointed=(), wandb_id=None, episodes=0)` adds `"episodes": int(episodes)` to the payload; `load_resume` returns `..., payload.get("wandb_id"), int(payload.get("episodes", 0))`.

`replay_sharpening`: replace the first four body lines after the `online` guard with:

```python
    state = model.state_input(sample["s"], sample.get("state_struct"), sample.get("proprio"))[online]
    z_all, base_all = mlp_float(sample["z_all"])[online], mlp_float(sample["a_base_all"])[online]
    pool = model.plan_input(sample.get("plan_pool"), sample.get("plan_struct"))
    batch, k = z_all.shape[:2]
    state_k = state.unsqueeze(1).expand(batch, k, -1).reshape(batch * k, -1)
    pool_k = None if pool is None else pool[online].unsqueeze(1).expand(batch, k, -1).reshape(batch * k, -1)
```

and `pools_all = mlp_float(sample["plan_pools_all"])[online]` with `pools_all = model.plan_input(sample["plan_pools_all"], sample.get("plan_struct_all"))[online]`.

`_update_from_buffer`:

```python
        sample = _sample_batch(buffer, expert_ratio, device)
        state = model.state_input(sample["s"], sample["state_struct"], sample["proprio"])
        next_state = model.state_input(sample["s_next"], sample["state_struct_next"], sample["proprio_next"])
        critic_state = model.critic_state(state, model.plan_input(sample["plan_pool"], sample["plan_struct"]))
        next_critic_state = model.critic_state(next_state, model.plan_input(sample["plan_pool_next"], sample["plan_struct_next"]))
        target = model.n_step_target(
            sample["reward"], sample["done"], next_critic_state,
            sample["z_next_all"], sample["a_base_next_all"], sample["n_steps"])
        if (step + 1) % ACTOR_EVERY == 0:
            info.update(model.update_actor(
                state, critic_state, sample["a"], sample["z_all"], sample["a_base_all"], sample["is_expert"],
                sample["mc_return"], filter_active))
        info.update(model.update_critic(critic_state, sample["a"], target))
        if model.plan_critic is not None:
            info.update(model.update_plan_critic(
                state, model.plan_input(sample["plan_pools_all"], sample["plan_struct_all"]), sample["z_all"], sample["plan_a_base_all"]))
```

`_train_eval(policy, tasks, norm, device, suite, env_steps, output_dir, episodes, proprio=False)`: `env = make_env(task["task_id"], suite, init_states=False, proprio=proprio)` and `row = run_rl_episode(policy, env, entry, task["instruction"], norm, device, proprio)`. `_maybe_sharpen`:

```python
def _maybe_sharpen(model, policy, batch, device):
    if policy.eval_plans > 1:
        decoded = policy.select_plan(batch, policy.eval_plans, 8, rank_fn=model.rank_plans)
    else:
        decoded = policy.decode_candidates(batch, k=8)
    state = model.state_input(decoded["s"].to(device), decoded.get("state_struct"), batch.get("proprio"))
    pool = model.plan_input(decoded["plan_pool"].to(device), decoded.get("plan_struct")) if policy.eval_plans > 1 else None
    a_base = decoded["a_base"].to(device=device, dtype=torch.float32)
    noise = decoded["z"].to(device=device, dtype=torch.float32)
    pooled = state
    if state.shape[0] == 1 and a_base.shape[0] > 1:
        state = state.expand(a_base.shape[0], -1)
    with torch.no_grad():
        action = apply_residual(a_base, model.actor(state, noise, a_base))
    if a_base.ndim == 3:
        a_base = a_base.unsqueeze(0)
        action = action.unsqueeze(0)
    else:
        pooled = state
    critic_state = model.critic_state(pooled, None if pool is None else pool.expand(pooled.shape[0], -1))
    return sharpening_metrics(model, critic_state, a_base, action)
```

- [ ] **Step 5: `train()`**

Build the model with `state_features=config.state_features, proprio=config.proprio`. Replace the suite block:

```python
    suite = None
    if config.plus_train:
        suite = _plus_suite()
    elif prepared.get("assets_path"):
        from script.lingbot_eval import describe_suite
        suite, suite_tasks = describe_suite(prepared["assets_path"])
        tasks = suite_tasks
    tasks = [task for task in tasks if task["task_id"] in config.task_ids]
    layout = None
    if config.proprio:
        rng = capture_rng()
        env = make_env(tasks[0]["task_id"], suite, init_states=False, proprio=True)
        try:
            env.reset(seed=config.seed)
            layout = object_layout(env._env.env)
        finally:
            env.close()
        restore_rng(rng)
    env_steps = 0
    chunks = 0
    episodes = 0
```

`load_resume` unpacks `env_steps, chunks, checkpointed, wandb_id, episodes`. `settings.json` gains `"object_layout": layout`. Every `save_resume(...)` call passes `episodes` as the last argument. In `maybe_checkpoint`: `_train_eval(..., config.train_eval_episodes, config.proprio)`, `env = make_env(tasks[0]["task_id"], suite, init_states=False, proprio=config.proprio)`, and the sharpening batch `rl_observation_batch(observation, tasks[0]["instruction"], device, config.proprio)`. Remove the `episodes = 0` line before the `while`. Episode start:

```python
        task = tasks[int(np.random.randint(len(tasks)))]
        task_id = task["task_id"]
        kind, variant, draw = draw_reset(config, episodes)
        env_task = task_id if variant is None else variant
        env = make_env(env_task, suite, init_states=kind == 2, plus=kind > 0, proprio=config.proprio)
        if kind == 2:
            env.init_state_id = int(draw * len(env._init_states))
        policy.reset()
        observation, _ = env.reset(seed=config.seed + env_steps)
```

Inside the chunk loop: `batch = rl_observation_batch(observation, task["instruction"], device, config.proprio)` then `sim_state = object_state(env._env.env, layout) if layout is not None else None`; `state = model.state_input(decoded["s"].to(device), decoded.get("state_struct"), batch.get("proprio"))`; `pool = model.plan_input(decoded["plan_pool"].to(device), decoded.get("plan_struct")) if plans > 1 else None`; `pools_all = model.plan_input(decoded["plan_pools"].to(device), decoded.get("plan_struct_all"))`; `s_cpu = _host_array(decoded["s"])[0]`. The row:

```python
                buffer.add_online({
                    "s": s_cpu,
                    "z": _host_array(noise[star]),
                    "a_base": _host_array(a_base[star]),
                    "z_all": _host_array(noise),
                    "a_base_all": _host_array(a_base),
                    "a": _host_array(chosen)[0],
                    "reward": np.float32(chunk_reward),
                    "done": np.float32(done),
                    "s_next": s_cpu.copy(),
                    "task_id": task_id,
                    "n_env_actions": executed_n,
                    "is_expert": np.float32(0.0),
                    "reset_kind": kind,
                    "variant_index": env_task,
                    **({"proprio": _host_array(batch["proprio"])[0], "object_state": sim_state} if config.proprio else {}),
                    **{key: host_bf16(decoded[key][0]) for key in ("state_struct", "state_tokens", "state_latent") if key in decoded},
                    **({"plan_pool": _host_array(pool)[0], "plan_pools_all": _host_array(pools_all),
                        "plan_index": int(decoded["plan_index"]), "plan_a_base_all": _host_array(all_base)} if plans > 1 else {}),
                    **({"plan_struct": host_bf16(decoded["plan_struct"][0]), "plan_struct_all": host_bf16(decoded["plan_struct_all"]),
                        "plan_latent_all": host_bf16(decoded["plan_latents_all"])} if plans > 1 and "plan_struct_all" in decoded else {}),
                })
```

Update log: `**plan_info, **({f"{key}/{RESET_KINDS[kind]}": value for key, value in plan_info.items()} if config.plus_train else {}),`. Episode log:

```python
        name = RESET_KINDS[kind]
        run.log({
            "env_steps": env_steps, "episode_return": episode_return,
            "episode_success": episode_success, "episode_length": episode_length,
            "reset_kind": kind, "variant_index": env_task,
            **({f"episode_success_{name}": episode_success, f"episode_length_{name}": episode_length} if config.plus_train else {}),
        })
```

Note `plan_pool`, `plan_pools_all` and `s` stay the raw pools in both runs (the `decoded["plan_pool"]`/`decoded["plan_pools"]` tensors), while `pool`/`pools_all` are the model inputs used for ranking and the plan metrics; with the defaults the two coincide bit-for-bit.

- [ ] **Step 6: `brev/train.sh`**

Both usage strings append ` [--state-features pool|struct] [--proprio] [--plus]`. Defaults after `PLAN_CANDIDATES=1`: `STATE_FEATURES=pool`, `PROPRIO=0`, `PLUS=0`. Cases: `--state-features) STATE_FEATURES="$2"; shift 2 ;;`, `--proprio) PROPRIO=1; shift ;;`, `--plus) PLUS=1; shift ;;`. After `mkdir -p "$OUT"`:

```bash
PLUS_TRAIN='{}'
[ "$PLUS" = 1 ] && PLUS_TRAIN="$(PYTHONPATH="$DICE_REPO" python3 -c 'import json; from script.lingbot_rl_config import plus_train_variants; print(json.dumps({axis: list(variants) for axis, variants in plus_train_variants().items()}))')"
```

The config JSON python gains `, "state_features": sys.argv[10], "proprio": sys.argv[11] == "1", "plus_train": json.loads(sys.argv[12])` before the closing `}` and the arguments `"$STATE_FEATURES" "$PROPRIO" "$PLUS_TRAIN"` after `"$PLAN_CANDIDATES"`. The `run.sh` block becomes:

```bash
{
  echo '#!/usr/bin/env bash'
  echo 'set -euo pipefail'
  echo "source '$ENV_FILE'"
  echo "export PYTHONPATH='$DICE_REPO' HF_HOME='$DICE_DATA/cache/hub' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"
  echo "export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl LIBERO_CONFIG_PATH='$DICE_DATA/libero-config' TOKENIZERS_PARALLELISM=false"
  echo "export LEROBOT_SOURCE_ROOT='$DICE_DATA/lerobot' MODAL_PROFILE='$MODAL_PROFILE'"
  if [ "$PLUS" = 1 ]; then
    echo "source '$DICE_DATA/libero-plus-config/activate.sh'"
    echo "export PYTHONPATH='$DICE_REPO':\"\$PYTHONPATH\""
  fi
  echo "cd '$DICE_REPO'"
  if [ "$PLUS" = 1 ]; then
    printf 'exec "$PY" -m script.lingbot_rl_train train --config-json "$(cat %q)"' "$OUT/config.json"
  else
    printf 'exec %q -m script.lingbot_rl_train train --config-json "$(cat %q)"' "$PY" "$OUT/config.json"
  fi
  printf ' %q' "${ARGS[@]}"
  printf ' 2>&1 | tee -a %q\n' "$OUT/train.log"
} > "$OUT/run.sh"
```

`activate.sh` sets `PYTHONPATH`, `LIBERO_CONFIG_PATH` and `PY` for the plus install; the repo path is prepended afterwards and the exec line reads `$PY` at run time.

- [ ] **Step 7: Verify and commit**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py tests/test_lingbot_eval.py && .cache/eval-venv/bin/python -m script.lingbot_rl_config && bash -n brev/setup.sh brev/train.sh brev/sync.sh brev/pull.sh brev/eval.sh`

```bash
git add script/lingbot_rl_train.py brev/train.sh tests/test_lingbot_rl.py
git commit -m "Collect under LIBERO-Plus resets with proprioception, object poses and per-kind metrics; add --plus, --proprio and --state-features to train.sh."
```

---

### Task 7: Evaluation reconstructs the layout from the checkpoint

**Files:**
- Modify: `script/lingbot_rl_train.py` (`evaluate`), `docs/libero-plus/plus_eval.py`, `tests/test_lingbot_rl.py`
- Verify: `brev/eval.sh` (unchanged; the layout comes from the checkpoint, `--plans` and `--residual-input` stay as today)

**Interfaces:**
- Consumes: `DiceResidualModel.layout_of`, `load_inference_state_dict` mismatch check (Task 4); `make_env(proprio=)`, `run_rl_episode` (Task 6).
- Produces: `evaluate()` replaces `config.state_features` / `config.proprio` with the checkpoint layout, builds the env with `proprio`, runs `run_rl_episode`, and writes `"layout"` into `settings/shard_<n>.json`; the plus driver does the same with `residual_layout(args)` and records `"layout"` in `settings/run.json`.

- [ ] **Step 1: Write the failing test** (after `test_evaluate_records_video_exec_step_and_passes_it_to_the_policy`)

```python
def test_evaluate_reconstructs_the_input_layout_from_the_residual_checkpoint(tmp_path, monkeypatch):
    import script.lingbot_eval as eval_module
    import script.lingbot_rl_train as train
    envs = []

    class StubEnv:
        def __init__(self, **env_kwargs):
            envs.append(env_kwargs)

        def close(self):
            return None

    tasks = [{"task_id": task_id, "instruction": f"task-{task_id}", "initial_state_count": 50} for task_id in range(10)]
    stub = _StubPolicy()
    monkeypatch.setattr(train, "load_residual_policy", lambda *a: stub)
    monkeypatch.setattr(train, "LiberoEnv", StubEnv)
    monkeypatch.setattr(eval_module, "read_checkpoint_metadata", lambda path: {"architecture": {}, "normalization": _eval_normalization()})
    monkeypatch.setattr(eval_module, "describe_suite", lambda path: (object(), tasks))
    monkeypatch.setattr(train, "run_rl_episode", lambda policy, env, entry, instruction, norm, device, proprio, **k: {
        **entry, "success": True, "seconds": 1.0, "policy_steps": 3, "proprio": proprio})
    torch.save(DiceResidualModel(device="cpu", state_features="struct", proprio=True).inference_state_dict(), tmp_path / "residual.pt")
    prepared = tmp_path / "prepared.json"
    prepared.write_text(json.dumps({"checkpoint": "ckpt", "model_path": "model", "assets_path": "assets",
                                    "source_run": "libero30-sft", "checkpoint_step": 600}))
    summary = train.evaluate(config_from_dict({"task_ids": [9]}), prepared, tmp_path / "out",
                             residual_path=tmp_path / "residual.pt", shard=1, shards=50)
    assert summary["complete"] and (stub.residual_model.features, stub.residual_model.proprio) == ("struct", True)
    assert envs[0]["obs_type"] == "pixels_agent_pos"
    settings = json.loads((tmp_path / "out" / "settings" / "shard_1.json").read_text())
    assert settings["layout"] == {"state_features": "struct", "proprio": True} and settings["rl"]["state_features"] == "struct"
    episode = json.loads((tmp_path / "out" / "episodes" / "task_09" / "episode_001.json").read_text())
    assert episode["proprio"] is True
    pool_payload = DiceResidualModel(device="cpu").inference_state_dict()
    torch.save(pool_payload, tmp_path / "pool.pt")
    with pytest.raises(RuntimeError, match="layout mismatch"):
        train.evaluate(config_from_dict({"task_ids": [9], "plan_candidates": 2}), prepared, tmp_path / "bad",
                       residual_path=tmp_path / "pool.pt", shard=1, shards=50)
    pool_payload.pop("layout")
    torch.save(pool_payload, tmp_path / "legacy.pt")
    train.evaluate(config_from_dict({"task_ids": [9]}), prepared, tmp_path / "legacy", residual_path=tmp_path / "legacy.pt", shard=1, shards=50)
    assert (stub.residual_model.features, stub.residual_model.proprio) == ("pool", False) and envs[-1]["obs_type"] == "pixels"
```

- [ ] **Step 2: Run it**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py -k reconstructs_the_input_layout`
Expected: FAIL (`KeyError: 'layout'` in settings, or the model built as pool).

- [ ] **Step 3: Implement** `evaluate()` in `script/lingbot_rl_train.py`

Add `from dataclasses import replace` to the module imports and drop `run_episode` from the `script.lingbot_eval` import inside `evaluate` (it is no longer called there). Replace the model construction and loading (from `model = DiceResidualModel(...)` through the `except` clause) with:

```python
    residual_path = Path(residual_path or Path(output_dir) / "residual.pt")
    try:
        payload = torch.load(residual_path, map_location="cpu", weights_only=True)
        layout = DiceResidualModel.layout_of(payload)
        config = replace(config, **layout).validate()
        model = DiceResidualModel(device=device, residual_input=config.residual_input, epsilon=config.epsilon,
                                  plan_candidates=config.plan_candidates, state_features=config.state_features, proprio=config.proprio)
        model.load_inference_state_dict(payload)
    except Exception as exc:
        raise RuntimeError(f"Failed to load residual weights {residual_path}: {exc}") from exc
```

Add `"layout": layout,` to the shard settings dict. In the episode loop: `env = make_env(task["task_id"], suite, proprio=config.proprio)` and `rows.append(run_rl_episode(policy, env, entry, task["instruction"], metadata["normalization"], device, config.proprio))`.

- [ ] **Step 4: Plus driver** `docs/libero-plus/plus_eval.py`

Replace `load`:

```python
def residual_layout(args):
    from script.lingbot_rl_model import DiceResidualModel

    if not args.residual:
        return {"state_features": "pool", "proprio": False}
    return DiceResidualModel.layout_of(torch.load(args.residual, map_location="cpu", weights_only=True))


def load(args, prepared, architecture, layout):
    if args.policy == "sft":
        return load_policy(prepared["checkpoint"], prepared["model_path"], architecture, -1, args.video_steps, args.action_steps)
    from script.lingbot_rl_model import DiceResidualModel
    from script.lingbot_rl_policy import load_residual_policy

    policy = load_residual_policy(prepared["checkpoint"], prepared["model_path"], architecture, -1, args.video_steps, args.action_steps)
    model = DiceResidualModel(device="cuda", residual_input=args.residual_input, epsilon=args.epsilon,
                              plan_candidates=args.plan_candidates, **layout)
    model.load_inference_state_dict(torch.load(args.residual, map_location="cpu", weights_only=True))
    policy.residual_model = model
    policy.eval_candidates = args.eval_candidates
    policy.eval_plans = args.plan_candidates
    policy.eval_plan_ranker = args.plan_ranker
    return policy
```

In `main`: `layout = residual_layout(args)` before `settings`, with `"layout": layout` added to `settings`; `policy = load(args, prepared, metadata["architecture"], layout)`; the env line becomes `env = LiberoEnv(task_suite=suite, task_id=entry["variant_index"], init_states=entry["reset_rule"] == "stored_state", **{**ENV_KWARGS, "obs_type": "pixels_agent_pos" if layout["proprio"] else "pixels"})`; the episode line becomes `row = run_rl_episode(policy, env, entry, instruction, metadata["normalization"], "cuda", layout["proprio"], video_writer=writer)` with `from script.lingbot_rl_train import run_rl_episode` added to the imports (and `run_episode` dropped from the `script.lingbot_eval` import). The driver's `driver_sha256` changes, so an existing `--out` of the random-plan ablations must not be resumed with the new driver; do not overwrite `~/Documents/dice/libero-plus-config/plus_eval.py` on a box while an ablation is running.

- [ ] **Step 5: Verify and commit**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py tests/test_lingbot_eval.py && .cache/eval-venv/bin/python -m compileall -q script/ docs/libero-plus/plus_eval.py && bash -n brev/eval.sh`

```bash
git add script/lingbot_rl_train.py docs/libero-plus/plus_eval.py tests/test_lingbot_rl.py
git commit -m "Reconstruct the residual model's input layout from the checkpoint in evaluate and the LIBERO-Plus driver."
```

---

### Task 8: README: the variant split

**Files:**
- Modify: `docs/libero-plus/README.md`

- [ ] **Step 1: Append** after the "Files:" paragraph:

```markdown
## Variant split for training under perturbations

Within each axis the variants are ordered by suite index and every third one from the third is held out
(`script/lingbot_rl_config.py`: `plus_split`, `plus_train_variants`). The held-out sets cover every difficulty level
present on their axis.

- robot initial state, train (28): 640, 641, 643, 644, 646, 647, 649, 650, 652, 653, 655, 656, 658, 659, 661, 662, 664,
  665, 667, 668, 670, 671, 673, 674, 676, 677, 679, 680; test (14): 642, 645, 648, 651, 654, 657, 660, 663, 666, 669,
  672, 675, 678, 681.
- objects layout, train (26): 1982, 1983, 1985, 1986, 1988, 1989, 1991, 1992, 1994, 1995, 1997, 1998, 2000, 2001, 2003,
  2004, 2006, 2007, 2009, 2010, 2012, 2013, 2015, 2016, 2018, 2019; test (13): 1984, 1987, 1990, 1993, 1996, 1999, 2002,
  2005, 2008, 2011, 2014, 2017, 2020.

Collection (`brev/train.sh --plus`) draws per episode, with equal probability from `np.random.default_rng([seed, episode])`,
a standard reset (`task_id 9`, `init_states=False`, seed `config.seed + env_steps`), a robot-init training variant
(`init_states=False, is_libero_plus=True`, same seed rule as the plan above) or a layout training variant
(`init_states=True`, uniform over its stored states). The held-out plan over the test variants
(`task9-heldout-plan.json`) is built after the runs are launched.
```

- [ ] **Step 2: Commit**

```bash
git add docs/libero-plus/README.md
git commit -m "Document the LIBERO-Plus task 9 variant split used for training under perturbations."
```

---

### Task 9: GB10 checks, smoke and launch (controller)

Run on one GB10 (Jarvis) from `~/Documents/dice-rl-wam` after `git pull`; `DICE_DATA=~/Documents/dice`; the standard install is what `brev/env.sh` gives, the plus install is `source ~/Documents/dice/libero-plus-config/activate.sh`. Nothing here is run by implementers.

- [ ] **Step 1: Proprioception convention (gates the launch).** Standard install:

```bash
cd ~/Documents/dice-rl-wam && source brev/env.sh && export PYTHONPATH=$PWD HF_HOME=$DICE_DATA/cache/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl LIBERO_CONFIG_PATH=$DICE_DATA/libero-config
$DICE_DATA/lerobot/.venv/bin/python -m pytest -q tests/test_lingbot_rl.py -k demo_frame -rs
```

Expected: `1 passed` (not skipped). If it fails with gaps only in entries 3–5, the dataset's axis-angle convention differs from `proprio_from_observation`; stop and report the printed per-entry gap vector rather than loosening the tolerance.

- [ ] **Step 2: The plus install reproduces the standard task 9 (spec section 7).** Run this snippet under both installs (standard as above; plus: same exports then `source ~/Documents/dice/libero-plus-config/activate.sh; export PYTHONPATH=$PWD:$PYTHONPATH` and use `$PY`) and diff the two outputs:

```bash
$PY - <<'EOF'
import json, numpy as np
from lerobot.envs.libero import LiberoEnv, _get_suite
from script.lingbot_eval import seed_all
from script.lingbot_rl_train import object_layout, object_state, proprio_from_observation
suite = _get_suite("libero_10")
env = LiberoEnv(task_suite=suite, task_id=9, task_suite_name="libero_10", episode_length=520, observation_height=128, observation_width=128,
                obs_type="pixels_agent_pos", init_states=True, n_envs=1, num_steps_wait=10, control_freq=20, control_mode="relative", hard_reset=True)
env.init_state_id = 0
seed_all(0)
observation, _ = env.reset(seed=0)
inner = env._env.env
layout = object_layout(inner)
print(json.dumps({"task": suite.get_task(9).name, "n_tasks": suite.get_num_tasks(), "layout": layout,
                  "proprio": proprio_from_observation(observation).round(6).tolist(), "object_state": object_state(inner, layout).round(6).tolist(),
                  "pixels_sum": int(observation["pixels"]["image"].astype(int).sum())}, indent=1))
env.close()
EOF
```

Expected: `task` is `KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it` under both; `n_tasks` 10 (standard) vs the plus count; `layout`, `proprio`, `object_state` and `pixels_sum` identical. Record the printed `layout` (the object names and articulated joints for task 9 cannot be verified from the Mac) in docs/heldout-results.md with the run; if `joints` is empty, the microwave door joint was not found under its body and `object_layout` must be revisited before launch.

- [ ] **Step 3: Smoke, run A then run B** (plus install via `--plus`; `DICE_NO_SYNC=1` in env.sh as for the previous runs):

```bash
bash brev/train.sh smoke-plus-pool 9 --residual-input base --epsilon -0.3 --video-steps 5 --action-steps 10 --plan-candidates 4 --env-steps 320000 --plus --proprio --state-features pool --max-env-steps 2100 --no-tmux
bash brev/train.sh smoke-plus-struct 9 --residual-input base --epsilon -0.3 --video-steps 5 --action-steps 10 --plan-candidates 4 --env-steps 320000 --plus --proprio --state-features struct --max-env-steps 2100 --no-tmux
```

Each smoke first rebuilds `expert_features.pt` under recipe v3 (15–25 min) and then runs the first four episodes (kinds 0, 1, 0, 2 with variants 673 and 1989; 2100 steps covers four full 520-step episodes). Read:

- `$DICE_DATA/runs/smoke-plus-*/settings.json`: `config.proprio` true, `config.state_features`, `config.plus_train` with 28 + 26 entries, `recipe.collection_mix` = `thirds_standard_robot_init_layout`, `object_layout` equal to Step 2's, `runtime.cuda` starting with 13.
- W&B run `smoke-plus-pool` / `smoke-plus-struct` (or `$DICE_REPO/wandb/latest-run/files/wandb-summary.json`): `peak_gpu_memory_gb` and `seconds_per_env_step` for both (run B's first layers are about 320 M parameters; compare against the `dice-t9-plan-m4` smoke), `plan_agreement/<kind>` keys, and the episode rows with `reset_kind` 0, 1, 0, 2 and `variant_index` 9, 673, 9, 1989.
- Buffer keys and dtypes from the resume file:

```bash
$PY - <<'EOF'
import torch, collections
payload = torch.load("/home/$USER/Documents/dice/runs/smoke-plus-struct/resume/latest.pt", map_location="cpu", weights_only=False)
rows = [row for row in payload["replay"]["data"] if float(row["is_expert"]) == 0.0]
print(payload["episodes"], collections.Counter(row["reset_kind"] for row in rows), collections.Counter(row["variant_index"] for row in rows))
for key in ("s", "proprio", "proprio_next", "state_struct", "state_struct_next", "state_tokens", "state_latent", "plan_pool", "plan_struct", "plan_struct_all", "plan_struct_next", "plan_latent_all", "object_state"):
    value = rows[-1][key]
    print(key, tuple(value.shape), getattr(value, "dtype", None))
expert = next(row for row in payload["replay"]["data"] if float(row["is_expert"]) == 1.0)
print("expert", tuple(expert["proprio"].shape), tuple(expert["plan_struct"].shape), tuple(expert["object_state"].shape), float(abs(expert["proprio"]).max()))
EOF
```

Expected: `episodes` 4; reset kinds {0: 2, 1: 1, 2: 1}; shapes (3072,), (8,), (8,), (2, 3072) bf16, (2, 3072) bf16, (32, 3072) bf16, (48, 1, 8, 16) bf16, (3072,), (8, 3072) bf16, (4, 8, 3072) bf16, (8, 3072) bf16, (4, 48, 4, 8, 16) bf16, (7·len(bodies) + len(joints),) float32; the expert row has a non-zero `proprio`, `plan_struct` (8, 3072) and an empty `object_state`. Also record the resume file size (expect about 0.7 MB per online chunk).

- [ ] **Step 4: Launch** run A on Jarvis and run B on Watson:

```bash
bash brev/train.sh dice-t9-plus-m4-pool 9 --residual-input base --epsilon -0.3 --env-steps 320000 --video-steps 5 --action-steps 10 --plan-candidates 4 --plus --proprio --state-features pool
bash brev/train.sh dice-t9-plus-m4-struct 9 --residual-input base --epsilon -0.3 --env-steps 320000 --video-steps 5 --action-steps 10 --plan-candidates 4 --plus --proprio --state-features struct
```

The expert features rebuild once more per run directory (copy `smoke-plus-*/expert_features.pt` into the run directory first to skip it; the cache is keyed by the recipe fingerprint, not the run). Checkpoints every 80k; the 240k checkpoint is the decision point (spec section 5); resume files reach about 16 GB.

- [ ] **Step 5: Record** the smoke numbers (memory, seconds per env step for both runs), the object layout, and the launch in `docs/heldout-results.md`.

## Self-review notes

- Spec coverage: section 1 runs / budget → Task 9 commands (320k, 80k checkpoints from the defaults); collection distribution and the two reset rules → Task 6 (`draw_reset`, `make_env`, `init_state_id`); variant split → Task 1 constants and Task 8 README; proprioception convention and extraction → Task 6 (`proprio_from_observation`, GB10 test), expert side → Task 5; run A / run B inputs → Task 4 (`state_input`, `plan_input`, dims); stored features incl. `object_state`, `reset_kind`, `variant_index` → Tasks 3, 5, 6; W&B config and per-kind metrics → Task 6; evaluation reconstruction → Task 7; section 3 `brev/train.sh` flags → Task 6; README → Task 8; section 3 tests → Tasks 1 (fingerprint), 2 (structured means), 3 (buffer round trip), 6 (reset draw, proprio GB10), 5 (expert v3), 4 (layout); section 6 step 1 smoke → Task 9; section 7 reset-equality check → Task 9 Step 2. Not in this plan: the offline feature study at the 80k checkpoints and the held-out plan / baselines (spec sections 1 "offline study", 4, 6 steps 3–5), which the spec defers to after the runs.
- Type consistency: `state_input(pool, struct, proprio)` / `plan_input(pool, struct)` have the same signature in the model (Task 4), the policy helpers (Task 4), the stub (Task 6) and the trainer (Task 6); `plan_struct_all` is (M, 8, 3072) in the policy, buffer, expert rows (as `plan_struct` repeated) and `update_plan_critic` after `plan_input`; `draw_reset` returns `(kind, variant, draw)` everywhere; `load_resume` returns five values in Task 6 and its tests.
- Bit-for-bit with defaults: `ResidualActor` / `CriticEnsemble` / `PlanCritic` widths and construction order are unchanged; `state_input` returns `mlp_float(pool)` and `plan_input` `mlp_float(pool)`; `draw_reset` returns without touching `np.random`; the object probe only runs with `proprio`; `run_rl_episode` delegates to `run_episode`; the extra per-kind log keys are gated on `plus_train`; `s`, `plan_pool`, `plan_pools_all` store the same raw pools as before.
- Fingerprint: `state_features`, `proprio`, `plus_train`, `collection_mix` are protocol keys, so any change refuses an older resume (`test_load_resume_rejects_recipe_mismatch`).
- Placeholder scan: every code step shows the code; the only unverifiable items (dataset axis-angle convention, MuJoCo binding attribute names, task 9 object names, the plus suite's task 9) are pinned to Task 9 Steps 1–2 with the exact commands and the expected output.
