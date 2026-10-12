# Plan-level selection implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the residual recipe with M = 4 sampled video plans per chunk, ranked by a regression-trained plan critic, at the 5 / 10 sampler, on tasks 9 and 0, one GB10 each.

**Architecture:** The policy denoises M plans in one batch-2M pass sequence against a row-expanded KV cache, pools every denoised chunk through the existing history-free critic pool, decodes K candidates per plan in one batch, and keeps only the winner's cache rows. The model gains a plan critic V_plan(s, p) trained by regression onto max-over-candidates min-ensemble Q, and Q takes the plan pool as an extra input. Replay rows carry the M pools and M × K base actions; expert rows carry the pool of the demo's true next chunk.

**Tech Stack:** Torch 2.11 (cu130 on the GB10s), the pinned LeRobot LingBot-VA port, numpy replay, pytest with the mocked transformer in `tests/test_lingbot_rl.py`.

**Spec:** docs/superpowers/specs/2026-10-09-plan-selection-design.md.

## Global Constraints

- Zero comments and docstrings in `script/lingbot_rl_*.py`; lean code; tests must not call Modal or `.remote()`.
- Tests: only the ones listed, each answering one question we need answered; no helper-level or plumbing tests, no narrative names beyond what the assertion needs.
- Do not edit `.cache/lerobot`; do not change `script/lingbot_eval.py` protocol fields.
- `plan_candidates = 1` must reproduce today's recipe bit for bit (policy path, critic input width, buffer sampling, fingerprint apart from the new keys).
- The port's scheduler is `FlowMatchScheduler(shift, sigma_min=0, extra_one_step=True)`: n video steps = n denoising passes whose last lands on σ = 0, plus one padded write-only pass at t = 0. A `scheduler.step` at the padded t = 0 is a corruption (it applies −0.556 × pred at 5 steps); the padded pass must never advance when `video_exec_step = -1`.
- Sampler values stay in config (`video_steps`, `action_steps`, `video_exec_step`); the runs use 5 / 10 / −1.
- Every resume file saved before this change fails the recipe fingerprint; the runs use fresh names and the boxes run with `DICE_NO_SYNC=1`, so no Modal resume is pulled.
- Secrets never printed; no new experiments launched by implementers (Task 7 is run by the controller on a GB10).
- Verify with `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py tests/test_lingbot_eval.py`, `.cache/eval-venv/bin/python -m script.lingbot_rl_config`, `bash -n brev/setup.sh brev/train.sh brev/sync.sh brev/pull.sh brev/eval.sh`.

## File map

| file | responsibility after this plan |
|---|---|
| `script/lingbot_rl_config.py` | `plan_candidates`, `plan_random_prob`; validation; fingerprint keys |
| `script/lingbot_rl_model.py` | `CriticEnsemble(state_dim)`, `PlanCritic`, `critic_state`, `rank_plans`, `update_plan_critic`, plan-aware targets and checkpoints |
| `script/lingbot_rl_buffer.py` | plan keys with defaults applied on store and on load; `plan_pool_next` at finalisation |
| `script/lingbot_rl_policy.py` | batched M-plan sampling (`select_plan`), batched M × K decode, batch-capable `_pool_from_latent`, plan-aware `predict_action_chunk` |
| `script/lingbot_rl_data.py` | expert plan pools from published latents; expert recipe v2 |
| `script/lingbot_rl_train.py` | M-plan collection, random-plan schedule, V_plan update, plan and throughput metrics, plan-aware eval and sharpening |
| `brev/train.sh`, `brev/eval.sh` | `--plan-candidates` / `--plans` |

---

### Task 1: Config fields and fingerprints

**Files:**
- Modify: `script/lingbot_rl_config.py`, `brev/train.sh`, `brev/eval.sh`
- Test: `tests/test_lingbot_rl.py`

**Interfaces:**
- Produces: `RLConfig.plan_candidates: int = 1`, `RLConfig.plan_random_prob: float = 0.1`; protocol keys `plan_candidates`, `plan_random_prob`, `plan_critic`, `plan_pool`.

- [ ] **Step 1: Write the failing test**

```python
def test_config_plan_fields_enter_the_fingerprint():
    base = RLConfig().protocol()
    assert base["plan_candidates"] == 1 and base["plan_critic"] == "none" and base["plan_pool"] == "none"
    plan = RLConfig(plan_candidates=4).protocol()
    assert plan["plan_critic"] == "regression_max_k_min_q" and plan["plan_pool"] == "critic_pool_of_denoised_chunk"
    assert plan != base
    with pytest.raises(ValueError):
        RLConfig(plan_candidates=9).validate()
```

- [ ] **Step 2: Run it**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py -k plan_fields`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'plan_candidates'`.

- [ ] **Step 3: Implement** in `script/lingbot_rl_config.py`

Fields after `epsilon`:

```python
    plan_candidates: int = 1
    plan_random_prob: float = 0.1
```

In `validate()` before `return self`:

```python
        if type(self.plan_candidates) is not int or not 1 <= self.plan_candidates <= 8:
            raise ValueError("plan_candidates must be an integer in [1, 8]")
        if not 0.0 <= float(self.plan_random_prob) <= 1.0:
            raise ValueError("plan_random_prob must lie in [0, 1]")
```

In `protocol()` after `"epsilon": self.epsilon,`:

```python
            "plan_candidates": self.plan_candidates,
            "plan_random_prob": self.plan_random_prob,
            "plan_critic": "regression_max_k_min_q" if self.plan_candidates > 1 else "none",
            "plan_pool": "critic_pool_of_denoised_chunk" if self.plan_candidates > 1 else "none",
```

- [ ] **Step 4: Script flags**

`brev/train.sh`: `PLAN_CANDIDATES=1` beside `ACTION_STEPS=50`; case `--plan-candidates) PLAN_CANDIDATES="$2"; shift 2 ;;`; `[--plan-candidates M]` in both usage strings; the config JSON gains `"plan_candidates": int(sys.argv[9])` with `"$PLAN_CANDIDATES"` appended to the argument list.

`brev/eval.sh`: `PLANS=1`; case `--plans) PLANS="$2"; shift 2 ;;`; `[--plans M]` in `USAGE`; in the run branch add `"plan_candidates": int(sys.argv[6])` with `"$PLANS"` appended, and `POLICY="$RUN-step$STEP6-k$K-m$PLANS$SUFFIX"` when `PLANS` is not 1 (unchanged name when it is). `script/lingbot_eval_modal.py` is not extended: evaluation of plan runs is GB10-only.

- [ ] **Step 5: Verify and commit**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py && .cache/eval-venv/bin/python -m script.lingbot_rl_config && bash -n brev/train.sh brev/eval.sh`

```bash
git add script/lingbot_rl_config.py brev/train.sh brev/eval.sh tests/test_lingbot_rl.py
git commit -m "Add plan_candidates and plan_random_prob to the RL recipe and the Brev scripts."
```

---

### Task 2: Plan critic and plan-aware action critic

**Files:**
- Modify: `script/lingbot_rl_model.py`, `script/lingbot_rl_train.py` (call sites only), `tests/test_lingbot_rl.py`

**Interfaces:**
- Produces:
  - `CriticEnsemble(state_dim=STATE_DIM)`; forward unchanged `(state, action, return_all=False)`.
  - `PlanCritic()` with forward `(state, pool, return_all=False)`, inputs `(B, 3072)` each.
  - `DiceResidualModel(device, residual_input, epsilon, plan_candidates=1)` with `plans`, `plan_critic`, `plan_opt`, `plan_lr` (`None` when `plans == 1`).
  - `model.critic_state(state, pool)` → `mlp_float(state)` when `plans == 1`, else `cat([state, pool], -1)`; `pool` may be `None` when `plans == 1`.
  - `model.rank_plans(state (1, 3072), pools (M, 3072))` → `int` argmax of min-ensemble V_plan.
  - `model.n_step_target(reward, done, next_critic_state, z_next_all, a_base_next_all, n_steps)`.
  - `model.update_critic(critic_state, action, target_q)`.
  - `model.update_actor(state, critic_state, action_stored, z_all, a_base_all, is_expert, mc_return, filter_active)`.
  - `model.update_plan_critic(state, pools_all (B, M, 3072), z_all (B, K, 16, 30), plan_a_base_all (B, M, K, 16, 30))` → `{"plan_critic_loss", "v_plan_mean", "plan_target_mean"}`.
  - `inference_state_dict()` adds `"plan_critic"` when `plans > 1`; `resume_state_dict()` adds `"plan_critic"`, `"plan_opt"`, `"plan_lr"`.

- [ ] **Step 1: Write the failing test**

```python
def test_plan_critic_regresses_onto_max_candidate_min_q_and_leaves_q_and_actor_alone():
    torch.manual_seed(0)
    single = DiceResidualModel(device="cpu")
    model = DiceResidualModel(device="cpu", residual_input="base", plan_candidates=2)
    assert single.plan_critic is None and single.critic.heads[0][0].in_features == STATE_DIM + HORIZON * ACTION_DIM
    assert model.critic.heads[0][0].in_features == 2 * STATE_DIM + HORIZON * ACTION_DIM
    assert set(single.inference_state_dict()) == {"actor", "critic", "target_critic"}
    assert {"plan_critic", "plan_opt", "plan_lr"} <= set(model.resume_state_dict())
    for parameter in model.critic.parameters():
        torch.nn.init.normal_(parameter, std=0.05)
    state, pools = torch.randn(5, STATE_DIM), torch.randn(5, 2, STATE_DIM)
    z_all = torch.randn(5, 4, HORIZON, ACTION_DIM)
    a_base = mask_unused_dof(torch.randn(5, 2, 4, HORIZON, ACTION_DIM))
    critic_before = [p.detach().clone() for p in model.critic.parameters()]
    actor_before = [p.detach().clone() for p in model.actor.parameters()]
    with torch.no_grad():
        flat_state = state[:, None, None].expand(5, 2, 4, STATE_DIM).reshape(-1, STATE_DIM)
        flat_pool = pools[:, :, None].expand(5, 2, 4, STATE_DIM).reshape(-1, STATE_DIM)
        flat_z = z_all[:, None].expand(5, 2, 4, HORIZON, ACTION_DIM).reshape(-1, HORIZON, ACTION_DIM)
        flat_base = a_base.reshape(-1, HORIZON, ACTION_DIM)
        action = apply_residual(flat_base, model.actor(flat_state, flat_z, flat_base))
        expected = model.critic(model.critic_state(flat_state, flat_pool), action).reshape(5, 2, 4).max(dim=2).values
    losses = [model.update_plan_critic(state, pools, z_all, a_base)["plan_critic_loss"] for _ in range(30)]
    info = model.update_plan_critic(state, pools, z_all, a_base)
    assert losses[-1] < losses[0] and abs(info["plan_target_mean"] - float(expected.mean())) < 1e-5
    assert all(torch.equal(a, b) for a, b in zip(critic_before, model.critic.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(actor_before, model.actor.parameters()))
    values = model.plan_critic(state[:1].expand(2, -1), pools[0])
    assert model.rank_plans(state[:1], pools[0]) == int(values.reshape(-1).argmax())
```

- [ ] **Step 2: Run it**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py -k plan_critic_regresses`
Expected: FAIL with `TypeError` on `plan_candidates`.

- [ ] **Step 3: Implement** in `script/lingbot_rl_model.py`

```python
class CriticEnsemble(nn.Module):
    def __init__(self, state_dim=STATE_DIM):
        super().__init__()
        self.heads = nn.ModuleList([_mlp(state_dim + HORIZON * ACTION_DIM, 1) for _ in range(ENSEMBLE)])

    def forward(self, state, action, return_all=False):
        state = mlp_float(state)
        batch = state.shape[0]
        x = torch.cat([state, mask_unused_dof(action).reshape(batch, -1)], dim=-1)
        qs = [head(x) for head in self.heads]
        if return_all:
            return qs
        return torch.min(torch.stack(qs, dim=0), dim=0).values


class PlanCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([_mlp(2 * STATE_DIM, 1) for _ in range(ENSEMBLE)])

    def forward(self, state, pool, return_all=False):
        x = torch.cat([mlp_float(state), mlp_float(pool)], dim=-1)
        vs = [head(x) for head in self.heads]
        if return_all:
            return vs
        return torch.min(torch.stack(vs, dim=0), dim=0).values
```

In `DiceResidualModel.__init__(self, device="cpu", residual_input="z", epsilon=EPSILON, plan_candidates=1)`: `self.plans = int(plan_candidates)`; `self.critic = CriticEnsemble(STATE_DIM * (2 if self.plans > 1 else 1)).to(device)` before the target copy; after the schedulers:

```python
        self.plan_critic = self.plan_opt = self.plan_lr = None
        if self.plans > 1:
            self.plan_critic = PlanCritic().to(device)
            self.plan_opt = torch.optim.Adam(self.plan_critic.parameters(), lr=ADAM_LR, weight_decay=WEIGHT_DECAY)
            self.plan_lr = torch.optim.lr_scheduler.LambdaLR(self.plan_opt, cosine_restart_lr)

    def critic_state(self, state, pool):
        if self.plans == 1:
            return mlp_float(state)
        return torch.cat([mlp_float(state), mlp_float(pool)], dim=-1)

    def rank_plans(self, state, pools):
        with torch.no_grad():
            pools = mlp_float(pools)
            return int(self.plan_critic(mlp_float(state).expand(pools.shape[0], -1), pools).reshape(-1).argmax())
```

`n_step_target`: third parameter `next_critic_state`; expand it over `k` into `critic_k` exactly as `next_state` is expanded today; `state_k = critic_k[:, :STATE_DIM]` feeds the actor; `critic_k` feeds `self.target_critic`.

`update_critic(self, critic_state, action, target_q)`: body unchanged.

`update_actor(self, state, critic_state, action_stored, z_all, a_base_all, is_expert, mc_return, filter_active)`: build `critic_k` from `critic_state` as `state_k` is built from `state`; use `critic_k` in the two `self.critic(...)` calls on `action` and `base_flat`, and `critic_state` in the call on `action_stored`; the actor keeps `state_k`.

```python
    def update_plan_critic(self, state, pools_all, z_all, plan_a_base_all):
        state, pools_all = mlp_float(state), mlp_float(pools_all)
        z_all, plan_a_base_all = mlp_float(z_all), mlp_float(plan_a_base_all)
        batch, plans, k = plan_a_base_all.shape[:3]
        with torch.no_grad():
            flat_state = state[:, None, None].expand(batch, plans, k, STATE_DIM).reshape(-1, STATE_DIM)
            flat_pool = pools_all[:, :, None].expand(batch, plans, k, STATE_DIM).reshape(-1, STATE_DIM)
            flat_z = z_all[:, None].expand(batch, plans, k, HORIZON, ACTION_DIM).reshape(-1, HORIZON, ACTION_DIM)
            flat_base = plan_a_base_all.reshape(-1, HORIZON, ACTION_DIM)
            action = apply_residual(flat_base, self.actor(flat_state, flat_z, flat_base))
            target = self.critic(self.critic_state(flat_state, flat_pool), action).reshape(batch, plans, k).max(dim=2).values
        preds = self.plan_critic(
            state[:, None].expand(batch, plans, STATE_DIM).reshape(-1, STATE_DIM), pools_all.reshape(-1, STATE_DIM), return_all=True)
        loss = torch.stack([F.mse_loss(pred.reshape(batch, plans), target) for pred in preds]).sum()
        self.plan_opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.plan_critic.parameters(), MAX_GRAD_NORM)
        self.plan_opt.step()
        self.plan_lr.step()
        return {"plan_critic_loss": float(loss.detach()), "v_plan_mean": float(torch.stack(preds).mean().detach()),
                "plan_target_mean": float(target.mean())}
```

Checkpoints: `inference_state_dict` adds `"plan_critic"` when present; `load_inference_state_dict` loads it when both sides have it and raises `ValueError("plan critic weights missing")` when the model has one and the payload does not; `resume_state_dict` adds `plan_opt` and `plan_lr` when present; `load_resume_state_dict` restores them when present.

- [ ] **Step 4: Call sites**

`script/lingbot_rl_train.py` (finished in Task 6): in `_update_from_buffer` pass `sample["s"]` as both `state` and `critic_state`, `sample["s_next"]` as `next_critic_state`; `save_inference` allows `{"actor", "critic", "target_critic", "plan_critic"}`; `RESUME_KEYS` gains `"plan_critic", "plan_opt", "plan_lr"`. Update the existing direct callers of `update_actor` in the tests (pass the state twice) and give `_sum_critic_model` a `plan_candidates=1` parameter forwarded to `DiceResidualModel`.

- [ ] **Step 5: Verify and commit**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py`

```bash
git add script/lingbot_rl_model.py script/lingbot_rl_train.py tests/test_lingbot_rl.py
git commit -m "Add the regression-trained plan critic and the plan-pool input to the action critic."
```

---

### Task 3: Replay keys for plans

**Files:**
- Modify: `script/lingbot_rl_buffer.py`, `tests/test_lingbot_rl.py`

**Interfaces:**
- Produces: row keys `plan_pool (3072)`, `plan_pools_all (M, 3072)`, `plan_index (int)`, `plan_a_base_all (M, K, 16, 30)`, `plan_pool_next (3072)`; `ChunkReplay(capacity, device, plans=1)`; `sample()` returns them (`plan_index` as `torch.long`); rows loaded from an older `state_dict` get the same defaults.

- [ ] **Step 1: Write the failing test**

```python
def test_plan_rows_default_finalize_and_sample_with_one_shape():
    buffer = ChunkReplay(plans=3)
    first = _replay_row(0.0, 0.0)
    first["plan_pool"] = np.full(STATE_DIM, 1.0, np.float32)
    buffer.add_online(first)
    second = _replay_row(1.0, 1.0)
    second["plan_pool"] = np.full(STATE_DIM, 2.0, np.float32)
    second["plan_pools_all"] = np.stack([np.full(STATE_DIM, v, np.float32) for v in (2.0, 5.0, 7.0)])
    second["plan_index"] = 0
    buffer.add_online(second)
    buffer.finalize_episode()
    rows = buffer.rows()
    assert rows[0]["plan_pools_all"].shape == (3, STATE_DIM) and float(rows[0]["plan_pools_all"][2, 0]) == 1.0
    assert np.array_equal(rows[0]["plan_a_base_all"][1], rows[0]["a_base_all"])
    assert float(rows[0]["plan_pool_next"][0]) == 2.0 and float(rows[1]["plan_pool_next"][0]) == 2.0
    batch = buffer.sample(2, 0.0)
    assert batch["plan_pools_all"].shape == (2, 3, STATE_DIM) and batch["plan_index"].dtype == torch.long
    assert batch["plan_a_base_all"].shape == (2, 3, K_CANDIDATES, HORIZON, ACTION_DIM)
    legacy = ChunkReplay(plans=3)
    payload = buffer.state_dict()
    for row in payload["data"]:
        for key in ("plan_pool", "plan_pools_all", "plan_index", "plan_a_base_all", "plan_pool_next"):
            row.pop(key)
    legacy.load_state_dict(payload)
    assert legacy.sample(2, 0.0)["plan_pools_all"].shape == (2, 3, STATE_DIM)
```

- [ ] **Step 2: Run it**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py -k plan_rows`
Expected: FAIL (`TypeError` on `plans=`).

- [ ] **Step 3: Implement** in `script/lingbot_rl_buffer.py`

Import `STATE_DIM`. Extend `FLOAT_KEYS` with `"plan_pool", "plan_pools_all", "plan_a_base_all", "plan_pool_next"` and `KEYS` with those plus `"plan_index"`. `__init__(self, capacity=REPLAY_CAPACITY, device="cpu", plans=1)` sets `self.plans = int(plans)`. Add:

```python
    def _plan_defaults(self, stored):
        stored.setdefault("plan_pool", np.zeros(STATE_DIM, np.float32))
        stored.setdefault("plan_pools_all", np.repeat(_host_float(stored["plan_pool"])[None], self.plans, axis=0))
        stored.setdefault("plan_index", 0)
        stored.setdefault("plan_a_base_all", np.repeat(_host_float(stored["a_base_all"])[None], self.plans, axis=0))
        stored.setdefault("plan_pool_next", np.array(stored["plan_pool"], copy=True))
        if _host_float(stored["plan_pools_all"]).shape[0] != self.plans or \
                _host_float(stored["plan_a_base_all"]).shape[0] != self.plans:
            raise ValueError(f"Rows must carry exactly {self.plans} plans")
        stored["plan_index"] = int(stored["plan_index"])
        return stored
```

Call it in `_store` after the `a_base_all` defaults and before the `FLOAT_KEYS` loop. In `finalize_episode`, beside `s_next`: `row["plan_pool_next"] = np.array(episode[nxt]["plan_pool"], copy=True)`. In `sample`, treat `plan_index` like `task_id`. `state_dict` adds `"plans": self.plans`; `load_state_dict` restores `self.plans = payload.get("plans", self.plans)` and applies `_plan_defaults` plus the `FLOAT_KEYS` conversion for the plan keys to every loaded row.

- [ ] **Step 4: Verify and commit**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py`

```bash
git add script/lingbot_rl_buffer.py tests/test_lingbot_rl.py
git commit -m "Carry plan pools, plan candidates and the next executed pool in replay rows."
```

---

### Task 4: Batched M-plan sampling in the policy

**Files:**
- Modify: `script/lingbot_rl_policy.py`, `tests/test_lingbot_rl.py` (`_mock_policy` and three tests)

**Interfaces:**
- Consumes: `model.rank_plans(state, pools)` through `self.residual_model`.
- Produces:
  - `expand_conditional_kv(transformer, k, plans=1)`: the first `plans` rows each repeated `k` times (`repeat_interleave`); `plans=1` equals today.
  - `expand_kv_rows(transformer, reps)`: every row repeated `reps` times (`[c, u]` → `[c × reps, u × reps]`); `select_kv_rows(transformer, rows)`: keep the listed rows.
  - `_pool_from_latent(latent)` accepting batch `B` and `T` in {1, 4}, returning `(B, 3072)`; empty critic caches named `"critic"` for `B = 1` and `"critic_m"` otherwise, recreated only when the batch changes.
  - `_video_pass_batched(latents (M, ...), t, latent_cond, frame_st_id, write, advance)`.
  - `_decode_actions(actions, frame_st_id, k, plans=1)`: batch `plans * k`, plan-major.
  - `select_plan(batch, plans, k, rank_fn=None, random_plan=False, decode_all=False)` → dict with `s (1, 3072)`, `z (K, 16, 30)`, `a_base (K, 16, 30)` for the executed plan, `plan_pools (M, 3072)`, `plan_index`, `plan_pool (1, 3072)`, `plan_a_base_all (M, K, 16, 30)` only when `decode_all`, `video_noise (M, ...)`, `latents (1, ...)` of the executed plan, `first_chunk`; leaves the winner's 2-row history and the chunk attributes `decode_candidates` sets. With `plans == 1` it returns `decode_candidates` plus zero pools and index 0, no extra pass.
  - `predict_action_chunk` uses `select_plan` with `rank_fn=self.residual_model.rank_plans` when `getattr(self, "eval_plans", 1) > 1`; attribute `self.eval_plans = 1` in `__init__`.
  - `decode_plans` keeps its looped reference role; its pool key is renamed `last_pass_pools`.

- [ ] **Step 1: Make the mocked transformer row-wise** (`_mock_policy` in `tests/test_lingbot_rl.py`)

The mock reads the cache as one global mean and writes one global scalar to all rows, so batched and looped runs cannot agree. Replace the two lines:

```python
            bias = self.blocks[0].attn1.attn_caches["pos"]["k"].float().reshape(x.shape[0], -1).mean(dim=1)[:, None, None]
```

and

```python
                    cache["k"] = cache["k"] + hidden.float().mean(dim=(1, 2)).reshape(-1, 1, 1, 1)
```

Every call site keeps cache rows equal to payload rows (2 for CFG passes, `k` or `M·k` after `expand_conditional_kv`, `2M` after `expand_kv_rows`). Run `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py -k "bit_for_bit or decode_plans or decode_candidates"`; all three existing tests must still pass (the reference loop runs the same ops at the same batch sizes).

- [ ] **Step 2: Write the failing tests**

```python
def test_video_pass_batched_single_plan_equals_video_pass_bit_for_bit():
    torch.manual_seed(2)
    noise = torch.randn(1, 48, 4, 2, 4)
    single, batched = _mock_policy(), _mock_policy()
    timesteps, exec_step = single._video_timesteps()
    batched._video_timesteps()
    x_single, x_batched = noise.clone(), noise.clone()
    for i, t in enumerate(timesteps):
        last = i == len(timesteps) - 1
        advance = not last or exec_step != -1
        x_single = single._video_pass(x_single, t, None, 4, write=last, advance=advance)
        x_batched = batched._video_pass_batched(x_batched, t, None, 4, write=last, advance=advance)
    assert torch.equal(x_single, x_batched)
    for a, b in zip(single._snapshot_kv(), batched._snapshot_kv()):
        assert torch.equal(a["k"], b["k"]) and torch.equal(a["v"], b["v"])


def test_select_plan_batched_matches_looped_decode_plans_and_keeps_winner_rows():
    pool = lambda latent: latent.float().mean(dim=(1, 2, 3, 4)).reshape(-1, 1).expand(-1, STATE_DIM)
    begin = lambda batch: (False, None, 4, torch.zeros(1, 48, 1, 2, 4))
    looped = _mock_policy()
    looped._begin_chunk, looped._pool_from_latent = begin, pool
    torch.manual_seed(11)
    ref = looped.decode_plans({"task": ["t"]}, plans=2, exec_step=-1, k=3)
    policy = _mock_policy()
    policy._begin_chunk, policy._pool_from_latent = begin, pool
    torch.manual_seed(11)
    out = policy.select_plan({"task": ["t"]}, plans=2, k=3, rank_fn=lambda s, pools: 1, decode_all=True)
    assert out["plan_index"] == 1 and torch.equal(out["plan_pool"], out["plan_pools"][1:2])
    torch.testing.assert_close(out["plan_a_base_all"], ref["a_base"], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(out["latents"], ref["latents"][1:2], rtol=1e-5, atol=1e-6)
    cache = policy.transformer.blocks[0].attn1.attn_caches["pos"]["k"]
    assert cache.shape[0] == 2
    winner = _mock_policy()
    winner._begin_chunk, winner._pool_from_latent = begin, pool
    winner.decode_candidates({"task": ["t"]}, k=3, video_noise=ref["video_noise"][1:2], action_noise=mlp_to_model(ref["z"]))
    torch.testing.assert_close(cache, winner.transformer.blocks[0].attn1.attn_caches["pos"]["k"], rtol=1e-5, atol=1e-6)
    out = policy.select_plan({"task": ["t"]}, plans=2, k=3, rank_fn=lambda s, pools: 0)
    assert "plan_a_base_all" not in out and out["a_base"].shape == (3, HORIZON, ACTION_DIM)


def test_predict_action_chunk_ranks_plans_with_the_plan_critic():
    from script.lingbot_rl_policy import ResidualLingBotPolicy
    policy = ResidualLingBotPolicy.__new__(ResidualLingBotPolicy)
    policy.eval_candidates, policy.eval_plans = 2, 3
    policy.residual_model = DiceResidualModel(device="cpu", plan_candidates=3)
    captured = {}

    def fake_select(batch, plans, k, rank_fn=None, random_plan=False, decode_all=False):
        captured.update(plans=plans, k=k, rank=rank_fn, decode_all=decode_all)
        return {"s": torch.zeros(1, STATE_DIM), "z": torch.zeros(2, HORIZON, ACTION_DIM),
                "a_base": torch.zeros(2, HORIZON, ACTION_DIM), "first_chunk": False, "latents": torch.zeros(1),
                "plan_pool": torch.zeros(1, STATE_DIM), "plan_index": 0}

    policy.select_plan = fake_select
    policy._apply_residual_choice = lambda dec, index: torch.zeros(1, 16, USED_DOF)
    policy.predict_action_chunk({"task": ["t"]})
    assert captured == {"plans": 3, "k": 2, "rank": policy.residual_model.rank_plans, "decode_all": False}
```

Rename the key in `test_decode_plans_shapes_and_restores_policy_state` from `plan_pools` to `last_pass_pools`.

- [ ] **Step 3: Run them**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py -k "video_pass_batched or select_plan or ranks_plans"`
Expected: FAIL with `AttributeError`.

- [ ] **Step 4: Cache helpers**

```python
def expand_conditional_kv(transformer, k, plans=1):
    for block in transformer.blocks:
        cache = block.attn1.attn_caches.get("pos") if block.attn1.attn_caches else None
        if cache is None or "k" not in cache or cache["k"].shape[0] < plans:
            raise RuntimeError("action candidate batching failed")
        for key in ("k", "v"):
            cache[key] = cache[key][:plans].repeat_interleave(k, dim=0)


def expand_kv_rows(transformer, reps):
    for block in transformer.blocks:
        cache = block.attn1.attn_caches.get("pos") if block.attn1.attn_caches else None
        if cache is None or "k" not in cache:
            raise RuntimeError("plan batching failed")
        for key in ("k", "v"):
            cache[key] = cache[key].repeat_interleave(reps, dim=0)


def select_kv_rows(transformer, rows):
    index = list(rows)
    for block in transformer.blocks:
        cache = block.attn1.attn_caches.get("pos") if block.attn1.attn_caches else None
        if cache is None or "k" not in cache:
            raise RuntimeError("plan batching failed")
        for key in ("k", "v"):
            cache[key] = cache[key][index]
```

- [ ] **Step 5: Batch-capable `_pool_from_latent`**

Replace `self._critic_cache_ready = False` with `self._critic_cache_batches = {}`. `_ensure_critic_cache(self, batch_size=1)` picks `name = "critic" if batch_size == 1 else "critic_m"`, recreates that cache with `batch_size=batch_size` only when `self._critic_cache_batches.get(name) != batch_size`, records it, and returns `name`. In `_pool_from_latent`: `b = latent.shape[0]`; `name = self._ensure_critic_cache(b)`; after building the payload set `payload["text_emb"] = self._prompt_embeds.to(self.dtype).expand(b, -1, -1).clone()`, `payload["grid_id"] = payload["grid_id"][None].repeat(b, 1, 1)`, `payload["timesteps"] = payload["timesteps"][None].repeat(b, 1)`; call the transformer with `cache_name=name`; pool with `pool_critic_state(video_tokens, text_tokens.expand(b, -1, -1))`. The `b = 1` payload must be shaped exactly as today's.

- [ ] **Step 6: Batched video pass and plan-aware action decode**

```python
    def _video_pass_batched(self, latents, t, latent_cond, frame_st_id, write, advance):
        cfg = self.config
        latent_h, latent_w = self._latent_hw
        m = latents.shape[0]
        cond = None if latent_cond is None else latent_cond.expand(m, -1, -1, -1, -1)
        payload = self._prepare_latent_input(latents, None, t, t, cond, None, frame_st_id=frame_st_id)["latent_res_lst"]
        payload["noisy_latents"] = payload["noisy_latents"].repeat(2, 1, 1, 1, 1)
        payload["text_emb"] = torch.cat([
            self._prompt_embeds.to(self.dtype).expand(m, -1, -1), self._negative_prompt_embeds.to(self.dtype).expand(m, -1, -1)]).clone()
        payload["grid_id"] = payload["grid_id"][None].repeat(2 * m, 1, 1) if payload["grid_id"].ndim == 2 else payload["grid_id"].repeat(2, 1, 1)
        payload["timesteps"] = payload["timesteps"][None].repeat(2 * m, 1) if payload["timesteps"].ndim == 1 else payload["timesteps"].repeat(2, 1)
        pred = self.transformer(payload, update_cache=1 if write else 0, cache_name="pos", action_mode=False)
        if advance:
            pred = data_seq_to_patch(cfg.patch_size, pred, cfg.frame_chunk_size, latent_h, latent_w, batch_size=2 * m)
            pred = pred[m:] + cfg.guidance_scale * (pred[:m] - pred[m:]) if cfg.guidance_scale > 1 else pred[:m]
            latents = self._scheduler.step(pred, t, latents, return_dict=False)
        if frame_st_id == 0 and latent_cond is not None:
            latents[:, :, 0:1] = latent_cond
        return latents
```

Check `_prepare_latent_input` in `modeling_lingbot_va.py` for the `grid_id` / `timesteps` shapes it returns at batch `m` and keep whichever branch matches; the bit-for-bit test at `m = 1` is the check against `_repeat_input_for_cfg`'s layout.

`_decode_actions(self, actions, frame_st_id, k, plans=1)`: the `k == 1 and plans == 1` branch unchanged; the batched branch calls `expand_conditional_kv(self.transformer, k, plans)` and sizes `action_cond`, `text_emb`, `grid_id` and `timesteps` with `rows = plans * k`.

- [ ] **Step 7: `select_plan` and `predict_action_chunk`**

```python
    @torch.no_grad()
    def select_plan(self, batch, plans, k, rank_fn=None, random_plan=False, decode_all=False):
        if plans == 1:
            decoded = self.decode_candidates(batch, k=k)
            decoded["plan_pools"] = torch.zeros(1, STATE_DIM, device=decoded["s"].device)
            decoded["plan_pool"] = decoded["plan_pools"]
            decoded["plan_index"] = 0
            if decode_all:
                decoded["plan_a_base_all"] = decoded["a_base"][None]
            return decoded
        self.eval()
        self._ensure_frozen_modules()
        self._maybe_init_prompt(batch)
        cfg = self.config
        device = cfg.device
        timesteps, exec_step = self._video_timesteps()
        self._action_scheduler.set_timesteps(cfg.action_num_inference_steps)
        first, init_latent, frame_st_id, critic_latent = self._begin_chunk(batch)
        state = self._pool_from_latent(critic_latent)
        latent_h, latent_w = self._latent_hw
        video_noise = torch.randn(plans, 48, cfg.frame_chunk_size, latent_h, latent_w, device=device, dtype=self.dtype)
        action_noise = torch.randn(k, cfg.action_dim, cfg.frame_chunk_size, cfg.action_per_frame, 1, device=device, dtype=self.dtype)
        latent_cond = init_latent[:, :, 0:1].to(self.dtype) if frame_st_id == 0 and init_latent is not None else None
        expand_kv_rows(self.transformer, plans)
        latents = video_noise.clone()
        for i, t in enumerate(timesteps):
            last = i == len(timesteps) - 1
            latents = self._video_pass_batched(
                latents, t, latent_cond, frame_st_id, write=last, advance=not last or exec_step != -1)
        pools = self._pool_from_latent(latents).float()
        index = int(np.random.randint(plans)) if random_plan else (0 if rank_fn is None else int(rank_fn(state.float(), pools)))
        if not 0 <= index < plans:
            raise ValueError("rank_fn must return a plan index")
        out = {"s": state.float(), "z": model_to_mlp(action_noise).float(), "plan_pools": pools,
               "plan_pool": pools[index:index + 1], "plan_index": index, "video_noise": video_noise,
               "latents": latents[index:index + 1], "first_chunk": first}
        if decode_all:
            decoded = model_to_mlp(self._decode_actions(action_noise.repeat(plans, 1, 1, 1, 1), frame_st_id, k, plans)).float()
            out["plan_a_base_all"] = decoded.reshape(plans, k, HORIZON, ACTION_DIM)
            out["a_base"] = out["plan_a_base_all"][index]
            select_kv_rows(self.transformer, [index, plans + index])
        else:
            select_kv_rows(self.transformer, [index, plans + index])
            out["a_base"] = model_to_mlp(self._decode_actions(action_noise.clone(), frame_st_id, k)).float()
        if first:
            self._first_chunk = False
        self._exec_step = 0
        self._started = True
        return out
```

`_decode_actions` snapshots and restores the cache it is given, so after `decode_all` the cache still has `2M` rows until `select_kv_rows` keeps the winner. Add `self.eval_plans = 1` in `__init__`. In `predict_action_chunk`, when `self.residual_model is not None and getattr(self, "eval_plans", 1) > 1`: `decoded = self.select_plan(batch, self.eval_plans, k, rank_fn=self.residual_model.rank_plans)` and rank the K candidates with `self.residual_model.critic(self.residual_model.critic_state(state, decoded["plan_pool"].expand(k, -1)), executed)`; otherwise today's path. In `decode_plans` rename the returned key `plan_pools` to `last_pass_pools`.

- [ ] **Step 8: Verify and commit**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py && .cache/eval-venv/bin/python -m compileall -q script/`

```bash
git add script/lingbot_rl_policy.py tests/test_lingbot_rl.py
git commit -m "Sample M video plans in one batched pass, pool them, and decode candidates per plan in one batch."
```

---

### Task 5: Expert plan pools

**Files:**
- Modify: `script/lingbot_rl_data.py`, `tests/test_lingbot_rl.py`

**Interfaces:**
- Consumes: `policy.extract_critic_state_from_latent(latent (1, 48, 4, h, w), batch)`.
- Produces: expert rows with `plan_pool (3072)` = pool of the published latents of the chunk's plan frames; `EXPERT_RECIPE = "dice-rl-expert-v2"`.

The online first chunk pools `[cur, +1, +2, +3]` (frame 0 is overwritten with the current frame on every pass) and later chunks pool `[idx + 1 .. idx + 4]`, so expert plan frames are `idx .. idx + 3` for the first chunk and `idx + 1 .. idx + 4` afterwards, clamped to the last frame.

- [ ] **Step 1: Write the failing test**

```python
def test_expert_rows_pool_the_true_plan_frames(tmp_path):
    from script.lingbot_rl_data import featurize_experts
    calls = []

    class StubPolicy:
        config = type("C", (), {"device": "cpu"})()

        def reset(self):
            return None

        def extract_critic_state_from_latent(self, latent, batch):
            calls.append(tuple(latent.shape))
            return torch.full((1, STATE_DIM), float(latent[0, 0, 0, 0, 0]))

    latents = torch.arange(10, dtype=torch.float32).reshape(1, 1, 10, 1, 1).expand(1, 48, 10, 2, 2).clone()
    episode = {"actions": np.zeros((36, USED_DOF), np.float32), "task": "t", "task_id": 9, "latents": latents, "success": True}
    rows = featurize_experts(StubPolicy(), [episode], None, _eval_normalization(), tmp_path / "cache.pt")
    assert [row["s"][0] for row in rows] == [0.0, 3.0, 7.0]
    assert [row["plan_pool"][0] for row in rows] == [0.0, 4.0, 8.0]
    assert (1, 48, 4, 2, 2) in calls
```

- [ ] **Step 2: Run it**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py -k true_plan_frames`
Expected: FAIL (`KeyError: 'plan_pool'`).

- [ ] **Step 3: Implement** in `script/lingbot_rl_data.py`

`EXPERT_RECIPE = "dice-rl-expert-v2"`. In `_episode_rows`, inside the published-latents branch after `state`:

```python
            start = index if first else index + 1
            plan_frames = torch.clamp(torch.arange(start, start + 4), max=max(frame_count - 1, 0))
            plan_pool = _tensor_state(policy.extract_critic_state_from_latent(
                latents[:, :, plan_frames], {"task": [episode["task"]]}))
```

In the RGB-frames branch `plan_pool = state.copy()`. Add `"plan_pool": plan_pool` to the row. The recipe string is already in `expert_fingerprint`, so existing caches rebuild (about 4.5k extra pooling passes over the 300 manifest episodes, 15 to 25 min on a GB10).

- [ ] **Step 4: Verify and commit**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py`

```bash
git add script/lingbot_rl_data.py tests/test_lingbot_rl.py
git commit -m "Pool the demo's true plan frames as the expert plan pool."
```

---

### Task 6: Training loop, metrics and evaluation with plans

**Files:**
- Modify: `script/lingbot_rl_train.py`, `tests/test_lingbot_rl.py` (`_StubPolicy` and two tests)

**Interfaces:**
- Consumes: everything above.
- Produces: `train()` and `evaluate()` honouring `config.plan_candidates`; per-update log keys `plan_agreement`, `plan_regret`, `plan_value_spread`, `plan_random_rate`, `plan_critic_loss`, `v_plan_mean`, `plan_target_mean`, `seconds_per_env_step`, `peak_gpu_memory_gb`; `replay_plan_value_spread` in the 5k sharpening block; `settings.json` gains `"runtime": {"cuda", "flash_sdp_enabled", "machine"}`; startup raises on an aarch64 host whose `torch.version.cuda` does not start with `"13"`.

- [ ] **Step 1: Extend `_StubPolicy` and write the failing tests**

Give `_StubPolicy` a class attribute `plan_calls = []` and:

```python
    def select_plan(self, batch, plans, k, rank_fn=None, random_plan=False, decode_all=False):
        type(self).plan_calls.append((plans, k, random_plan, decode_all))
        base = self.decode_candidates(batch, k)
        pools = torch.stack([torch.full((STATE_DIM,), float(m)) for m in range(plans)])
        all_base = torch.stack([base["a_base"] + 10.0 * m for m in range(plans)])
        index = 1 if random_plan else int(rank_fn(base["s"], pools))
        return {**base, "plan_pools": pools, "plan_pool": pools[index:index + 1], "plan_index": index,
                "plan_a_base_all": all_base, "a_base": all_base[index]}
```

Tests:

```python
def test_train_with_plans_stores_plan_rows_trains_the_plan_critic_and_logs_metrics(tmp_path, monkeypatch):
    _StubPolicy.plan_calls = []
    out = _smoke_train(tmp_path / "ranked", monkeypatch, 42, model_factory=lambda **kw: DiceResidualModel(**kw),
                       config=RLConfig(plan_candidates=3, plan_random_prob=0.0, selection_warmup_steps=0, residual_input="base"))
    rows = [row for row in out["rows"] if float(row["is_expert"]) == 0.0]
    assert rows[0]["plan_pools_all"].shape == (3, STATE_DIM)
    assert rows[0]["plan_a_base_all"].shape == (3, K_CANDIDATES, HORIZON, ACTION_DIM)
    assert float(rows[0]["plan_pool"][0]) == float(rows[0]["plan_index"])
    assert all(call[2] is False and call[3] is True for call in _StubPolicy.plan_calls)
    logged = [entry for entry in out["logs"] if "plan_critic_loss" in entry]
    assert logged and {"plan_agreement", "plan_regret", "plan_value_spread", "plan_random_rate", "seconds_per_env_step"} <= set(logged[0])
    assert {"plan_critic", "plan_opt", "plan_lr"} <= set(out["resume"])
    _StubPolicy.plan_calls = []
    _smoke_train(tmp_path / "warm", monkeypatch, 42, model_factory=lambda **kw: DiceResidualModel(**kw),
                 config=RLConfig(plan_candidates=3, plan_random_prob=0.0, selection_warmup_steps=10_000))
    assert all(call[2] is True for call in _StubPolicy.plan_calls)


def test_single_plan_training_never_calls_select_plan(tmp_path, monkeypatch):
    _StubPolicy.plan_calls = []
    out = _smoke_train(tmp_path, monkeypatch, 42, model_factory=_sum_critic_model, config=RLConfig(selection_warmup_steps=0))
    assert _StubPolicy.plan_calls == [] and not any("plan_agreement" in entry for entry in out["logs"])
```

- [ ] **Step 2: Run them**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py -k "train_with_plans or single_plan_training"`
Expected: FAIL (`KeyError: 'plan_pools_all'`).

- [ ] **Step 3: Startup, collection and metrics** in `train()`

After `device = _device()`:

```python
    runtime = {"cuda": torch.version.cuda, "machine": platform.machine(),
               "flash_sdp_enabled": bool(torch.backends.cuda.flash_sdp_enabled()) if device == "cuda" else None}
    if device == "cuda" and runtime["machine"] == "aarch64" and not str(runtime["cuda"]).startswith("13"):
        raise RuntimeError("GB10 hosts need a CUDA 13 torch build (DICE_TORCH_INDEX)")
```

and write `"runtime": runtime` into `settings.json`. Build `model = DiceResidualModel(device=device, residual_input=config.residual_input, epsilon=config.epsilon, plan_candidates=config.plan_candidates)`, `buffer = ChunkReplay(plans=config.plan_candidates)`, `policy.eval_plans = config.plan_candidates`. Keep `chunk_seconds = 0.0` and `chunk_actions = 0` accumulators reset at every update log.

Replace the `decode_candidates` call in the episode loop:

```python
                started = time.perf_counter()
                plans = config.plan_candidates
                if plans > 1:
                    random_plan = env_steps < config.selection_warmup_steps or np.random.random() < config.plan_random_prob
                    decoded = policy.select_plan(batch, plans, config.k_candidates, rank_fn=model.rank_plans,
                                                 random_plan=random_plan, decode_all=True)
                else:
                    decoded = policy.decode_candidates(batch, k=config.k_candidates)
```

Keep `state`, `noise`, `a_base`; add `pool = decoded["plan_pool"].to(device=device, dtype=torch.float32) if plans > 1 else None`; rank candidates with `model.critic(model.critic_state(state_k, None if pool is None else pool.expand(k, -1)), executed)`. When `plans > 1`, under `no_grad` after `star`:

```python
                    pools_all = decoded["plan_pools"].to(device=device, dtype=torch.float32)
                    all_base = decoded["plan_a_base_all"].to(device=device, dtype=torch.float32)
                    flat_state = state.expand(plans * k, -1)
                    flat_base = all_base.reshape(plans * k, HORIZON, ACTION_DIM)
                    q_all = model.critic(model.critic_state(flat_state, pools_all.repeat_interleave(k, dim=0)),
                                         apply_residual(flat_base, model.actor(flat_state, noise.repeat(plans, 1, 1), flat_base))).reshape(plans, k).max(dim=1).values
                    v_all = model.plan_critic(state.expand(plans, -1), pools_all).reshape(-1)
                    plan_info = {"plan_agreement": float(int(v_all.argmax()) == int(q_all.argmax())),
                                 "plan_regret": float(q_all.max() - q_all[decoded["plan_index"]]),
                                 "plan_value_spread": float(v_all.max() - v_all.min()),
                                 "plan_random_rate": float(random_plan)}
```

After the env loop: `chunk_seconds += time.perf_counter() - started; chunk_actions += executed_n`. Add to the `buffer.add_online` row when `plans > 1`: `"plan_pool": _host_array(pool)[0]`, `"plan_pools_all": _host_array(pools_all)`, `"plan_index": int(decoded["plan_index"])`, `"plan_a_base_all": _host_array(all_base)`. In the update log merge `plan_info` (when `plans > 1`), `"seconds_per_env_step": chunk_seconds / max(chunk_actions, 1)`, and `"peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0`, then reset the accumulators.

- [ ] **Step 4: Updates, sharpening, checkpoints, evaluation**

`_update_from_buffer`: `critic_state = model.critic_state(sample["s"], sample.get("plan_pool"))`, `next_critic_state = model.critic_state(sample["s_next"], sample.get("plan_pool_next"))`; pass them to `n_step_target`, `update_critic` and `update_actor` as Task 2 defines; after `update_critic`, when `model.plan_critic is not None`: `info.update(model.update_plan_critic(sample["s"], sample["plan_pools_all"], sample["z_all"], sample["plan_a_base_all"]))`.

`replay_sharpening`: critic calls use `model.critic_state(state_k, pool_k)` with `pool_k` from `sample.get("plan_pool")` expanded like `state_k` (`None` when absent); when `model.plan_critic is not None` add `"replay_plan_value_spread"`: mean over online rows of max − min of `model.plan_critic(state_m, pools_all_m)` over the row's `plan_pools_all`. `sharpening_metrics` and `_maybe_sharpen`: `_maybe_sharpen` uses `policy.select_plan(batch, policy.eval_plans, 8, rank_fn=model.rank_plans)` when `policy.eval_plans > 1` and passes `model.critic_state(state, decoded["plan_pool"].expand(...))` to `sharpening_metrics` as the critic state.

`maybe_checkpoint`: leave `policy.eval_plans = config.plan_candidates` (the step-0 train-eval ranks with an untrained plan critic; it is a sanity check, noted in the results).

`evaluate()`: build the model with `plan_candidates=config.plan_candidates`, set `policy.eval_plans = config.plan_candidates`, add `"eval_plans": policy.eval_plans` to the shard settings.

- [ ] **Step 5: Verify and commit**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py tests/test_lingbot_eval.py && .cache/eval-venv/bin/python -m script.lingbot_rl_config`

```bash
git add script/lingbot_rl_train.py tests/test_lingbot_rl.py
git commit -m "Collect with M sampled plans, train the plan critic, and evaluate with plan selection."
```

---

### Task 7: GB10 smoke and launch (controller)

- [ ] Step 1: sync main to both boxes; on Jarvis run `bash brev/train.sh smoke-plan 9 --residual-input base --epsilon -0.3 --video-steps 5 --action-steps 10 --plan-candidates 4 --env-steps 320000 --max-env-steps 300 --no-tmux`; confirm `settings.json` runtime shows CUDA 13, `plan_critic_loss` falls, `plan_agreement` and `plan_value_spread` are present and the spread is not near zero, `seconds_per_env_step` and `peak_gpu_memory_gb` (expect about 25 GB of KV plus weights); repeat with `--plan-candidates 1` for the throughput comparison.
- [ ] Step 2: expert featurisation rebuilds under recipe v2 (15 to 25 min); confirm `expert_features.pt` rows carry `plan_pool`.
- [ ] Step 3: launch `dice-t9-plan-m4` on Jarvis and `dice-t0-plan-m4` on Watson: `bash brev/train.sh dice-tX-plan-m4 X --residual-input base --epsilon -0.3 --env-steps 320000 --video-steps 5 --action-steps 10 --plan-candidates 4`. Resume files are about 3.5 GB by 320k and are rewritten every 10 episodes.
- [ ] Step 4: record the runs, the SFT re-bench numbers and the smoke throughput in `docs/heldout-results.md`.

## Self-review notes

- Spec coverage: §4 → Tasks 4 and 6 (runtime record, throughput metrics); §5/§6 → Tasks 4 and 6; §7 → Task 2; §8 → Task 5; §9 → Task 3; §10 → Task 1; §12 runs → Task 7.
- The padded t = 0 write pass never advances (`advance=not last or exec_step != -1`), matching `_video_pass`.
- The plan pool is the history-free critic pool of the denoised chunk for online and expert rows; `decode_plans`' `last_pass_pools` is a different quantity and is only a test reference.
- `critic_state(state, pool)` is the only place that widens the critic input; `rank_plans(state, pools)` matches `rank_fn(state, pools)`; `plan_a_base_all` is `(M, K, 16, 30)` in the policy output, the buffer and `update_plan_critic`.
- Tests: nine new test functions in total, one per question (fingerprint, plan-critic target and isolation, buffer row contract including legacy load, batched pass equals single pass, batched equals looped with winner rows, eval ranking plumbing, expert frame alignment, training with and without plans).
