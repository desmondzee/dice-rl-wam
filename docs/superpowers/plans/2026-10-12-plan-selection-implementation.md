# Plan-level selection implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the residual recipe with M = 4 sampled video plans per chunk, ranked by a regression-trained plan critic, at the 5 / 10 sampler, on tasks 9 and 0, one GB10 each.

**Architecture:** The policy denoises M plans in one batch-2M pass sequence against a row-expanded KV cache, pools every denoised chunk through the existing history-free critic pool, decodes K candidates per plan in one batch, and keeps only the winner's cache rows. The model gains a plan critic V_plan(s, p) trained by regression onto max-over-candidates min-ensemble Q, and Q takes the plan pool as an extra input. Replay rows carry the M pools and M × K base actions; expert rows carry the pool of the demo's true next chunk.

**Tech Stack:** Torch 2.11 (cu130 on the GB10s), the pinned LeRobot LingBot-VA port, numpy replay, pytest with the mocked transformer in `tests/test_lingbot_rl.py`.

**Spec:** docs/superpowers/specs/2026-10-09-plan-selection-design.md (commit f203882).

## Global Constraints

- Zero comments and docstrings in `script/lingbot_rl_*.py`; lean code; tests must not call Modal or `.remote()`.
- Do not edit `.cache/lerobot`; do not change `script/lingbot_eval.py` protocol fields.
- `plan_candidates = 1` must reproduce today's recipe bit for bit (policy path, critic input width, buffer sampling, fingerprint apart from the new keys).
- Sampler values stay in config (`video_steps`, `action_steps`, `video_exec_step`); the runs use 5 / 10 / −1.
- Secrets never printed; no new experiments launched by implementers (Task 7's smoke is run by the controller on a GB10).
- Verify with `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py tests/test_lingbot_eval.py`, `.cache/eval-venv/bin/python -m script.lingbot_rl_config`, `bash -n brev/setup.sh brev/train.sh brev/sync.sh brev/pull.sh brev/eval.sh`.

## File map

| file | responsibility after this plan |
|---|---|
| `script/lingbot_rl_config.py` | `plan_candidates`, `plan_random_prob`; validation; fingerprint keys |
| `script/lingbot_rl_model.py` | `CriticEnsemble(state_dim)`, `PlanCritic`, `critic_state`, `rank_plans`, `update_plan_critic`, plan-aware targets and checkpoints |
| `script/lingbot_rl_buffer.py` | plan keys with expert defaults; `plan_pool_next` at finalisation |
| `script/lingbot_rl_policy.py` | batched M-plan sampling (`select_plan`), batched M × K decode, batch-capable `_pool_from_latent`, plan-aware `predict_action_chunk` |
| `script/lingbot_rl_data.py` | expert plan pools from published latents; expert recipe v2 |
| `script/lingbot_rl_train.py` | M-plan collection, random-plan schedule, V_plan update, plan metrics, plan-aware eval and sharpening |
| `brev/train.sh`, `brev/eval.sh` | `--plan-candidates` / `--plans` |

---

### Task 1: Config fields and fingerprints

**Files:**
- Modify: `script/lingbot_rl_config.py`
- Modify: `brev/train.sh`, `brev/eval.sh`
- Test: `tests/test_lingbot_rl.py`

**Interfaces:**
- Produces: `RLConfig.plan_candidates: int = 1`, `RLConfig.plan_random_prob: float = 0.1`; protocol keys `plan_candidates`, `plan_random_prob`, `plan_critic`, `plan_pool`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_lingbot_rl.py`)

```python
def test_config_plan_fields_enter_the_fingerprint():
    base = RLConfig().protocol()
    assert base["plan_candidates"] == 1 and base["plan_random_prob"] == 0.1
    assert base["plan_critic"] == "none" and base["plan_pool"] == "none"
    plan = RLConfig(plan_candidates=4).protocol()
    assert plan["plan_candidates"] == 4
    assert plan["plan_critic"] == "regression_max_k_min_q" and plan["plan_pool"] == "critic_pool_of_denoised_chunk"
    assert plan != base
    assert config_from_dict({"plan_candidates": 4, "plan_random_prob": 0.25}).plan_random_prob == 0.25


@pytest.mark.parametrize("changes", [
    {"plan_candidates": 0}, {"plan_candidates": 9}, {"plan_candidates": 2.0},
    {"plan_random_prob": -0.1}, {"plan_random_prob": 1.5},
])
def test_config_rejects_plan_field_drift(changes):
    with pytest.raises(ValueError):
        RLConfig(**changes).validate()
```

- [ ] **Step 2: Run them**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py -k plan_field`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'plan_candidates'`.

- [ ] **Step 3: Implement** in `script/lingbot_rl_config.py`

Add the fields after `epsilon`:

```python
    plan_candidates: int = 1
    plan_random_prob: float = 0.1
```

Add to `validate()` before `return self`:

```python
        if type(self.plan_candidates) is not int or not 1 <= self.plan_candidates <= 8:
            raise ValueError("plan_candidates must be an integer in [1, 8]")
        if not 0.0 <= float(self.plan_random_prob) <= 1.0:
            raise ValueError("plan_random_prob must lie in [0, 1]")
```

Add to `protocol()` after `"epsilon": self.epsilon,`:

```python
            "plan_candidates": self.plan_candidates,
            "plan_random_prob": self.plan_random_prob,
            "plan_critic": "regression_max_k_min_q" if self.plan_candidates > 1 else "none",
            "plan_pool": "critic_pool_of_denoised_chunk" if self.plan_candidates > 1 else "none",
```

- [ ] **Step 4: Script flags**

`brev/train.sh`: add `PLAN_CANDIDATES=1` beside `ACTION_STEPS=50`, the case `--plan-candidates) PLAN_CANDIDATES="$2"; shift 2 ;;`, `[--plan-candidates M]` in both usage strings, and extend the config line so the JSON gains `"plan_candidates": int(sys.argv[9])` with `"$PLAN_CANDIDATES"` appended to the argument list.

`brev/eval.sh`: add `PLANS=1`, the case `--plans) PLANS="$2"; shift 2 ;;`, `[--plans M]` in `USAGE`; for the run branch add `"plan_candidates": int(sys.argv[6])` to the JSON with `"$PLANS"` appended, and name the policy `POLICY="$RUN-step$STEP6-k$K-m$PLANS$SUFFIX"` when `PLANS` is not 1 (unchanged name when it is 1).

- [ ] **Step 5: Verify and commit**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py && .cache/eval-venv/bin/python -m script.lingbot_rl_config && bash -n brev/train.sh brev/eval.sh`
Expected: all pass; the config JSON prints `"plan_candidates": 1`.

```bash
git add script/lingbot_rl_config.py brev/train.sh brev/eval.sh tests/test_lingbot_rl.py
git commit -m "Add plan_candidates and plan_random_prob to the RL recipe and the Brev scripts."
```

---

### Task 2: Plan critic and plan-aware action critic

**Files:**
- Modify: `script/lingbot_rl_model.py`
- Test: `tests/test_lingbot_rl.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `CriticEnsemble(state_dim=STATE_DIM)`; forward unchanged `(state, action, return_all=False)`.
  - `PlanCritic()` with forward `(state, pool, return_all=False)`, inputs `(B, 3072)` and `(B, 3072)`.
  - `DiceResidualModel(device, residual_input, epsilon, plan_candidates=1)` with attributes `plans`, `plan_critic`, `plan_opt`, `plan_lr` (all `None` when `plans == 1`).
  - `model.critic_state(state, pool)` → `state` when `plans == 1`, else `cat([state, pool], -1)`.
  - `model.rank_plans(state, pools)` → `int` argmax over `(M, 3072)` pools of min-ensemble V_plan; `state` is `(1, 3072)`.
  - `model.n_step_target(reward, done, next_critic_state, z_next_all, a_base_next_all, n_steps)`: the third argument is already `critic_state(s_next, plan_pool_next)`.
  - `model.update_critic(critic_state, action, target_q)`.
  - `model.update_actor(state, critic_state, action_stored, z_all, a_base_all, is_expert, mc_return, filter_active)`.
  - `model.update_plan_critic(state, pools_all, z_all, plan_a_base_all)` with `pools_all (B, M, 3072)`, `z_all (B, K, 16, 30)`, `plan_a_base_all (B, M, K, 16, 30)`; returns `{"plan_critic_loss", "v_plan_mean", "plan_target_mean"}`.
  - `inference_state_dict()` adds `"plan_critic"` when `plans > 1`; `resume_state_dict()` adds `"plan_critic"`, `"plan_opt"`, `"plan_lr"`.

- [ ] **Step 1: Write the failing tests**

```python
def test_critic_state_concatenates_pool_only_in_plan_mode():
    single = DiceResidualModel(device="cpu")
    plan = DiceResidualModel(device="cpu", plan_candidates=4)
    s, p = torch.randn(3, STATE_DIM), torch.randn(3, STATE_DIM)
    assert torch.equal(single.critic_state(s, p), s)
    assert plan.critic_state(s, p).shape == (3, 2 * STATE_DIM)
    assert single.plan_critic is None and plan.plan_critic is not None
    assert single.critic.heads[0][0].in_features == STATE_DIM + HORIZON * ACTION_DIM
    assert plan.critic.heads[0][0].in_features == 2 * STATE_DIM + HORIZON * ACTION_DIM
    assert set(single.inference_state_dict()) == {"actor", "critic", "target_critic"}
    assert set(plan.inference_state_dict()) == {"actor", "critic", "target_critic", "plan_critic"}
    assert {"plan_opt", "plan_lr"} <= set(plan.resume_state_dict())


def test_plan_critic_regresses_onto_max_candidate_min_q_without_touching_q_or_actor():
    torch.manual_seed(0)
    model = DiceResidualModel(device="cpu", residual_input="base", plan_candidates=2)
    for parameter in model.critic.parameters():
        torch.nn.init.normal_(parameter, std=0.05)
    state = torch.randn(5, STATE_DIM)
    pools = torch.randn(5, 2, STATE_DIM)
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
        q = model.critic(model.critic_state(flat_state, flat_pool), action).reshape(5, 2, 4)
        expected_target = q.max(dim=2).values
    losses = [model.update_plan_critic(state, pools, z_all, a_base)["plan_critic_loss"] for _ in range(30)]
    assert losses[-1] < losses[0]
    info = model.update_plan_critic(state, pools, z_all, a_base)
    assert abs(info["plan_target_mean"] - float(expected_target.mean())) < 1e-5
    assert all(torch.equal(a, b) for a, b in zip(critic_before, model.critic.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(actor_before, model.actor.parameters()))


def test_rank_plans_returns_argmax_of_min_ensemble_plan_value():
    torch.manual_seed(1)
    model = DiceResidualModel(device="cpu", plan_candidates=3)
    for parameter in model.plan_critic.parameters():
        torch.nn.init.normal_(parameter, std=0.05)
    state, pools = torch.randn(1, STATE_DIM), torch.randn(3, STATE_DIM)
    values = model.plan_critic(state.expand(3, -1), pools)
    assert model.rank_plans(state, pools) == int(values.reshape(-1).argmax())
    assert values.shape == (3, 1)
```

- [ ] **Step 2: Run them**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py -k "critic_state or plan_critic or rank_plans"`
Expected: FAIL with `TypeError` on `plan_candidates`.

- [ ] **Step 3: Implement** in `script/lingbot_rl_model.py`

Replace `CriticEnsemble` and add `PlanCritic`:

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

In `DiceResidualModel.__init__` take `plan_candidates=1`, set `self.plans = int(plan_candidates)`, build `self.critic = CriticEnsemble(STATE_DIM * (2 if self.plans > 1 else 1)).to(device)` before the target copy, and after the schedulers:

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
            values = self.plan_critic(mlp_float(state).expand(pools.shape[0], -1), pools)
            return int(values.reshape(-1).argmax())
```

`n_step_target`: rename the third parameter to `next_critic_state`, expand it over `k` exactly as `next_state` is today, and pass the expanded tensor to `self.target_critic` while the actor still receives `state_k` built from `next_critic_state[:, :STATE_DIM]`:

```python
    def n_step_target(self, reward, done, next_critic_state, z_next_all, a_base_next_all, n_steps):
        with torch.no_grad():
            next_critic_state = mlp_float(next_critic_state)
            z_next_all = mlp_float(z_next_all)
            a_base_next_all = mlp_float(a_base_next_all)
            batch, k = z_next_all.shape[0], z_next_all.shape[1]
            critic_k = next_critic_state.unsqueeze(1).expand(batch, k, -1).reshape(batch * k, -1)
            state_k = critic_k[:, :STATE_DIM]
            z_flat = z_next_all.reshape(batch * k, HORIZON, ACTION_DIM)
            base_flat = a_base_next_all.reshape(batch * k, HORIZON, ACTION_DIM)
            a_next = apply_residual(base_flat, self.actor(state_k, z_flat, base_flat))
            backup = self.target_critic(critic_k, a_next).reshape(batch, k, 1).mean(dim=1)
            return reward + (GAMMA ** n_steps) * (1.0 - done) * backup
```

`update_critic(self, critic_state, action, target_q)`: body unchanged apart from the parameter name.

`update_actor(self, state, critic_state, action_stored, z_all, a_base_all, is_expert, mc_return, filter_active)`: build `critic_k` from `critic_state` the same way `state_k` is built from `state`, and use `critic_k` in the three `self.critic(...)` calls on `action`, `base_flat` and `critic_state` with `action_stored`; the actor keeps `state_k`.

Add:

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

Checkpoints: in `inference_state_dict` add `"plan_critic": self.plan_critic.state_dict()` when `self.plan_critic is not None`; `load_inference_state_dict` loads it when both the payload and the model have it and raises `ValueError("plan critic weights missing")` when the model has one and the payload does not; `resume_state_dict` adds `plan_opt` and `plan_lr` state dicts when present; `load_resume_state_dict` restores them when present.

- [ ] **Step 4: Update callers and existing tests**

`script/lingbot_rl_train.py` (temporary, finished in Task 6): pass `sample["s"]` as both `state` and `critic_state` in `_update_from_buffer`, and `sample["s_next"]` as `next_critic_state`, so the suite runs. `save_inference` must accept `{"actor", "critic", "target_critic", "plan_critic"}`. Fix `test_actor_critic_one_step_update_on_random_tensors`, `test_actor_q_term_is_normalised_by_online_mean_abs_q`, `test_bc_filter_drops_better_overestimated_online_rows_only`, `test_n_step_target_averages_current_policy_over_candidates` and any other direct caller for the new `update_actor` signature (pass the state twice).

- [ ] **Step 5: Verify and commit**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py`
Expected: all pass.

```bash
git add script/lingbot_rl_model.py script/lingbot_rl_train.py tests/test_lingbot_rl.py
git commit -m "Add the regression-trained plan critic and the plan-pool input to the action critic."
```

---

### Task 3: Replay keys for plans

**Files:**
- Modify: `script/lingbot_rl_buffer.py`
- Test: `tests/test_lingbot_rl.py`

**Interfaces:**
- Produces: row keys `plan_pool (3072)`, `plan_pools_all (M, 3072)`, `plan_index (int)`, `plan_a_base_all (M, K, 16, 30)`, `plan_pool_next (3072)`; `ChunkReplay(capacity, device, plans=1)`; `sample()` returns them as tensors (`plan_index` long).

- [ ] **Step 1: Write the failing tests**

```python
def test_plan_keys_default_to_repeated_single_plan_and_finalize_fills_next_pool():
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
    assert rows[0]["plan_a_base_all"].shape == (3, K_CANDIDATES, HORIZON, ACTION_DIM)
    assert np.array_equal(rows[0]["plan_a_base_all"][1], rows[0]["a_base_all"])
    assert float(rows[0]["plan_pool_next"][0]) == 2.0 and float(rows[1]["plan_pool_next"][0]) == 2.0
    assert int(rows[1]["plan_index"]) == 0
    batch = buffer.sample(2, 0.0)
    assert batch["plan_pools_all"].shape == (2, 3, STATE_DIM) and batch["plan_index"].dtype == torch.long
    assert batch["plan_a_base_all"].shape == (2, 3, K_CANDIDATES, HORIZON, ACTION_DIM)


def test_plan_keys_absent_rows_get_zero_pool_in_single_plan_mode():
    buffer = _online_buffer()
    row = buffer.rows()[0]
    assert row["plan_pool"].shape == (STATE_DIM,) and float(np.abs(row["plan_pool"]).sum()) == 0.0
    assert row["plan_pools_all"].shape == (1, STATE_DIM)
```

- [ ] **Step 2: Run them**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py -k plan_keys`
Expected: FAIL (`TypeError` on `plans=`, then `KeyError: 'plan_pool'`).

- [ ] **Step 3: Implement** in `script/lingbot_rl_buffer.py`

Import `STATE_DIM` from the model module. Extend `FLOAT_KEYS` with `"plan_pool", "plan_pools_all", "plan_a_base_all", "plan_pool_next"` and `KEYS` with those plus `"plan_index"`. `__init__(self, capacity=REPLAY_CAPACITY, device="cpu", plans=1)` stores `self.plans = int(plans)`. In `_store`, after the `a_base_all` defaults:

```python
        stored.setdefault("plan_pool", np.zeros(STATE_DIM, np.float32))
        stored.setdefault("plan_pools_all", np.repeat(_host_float(stored["plan_pool"])[None], self.plans, axis=0))
        stored.setdefault("plan_index", 0)
        stored.setdefault("plan_a_base_all", np.repeat(_host_float(stored["a_base_all"])[None], self.plans, axis=0))
        stored.setdefault("plan_pool_next", np.array(stored["plan_pool"], copy=True))
        if _host_float(stored["plan_pools_all"]).shape[0] != self.plans or \
                _host_float(stored["plan_a_base_all"]).shape[0] != self.plans:
            raise ValueError(f"Rows must carry exactly {self.plans} plans")
        stored["plan_index"] = int(stored["plan_index"])
```

In `finalize_episode`, beside `s_next`: `row["plan_pool_next"] = np.array(episode[nxt]["plan_pool"], copy=True)`. In `sample`, treat `plan_index` like `task_id` (integer tensor, `torch.long`). `state_dict` adds `"plans": self.plans`; `load_state_dict` restores it with `payload.get("plans", 1)`.

- [ ] **Step 4: Verify and commit**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py`
Expected: all pass (the resume round-trip tests still pass because defaults fill the new keys).

```bash
git add script/lingbot_rl_buffer.py tests/test_lingbot_rl.py
git commit -m "Carry plan pools, plan candidates and the next executed pool in replay rows."
```

---

### Task 4: Batched M-plan sampling in the policy

**Files:**
- Modify: `script/lingbot_rl_policy.py`
- Test: `tests/test_lingbot_rl.py`

**Interfaces:**
- Consumes: `model.rank_plans(state, pools)` (Task 2) through `self.residual_model`.
- Produces:
  - `expand_conditional_kv(transformer, k, plans=1)`: each of the first `plans` cache rows repeated `k` times (`repeat_interleave`), so `plans=1` equals today's behaviour.
  - `expand_kv_rows(transformer, reps)`: every cache row repeated `reps` times in place (`[c, u]` → `[c × reps, u × reps]`).
  - `select_kv_rows(transformer, rows)`: keep the listed rows.
  - `_pool_from_latent(latent)` accepting batch `B` and `T` in {1, 4}, returning `(B, 3072)`.
  - `_decode_actions(actions, frame_st_id, k, plans=1)`: `actions` has batch `plans * k`, plan-major; returns the same layout.
  - `select_plan(batch, plans, k, rank_fn=None, random_plan=False, decode_all=False)` → dict with `s (1, 3072)`, `z (K, 16, 30)`, `a_base (K, 16, 30)` for the executed plan, `plan_pools (M, 3072)`, `plan_index`, `plan_pool (1, 3072)`, `plan_a_base_all (M, K, 16, 30)` (only when `decode_all`), `video_noise (M, 48, 4, h, w)`, `latents (1, 48, 4, h, w)` of the executed plan, `first_chunk`. Leaves the policy with the winner's 2-row history and the same chunk attributes `decode_candidates` sets.
  - `predict_action_chunk` uses `select_plan` with `rank_fn=self.residual_model.rank_plans` when `self.eval_plans > 1`, else today's path. New attribute `self.eval_plans = 1`.

- [ ] **Step 1: Write the failing tests**

```python
def test_expand_and_select_kv_rows_keep_cfg_order():
    from script.lingbot_rl_policy import expand_conditional_kv, expand_kv_rows, select_kv_rows
    policy = _mock_policy()
    for block in policy.transformer.blocks:
        block.attn1.attn_caches["pos"] = {"k": torch.tensor([[1.0], [2.0]]), "v": torch.tensor([[10.0], [20.0]])}
    expand_kv_rows(policy.transformer, 3)
    cache = policy.transformer.blocks[0].attn1.attn_caches["pos"]
    assert cache["k"].reshape(-1).tolist() == [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
    select_kv_rows(policy.transformer, [1, 4])
    assert cache["k"].reshape(-1).tolist() == [1.0, 2.0] and cache["v"].reshape(-1).tolist() == [10.0, 20.0]
    for block in policy.transformer.blocks:
        block.attn1.attn_caches["pos"] = {"k": torch.tensor([[1.0], [3.0], [2.0], [4.0]]), "v": torch.tensor([[1.0], [3.0], [2.0], [4.0]])}
    expand_conditional_kv(policy.transformer, 2, plans=2)
    assert cache["k"].reshape(-1).tolist() == [1.0, 1.0, 3.0, 3.0]


def test_select_plan_single_plan_matches_decode_candidates_bit_for_bit():
    torch.manual_seed(3)
    reference = _mock_policy()
    reference._begin_chunk = lambda batch: (False, None, 4, torch.zeros(1, 48, 1, 2, 4))
    reference._pool_from_latent = lambda latent: torch.zeros(latent.shape[0], STATE_DIM)
    torch.manual_seed(5)
    expected = reference.decode_candidates({"task": ["t"]}, k=4)
    policy = _mock_policy()
    policy._begin_chunk = lambda batch: (False, None, 4, torch.zeros(1, 48, 1, 2, 4))
    policy._pool_from_latent = lambda latent: torch.zeros(latent.shape[0], STATE_DIM)
    torch.manual_seed(5)
    out = policy.select_plan({"task": ["t"]}, plans=1, k=4)
    assert torch.equal(out["a_base"], expected["a_base"]) and torch.equal(out["z"], expected["z"])
    assert torch.equal(out["latents"], expected["latents"]) and out["plan_index"] == 0
    for a, b in zip(policy._snapshot_kv(), reference._snapshot_kv()):
        assert torch.equal(a["k"], b["k"])


def test_select_plan_batched_matches_looped_decode_plans_and_keeps_winner_rows():
    torch.manual_seed(7)
    looped = _mock_policy()
    looped._begin_chunk = lambda batch: (False, None, 4, torch.zeros(1, 48, 1, 2, 4))
    looped._pool_from_latent = lambda latent: latent.float().mean(dim=(1, 2, 3, 4), keepdim=True).reshape(latent.shape[0], 1).expand(-1, STATE_DIM)
    torch.manual_seed(11)
    ref = looped.decode_plans({"task": ["t"]}, plans=2, exec_step=-1, k=3)
    policy = _mock_policy()
    policy._begin_chunk = lambda batch: (False, None, 4, torch.zeros(1, 48, 1, 2, 4))
    policy._pool_from_latent = looped._pool_from_latent
    torch.manual_seed(11)
    out = policy.select_plan({"task": ["t"]}, plans=2, k=3, rank_fn=lambda s, pools: 1, decode_all=True)
    assert out["plan_index"] == 1
    torch.testing.assert_close(out["plan_a_base_all"], ref["a_base"], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(out["a_base"], ref["a_base"][1], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(out["latents"], ref["latents"][1:2], rtol=1e-5, atol=1e-6)
    assert out["plan_pools"].shape == (2, STATE_DIM) and torch.equal(out["plan_pool"], out["plan_pools"][1:2])
    cache = policy.transformer.blocks[0].attn1.attn_caches["pos"]["k"]
    assert cache.shape[0] == 2
    winner = _mock_policy()
    winner._begin_chunk = lambda batch: (False, None, 4, torch.zeros(1, 48, 1, 2, 4))
    winner._pool_from_latent = looped._pool_from_latent
    winner.decode_candidates({"task": ["t"]}, k=3, video_noise=ref["video_noise"][1:2], action_noise=mlp_to_model(ref["z"]))
    torch.testing.assert_close(cache, winner.transformer.blocks[0].attn1.attn_caches["pos"]["k"], rtol=1e-5, atol=1e-6)


def test_select_plan_eval_decodes_only_the_chosen_plan_and_random_plan_is_uniform(monkeypatch):
    policy = _mock_policy()
    policy._begin_chunk = lambda batch: (False, None, 4, torch.zeros(1, 48, 1, 2, 4))
    policy._pool_from_latent = lambda latent: torch.zeros(latent.shape[0], STATE_DIM)
    out = policy.select_plan({"task": ["t"]}, plans=3, k=2, rank_fn=lambda s, pools: 2)
    assert "plan_a_base_all" not in out and out["a_base"].shape == (2, HORIZON, ACTION_DIM)
    action_calls = [call for call in policy.transformer.calls if call[0]]
    assert len(action_calls) == policy.config.action_num_inference_steps
    monkeypatch.setattr(np.random, "randint", lambda n: 1)
    out = policy.select_plan({"task": ["t"]}, plans=3, k=2, rank_fn=lambda s, pools: 2, random_plan=True)
    assert out["plan_index"] == 1


def test_predict_action_chunk_uses_plan_critic_when_eval_plans_set():
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
    assert captured["plans"] == 3 and captured["k"] == 2 and captured["decode_all"] is False
    assert captured["rank"] == policy.residual_model.rank_plans
```

- [ ] **Step 2: Run them**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py -k "kv_rows or select_plan or eval_plans"`
Expected: FAIL with `ImportError` / `AttributeError: select_plan`.

- [ ] **Step 3: Implement the cache helpers** in `script/lingbot_rl_policy.py`

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
            cache[key] = cache[key][index].clone()
```

Keep `test_expand_conditional_kv_repeats_conditional_batch_only` passing: `repeat_interleave` of one row equals `repeat`.

- [ ] **Step 4: Make `_pool_from_latent` batch-capable**

Replace `_critic_cache_ready = False` with `self._critic_cache_batch = 0` in `__init__`; `_ensure_critic_cache(self, batch_size=1)` recreates the `"critic"` cache with `batch_size=batch_size` whenever `batch_size != self._critic_cache_batch` and records it. In `_pool_from_latent`, call `self._ensure_critic_cache(latent.shape[0])`, build the payload as today, then `payload["grid_id"] = payload["grid_id"][None].repeat(latent.shape[0], 1, 1)`, `payload["timesteps"] = payload["timesteps"][None].repeat(latent.shape[0], 1)`, and pool with `pool_critic_state(video_tokens, text_tokens.expand(latent.shape[0], -1, -1))`. If `_prepare_latent_input` already returns batched `grid_id` for batched latents (check `modeling_lingbot_va.py`), keep the shapes consistent with the batch-1 path, which must stay bit for bit identical.

- [ ] **Step 5: Batched video passes and `_decode_actions(..., plans)`**

Add `_video_pass_batched(self, latents, t, latent_cond, frame_st_id, write)` for `latents (M, ...)`: payload from `self._prepare_latent_input(latents, None, t, t, latent_cond_m, None, frame_st_id=frame_st_id)["latent_res_lst"]` where `latent_cond_m = None if latent_cond is None else latent_cond.expand(latents.shape[0], -1, -1, -1, -1)`; then set `payload["noisy_latents"] = payload["noisy_latents"].repeat(2, 1, 1, 1, 1)`, `payload["text_emb"] = torch.cat([self._prompt_embeds.to(self.dtype).expand(M, -1, -1), self._negative_prompt_embeds.to(self.dtype).expand(M, -1, -1)]).clone()`, and `grid_id` / `timesteps` repeated to `2M` rows following the layout `_repeat_input_for_cfg` produces for `M = 1`. Call `self.transformer(payload, update_cache=1 if write else 0, cache_name="pos", action_mode=False)`, `data_seq_to_patch(..., batch_size=2 * M)`, combine `pred[M:] + cfg.guidance_scale * (pred[:M] - pred[M:])` (or `pred[:M]` when guidance is 1), `self._scheduler.step(...)`, and reapply `latent_cond` to `latents[:, :, 0:1]` when `frame_st_id == 0`. When `M == 1` this must equal `_video_pass(..., advance=True)`; add `assert` nothing, the bit-for-bit test covers it.

`_decode_actions(self, actions, frame_st_id, k, plans=1)`: the `k == 1 and plans == 1` branch unchanged; the batched branch calls `expand_conditional_kv(self.transformer, k, plans)` and uses `rows = plans * k` wherever `k` sizes `action_cond`, `text_emb`, `grid_id` and `timesteps`.

- [ ] **Step 6: `select_plan`**

```python
    @torch.no_grad()
    def select_plan(self, batch, plans, k, rank_fn=None, random_plan=False, decode_all=False):
        if plans == 1:
            decoded = self.decode_candidates(batch, k=k)
            decoded["plan_pools"] = self._pool_from_latent(decoded["latents"]).float()
            decoded["plan_pool"] = decoded["plan_pools"][:1]
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
            latents = self._video_pass_batched(latents, t, latent_cond, frame_st_id, write=i == len(timesteps) - 1)
        pools = self._pool_from_latent(latents).float()
        index = int(np.random.randint(plans)) if random_plan else (0 if rank_fn is None else int(rank_fn(state.float(), pools)))
        if not 0 <= index < plans:
            raise ValueError("rank_fn must return a plan index")
        out = {"s": state.float(), "z": model_to_mlp(action_noise).float(), "plan_pools": pools,
               "plan_pool": pools[index:index + 1], "plan_index": index, "video_noise": video_noise,
               "latents": latents[index:index + 1], "first_chunk": first}
        if decode_all:
            stacked = action_noise.repeat(plans, 1, 1, 1, 1)
            decoded = model_to_mlp(self._decode_actions(stacked, frame_st_id, k, plans)).float()
            out["plan_a_base_all"] = decoded.reshape(plans, k, HORIZON, ACTION_DIM)
            out["a_base"] = out["plan_a_base_all"][index]
        else:
            select_kv_rows(self.transformer, [index, plans + index])
            out["a_base"] = model_to_mlp(self._decode_actions(action_noise.clone(), frame_st_id, k)).float()
        if decode_all:
            select_kv_rows(self.transformer, [index, plans + index])
        if first:
            self._first_chunk = False
        self._exec_step = 0
        self._started = True
        return out
```

`_decode_actions` snapshots and restores the cache it was given, so after `decode_all` the cache still has `2M` rows and the `select_kv_rows` call reduces it to the winner. In `__init__` add `self.eval_plans = 1`. In `predict_action_chunk`, when `self.residual_model is not None and self.eval_plans > 1`, call `self.select_plan(batch, self.eval_plans, k, rank_fn=self.residual_model.rank_plans)` and rank the K candidates with `self.residual_model.critic(self.residual_model.critic_state(state, decoded["plan_pool"].expand(k, -1)), executed)`; otherwise today's path.

- [ ] **Step 7: Verify and commit**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py && .cache/eval-venv/bin/python -m compileall -q script/`
Expected: all pass, including `test_single_plan_infer_reproduces_baseline_bit_for_bit` and `test_decode_plans_shapes_and_restores_policy_state` unchanged.

```bash
git add script/lingbot_rl_policy.py tests/test_lingbot_rl.py
git commit -m "Sample M video plans in one batched pass, pool them, and decode candidates per plan in one batch."
```

---

### Task 5: Expert plan pools

**Files:**
- Modify: `script/lingbot_rl_data.py`
- Test: `tests/test_lingbot_rl.py`

**Interfaces:**
- Consumes: `policy.extract_critic_state_from_latent(latent (1, 48, T, h, w), batch)` accepting `T = 4` (Task 4's `_pool_from_latent`).
- Produces: expert rows with `plan_pool (3072)` = pool of the published latents of the next 4 frames; `EXPERT_RECIPE = "dice-rl-expert-v2"`.

- [ ] **Step 1: Write the failing test** (extend the pattern of `test_featurize_experts_pools_published_latents`)

```python
def test_expert_rows_carry_the_pool_of_the_true_next_chunk(tmp_path):
    from script.lingbot_rl_data import EXPERT_RECIPE, featurize_experts
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
    assert EXPERT_RECIPE == "dice-rl-expert-v2"
    assert [row["s"][0] for row in rows] == [0.0, 3.0, 7.0]
    assert [row["plan_pool"][0] for row in rows] == [1.0, 4.0, 8.0]
    assert (1, 48, 4, 2, 2) in calls
```

The three chunks start at env offsets 0, 12, 28, so frame indices 0, 3, 7; the plan latent for a chunk is frames `idx + 1 .. idx + 4`, clamped to the last frame and padded by repeating it.

- [ ] **Step 2: Run it**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py -k true_next_chunk`
Expected: FAIL (`KeyError: 'plan_pool'`).

- [ ] **Step 3: Implement** in `script/lingbot_rl_data.py`

Set `EXPERT_RECIPE = "dice-rl-expert-v2"`. In `_episode_rows`, inside the published-latents branch after computing `state`:

```python
            plan_index = torch.clamp(torch.arange(index + 1, index + 5), max=max(frame_count - 1, 0))
            plan_pool = _tensor_state(policy.extract_critic_state_from_latent(
                latents[:, :, plan_index], {"task": [episode["task"]]}))
```

and in the RGB-frames branch `plan_pool = state.copy()`. Add `"plan_pool": plan_pool` to the row dict. Nothing else changes; `expert_fingerprint` already embeds the recipe string, so existing caches rebuild.

- [ ] **Step 4: Verify and commit**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py`
Expected: all pass (`test_featurize_experts_rebuilds_corrupt_cache` and the published-latents test still hold).

```bash
git add script/lingbot_rl_data.py tests/test_lingbot_rl.py
git commit -m "Pool the demo's true next chunk as the expert plan pool."
```

---

### Task 6: Training loop, metrics and evaluation with plans

**Files:**
- Modify: `script/lingbot_rl_train.py`
- Test: `tests/test_lingbot_rl.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `train()` and `evaluate()` honouring `config.plan_candidates`; W&B keys `plan_agreement`, `plan_regret`, `plan_value_spread`, `plan_random_rate`, `plan_critic_loss`, `v_plan_mean`, `plan_target_mean`; `settings.json` gains `"runtime": {"cuda": torch.version.cuda, "flash_sdp": torch.backends.cuda.flash_sdp_enabled()}` on CUDA hosts; `RESUME_KEYS` gains `plan_critic`, `plan_opt`, `plan_lr`.

- [ ] **Step 1: Write the failing tests**

Extend `_StubPolicy` with a `select_plan` that returns three plans whose pools are `torch.full((1, STATE_DIM), m)` and whose candidates are `decode_candidates`' candidates offset by `10 * m` in channel 0, records `(plans, k, random_plan, decode_all)` in a class-level list, and executes the index `rank_fn` returns (or `1` when `random_plan`). Then:

```python
def test_train_with_plans_stores_plan_rows_updates_plan_critic_and_logs_plan_metrics(tmp_path, monkeypatch):
    out = _smoke_train(tmp_path, monkeypatch, 36, model_factory=lambda **kw: DiceResidualModel(**kw),
                       config=RLConfig(plan_candidates=3, plan_random_prob=0.0, selection_warmup_steps=0,
                                       residual_input="base"))
    rows = [row for row in out["rows"] if float(row["is_expert"]) == 0.0]
    assert rows and rows[0]["plan_pools_all"].shape == (3, STATE_DIM)
    assert rows[0]["plan_a_base_all"].shape == (3, K_CANDIDATES, HORIZON, ACTION_DIM)
    assert float(rows[0]["plan_pool"][0]) == float(rows[0]["plan_index"])
    assert _StubPolicy.plan_calls and all(call[3] is True for call in _StubPolicy.plan_calls)
    logged = [entry for entry in out["logs"] if "plan_agreement" in entry]
    assert logged and {"plan_regret", "plan_value_spread", "plan_random_rate"} <= set(logged[0])
    assert any("plan_critic_loss" in entry for entry in out["logs"])
    assert {"plan_critic", "plan_opt", "plan_lr"} <= set(out["resume"])


def test_train_random_plan_during_warmup_and_with_probability_afterwards(tmp_path, monkeypatch):
    _StubPolicy.plan_calls = []
    _smoke_train(tmp_path / "warm", monkeypatch, 36, model_factory=lambda **kw: DiceResidualModel(**kw),
                 config=RLConfig(plan_candidates=3, plan_random_prob=0.0, selection_warmup_steps=10_000))
    assert all(call[2] is True for call in _StubPolicy.plan_calls)
    _StubPolicy.plan_calls = []
    monkeypatch.setattr(np.random, "random", lambda: 0.05)
    _smoke_train(tmp_path / "rand", monkeypatch, 36, model_factory=lambda **kw: DiceResidualModel(**kw),
                 config=RLConfig(plan_candidates=3, plan_random_prob=0.1, selection_warmup_steps=0))
    assert all(call[2] is True for call in _StubPolicy.plan_calls)


def test_single_plan_training_path_is_unchanged(tmp_path, monkeypatch):
    _StubPolicy.plan_calls = []
    out = _smoke_train(tmp_path, monkeypatch, 36, model_factory=_sum_critic_model, config=RLConfig(selection_warmup_steps=0))
    assert _StubPolicy.plan_calls == []
    assert [float(row["a"][0][0]) for row in out["rows"]] == [3.0, 3.0, 3.0]
    assert not any("plan_agreement" in entry for entry in out["logs"])


def test_evaluate_sets_eval_plans_and_records_them(tmp_path, monkeypatch):
    # mirror test_evaluate_records_video_exec_step_and_passes_it_to_the_policy with RLConfig(plan_candidates=4)
    # and assert policy.eval_plans == 4 and settings["eval_plans"] == 4 and settings["rl"]["plan_candidates"] == 4
```

Write the last test fully by copying the referenced test and changing the config and assertions.

- [ ] **Step 2: Run them**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py -k "plans or single_plan_training"`
Expected: FAIL (`KeyError: 'plan_pools_all'` / missing log keys).

- [ ] **Step 3: Implement collection** in `train()`

Construct `model = DiceResidualModel(device=device, residual_input=config.residual_input, epsilon=config.epsilon, plan_candidates=config.plan_candidates)` and `buffer = ChunkReplay(plans=config.plan_candidates)`; set `policy.eval_plans = config.plan_candidates`. Replace the `decode_candidates` call in the episode loop with:

```python
                plans = config.plan_candidates
                if plans > 1:
                    random_plan = env_steps < config.selection_warmup_steps or np.random.random() < config.plan_random_prob
                    decoded = policy.select_plan(batch, plans, config.k_candidates, rank_fn=model.rank_plans,
                                                 random_plan=random_plan, decode_all=True)
                else:
                    decoded = policy.decode_candidates(batch, k=config.k_candidates)
```

Keep `state`, `noise`, `a_base` as today; add `pool = decoded["plan_pool"].to(device=device, dtype=torch.float32) if plans > 1 else None`, `critic_k = model.critic_state(state_k, None if pool is None else pool.expand(k, -1))`, and rank candidates with `model.critic(critic_k, executed)`. After choosing `star`, when `plans > 1` compute under `no_grad`:

```python
                    pools_all = decoded["plan_pools"].to(device=device, dtype=torch.float32)
                    all_base = decoded["plan_a_base_all"].to(device=device, dtype=torch.float32)
                    flat_state = state.expand(plans * k, -1)
                    flat_pool = pools_all.repeat_interleave(k, dim=0)
                    flat_z = noise.repeat(plans, 1, 1)
                    flat_base = all_base.reshape(plans * k, HORIZON, ACTION_DIM)
                    q_all = model.critic(model.critic_state(flat_state, flat_pool),
                                         apply_residual(flat_base, model.actor(flat_state, flat_z, flat_base))).reshape(plans, k).max(dim=1).values
                    v_all = model.plan_critic(state.expand(plans, -1), pools_all).reshape(-1)
                    plan_info = {
                        "plan_agreement": float(int(v_all.argmax()) == int(q_all.argmax())),
                        "plan_regret": float(q_all.max() - q_all[decoded["plan_index"]]),
                        "plan_value_spread": float(v_all.max() - v_all.min()),
                        "plan_random_rate": float(random_plan),
                    }
```

Add to the `buffer.add_online` row when `plans > 1`: `"plan_pool": _host_array(pool)[0]`, `"plan_pools_all": _host_array(pools_all)`, `"plan_index": int(decoded["plan_index"])`, `"plan_a_base_all": _host_array(all_base)`. Log `plan_info` with the per-chunk `run.log` of the update block (merge into `update_info` when an update happens; otherwise log it on its own with `env_steps`).

- [ ] **Step 4: Updates, sharpening, checkpoints**

`_update_from_buffer`: build `critic_state = model.critic_state(sample["s"], sample["plan_pool"])` and `next_critic_state = model.critic_state(sample["s_next"], sample["plan_pool_next"])`; pass `next_critic_state` to `n_step_target`, `critic_state` to `update_critic`, and `(sample["s"], critic_state, ...)` to `update_actor`. After `update_critic`, when `model.plan_critic is not None`: `info.update(model.update_plan_critic(sample["s"], sample["plan_pools_all"], sample["z_all"], sample["plan_a_base_all"]))`.

`replay_sharpening` and `sharpening_metrics`: take the critic state from `model.critic_state(sample["s"], sample["plan_pool"])` (expanded over `k` like `state_k`) for both `model.critic` calls; `_maybe_sharpen` uses `policy.select_plan(batch, policy.eval_plans, 8, rank_fn=model.rank_plans)` when `policy.eval_plans > 1` and passes `model.critic_state(state, decoded["plan_pool"].expand(...))` as the critic state.

`RESUME_KEYS` gains `"plan_critic", "plan_opt", "plan_lr"`; `save_inference` allows `"plan_critic"`. In `maybe_checkpoint` keep `policy.eval_plans = config.plan_candidates` (train-eval uses the critic from the first update; during the selection warm-up the policy still ranks with the untrained plan critic, which is acceptable for the 10-episode train-eval and matches `eval_candidates` handling).

Record runtime in `settings.json`: `{"config": ..., "recipe": ..., "runtime": {"cuda": torch.version.cuda, "flash_sdp": bool(torch.backends.cuda.flash_sdp_enabled()) if device == "cuda" else None}}`.

- [ ] **Step 5: Evaluation**

In `evaluate()`: build the model with `plan_candidates=config.plan_candidates`, set `policy.eval_plans = config.plan_candidates`, and add `"eval_plans": policy.eval_plans` to the shard settings JSON.

- [ ] **Step 6: Verify and commit**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py tests/test_lingbot_eval.py && .cache/eval-venv/bin/python -m script.lingbot_rl_config`
Expected: all pass.

```bash
git add script/lingbot_rl_train.py tests/test_lingbot_rl.py
git commit -m "Collect with M sampled plans, train the plan critic, and evaluate with plan selection."
```

---

### Task 7: GB10 smoke and launch (controller, not a subagent)

- [ ] Step 1: sync main to both boxes; on Jarvis run `bash brev/train.sh smoke-plan 9 --residual-input base --epsilon -0.3 --video-steps 5 --action-steps 10 --plan-candidates 4 --env-steps 320000 --max-env-steps 300 --no-tmux`; confirm `settings.json` runtime shows cuda 13 and flash SDPA true, `plan_critic_loss` logged and falling, `plan_agreement` present, seconds per env step from the log; repeat with `--plan-candidates 1` for the throughput comparison.
- [ ] Step 2: expert featurisation rebuilds under recipe v2 (about 900 extra pooling passes); confirm `expert_features.pt` rows carry `plan_pool`.
- [ ] Step 3: launch `dice-t9-plan-m4` on Jarvis and `dice-t0-plan-m4` on Watson: `bash brev/train.sh dice-tX-plan-m4 X --residual-input base --epsilon -0.3 --env-steps 320000 --video-steps 5 --action-steps 10 --plan-candidates 4`.
- [ ] Step 4: record the runs, the SFT re-bench numbers and the smoke throughput in `docs/heldout-results.md`.

## Self-review notes

- Spec coverage: §4 (batched passes, batched decode) → Task 4; §5/§6 → Tasks 4 and 6; §7 → Task 2; §8 → Task 5; §9 → Task 3; §10 → Task 1; §11 tests → each task; §12 runs → Task 7.
- The plan pool is the history-free critic pool of the denoised chunk for both online and expert rows (spec §7's "plan pool"), computed by the same function as `s`, so no `proj_out` capture hook is needed in the batched passes.
- Type consistency: `critic_state(state, pool)` is the only place that widens the critic input; `rank_plans(state, pools)` matches `rank_fn(state, pools)` in `select_plan`; `plan_a_base_all` is `(M, K, 16, 30)` in the policy output, the buffer and `update_plan_critic`.
