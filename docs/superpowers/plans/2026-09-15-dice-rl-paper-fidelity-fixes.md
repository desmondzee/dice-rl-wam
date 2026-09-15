# DICE-RL Paper-Fidelity Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the five implementation deviations from the DICE-RL paper (arXiv 2603.10263v2) identified in the results analysis, so the LingBot LIBERO-10 RL run can be relaunched with paper-faithful training.

**Architecture:** All fixes stay inside the existing residual-RL stack (`script/lingbot_rl_*.py`) and keep the pinned constraint that the frozen 5B prior is never called inside the update loop. Multi-sample expectation training is approximated by storing all K=4 collected action candidates per chunk in the replay buffer and recomputing residuals with the *current* actor at update time (both for critic targets at next states and the actor objective at current states). The BC-loss filter is re-anchored to a Monte-Carlo return-to-go computed at episode finalization. UTD becomes fresh-minibatch-per-gradient-step. The residual actor's output layer is zero-initialized. Evaluation gains critic-ranked best-of-N.

**Tech Stack:** PyTorch (CPU for tests), numpy, pytest. Local verification venv: `.cache/eval-venv/bin/python`. No GPU, no Modal calls, no network during tests.

**Spec:** `docs/dice-rl-results-analysis.md` (sections "Implementation deviations" and "Recommended next steps"). Where this plan conflicts with the original design doc `docs/superpowers/specs/2026-09-14-lingbot-dice-rl-design.md` (specifically its row "Multi-sample K / best-of-K → train uses stored `(s,z,a_base)`"), this plan supersedes it: single-sample training is the deviation being fixed.

## Global Constraints

- The frozen prior is never invoked inside `_update_from_buffer` / `DiceResidualModel` — training must still run without the 5B transformer loaded (resume-without-prior invariant).
- K stays 4 (`RLConfig.k_candidates` pinned; validation must keep rejecting other values).
- Pinned hyperparameters stay unchanged: β=100, ε=-0.5, γ=0.99, n_step=3, UTD=10, τ=0.01, LR=1e-4, batch 256, ensemble 10, RLPD 0.5→0.1, budget 100,000 env steps.
- The SFT-parity inference path must be untouched: with `residual_model is None`, `predict_action_chunk` must still decode exactly one candidate (this reproduces the 69% SFT eval).
- Verification command for every task: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py` (targeted `-k`/node selections per step; full file at the end of each task).
- No comments or docstrings in new or modified code — the user wants lean, minimal code; the code and its tests are the documentation. This applies to implementation and test code alike.
- Commit messages: plain imperative sentences matching repo style (e.g. "Zero-initialize the DICE-RL residual actor output layer."), no conventional-commit prefixes.
- The protocol/recipe fingerprint changes in Task 6; old `resume/latest.pt` files and the old `expert_features.pt` cache become intentionally unresumable/rebuildable. Do not add backward-compatibility shims.

---

### Task 0: Commit the pending resume/checkpoint work

The working tree already contains finished, tested changes (per-eval-point checkpoints, RNG capture, W&B id resume) in `script/lingbot_rl_train.py` and `tests/test_lingbot_rl.py`. Land them first so the fidelity fixes are reviewable on their own.

**Files:**
- Modify: none (commit only)

- [ ] **Step 1: Run the full suite on the current tree**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py`
Expected: all tests pass. If anything fails, stop and report — do not start Task 1 on a red tree.

- [ ] **Step 2: Commit**

```bash
git add script/lingbot_rl_train.py tests/test_lingbot_rl.py
git commit -m "Save per-eval-point RL checkpoints and make resume restore RNG, eval schedule, and the W&B run id."
```

---

### Task 1: Zero-initialize the residual actor output layer

Fixes finding 4 of the spec: at step 0 the RL policy must equal the frozen prior exactly (the baseline run opened at 1/10 train-eval vs ~7/10 expected because the random residual corrupted actions).

**Files:**
- Modify: `script/lingbot_rl_model.py` (class `ResidualActor`, lines 51–61)
- Test: `tests/test_lingbot_rl.py`

**Interfaces:**
- Consumes: `_mlp(in_dim, out_dim)` as defined in `lingbot_rl_model.py` (final element of the `nn.Sequential` is the output `nn.Linear`).
- Produces: `ResidualActor()` whose `forward` returns exactly zeros for any input at initialization. Later tasks (3, 4) rely on this for deterministic target tests.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_lingbot_rl.py`:

```python
def test_residual_actor_initializes_to_zero_so_policy_starts_at_prior():
    torch.manual_seed(3)
    model = DiceResidualModel(device="cpu")
    state = torch.randn(5, STATE_DIM)
    noise = torch.randn(5, HORIZON, ACTION_DIM)
    residual = model.actor(state, noise)
    torch.testing.assert_close(residual, torch.zeros(5, HORIZON, ACTION_DIM))
    a_base = mask_unused_dof(torch.randn(5, HORIZON, ACTION_DIM))
    torch.testing.assert_close(apply_residual(a_base, residual), a_base)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py::test_residual_actor_initializes_to_zero_so_policy_starts_at_prior -v`
Expected: FAIL on the first `assert_close` (nonzero residual from default Linear init).

- [ ] **Step 3: Zero-init the output layer**

In `script/lingbot_rl_model.py`, change `ResidualActor.__init__`:

```python
class ResidualActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = _mlp(STATE_DIM + HORIZON * ACTION_DIM, HORIZON * ACTION_DIM)
        final = self.net[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
```

Do NOT touch `CriticEnsemble` (zero-initializing critics would break learning).

- [ ] **Step 4: Run test to verify it passes, plus neighbors**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py -k "residual_actor_initializes or actor_critic_one_step or cast_bfloat16" -v`
Expected: PASS. (`test_actor_critic_one_step_update_on_random_tensors` asserts actor parameters change after one update; the gradient through the zeroed layer's inputs is nonzero, so it still passes. If it fails, that is a real problem — report it, don't loosen the test.)

- [ ] **Step 5: Commit**

```bash
git add script/lingbot_rl_model.py tests/test_lingbot_rl.py
git commit -m "Zero-initialize the DICE-RL residual actor output layer so training starts at the frozen prior."
```

---

### Task 2: Store per-chunk candidate sets and Monte-Carlo returns in the replay buffer

Fixes the data layer for findings 1 and 2: every row carries all K=4 collected candidates `(z_all, a_base_all)`, the next-chunk candidate set for target computation, and the discounted Monte-Carlo return-to-go `mc_return` computed at episode finalization.

**Files:**
- Modify: `script/lingbot_rl_buffer.py`
- Test: `tests/test_lingbot_rl.py`

**Interfaces:**
- Consumes: `K_CANDIDATES` (new constant, add to `script/lingbot_rl_model.py`: `K_CANDIDATES = 4` next to the other pinned constants).
- Produces: `buffer.sample(batch_size, expert_ratio)` batches with new tensors:
  - `z_all`, `a_base_all`: shape `(B, 4, HORIZON, ACTION_DIM)` float32 — the K candidates proposed at the row's state.
  - `z_next_all`, `a_base_next_all`: shape `(B, 4, HORIZON, ACTION_DIM)` — the candidates proposed at the n-step bootstrap state.
  - `mc_return`: shape `(B, 1)` — discounted return-to-go from this chunk to episode end; `0.0` for rows of a not-yet-finalized episode (conservative: the BC filter can never disable on them).
  - Rows missing `z_all`/`a_base_all` at insert (expert rows, old test helpers) default to the single `z`/`a_base` repeated 4 times.
  - `a_next` is kept unchanged (diagnostics and existing tests), no longer consumed by the model after Task 3.

- [ ] **Step 1: Add `K_CANDIDATES` to the model constants**

In `script/lingbot_rl_model.py`, after `REPLAY_CAPACITY = 100_000` add:

```python
K_CANDIDATES = 4
```

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_lingbot_rl.py` (also add `K_CANDIDATES` to the existing `from script.lingbot_rl_model import (...)` import list):

```python
def test_finalize_stores_monte_carlo_return_and_next_candidates():
    buf = ChunkReplay(capacity=32)
    for step, (reward, done) in enumerate(((0, 0), (0, 0), (1, 1))):
        row = _replay_row(reward, done)
        row["z_all"] = np.full((K_CANDIDATES, HORIZON, ACTION_DIM), float(step), np.float32)
        row["a_base_all"] = np.full((K_CANDIDATES, HORIZON, ACTION_DIM), 10.0 + step, np.float32)
        buf.add_online(row)
    buf.finalize_episode()
    rows = buf.rows()
    assert [float(row["mc_return"]) for row in rows] == pytest.approx([GAMMA ** 2, GAMMA, 1.0])
    np.testing.assert_array_equal(
        rows[0]["z_next_all"], np.full((K_CANDIDATES, HORIZON, ACTION_DIM), 2.0, np.float32))
    np.testing.assert_array_equal(
        rows[0]["a_base_next_all"], np.full((K_CANDIDATES, HORIZON, ACTION_DIM), 12.0, np.float32))


def test_store_defaults_repeat_single_candidate_for_experts():
    buf = ChunkReplay(capacity=8)
    row = _replay_row(0, 1, expert=True)
    row["z"] = np.full((HORIZON, ACTION_DIM), 7.0, np.float32)
    row["a_base"] = np.full((HORIZON, ACTION_DIM), 8.0, np.float32)
    buf.add_expert(row)
    buf.finalize_episode()
    stored = buf.rows()[0]
    assert stored["z_all"].shape == (K_CANDIDATES, HORIZON, ACTION_DIM)
    np.testing.assert_array_equal(stored["z_all"][3], row["z"])
    np.testing.assert_array_equal(stored["a_base_all"][0], stored["a_base_all"][3])


def test_open_episode_view_has_zero_mc_return_and_successor_candidates():
    buf = ChunkReplay(capacity=32)
    first = _replay_row(0, 0)
    second = _replay_row(0, 0)
    second["z_all"] = np.full((K_CANDIDATES, HORIZON, ACTION_DIM), 5.0, np.float32)
    buf.add_online(first)
    buf.add_online(second)
    batch = buf.sample(1, expert_ratio=0.0)
    assert batch["mc_return"].shape == (1, 1)
    assert float(batch["mc_return"][0]) == 0.0
    assert batch["z_all"].shape == (1, K_CANDIDATES, HORIZON, ACTION_DIM)
    np.testing.assert_array_equal(
        batch["z_next_all"][0].numpy(), np.full((K_CANDIDATES, HORIZON, ACTION_DIM), 5.0, np.float32))


def test_store_rejects_wrong_candidate_count():
    buf = ChunkReplay(capacity=8)
    row = _replay_row(0, 0)
    row["z_all"] = np.zeros((2, HORIZON, ACTION_DIM), np.float32)
    row["a_base_all"] = np.zeros((2, HORIZON, ACTION_DIM), np.float32)
    with pytest.raises(ValueError, match="candidate"):
        buf.add_online(row)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py -k "monte_carlo or repeat_single_candidate or zero_mc_return or wrong_candidate_count" -v`
Expected: FAIL with `KeyError: 'mc_return'` / missing keys.

- [ ] **Step 4: Implement the buffer changes**

In `script/lingbot_rl_buffer.py`:

Change the import and key tuples:

```python
from script.lingbot_rl_model import GAMMA, K_CANDIDATES, N_STEP_CHUNKS, REPLAY_CAPACITY

FLOAT_KEYS = (
    "s", "z", "a_base", "a", "reward", "done", "s_next", "a_next", "n_steps", "is_expert",
    "z_all", "a_base_all", "z_next_all", "a_base_next_all", "mc_return",
)

KEYS = (
    "s", "z", "a_base", "a", "reward", "done", "s_next", "a_next",
    "n_steps", "is_expert", "task_id", "n_env_actions",
    "z_all", "a_base_all", "z_next_all", "a_base_next_all", "mc_return",
)
```

In `_store`, after the existing `setdefault` lines and before the `for key in FLOAT_KEYS` loop:

```python
        stored.setdefault(
            "z_all", np.repeat(_host_float(stored["z"])[None], K_CANDIDATES, axis=0))
        stored.setdefault(
            "a_base_all", np.repeat(_host_float(stored["a_base"])[None], K_CANDIDATES, axis=0))
        if _host_float(stored["z_all"]).shape[0] != K_CANDIDATES or \
                _host_float(stored["a_base_all"]).shape[0] != K_CANDIDATES:
            raise ValueError(f"Rows must carry exactly {K_CANDIDATES} candidate proposals")
        stored.setdefault("z_next_all", np.array(stored["z_all"], copy=True))
        stored.setdefault("a_base_next_all", np.array(stored["a_base_all"], copy=True))
        stored.setdefault("mc_return", np.float32(0.0))
```

In `_n_step_view`, after the existing `view["a_next"] = ...` line:

```python
        view["z_next_all"] = np.array(episode[nxt]["z_all"], copy=True)
        view["a_base_next_all"] = np.array(episode[nxt]["a_base_all"], copy=True)
        view["mc_return"] = np.float32(0.0)
```

In `finalize_episode`, after `dones = [...]` add the return-to-go pass, and extend the per-row assignment:

```python
        mc_returns = [0.0] * length
        running = 0.0
        for index in range(length - 1, -1, -1):
            running = rewards[index] + GAMMA * running
            mc_returns[index] = running
```

and inside the `for index, row in enumerate(episode):` loop, after `row["a_next"] = ...`:

```python
            row["z_next_all"] = np.array(episode[nxt]["z_all"], copy=True)
            row["a_base_next_all"] = np.array(episode[nxt]["a_base_all"], copy=True)
            row["mc_return"] = np.float32(mc_returns[index])
```

(`rewards` is captured before the loop overwrites `row["reward"]` with n-step sums, so `mc_returns` is computed from raw per-chunk rewards — do not move the capture.)

In `sample`, extend the unsqueeze condition:

```python
            if key in ("reward", "done", "n_steps", "is_expert", "mc_return") and tensor.ndim == 1:
                tensor = tensor.unsqueeze(-1)
```

- [ ] **Step 5: Extend the bf16 conversion test to the new keys**

In `test_replay_sample_converts_bf16_tensors_to_float32`, change the key loop to:

```python
    for key in ("s", "z", "a_base", "a", "s_next", "a_next",
                "z_all", "a_base_all", "z_next_all", "a_base_next_all", "mc_return"):
```

- [ ] **Step 6: Run the buffer tests**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py -k "buffer or replay or n_step or monte_carlo or candidate or mc_return or rlpd_mix or expert_n_step or open_episode" -v`
Expected: PASS, including all pre-existing buffer tests (they rely on the repeat-defaults).

- [ ] **Step 7: Commit**

```bash
git add script/lingbot_rl_model.py script/lingbot_rl_buffer.py tests/test_lingbot_rl.py
git commit -m "Store per-chunk candidate sets and Monte-Carlo returns in the DICE-RL replay buffer."
```

---

### Task 3: Multi-candidate n-step critic target (paper Eq. 4)

Fixes finding 1's critic half: the bootstrap evaluates the *current* policy — residuals recomputed with the current actor on the stored next-state candidates — averaged over K, instead of the stale executed `a_next`.

**Files:**
- Modify: `script/lingbot_rl_model.py` (`DiceResidualModel.n_step_target`)
- Test: `tests/test_lingbot_rl.py`

**Interfaces:**
- Consumes: batch fields `z_next_all`, `a_base_next_all` `(B, K, HORIZON, ACTION_DIM)` from Task 2; zero-init actor from Task 1.
- Produces: `n_step_target(reward, done, next_state, z_next_all, a_base_next_all, n_steps) -> (B, 1)` where the backup is `mean_k target_critic(s', apply_residual(a_base'_k, actor(s', z'_k)))`. Task 5 calls it with these names.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_lingbot_rl.py`:

```python
def test_n_step_target_averages_current_policy_over_candidates():
    torch.manual_seed(11)
    model = DiceResidualModel(device="cpu")
    next_state = torch.randn(2, STATE_DIM)
    z_next_all = torch.randn(2, K_CANDIDATES, HORIZON, ACTION_DIM)
    a_base_next_all = mask_unused_dof(torch.randn(2, K_CANDIDATES, HORIZON, ACTION_DIM))
    reward = torch.tensor([[0.0], [1.0]])
    done = torch.tensor([[0.0], [1.0]])
    n_steps = torch.tensor([[3.0], [1.0]])
    target = model.n_step_target(reward, done, next_state, z_next_all, a_base_next_all, n_steps)
    assert target.shape == (2, 1)
    with torch.no_grad():
        qs = torch.stack([
            model.target_critic(next_state, a_base_next_all[:, k]) for k in range(K_CANDIDATES)
        ], dim=0).mean(dim=0)
    torch.testing.assert_close(target[0], (GAMMA ** 3) * qs[0])
    torch.testing.assert_close(target[1], torch.tensor([1.0]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py::test_n_step_target_averages_current_policy_over_candidates -v`
Expected: FAIL (old signature treats `z_next_all` as a state-shaped action tensor; shape error or wrong value).

- [ ] **Step 3: Implement the new target**

Replace `n_step_target` in `script/lingbot_rl_model.py`:

```python
    def n_step_target(self, reward, done, next_state, z_next_all, a_base_next_all, n_steps):
        with torch.no_grad():
            next_state = mlp_float(next_state)
            z_next_all = mlp_float(z_next_all)
            a_base_next_all = mlp_float(a_base_next_all)
            batch, k = z_next_all.shape[0], z_next_all.shape[1]
            state_k = next_state.unsqueeze(1).expand(batch, k, next_state.shape[-1]).reshape(batch * k, -1)
            z_flat = z_next_all.reshape(batch * k, HORIZON, ACTION_DIM)
            base_flat = a_base_next_all.reshape(batch * k, HORIZON, ACTION_DIM)
            a_next = apply_residual(base_flat, self.actor(state_k, z_flat))
            backup = self.target_critic(state_k, a_next).reshape(batch, k, 1).mean(dim=1)
            return reward + (GAMMA ** n_steps) * (1.0 - done) * backup
```

- [ ] **Step 4: Update the two existing callers in tests**

Replace `test_n_step_chunk_return_sparse_terminal` with:

```python
def test_n_step_chunk_return_sparse_terminal():
    model = DiceResidualModel(device="cpu")
    done = torch.tensor([[1.0], [1.0], [1.0]])
    next_state = torch.zeros(3, STATE_DIM)
    z_next_all = torch.zeros(3, K_CANDIDATES, HORIZON, ACTION_DIM)
    a_base_next_all = torch.zeros(3, K_CANDIDATES, HORIZON, ACTION_DIM)
    n_steps = torch.tensor([[3.0], [2.0], [1.0]])
    summed = torch.tensor([[0.99 ** 2], [0.99], [1.0]])
    target = model.n_step_target(summed, done, next_state, z_next_all, a_base_next_all, n_steps)
    assert target.shape == (3, 1)
    torch.testing.assert_close(target[2], torch.tensor([1.0]))
    assert target[0].item() == pytest.approx(0.99 ** 2)
    assert target[1].item() == pytest.approx(0.99)
```

In `test_actor_critic_one_step_update_on_random_tensors`, replace the two lines

```python
    next_action = mask_unused_dof(torch.randn(8, HORIZON, ACTION_DIM))
    target = model.n_step_target(reward, done, next_state, next_action, n_steps)
```

with

```python
    z_next_all = torch.randn(8, K_CANDIDATES, HORIZON, ACTION_DIM)
    a_base_next_all = mask_unused_dof(torch.randn(8, K_CANDIDATES, HORIZON, ACTION_DIM))
    target = model.n_step_target(reward, done, next_state, z_next_all, a_base_next_all, n_steps)
```

(The `update_actor` call two lines below changes in Task 4 — leave it for now; this test will be red until Task 4 Step 3, which is acceptable only within the same session. If executing tasks in separate sessions, do Task 4 immediately after.)

- [ ] **Step 5: Run the model tests**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py -k "n_step_target or n_step_chunk" -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add script/lingbot_rl_model.py tests/test_lingbot_rl.py
git commit -m "Bootstrap the DICE-RL critic target from the current policy averaged over stored next-state candidates."
```

---

### Task 4: Multi-candidate actor objective with the Monte-Carlo-anchored BC filter (paper Eq. 5–8)

Fixes finding 1's actor half and finding 2: the actor maximizes mean Q over the K stored candidates at the current state, and the BC-loss filter's overestimation guard compares against `mc_return` instead of the bootstrapped TD target.

**Files:**
- Modify: `script/lingbot_rl_model.py` (`update_actor`; new module function `bc_filter_keep`)
- Modify: `script/lingbot_rl_train.py:236` (the `update_actor` call inside `_update_from_buffer` — full rewrite lands in Task 5, but the call must compile now)
- Test: `tests/test_lingbot_rl.py`

**Interfaces:**
- Consumes: batch fields `z_all`, `a_base_all` `(B, K, HORIZON, ACTION_DIM)`, `mc_return` `(B, 1)` from Task 2.
- Produces:
  - `bc_filter_keep(q_a, q_base, mc_return) -> tensor` — 1.0 where the BC penalty applies, 0.0 where it is disabled (Eq. 6: disabled iff `q_a > q_base` AND `q_a - mc_return < EPSILON`). Broadcasts `(B, K)` against `(B, 1)`.
  - `update_actor(state, z_all, a_base_all, is_expert, mc_return) -> dict` with the same metric keys as before (`actor_loss`, `residual_rms`, `q_mean`, `q_min`, `bc_filter_rate`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_lingbot_rl.py`:

```python
def test_bc_filter_disables_only_on_better_and_mc_underestimated():
    from script.lingbot_rl_model import bc_filter_keep

    q_a = torch.tensor([[0.9, 0.9, 0.3, 0.9]])
    q_base = torch.tensor([[0.5, 0.5, 0.5, 0.95]])
    mc_return = torch.tensor([[1.5]])
    keep = bc_filter_keep(q_a, q_base, mc_return)
    torch.testing.assert_close(keep, torch.tensor([[0.0, 0.0, 1.0, 1.0]]))
    keep_low_mc = bc_filter_keep(q_a, q_base, torch.tensor([[0.9]]))
    torch.testing.assert_close(keep_low_mc, torch.ones(1, 4))


def test_update_actor_consumes_candidate_sets_and_mc_return():
    torch.manual_seed(4)
    model = DiceResidualModel(device="cpu")
    state = torch.randn(6, STATE_DIM)
    z_all = torch.randn(6, K_CANDIDATES, HORIZON, ACTION_DIM)
    a_base_all = mask_unused_dof(torch.randn(6, K_CANDIDATES, HORIZON, ACTION_DIM))
    is_expert = torch.zeros(6, 1)
    is_expert[:3] = 1
    mc_return = torch.zeros(6, 1)
    before = {k: v.detach().clone() for k, v in model.actor.named_parameters()}
    info = model.update_actor(state, z_all, a_base_all, is_expert, mc_return)
    assert torch.isfinite(torch.tensor(info["actor_loss"]))
    assert 0.0 <= info["bc_filter_rate"] <= 1.0
    assert any(not torch.equal(before[k], v) for k, v in model.actor.named_parameters())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py -k "bc_filter_disables or consumes_candidate_sets" -v`
Expected: FAIL (`bc_filter_keep` undefined; `update_actor` rejects 4-D candidate tensors).

- [ ] **Step 3: Implement**

In `script/lingbot_rl_model.py`, add after `apply_residual`:

```python
def bc_filter_keep(q_a, q_base, mc_return):
    better = (q_a > q_base).float()
    underestimated = ((q_a - mc_return) < EPSILON).float()
    return 1.0 - better * underestimated
```

Replace `update_actor`:

```python
    def update_actor(self, state, z_all, a_base_all, is_expert, mc_return):
        state = mlp_float(state)
        z_all = mlp_float(z_all)
        a_base_all = mlp_float(a_base_all)
        batch, k = z_all.shape[0], z_all.shape[1]
        state_k = state.unsqueeze(1).expand(batch, k, state.shape[-1]).reshape(batch * k, -1)
        z_flat = z_all.reshape(batch * k, HORIZON, ACTION_DIM)
        base_flat = a_base_all.reshape(batch * k, HORIZON, ACTION_DIM)
        residual = self.actor(state_k, z_flat)
        action = apply_residual(base_flat, residual)
        q_a = self.critic(state_k, action).reshape(batch, k)
        with torch.no_grad():
            q_base = self.critic(state_k, base_flat).reshape(batch, k)
            bc_keep = bc_filter_keep(q_a, q_base, mlp_float(mc_return))
        online = (is_expert == 0).float()
        q_term = -(q_a.mean(dim=1, keepdim=True) * online).sum() / online.sum().clamp(min=1.0)
        mse = ((action - base_flat) ** 2).mean(dim=(1, 2)).reshape(batch, k)
        bc = (bc_keep * mse).mean()
        loss = q_term + BETA * bc
        self.actor_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.actor_opt.step()
        return {
            "actor_loss": float(loss.detach()),
            "residual_rms": float(((action - base_flat).detach() ** 2).mean().sqrt()),
            "q_mean": float(q_a.detach().mean()),
            "q_min": float(q_a.detach().min()),
            "bc_filter_rate": float(bc_keep.mean()),
        }
```

In `script/lingbot_rl_train.py` `_update_from_buffer`, change the actor call to the new fields so the module keeps working until Task 5 rewrites the function:

```python
    actor_info = model.update_actor(
        sample["s"], sample["z_all"], sample["a_base_all"], sample["is_expert"], sample["mc_return"])
```

- [ ] **Step 4: Update the remaining old-signature caller**

In `test_actor_critic_one_step_update_on_random_tensors`, replace

```python
    actor_info = model.update_actor(state, noise, a_base, is_expert, target)
```

with

```python
    mc_return = torch.zeros(8, 1)
    actor_info = model.update_actor(
        state, noise.unsqueeze(1).repeat(1, K_CANDIDATES, 1, 1),
        a_base.unsqueeze(1).repeat(1, K_CANDIDATES, 1, 1), is_expert, mc_return)
```

The `model.update_critic(...)` line in that test needs no edit — its signature is unchanged.

- [ ] **Step 5: Run the model tests**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py -k "bc_filter or update_actor or one_step_update or sharpening" -v`
Expected: PASS (`test_sharpening_metrics_not_residual_rms` does not call `update_actor`; if it fails, inspect before changing anything).

- [ ] **Step 6: Commit**

```bash
git add script/lingbot_rl_model.py script/lingbot_rl_train.py tests/test_lingbot_rl.py
git commit -m "Optimize the DICE-RL actor over all stored candidates and anchor the BC-loss filter to Monte-Carlo returns."
```

---

### Task 5: Fresh minibatch per critic gradient step, and store candidate sets at collection

Fixes finding 3 (UTD reused one batch and one fixed target ten times) and finishes finding 1's data path (the rollout writes `z_all`/`a_base_all` from the K decoded candidates).

**Files:**
- Modify: `script/lingbot_rl_train.py` (`_update_from_buffer`, lines 227–237; online row construction, lines 442–453)
- Test: `tests/test_lingbot_rl.py`

**Interfaces:**
- Consumes: `n_step_target` (Task 3) and `update_actor` (Task 4) signatures; buffer fields (Task 2).
- Produces: `_update_from_buffer(model, buffer, expert_ratio, device)` performing `UTD` critic updates each on a freshly sampled batch with a freshly computed target, then one actor update on one more fresh batch. Online rows carry `"z_all": _host_array(noise)` and `"a_base_all": _host_array(a_base)` (all K candidates; `"z"`/`"a_base"` keep the executed star candidate).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_lingbot_rl.py`:

```python
def test_update_from_buffer_samples_a_fresh_minibatch_per_gradient_step():
    from script.lingbot_rl_train import _update_from_buffer

    torch.manual_seed(7)
    model = DiceResidualModel(device="cpu")
    buf = ChunkReplay(capacity=32)
    for reward, done in ((0, 0), (0, 0), (1, 1)):
        buf.add_online(_replay_row(reward, done))
    buf.finalize_episode()
    calls = []
    original = buf.sample

    def spy(batch_size, expert_ratio):
        calls.append(batch_size)
        return original(batch_size, expert_ratio)

    buf.sample = spy
    critic_info, actor_info = _update_from_buffer(model, buf, expert_ratio=0.0, device="cpu")
    assert len(calls) == UTD + 1
    assert torch.isfinite(torch.tensor(critic_info["critic_loss"]))
    assert torch.isfinite(torch.tensor(actor_info["actor_loss"]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py::test_update_from_buffer_samples_a_fresh_minibatch_per_gradient_step -v`
Expected: FAIL with `assert 1 == 11` (one sample call today).

- [ ] **Step 3: Rewrite `_update_from_buffer`**

Replace the function in `script/lingbot_rl_train.py`:

```python
def _sample_batch(buffer, expert_ratio, device):
    return _to_model_device(buffer.sample(min(BATCH, len(buffer)), expert_ratio), device)


def _update_from_buffer(model, buffer, expert_ratio, device):
    critic_info = None
    for _ in range(UTD):
        sample = _sample_batch(buffer, expert_ratio, device)
        target = model.n_step_target(
            sample["reward"], sample["done"], sample["s_next"],
            sample["z_next_all"], sample["a_base_next_all"], sample["n_steps"])
        critic_info = model.update_critic(sample["s"], sample["a"], target, sample["is_expert"])
    sample = _sample_batch(buffer, expert_ratio, device)
    actor_info = model.update_actor(
        sample["s"], sample["z_all"], sample["a_base_all"], sample["is_expert"], sample["mc_return"])
    return critic_info, actor_info
```

- [ ] **Step 4: Store the candidate sets in the online rollout row**

In the `train()` while-loop where `row` is built (currently `"z": _host_array(noise[star])` etc.), add two fields:

```python
                row = {
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
                }
```

(No change is needed in `script/lingbot_rl_data.py`: expert rows fall through to the buffer's repeat-defaults from Task 2, which is the intended semantics — a demo state has one proposal, repeated.)

- [ ] **Step 5: Run the training-loop tests**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py -k "update_from_buffer or train_mocked or train_resume or train_eval_writes or save_inference" -v`
Expected: PASS. The mocked train tests exercise the new row fields end-to-end; if a mocked `decode_candidates` returns fewer than 4 candidates, fix the mock, not the code.

- [ ] **Step 6: Commit**

```bash
git add script/lingbot_rl_train.py tests/test_lingbot_rl.py
git commit -m "Sample a fresh minibatch per DICE-RL gradient step and store all collected candidates in replay."
```

---

### Task 6: Critic-ranked best-of-N at evaluation

Fixes finding 5: `predict_action_chunk` (used by `select_action` in both the train-time eval and the 200-episode comparison eval) decodes K candidates and executes the argmax-Q one whenever a residual model is attached. The SFT-parity path (`residual_model is None`) stays single-candidate.

**Files:**
- Modify: `script/lingbot_rl_policy.py` (`ResidualLingBotPolicy.__init__`, `predict_action_chunk`)
- Modify: `script/lingbot_rl_train.py` (wire `eval_candidates` in `train()` and `evaluate()`)
- Test: `tests/test_lingbot_rl.py`

**Interfaces:**
- Consumes: `decode_candidates(batch, k)` returning `{"s": (1, STATE_DIM), "z": (k, H, A), "a_base": (k, H, A), "first_chunk": bool, ...}`; `DiceResidualModel.actor/critic`.
- Produces: attribute `self.eval_candidates` (default 4) on `ResidualLingBotPolicy`; `predict_action_chunk` selects `argmax_k min-ensemble Q(s, a_base_k + s_θ(s, z_k))` when `residual_model` is set.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_lingbot_rl.py`:

```python
def test_predict_action_chunk_executes_highest_q_candidate():
    from script.lingbot_rl_policy import ResidualLingBotPolicy

    torch.manual_seed(9)
    policy = ResidualLingBotPolicy.__new__(ResidualLingBotPolicy)
    policy.eval_candidates = 4
    model = DiceResidualModel(device="cpu")
    for parameter in model.critic.parameters():
        torch.nn.init.normal_(parameter, std=0.05)
    policy.residual_model = model
    decoded = {
        "s": torch.zeros(1, STATE_DIM),
        "z": torch.randn(4, HORIZON, ACTION_DIM),
        "a_base": mask_unused_dof(torch.randn(4, HORIZON, ACTION_DIM)),
        "first_chunk": False,
        "latents": torch.zeros(1),
    }
    captured = {}

    def fake_decode(batch, k):
        captured["k"] = k
        return decoded

    def fake_apply(dec, index):
        captured["index"] = index
        return torch.zeros(1, 12, USED_DOF)

    policy.decode_candidates = fake_decode
    policy._apply_residual_choice = fake_apply
    policy.predict_action_chunk({"task": ["t"]})
    state = decoded["s"].expand(4, -1)
    executed = apply_residual(decoded["a_base"], model.actor(state, decoded["z"]))
    expected = int(model.critic(state, executed).reshape(-1).argmax())
    assert captured["k"] == 4
    assert captured["index"] == expected


def test_predict_action_chunk_stays_single_candidate_without_residual_model():
    from script.lingbot_rl_policy import ResidualLingBotPolicy

    policy = ResidualLingBotPolicy.__new__(ResidualLingBotPolicy)
    policy.eval_candidates = 4
    policy.residual_model = None
    captured = {}

    def fake_decode(batch, k):
        captured["k"] = k
        return {"s": torch.zeros(1, STATE_DIM), "z": torch.zeros(1, HORIZON, ACTION_DIM),
                "a_base": torch.zeros(1, HORIZON, ACTION_DIM), "first_chunk": False}

    def fake_apply(dec, index):
        captured["index"] = index
        return torch.zeros(1, 16, USED_DOF)

    policy.decode_candidates = fake_decode
    policy._apply_residual_choice = fake_apply
    policy.predict_action_chunk({"task": ["t"]})
    assert captured["k"] == 1
    assert captured["index"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py -k "predict_action_chunk" -v`
Expected: FAIL (`eval_candidates` unused; current code always decodes k=1 and picks index 0).

- [ ] **Step 3: Implement**

In `script/lingbot_rl_policy.py`, add to `ResidualLingBotPolicy.__init__` after `self._critic_cache_ready = False`:

```python
        self.eval_candidates = 4
```

Replace `predict_action_chunk`:

```python
    @torch.no_grad()
    def predict_action_chunk(self, batch, **kwargs):
        k = self.eval_candidates if self.residual_model is not None else 1
        decoded = self.decode_candidates(batch, k=k)
        index = 0
        if self.residual_model is not None and decoded["a_base"].shape[0] > 1:
            a_base = decoded["a_base"].float()
            noise = decoded["z"].float()
            state = decoded["s"].float()
            if state.shape[0] != a_base.shape[0]:
                state = state[:1].expand(a_base.shape[0], -1)
            executed = apply_residual(a_base, self.residual_model.actor(state, noise))
            index = int(self.residual_model.critic(state, executed).reshape(-1).argmax())
        return self._apply_residual_choice(decoded, index)
```

In `script/lingbot_rl_train.py`:
- in `train()`, directly after `policy.residual_model = model` add `policy.eval_candidates = config.k_candidates`;
- in `evaluate()`, directly after `policy.residual_model = model` add `policy.eval_candidates = config.k_candidates`.

- [ ] **Step 4: Run the policy tests**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py -k "predict_action_chunk or policy_class or commit_executed" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add script/lingbot_rl_policy.py script/lingbot_rl_train.py tests/test_lingbot_rl.py
git commit -m "Select the highest-value candidate at DICE-RL evaluation time with critic-ranked best-of-N."
```

---

### Task 7: Bump the recipe to version 2 and run full verification

The recipe fingerprint must reflect the new algorithm so old resume files and expert-feature caches cannot silently mix with v2 runs, and the pinned-protocol test must document the new fields.

**Files:**
- Modify: `script/lingbot_rl_config.py` (`protocol()`)
- Test: `tests/test_lingbot_rl.py` (`test_config_pins_released_libero_sampler_and_step_600`)

**Interfaces:**
- Produces: `protocol()["version"] == 2` plus keys `multi_sample_candidates`, `bc_filter_anchor`, `utd_sampling`, `actor_final_init`, `eval_best_of_n`. `load_resume` already rejects any fingerprint mismatch, so v1 resumes fail loudly with "Resume recipe fingerprint mismatch" — that is the desired behavior, add nothing.

- [ ] **Step 1: Extend the pinned-protocol test (failing first)**

In `test_config_pins_released_libero_sampler_and_step_600`, add:

```python
    assert proto["version"] == 2
    assert proto["multi_sample_candidates"] == 4
    assert proto["bc_filter_anchor"] == "mc_return"
    assert proto["utd_sampling"] == "fresh_minibatch_per_step"
    assert proto["actor_final_init"] == "zeros"
    assert proto["eval_best_of_n"] == 4
```

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py::test_config_pins_released_libero_sampler_and_step_600 -v`
Expected: FAIL (`version` is 1, keys missing).

- [ ] **Step 2: Update `protocol()`**

In `script/lingbot_rl_config.py`, change `"version": 1,` to `"version": 2,` and add after the `"residual_input": "z",` line:

```python
            "multi_sample_candidates": self.k_candidates,
            "bc_filter_anchor": "mc_return",
            "utd_sampling": "fresh_minibatch_per_step",
            "actor_final_init": "zeros",
            "eval_best_of_n": self.k_candidates,
```

- [ ] **Step 3: Run the full suite and compile check**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py`
Expected: all tests pass.

Run: `.cache/eval-venv/bin/python -m compileall -q script/lingbot_rl_buffer.py script/lingbot_rl_config.py script/lingbot_rl_data.py script/lingbot_rl_model.py script/lingbot_rl_policy.py script/lingbot_rl_train.py`
Expected: no output (success).

Also confirm the SFT suite is untouched: `.cache/sft-venv/bin/python -m pytest -q tests/test_lingbot_sft.py`
Expected: all tests pass (no shared code changed, this is a guard).

- [ ] **Step 4: Commit**

```bash
git add script/lingbot_rl_config.py tests/test_lingbot_rl.py
git commit -m "Bump the DICE-RL recipe to version 2 for the paper-fidelity training changes."
```

---

## Relaunch notes (operator steps, not part of code execution)

- The v2 recipe cannot resume the v1 run (`Resume recipe fingerprint mismatch` by design). The old `expert_features.pt` cache stays valid: its fingerprint covers only sampler/normalization fields, and cached rows lacking the new candidate fields are filled by the buffer's repeat-defaults at ingest, which is the intended expert semantics.
- Launch a fresh run name (e.g. `libero30-dice-v2`) via the existing Modal launcher; same 100,000-env-step budget is fine for a first comparison, since the v1 run peaked well before that.
- After training, evaluate **every** `train_eval/step_*/residual.pt` (saved by the Task 0 commit) with the pinned 200-episode protocol via `python -m script.lingbot_rl_train eval --residual-path <...>`, not only the final checkpoint. The spec's headline failure was evaluating only a post-peak endpoint.
- Watch `bc_filter_rate`, `residual_rms`, and `q_mean` on W&B; sustained growth of all three late in training was the v1 drift signature and should be gone.
- Replay rows grow by ~4 candidate tensors (~30 KB/row, ~400 MB at the ~12k rows a 100k-step run collects) — no capacity change needed.
