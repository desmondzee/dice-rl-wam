# Single-task DICE-RL on LingBot-VA (LIBERO-10 tasks 0 and 4) — design

Date: 2026-09-22
Status: draft for review

Repository: `dice-rl-wam`. Reference: DICE-RL, arXiv 2603.10263v2 (§4, §5.1, Appendix A–C).

## 1. Goal

Establish whether DICE-RL residual finetuning consistently improves the frozen
`libero30-sft` step-600 prior on a **single LIBERO-10 task**, at a per-task
interaction budget comparable to the paper, under an evaluation protocol whose
initial states the policy never trained on. Two independent runs, tasks 0 and
4, one H100 each, on Brev. Success is a paired, held-out improvement over the
SFT baseline on both tasks; a null on either is a finding, not a failure of the
phase.

## 2. Findings that motivate this phase

1. **Every previous RL training episode started from LIBERO init state 0.**
   `script/lingbot_rl_train.py:403` builds a fresh `LiberoEnv` each episode;
   `make_env` (`:34`) uses the default `episode_index=0`, and `LiberoEnv` sets
   `init_state_id = episode_index` at construction (LeRobot
   `envs/libero.py:175`). `env.reset(seed=...)` seeds MuJoCo, not the init
   index. The 200-episode protocol evaluates init states 1–20, none of which
   were ever visited. The train-time eval (`_train_eval`, `:248`) uses
   `stage="smoke"` → offset 0 → the trained-on state, which is why it read
   8–9/10 while held-out success sat at or below baseline.
2. **Budget was ~1–2% of the paper's per-task interaction.** Paper: Tool Hang
   45% → >90% in ~2,000 online episodes single-task; LIBERO-10 multitask run
   is ~1.15M env steps (≈115k, ≈330 episodes per task); Robomimic `T_ratio`
   alone is 160k–640k env steps. v2: 100k env steps over ten tasks, ≈30
   episodes per task.
3. **Tasks 0 and 4 are among the paper's biggest LIBERO movers.** Fig. 12:
   LIVING_ROOM_SCENE2 (soup + sauce, task 0) 40 → 70%; LIVING_ROOM_SCENE5 (two
   mugs, task 4) 52 → 78%. Every task starting ≥78% stays flat for the whole
   run. Our SFT: task 0 = 12/20, task 4 = 13/20.
4. **Multi-sample K is second-order.** Fig. 15: K ∈ {1, 4, 16} indistinguishable
   on Can/Square; on Tool Hang K=16 ends ~91%, K=4 ~85%, K=1 ~78%. It never
   changes curve shape and does not explain a dip below baseline.
5. **Fresh latent resampling per gradient step is not feasible on a 5B WAM.**
   π_pre(s, z) needs the AR video KV context + 20-step CFG video denoise +
   50-step action denoise (~3.7 s per state measured at k=1). Batch 256 × UTD
   10 × ~41k chunks ≈ 10⁵ GPU-hours. Storing post-video KV caches per replay
   row (GB per state), rebuilding context from stored frames (~same cost), or
   refreshing a small random subset in the background (touches ~20% of rows
   once) do not change this. Stored candidates are accepted.

## 3. Decisions (locked)

| decision | value |
|---|---|
| tasks | 0 and 4, one run each, same config |
| online budget | 660,000 env action steps per run (≈1,670 / ≈1,880 episodes at 395 / 351 mean steps) |
| training initial states | **procedural** (`init_states=False`, `hard_reset=True`): BDDL placement sampler on every reset |
| held-out eval set | all 50 canonical LIBERO init states × 2 policy seeds = **100 episodes** per checkpoint (collaborator's floor: 50; paper's LIBERO count: 100) |
| SFT baseline | same 100-episode protocol, k=1, existing step-600 prior, run on Modal |
| K candidates | 4 (unchanged) |
| prior / D_demo | step-600 multitask SFT, unchanged; D_demo = the 30 SFT demos of the task |
| compute | Brev on-demand H100 per task, state on instance disk, rsync down; evals on Modal |
| parallel envs | 1 (follow-up, see §11) |
| stop rule | held-out curve flat at 330k (3× the paper's per-task LIBERO exposure) → kill the run |

## 4. Non-goals

- Re-SFT at 50 demos/task or any SFT checkpoint sweep.
- K=16, fresh-latent resampling, or any candidate-refresh mechanism.
- Multi-task RL. Extending beyond tasks 0 and 4.
- Parallel or batched environments; actor/learner process split.
- Changing the inference sampler (20/50 full-video) or the residual/critic
  architecture. Critic-state changes only via the gated probe in §8.
- Hyperparameter sweeps (β, ε, n-step, UTD). One config per task.

## 5. Evaluation protocol (version 2)

### 5.1 Episode identity

An eval episode is `(suite, base_seed, task_id, init_state_id, policy_seed_index)`.
`episode_plan` keeps its SHA-256 seed derivation and adds the seed index, so
`policy_seed_index=0` reproduces today's seeds bit-for-bit: canonical states
1–20 at index 0 are the exact episodes behind the 12/20 and 13/20 numbers.

### 5.2 Held-out set

Init states 0–49 (offset 0) × seed indices {0, 1} = 100 episodes per task.
Training never uses canonical states (procedural inits only), so the entire
set is held out. Every arm (SFT k=1, RL best-of-4, RL k=1) runs the identical
100 `(state, seed)` pairs → paired comparison.

### 5.3 Arms and cadence

| arm | when | episodes |
|---|---|---|
| SFT baseline, k=1 | once, before RL | 100 per task |
| RL checkpoint, best-of-4 (critic argmax, as in training) | every 80k env steps plus the final step: 0, 80k, …, 640k, 660k (10 points) | 100 per checkpoint |
| RL checkpoint, k=1 (residual only) | at 320k and final | 100 |

Checkpoint 0 (zero residual, untrained critic, best-of-4) is the step-0 gate:
it must land within the SFT baseline's CI. Below it → best-of-N is harmful
cold, and the run is stopped before spending the budget (see §10).

### 5.4 Reporting

Per checkpoint and task: held-out success with a Wilson 95% interval; paired
difference vs SFT with an exact McNemar p-value and a bootstrap CI over the
100 pairs; the k=1 vs best-of-4 difference where both exist. Curves, not
endpoints. Primary endpoint is the **final** checkpoint; no selection on the
held-out numbers. Inline train-eval (§7) may be used to pick a secondary
"best" checkpoint, declared before its held-out eval is opened.

### 5.5 Inline train-eval (validation, not test)

10 episodes on fresh procedural inits, 1 seed, best-of-4, every 80k env
steps, from the training process. Logged to W&B as `train_eval/*`. Disjoint
from the canonical 50 by construction. This replaces the current 1-episode-
per-task train-eval on init state 0.

## 6. Recipe (RL protocol version 3)

Unchanged from v2 unless listed.

| field | v2 | v3 | reason |
|---|---|---|---|
| `task_ids` | all 10, random per episode | `[t]`, single task | collaborator: task-specific RL |
| training inits | canonical state 0 (bug) | procedural (`init_states=False`) | §2.1; paper's Robomimic held-out regime |
| `online_env_steps` | 100,000 | 660,000 | §2.2 |
| `rlpd_t_ratio` | = `online_env_steps` | 320,000 | paper: decay spans the warm-start only (Transport/Square) |
| truncation at 520 | terminal | `done=0`, bootstrap from the final observed state | paper silent; matters at 2,000 episodes |
| `train_eval_every` | 25,000 | 80,000 | matches checkpoint cadence |
| `train_eval_episodes_per_task` | 1 on state 0 | 10 on procedural inits | §5.5 |
| checkpoints | every train-eval point | every 80k, `residual.pt` + resume state | eval cadence, transport |
| D_demo | 300 demos, all tasks | the task's 30 manifest demos | single-task RLPD, paper's D_demo = pretraining set |
| `version` | 2 | 3 | |

Unchanged: K=4 with all 4 in Eq. 4/5, β=100, ε=−0.5, n-step 3, γ=0.99, UTD 10,
τ=0.01, Adam 1e-4, batch 256, ensemble 10, replay 100k, RLPD 0.5 → 0.1,
zero-init actor, MC-return BC anchor, best-of-4 at collection, 20/50 sampler,
3072-d critic state.

`RLConfig.validate` currently hard-errors on anything but the v2 constants;
those become validated ranges (`online_env_steps ≥ 1`, `rlpd_t_ratio ≤
online_env_steps`, `task_ids ⊆ 0..9` non-empty, `train_eval_every ≤ online_env_steps`); a
checkpoint is always written at the final step whether or not the cadence
divides the budget. `k_candidates` stays pinned to 4.

## 7. Code changes

Files under `script/` unless noted.

**Phase A — Brev training** (built and smoked first): `lingbot_rl_config.py`,
`lingbot_rl_train.py`, `lingbot_rl_data.py`, `lingbot_rl_buffer.py`, the
`brev/` scripts, and their tests. Nothing in Phase A depends on the eval
protocol change; the inline train-eval only needs `init_states=False`.

**Phase B — Modal evaluation** (built while the Brev runs are in their first
80k interval): `lingbot_eval_config.py`, `lingbot_eval.py`,
`lingbot_rl_modal.py`, `evaluate`, `analysis/`. The SFT baseline and the
checkpoint-0 gate must land before either run reaches 80k (~9 h in).

- **`lingbot_eval_config.py`** — `EvalConfig` gains `task_ids`,
  `episodes_per_task`, `initial_state_offset`, `policy_seeds` as explicit
  fields (stage presets: `smoke`, `eval` = legacy 20/offset-1/1-seed,
  `heldout` = 50/offset-0/2-seeds). `episode_plan` emits one entry per
  `(init_state_id, seed_index)`; seed derivation unchanged at index 0.
  Protocol `version` → 2 for `heldout`.
- **`lingbot_eval.py`** — `run_episode` and result aggregation carry
  `seed_index`; task filtering from config. The 100-episode SFT baseline runs
  through the existing Modal eval launcher with `--stage heldout --tasks 0,4`.
- **`lingbot_rl_config.py`** — fields from §6; `rlpd_expert_ratio` decays over
  `rlpd_t_ratio` then holds; `protocol()` version 3.
- **`lingbot_rl_train.py`**
  - `make_env(task_id, suite, init_states)`; the training env is built once
    per run with `init_states=False` and reused across episodes.
  - task draw removed; `tasks` filtered to `config.task_ids`.
  - episode end: `done=1` only on success or env `terminated`. Today `:443`
    also sets it on `truncated`, `episode_length >= 520`, and budget
    exhaustion; those rows keep `done=0` and bootstrap from the final
    observed state (the buffer already points the tail's `s_next` at the
    last row). `finalize_episode` already runs after every episode (`:478`), so the only
    change is which rows carry `done=1`.
  - `_train_eval` → procedural inits, `train_eval_episodes_per_task`
    episodes, best-of-4.
  - checkpoint every `train_eval_every`: `checkpoints/step_XXXXXX/residual.pt`
    plus `resume/latest.pt`, and at the final step; if `DICE_SYNC_CMD` is set
    in the environment it runs after each save (non-fatal on failure).
  - `evaluate` accepts a `stage`, `task_ids`, and `k` (1 or 4) instead of
    asserting the legacy protocol; writes `eval/<stage>-k<k>/`.
- **`lingbot_rl_data.py`** — `load_manifest_episodes` filters by `task_ids`
  (30 episodes for one task, not 300); fingerprint includes the task set.
- **`lingbot_rl_buffer.py`** — no schema change. `finalize_episode` respects
  the `done=0` truncation tail. Test for the truncation target.
- **`lingbot_rl_modal.py`** — `run_eval` accepts a residual path uploaded to
  the results volume and the eval stage/k; no training on Modal this phase.
- **`brev/setup.sh`** — idempotent: apt deps (as in `build_image`), `uv`,
  LeRobot at `LEROBOT_REVISION` with the same `uv sync` extras and exported
  constraints, `wandb` login from `WANDB_API_KEY`, HF login from `HF_TOKEN`,
  base model + LIBERO assets + the task's demo latents from HF into
  `/data/cache`, SFT step-600 into `/data/sft` (from the Modal volume via
  `modal volume get`, or `scp` from the local copy), `LIBERO_CONFIG_PATH` /
  `MUJOCO_GL=egl`, then `--max-env-steps 32` smoke.
- **`brev/train.sh`** — starts (or attaches to) a named tmux session running
  `python -m script.lingbot_rl_train train --config-json ...`, adding
  `--resume` when `resume/latest.pt` exists; W&B run id restored from the
  resume payload. Logs tee'd to `/data/runs/<run>/train.log`.
- **`brev/sync.sh`** — called by the sync hook and at exit: `modal volume
  put` the new `residual.pt` to the results volume so the Modal eval can
  start (Phase B); nothing else leaves the box automatically.
- **`brev/pull.sh`** (runs on the Mac) — `pull.sh <host> <run> [--resume]`
  rsyncs `checkpoints/`, `train_eval/`, `train.log`, and the run's config
  from the instance into `result/brev/<run>/`; `--resume` also pulls
  `resume/latest.pt` (several GB). Idempotent; safe to run mid-training.
  Modal-side eval results come down through the existing `--stage download`.
- **`analysis/`** (new) — paired McNemar/bootstrap over episode JSONs; curve
  plots; the per-checkpoint table in §5.4.
- **`tests/`** — episode-plan seed-compatibility at index 0; procedural env
  construction; truncation target; single-task manifest filtering; RLPD
  `t_ratio` schedule; config validation ranges.

Secrets never enter the repo: `WANDB_API_KEY`, `HF_TOKEN`, Modal token are
read from the Brev instance environment.

## 8. Gates (kept lean)

1. **Brev smoke** (~15 min after `setup.sh`): `--max-env-steps 32` train on
   procedural inits, no train-eval, a checkpoint written, the sync hook
   fires, the W&B run is visible. Two consecutive resets must produce
   different object placements (asserted in the smoke, not a separate probe).
2. **Step-0 gate** (Phase B, on Modal): the checkpoint-0 held-out eval (§5.3)
   must land within the SFT baseline's CI before the run passes 80k.

Dropped as up-front gates (see §11): the procedural-init distribution check
and the critic-state linear probe. Both are cheap and are run only if the
decision rules in §10 point at them.

## 9. Diagnostics logged per checkpoint

`bc_filter_rate`, `residual_rms`, `q_mean`, `q_min`, critic loss, ΔV and ΔH
on 10 procedural states, RLPD ratio, replay size, episodes collected, mean
episode length, train-eval success. All on W&B under the run's name; the
same numbers are in `train_eval/step_XXXXXX/*.json` for offline analysis.

## 10. Decision rules

| observation | reading | next |
|---|---|---|
| held-out climbs and stays above SFT on both tasks | recipe works; init-state collapse was the story | extend to tasks 8/9, consider parallel envs |
| checkpoint 0 below SFT CI | best-of-N harmful with a cold critic | stop; warm the critic before enabling selection |
| flat, `bc_filter_rate` ≈ 0 | residual pinned to the prior by construction | β / ε are the lever |
| flat, filter fires, ΔH does not fall | critic cannot discriminate | run the critic-state linear probe (§11); add proprio + history if R² is low |
| climbs on one task only | task-dependent; not a recipe failure | report both; choose a third task by §2.3 |
| flat at 330k | stop rule | kill; do not spend the second half |

## 11. Follow-ups explicitly deferred

- Parallel collection: the policy wrapper is a single AR stream with KV
  snapshots (`_snapshot_kv`, `_init_streaming_cache`); batching streams is
  wrapper surgery, and time-sliced collector processes are a smaller win than
  they look. Revisit if wall-clock (not $) becomes binding.
- K=16 at collection (memory headroom exists: 13.7 GB peak at k=1).
- Re-SFT at 50 demos/task with the full checkpoint ladder (paper Fig. 4).
- Critic-state linear probe (frozen 3072-d state → MC return-to-go on the
  task's demos, ~1 GPU-h) and, if R² < 0.3, proprioception + frame history
  in the critic state (replay-schema change).
- Procedural-init distribution check against the canonical 50, if a
  held-out/train-eval gap appears that the curves cannot explain.

## 12. Cost and timeline

| item | estimate |
|---|---|
| SFT baseline, 2 tasks × 100 episodes, Modal | ~4.6 GPU-h, ≈ $20 |
| RL training, per task, Brev on-demand H100 | ~73 h at the measured 0.396 s/env-step (v2), 660k steps |
| checkpoint evals, per task, Modal | 10 × best-of-4 + 2 × k=1 = 1,200 episodes ≈ 30–35 GPU-h, ≈ $130 |
| data down per task | 9 × 485 MB `residual.pt` + resume state (replay ≈ 3 GB at K=4) + JSON |

Wall clock: Phase A code + Brev smoke, then both runs start; Phase B is
built during the first interval and the baseline + step-0 evals land before
80k (~9 h). Runs take ~3–4 days in parallel, evals trailing each checkpoint
by ~3 h; `brev/pull.sh` at the end.
