# DICE-RL port audit: LingBot-VA port vs the vendored reference

Date: 2026-09-23. Branch `single-task-dice-rl-brev`. Audit only; no code changed.

Reference = `model/rl/distill_residual_rl.py`, `model/rl/distill_residual_rl_img.py`,
`agent/finetune/train_distill_residual_flow_agent.py` (run loop, inherited by the img agent),
`agent/finetune/train_distill_residual_flow_img_agent.py`, `util/hybrid_replay_buffer.py`,
`env/gym_utils/wrapper/multi_step_full.py` (the agent swaps `multi_step` for it), `env/gym_utils/wrapper/robomimic_image.py`,
`cfg/robomimic/finetune/tool_hang/ft_distill_residual_flow_unet_img.yaml` (TH) and
`cfg/robomimic/finetune/transport/ft_distill_residual_flow_unet_img.yaml` (TR; the config closest to ours: n_envs 4, n_step 3, ratio anneal over 320k env steps).

Ours = `script/lingbot_rl_model.py`, `script/lingbot_rl_buffer.py`, `script/lingbot_rl_train.py`, `script/lingbot_rl_config.py`,
`script/lingbot_rl_data.py`, `script/lingbot_rl_policy.py`, line numbers from the working tree at audit time
(it carries an uncommitted GPU-ring replay refactor in buffer/train/model; the algorithm is unchanged by it).

## Units

The reference counts `training_step` in vectorised env steps: one step = `n_envs` chunks of `act_steps = 8` env steps
(64 env steps on TH, 32 on TR). Every warmup and schedule below is converted to env steps with that factor.
Our chunk is 16 env steps (12 on the first chunk) and `n_envs` slots run in lockstep.

| Reference knob | Value | TH env steps | TR env steps | Ours |
|---|---|---|---|---|
| `replay_flow_warmup_steps` (no critic selection) | 3000 | 192k | 96k | 0 |
| `q_filtering_warmup_steps` (no BC filter) | 4500 / 4000 | 288k | 128k | 0 |
| `adaptive_expert_ratio` start→end over steps | 0.5→0.1 / 0.9→0.1 over 10000 | 640k | 320k | 0.5→0.1 over 320k |
| `gradient_steps` per training step | 10 | 10 / 64 = 0.16 per env step | 10 / 32 = 0.31 per env step | 10 / 16 = 0.63 per env step |
| actor updates per training step | 5 (bursts of 10 on even steps) | 1 actor : 2 critic | 1 : 2 | 1 : 10 |
| Polyak updates per training step | 5 (bursts, tau 0.01) | tau_eff 0.005 per critic step | 0.005 | 0.01 per critic step |

## Table

Overest. = does the difference plausibly cause or worsen the observed critic overestimation.

| # | Detail | Reference | Ours | Verdict | Overest. |
|---|---|---|---|---|---|
| 1 | TD target | `loss()` `distill_residual_rl.py:993-1032`: K=4 fresh z at s', `a' = π_pre(s',z) + r_θ(s',z)` with the current actor, `target_critic` returns the ensemble **min** (`:163-202`), then **mean over K**; `y = r + γ^n (1-d) Q'`; no clipping | `n_step_target` `lingbot_rl_model.py:102-113`: same formula, min then mean over the K=4 candidates stored at s' (`z_next_all`, `a_base_next_all`) | SAME (modulo row 17) | no |
| 2 | Critic loss | `critic_loss` `:840-916`: MSE per head against the shared target on the stored action, summed over 10 heads; `data_source` never passed from `loss()` (`:1038`) so `disable_td_loss_for_expert_data` is inert; `critic_weight` 1 | `update_critic` `:115-127`: identical | SAME | no |
| 3 | Done at horizon | `multi_step_full.py:100-112`: `truncated` at `max_episode_steps` is folded into `done`; buffer `hybrid_replay_buffer.py:333-335` marks the chunk done → no bootstrap. Truncation is terminal | `lingbot_rl_train.py:545` done at success, env terminal, or 520 steps; `truncation: terminal` in `protocol()` `lingbot_rl_config.py:79` (since a0bdadd) | SAME | no (Run A's self-loop was this row; now fixed) |
| 4 | n-step semantics | Buffer `:410-460`: walks upsampled chunks `idx+k`, k < (n-1)·8+1, discount `γ^((k+7)//8)` (chunk-level γ), stops after a done chunk, `s_next` = chunk `idx+(n-1)·8`'s next state, `n_steps` constant n; `use_n_step` true, n=5 (TH) / 3 (TR); expert rows same in the dataset `agent/dataset/sequence.py:317-351` | `finalize_episode` `lingbot_rl_buffer.py:92-98`: `Σ_{lag<n} γ^lag r`, `done_n = any(dones)`, successor = `idx+n` (clamped to the last row, masked by done), `n = min(3, remaining)`; `γ^n` in the target | SAME (n=3 = TR) | no |
| 5 | Reward scale / discount unit | `robomimic_image.py:265-297` sparse reward 1 per success step, `success_steps_before_termination: 1` (cfg `:41`) → exactly one reward of 1 then terminal; chunk reward = within-chunk sum (`:316-318`); γ=0.99 per **chunk** of 8 env steps | `lingbot_rl_train.py:541-547`: 1.0 once at first success then terminal; γ=0.99 per chunk of 16 env steps | SAME bound (true Q ≤ 1); per-env-step discount differs (chunk length) | no |
| 6 | Ensemble reduction: target / actor / selection | min-then-mean(K) / min (`return_mean` forced False, `:163`) / max over samples of min (`img:199-206`) | min-then-mean / min (`:139`) / argmax of min (`train:516-517`) | SAME | no |
| 7 | Actor Q term | `actor_loss` `:774-784`: `-mean_k Q(s,a_k)` masked to online rows (`disable_q_loss_for_expert_data: true`), then `.mean()` over the **full batch B** (masked rows contribute 0, so the term is scaled by the online fraction) | `update_actor` `:143-144`: `-(mean_k Q · online).sum() / online.sum()` (averaged over online rows only) | DIFFERENT | weak yes: at ratio 0.5 our Q term carries 2× the reference weight relative to BC |
| 8 | Actor Q normalisation | `:786-804`, cfg `use_q_normalization: true` ("this has to be true"): Q term divided by `mean |Q|` over the online (B,K) samples, detached | none | DIFFERENT | **yes**: the reference actor gradient is `∂Q/|Q|`; without it the actor's pull toward critic-favoured actions grows in proportion to Q, so an inflating critic recruits a more exploiting actor, which feeds the next-state target; a positive feedback the reference lacks |
| 9 | Actor BC term | `:806-818`: `β · mean_{B,K}(mean_{H,A}(a_k − a_pre,k)²)`, β=100, uniform weights over K | `:145-147`: same MSE, β=100 | SAME | no |
| 10 | K at actor update | `:649-671`: K=16 **fresh** z per state, `a_pre` recomputed by the frozen flow policy for each | K=4 candidates stored at collection (`z_all`, `a_base_all`), reused on every sample | DIFFERENT, LingBot-specific (WAM decode is the cost) | no: the actor maximises over a smaller, fixed set; if anything less exploitation |
| 11 | BC filter (soft Q filter) | `:746-772`: needs `use_soft_q_filtering` — **false in both configs** (`TH:162`), so `bc_filter ≡ 1` and BC is never relaxed in the reference runs. When enabled: `better_k = Q(s,a_k) > Q(s,a_pre,k)` AND `q_overestimation < threshold` where `q_overestimation = Q_min(s, a_stored) − mc_return` (agent `:306-313`), thresholds −0.85 (TH) / −0.55 (TR); expert rows always keep BC (`always_retain_bc_loss_for_expert_data: true`); only after `q_filtering_warmup_steps` (`:674`) | `bc_filter_keep` `:41-44`: enabled from step 0, `better_k` same, underestimation anchored on `Q(s,a_k) − mc_return < −0.5` per candidate; no expert override; observed keep rate 1.0 | DIFFERENT | no: the filter only removes BC when Q is *below* MC; with an overestimating critic it never fires in either code base. Not a cause; the `bc_filter_rate = 1.0` symptom is also the reference's steady state |
| 12 | Self-imitation | Not implemented (only mentioned in docstrings `:640-644`, `:731`); the post-warmup loss is Q + filtered BC | none | SAME | no |
| 13 | Exploration / selection at collection | `img:134-236`: for `training_step ≤ replay_flow_warmup_steps` (3000 → 192k / 96k env steps) a **single** random z is executed with no critic; afterwards 16 samples, argmax of the ensemble-min Q (`max_q_min`) | `train:513-518`: from the first chunk, K=4 candidates, argmax of the ensemble-min Q | DIFFERENT | **yes**: best-of-K by an untrained critic selects the candidates the critic overrates (optimiser's curse); every stored `a` is a critic-argmax and the actor then maximises the same critic over the same candidates. The reference collects ~100–200k env steps of unselected π_pre + r data first |
| 14 | Selection at evaluation | `evaluate_strategy: max_q_min`, 16 samples | best-of-4 (`lingbot_rl_policy.py:501-514`) | DIFFERENT, LingBot-specific | no |
| 15 | Expert data: actions and reward | `sequence.py`: dataset actions (chunks of 8), reward 1 only at the truncated success step (`ph_finetune` / "exactly one step of success", TR `:188`), last step done=1, chunked MC return; buffer `_sample_expert_data` `:595-635` | `_episode_rows` `lingbot_rl_data.py:104-124`: `a = a_base` = expert chunk, reward 1 on the last row, done 1 on the last row, MC at finalize | SAME | no (expert Q ≈ MC in both runs) |
| 16 | Expert data: noise / base action in actor loss and target | No noise stored; actor loss and target draw fresh z and use `π_pre(s,z)` — on expert states the residual is regularised toward 0 over the noise distribution and the target bootstraps `Q(s', π_pre(s',z) + r)` | `z = 0`, `a_base = a_base_all = expert action` (`:106-108`, buffer `:67-70`): BC pushes `r(s,0) → 0` only at z=0; the target bootstraps `Q(s', a_expert' + r(s',0))` | DIFFERENT, LingBot-specific (no cheap decode on demo frames) | no: expert rows are calibrated (0.87 vs 0.91); worth a question to the authors |
| 17 | Next-state candidates for the target | fresh z at s' every update (`:1013-1022`) | the 4 candidates decoded at s' during collection (`z_next_all`, `a_base_next_all`); the last row of an episode points at itself, masked by done | DIFFERENT, LingBot-specific | no |
| 18 | RLPD ratio | `get_current_expert_ratio` `agent:200-222`: linear in training steps, TH 0.5→0.1 over 640k env steps, TR 0.9→0.1 over 320k; `expert_batch = int(B·ratio)` (`buffer:547-549`) | `rlpd_expert_ratio` `config:67-69`: 0.5→0.1 over 320k env steps; `round(B·ratio)` | SAME in spirit (TH start, TR duration) | no |
| 19 | Update start condition | `agent:288-289` and `:991`: no gradient step until the **online** buffer holds ≥ `batch_size` (256) upsampled chunks (~2 episodes); `_process_complete_episode` only runs at episode end | `train:567`: updates start once one online episode is finalised (`has_ready_online`, buffer `:127-128`); batch = `min(256, len(buffer))` where `len` includes the ~800 expert rows, so the online half (128 rows) is drawn from the first episode's ~15–30 rows with replacement (`_choose` `:34-37`) | DIFFERENT | **yes**: the first thousands of critic steps fit a handful of online rows replicated ~5–8× per batch at UTD 10 per chunk; classic early overfit of the sparse-failure signal |
| 20 | UTD (critic steps per collected data) | `gradient_steps: 10` per training step (`TH:81`, `agent:997`) = 10 per `n_envs` chunks = 1.25 (TH) / 2.5 (TR) per chunk; 0.16 / 0.31 per env step | `UTD = 10` per collected chunk (`_update_from_buffer` `train:253-261`, called per slot per chunk `:567-568`); 0.63 per env step (0.83 on 12-step first chunks) | DIFFERENT | **yes**: 4–8× the reference's critic UTD per chunk on the same ensemble size; high UTD with a small buffer is the standard overestimation amplifier (REDQ/DroQ literature) |
| 21 | Actor update frequency | `actor_update_freq: 2` gated on the venv-step index (`agent:355`, constant across the 10 gradient steps) → 10 actor steps on even venv steps, 0 on odd: 1 actor : 2 critic on average; `actor_total.backward()` still runs on skipped steps (`:373`, discarded) | 1 actor step per 10 critic steps (`train:262-264`) | DIFFERENT | no direct effect (ours is more conservative); port for fidelity with row 20 |
| 22 | Target update | `critic_target_update_freq: 2`, same burst gating (`:393-394`), tau 0.01 → 0.005 per critic step on average | Polyak every critic step, tau 0.01 (`update_critic` `:121`) | DIFFERENT | weak yes: the target tracks the online critic 2× faster, shortening the lag that damps bootstrapped inflation |
| 23 | Optimiser / clipping / schedule | Adam lr 1e-4, `weight_decay 1e-5` (`TH:84-89`), `max_grad_norm 1.0` on actor and critic (`TH:109`, `agent:367-368`, `:388-389`), cosine-annealing-with-warm-restarts `first_cycle_steps 1000`, `warmup 10`, `min_lr 1e-6` (`TH:92-96`, stepped per critic step / per actor step `agent:396-401`; mean lr ≈ 5e-5) | Adam lr 1e-4, no weight decay, no clipping, constant lr (`model:99-100`) | DIFFERENT | weak yes: no clipping means a single batch containing the rare reward-1 rows can take an unbounded step; the schedule halves the mean lr |
| 24 | Batch size | 256 (`TH:76`), fresh minibatch per gradient step (`agent:296-304`) | 256, fresh minibatch per step (`train:256`) | SAME | no |
| 25 | Critic input / state | `Q(s,a)`, `q_depends_on_noise: false`; s = low-dim proprio (9/18-d) + visual features from the BC-trained ResNet of the frozen prior (`img:296-346`), cond_steps 1 | `Q(s,a)`; s = 3072-d mean pool of pre-`proj_out` video tokens + text tokens of the current frame (`policy:33-36`, `:314-336`); a = 16×30 with 23 channels masked to 0 (`model:31-34`) | DIFFERENT, LingBot-specific | open: the ResNet features were trained on the task and encode progress; a frozen video-model pool may alias failing online states to demo-like states — consistent with Run B (mid-trajectory Q 0.82, MC 0.11, self-consistent TD target). Not a port bug |
| 26 | Actor input / init | `r_θ(s, z)` (`condition_residual_on_base_action: false`), default MLP init, GELU + LayerNorm 1024×3 | `r_θ(s, z)`, final layer zero-initialised (`model:62-64`), same widths | SAME (init more conservative) | no |
| 27 | Replay layout | Upsampled: a chunk at **every** env step of each episode (`buffer:298`), sampling uniform over ~8× more rows; capacity 2–4M | one row per executed chunk; capacity 100k | DIFFERENT, LingBot-specific (cannot re-decode intermediate starts) | no |
| 28 | Normalisation of Q / targets / rewards | none anywhere (Identity output, MSE, no clipping) | none | SAME | no |
| 29 | Stored `a` | executed action (prior + residual, selected sample) | executed candidate `chosen` (`train:559-561`) | SAME | no |
| 30 | Success handling mid-chunk | `multi_step_full.py:85-88` stops executing the chunk at done | `train:545-547` stops stepping the slot at done | SAME | no |

## Ranked fix list

Ordered by expected effect on the critic ratchet. Every item names its table row.

1. **Selection warmup at collection (row 13).** Execute a uniformly random one of the K=4 decoded candidates (still store all four) until `env_steps ≥ 96_000` (TR anchor; TH would be 192k), then switch to the ensemble-min argmax. Port of `replay_flow_warmup_steps`.
2. **Minimum online data before updates (row 19).** No gradient steps until ≥ 256 finalised online rows exist (reference `get_total_transitions() ≥ batch_size`), so the online half of a batch is never a handful of rows replicated.
3. **UTD in env-step units, with the reference actor/target cadence (rows 20, 21, 22).** 10 gradient steps per 32 collected env steps (TR: 2.5 per 16-step chunk) instead of 10 per chunk; actor step and Polyak (tau 0.01) on every second gradient step. Recipe fingerprint changes (`utd`).
4. **Actor Q-loss normalisation and batch-mean masking (rows 7, 8).** `q_term = −Σ_online mean_k Q / B`, divided by `mean|Q|` over online (B,K) samples (detached); keep β=100 BC.
5. **Optimiser regularisation (row 23).** Grad-norm clip 1.0 on actor and critic; Adam `weight_decay=1e-5`; cosine-with-warm-restarts lr (1000-step cycles, 10-step warmup, floor 1e-6) stepped per critic step and per actor step. Clipping is the part that matters; the schedule is fidelity.
6. **BC filter gating and anchor (row 11).** Anchor underestimation on `Q_min(s, a_stored) − mc_return < ε` per row, keep BC on expert rows unconditionally, and enable the filter only after 128k env steps (TR `q_filtering_warmup_steps`). Cannot affect overestimation; done for fidelity. Note the reference configs disable the filter outright (`use_soft_q_filtering: false`).

Keep LingBot-specific (rows 10, 14, 16, 17, 25, 27): the stored K=4 candidates for the actor loss and for the next-state target instead of fresh draws, the WAM decode and 16-step chunk, the pooled 3072-d critic state, best-of-4 evaluation, one row per executed chunk, expert rows with `z = 0` and `a_base = expert action`, the zero-initialised residual head.

## Open questions for the authors

- Row 11: `use_soft_q_filtering: false` in both shipped configs — were the paper's numbers produced without the Q filter?
- Rows 20–22: the intended UTD unit (gradient steps per vectorised step, i.e. `10 / (8·n_envs)` per env step), and whether the burst actor/target cadence (10 on even steps, 0 on odd) is deliberate or an artefact of gating on `training_step`.
- Row 25: the critic consumes BC-trained ResNet features that encode task progress; is there evidence the method survives a frozen, task-agnostic state (our pooled WAM tokens)? Run B's calibrated expert rows but uncalibrated online mid-trajectory rows point at state aliasing rather than at the update rule.
- Row 16: on expert states the reference regularises the residual toward `π_pre` samples (RLPD-style), never toward the demo action; is that the intent, and does the demo action belong in the critic only?
- Row 4: with upsampled chunks the n-step walk over `idx+k` counts each env-step reward once only because episodes hold exactly one reward; is `(k+7)//8` discounting intended for dense-reward tasks?
- Row 23: is the 1000-step cosine restart schedule load-bearing, or incidental?

## Notes for the fix phase

- The working tree holds an uncommitted replay/logging refactor (GPU ring buffer, `bc_keep_rate`/`q_base`/`q_target` metrics, `eval/` prefixes) plus a concurrent edit to `lingbot_rl_policy.py`; the branch may be reset before fixes land, so the line numbers above are for orientation, not patch anchors.
- `result/brev/dice-t0-bootstrap-diverged/resume/` currently holds a partially copied `.latest.pt.*` temp file and `dice-t0/resume/` is empty; the offline replay check can run once both `latest.pt` files land.
- `timeout` is not available on this macOS shell; run the test suite without it.
