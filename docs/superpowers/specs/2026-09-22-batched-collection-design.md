# Batched LIBERO collection for single-task DICE-RL — design

Date: 2026-09-22. Branch `single-task-dice-rl-brev`. Scope: wall-clock only; the learning recipe (K=4, best-of-4 critic argmax, UTD=10 critic + 1 actor step per collected chunk, rewards, resets, checkpoint cadence, `protocol()` fingerprint) is unchanged.

## Why the GPU idles

One collection stream drives the 5B transformer at batch 2 (video CFG) and batch 4 (action candidates). Each of the 20 video and 50 action denoise steps is a short launch-bound forward, so the H100 sits at 75–79% with dips to 9–25% between chunks; the measured rate is 0.40 s per env step including updates. Only a larger batch per forward can lift utilization.

## Strategy: batch N streams through one transformer forward

`StreamBatch(policy, n)` in `script/lingbot_rl_policy.py` runs N LIBERO episodes in lockstep and issues one forward per denoise step with batch 2N for video (rows `[cond_0..cond_{N-1}, uncond_0..uncond_{N-1}]`) and N×K for the action head (`expand_conditional_kv(transformer, k, n)` repeat-interleaves the N conditional rows). Alternatives rejected: N separate KV caches swapped per stream (no batching, no speedup); an actor/learner process split (does not raise the per-forward batch). The trainer talks to streams through one small interface — `start()`, `decode(batches, k)`, `commit(chosen, first_chunk)`, `observe(batches)`, `drop(position)`, `size` — and `open_streams(policy, n)` returns `SingleStream(policy)` for `n_envs=1`, which forwards to the exact `reset`/`decode_candidates`/`commit_executed`/`observe_env_step` calls the trainer makes today, so the single-env path is byte-for-byte the current behaviour.

## Per-stream state layout

| state | single stream today | batched |
|---|---|---|
| transformer KV cache `"pos"` | k/v `[2, T, H, D]` | k/v `[2N, T, H, D]`, created once per batch; slot arrays `mask`/`id`/`is_pred` stay shared |
| streaming VAE `feat_cache` | rows `[cam0, cam1]` | rows cam-major `[cam0_s0..cam0_s{N-1}, cam1_s0..]` |
| `_init_latent`, `_executed_actions` | batch 1 | batch N |
| `_obs_buffer` | one list | one list per stream |
| prompt embeds | `[1, L, D]` | `[N, L, D]` (per-stream task), negative shared |
| `_first_chunk`, `_frame_st_id`, `_exec_step`, `_prev_j` | scalars | scalars shared by the batch (lockstep) |
| critic cache `"critic"` | batch 1 | re-created at the needed batch size; `_pool_from_latent` takes the per-stream prompt embeds |

The KV cache's slot bookkeeping (`mask`, `id`, `is_pred`) is per slot, not per batch row, so every row in a cache must be at the same chunk index. That is what forces lockstep and decides episode-end handling.

## Episode end mid-batch: drop until the sync point

When a stream's episode ends (success, terminated, 520 steps, or budget) its rows are removed from every block's k/v cache, from the VAE `feat_cache`, and from the per-stream tensors; the surviving streams keep going with a smaller batch. New episodes start only at the sync point, when every stream has ended (or the budget is hit). Re-seeding a fresh stream into a live batch would need per-row attention masks and per-row `frame_st_id`, which the read-only cache API cannot express; dropping is exact and simple. Cost: with mean episode length ≈ 380 and cap 520, slots are occupied ≈ 75% of the time; because the forwards are launch-bound the partially filled tail is still cheap. Train-eval, sharpening, and the throttled resume save run at sync points (they reset the policy and would clobber live streams); with `n_envs=1` every episode end is a sync point, so cadence is unchanged. Seeds are `config.seed + env_steps + slot` at the sync point: slots are `< n_envs` and every batch advances `env_steps` by at least `n_envs`, so seeds stay unique.

## Replay: multiple open episodes, same on-disk format

`ChunkReplay` gains a `stream` argument on `add_online`/`add_expert`/`finalize_episode` (default 0). In-progress rows live in `_open[stream]`; `finalize_episode(stream)` computes MC returns and n-step targets and appends them to `_data`. Open rows are sampled with the same bootstrapped successor views as today (all but the last row of each open episode), so nothing changes about what is sampleable, and updates start with the second chunk exactly as now. `state_dict()` writes `_data` plus stream 0's open rows under `data` with `episode_start` marking them — the layout the single-env code writes today — and `load_state_dict` moves any tail after `episode_start` back into `_open[0]`, so a replay saved by the running single-env job loads and continues. All saves in the new loop happen at sync points, where no episode is open.

## Budget

`RLConfig.n_envs` (1..8, default 1) is a config field, not a protocol field, so the recipe fingerprint is byte-identical and the live run can resume with `--n-envs N`. Memory per stream: "pos" cache ≈ 1.6 GB (2 CFG rows), the N×K expanded conditional cache ≈ 3.2 GB transient, critic cache ≈ 0.8 GB; N=4 is ≈ 25 GB on top of the 10 GB weights. Expected speedup: N=2 ≈ 1.7×, N=4 ≈ 2–2.5× after the tail effect, to be measured. Env stepping stays sequential on the main process (≈ 0.6 s per chunk for N=4 versus several seconds of decode).

## Risks

Hidden batch coupling inside the frozen VAE or transformer (none found: attention, norms, and causal convs are per row; slot arrays are shared by construction). CPU contention from N MuJoCo envs on one core. Partially filled tails at large N. Multi-task configs multiply env instances (one per slot per task).
