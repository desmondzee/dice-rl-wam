# Resumable residual training: design

Date: 2026-10-23. A bounded change to milestone 2's agent subclass (`script/dicerl_libero_rl.py`), required before any further residual run (an extension of the current runs, the control's successor, or the layout axis). Spec of record for the milestone: `docs/superpowers/specs/2026-10-22-libero-residual-robot-init-design.md`.

## Intent

A residual run must be continuable from its last checkpoint in a fresh process with the same training trajectory it would have had uninterrupted. The vendored DICE-RL checkpoint (`model_step_<N>.pth`: weights, optimisers, schedulers, step) is enough to evaluate or warm-start but not to continue: the replay buffer is never written, and the vendored training loop keeps its iteration counter as a local variable that always starts at zero, which drives the warm-ups, the expert-ratio schedule, the shift ramp and the W&B step. Today the only way to extend a run is never to stop it.

## Goals

- `ShiftedResidualAgent` saves a full training state beside every `model_step_<N>.pth` and can resume from it: replay buffer contents, the iteration counter, the shift ramp's magnitude and the per-worker shift seeds and reset counts, the expert-ratio schedule position (a function of the counter), the numpy and torch RNG states, and the W&B run id for continued logging.
- A resumed process reproduces the uninterrupted run's schedule exactly (same magnitude at every iteration, same warm-up boundaries, same evaluation and save iterations) and continues the replay buffer rather than refilling it.
- The vendored files stay untouched; the subclass owns a copy of the training loop with a resume offset, and that copy is the only vendored logic we carry ourselves.

## Non-goals and anti-goals

- No bit-exact reproduction of environment physics or policy sampling after a resume (the simulator is not bitwise reproducible across resets, as gate A showed); the contract is the schedule, the buffer and the optimiser state, checked statistically.
- No change to the vendored loop's semantics beyond the offset: evaluation cadence, save cadence, the single-sample warm-up, the Q-filter warm-up, the expert-ratio decay and the update gating stay as released. Any divergence between the copied loop and the vendored one is a defect, enforced by a line-by-line diff recorded in the review.
- No retrofit of the two live runs; their processes hold the old class.
- Anti-goal: a resumed run that silently restarts its ramp or warm-up. Resume fails loudly when the state file is missing or its counter disagrees with the checkpoint's step.

## Design

### State file

`state_step_<N>.pt` next to `model_step_<N>.pth`, written by `save_state(step)` immediately after the vendored `save_model(step)` runs (the subclass overrides `save_model` to call the parent then `save_state`). Contents:

- `step` (the iteration counter), `env_steps` (the W&B step), `wandb_run_id`;
- `replay`: the buffer's tensors truncated to `size` (`chunk_states`, `chunk_actions`, `chunk_rewards`, `chunk_next_states`, `chunk_dones`, `chunk_mc_returns` when present) moved to CPU, plus `ptr`, `size`, `num_episodes`; in-flight partial episodes (`ongoing_episodes`) are dropped, because the resumed process resets every environment before stepping, and the drop is logged with the number of partial episodes discarded;
- `shift`: the current magnitude and the per-worker `(shift_seed, reset_count)` pairs read back through `call_sync("shift_state")` (a new wrapper method returning both);
- `rng`: `np.random.get_state()`, `torch.get_rng_state()`, `torch.cuda.get_rng_state_all()`;
- `completed_episode_stats`: the loop's rolling `completed_episodes_rewards` and `completed_episodes_lengths` lists (100 entries at most), so logging continues without a gap.

Written atomically (temporary file in the run directory, then `os.replace`). Size is about 1 GB at 620k transitions (feature vectors, not images), which is why it is written at the checkpoint cadence (every 1,000 iterations) and not more often.

### Resume

The launcher gains `--resume <run_dir>`: it finds the latest `model_step_<N>.pth` that has a matching `state_step_<N>.pt`, passes `resume_from=<run_dir>` and `resume_step=N` as hydra overrides, reuses the run directory's `.hydra/config.yaml` (no config edits on resume) and restarts the watcher on the same run directory. The agent, when `cfg.resume_from` is set: loads the model checkpoint through the vendored loader (weights, optimisers, schedulers), loads the state file, restores the buffer tensors into the buffer's preallocated storage, restores the RNG states, sends the per-worker shift seeds and reset counts to the workers (`call_sync_arg("set_shift_state", ...)`) and the magnitude, resumes the W&B run by id, and runs the loop from `step`. The in-training evaluation at the resume iteration is skipped (it already ran before the save).

### The loop copy

`ShiftedResidualAgent.run()` is a copy of the vendored `run()` from `agent/finetune/train_distill_residual_flow_agent.py` (lines 871–1054 at commit 64f5593) with exactly these differences, each marked in the review diff: `total_steps` starts at `self.resume_step`; the initial `reset_env_all` and buffer warm-start happen as before; the vendored expert-trajectory bookkeeping at the top of `run()` is kept; the body is otherwise identical. The subclass's `update_networks` override (ramp and shift logging) is unchanged.

### Gate

`script/dicerl_libero_gate_resume.py`, run on the box with the smoke config (`train.num_train_steps=300`, `save_freq=100`): run A trains 300 iterations uninterrupted; run B trains to 100, is stopped by its session, and is resumed to 300. Checks: B's state file at 100 reloads with the same buffer size, pointer and counters; B's logged shift magnitude and expert ratio at iterations 101–300 equal A's; B's evaluation and save iterations equal A's; B's buffer size at 300 equals A's; the W&B run of B continues under one id. The in-training success at 300 is reported for both but not compared (sampling noise).

## Builder decisions

Left to the builder: the exact attribute list read from `HybridReplayBuffer` (verified against `util/hybrid_replay_buffer.py` at implementation time, with an assertion that every tensor attribute of the buffer is either saved or explicitly listed as derived), the wrapper's `shift_state`/`set_shift_state` method shapes, how W&B resume is passed (`wandb.init(id=..., resume="must")` through the config the vendored agent reads, or the subclass re-initialising), the launcher's resume argument parsing, and the unit tests in `tests/test_dicerl_libero_rl.py` (state save/load round trip on a fake buffer; resume refuses a mismatched step; the loop copy's offset; the launcher's latest-resumable-checkpoint selection on a fake run directory).

Code rules: no comments or docstrings; lean; vendored files untouched; commit without attribution trailers; never `pkill -f`; the gate runs on an idle box only.
