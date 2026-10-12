# Plan-selection probes on the GB10s

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run probes P0–P2 of the plan-selection design on the two GB10 units so that E is chosen and the across-plan spread is measured, with every code path they need built once and reused by the full implementation.

**Architecture:** A no-sudo, aarch64 setup path for the Brev scripts; a local held-out eval runner with a `video_exec_step` option; the M-plan sampler inside the residual policy (M video noises each fully denoised to E against one history cache, final-step token pooling, K decodes per plan).

**Tech Stack:** LeRobot eval environment (Torch 2.11 cu128 aarch64 wheels from the pinned lock), MuJoCo EGL, uv, tmux, rsync over the Brev tunnel.

**Spec:** docs/superpowers/specs/2026-10-09-plan-selection-design.md; method text in docs/methodology.tex.

## Global Constraints

- Zero comments and docstrings in `script/lingbot_rl_*.py`; lean code; tests must not call Modal.
- Do not edit `.cache/lerobot`. `script/lingbot_eval.py` protocol fields may gain `video_exec_step` with default −1 only; no other protocol change.
- Compute is the two GB10s only (`ssh -F ~/.brev/ssh_config watson|jarvis`, aarch64, no sudo, work under `~/Documents`). No Modal compute.
- Secrets never printed; `brev/env.sh` is written over ssh stdin from local sources.
- Large files move by rsync over ssh (checksummed), never by Modal volume downloads.
- Held-out protocol unchanged: 100 episodes = 50 canonical states × 2 seeds, identical seeds across policies.

---

### Task 1: No-sudo GB10 setup path and P0 smoke

**Files:**
- Modify: `brev/setup.sh` (skip apt when `DICE_SKIP_APT=1`; create `DICE_DATA` without sudo when its parent is writable; install uv user-space; take the SFT checkpoint from a local directory `DICE_SFT_SRC` when set instead of `modal volume get`)
- Modify: `brev/env.example` (document `DICE_DATA=$HOME/Documents/dice`, `DICE_SKIP_APT=1`, `DICE_SFT_SRC`)
- Test: `bash -n brev/setup.sh`; the smoke itself

- [ ] Step 1: Edit `brev/setup.sh`: wrap the two `sudo apt-get` lines and the sudo mkdir in `if [ "${DICE_SKIP_APT:-0}" != "1" ]`; replace the `modal volume get` of the SFT checkpoint with `rsync -a "$DICE_SFT_SRC"/ "$SFT"/` when `DICE_SFT_SRC` is set.
- [ ] Step 2: `bash -n brev/setup.sh brev/train.sh brev/sync.sh brev/pull.sh`.
- [ ] Step 3: rsync the repo (excluding .cache, result, checkpoints, wandb) to `watson:~/Documents/dice-rl-wam` and the verified SFT checkpoint (`checkpoints/lingbot-sft/libero30-sft/step_000600`, transformer sha `220501a8…`) to `watson:~/Documents/dice/sft/libero30-sft/checkpoints/step_000600`; verify the transformer sha on the box.
- [ ] Step 4: write `~/Documents/dice-rl-wam/brev/env.sh` on the box over ssh stdin (W&B key from `brev/.env`, HF token from `~/.cache/huggingface/token`, `DICE_DATA`, `DICE_SKIP_APT=1`, `DICE_SFT_SRC` unset since the checkpoint is already in place, `MODAL_PROFILE` unused).
- [ ] Step 5: run `bash brev/setup.sh` in tmux on Watson; confirm `setup complete`, `python -c "from lerobot.policies.lingbot_va.modeling_lingbot_va import LingBotVAPolicy"` and `nvidia-smi` visible from the venv's torch.
- [ ] Step 6: P0: `bash brev/train.sh smoke-gb10 9 --residual-input base --epsilon -0.3 --env-steps 320000 --max-env-steps 300 --no-tmux`; record s/env-step from the log timestamps and GPU utilisation; record the step-0 sanity (the smoke does no train-eval, so run the 10-episode train-eval separately via `script.lingbot_rl_train eval` with `--eval-candidates 1` on 10 plan entries and compare with 4/10).
- [ ] Step 7: repeat Steps 3–5 on Jarvis (rsync from Watson to Jarvis over ssh if faster than from the Mac).
- [ ] Step 8: commit `brev/setup.sh` and `brev/env.example`.

### Task 2: Execution-step option and a local held-out runner

**Files:**
- Modify: `script/lingbot_eval_config.py` (`video_exec_step: int = -1` field, validated in {-1} ∪ [1, 20], written into `protocol()`), `script/lingbot_eval.py` and `script/lingbot_rl_policy.py`/`script/lingbot_rl_train.py` (pass the field into the LingBotVA policy config when loading), `script/lingbot_eval_modal.py` and `brev/train.sh` (option plumbing)
- Create: `brev/eval.sh` (`RUN|sft STEP K SHARD SHARDS [--video-exec-step E] [--residual-input base] [--plan-candidates M]`, runs the held-out protocol locally into `$DICE_DATA/heldout/<policy>/task_09`, shard layout identical to the Modal path so `result/heldout` merging works)
- Test: `tests/test_lingbot_eval.py` (default protocol unchanged apart from the new key; non-default value recorded in settings), `tests/test_lingbot_rl.py` (RL eval passes the value through)

- [ ] Step 1: write the failing tests: `EvalConfig(video_exec_step=10).protocol()["video_exec_step"] == 10`; default −1; `EvalConfig(video_exec_step=0)` raises; the RL `evaluate` settings record it.
- [ ] Step 2: run them, see them fail.
- [ ] Step 3: implement the field and the two loader pass-throughs (`LingBotVAConfig(..., video_exec_step=cfg.video_exec_step)`).
- [ ] Step 4: `brev/eval.sh`: assemble the config JSON with `stage: heldout`, `task_ids`, `shard`, `shards`, `video_exec_step`; call `script.lingbot_eval run` for `sft` and `script.lingbot_rl_train eval --residual-path ... --eval-candidates K --shard --shards` for a run; write `summary/shard_N.json`.
- [ ] Step 5: tests green; `bash -n brev/eval.sh`; commit.

### Task 3: Run the probes

- [ ] P0 on Watson (Task 1, Step 6): s/env-step and GPU utilisation at 20 video steps.
- [ ] P1: `brev/eval.sh sft 600 1 0 1 --video-exec-step {20,10,6,3}` on task 9, 100 episodes each; cuts 20 and 10 on Watson, 6 and 3 on Jarvis, in tmux; pull each `heldout/` directory to `result/heldout/sft-step000600-exec{E}/task_09` with rsync; paired comparison against the 20-step result on the same box; choose E per the spec rule (42 ± 10 band of the GB10's own 20-step run).
- [ ] P2 on Jarvis: `decode_plans` over 200 train-eval states, M = 4, at E; summarise across-plan vs within-plan spread of the decoded base actions.
- [ ] Record all numbers in `docs/heldout-results.md` (new section "Probes for plan-level selection") and in the spec's decisions table; commit.

## Decisions carried into the run (from the spec)

M = 4, ε = −0.3, 320k env steps, checkpoints every 80k, train-evals at 0/80k/240k, rolling resume, W&B as today, plan_random_prob 0.1 constant during collection only, argmax plan at evaluation.
