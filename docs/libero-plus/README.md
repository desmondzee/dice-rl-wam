# LIBERO-Plus task 9 protocol

`task9-plan.json` fixes a 100-episode evaluation of the LIBERO-10 task 9 base task
(`KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it`) on two
LIBERO-Plus perturbation axes of the plus `libero_10` suite (clone `sylvestf/LIBERO-plus` at `4976dc3`):

- robot initial state: 42 variants, suite indices 640–681 (`..._view_0_0_100_0_0_initstate_N`, arm pose `MountedPandaN.init_qpos`);
- objects layout: 39 variants, suite indices 1982–2020 (21 `_add_K` confounder objects, 18 `_levelL_sampleS` displacements).

Every variant runs once (81 episodes, robot-init ascending then layout ascending), then the first 10 robot-init and
first 9 layout variants by suite index run a second time (19 repeats, pass 2). `episode_index` is the plan order 0..99.

Reset rule (our reading; LIBERO-Plus ships no evaluation harness):

- `robot_init` entries, `reset_rule = variant_pose_sampled_objects`: `LiberoEnv(init_states=False, is_libero_plus=True)`.
  The arm starts at the variant pose and the object layout is sampled from the BDDL regions with `np.random` seeded
  by the entry seed, so these episodes pair across policies by seed, not by canonical init state (`init_state_id` is null).
  Each episode uses a freshly constructed env and exactly one reset: the first seeded reset of a fresh env is
  reproducible (verified pixel-exact), repeated resets of one env instance are not.
- `layout` entries, `reset_rule = stored_state`: `LiberoEnv(init_states=True)` with the variant's single stored state (`init_state_id` 0).

Seeds use the standard `episode_plan` formula (`sha256(json.dumps(["libero_10", 42, 9, slot] + [1] * repeat))[:4]`)
with `slot` = the variant's rank within its axis; repeats use `repeat = 1`. The policy receives the base task's
instruction, not the suffixed `task.language` of the variant. Horizon 520 steps, 10 settle steps, relative control at
20 Hz, 128x128 cameras, and the 5/10 sampler, as in the standard held-out evaluation.

Files: `make_plan.py` regenerates the plan on a box with the plus package active; `plus_eval.py` is the driver
(copy of `~/Documents/dice/libero-plus-config/plus_eval.py` on jarvis); `check_reset.py` verifies that the seeded
`init_states=False` reset is deterministic. Per-episode JSON matches `script/lingbot_eval.py` output plus the plan keys
and a `reset` record (robot model, `init_qpos`, joints after reset, object body positions); `summary.json` reports
successes per axis and overall.
