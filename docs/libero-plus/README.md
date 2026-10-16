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
(`init_states=True`, uniform over its stored states). The draw schedule pairs the two runs per distribution, not per
episode: episode `i` draws the same kind, variant and stored state in both runs, but the simulator's object sampling
is seeded by `config.seed + env_steps`, which diverges with the runs' episode lengths. The buffer's `object_state`
stores MuJoCo `body_xquat` in w-x-y-z order while `proprio` carries robosuite's x-y-z-w end-effector quaternion
converted to axis-angle (`settings.json` records `object_layout.body_quaternion`). The held-out plan over the test
variants (`task9-heldout-plan.json`) is built after the runs are launched.
