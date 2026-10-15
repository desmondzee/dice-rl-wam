# Benchmark candidates for LingBot-VA DICE-RL beyond LIBERO-10

Date: 2026-10-15. Setting: LingBot-VA 5B SFT prior (30 demos/task, empty text embedding, two 128×128 cameras,
7-DoF OSC, chunk 16) at 60–80 % on LIBERO-10 tasks 0/4/9 at the 5/10 sampler (task 8 at 12 %), residual RL plus
plan selection on top, evaluation on two DGX Spark GB10 (aarch64, EGL, 65–130 s/episode), LeRobot env interface.
Target: a prior at 40–50 % on an accepted benchmark whose demos convert to LeRobot and whose simulator runs on aarch64.

## 1. Hard filter: simulator on aarch64 Linux with EGL

| simulator | aarch64 status | consequence |
|---|---|---|
| MuJoCo / robosuite (LIBERO, Robomimic, MimicGen, RoboCasa, MetaWorld, VLABench) | wheels exist; `MUJOCO_GL=egl` is what we run today | viable |
| PyBullet (CALVIN) | builds from source on aarch64; EGL plugin for GPU render | viable, untested here |
| SAPIEN 3.0.3 (ManiSkill3, RoboTwin 2.0, SimplerEnv, RoboMME) | PyPI ships only manylinux x86_64, win_amd64, macOS universal2; no Linux aarch64 wheel | blocked unless SAPIEN+PhysX source build (unverified) |
| Isaac Sim 5.1 / Isaac Lab 2.3 (RoboDojo) | aarch64 builds "only supported on DGX Spark" (DGX OS 7, RT-core GPU); Isaac Lab 2.3 lists Spark caveats (CUDA 13 wheels, no cuRobo/SkillGen); the RoboDojo fork notes DLSS denoiser needs `libnvidia-ngx` | possible in principle, heavy and unvalidated for camera policies |
| Isaac Sim 4.5 / OmniGibson (BEHAVIOR-1K) | no aarch64 build of 4.5 | blocked |
| CoppeliaSim 4.1 Edu + PyRep (RLBench) | x86_64 Linux tarball only, Edu non-commercial licence | blocked |

Sources: pypi.org/project/sapien (files), docs.isaacsim.omniverse.nvidia.com requirements, isaac-sim.github.io/IsaacLab v2.3.2 install,
huggingface.co/docs/lerobot (libero, libero_plus, metaworld, robotwin), robomimic.github.io/docs/datasets/robosuite.html.

## 2. What the offline-to-online RL papers used (de facto standard)

| paper | sim benchmarks | prior and its success before RL |
|---|---|---|
| DSRL 2506.15799 | Robomimic Lift/Can/Square/Transport (DPPO checkpoints; Can/Square MH for offline-to-online), OGBench, LIBERO-90 one task, ALOHA sim | DP; π0 fine-tuned on 40 LIBERO tasks (not libero_90) ≈20 % on "pick up the cream cheese and put it in the tray" |
| DICE-RL 2603.10263 | Robomimic Can (20 demos), Square (50 PH), Transport (50 PH), Tool Hang (50 PH), pixel 96→84 crop (Tool Hang 240→216); LIBERO-10 appendix (flow BC 50/task; π0 SFT 30/task) | flow BC; Tool Hang 45 % → >90 % in ≈2k episodes |
| FASTER 2604.19730 | Robomimic Lift (10-episode subset)/Can (MH)/Square (PH)/Tool Hang (PH); 5 held-out libero_90 tasks with `pi05_libero` | π0.5 at 40–60 % on tasks 2, 28, 29, 60, 79 of libero_90 (bowl→top drawer ×2 scenes, close top drawer, bowl→tray, book→caddy) |
| πRL 2510.25889 | LIBERO 4 suites few-shot; ManiSkill3 pick-place (4,352 combos, 16k MPLib demos, RGB) | π0 57.6 % / π0.5 77.1 % (LIBERO avg; Long 51.2 / 43.9); ManiSkill π0 38.4 %, π0.5 40.1 % |
| Q-VGM 2606.08015, RLinf-VLA | LIBERO few-shot (πRL checkpoints); RLinf covers ManiSkill, LIBERO, RoboCasa, MetaWorld | OpenVLA-OFT released ckpt 42.7 % on libero_90 → 98 % after GRPO (rlinf docs) |

Robomimic (image) plus LIBERO few-shot/held-out is the standard; nothing else recurs across these papers.

## 3. Candidates

### RoboDojo (arXiv 2607.04434, 5 Jul 2026, Chen et al., 44 authors; XPolicyLab 2608.09892)
Single thing by that name. 42 Isaac Sim/Isaac Lab tasks over five axes (generalisation, memory, precision, long-horizon, open) plus 18
real tasks (Piper X, Piper, ARX X5 via cloud RealEval). Sim embodiment is a bimanual ARX X5 pair (bases 0.6 m apart), 25 Hz, 100
trajectories per training task (34 tasks; 3,500 total, HF-hosted). Repo RoboDojo-Benchmark/RoboDojo says MIT; the neubig fork README
cites a "RoboDojo Non-Commercial Research License"; check LICENSE. Release is eval-only; policies run as XPolicyLab servers.
Sim leaderboard (Table 1, success): π0.5 6.9 %, X-VLA 6.5 %, LingBot-VLA 3.0 %, Fast-WAM 2.0 %, π0 1.5 %, best entry 8.8 %, human 76 %.
Verdict: bimanual, Isaac Sim, every prior below 10 %, no tasks in band. Not usable for us.

### Robomimic (robomimic.github.io; Lift, Can, Square, Transport, Tool Hang)
Panda, OSC delta-EE 7-dim (Transport is two Pandas, 14-dim: excluded). PH 200 demos/task, MH 300 (no MH Tool Hang). HDF5 of MuJoCo
states; images re-rendered by `dataset_states_to_obs.py --camera_names agentview robot0_eye_in_hand --camera_height H --camera_width W`,
any resolution. v0.1 data now regenerated for robosuite v1.5.1 (LIBERO/MimicGen pin v1.4.1: two robosuite builds in one env).
Code MIT; dataset licence not stated on the docs page (secondary sources say MIT).
Image priors with full data (DP paper 2303.04137 Table 2, max/avg): Square PH 0.98/0.92, MH 0.98/0.84; Tool Hang PH 0.95/0.73 (CNN),
0.76/0.47 (Transformer); Lift and Can 1.00. Low-data priors: DICE-RL Tool Hang 50 PH at 45 %; one 20-demo report puts DP Square ≈43 %
and Tool Hang ≈15 % (secondary, uncited). In band: Tool Hang at 50 PH (45 %, but at 240 px; at 128 px probably lower), Square at
≤30 PH (≈40–50 %, needs a sweep). Gotchas: Tool Hang horizon 700 (≈2× our episode time), fine insertion at 128 px; robosuite version
split; no LeRobot env wrapper for plain robosuite (LeRobot ships libero, metaworld, robocasa, robotwin, robomme, vlabench only).

### MimicGen core datasets (arXiv 2310.17596, CoRL 2023; code NVIDIA Source Code License, datasets CC-BY 4.0)
Same stack as LIBERO: robosuite v1.4.1 (v1.5+ unsupported), robomimic v0.3, mujoco 2.3.2, Panda, delta-EE 7-dim, agentview + eye-in-hand.
10 human source demos per task, 1,000 generated per variant, 26 core variants (D0 < D1 < D2 reset distributions), 48k demos total,
HF `amandlek/mimicgen_datasets`, robosuite HDF5 with states (re-render at 128×128 as above). Kitchen and Hammer Cleanup need
robosuite_task_zoo. BC-RNN image on 1,000 demos (max of 3 seeds):

| task | D0 | D1 | D2 |
|---|---|---|---|
| Square (= Robomimic Square, wider resets) | 90.7 | 73.3 | 49.3 |
| Threading | 98.0 | 60.7 | 38.0 |
| Three Piece Assembly | 82.0 | 62.7 | 13.3 |
| Coffee | 100 | 90.7 | 77.3 |
| Mug Cleanup / Hammer Cleanup / Kitchen | 80.0 / 100 / 100 | 64.0 / 62.7 / 76.0 | – |
| Nut Assembly / Pick Place / Coffee Preparation | 53.3 / 50.7 / 97.3 | – / – / 42.0 | – |

Two dials (reset distribution, demo count up to 1,000) put a strong prior in band without changing the task. Not yet a standard for RL
papers; it is the standard data-scaling IL suite and Square D0 is the DSRL/FASTER/DICE-RL task. No DP/π0/WAM numbers for D1/D2 found.

### LIBERO-90 and perturbed LIBERO
LIBERO-90: 90 short tasks, 50 demos each, same hf-libero/BDDL stack, 400-step horizon. Multitask priors are high (DP 75.4 %, π0 92.1 %,
2606.23420 Table) but held-out use lands in band: FASTER's five tasks with π0.5 at 40–60 %, DSRL's π0 at ≈20 %. Our SFT has no language
conditioning, so the prior must be trained on the chosen tasks and scene-sharing tasks (e.g. task 28 "close the top drawer" and 29
"bowl into top drawer") are ambiguous; pick one task per scene.
LIBERO-Plus (2510.13626, CVPR 2026; repo sylvestf/LIBERO-plus; LeRobot doc + `lerobot/libero_plus`): 10,030 variants over seven
perturbation axes; installing replaces hf-libero. Success under one axis (Table 2): robot-initial-state OpenVLA-OFT 31.9, UniVLA 46.2,
π0-fast 21.6, π0 6.0; layout 68.7–77.6; camera π0 13.8 (LingBot-VA 28.9 under camera per 2603.22078). Widely cited in 2026 papers.
LIBERO-PRO (2510.03827; Zxy-MLlab/LIBERO-PRO): object/initial-state/instruction/environment swaps; π0 and OpenVLA fall to 0 beyond 0.2
displacement, π0.5 0.38 on libero-goal position change; mostly 0 %, less adopted. Perturbed-init on LIBERO-10 is a dial, not a split,
and a reviewer will read it as a custom protocol unless named as a LIBERO-Plus sub-component.

### RoboCasa (2406.02523, RSS 2024, CC BY 4.0) and RoboCasa365 (2603.04356, ICLR 2026)
robosuite/MuJoCo, Panda on Omron mobile base, 12-dim action (base 4, mode 1, EE 6, gripper 1 per LeRobot wrapper), three 256×256
cameras, 20 Hz, LeRobot wrapper present in the pinned checkout. 50 human demos per atomic task (25 tasks) + 3,000 MimicGen each.
BC-Transformer Human-50: avg 28.8 %; in band OpenSingleDoor 46, OpenDrawer 42, TurnOffSinkFaucet 50, TurnSinkSpout 54,
CoffeePressButton 48, TurnOnSinkFaucet 38; Gen-3000 avg 47.6 %. RoboCasa365 multi-task over 300 tasks: π0.5 39.6 % atomic-seen,
π0 36.3, GR00T N1.5 43.0, DP 15.7. Gotchas: 12-dim action needs a new action head and full re-SFT; big asset pack; slow kitchens.

### Others
- ManiSkill3 (2410.00425, Apache 2.0): πRL pick-place prior 38–40 % is in band, but SAPIEN has no aarch64 wheel; DP baselines
  disagree across papers (PickCube 0.44–0.99, StackCube 0–0.41, PegInsertion ≈0). Blocked.
- RoboTwin 2.0 (2506.18088; LeRobot wrapper, `lerobot/robotwin_unified`): Aloha-AgileX 14-dim joint, 50 clean demos; π0 46.4 easy/
  16.3 hard, DP 28.0/0.6, DP3 55.2/5.0; LingBot-VA 92.2 % with 50 clean + 500 randomised demos per task. Bimanual and SAPIEN. Blocked.
- SimplerEnv: SAPIEN real-to-sim eval only (no training demos); π0 WidowX ≈27 % with high run variance. Blocked.
- BEHAVIOR-1K: Isaac Sim 4.5, R1 mobile bimanual, 50 tasks, 10k demos; π0.5-based 2025 entry Q 0.345. Blocked.
- RLBench: CoppeliaSim x86_64 only. Blocked.
- CALVIN (2112.03227, MIT, PyBullet+EGL, Franka, 7-dim relative EE, 200×200 + 84×84 cameras, 30 Hz): metric is 5-step chain length
  (DP 1.35, GR-1 3.06, SDP 4.49), first-step success 85–99 %; no per-task few-shot protocol. Poor fit.
- MetaWorld MT50 (LeRobot wrapper + `lerobot/metaworld_mt50`, single 480×480 corner2 camera, Sawyer 4-dim action): DP 10.5 %, π0 50.8,
  π0.5 43.8, SmolVLA 68.2 (rlinf docs/SmolVLA 2506.01844). In band for VLAs but 4-DoF and toy visuals. Poor fit.
- VLABench (2412.18194, MuJoCo/dm_control, LeRobot wrapper): language-conditioned long-horizon; no few-shot prior numbers found.

## 4. Ranking for our purpose

| rank | benchmark | why | engineer-days |
|---|---|---|---|
| 1 | MimicGen core D1/D2 (Square D1/D2, Threading D1, Three Piece Assembly D1, Coffee D2, Mug Cleanup D1) | identical robosuite 1.4.1/MuJoCo/Panda/two-camera stack as LIBERO, Square is the DSRL/FASTER/DICE-RL task, and reset-distribution plus demo-count dials put a prior in band without inventing a protocol | 4–6 (robosuite env wrapper 2, HDF5→LeRobot converter from the LIBERO one 1, 128×128 re-render + stored init states for our seed protocol 1, harness/tests 1) + SFT compute |
| 2 | LIBERO-90 held-out subset (FASTER tasks 2, 29, 60, 79 plus DSRL cream-cheese; drop 28, same scene as 29) with LIBERO-Plus layout/robot-init as the difficulty dial | zero simulator work, two RL papers already report π0/π0.5 priors at 20–60 % on exactly these tasks, LIBERO-Plus is CVPR-accepted | 1–2 (convert 5 tasks, scene-unique task selection, LIBERO-Plus fork + asset pack + determinism check) + SFT compute; prior calibration uncertain because our SFT trains on the tasks |
| 3 | Robomimic Tool Hang (50 PH) + Square (20–30 PH) | DICE-RL's own 45 % prior on Tool Hang, direct comparability with DSRL/FASTER | 5–7 (as rank 1 plus robosuite 1.5.1 vs 1.4.1 handling and a 128 px vs 240 px study); Tool Hang episodes ≈2× longer |

RoboCasa is fourth (6–10 days plus a 12-dim action head). SAPIEN- and Isaac-based suites are blocked on the GB10s.

Recommended task set: MimicGen core Square D1, Square D2, Threading D1, Three Piece Assembly D1 (Panda, agentview + eye-in-hand at
128×128, 7-dim OSC, 100 generated demos per task, held-out eval on 50 stored D-variant reset states × 2 seeds). Expected prior: 40–55 %
(BC-RNN with 1,000 demos reaches 49–73 % on these; a 5B prior on 100 demos should land below that). Calibrate with one SFT sweep
(30/100/300 demos on Square D1) before committing; if 100 demos overshoot, drop to 30 or move Square to D2.
