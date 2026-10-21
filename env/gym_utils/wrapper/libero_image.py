import importlib.util
import json
import os
from pathlib import Path

import gym
import imageio
import numpy as np
from gym import spaces

NOOP = np.array([0, 0, 0, 0, 0, 0, -1.0])
SETTLE_STEPS = 10
FLIP = "vertical"


def libero_task_files(task_name):
    spec = importlib.util.find_spec("libero")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("The libero package is required")
    root = Path(next(iter(spec.submodule_search_locations))) / "libero"
    config_dir = Path(os.environ.setdefault("LIBERO_CONFIG_PATH", "/tmp/dicerl-libero-config"))
    config = config_dir / "config.yaml"
    if not config.exists():
        config_dir.mkdir(parents=True, exist_ok=True)
        config.write_text(json.dumps({
            "benchmark_root": str(root), "bddl_files": str(root / "bddl_files"),
            "init_states": str(root / "init_files"), "assets": str(root / "assets"),
            "datasets": str(config_dir / "unused-datasets"),
        }))
    from libero.libero import benchmark
    suite = benchmark.get_benchmark("libero_10")()
    names = [suite.get_task(i).name for i in range(len(suite.get_task_names()))]
    if task_name not in names:
        raise ValueError(f"Unknown LIBERO-10 task {task_name!r}")
    index = names.index(task_name)
    return suite.get_task_bddl_file_path(index), np.asarray(suite.get_task_init_states(index))


def quat_to_axis_angle(quat_xyzw):
    from robosuite.utils.transform_utils import quat2axisangle
    return np.asarray(quat2axisangle(np.asarray(quat_xyzw, dtype=np.float64)))


def orient(image, flip):
    if flip == "none":
        return image
    if flip == "vertical":
        return image[::-1]
    if flip == "both":
        return image[::-1, ::-1]
    raise ValueError(f"Unknown flip {flip!r}")


def make_libero_env(task_name, camera_names=("agentview", "robot0_eye_in_hand"), image_hw=128):
    bddl, _ = libero_task_files(task_name)
    from libero.libero.envs import OffScreenRenderEnv
    return OffScreenRenderEnv(
        bddl_file_name=bddl, camera_heights=image_hw, camera_widths=image_hw, camera_names=list(camera_names),
    )


class LiberoImageWrapper(gym.Env):
    def __init__(
        self,
        env,
        shape_meta,
        normalization_path=None,
        init_states=None,
        image_keys=("agentview_image", "robot0_eye_in_hand_image"),
        success_steps_before_termination=5,
        max_episode_steps=520,
        flip=FLIP,
        render_camera_name="agentview",
    ):
        self.env = env
        self.shape_meta = shape_meta
        self.init_states = None if init_states is None else np.asarray(init_states)
        self.image_keys = list(image_keys)
        self.success_steps_before_termination = success_steps_before_termination
        self.max_episode_steps = max_episode_steps
        self.flip = flip
        self.render_camera_name = render_camera_name
        self.has_reset_before = False
        self.video_writer = None
        self.last_raw_obs = None
        self.success_count = 0
        self.step_count = 0
        self.ever_succeeded = False
        self.normalize = normalization_path is not None
        if self.normalize:
            normalization = np.load(normalization_path)
            self.obs_min = normalization["obs_min"]
            self.obs_max = normalization["obs_max"]
            self.action_min = normalization["action_min"]
            self.action_max = normalization["action_max"]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        if shape_meta is not None:
            observation_space = spaces.Dict()
            for key, value in shape_meta["obs"].items():
                if key.endswith("rgb"):
                    low, high = 0, 1
                elif key.endswith("state"):
                    low, high = -1, 1
                else:
                    raise RuntimeError(f"Unsupported type {key}")
                observation_space[key] = spaces.Box(low=low, high=high, shape=tuple(value["shape"]), dtype=np.float32)
            self.observation_space = observation_space

    def normalize_obs(self, obs):
        return 2 * ((obs - self.obs_min) / (self.obs_max - self.obs_min + 1e-6) - 0.5)

    def unnormalize_action(self, action):
        return (action + 1) / 2 * (self.action_max - self.action_min) + self.action_min

    def get_observation(self, raw_obs):
        self.last_raw_obs = raw_obs
        rgb = np.concatenate([orient(raw_obs[key], self.flip) for key in self.image_keys], axis=-1)
        state = np.concatenate([
            np.asarray(raw_obs["robot0_eef_pos"]),
            quat_to_axis_angle(raw_obs["robot0_eef_quat"]),
            np.asarray(raw_obs["robot0_gripper_qpos"]),
        ])
        if self.normalize:
            state = self.normalize_obs(state)
        return {"rgb": np.ascontiguousarray(rgb).astype(np.uint8), "state": state}

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed=seed)
        else:
            np.random.seed()

    def get_sim_state(self):
        return self.env.get_sim_state()

    def reset(self, options={}, **kwargs):
        if self.video_writer is not None:
            self.video_writer.close()
            self.video_writer = None
        if "video_path" in options:
            self.video_writer = imageio.get_writer(options["video_path"], fps=30)
        seed = options.get("seed", None)
        state = options.get("init_state", None)
        if seed is not None:
            self.seed(seed)
        if state is None and self.init_states is not None:
            index = seed % len(self.init_states) if seed is not None else np.random.randint(len(self.init_states))
            state = self.init_states[index]
        if not self.has_reset_before:
            self.env.reset()
            self.has_reset_before = True
        raw_obs = self.env.set_init_state(state) if state is not None else self.env.reset()
        for _ in range(SETTLE_STEPS):
            raw_obs, _, _, _ = self.env.step(NOOP)
        self.success_count = 0
        self.step_count = 0
        self.ever_succeeded = False
        return self.get_observation(raw_obs)

    def step(self, action):
        if self.normalize:
            action = self.unnormalize_action(action)
        raw_obs, _, _, _ = self.env.step(action)
        obs = self.get_observation(raw_obs)
        reward = float(self.env.check_success())
        self.step_count += 1
        info = {}
        if self.step_count >= self.max_episode_steps:
            info["TimeLimit.truncated"] = True
        terminated = False
        if reward > 0:
            self.ever_succeeded = True
            self.success_count += 1
            if self.success_count >= self.success_steps_before_termination:
                terminated = True
                self.success_count = 0
                self.step_count = 0
                self.ever_succeeded = False
        elif not self.ever_succeeded:
            self.success_count = 0
        if self.video_writer is not None:
            self.video_writer.append_data(self.render(mode="rgb_array"))
        return obs, reward, terminated, info

    def render(self, mode="rgb_array"):
        return orient(self.last_raw_obs[f"{self.render_camera_name}_image"], self.flip)
