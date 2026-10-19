from importlib import import_module

import numpy as np

BASE_QPOS = np.array([0, -1.61037389e-01, 0.0, -2.44459747e00, 0.0, 2.22675220e00, np.pi / 4])
PANDA_MODULE = "libero.libero.envs.robots.mounted_panda"


def procedural_offset(rng, magnitude_lo, cap):
    direction = rng.normal(size=7)
    direction /= np.linalg.norm(direction)
    return rng.uniform(magnitude_lo, cap) * direction


class PoseOverride:
    def __init__(self):
        self.offset = None

    def set(self, offset):
        mounted_panda = import_module(PANDA_MODULE)
        self.offset = np.asarray(offset, dtype=np.float64).copy()
        pose = BASE_QPOS + self.offset
        mounted_panda.MountedPanda.init_qpos = property(lambda self: pose.copy())

    def clear(self):
        mounted_panda = import_module(PANDA_MODULE)
        self.offset = None
        mounted_panda.MountedPanda.init_qpos = property(lambda self: BASE_QPOS.copy())

    def current(self):
        return self.offset


class Curriculum:
    def __init__(self, cap_start, cap_end, trigger_success, trigger_window):
        self.cap_start = cap_start
        self.cap_end = cap_end
        self.trigger_success = trigger_success
        self.trigger_window = trigger_window
        self.cap = cap_start
        self.history = []

    def record(self, success):
        self.history.append(float(success))
        recent = self.history[-self.trigger_window:]
        if len(recent) >= self.trigger_window and float(np.mean(recent)) >= self.trigger_success:
            self.cap = self.cap_end

    def state_dict(self):
        return {"cap": self.cap, "history": list(self.history)}

    def load_state_dict(self, state):
        self.cap = state["cap"]
        self.history = list(state["history"])
