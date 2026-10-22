import numpy as np

from test_dicerl_libero import FakeLibero, needs_gym

CANONICAL = np.array([-0.031, -0.133, 0.015, -2.439, 0.0, 2.207, 0.778])


class FakeController:
    def __init__(self):
        self.calls = []

    def update_initial_joints(self, joints):
        self.calls.append(np.asarray(joints))


class FakeGripper:
    dof = 1

    def __init__(self):
        self.current_action = np.array([0.75, -0.75])
        self.resets = 0

    def __setattr__(self, name, value):
        if name == "current_action" and np.array_equal(value, np.zeros(self.dof)):
            object.__setattr__(self, "resets", self.resets + 1)
        object.__setattr__(self, name, value)


class FakeRobot:
    def __init__(self):
        self.controller = FakeController()
        self.gripper = FakeGripper()


class FakeInner:
    def __init__(self):
        self.robots = [FakeRobot()]


class FakeRLLibero(FakeLibero):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.env = FakeInner()

    @property
    def controller_calls(self):
        return self.env.robots[0].controller.calls


def make_rl_wrapper(tmp_path, **kw):
    from test_dicerl_libero import make_wrapper
    base = np.zeros(47)
    base[1:8] = CANONICAL
    return make_wrapper(tmp_path, env=FakeRLLibero(), init_states=np.tile(base, (50, 1)), **kw)


@needs_gym
def test_controller_reference_follows_loaded_joints(tmp_path):
    env, w = make_rl_wrapper(tmp_path)
    canonical = w.init_states[0]
    w.reset(options={"init_state": canonical})
    assert env.controller_calls == []
    shifted = canonical.copy(); shifted[1:8] += 0.1
    w.reset(options={"init_state": shifted})
    assert len(env.controller_calls) == 1 and np.allclose(env.controller_calls[0], shifted[1:8])


@needs_gym
def test_controller_reference_update_tolerance_is_absolute_1e9(tmp_path):
    env, w = make_rl_wrapper(tmp_path)
    canonical = w.init_states[0]
    tiny = canonical.copy(); tiny[4] += 1e-7
    w.reset(options={"init_state": tiny})
    assert len(env.controller_calls) == 1
    inside = canonical.copy(); inside[1:8] += 1e-10
    w.reset(options={"init_state": inside})
    assert len(env.controller_calls) == 1


@needs_gym
def test_hard_reset_flag_is_stored_and_passed():
    from env.gym_utils.wrapper.libero_image import LiberoImageWrapper
    w = LiberoImageWrapper(FakeRLLibero(), None, hard_reset=False)
    assert w.hard_reset is False
    assert LiberoImageWrapper(FakeRLLibero(), None).hard_reset is True


@needs_gym
def test_soft_reset_resets_gripper_integrator_every_reset(tmp_path):
    env, w = make_rl_wrapper(tmp_path, hard_reset=False)
    gripper = env.env.robots[0].gripper
    for n in (1, 2, 3):
        gripper.current_action = np.array([0.75, -0.75])
        w.reset(options={"seed": n})
        assert np.array_equal(gripper.current_action, np.zeros(1))
        assert gripper.resets == n


@needs_gym
def test_hard_reset_leaves_gripper_alone(tmp_path):
    env, w = make_rl_wrapper(tmp_path)
    w.reset(options={"seed": 1})
    assert env.env.robots[0].gripper.resets == 0
