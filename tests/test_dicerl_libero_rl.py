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


class FakeModel:
    jnt_range = np.tile([-2.9, 2.9], (7, 1))
    jnt_qposadr = np.arange(7)
    names = ["robot0_link4_collision", "gripper0_finger1_collision", "gripper0_finger2_collision", "microwave_1_body", "table_collision"]

    def geom_id2name(self, i):
        return self.names[i]


class FakeContact:
    def __init__(self, geom1, geom2):
        self.geom1 = geom1
        self.geom2 = geom2


class FakeData:
    def __init__(self, env):
        self.env = env
        self.contact = []

    @property
    def ncon(self):
        return len(self.contact)

    @property
    def qpos(self):
        return self.env.state[1:8]


class FakeSim:
    def __init__(self, env):
        self.model = FakeModel()
        self.data = FakeData(env)
        self.always_contact = False

    def add_contact(self, name1, name2):
        self.data.contact.append(FakeContact(self.model.names.index(name1), self.model.names.index(name2)))

    def clear_contacts(self):
        self.data.contact = []

    def forward(self):
        self.clear_contacts()
        if self.always_contact:
            self.add_contact("robot0_link4_collision", "table_collision")


class FakeRLLibero(FakeLibero):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.env = FakeInner()
        self.sim = FakeSim(self)

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


@needs_gym
def test_sample_offset_norm_and_zero():
    from env.gym_utils.wrapper.libero_shift import sample_offset
    rng = np.random.RandomState(0)
    norms = [np.linalg.norm(sample_offset(rng, 0.6)) for _ in range(500)]
    assert max(norms) <= 0.6 + 1e-9 and min(norms) >= 0 and abs(np.mean(norms) - 0.3) < 0.03
    assert np.array_equal(sample_offset(rng, 0.0), np.zeros(7))


@needs_gym
def test_shift_state_changes_only_joints_and_not_input():
    from env.gym_utils.wrapper.libero_shift import shift_state
    s = np.arange(47, dtype=float); s.setflags(write=False)
    out = shift_state(s, np.ones(7))
    assert np.array_equal(out[1:8], s[1:8] + 1) and np.array_equal(np.delete(out, range(1, 8)), np.delete(s, range(1, 8))) and out is not s


@needs_gym
def test_valid_start_rejects_contact_and_range():
    from env.gym_utils.wrapper.libero_shift import valid_start
    env = FakeRLLibero()
    assert valid_start(env)
    env.sim.add_contact("robot0_link4_collision", "microwave_1_body")
    assert not valid_start(env)
    env.sim.clear_contacts(); env.sim.add_contact("gripper0_finger1_collision", "gripper0_finger2_collision")
    assert valid_start(env)
    env.sim.clear_contacts(); env.state[1] = env.sim.model.jnt_range[0][1] - 0.01
    assert not valid_start(env)


@needs_gym
def test_training_shift_draws_shifted_stored_state_and_counts(tmp_path):
    env, w = make_rl_wrapper(tmp_path)
    w.set_shift_seed(7); w.set_shift(0.3)
    w.reset(options={"seed": 3})
    assert 0 < np.linalg.norm(env.state[1:8] - w.init_states[3][1:8]) <= 0.3
    assert w.shift_stats() == {"draws": 1, "rejections": 0, "fallbacks": 0}


@needs_gym
def test_explicit_init_state_ignores_training_shift(tmp_path):
    env, w = make_rl_wrapper(tmp_path)
    w.set_shift_seed(7); w.set_shift(0.3)
    w.reset(options={"init_state": w.init_states[5]})
    assert np.array_equal(env.state, w.init_states[5])


@needs_gym
def test_shift_option_applies_to_explicit_state_with_fallback(tmp_path):
    env, w = make_rl_wrapper(tmp_path)
    env.sim.always_contact = True
    w.reset(options={"init_state": w.init_states[5], "shift": {"magnitude": 0.3, "seed": 11}})
    stats = w.shift_stats()
    assert np.array_equal(env.state, w.init_states[5]) and stats["fallbacks"] == 1 and stats["rejections"] == 20
    assert env.calls.count("reset") == 2


@needs_gym
def test_ramp_magnitude_segments():
    from env.gym_utils.wrapper.libero_shift import ramp_magnitude
    assert ramp_magnitude(0, 3000, 6000, 0.6) == 0.0 and ramp_magnitude(2999, 3000, 6000, 0.6) == 0.0
    assert abs(ramp_magnitude(4500, 3000, 6000, 0.6) - 0.3) < 1e-12
    assert ramp_magnitude(6000, 3000, 6000, 0.6) == 0.6 and ramp_magnitude(20000, 3000, 6000, 0.6) == 0.6
