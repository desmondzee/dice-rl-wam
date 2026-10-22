import numpy as np
import pytest

from test_dicerl_libero import FakeLibero, FakeVenv, needs_gym

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


def shifted_joints(w, env, n):
    out = []
    for _ in range(n):
        w.reset(options={"seed": 3})
        out.append(env.state[1:8].copy())
    return out


@needs_gym
def test_set_shift_seed_zeroes_reset_count(tmp_path):
    env, w = make_rl_wrapper(tmp_path)
    w.reset(options={"seed": 1}); w.reset(options={"seed": 2})
    assert w.reset_count == 2
    w.set_shift_seed(5)
    assert w.reset_count == 0 and w.shift_seed == 5


@needs_gym
def test_shift_stream_is_deterministic_per_seed_and_reset(tmp_path):
    runs = {}
    for seed in (7, 7, 8):
        env, w = make_rl_wrapper(tmp_path)
        w.set_shift_seed(seed); w.set_shift(0.3)
        runs.setdefault(seed, []).append(shifted_joints(w, env, 4))
    a, b = runs[7]
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not np.array_equal(a[0], a[1])
    c = runs[8][0]
    for k in range(3):
        assert not np.array_equal(a[k + 1], c[k]) and not np.array_equal(c[k + 1], a[k])
    assert not np.array_equal(a[0], c[0])


@needs_gym
@pytest.mark.parametrize("off", [0, None])
def test_set_shift_zero_or_none_disables(tmp_path, off):
    env, w = make_rl_wrapper(tmp_path)
    w.set_shift_seed(7); w.set_shift(0.3); w.set_shift(off)
    assert w.shift is None
    w.reset(options={"seed": 3})
    assert np.array_equal(env.state, w.init_states[3]) and w.shift_stats() == {"draws": 0, "rejections": 0, "fallbacks": 0}


@needs_gym
def test_shift_stats_clears_on_read(tmp_path):
    env, w = make_rl_wrapper(tmp_path)
    w.set_shift_seed(7); w.set_shift(0.3)
    w.reset(options={"seed": 3})
    assert w.shift_stats()["draws"] == 1
    assert w.shift_stats() == {"draws": 0, "rejections": 0, "fallbacks": 0}


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


def test_states_file_validation(tmp_path):
    from script.dicerl_libero_eval import load_states
    np.savez(tmp_path / "s.npz", states=np.zeros((3, 47)), labels=np.array(["a", "b", "c"]))
    s, l = load_states(tmp_path / "s.npz"); assert s.shape == (3, 47) and list(l) == ["a", "b", "c"]
    np.savez(tmp_path / "bad.npz", states=np.zeros((3, 47)), labels=np.array(["a"]))
    with pytest.raises(ValueError): load_states(tmp_path / "bad.npz")


def test_single_seed_plan():
    from script.dicerl_libero_eval import episode_plan
    assert len(episode_plan(4, (0,))) == 4


def test_run_protocol_shift_option_and_labels():
    from script.dicerl_libero_eval import run_protocol, episode_plan
    venv = FakeVenv(2, succeed_at=[10 ** 6, 10 ** 6]); venv.options = []
    def reset_arg(options_list): venv.options.append(options_list); return FakeVenv.reset_arg(venv, options_list)
    venv.reset_arg = reset_arg
    rows = run_protocol(lambda o: np.zeros((2, 8, 7)), venv, np.zeros((2, 47)), episode_plan(2, (0,)), 2, max_steps=16, shift={"magnitude": 0.3, "seed": 1000}, labels=["x", "y"])
    assert [o["shift"]["seed"] for o in venv.options[0]] == [1000, 1001] and venv.options[0][0]["shift"]["magnitude"] == 0.3
    assert [r["label"] for r in rows] == ["x", "y"]


def test_run_protocol_default_labels_and_no_shift_option():
    from script.dicerl_libero_eval import run_protocol, episode_plan
    venv = FakeVenv(2, succeed_at=[10 ** 6, 10 ** 6]); venv.options = []
    def reset_arg(options_list): venv.options.append(options_list); return FakeVenv.reset_arg(venv, options_list)
    venv.reset_arg = reset_arg
    rows = run_protocol(lambda o: np.zeros((2, 8, 7)), venv, np.zeros((2, 47)), episode_plan(2, (0,)), 2, max_steps=16)
    assert [r["label"] for r in rows] == ["0", "1"] and all("shift" not in o for o in venv.options[0])


def test_build_plus_states(tmp_path):
    from script.dicerl_libero_plus_states import build
    canonical = np.tile(np.arange(47, dtype=float), (50, 1)); qpos = np.full((42, 7), 9.0); variants = np.arange(640, 682)
    full, monitor = build(canonical, qpos, variants)
    assert full["states"].shape == (294, 47) and monitor["states"].shape == (84, 47)
    assert full["labels"][0] == "640:0" and np.array_equal(full["states"][0][1:8], np.full(7, 9.0)) and np.array_equal(full["states"][0][8:], canonical[0][8:])
    assert monitor["labels"][-1] == "681:1"


def test_variant_breakdown():
    from script.dicerl_libero_plus_states import by_variant
    rows = [{"label": "640:0", "success": "True"}, {"label": "640:1", "success": "False"}, {"label": "641:0", "success": "True"}]
    assert by_variant(rows) == {"640": 0.5, "641": 1.0}


def test_iteration_from_checkpoint_name():
    from script.dicerl_libero_rl_eval import iteration_of
    assert iteration_of("/a/b/model_step_12000.pth") == 12000
    with pytest.raises(ValueError): iteration_of("/a/b/state_100.pt")


def test_augment_concatenates_state_and_visual_features():
    import torch
    from script.dicerl_libero_rl_eval import augment
    seen = {}
    def extract(cond): seen.update(cond); return torch.arange(6, dtype=torch.float32).reshape(2, 3)
    model = type("M", (), {})(); model.pretrained_flow_policy = type("P", (), {})(); model.pretrained_flow_policy.network = type("N", (), {"extract_visual_features": staticmethod(extract)})()
    obs = {"rgb": np.zeros((2, 1, 16, 16, 6), np.uint8), "state": np.ones((2, 1, 8))}
    out = augment(model, obs, "cpu")
    assert out.shape == (2, 1, 11) and torch.equal(out[:, 0, 8:], torch.arange(6, dtype=torch.float32).reshape(2, 3)) and torch.equal(out[:, 0, :8], torch.ones(2, 8))
    assert seen["rgb"].shape == (2, 1, 6, 16, 16) and seen["state"].shape == (2, 1, 8)


def test_residual_policy_slices_act_steps():
    import torch
    from script.dicerl_libero_rl_eval import residual_policy
    calls = {}
    class M:
        class pretrained_flow_policy:
            class network:
                extract_visual_features = staticmethod(lambda cond: torch.zeros(cond["state"].shape[0], 3))
        def get_exploration_action(self, state, num_samples, exploration_strategy, training_step):
            calls.update(num_samples=num_samples, strategy=exploration_strategy, step=training_step)
            return torch.arange(2 * 16 * 7, dtype=torch.float32).reshape(2, 16, 7), None
    obs = {"rgb": np.zeros((2, 1, 16, 16, 6), np.uint8), "state": np.zeros((2, 1, 8))}
    act = residual_policy(M(), 8, "cpu")(obs)
    assert act.shape == (2, 8, 7) and calls == {"num_samples": 16, "strategy": "max_q_min", "step": 10 ** 6}


def test_shift_stats_summed_into_summary(tmp_path):
    import json, types
    from script.dicerl_libero_eval import shift_totals, write_results
    venv = types.SimpleNamespace(call_sync=lambda name: [({"draws": 3, "rejections": 1, "fallbacks": 0}, True), ({"draws": 4, "rejections": 2, "fallbacks": 1}, True)])
    assert shift_totals(venv, None) == {"draws": 0, "rejections": 0, "fallbacks": 0}
    stats = shift_totals(venv, {"magnitude": 0.6, "seed": 1})
    assert stats == {"draws": 7, "rejections": 3, "fallbacks": 1}
    args = types.SimpleNamespace(ckpt="c/state_1.pt", out=str(tmp_path), envs=2, seeds=[0], states=None, shift=0.6, shift_seed=1)
    cfg = types.SimpleNamespace(env=types.SimpleNamespace(name="t", max_episode_steps=16), act_steps=8)
    write_results(args, cfg, [{"init_state_id": 0, "label": "0", "seed": 0, "success": True, "length": 8}], {"weights": "model", "shift_stats": stats})
    assert json.load(open(tmp_path / "summary.json"))["shift_stats"] == stats
