import importlib.util

import numpy as np
import pytest

needs_h5py = pytest.mark.skipif(importlib.util.find_spec("h5py") is None, reason="h5py not installed")
needs_gym = pytest.mark.skipif(importlib.util.find_spec("gym") is None, reason="gym not installed")


class FakeLibero:
    def __init__(self, success_after=None, state_dim=47):
        self.t = 0; self.success_after = success_after; self.state = np.zeros(state_dim); self.actions = []; self.calls = []
    def reset(self): self.calls.append("reset"); self.t = 0; return self._obs()
    def set_init_state(self, state): self.calls.append("set_init_state"); self.state = np.asarray(state); self.t = 0; return self._obs()
    def step(self, action):
        self.calls.append("step"); self.actions.append(np.asarray(action)); self.t += 1
        return self._obs(), 0.0, False, {}
    def check_success(self): return self.success_after is not None and self.t >= self.success_after
    def get_sim_state(self): return self.state
    def _obs(self):
        return {"agentview_image": np.full((128, 128, 3), 10, np.uint8), "robot0_eye_in_hand_image": np.full((128, 128, 3), 20, np.uint8),
                "robot0_eef_pos": np.array([0.1, 0.2, 0.9]), "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]), "robot0_gripper_qpos": np.array([0.04, -0.04])}


def make_wrapper(tmp_path, **kw):
    from env.gym_utils.wrapper.libero_image import LiberoImageWrapper
    np.savez(tmp_path / "norm.npz", obs_min=np.full(8, -1.0), obs_max=np.full(8, 1.0), action_min=np.full(7, -1.0), action_max=np.full(7, 1.0))
    env = kw.pop("env", FakeLibero())
    return env, LiberoImageWrapper(env, shape_meta={"obs": {"rgb": {"shape": [128, 128, 6]}, "state": {"shape": [8]}}, "action": {"shape": [7]}},
                                   normalization_path=str(tmp_path / "norm.npz"), init_states=np.arange(50)[:, None] * np.ones((50, 47)), **kw)


@needs_gym
def test_observation_layout(tmp_path):
    env, w = make_wrapper(tmp_path)
    obs = w.reset(options={"seed": 3})
    assert obs["rgb"].shape == (128, 128, 6) and obs["rgb"].dtype == np.uint8
    assert obs["rgb"][0, 0, 0] == 10 and obs["rgb"][0, 0, 3] == 20
    assert obs["state"].shape == (8,) and np.allclose(obs["state"][:3], [0.1, 0.2, 0.9], atol=1e-5) and np.allclose(obs["state"][3:6], 0.0, atol=1e-5)
    assert np.allclose(obs["state"][6:], [0.04, -0.04], atol=1e-5)


@needs_gym
def test_reset_seed_selects_stored_state_and_settles(tmp_path):
    env, w = make_wrapper(tmp_path)
    w.reset(options={"seed": 53})
    assert np.allclose(env.state, 3.0)
    assert len(env.actions) == 10 and np.allclose(env.actions[0], [0, 0, 0, 0, 0, 0, -1])


@needs_gym
def test_every_reset_resets_raw_env_before_set_init_state_and_settles(tmp_path):
    env, w = make_wrapper(tmp_path)
    w.reset(options={"seed": 1})
    w.reset(options={"seed": 2})
    assert env.calls == (["reset", "set_init_state"] + ["step"] * 10) * 2
    assert np.allclose(env.state, 2.0)


@needs_gym
def test_read_only_init_states_are_kept_without_copy(tmp_path):
    from env.gym_utils.wrapper.libero_image import LiberoImageWrapper
    states = np.arange(50)[:, None] * np.ones((50, 47))
    states.setflags(write=False)
    w = LiberoImageWrapper(FakeLibero(), shape_meta=None, init_states=states)
    assert w.init_states.flags.writeable is False and np.shares_memory(w.init_states, states)
    w.reset(seed=3)
    assert states.flags.writeable is False and np.array_equal(states[3], 3 * np.ones(47))


@needs_gym
def test_reset_init_state_option(tmp_path):
    env, w = make_wrapper(tmp_path)
    w.reset(options={"init_state": np.full(47, 7.0)})
    assert np.allclose(env.state, 7.0)


@needs_gym
def test_termination_after_five_cumulative_successes(tmp_path):
    env, w = make_wrapper(tmp_path, env=FakeLibero(success_after=13))
    w.reset(options={"seed": 0})
    rewards, terms = [], []
    for _ in range(8):
        _, r, term, info = w.step(np.zeros(7)); rewards.append(r); terms.append(term)
    assert rewards[:2] == [0.0, 0.0] and rewards[2] == 1.0
    assert terms.index(True) == 6 and sum(rewards) >= 1


@needs_gym
def test_truncation_at_max_steps(tmp_path):
    env, w = make_wrapper(tmp_path, max_episode_steps=4)
    w.reset(options={"seed": 0})
    out = [w.step(np.zeros(7))[3].get("TimeLimit.truncated", False) for _ in range(4)]
    assert out == [False, False, False, True]


@needs_gym
def test_orient():
    from env.gym_utils.wrapper.libero_image import orient
    img = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    assert np.array_equal(orient(img, "vertical"), img[::-1]) and np.array_equal(orient(img, "both"), img[::-1, ::-1]) and np.array_equal(orient(img, "none"), img)


@needs_gym
def test_action_unnormalize(tmp_path):
    env, w = make_wrapper(tmp_path)
    w.action_min = np.full(7, -0.5); w.action_max = np.full(7, 0.5)
    assert np.allclose(w.unnormalize_action(np.ones(7)), 0.5) and np.allclose(w.unnormalize_action(-np.ones(7)), -0.5)


@needs_gym
def test_wrapper_registered():
    from env.gym_utils.wrapper import wrapper_dict
    from env.gym_utils.wrapper.libero_image import LiberoImageWrapper
    assert wrapper_dict["libero_image"] is LiberoImageWrapper


@needs_gym
def test_write_config_is_atomic(tmp_path):
    from unittest import mock
    from env.gym_utils.wrapper import libero_image
    target = tmp_path / "cfg" / "config.yaml"
    libero_image.write_config(target, '{"a": 1}')
    assert target.read_text() == '{"a": 1}' and [p.name for p in target.parent.iterdir()] == ["config.yaml"]
    with mock.patch.object(libero_image.os, "replace", side_effect=OSError("boom")):
        with pytest.raises(OSError):
            libero_image.write_config(target, '{"a": 2}')
    assert target.read_text() == '{"a": 1}' and [p.name for p in target.parent.iterdir()] == ["config.yaml"]


def make_demo_h5(path, lengths=(5, 7)):
    import h5py
    with h5py.File(path, "w") as f:
        for i, n in enumerate(lengths):
            g = f.create_group(f"data/demo_{i}")
            o = g.create_group("obs")
            o["agentview_rgb"] = np.full((n, 128, 128, 3), i + 1, np.uint8)
            o["eye_in_hand_rgb"] = np.full((n, 128, 128, 3), i + 11, np.uint8)
            o["ee_pos"] = np.tile([0.0, 0.5, 1.0], (n, 1)) + i
            o["ee_ori"] = np.zeros((n, 3)); o["gripper_states"] = np.tile([0.02, -0.02], (n, 1))
            g["actions"] = np.linspace(-0.5, 0.5, n * 7).reshape(n, 7)
            g["rewards"] = np.r_[np.zeros(n - 1), 1.0]; g["dones"] = np.r_[np.zeros(n - 1), 1.0]
            g["states"] = np.zeros((n, 47))


@needs_h5py
def test_convert_shapes_and_order(tmp_path):
    from script.dataset.process_libero_dataset import convert
    make_demo_h5(tmp_path / "d.hdf5")
    convert(str(tmp_path / "d.hdf5"), str(tmp_path / "out"))
    d = np.load(tmp_path / "out" / "train.npz")
    assert d["states"].shape == (12, 8) and d["actions"].shape == (12, 7) and d["images"].shape == (12, 6, 128, 128)
    assert d["images"].dtype == np.uint8 and d["images"][0, 0, 0, 0] == 1 and d["images"][0, 3, 0, 0] == 11 and d["images"][5, 0, 0, 0] == 2
    assert list(d["traj_lengths"]) == [5, 7] and d["rewards"].sum() == 2 and d["terminals"].sum() == 2


@needs_h5py
def test_convert_max_episodes_caps_demos_and_stats(tmp_path):
    from script.dataset.process_libero_dataset import convert
    make_demo_h5(tmp_path / "d.hdf5", lengths=(5, 7, 4))
    assert convert(str(tmp_path / "d.hdf5"), str(tmp_path / "out"), max_episodes=2) == {"episodes": 2, "steps": 12}
    d = np.load(tmp_path / "out" / "train.npz"); n = np.load(tmp_path / "out" / "normalization.npz")
    assert list(d["traj_lengths"]) == [5, 7] and d["states"].shape[0] == 12
    assert np.allclose(n["obs_max"][:3], [1.0, 1.5, 2.0])


@needs_h5py
def test_normalization_true_extremes(tmp_path):
    from script.dataset.process_libero_dataset import convert
    make_demo_h5(tmp_path / "d.hdf5")
    convert(str(tmp_path / "d.hdf5"), str(tmp_path / "out"))
    n = np.load(tmp_path / "out" / "normalization.npz"); d = np.load(tmp_path / "out" / "train.npz")
    assert np.allclose(n["obs_min"][:3], [0.0, 0.5, 1.0]) and np.allclose(n["obs_max"][:3], [1.0, 1.5, 2.0])
    assert n["obs_min"][2] > 0
    assert d["states"].min() >= -1 - 1e-6 and d["states"].max() <= 1 + 1e-6 and np.isfinite(d["states"]).all()
    assert np.isclose(n["action_min"].min(), -0.5) and np.isclose(n["action_max"].max(), 0.5)


@needs_h5py
def test_convert_orders_demos_numerically_and_flips(tmp_path):
    import h5py
    from script.dataset.process_libero_dataset import convert
    make_demo_h5(tmp_path / "d.hdf5", lengths=(2,) * 11)
    with h5py.File(tmp_path / "d.hdf5", "a") as f:
        f["data/demo_10/obs/agentview_rgb"][:, 0] = 99
    convert(str(tmp_path / "d.hdf5"), str(tmp_path / "out"))
    d = np.load(tmp_path / "out" / "train.npz")
    assert d["images"][20, 0, 0, 0] == 99 and d["images"][20, 0, 127, 0] == 11
    convert(str(tmp_path / "d.hdf5"), str(tmp_path / "vertical"), flip="vertical")
    v = np.load(tmp_path / "vertical" / "train.npz")["images"]
    assert v[20, 0, 127, 0] == 99 and v[20, 0, 0, 0] == 11


@needs_gym
def test_flip_matches_wrapper_orient(tmp_path):
    from env.gym_utils.wrapper.libero_image import orient
    from script.dataset.process_libero_dataset import oriented
    frames = np.random.default_rng(0).integers(0, 255, (3, 8, 8, 3), dtype=np.uint8)
    for flip in ("none", "vertical", "both"):
        assert np.array_equal(oriented(frames, flip), np.stack([orient(f, flip) for f in frames]))


def test_pick_flip_accepts_clear_match_and_rejects_ambiguous():
    from script.dicerl_libero_gate0 import pick_flip
    rng = np.random.RandomState(0)
    img = rng.randint(0, 255, (128, 128, 3), np.uint8)
    other = rng.randint(0, 255, (128, 128, 3), np.uint8)
    rendered = {"agentview": img, "wrist": img}
    flip, scores = pick_flip(rendered, {"agentview": img, "wrist": img})
    assert flip == "none" and scores["none"] == 0 and scores["vertical"] > 0.05
    flip, scores = pick_flip(rendered, {"agentview": img[::-1], "wrist": img[::-1]})
    assert flip == "vertical" and scores["vertical"] < 0.05
    with pytest.raises(ValueError):
        pick_flip(rendered, {"agentview": other, "wrist": other})
    sym = np.concatenate([img[:64], img[:64][::-1]])
    with pytest.raises(ValueError):
        pick_flip({"agentview": sym, "wrist": sym}, {"agentview": sym, "wrist": sym})
    flat = 100 + rng.randint(-3, 4, (128, 128, 3))
    stored_flat = (100 + rng.randint(-3, 4, (128, 128, 3))).astype(np.uint8)
    with pytest.raises(ValueError):
        pick_flip({"agentview": flat.astype(np.uint8), "wrist": flat.astype(np.uint8)}, {"agentview": stored_flat, "wrist": stored_flat})


@needs_gym
def test_make_libero_env_ignores_robosuite_horizon(monkeypatch):
    import sys
    import types
    from env.gym_utils.wrapper import libero_image
    seen = {}
    fake = types.ModuleType("libero.libero.envs")
    fake.OffScreenRenderEnv = lambda **kw: seen.update(kw)
    monkeypatch.setitem(sys.modules, "libero.libero.envs", fake)
    monkeypatch.setattr(libero_image, "libero_task_files", lambda name: ("task.bddl", None))
    libero_image.make_libero_env("task")
    assert seen["ignore_done"] is True and seen["bddl_file_name"] == "task.bddl"


def test_episode_plan_order():
    from script.dicerl_libero_eval import episode_plan
    plan = episode_plan(3, (0, 1))
    assert plan == [{"init_state_id": 0, "seed": 0}, {"init_state_id": 1, "seed": 0}, {"init_state_id": 2, "seed": 0}, {"init_state_id": 0, "seed": 1}, {"init_state_id": 1, "seed": 1}, {"init_state_id": 2, "seed": 1}]


def test_episode_plan_seed_major_with_custom_seeds():
    from script.dicerl_libero_eval import episode_plan
    assert episode_plan(2, (2, 3)) == [{"init_state_id": 0, "seed": 2}, {"init_state_id": 1, "seed": 2}, {"init_state_id": 0, "seed": 3}, {"init_state_id": 1, "seed": 3}]


class FakeVenv:
    def __init__(self, n, succeed_at):
        self.n = n; self.succeed_at = succeed_at; self.resets = []; self.t = 0
    def _obs(self):
        return {"rgb": np.zeros((self.n, 1, 128, 128, 6), np.uint8), "state": np.zeros((self.n, 1, 8))}
    def reset_arg(self, options_list):
        self.resets.append([o["init_state"][0] for o in options_list]); self.t = 0
        return self._obs()
    def step(self, actions):
        self.t += 1
        r = np.array([1.0 if self.t == self.succeed_at[i] else 0.0 for i in range(self.n)])
        term = np.array([self.t >= self.succeed_at[i] + 4 for i in range(self.n)])
        return self._obs(), r, term, np.zeros(self.n, bool), [{}] * self.n


def test_eval_resets_with_init_state_every_episode():
    from script.dicerl_libero_eval import episode_plan, run_protocol, summarise
    venv = FakeVenv(2, succeed_at=[3, 10 ** 6])
    init_states = np.arange(4)[:, None] * np.ones((4, 47))
    rows = run_protocol(lambda obs: np.zeros((2, 8, 7)), venv, init_states, episode_plan(4, (0,)), n_envs=2, max_steps=80)
    assert venv.resets == [[0.0, 1.0], [2.0, 3.0]]
    assert [r["success"] for r in rows] == [True, False, True, False] and rows[0]["length"] == 7 * 8 and rows[1]["length"] == 80
    s = summarise(rows); assert s["n"] == 4 and s["success_rate"] == 0.5 and abs(s["stderr"] - 0.25) < 1e-9


def test_select_checkpoint_none_in_band():
    from script.dicerl_libero_watch import select_checkpoint, in_band
    assert select_checkpoint([{"epoch": 1000, "success_rate": 0.7}, {"epoch": 2000, "success_rate": 0.65}]) is None
    assert not in_band(0.39) and in_band(0.40) and in_band(0.60) and not in_band(0.61)


def test_select_checkpoint_closest_to_half_earlier_on_ties():
    from script.dicerl_libero_watch import select_checkpoint
    rows = [{"epoch": 1000, "success_rate": 0.42}, {"epoch": 2000, "success_rate": 0.56}, {"epoch": 3000, "success_rate": 0.44}]
    assert select_checkpoint(rows)["epoch"] == 2000
    rows = [{"epoch": 1000, "success_rate": 0.45}, {"epoch": 2000, "success_rate": 0.55}]
    assert select_checkpoint(rows)["epoch"] == 1000


def test_schedule_epochs():
    from script.dicerl_libero_watch import schedule_epochs
    e = schedule_epochs("100:1000,1000:8000")
    assert e[:11] == [100 * i for i in range(1, 10)] + [1000, 2000] and e[-1] == 8000 and e.count(1000) == 1 and len(e) == 9 + 8
    assert schedule_epochs("1000:3000") == [1000, 2000, 3000]


def test_checkpoint_epochs_follow_schedule(tmp_path):
    from script.dicerl_libero_watch import checkpoint_epochs
    (tmp_path / "checkpoint").mkdir()
    for e in (100, 150, 1000, 1500, 2000):
        (tmp_path / "checkpoint" / f"state_{e}.pt").write_text("x")
    assert checkpoint_epochs(str(tmp_path), {100, 1000, 2000}) == [100, 1000, 2000]


def _watch_stubs(monkeypatch, alive=True):
    import script.dicerl_libero_watch as w
    calls = {"stop": 0, "eval": []}
    monkeypatch.setattr(w, "training_pgid", lambda session: None)
    monkeypatch.setattr(w, "stop_training", lambda session, pgid: calls.__setitem__("stop", calls["stop"] + 1))
    monkeypatch.setattr(w, "session_alive", lambda session: alive)
    monkeypatch.setattr(w, "evaluate", lambda run_dir, epoch, envs: calls["eval"].append(epoch) or {"epoch": epoch, "success_rate": 0.9, "stderr": 0.0})
    monkeypatch.setattr(w.time, "sleep", lambda s: None)
    return w, calls


def test_watch_restart_with_in_band_row_stops_without_evaluating(tmp_path, monkeypatch):
    import json
    w, calls = _watch_stubs(monkeypatch)
    (tmp_path / "prior_sweep.json").write_text(json.dumps([{"epoch": 1000, "success_rate": 0.5, "stderr": 0.1}]))
    assert w.watch(str(tmp_path), w.schedule_epochs("100:1000,1000:8000"), 25, "s", 0) == 0
    assert calls["stop"] == 1 and calls["eval"] == []


def test_watch_restart_with_final_epoch_done_exits_one(tmp_path, monkeypatch):
    import json
    w, calls = _watch_stubs(monkeypatch)
    (tmp_path / "prior_sweep.json").write_text(json.dumps([{"epoch": 8000, "success_rate": 0.9, "stderr": 0.1}]))
    assert w.watch(str(tmp_path), w.schedule_epochs("100:1000,1000:8000"), 25, "s", 0) == 1
    assert calls["stop"] == 0 and calls["eval"] == []


def test_watch_dead_session_without_band_row_exits_one(tmp_path, monkeypatch):
    w, calls = _watch_stubs(monkeypatch, alive=False)
    assert w.watch(str(tmp_path), w.schedule_epochs("100:1000,1000:8000"), 25, "s", 0) == 1
    assert calls["stop"] == 0


def test_watch_evaluates_new_checkpoint_then_notices_dead_session(tmp_path, monkeypatch):
    import json
    w, calls = _watch_stubs(monkeypatch, alive=False)
    (tmp_path / "checkpoint").mkdir()
    (tmp_path / "checkpoint" / "state_1000.pt").write_text("x")
    assert w.watch(str(tmp_path), w.schedule_epochs("100:1000,1000:8000"), 25, "s", 0) == 1
    assert calls["eval"] == [1000] and json.loads((tmp_path / "prior_sweep.json").read_text())[0]["epoch"] == 1000
