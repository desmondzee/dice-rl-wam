import numpy as np
import pytest

pytest.importorskip("gym")


class FakeLibero:
    def __init__(self, success_after=None, state_dim=47):
        self.t = 0; self.success_after = success_after; self.state = np.zeros(state_dim); self.actions = []
    def reset(self): self.t = 0; return self._obs()
    def set_init_state(self, state): self.state = np.asarray(state); self.t = 0; return self._obs()
    def step(self, action):
        self.actions.append(np.asarray(action)); self.t += 1
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


def test_observation_layout(tmp_path):
    env, w = make_wrapper(tmp_path)
    obs = w.reset(options={"seed": 3})
    assert obs["rgb"].shape == (128, 128, 6) and obs["rgb"].dtype == np.uint8
    assert obs["rgb"][0, 0, 0] == 10 and obs["rgb"][0, 0, 3] == 20
    assert obs["state"].shape == (8,) and np.allclose(obs["state"][:3], [0.1, 0.2, 0.9], atol=1e-5) and np.allclose(obs["state"][3:6], 0.0, atol=1e-5)
    assert np.allclose(obs["state"][6:], [0.04, -0.04], atol=1e-5)


def test_reset_seed_selects_stored_state_and_settles(tmp_path):
    env, w = make_wrapper(tmp_path)
    w.reset(options={"seed": 53})
    assert np.allclose(env.state, 3.0)
    assert len(env.actions) == 10 and np.allclose(env.actions[0], [0, 0, 0, 0, 0, 0, -1])


def test_reset_init_state_option(tmp_path):
    env, w = make_wrapper(tmp_path)
    w.reset(options={"init_state": np.full(47, 7.0)})
    assert np.allclose(env.state, 7.0)


def test_termination_after_five_cumulative_successes(tmp_path):
    env, w = make_wrapper(tmp_path, env=FakeLibero(success_after=13))
    w.reset(options={"seed": 0})
    rewards, terms = [], []
    for _ in range(8):
        _, r, term, info = w.step(np.zeros(7)); rewards.append(r); terms.append(term)
    assert rewards[:2] == [0.0, 0.0] and rewards[2] == 1.0
    assert terms.index(True) == 6 and sum(rewards) >= 1


def test_truncation_at_max_steps(tmp_path):
    env, w = make_wrapper(tmp_path, max_episode_steps=4)
    w.reset(options={"seed": 0})
    out = [w.step(np.zeros(7))[3].get("TimeLimit.truncated", False) for _ in range(4)]
    assert out == [False, False, False, True]


def test_orient():
    from env.gym_utils.wrapper.libero_image import orient
    img = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    assert np.array_equal(orient(img, "vertical"), img[::-1]) and np.array_equal(orient(img, "both"), img[::-1, ::-1]) and np.array_equal(orient(img, "none"), img)


def test_action_unnormalize(tmp_path):
    env, w = make_wrapper(tmp_path)
    w.action_min = np.full(7, -0.5); w.action_max = np.full(7, 0.5)
    assert np.allclose(w.unnormalize_action(np.ones(7)), 0.5) and np.allclose(w.unnormalize_action(-np.ones(7)), -0.5)


def test_wrapper_registered():
    from env.gym_utils.wrapper import wrapper_dict
    from env.gym_utils.wrapper.libero_image import LiberoImageWrapper
    assert wrapper_dict["libero_image"] is LiberoImageWrapper


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
