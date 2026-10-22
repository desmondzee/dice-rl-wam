import numpy as np

JOINTS = slice(1, 8)
MARGIN = 0.02
MAX_TRIES = 20


def sample_offset(rng, magnitude):
    if not magnitude:
        return np.zeros(7)
    direction = rng.normal(size=7)
    direction /= np.linalg.norm(direction)
    return direction * rng.uniform(0.0, magnitude)


def shift_state(state, offset):
    out = np.array(state, dtype=np.float64)
    out[JOINTS] += offset
    return out


def valid_start(env):
    model, data = env.sim.model, env.sim.data
    for j in range(7):
        q = data.qpos[model.jnt_qposadr[j]]
        low, high = model.jnt_range[j]
        if q < low + MARGIN or q > high - MARGIN:
            return False
    for i in range(data.ncon):
        contact = data.contact[i]
        names = [model.geom_id2name(contact.geom1) or "", model.geom_id2name(contact.geom2) or ""]
        for a, b in (names, names[::-1]):
            if a.startswith(("robot0_", "gripper0_")) and not b.startswith("gripper0_"):
                return False
    return True


def ramp_magnitude(step, start, end, magnitude):
    if step < start:
        return 0.0
    if step >= end:
        return magnitude
    return magnitude * (step - start) / (end - start)
