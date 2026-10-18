import numpy as np

INSIDE_RADIUS = 0.12
TASKS = {9: {"mug": "white_yellow_mug_1", "microwave": "microwave_1", "door": "microwave_1_microjoint"}}


def body_position(state, layout, name):
    start = 7 * layout["bodies"].index(name)
    return np.asarray(state[start:start + 3], dtype=np.float64)


def joint_value(state, layout, name):
    return float(state[7 * len(layout["bodies"]) + layout["joints"].index(name)])


def phi_shape(calibration, layout, state):
    distance = float(np.linalg.norm(
        body_position(state, layout, calibration["mug"]) - body_position(state, layout, calibration["microwave"])))
    inside = float(distance < INSIDE_RADIUS)
    door_open = (joint_value(state, layout, calibration["door"]) - calibration["door_closed"]) / (
        calibration["door_open"] - calibration["door_closed"])
    return -distance + inside + (1.0 - door_open) * inside


def phi(calibration, layout, state, success):
    return phi_shape(calibration, layout, state) + 2.0 * float(success)


def shaping_reward(calibration, layout, before, after, gamma):
    return gamma * phi_shape(calibration, layout, after) - phi_shape(calibration, layout, before)


def calibrate(task_id, inner, layout, initial):
    if task_id not in TASKS:
        raise ValueError(f"No progress function for task {task_id}")
    names = TASKS[task_id]
    model = inner.sim.model
    low, high = (float(limit) for limit in model.jnt_range[model.joint_name2id(names["door"])])
    closed, opened = (low, high) if abs(low) <= abs(high) else (high, low)
    calibration = {"task_id": int(task_id), **names, "door_closed": closed, "door_open": opened,
                   "door_initial": joint_value(initial, layout, names["door"])}
    calibration["phi_initial"] = phi(calibration, layout, initial, False)
    return calibration


def progress(calibration, layout, before, after, success):
    return phi(calibration, layout, after, success) - phi(calibration, layout, before, False)
