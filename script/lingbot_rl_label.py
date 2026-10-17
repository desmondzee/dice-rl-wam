import multiprocessing
import os
import traceback

import numpy as np


def build_suite(spec):
    if spec is None:
        return None
    if spec["plus"]:
        from script.lingbot_rl_train import _plus_suite
        return _plus_suite(spec["base"])[0]
    from script.lingbot_eval import describe_suite
    return describe_suite(spec["assets_path"])[0]


def pack_snapshot(snapshot):
    state = snapshot["state"]
    return {**snapshot, "state": (state.time, np.array(state.qpos), np.array(state.qvel))}


def unpack_snapshot(packed, sim):
    time, qpos, qvel = packed["state"]
    return {**packed, "state": type(sim.get_state())(time=time, qpos=qpos, qvel=qvel)}


def _serve(connection, spec, norm, layout, calibration, shadow_factory):
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from script.lingbot_rl_train import _timed, label_plans, make_shadow
    suite, cached, shadow = None, None, None
    while True:
        message = connection.recv()
        if message is None:
            break
        try:
            snapshot, before, candidates, first_chunk, env_task, init_states, plus, seed = message
            key = (env_task, init_states, plus)
            if key != cached:
                if shadow is not None:
                    shadow.close()
                if suite is None:
                    suite = build_suite(spec)
                shadow, cached = None, None
                shadow = (shadow_factory or make_shadow)(env_task, suite, init_states, plus, seed)
                cached = key
            reply = _timed(label_plans, shadow, unpack_snapshot(snapshot, shadow._env.env.sim), before, torch.from_numpy(candidates),
                           first_chunk, norm, layout, calibration)
            connection.send((True, reply))
        except Exception as exc:
            try:
                error = type(exc)(f"{exc}\n{traceback.format_exc()}")
            except Exception:
                error = RuntimeError(traceback.format_exc())
            connection.send((False, error))
    if shadow is not None:
        shadow.close()


class LabelHandle:
    def __init__(self, connections, shape):
        self.connections, self.shape, self.outcome = connections, shape, None

    def wait(self):
        if self.outcome is None:
            replies = [connection.recv() for connection in self.connections]
            errors = [payload for ok, payload in replies if not ok]
            if errors:
                self.outcome = (errors[0], None)
            else:
                labels = np.concatenate([payload[0].reshape(-1) for _, payload in replies]).reshape(self.shape)
                self.outcome = (None, (labels, max(payload[1] for _, payload in replies)))
        return self

    def result(self):
        error, value = self.wait().outcome
        if error is not None:
            raise error
        return value


class LabelPool:
    def __init__(self, workers, spec, norm, layout, calibration, shadow_factory=None):
        context = multiprocessing.get_context("spawn")
        self.connections, self.processes = [], []
        for _ in range(int(workers)):
            parent, child = context.Pipe()
            process = context.Process(target=_serve, args=(child, spec, norm, layout, calibration, shadow_factory), daemon=True)
            process.start()
            child.close()
            self.connections.append(parent)
            self.processes.append(process)

    def submit(self, snapshot, before, candidates, first_chunk, env_task, init_states, plus, seed):
        plans, k = candidates.shape[:2]
        flat = np.ascontiguousarray(candidates, dtype=np.float32).reshape(plans * k, 1, *candidates.shape[2:])
        packed = pack_snapshot(snapshot)
        targets = [(connection, part) for connection, part in zip(self.connections, np.array_split(np.arange(plans * k), len(self.connections)))
                   if len(part)]
        for connection, part in targets:
            connection.send((packed, before, flat[part[0]:part[-1] + 1], first_chunk, env_task, init_states, plus, seed))
        return LabelHandle([connection for connection, _ in targets], (plans, k))

    def close(self):
        for connection in self.connections:
            try:
                connection.send(None)
            except (OSError, EOFError):
                pass
        for process in self.processes:
            process.join(5)
            if process.is_alive():
                process.terminate()
                process.join()
        for connection in self.connections:
            connection.close()
