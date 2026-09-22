import numpy as np
import torch

from script.lingbot_rl_model import GAMMA, K_CANDIDATES, N_STEP_CHUNKS, REPLAY_CAPACITY

FLOAT_KEYS = (
    "s", "z", "a_base", "a", "reward", "done", "s_next", "a_next", "n_steps", "is_expert",
    "z_all", "a_base_all", "z_next_all", "a_base_next_all", "mc_return",
)


def _host_float(value):
    if torch.is_tensor(value):
        value = value.detach().float().cpu().numpy()
    array = np.asarray(value)
    if array.dtype == object:
        raise TypeError("Replay field must be a numeric array, not object dtype")
    return np.array(array, dtype=np.float32, copy=True)

KEYS = (
    "s", "z", "a_base", "a", "reward", "done", "s_next", "a_next",
    "n_steps", "is_expert", "task_id", "n_env_actions",
    "z_all", "a_base_all", "z_next_all", "a_base_next_all", "mc_return",
)


class ChunkReplay:
    def __init__(self, capacity=REPLAY_CAPACITY, device="cpu"):
        self.capacity = capacity
        self.device = device
        self._data = []
        self._open = {}

    def __len__(self):
        return len(self._data) + sum(len(rows) for rows in self._open.values())

    def _open_rows(self):
        return [row for stream in sorted(self._open) for row in self._open[stream]]

    def rows(self):
        return self._data + self._open_rows()

    def _trim(self):
        extra = len(self) - self.capacity
        if extra > 0:
            self._data = self._data[extra:]

    def _store(self, row, is_expert, stream):
        stored = {**row, "is_expert": np.float32(is_expert)}
        stored.setdefault("a_next", stored["a"])
        stored.setdefault("n_steps", np.float32(1.0))
        stored.setdefault("s_next", stored["s"])
        stored.setdefault(
            "z_all", np.repeat(_host_float(stored["z"])[None], K_CANDIDATES, axis=0))
        stored.setdefault(
            "a_base_all", np.repeat(_host_float(stored["a_base"])[None], K_CANDIDATES, axis=0))
        if _host_float(stored["z_all"]).shape[0] != K_CANDIDATES or \
                _host_float(stored["a_base_all"]).shape[0] != K_CANDIDATES:
            raise ValueError(f"Rows must carry exactly {K_CANDIDATES} candidate proposals")
        stored.setdefault("z_next_all", np.array(stored["z_all"], copy=True))
        stored.setdefault("a_base_next_all", np.array(stored["a_base_all"], copy=True))
        stored.setdefault("mc_return", np.float32(0.0))
        for key in FLOAT_KEYS:
            stored[key] = _host_float(stored[key])
        stored["task_id"] = int(stored["task_id"])
        stored["n_env_actions"] = int(stored["n_env_actions"])
        self._open.setdefault(int(stream), []).append(stored)
        self._trim()

    def add_online(self, row, stream=0):
        self._store(row, 0.0, stream)

    def add_expert(self, row, stream=0):
        self._store(row, 1.0, stream)

    def _ready(self):
        ready = [(None, index) for index in range(len(self._data))]
        for stream in sorted(self._open):
            episode = self._open[stream]
            for index, row in enumerate(episode):
                if index == len(episode) - 1 and float(row["done"]) != 1.0:
                    continue
                ready.append((stream, index))
        return ready

    def _row(self, stream, index):
        return self._data[index] if stream is None else self._open[stream][index]

    def has_ready_online(self):
        return any(float(self._row(stream, index)["is_expert"]) == 0.0 for stream, index in self._ready())

    def _n_step_view(self, stream, index):
        row = self._row(stream, index)
        if stream is None:
            return row
        episode = self._open[stream]
        length = len(episode)
        n_steps = min(N_STEP_CHUNKS, length - index)
        ret = 0.0
        done_n = 0.0
        used = n_steps
        for lag in range(n_steps):
            ret += (GAMMA ** lag) * float(episode[index + lag]["reward"])
            if float(episode[index + lag]["done"]) == 1.0:
                done_n = 1.0
                used = lag + 1
                break
        nxt = index + used if index + used < length else length - 1
        view = dict(row)
        view["reward"] = np.float32(ret)
        view["done"] = np.float32(done_n)
        view["n_steps"] = np.float32(used)
        view["s_next"] = np.array(episode[nxt]["s"], copy=True)
        view["a_next"] = np.array(episode[nxt]["a"], copy=True)
        view["z_next_all"] = np.array(episode[nxt]["z_all"], copy=True)
        view["a_base_next_all"] = np.array(episode[nxt]["a_base_all"], copy=True)
        view["mc_return"] = np.float32(0.0)
        return view

    def finalize_episode(self, stream=0):
        episode = self._open.pop(int(stream), [])
        length = len(episode)
        rewards = [float(row["reward"]) for row in episode]
        dones = [float(row["done"]) for row in episode]
        mc_returns = [0.0] * length
        running = 0.0
        for index in range(length - 1, -1, -1):
            running = rewards[index] + GAMMA * running
            mc_returns[index] = running
        for index, row in enumerate(episode):
            n_steps = min(N_STEP_CHUNKS, length - index)
            ret = 0.0
            done_n = 0.0
            for lag in range(n_steps):
                ret += (GAMMA ** lag) * rewards[index + lag]
                if dones[index + lag]:
                    done_n = 1.0
            nxt = index + n_steps if index + n_steps < length else length - 1
            row["reward"] = np.float32(ret)
            row["done"] = np.float32(done_n)
            row["n_steps"] = np.float32(n_steps)
            row["s_next"] = np.array(episode[nxt]["s"], copy=True)
            row["a_next"] = np.array(episode[nxt]["a"], copy=True)
            row["z_next_all"] = np.array(episode[nxt]["z_all"], copy=True)
            row["a_base_next_all"] = np.array(episode[nxt]["a_base_all"], copy=True)
            row["mc_return"] = np.float32(mc_returns[index])
        self._data.extend(episode)

    def sample(self, batch_size, expert_ratio):
        ready = self._ready()
        online = [i for i, key in enumerate(ready) if float(self._row(*key)["is_expert"]) == 0.0]
        expert = [i for i, key in enumerate(ready) if float(self._row(*key)["is_expert"]) == 1.0]
        if not online:
            raise ValueError("Replay has no online chunks")
        n_expert = int(round(batch_size * expert_ratio)) if expert else 0
        n_expert = min(n_expert, batch_size)
        n_online = batch_size - n_expert
        indices = list(np.random.choice(online, size=n_online, replace=len(online) < n_online))
        if n_expert:
            indices += list(np.random.choice(expert, size=n_expert, replace=len(expert) < n_expert))
        views = [self._n_step_view(*ready[i]) for i in indices]
        batch = {}
        for key in KEYS:
            stacked = np.stack([view[key] for view in views])
            if key in FLOAT_KEYS:
                stacked = np.asarray(stacked, dtype=np.float32)
            tensor = torch.from_numpy(np.ascontiguousarray(stacked)).to(self.device)
            if key in ("reward", "done", "n_steps", "is_expert", "mc_return") and tensor.ndim == 1:
                tensor = tensor.unsqueeze(-1)
            batch[key] = tensor
        return batch

    def state_dict(self):
        return {
            "capacity": self.capacity,
            "episode_start": len(self._data),
            "data": self._data + list(self._open.get(0, [])),
        }

    def load_state_dict(self, payload):
        self.capacity = payload["capacity"]
        start = int(payload["episode_start"])
        data = list(payload["data"])
        self._data = data[:start]
        self._open = {0: data[start:]} if data[start:] else {}
