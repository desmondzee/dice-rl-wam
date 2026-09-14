import numpy as np
import torch

from script.lingbot_rl_model import GAMMA, N_STEP_CHUNKS, REPLAY_CAPACITY

FLOAT_KEYS = ("s", "z", "a_base", "a", "reward", "done", "s_next", "a_next", "n_steps", "is_expert")


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
)


class ChunkReplay:
    def __init__(self, capacity=REPLAY_CAPACITY, device="cpu"):
        self.capacity = capacity
        self.device = device
        self._data = []
        self._episode_start = 0

    def __len__(self):
        return len(self._data)

    def rows(self):
        return self._data

    def _trim(self):
        extra = len(self._data) - self.capacity
        if extra > 0:
            self._data = self._data[extra:]
            self._episode_start = max(0, self._episode_start - extra)

    def _store(self, row, is_expert):
        stored = {**row, "is_expert": np.float32(is_expert)}
        stored.setdefault("a_next", stored["a"])
        stored.setdefault("n_steps", np.float32(1.0))
        stored.setdefault("s_next", stored["s"])
        for key in FLOAT_KEYS:
            stored[key] = _host_float(stored[key])
        stored["task_id"] = int(stored["task_id"])
        stored["n_env_actions"] = int(stored["n_env_actions"])
        self._data.append(stored)
        self._trim()

    def add_online(self, row):
        self._store(row, 0.0)

    def add_expert(self, row):
        self._store(row, 1.0)

    def _is_incomplete_last(self, index):
        return (
            index >= self._episode_start
            and index == len(self._data) - 1
            and float(self._data[index]["done"]) != 1.0
        )

    def ready_indices(self):
        return [index for index in range(len(self._data)) if not self._is_incomplete_last(index)]

    def has_ready_online(self):
        return any(float(self._data[index]["is_expert"]) == 0.0 for index in self.ready_indices())

    def _n_step_view(self, index):
        row = self._data[index]
        if index < self._episode_start:
            return row
        episode = self._data[self._episode_start:]
        local = index - self._episode_start
        length = len(episode)
        n_steps = min(N_STEP_CHUNKS, length - local)
        ret = 0.0
        done_n = 0.0
        used = n_steps
        for lag in range(n_steps):
            ret += (GAMMA ** lag) * float(episode[local + lag]["reward"])
            if float(episode[local + lag]["done"]) == 1.0:
                done_n = 1.0
                used = lag + 1
                break
        nxt = local + used if local + used < length else length - 1
        view = dict(row)
        view["reward"] = np.float32(ret)
        view["done"] = np.float32(done_n)
        view["n_steps"] = np.float32(used)
        view["s_next"] = np.array(episode[nxt]["s"], copy=True)
        view["a_next"] = np.array(episode[nxt]["a"], copy=True)
        return view

    def finalize_episode(self):
        episode = self._data[self._episode_start:]
        length = len(episode)
        rewards = [float(row["reward"]) for row in episode]
        dones = [float(row["done"]) for row in episode]
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
        self._episode_start = len(self._data)

    def sample(self, batch_size, expert_ratio):
        ready = self.ready_indices()
        online = [i for i in ready if float(self._data[i]["is_expert"]) == 0.0]
        expert = [i for i in ready if float(self._data[i]["is_expert"]) == 1.0]
        if not online:
            raise ValueError("Replay has no online chunks")
        n_expert = int(round(batch_size * expert_ratio)) if expert else 0
        n_expert = min(n_expert, batch_size)
        n_online = batch_size - n_expert
        indices = list(np.random.choice(online, size=n_online, replace=len(online) < n_online))
        if n_expert:
            indices += list(np.random.choice(expert, size=n_expert, replace=len(expert) < n_expert))
        views = [self._n_step_view(i) for i in indices]
        batch = {}
        for key in KEYS:
            stacked = np.stack([view[key] for view in views])
            if key in FLOAT_KEYS:
                stacked = np.asarray(stacked, dtype=np.float32)
            tensor = torch.from_numpy(np.ascontiguousarray(stacked)).to(self.device)
            if key in ("reward", "done", "n_steps", "is_expert") and tensor.ndim == 1:
                tensor = tensor.unsqueeze(-1)
            batch[key] = tensor
        return batch

    def state_dict(self):
        return {
            "capacity": self.capacity,
            "episode_start": self._episode_start,
            "data": self._data,
        }

    def load_state_dict(self, payload):
        self.capacity = payload["capacity"]
        self._episode_start = payload["episode_start"]
        self._data = payload["data"]
