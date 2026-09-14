import numpy as np
import torch

from script.lingbot_rl_model import GAMMA, N_STEP_CHUNKS, REPLAY_CAPACITY

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

    def add_online(self, row):
        stored = {**row, "is_expert": np.float32(0.0)}
        stored.setdefault("a_next", np.array(stored["a"], copy=True))
        stored.setdefault("n_steps", np.float32(1.0))
        self._data.append(stored)
        self._trim()

    def add_expert(self, row):
        stored = {
            **row,
            "is_expert": np.float32(1.0),
            "n_steps": np.float32(1.0),
            "a_next": np.array(row["a"], copy=True),
        }
        self._data.append(stored)
        self._trim()
        self._episode_start = len(self._data)

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
        online = [i for i, row in enumerate(self._data) if float(row["is_expert"]) == 0.0]
        expert = [i for i, row in enumerate(self._data) if float(row["is_expert"]) == 1.0]
        if not online:
            raise ValueError("Replay has no online chunks")
        n_expert = int(round(batch_size * expert_ratio)) if expert else 0
        n_expert = min(n_expert, batch_size)
        n_online = batch_size - n_expert
        indices = list(np.random.choice(online, size=n_online, replace=len(online) < n_online))
        if n_expert:
            indices += list(np.random.choice(expert, size=n_expert, replace=len(expert) < n_expert))
        batch = {}
        for key in KEYS:
            stacked = np.stack([self._data[i][key] for i in indices])
            tensor = torch.from_numpy(np.asarray(stacked)).to(self.device)
            if key in ("reward", "done", "n_steps", "is_expert") and tensor.ndim == 1:
                tensor = tensor.unsqueeze(-1)
            batch[key] = tensor
        return batch
