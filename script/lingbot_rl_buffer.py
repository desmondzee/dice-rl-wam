import numpy as np
import torch

from script.lingbot_rl_model import GAMMA, K_CANDIDATES, N_STEP_CHUNKS, REPLAY_CAPACITY, STATE_DIM

PLAN_FLOAT_KEYS = ("plan_pool", "plan_pools_all", "plan_a_base_all", "plan_pool_next")
FLOAT_KEYS = (
    "s", "z", "a_base", "a", "reward", "done", "s_next", "a_next", "n_steps", "is_expert",
    "z_all", "a_base_all", "z_next_all", "a_base_next_all", "mc_return", *PLAN_FLOAT_KEYS, "proprio", "proprio_next",
)
BF16_KEYS = ("state_struct", "state_struct_next", "plan_struct", "plan_struct_all", "plan_struct_next")
STORED_BF16_KEYS = ("state_tokens", "state_latent", "plan_latent_all")


def _host_float(value):
    if torch.is_tensor(value):
        value = value.detach().float().cpu().numpy()
    array = np.asarray(value)
    if array.dtype == object:
        raise TypeError("Replay field must be a numeric array, not object dtype")
    return np.array(array, dtype=np.float32, copy=True)


def host_bf16(value):
    return torch.as_tensor(value).detach().to(device="cpu", dtype=torch.bfloat16).contiguous()


def _default(stored, key, make):
    if key not in stored:
        stored[key] = make()

KEYS = (
    "s", "z", "a_base", "a", "reward", "done", "s_next", "a_next",
    "n_steps", "is_expert", "task_id", "n_env_actions",
    "z_all", "a_base_all", "z_next_all", "a_base_next_all", "mc_return", *PLAN_FLOAT_KEYS, "plan_index",
    "proprio", "proprio_next", *BF16_KEYS,
)


class ChunkReplay:
    def __init__(self, capacity=REPLAY_CAPACITY, device="cpu", plans=1):
        self.capacity = capacity
        self.device = device
        self.plans = int(plans)
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

    def _plan_defaults(self, stored):
        _default(stored, "plan_pool", lambda: np.zeros(STATE_DIM, np.float32))
        _default(stored, "plan_pools_all", lambda: np.repeat(_host_float(stored["plan_pool"])[None], self.plans, axis=0))
        stored.setdefault("plan_index", 0)
        _default(stored, "plan_a_base_all", lambda: np.repeat(_host_float(stored["a_base_all"])[None], self.plans, axis=0))
        _default(stored, "plan_pool_next", lambda: np.array(stored["plan_pool"], copy=True))
        if _host_float(stored["plan_pools_all"]).shape[0] != self.plans or \
                _host_float(stored["plan_a_base_all"]).shape[0] != self.plans:
            raise ValueError(f"Rows must carry exactly {self.plans} plans")
        stored["plan_index"] = int(stored["plan_index"])
        return stored

    def _feature_defaults(self, stored):
        _default(stored, "proprio", lambda: np.zeros(8, np.float32))
        _default(stored, "proprio_next", lambda: np.array(stored["proprio"], copy=True))
        _default(stored, "state_struct", lambda: np.repeat(_host_float(stored["s"])[None], 2, axis=0))
        _default(stored, "state_struct_next", lambda: torch.as_tensor(stored["state_struct"]).clone())
        _default(stored, "plan_struct", lambda: np.repeat(_host_float(stored["plan_pool"])[None], 8, axis=0))
        _default(stored, "plan_struct_all", lambda: np.repeat(_host_float(stored["plan_struct"])[None], self.plans, axis=0))
        _default(stored, "plan_struct_next", lambda: torch.as_tensor(stored["plan_struct"]).clone())
        plan_latent = stored.pop("plan_latent", None)
        if plan_latent is not None:
            _default(stored, "plan_latent_all", lambda: np.repeat(_host_float(plan_latent)[None], self.plans, axis=0))
        for key in STORED_BF16_KEYS:
            _default(stored, key, lambda: torch.zeros(0, dtype=torch.bfloat16))
        stored.setdefault("object_state", np.zeros(0, np.float32))
        stored.setdefault("reset_kind", 0)
        stored.setdefault("variant_index", int(stored["task_id"]))
        for key in (*BF16_KEYS, *STORED_BF16_KEYS):
            stored[key] = host_bf16(stored[key])
        if stored["plan_struct_all"].shape[0] != self.plans or \
                stored["plan_latent_all"].numel() and stored["plan_latent_all"].shape[0] != self.plans:
            raise ValueError(f"Rows must carry exactly {self.plans} plans")
        stored["object_state"] = _host_float(stored["object_state"])
        stored["reset_kind"] = int(stored["reset_kind"])
        stored["variant_index"] = int(stored["variant_index"])
        return stored

    def _store(self, row, is_expert):
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
        self._plan_defaults(stored)
        self._feature_defaults(stored)
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

    def finalized_indices(self):
        return range(self._episode_start)

    def ready_online_count(self):
        return sum(float(self._data[index]["is_expert"]) == 0.0 for index in self.finalized_indices())

    def finalize_episode(self):
        episode = self._data[self._episode_start:]
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
            row["plan_pool_next"] = np.array(episode[nxt]["plan_pool"], copy=True)
            row["proprio_next"] = np.array(episode[nxt]["proprio"], copy=True)
            row["state_struct_next"] = episode[nxt]["state_struct"].clone()
            row["plan_struct_next"] = episode[nxt]["plan_struct"].clone()
            row["a_next"] = np.array(episode[nxt]["a"], copy=True)
            row["z_next_all"] = np.array(episode[nxt]["z_all"], copy=True)
            row["a_base_next_all"] = np.array(episode[nxt]["a_base_all"], copy=True)
            row["mc_return"] = np.float32(mc_returns[index])
        self._episode_start = len(self._data)

    def sample(self, batch_size, expert_ratio):
        ready = self.finalized_indices()
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
        batch = {}
        for key in KEYS:
            if key in BF16_KEYS:
                batch[key] = torch.stack([self._data[i][key] for i in indices]).to(self.device)
                continue
            stacked = np.stack([self._data[i][key] for i in indices])
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
            "plans": self.plans,
            "episode_start": self._episode_start,
            "data": self._data,
        }

    def load_state_dict(self, payload):
        self.capacity = payload["capacity"]
        self.plans = int(payload.get("plans", self.plans))
        self._episode_start = payload["episode_start"]
        self._data = payload["data"]
        for row in self._data:
            self._plan_defaults(row)
            self._feature_defaults(row)
            for key in PLAN_FLOAT_KEYS:
                row[key] = _host_float(row[key])
