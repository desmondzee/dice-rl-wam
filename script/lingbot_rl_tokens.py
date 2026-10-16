import json
from pathlib import Path

import numpy as np
import torch

SHARD_CHUNKS = 256
FIELDS = ("chunk", "episode", "step", "reset_kind", "variant_index", "plan_index")


class TokenStore:
    def __init__(self, root, ordinal):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index = self.root / "index.jsonl"
        lines = self.index.read_text().splitlines() if self.index.is_file() else []
        if len(lines) < ordinal:
            raise RuntimeError("token index behind the replay")
        self.index.write_text("".join(line + "\n" for line in lines[:ordinal]))
        self.ordinal = int(ordinal)

    def shard_path(self, shard):
        return self.root / f"shard_{shard:04d}.npy"

    def append(self, tokens, **entry):
        if entry["chunk"] != self.ordinal:
            raise ValueError(f"token chunk {entry['chunk']} does not match ordinal {self.ordinal}")
        shard, offset = divmod(self.ordinal, SHARD_CHUNKS)
        array = tokens.detach().to(device="cpu", dtype=torch.bfloat16).contiguous().view(torch.int16).numpy()
        path = self.shard_path(shard)
        if offset == 0 and not path.is_file():
            store = np.lib.format.open_memmap(path, mode="w+", dtype=np.int16, shape=(SHARD_CHUNKS, *array.shape))
        else:
            store = np.load(path, mmap_mode="r+")
        store[offset] = array
        store.flush()
        del store
        entry = {**{key: int(entry[key]) for key in FIELDS}, "shard": shard, "offset": offset}
        with self.index.open("a") as handle:
            handle.write(json.dumps(entry) + "\n")
        self.ordinal += 1
        return entry

    def read(self, entry):
        array = np.array(np.load(self.shard_path(entry["shard"]), mmap_mode="r")[entry["offset"]])
        return torch.from_numpy(array).view(torch.bfloat16)
