import numpy as np
import torch

STRUCT_STD_FLOOR = 1e-3
OBJECT_STD_FLOOR = 0.05


def expert_standardization(rows):
    expert = [row for row in rows if float(row["is_expert"]) == 1.0]
    if any(np.asarray(row["object_state"]).size == 0 for row in expert):
        raise ValueError("expert rows carry no object state")
    count = total = squares = 0.0
    for row in expert:
        values = row["state_struct"].to(torch.float64)
        count += values.numel()
        total += float(values.sum())
        squares += float((values * values).sum())
    struct_mean = total / count
    struct_std = max(max(squares / count - struct_mean * struct_mean, 0.0) ** 0.5, STRUCT_STD_FLOOR)
    objects = np.stack([np.asarray(row["object_state"], np.float64) for row in expert])
    return {"struct_mean": struct_mean, "struct_std": struct_std,
            "object_mean": torch.from_numpy(objects.mean(0).astype(np.float32)),
            "object_std": torch.from_numpy(np.maximum(objects.std(0), OBJECT_STD_FLOOR).astype(np.float32))}
