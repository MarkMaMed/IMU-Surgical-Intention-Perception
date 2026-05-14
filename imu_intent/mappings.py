from __future__ import annotations

import numpy as np

INTENT_LABELS = ["IDLE", "PREPARE_LOCK", "PREPARE_UNLOCK", "FINE_OPERATE"]


def map_values_with_dict(values: np.ndarray, mapping: dict[str, str], default_intent: str = "IDLE") -> np.ndarray:
    return np.array([mapping.get(str(v), default_intent) for v in values], dtype=object)


def map_values_with_ranges(values: np.ndarray, ranges: list[dict], default_intent: str = "IDLE") -> np.ndarray:
    out = np.full(values.shape, default_intent, dtype=object)
    for r in ranges:
        start = int(r["start"])
        end = int(r["end"])
        intent = str(r["intent"])
        mask = (values >= start) & (values <= end)
        out[mask] = intent
    return out

