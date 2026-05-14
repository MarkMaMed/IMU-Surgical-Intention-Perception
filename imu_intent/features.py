from __future__ import annotations

import numpy as np

FEATURE_KEYS = [
    "mean",
    "std",
    "min",
    "max",
    "range",
    "rms",
    "mean_abs",
    "energy",
    "diff_rms",
    "zero_cross_rate",
]


def _zero_cross_rate(x: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    return float(np.mean((x[:-1] * x[1:]) < 0))


def extract_window_features(window: np.ndarray) -> np.ndarray:
    """Extract fixed-size, explainable features from [window_size, channels]."""
    feats: list[float] = []
    for c in range(window.shape[1]):
        x = window[:, c]
        dx = np.diff(x)
        feats.extend(
            [
                float(np.mean(x)),
                float(np.std(x)),
                float(np.min(x)),
                float(np.max(x)),
                float(np.ptp(x)),
                float(np.sqrt(np.mean(np.square(x)))),
                float(np.mean(np.abs(x))),
                float(np.mean(np.square(x))),
                float(np.sqrt(np.mean(np.square(dx)))) if dx.size > 0 else 0.0,
                _zero_cross_rate(x),
            ]
        )
    return np.array(feats, dtype=np.float32)


def feature_names(channels: list[str]) -> list[str]:
    output: list[str] = []
    for ch in channels:
        for key in FEATURE_KEYS:
            output.append(f"{ch}_{key}")
    return output

