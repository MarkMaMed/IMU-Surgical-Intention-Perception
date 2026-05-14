from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import SequenceRecord


@dataclass
class MockIntentSegment:
    intent: str
    duration_s: float


def _segment_signal(intent: str, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n, endpoint=False)
    if intent == "PREPARE_LOCK":
        base = 0.02 * rng.normal(size=(n, 6))
    elif intent == "PREPARE_UNLOCK":
        wave = np.stack(
            [
                0.7 * np.sin(2 * np.pi * 1.0 * t),
                0.5 * np.cos(2 * np.pi * 1.3 * t),
                0.6 * np.sin(2 * np.pi * 0.8 * t + 0.3),
                0.3 * np.sin(2 * np.pi * 1.8 * t),
                0.2 * np.cos(2 * np.pi * 2.2 * t),
                0.25 * np.sin(2 * np.pi * 1.5 * t),
            ],
            axis=1,
        )
        base = wave + 0.08 * rng.normal(size=(n, 6))
    elif intent == "FINE_OPERATE":
        wave = np.stack(
            [
                0.15 * np.sin(2 * np.pi * 6.0 * t),
                0.12 * np.cos(2 * np.pi * 7.0 * t),
                0.10 * np.sin(2 * np.pi * 8.0 * t + 0.2),
                0.20 * np.sin(2 * np.pi * 9.0 * t),
                0.18 * np.cos(2 * np.pi * 10.0 * t),
                0.16 * np.sin(2 * np.pi * 8.5 * t),
            ],
            axis=1,
        )
        base = wave + 0.03 * rng.normal(size=(n, 6))
    else:
        base = 0.01 * rng.normal(size=(n, 6))
    return base.astype(np.float32)


def make_synthetic_records(sample_rate_hz: float = 100.0) -> list[SequenceRecord]:
    records: list[SequenceRecord] = []
    plans = [
        [
            MockIntentSegment("PREPARE_LOCK", 4.0),
            MockIntentSegment("FINE_OPERATE", 4.0),
            MockIntentSegment("PREPARE_UNLOCK", 4.0),
            MockIntentSegment("PREPARE_LOCK", 3.0),
        ],
        [
            MockIntentSegment("PREPARE_UNLOCK", 3.0),
            MockIntentSegment("PREPARE_LOCK", 3.5),
            MockIntentSegment("FINE_OPERATE", 3.5),
            MockIntentSegment("PREPARE_UNLOCK", 3.0),
        ],
    ]

    for ridx, plan in enumerate(plans):
        xs: list[np.ndarray] = []
        ys: list[str] = []
        for sidx, seg in enumerate(plan):
            n = int(seg.duration_s * sample_rate_hz)
            xs.append(_segment_signal(seg.intent, n=n, seed=100 + ridx * 10 + sidx))
            ys.extend([seg.intent] * n)
        signals = np.vstack(xs)
        labels = np.array(ys, dtype=object)
        records.append(
            SequenceRecord(
                dataset="SYNTHETIC",
                sequence_id=f"synthetic_{ridx}",
                signals=signals,
                labels=labels,
                sample_rate_hz=sample_rate_hz,
            )
        )
    return records

