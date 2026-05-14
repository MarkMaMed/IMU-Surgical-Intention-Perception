from __future__ import annotations

from collections import Counter

import numpy as np

from .features import extract_window_features
from .types import SequenceRecord


def build_window_dataset(
    records: list[SequenceRecord],
    window_size: int,
    stride: int,
    majority_ratio: float,
    max_windows_per_sequence: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    X_rows: list[np.ndarray] = []
    y_rows: list[str] = []
    meta: list[dict] = []

    for rec in records:
        T = rec.signals.shape[0]
        if T < window_size:
            continue
        kept_in_seq = 0
        for start in range(0, T - window_size + 1, stride):
            end = start + window_size
            labels = rec.labels[start:end]
            count = Counter(labels.tolist())
            major_label, major_count = count.most_common(1)[0]
            ratio = major_count / float(window_size)
            if ratio < majority_ratio:
                continue
            feat = extract_window_features(rec.signals[start:end])
            X_rows.append(feat)
            y_rows.append(str(major_label))
            meta.append(
                {
                    "dataset": rec.dataset,
                    "sequence_id": rec.sequence_id,
                    "signal_source": rec.signal_source,
                    "start": start,
                    "end": end,
                    "sample_rate_hz": rec.sample_rate_hz,
                }
            )
            kept_in_seq += 1
            if max_windows_per_sequence > 0 and kept_in_seq >= max_windows_per_sequence:
                break

    if not X_rows:
        return np.empty((0, 0), dtype=np.float32), np.array([], dtype=object), []
    return np.vstack(X_rows), np.array(y_rows, dtype=object), meta
