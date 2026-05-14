from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SequenceRecord:
    dataset: str
    sequence_id: str
    signals: np.ndarray  # shape: [time, channels]
    labels: np.ndarray  # shape: [time], string labels
    sample_rate_hz: float
    signal_source: str = "raw"
