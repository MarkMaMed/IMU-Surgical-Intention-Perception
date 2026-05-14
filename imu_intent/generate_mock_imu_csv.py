from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .synthetic import make_synthetic_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic IMU CSV for quick inference demo.")
    parser.add_argument("--output", default="demo/mock_imu_stream.csv")
    parser.add_argument("--sample-rate", type=float, default=100.0)
    args = parser.parse_args()

    records = make_synthetic_records(sample_rate_hz=args.sample_rate)
    first = records[0]
    df = pd.DataFrame(
        first.signals,
        columns=["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
    )
    df["true_intent"] = first.labels
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Mock IMU stream saved to: {output}")


if __name__ == "__main__":
    main()

