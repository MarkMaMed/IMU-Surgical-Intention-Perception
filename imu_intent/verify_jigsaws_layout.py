from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

from .loaders import inspect_jigsaws_layout


def _load_root_from_config(config_path: Path) -> Path:
    with config_path.open("rb") as f:
        cfg = tomllib.load(f)
    ds = cfg.get("dataset", {}).get("jigsaws", {})
    root = ds.get("root", "data/public_imu/JIGSAWS")
    return Path(root)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify JIGSAWS layout for kinematics/transcriptions readiness.")
    parser.add_argument("--config", default="config/imu_multidataset.toml")
    parser.add_argument("--root", default="")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    root = Path(args.root) if args.root else _load_root_from_config(Path(args.config))
    if not root.exists():
        raise FileNotFoundError(f"JIGSAWS root not found: {root}")

    layout = inspect_jigsaws_layout(root)
    mode = "kinematics" if layout["has_official_like_kinematics"] else "video_fallback"
    result = {
        "root": str(root.resolve()),
        "mode": mode,
        "layout": layout,
        "next_step": (
            "Disable/ignore fallback and retrain."
            if mode == "kinematics"
            else "Import official archives (or provide official links) and rerun verify."
        ),
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved to: {out}")


if __name__ == "__main__":
    main()

