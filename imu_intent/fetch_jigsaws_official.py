from __future__ import annotations

import argparse
import shutil
import sys
import tomllib
import urllib.request
import zipfile
from pathlib import Path

from .loaders import inspect_jigsaws_layout


def _load_root_from_config(config_path: Path) -> Path:
    with config_path.open("rb") as f:
        cfg = tomllib.load(f)
    ds = cfg.get("dataset", {}).get("jigsaws", {})
    root = ds.get("root", "data/public_imu/JIGSAWS")
    return Path(root)


def _download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as resp:  # nosec B310
        if getattr(resp, "status", 200) >= 400:
            raise RuntimeError(f"HTTP error {resp.status} for {url}")
        with out_path.open("wb") as f:
            shutil.copyfileobj(resp, f, length=1024 * 1024)


def _extract_zip(zip_path: Path, out_dir: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)


def _has_input(args: argparse.Namespace) -> bool:
    return any(
        [
            args.suturing_url,
            args.knot_url,
            args.needle_url,
            args.suturing_zip,
            args.knot_zip,
            args.needle_zip,
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch official JIGSAWS archives from email links and extract to dataset root. "
            "This script cannot bypass the official reCAPTCHA form."
        )
    )
    parser.add_argument("--config", default="config/imu_multidataset.toml")
    parser.add_argument("--root", default="")
    parser.add_argument("--tmp-dir", default="data/public_imu/tmp/jigsaws_official")
    parser.add_argument("--suturing-url", default="")
    parser.add_argument("--knot-url", default="")
    parser.add_argument("--needle-url", default="")
    parser.add_argument("--suturing-zip", default="")
    parser.add_argument("--knot-zip", default="")
    parser.add_argument("--needle-zip", default="")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    root = Path(args.root) if args.root else _load_root_from_config(Path(args.config))
    tmp_dir = Path(args.tmp_dir)
    root.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print("JIGSAWS dataset root:", root.resolve())
    print("Official access form: https://www.cs.jhu.edu/~los/jigsaws/info.php")
    print("Note: official source requires form + reCAPTCHA + emailed links.")

    if args.verify_only or not _has_input(args):
        layout = inspect_jigsaws_layout(root)
        print("Current layout:", layout)
        if not _has_input(args):
            print(
                "\nNo download input provided. After receiving official links, run for example:\n"
                "python3 -m imu_intent.fetch_jigsaws_official "
                "--suturing-url '<email_link_1>' "
                "--knot-url '<email_link_2>' "
                "--needle-url '<email_link_3>'"
            )
        return

    items = [
        ("Suturing", args.suturing_url, args.suturing_zip),
        ("Knot_Tying", args.knot_url, args.knot_zip),
        ("Needle_Passing", args.needle_url, args.needle_zip),
    ]
    acquired: list[Path] = []

    for name, url, local_zip in items:
        if local_zip:
            p = Path(local_zip)
            if not p.exists():
                raise FileNotFoundError(f"Local zip not found for {name}: {p}")
            acquired.append(p)
            print(f"[{name}] using local zip:", p)
            continue

        if not url:
            continue
        out_path = tmp_dir / f"{name}.zip"
        print(f"[{name}] downloading from url ...")
        _download(url=url, out_path=out_path)
        print(f"[{name}] downloaded to:", out_path)
        acquired.append(out_path)

    if not acquired:
        print("No zip archives acquired. Nothing to extract.")
        return

    for z in acquired:
        print("Extracting:", z)
        _extract_zip(zip_path=z, out_dir=root)

    layout = inspect_jigsaws_layout(root)
    print("Post-extract layout:", layout)
    if not layout["has_official_like_kinematics"]:
        print(
            "Warning: official-like kinematics/transcriptions still not detected. "
            "Check whether emailed links are complete dataset archives."
        )
        sys.exit(2)

    print("Success: official-like kinematics/transcriptions detected. Fallback can be disabled.")


if __name__ == "__main__":
    main()

