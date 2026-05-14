from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.io import loadmat

from .mappings import map_values_with_dict, map_values_with_ranges
from .types import SequenceRecord

CANONICAL_CHANNELS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]


def _normalize_channels(data: np.ndarray, col_indices: list[int] | None = None) -> np.ndarray:
    if data.ndim != 2:
        raise ValueError("Expected 2D data [time, channels].")
    if col_indices is not None and len(col_indices) > 0:
        data = data[:, col_indices]
    if data.shape[1] >= 6:
        return data[:, :6].astype(np.float32)
    # Pad missing channels with zeros to keep model input fixed.
    padded = np.zeros((data.shape[0], 6), dtype=np.float32)
    padded[:, : data.shape[1]] = data.astype(np.float32)
    return padded


def inspect_jigsaws_layout(root: Path) -> dict:
    kin_files: list[Path] = []
    trans_files: list[Path] = []
    video_files: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        low = p.as_posix().lower()
        if p.suffix.lower() in {".avi", ".mp4", ".mov"} and "video" in low:
            video_files.append(p)
        elif p.suffix.lower() == ".txt" and "kinemat" in low:
            kin_files.append(p)
        elif p.suffix.lower() == ".txt" and "trans" in low:
            trans_files.append(p)

    trans_by_stem = {p.stem for p in trans_files}
    paired_kin = sum(1 for p in kin_files if p.stem in trans_by_stem)
    return {
        "kinematics_file_count": len(kin_files),
        "transcriptions_file_count": len(trans_files),
        "paired_trials": int(paired_kin),
        "video_file_count": len(video_files),
        "has_official_like_kinematics": bool(paired_kin > 0),
    }


def load_pamap2_records(cfg: dict) -> list[SequenceRecord]:
    root = Path(cfg["root"])
    if not root.exists():
        return []
    records: list[SequenceRecord] = []
    file_glob = cfg.get("file_glob", "*.dat")
    signal_cols = [int(x) for x in cfg.get("signal_column_indices", [7, 8, 9, 10, 11, 12])]
    label_col = int(cfg.get("label_column_index", 1))
    signal_names = list(cfg.get("signal_column_names", []))
    label_name = str(cfg.get("label_column_name", "")).strip()
    mapping = {str(k): str(v) for k, v in cfg.get("intent_map", {}).items()}
    default_intent = str(cfg.get("default_intent", "IDLE"))
    sample_rate = float(cfg.get("sample_rate_hz", 100.0))

    for path in sorted(root.glob(file_glob)):
        if path.suffix.lower() == ".csv":
            if signal_names and label_name:
                usecols = signal_names + [label_name]
                df = pd.read_csv(path, usecols=usecols)
                signals = df[signal_names].to_numpy(dtype=float)
                labels_raw = df[label_name].fillna(0).to_numpy(dtype=int)
            else:
                max_idx = max(signal_cols + [label_col])
                df = pd.read_csv(path)
                if df.shape[1] <= max_idx:
                    continue
                signals = df.iloc[:, signal_cols].to_numpy(dtype=float)
                labels_raw = df.iloc[:, label_col].fillna(0).to_numpy(dtype=int)
        else:
            max_idx = max(signal_cols + [label_col])
            usecols = sorted(set(signal_cols + [label_col]))
            df = pd.read_csv(path, sep=r"\s+", header=None, engine="python", usecols=usecols)
            if df.shape[1] < len(usecols):
                continue
            col_to_pos = {c: i for i, c in enumerate(usecols)}
            signals = df.iloc[:, [col_to_pos[c] for c in signal_cols]].to_numpy(dtype=float)
            labels_raw = df.iloc[:, col_to_pos[label_col]].fillna(0).to_numpy(dtype=int)

        valid_mask = np.isfinite(signals).all(axis=1)
        signals = signals[valid_mask]
        labels_raw = labels_raw[valid_mask]
        labels = map_values_with_dict(labels_raw, mapping=mapping, default_intent=default_intent)
        records.append(
            SequenceRecord(
                dataset="PAMAP2",
                sequence_id=path.stem,
                signals=_normalize_channels(signals),
                labels=labels,
                sample_rate_hz=sample_rate,
                signal_source="raw",
            )
        )
    return records


def load_opportunity_records(cfg: dict) -> list[SequenceRecord]:
    root = Path(cfg["root"])
    if not root.exists():
        return []
    records: list[SequenceRecord] = []
    file_glob = cfg.get("file_glob", "S*-ADL*.dat")
    signal_cols = [int(x) for x in cfg.get("signal_column_indices", [37, 38, 39, 40, 41, 42])]
    label_col = int(cfg.get("label_column_index", 243))
    mapping = {str(k): str(v) for k, v in cfg.get("intent_map", {}).items()}
    default_intent = str(cfg.get("default_intent", "IDLE"))
    sample_rate = float(cfg.get("sample_rate_hz", 30.0))

    for path in sorted(root.glob(file_glob)):
        usecols = sorted(set(signal_cols + [label_col]))
        df = pd.read_csv(path, sep=r"\s+", header=None, engine="python", usecols=usecols)
        if df.shape[1] < len(usecols):
            continue
        col_to_pos = {c: i for i, c in enumerate(usecols)}
        signals = df.iloc[:, [col_to_pos[c] for c in signal_cols]].to_numpy(dtype=float)
        labels_raw = df.iloc[:, col_to_pos[label_col]].fillna(0).to_numpy(dtype=int)
        valid_mask = np.isfinite(signals).all(axis=1)
        signals = signals[valid_mask]
        labels_raw = labels_raw[valid_mask]
        labels = map_values_with_dict(labels_raw, mapping=mapping, default_intent=default_intent)
        records.append(
            SequenceRecord(
                dataset="Opportunity",
                sequence_id=path.stem,
                signals=_normalize_channels(signals),
                labels=labels,
                sample_rate_hz=sample_rate,
                signal_source="raw",
            )
        )
    return records


def load_ninapro_records(cfg: dict) -> list[SequenceRecord]:
    root = Path(cfg["root"])
    if not root.exists():
        return []

    records: list[SequenceRecord] = []
    file_glob = cfg.get("file_glob", "**/*.mat")
    signal_key_priority = list(cfg.get("signal_key_priority", ["acc", "inclinometer", "glove", "emg"]))
    label_key_priority = list(cfg.get("label_key_priority", ["restimulus", "stimulus", "label"]))
    signal_cols = [int(x) for x in cfg.get("signal_column_indices", [0, 1, 2, 3, 4, 5])]
    ranges = list(cfg.get("intent_ranges", []))
    default_intent = str(cfg.get("default_intent", "IDLE"))
    sample_rate = float(cfg.get("sample_rate_hz", 100.0))

    for path in sorted(root.glob(file_glob)):
        mat = loadmat(path)
        signal_key = next((k for k in signal_key_priority if k in mat and isinstance(mat[k], np.ndarray)), None)
        label_key = next((k for k in label_key_priority if k in mat and isinstance(mat[k], np.ndarray)), None)
        if signal_key is None or label_key is None:
            continue
        signals = np.asarray(mat[signal_key], dtype=float)
        labels_raw = np.asarray(mat[label_key]).reshape(-1)
        if signals.ndim != 2:
            continue
        min_len = min(signals.shape[0], labels_raw.shape[0])
        signals = signals[:min_len]
        labels_raw = labels_raw[:min_len].astype(int)
        labels = map_values_with_ranges(labels_raw, ranges=ranges, default_intent=default_intent)
        records.append(
            SequenceRecord(
                dataset="NinaPro",
                sequence_id=path.stem,
                signals=_normalize_channels(signals, col_indices=signal_cols),
                labels=labels,
                sample_rate_hz=sample_rate,
                signal_source="raw",
            )
        )
    return records


def load_jigsaws_records(cfg: dict) -> list[SequenceRecord]:
    root = Path(cfg["root"])
    if not root.exists():
        return []

    records: list[SequenceRecord] = []
    intent_map = {str(k): str(v) for k, v in cfg.get("intent_map", {}).items()}
    default_intent = str(cfg.get("default_intent", "IDLE"))
    sample_rate = float(cfg.get("sample_rate_hz", 30.0))
    col_indices = [int(x) for x in cfg.get("signal_column_indices", [0, 1, 2, 3, 4, 5])]
    index_base = int(cfg.get("transcription_index_base", 0))

    layout = inspect_jigsaws_layout(root)
    kin_files = []
    for p in root.rglob("*.txt"):
        low = p.as_posix().lower()
        if "kinemat" in low:
            kin_files.append(p)
    trans_files = []
    for p in root.rglob("*.txt"):
        low = p.as_posix().lower()
        if "trans" in low:
            trans_files.append(p)

    trans_by_stem = {p.stem: p for p in trans_files}
    for kin_path in sorted(kin_files):
        trans_path = trans_by_stem.get(kin_path.stem)
        if trans_path is None:
            continue
        try:
            signals_raw = np.loadtxt(kin_path)
        except Exception:
            continue
        if signals_raw.ndim == 1:
            signals_raw = signals_raw.reshape(-1, 1)
        signals = _normalize_channels(signals_raw, col_indices=col_indices)
        labels = np.full(signals.shape[0], default_intent, dtype=object)

        try:
            with trans_path.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    toks = line.strip().split()
                    if len(toks) < 3:
                        continue
                    start = int(float(toks[0])) - index_base
                    end = int(float(toks[1])) - index_base
                    gesture = toks[2]
                    intent = intent_map.get(gesture, default_intent)
                    s = max(0, start)
                    e = min(signals.shape[0] - 1, end)
                    if s <= e:
                        labels[s : e + 1] = intent
        except Exception:
            continue

        records.append(
            SequenceRecord(
                dataset="JIGSAWS",
                sequence_id=kin_path.stem,
                signals=signals,
                labels=labels,
                sample_rate_hz=sample_rate,
                signal_source="kinematics",
            )
        )

    # Fallback for mirrors without kinematics/transcriptions: extract coarse motion kinematics from videos.
    if records:
        return records
    if not bool(cfg.get("use_video_fallback", False)):
        return records
    if bool(cfg.get("warn_when_fallback", True)):
        print(
            "[JIGSAWS] kinematics/transcriptions not found, switching to video fallback.",
            f"layout={layout}",
        )

    video_glob = str(cfg.get("video_glob", "**/video/*.avi"))
    sample_every = max(1, int(cfg.get("video_sample_every_n_frames", 5)))
    w = int(cfg.get("video_resize_width", 256))
    h = int(cfg.get("video_resize_height", 192))
    max_files = max(1, int(cfg.get("video_max_files", 24)))
    prep_ratio = float(cfg.get("video_phase_prepare_ratio", 0.15))
    default_rate = float(cfg.get("sample_rate_hz", 30.0))

    def _video_signals(path: Path) -> tuple[np.ndarray, float]:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return np.empty((0, 6), dtype=np.float32), default_rate
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        fps = fps if fps > 1e-3 else default_rate

        prev = None
        rows: list[list[float]] = []
        fidx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if fidx % sample_every != 0:
                fidx += 1
                continue
            gray = cv2.cvtColor(cv2.resize(frame, (w, h)), cv2.COLOR_BGR2GRAY)
            grayf = gray.astype(np.float32)
            if prev is None:
                prev = grayf
                fidx += 1
                continue
            diff = cv2.absdiff(prev.astype(np.uint8), gray.astype(np.uint8))
            _, mask = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
            motion_energy = float(np.count_nonzero(mask)) / float(mask.size)

            m = cv2.moments(mask)
            if m["m00"] > 1e-6:
                cx = float(m["m10"] / m["m00"]) / float(w)
                cy = float(m["m01"] / m["m00"]) / float(h)
            else:
                cx = 0.5
                cy = 0.5
            shift, _ = cv2.phaseCorrelate(prev, grayf)
            dx, dy = float(shift[0]) / float(w), float(shift[1]) / float(h)
            mean_diff = float(np.mean(diff)) / 255.0
            std_diff = float(np.std(diff)) / 255.0

            rows.append([motion_energy, cx, cy, dx, dy, mean_diff + std_diff])
            prev = grayf
            fidx += 1
        cap.release()
        return np.array(rows, dtype=np.float32), fps / float(sample_every)

    videos = sorted(root.glob(video_glob))[:max_files]
    for v in videos:
        sig, sr = _video_signals(v)
        if sig.shape[0] < 20:
            continue
        n = sig.shape[0]
        a = int(n * prep_ratio)
        b = int(n * (1.0 - prep_ratio))
        labels = np.array(["PREPARE_LOCK"] * n, dtype=object)
        labels[a:b] = "FINE_OPERATE"
        labels[b:] = "PREPARE_UNLOCK"
        records.append(
            SequenceRecord(
                dataset="JIGSAWS",
                sequence_id=v.stem,
                signals=sig,
                labels=labels,
                sample_rate_hz=sr,
                signal_source="video_fallback",
            )
        )
    return records


def load_all_enabled_records(cfg: dict) -> list[SequenceRecord]:
    ds_cfg = cfg.get("dataset", {})
    all_records: list[SequenceRecord] = []

    if ds_cfg.get("jigsaws", {}).get("enabled", False):
        all_records.extend(load_jigsaws_records(ds_cfg["jigsaws"]))
    if ds_cfg.get("opportunity", {}).get("enabled", False):
        all_records.extend(load_opportunity_records(ds_cfg["opportunity"]))
    if ds_cfg.get("ninapro", {}).get("enabled", False):
        all_records.extend(load_ninapro_records(ds_cfg["ninapro"]))
    if ds_cfg.get("pamap2", {}).get("enabled", False):
        all_records.extend(load_pamap2_records(ds_cfg["pamap2"]))
    return all_records
