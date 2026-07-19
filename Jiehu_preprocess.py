"""
jiehu_tremor_preprocessor.py
─────────────────────────────
Preprocessing pipeline for the Jiehu (Rima) dataset.
Follows the same PDF pipeline spec as kaggle_tremor_preprocessor.py.

Key differences from the Kaggle dataset:
- Single CSV (df_all_timesteps.csv) with per-sample tremor labels already present
  — no need for AR PSD label generation (Steps 2-4)
- Already at 50 Hz — Step 0 upsamples 50→100 Hz
- Split on segment_id (patient/session) instead of per-patient files

Usage
─────
    python jiehu_tremor_preprocessor.py \
        --csv_path  "Jiehu Set (Rima)/df_all_timesteps.csv" \
        --out_dir   artifacts
"""

import argparse
import os
from math import gcd

import numpy as np
import pandas as pd
from scipy.signal import firwin, filtfilt, resample_poly
from sklearn.model_selection import train_test_split


FS_SOURCE  = 50            # Hz – Jiehu dataset sampling rate
FS_TARGET  = 100           # Hz
WIN_SEC    = 3.0
OVERLAP    = 0.90
WIN_SAMPLES = int(FS_TARGET * WIN_SEC)   # 300
HOP_SAMPLES = max(1, int(WIN_SAMPLES * (1.0 - OVERLAP)))  # 30
RANDOM_SEED = 42


# ── Step 0: Upsample 50 → 100 Hz ─────────────────────────────────────────────

def resample_to_100hz(signal: np.ndarray, fs_src: float = FS_SOURCE) -> np.ndarray:
    if abs(fs_src - FS_TARGET) < 0.5:
        return signal.astype(np.float64)

    lp_cutoff = min(fs_src / 2.0, FS_TARGET / 2.0) - 0.5
    lp_cutoff = max(lp_cutoff, 1.0)

    n = len(signal)
    numtaps = min(64 * 2 + 1, (n - 1) // 3)
    if numtaps % 2 == 0:
        numtaps -= 1
    numtaps = max(numtaps, 3)
    lp = firwin(numtaps, lp_cutoff / (fs_src / 2.0))
    signal = filtfilt(lp, [1.0], signal.astype(np.float64))

    fs_src_int = int(round(fs_src))
    g    = gcd(fs_src_int, FS_TARGET)
    up   = FS_TARGET   // g
    down = fs_src_int  // g
    return resample_poly(signal, up, down, padtype="line")


def resample_labels(labels: np.ndarray, fs_src: float = FS_SOURCE) -> np.ndarray:
    """Upsample binary labels by nearest-neighbour (no interpolation)."""
    if abs(fs_src - FS_TARGET) < 0.5:
        return labels.astype(np.int8)
    fs_src_int = int(round(fs_src))
    g    = gcd(fs_src_int, FS_TARGET)
    up   = FS_TARGET   // g
    down = fs_src_int  // g
    # repeat each sample `up` times then thin by `down`
    upsampled = np.repeat(labels, up)
    return upsampled[::down].astype(np.int8)


# ── Step 1: Drift removal + bandpass ─────────────────────────────────────────

def drift_remove_ma(x: np.ndarray, fs: int = FS_TARGET, seconds: float = 5.0) -> np.ndarray:
    L = max(int(fs * seconds), 3)
    L = min(L, max(3, (len(x) - 1) // 3))
    ma = np.ones(L, dtype=float) / L
    return x - filtfilt(ma, [1.0], x)


def bandpass_fir(x: np.ndarray, fs: int = FS_TARGET,
                 lo: float = 1.0, hi: float = 30.0, numtaps: int = 201) -> np.ndarray:
    numtaps = min(numtaps, max(3, (len(x) - 1) // 3))
    if numtaps % 2 == 0:
        numtaps -= 1
    numtaps = max(numtaps, 3)
    bp = firwin(numtaps, [lo, hi], pass_zero=False, fs=fs)
    return filtfilt(bp, [1.0], x)


# ── Windowing ─────────────────────────────────────────────────────────────────

def window_xyz(ax, ay, az, y):
    win = WIN_SAMPLES
    hop = HOP_SAMPLES
    ham = np.hamming(win).astype(np.float32)

    X_out, y_out = [], []
    for i in range(0, len(ax) - win + 1, hop):
        seg = np.stack([ax[i:i+win], ay[i:i+win], az[i:i+win]], axis=1)
        X_out.append((seg * ham[:, None]).astype(np.float32))
        lab = y[i:i+win]
        y_out.append(int(np.bincount(lab.astype(int)).argmax()))

    return np.asarray(X_out, dtype=np.float32), np.asarray(y_out, dtype=np.int64)


# ── Per-segment pipeline ──────────────────────────────────────────────────────

def process_segment(ax, ay, az, labels):
    """Run Steps 0–1 on one segment's raw arrays. Labels upsampled by NN."""
    ax = resample_to_100hz(ax)
    ay = resample_to_100hz(ay)
    az = resample_to_100hz(az)
    y  = resample_labels(labels)

    # Trim to same length in case of off-by-one after resampling
    n = min(len(ax), len(ay), len(az), len(y))
    ax, ay, az, y = ax[:n], ay[:n], az[:n], y[:n]

    ax = bandpass_fir(drift_remove_ma(ax))
    ay = bandpass_fir(drift_remove_ma(ay))
    az = bandpass_fir(drift_remove_ma(az))

    return ax, ay, az, y


# ── Main ──────────────────────────────────────────────────────────────────────

def main(csv_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    print("Loading CSV...")
    df = pd.read_csv(csv_path)

    # Segment-level label for stratified split (majority vote per segment)
    seg_labels = (df.groupby("segment_id")["tremor"]
                    .agg(lambda x: int(x.mean() > 0.5))
                    .reset_index()
                    .rename(columns={"tremor": "seg_label"}))

    seg_ids    = seg_labels["segment_id"].tolist()
    seg_y      = seg_labels["seg_label"].tolist()

    print(f"Total segments: {len(seg_ids)}  "
          f"(tremor={sum(seg_y)}, control={len(seg_y)-sum(seg_y)})")

    # Patient-level stratified split
    indices = np.arange(len(seg_ids))
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.30, stratify=seg_y, random_state=RANDOM_SEED
    )
    temp_y = [seg_y[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, stratify=temp_y, random_state=RANDOM_SEED
    )

    split_map = {}
    for i in train_idx: split_map[seg_ids[i]] = "train"
    for i in val_idx:   split_map[seg_ids[i]] = "val"
    for i in test_idx:  split_map[seg_ids[i]] = "test"

    print(f"Split: {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test segments")

    def build_arrays(split_name):
        ids = [seg_ids[i] for i in (train_idx if split_name=="train"
                                    else val_idx if split_name=="val"
                                    else test_idx)]
        X_parts, y_parts = [], []
        for sid in ids:
            seg_df = df[df["segment_id"] == sid]
            ax = seg_df["x"].to_numpy(dtype=float)
            ay = seg_df["y"].to_numpy(dtype=float)
            az = seg_df["z"].to_numpy(dtype=float)
            labels = seg_df["tremor"].to_numpy(dtype=np.int8)

            try:
                ax, ay, az, y = process_segment(ax, ay, az, labels)
                X_w, y_w = window_xyz(ax, ay, az, y)
                if len(X_w) > 0:
                    X_parts.append(X_w)
                    y_parts.append(y_w)
            except Exception as e:
                print(f"  [ERR] segment {sid}: {e}")

        X = np.concatenate(X_parts, axis=0)
        y = np.concatenate(y_parts, axis=0)
        print(f"  {split_name}: windows={len(X)}  tremor={y.sum()}  control={(y==0).sum()}")
        return X, y

    print("\nProcessing train...")
    X_train, y_train = build_arrays("train")
    print("Processing val...")
    X_val,   y_val   = build_arrays("val")
    print("Processing test...")
    X_test,  y_test  = build_arrays("test")

    for name, arr in [
        ("X_train", X_train), ("y_train", y_train),
        ("X_val",   X_val),   ("y_val",   y_val),
        ("X_test",  X_test),  ("y_test",  y_test),
    ]:
        np.save(os.path.join(out_dir, f"{name}.npy"), arr)

    print(f"\nDone. Saved to: {out_dir}")
    print(f"  X_train {X_train.shape}  X_val {X_val.shape}  X_test {X_test.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True,
                        help="Path to df_all_timesteps.csv")
    parser.add_argument("--out_dir",  default="artifacts")
    args = parser.parse_args()
    main(args.csv_path, args.out_dir)
