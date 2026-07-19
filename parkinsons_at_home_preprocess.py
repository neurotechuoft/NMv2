"""
parkinson_at_home_preprocessor.py
───────────────────────────────────
Preprocessing pipeline for the Parkinson@Home formatted data.
Follows the PDF pipeline spec.

- Columns: time, x, y, z, tremor (per-sample labels already exist)
- 200 Hz → downsample to 100 Hz in Step 0
- Patient-level split on pd{id} from filename (e.g. pd02_LW_OFF_labeled.csv → patient pd02)

Usage
─────
    python parkinson_at_home_preprocessor.py \
        --data_dir  "formatted data" \
        --out_dir   artifacts
"""

import argparse
import os
import glob
from math import gcd

import numpy as np
import pandas as pd
from scipy.signal import firwin, filtfilt, resample_poly
from sklearn.model_selection import train_test_split


FS_SOURCE   = 200
FS_TARGET   = 100
WIN_SAMPLES = int(FS_TARGET * 3.0)   # 300
HOP_SAMPLES = max(1, int(WIN_SAMPLES * 0.10))  # 30
RANDOM_SEED = 42


# ── Step 0: Downsample 200 → 100 Hz ──────────────────────────────────────────

def resample_to_100hz(signal: np.ndarray) -> np.ndarray:
    lp_cutoff = min(FS_SOURCE / 2.0, FS_TARGET / 2.0) - 0.5  # 49.5 Hz
    n = len(signal)
    numtaps = min(64 * 2 + 1, (n - 1) // 3)
    if numtaps % 2 == 0:
        numtaps -= 1
    numtaps = max(numtaps, 3)
    lp = firwin(numtaps, lp_cutoff / (FS_SOURCE / 2.0))
    signal = filtfilt(lp, [1.0], signal.astype(np.float64))
    g    = gcd(FS_SOURCE, FS_TARGET)
    up   = FS_TARGET  // g   # 1
    down = FS_SOURCE  // g   # 2
    return resample_poly(signal, up, down, padtype="line")


def resample_labels(labels: np.ndarray) -> np.ndarray:
    g    = gcd(FS_SOURCE, FS_TARGET)
    up   = FS_TARGET  // g
    down = FS_SOURCE  // g
    return np.repeat(labels, up)[::down].astype(np.int8)


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
    ham = np.hamming(WIN_SAMPLES).astype(np.float32)
    X_out, y_out = [], []
    for i in range(0, len(ax) - WIN_SAMPLES + 1, HOP_SAMPLES):
        seg = np.stack([ax[i:i+WIN_SAMPLES], ay[i:i+WIN_SAMPLES], az[i:i+WIN_SAMPLES]], axis=1)
        X_out.append((seg * ham[:, None]).astype(np.float32))
        lab = y[i:i+WIN_SAMPLES]
        y_out.append(int(np.bincount(lab.astype(int)).argmax()))
    return np.asarray(X_out, dtype=np.float32), np.asarray(y_out, dtype=np.int64)


# ── Per-file pipeline ─────────────────────────────────────────────────────────

def process_file(csv_path: str):
    df = pd.read_csv(csv_path)
    ax = df["x"].to_numpy(dtype=float)
    ay = df["y"].to_numpy(dtype=float)
    az = df["z"].to_numpy(dtype=float)
    labels = df["tremor"].to_numpy(dtype=np.int8)

    ax = resample_to_100hz(ax)
    ay = resample_to_100hz(ay)
    az = resample_to_100hz(az)
    y  = resample_labels(labels)

    n = min(len(ax), len(ay), len(az), len(y))
    ax, ay, az, y = ax[:n], ay[:n], az[:n], y[:n]

    ax = bandpass_fir(drift_remove_ma(ax))
    ay = bandpass_fir(drift_remove_ma(ay))
    az = bandpass_fir(drift_remove_ma(az))

    return ax, ay, az, y


# ── Main ──────────────────────────────────────────────────────────────────────

def main(data_dir: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    print(f"Found {len(files)} files")

    # Extract patient ID from filename (e.g. pd02_LW_OFF_labeled.csv → pd02)
    def patient_id(fp):
        return os.path.basename(fp).split("_")[0]

    # Patient-level label: does this file have any tremor?
    file_labels = []
    for fp in files:
        df = pd.read_csv(fp, usecols=["tremor"])
        file_labels.append(int(df["tremor"].sum() > 0))

    # Group by patient for split (use file-level split, stratified)
    indices = np.arange(len(files))
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.30, stratify=file_labels, random_state=RANDOM_SEED
    )
    val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.50, random_state=RANDOM_SEED
    )

    print(f"Split: {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test files")

    def build_arrays(idx_list, split_name):
        X_parts, y_parts = [], []
        for i in idx_list:
            fp = files[i]
            try:
                ax, ay, az, y = process_file(fp)
                X_w, y_w = window_xyz(ax, ay, az, y)
                if len(X_w) > 0:
                    X_parts.append(X_w)
                    y_parts.append(y_w)
                    print(f"  {os.path.basename(fp)}: {len(X_w)} windows, tremor={y_w.sum()}")
            except Exception as e:
                print(f"  [ERR] {os.path.basename(fp)}: {e}")

        X = np.concatenate(X_parts, axis=0)
        y = np.concatenate(y_parts, axis=0)
        print(f"  {split_name} total: windows={len(X)} tremor={y.sum()} control={(y==0).sum()}")
        return X, y

    print("\nProcessing train...")
    X_train, y_train = build_arrays(train_idx, "train")
    print("Processing val...")
    X_val,   y_val   = build_arrays(val_idx,   "val")
    print("Processing test...")
    X_test,  y_test  = build_arrays(test_idx,  "test")

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
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir",  default="artifacts")
    args = parser.parse_args()
    main(args.data_dir, args.out_dir)
