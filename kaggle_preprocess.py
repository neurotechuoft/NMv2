"""
kaggle_tremor_preprocessor.py
──────────────────────────────
Preprocessing pipeline for the Kaggle (Yunji) wrist accelerometer dataset.
Follows the pipeline spec in data_processing__acc__to_freq_.pdf:

  Step 0 – Resample each file to 100 Hz:
             - Apply ideal LP anti-aliasing filter at cutoff = min(fs_src/2, 50) Hz
             - Resample with resample_poly (polyphase, avoids FFT periodicity artefacts)
  Step 1 – Drift removal (5 s moving-average zero-phase filter).
           Bandpass FIR 1–30 Hz.
  Step 2 – Sliding 3 s Hamming-windowed segments (90% overlap).
           AR(6) Burg PSD per segment → peak detection in 3–8 Hz.
           Assign 1 Hz baseline if no significant peak found
           (significant = band peak > global_max_power / 10).
  Step 3 – Map per-window peak freq back to a continuous 100 Hz profile
           (nearest-neighbour, full signal length). Smooth with 3-point MA.
           Multiply the three axis profiles together.
  Step 4 – Threshold combined profile at 3.5 Hz → binary pulse.
           Duration filter: only pulses ≥ 3 s count as tremor.
  Output – Patient-level train/val/test split (no leakage), then windowed
            arrays X (N, 300, 3) and y (N,) ready for the CNN + RF notebooks.

Usage
─────
    python kaggle_tremor_preprocessor.py \
        --data_dir  "Kaggle Set (Yunji)/patient_tremor_data" \
        --id_csv    "Kaggle Set (Yunji)/tremorPatientID.csv" \
        --out_dir   artifacts
"""

import argparse
import os
from math import gcd

import numpy as np
import pandas as pd
from scipy.signal import firwin, filtfilt, resample_poly
from sklearn.model_selection import train_test_split


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
FS_TARGET   = 100          # Hz – everything downstream assumes this
WIN_SEC     = 3.0          # seconds per window
OVERLAP     = 0.90         # 90 % overlap → hop = 30 samples
WIN_SAMPLES = int(FS_TARGET * WIN_SEC)   # 300
HOP_SAMPLES = max(1, int(WIN_SAMPLES * (1.0 - OVERLAP)))  # 30

TREMOR_LO   = 3.0          # Hz – lower bound for PD tremor peak search
TREMOR_HI   = 8.0          # Hz
FREQ_THR    = 3.5          # Hz – final threshold for binary label
DUR_MIN_S   = 3.0          # seconds – minimum pulse duration to count as tremor
AR_ORDER    = 6            # AR model order (Burg method)
RANDOM_SEED = 42


# ──────────────────────────────────────────────────────────────────────────────
# Step 0 – Resampling with anti-aliasing LP applied before resample_poly
# ──────────────────────────────────────────────────────────────────────────────
def estimate_fs(time_col: np.ndarray) -> float:
    """Estimate sampling rate from the TIME column (seconds)."""
    diffs = np.diff(time_col)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return FS_TARGET
    return 1.0 / np.median(diffs)


def resample_to_100hz(signal: np.ndarray, fs_src: float) -> np.ndarray:
    """
    Resample a 1-D signal from fs_src → FS_TARGET (100 Hz).

    Per the PDF spec:
      1. Apply ideal LP anti-aliasing filter at cutoff = min(fs_src/2, fs_target/2)
         BEFORE resampling, to prevent aliasing of out-of-band noise into the
         tremor frequency range.
      2. Resample with resample_poly (polyphase), not resample (FFT-based),
         to avoid periodicity artefacts at signal edges.
    """
    if abs(fs_src - FS_TARGET) < 0.5:
        return signal.astype(np.float64)

    # LP cutoff: min of Nyquist of source and target
    lp_cutoff = min(fs_src / 2.0, FS_TARGET / 2.0)
    # Small guard band so the FIR doesn't sit right at Nyquist
    lp_cutoff = max(lp_cutoff - 0.5, 1.0)

    # Apply anti-aliasing LP filter on the original-rate signal before resampling.
    # scipy filtfilt padlen can be up to 3*numtaps for edge cases near Nyquist,
    # so use the conservative cap: numtaps < n/3, i.e. numtaps = (n-1)//3.
    n_src = len(signal)
    numtaps = min(64 * 2 + 1, (n_src - 1) // 3)   # 129 taps, or less for short signals
    if numtaps % 2 == 0:
        numtaps -= 1
    numtaps = max(numtaps, 3)
    lp = firwin(numtaps, lp_cutoff / (fs_src / 2.0))
    signal_filtered = filtfilt(lp, [1.0], signal.astype(np.float64))


    # Integer up/down ratio via GCD for resample_poly
    fs_src_int = int(round(fs_src))
    g    = gcd(fs_src_int, FS_TARGET)
    up   = FS_TARGET   // g
    down = fs_src_int  // g

    resampled = resample_poly(signal_filtered, up, down, padtype="line")
    return resampled.astype(np.float64)


# ──────────────────────────────────────────────────────────────────────────────
# Step 1 – Drift removal & bandpass (at 100 Hz)
# ──────────────────────────────────────────────────────────────────────────────
def drift_remove_ma(x: np.ndarray, fs: int = FS_TARGET, seconds: float = 5.0) -> np.ndarray:
    """Zero-phase 5-second moving average subtraction to remove DC drift."""
    L = max(int(fs * seconds), 3)
    # filtfilt padlen can reach 3*L for near-Nyquist filters; use conservative cap
    max_L = max(3, (len(x) - 1) // 3)
    L = min(L, max_L)
    ma = np.ones(L, dtype=float) / L
    trend = filtfilt(ma, [1.0], x)
    return x - trend


def bandpass_fir(x: np.ndarray, fs: int = FS_TARGET,
                 lo: float = 1.0, hi: float = 30.0, numtaps: int = 201) -> np.ndarray:
    """Zero-phase FIR bandpass 1–30 Hz."""
    # filtfilt padlen can reach 3*numtaps; use conservative cap
    max_taps = max(3, (len(x) - 1) // 3)
    numtaps = min(numtaps, max_taps)
    if numtaps % 2 == 0:
        numtaps -= 1  # firwin requires odd numtaps for bandpass
    numtaps = max(numtaps, 3)
    bp = firwin(numtaps, [lo, hi], pass_zero=False, fs=fs)
    return filtfilt(bp, [1.0], x)


# ──────────────────────────────────────────────────────────────────────────────
# Step 2 – AR(6) Burg PSD + peak detection in 3–8 Hz
# ──────────────────────────────────────────────────────────────────────────────
def burg_ar_psd(segment: np.ndarray, order: int = AR_ORDER,
                fs: int = FS_TARGET, nfft: int = 512):
    """
    Estimate the PSD of `segment` using the Burg AR method (Berg method).
    Returns (freqs, power) arrays.
    """
    x  = (segment - segment.mean()).astype(np.float64)
    N  = len(x)

    ef = x.copy()
    eb = x.copy()
    a  = np.zeros(order + 1, dtype=np.float64)
    a[0] = 1.0
    P  = float(np.dot(x, x)) / N

    for m in range(1, order + 1):
        ef_m = ef[m:]      # forward prediction errors, length N-m
        eb_m = eb[:-m]     # backward prediction errors, length N-m (aligned)

        num = -2.0 * np.dot(eb_m, ef_m)
        den = np.dot(ef_m, ef_m) + np.dot(eb_m, eb_m)

        if abs(den) < 1e-10:
            break

        km = num / den

        # Update AR coefficients via Levinson step
        a_old = a[:m + 1].copy()
        a[:m + 1] = a_old + km * a_old[::-1]

        # Update error vectors
        ef_new = ef_m + km * eb_m
        eb_new = eb_m + km * ef_m
        ef, eb = ef_new, eb_new

        P *= (1.0 - km ** 2)

    # PSD = noise_power / |H(e^jw)|^2
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    h     = np.fft.rfft(a, n=nfft)
    power = max(P, 0.0) / (np.abs(h) ** 2 + 1e-12)
    return freqs, power


def peak_freq_in_band(freqs: np.ndarray, power: np.ndarray,
                      lo: float = TREMOR_LO, hi: float = TREMOR_HI) -> float:
    """
    Return the dominant peak frequency in [lo, hi] Hz.

    Per the PDF spec:
      - Compute the max power in the 3–8 Hz band.
      - Threshold T = global_max_power / 10.
      - If no peak in band, or if band_max_power < T, assign baseline 1 Hz.
      - Otherwise return the frequency of the band peak.
    """
    mask = (freqs >= lo) & (freqs <= hi)
    if not mask.any():
        return 1.0

    band_power = power[mask]
    band_freqs = freqs[mask]
    band_max   = band_power.max()

    # Threshold is 1/10th of the GLOBAL max power (not band max)
    global_thr = power.max() / 10.0

    if band_max < global_thr:
        return 1.0  # no significant tremor peak

    peak_idx = np.argmax(band_power)
    return float(band_freqs[peak_idx])


# ──────────────────────────────────────────────────────────────────────────────
# Step 3 – Time-frequency profile, smoothing, axis multiplication
# ──────────────────────────────────────────────────────────────────────────────
def compute_tf_profile(axis_signal: np.ndarray,
                       fs: int      = FS_TARGET,
                       win_s: float = WIN_SEC,
                       overlap: float = OVERLAP) -> np.ndarray:
    """
    Slide a Hamming-weighted window over the signal, compute the AR PSD peak
    frequency per window, smooth with a 3-point MA, then upsample back to the
    full signal length (100 Hz) by nearest-neighbour interpolation.

    Returns a profile array of length == len(axis_signal).
    """
    win = int(fs * win_s)            # 300 samples
    hop = max(1, int(win * (1.0 - overlap)))  # 30 samples
    N   = len(axis_signal)
    ham = np.hamming(win)

    centres    = []
    peak_freqs = []

    for start in range(0, N - win + 1, hop):
        seg = axis_signal[start:start + win] * ham
        fr, pw = burg_ar_psd(seg, order=AR_ORDER, fs=fs)
        pf = peak_freq_in_band(fr, pw)
        centres.append(start + win // 2)
        peak_freqs.append(pf)

    if len(centres) == 0:
        return np.ones(N, dtype=np.float64)

    centres    = np.array(centres,    dtype=np.float64)
    peak_freqs = np.array(peak_freqs, dtype=np.float64)

    # 3-point moving average smoothing on the per-window values
    kernel   = np.ones(3) / 3.0
    smoothed = np.convolve(peak_freqs, kernel, mode="same")

    # Upsample back to full 100 Hz sample rate via nearest-neighbour interpolation.
    # np.interp clamps to boundary values outside the range, so edge samples
    # get the first/last window's value — correct behaviour.
    sample_indices = np.arange(N, dtype=np.float64)
    profile = np.interp(sample_indices, centres, smoothed)

    return profile


# ──────────────────────────────────────────────────────────────────────────────
# Step 4 – Threshold + duration filter → binary label per sample
# ──────────────────────────────────────────────────────────────────────────────
def label_from_combined_profile(combined: np.ndarray,
                                fs: int         = FS_TARGET,
                                freq_thr: float = FREQ_THR,
                                dur_min_s: float = DUR_MIN_S) -> np.ndarray:
    """
    Apply 3.5 Hz threshold to the combined axis profile → binary pulse waveform.
    Then apply duration filter: only pulses lasting ≥ 3 seconds are kept as tremor.
    """
    raw   = (combined >= freq_thr).astype(np.int8)
    label = np.zeros(len(raw), dtype=np.int8)
    min_samples = int(dur_min_s * fs)

    i = 0
    while i < len(raw):
        if raw[i] == 1:
            j = i
            while j < len(raw) and raw[j] == 1:
                j += 1
            if (j - i) >= min_samples:
                label[i:j] = 1
            i = j
        else:
            i += 1

    return label


# ──────────────────────────────────────────────────────────────────────────────
# Full per-file pipeline
# ──────────────────────────────────────────────────────────────────────────────
def process_file(csv_path: str):
    """
    Load one patient CSV, run the full PDF pipeline.
    Returns (ax, ay, az, label) – all at 100 Hz, same length.
    """
    df = pd.read_csv(csv_path)

    time = df["TIME"].to_numpy(dtype=float)
    ax   = df["WRIST.x"].to_numpy(dtype=float)
    ay   = df["WRIST.y"].to_numpy(dtype=float)
    az   = df["WRIST.z"].to_numpy(dtype=float)

    # Step 0 – anti-alias LP then resample to 100 Hz
    fs_src = estimate_fs(time)
    ax = resample_to_100hz(ax, fs_src)
    ay = resample_to_100hz(ay, fs_src)
    az = resample_to_100hz(az, fs_src)

    # Step 1 – drift removal + bandpass
    ax = bandpass_fir(drift_remove_ma(ax))
    ay = bandpass_fir(drift_remove_ma(ay))
    az = bandpass_fir(drift_remove_ma(az))

    # Steps 2+3 – per-axis TF profile
    prof_x = compute_tf_profile(ax)
    prof_y = compute_tf_profile(ay)
    prof_z = compute_tf_profile(az)

    # Multiply axes (amplifies signals consistent across all three spatial dimensions)
    combined = prof_x * prof_y * prof_z

    # Step 4 – threshold + duration filter → binary label
    label = label_from_combined_profile(combined)

    return ax, ay, az, label


# ──────────────────────────────────────────────────────────────────────────────
# Windowing (3 s, 90% overlap, Hamming – same as original at 100 Hz)
# ──────────────────────────────────────────────────────────────────────────────
def window_xyz(ax: np.ndarray, ay: np.ndarray, az: np.ndarray,
               y: np.ndarray):
    win = WIN_SAMPLES   # 300
    hop = HOP_SAMPLES   # 30
    ham = np.hamming(win).astype(np.float32)

    X_out, y_out = [], []
    for i in range(0, len(ax) - win + 1, hop):
        seg = np.stack([ax[i:i+win], ay[i:i+win], az[i:i+win]], axis=1)
        seg = (seg * ham[:, None]).astype(np.float32)
        X_out.append(seg)
        lab = y[i:i+win]
        y_out.append(int(np.bincount(lab.astype(int)).argmax()))

    return np.array(X_out, dtype=np.float32), np.array(y_out, dtype=np.int64)


# ──────────────────────────────────────────────────────────────────────────────
# Patient-level split (stratified, no leakage)
# ──────────────────────────────────────────────────────────────────────────────
def patient_level_split(file_list, labels,
                        val_frac: float = 0.15, test_frac: float = 0.15):
    """
    Stratified split at the patient (file) level before windowing.
    Prevents leakage from overlapping windows crossing split boundaries.
    """
    indices = np.arange(len(file_list))
    train_idx, temp_idx = train_test_split(
        indices, test_size=val_frac + test_frac,
        stratify=labels, random_state=RANDOM_SEED
    )
    temp_labels    = [labels[i] for i in temp_idx]
    relative_test  = test_frac / (val_frac + test_frac)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=relative_test,
        stratify=temp_labels, random_state=RANDOM_SEED
    )
    to_paths = lambda idx: [file_list[i] for i in idx]
    return to_paths(train_idx), to_paths(val_idx), to_paths(test_idx)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main(data_dir: str, id_csv: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    id_df = pd.read_csv(id_csv)
    file_paths     = [os.path.join(data_dir, fn) for fn in id_df["data_file_name"]]
    patient_labels = id_df["tremor"].tolist()

    print(f"Total patients: {len(file_paths)}  "
          f"(tremor={sum(patient_labels)}, control={len(patient_labels)-sum(patient_labels)})")

    train_files, val_files, test_files = patient_level_split(file_paths, patient_labels)
    print(f"Split: {len(train_files)} train / {len(val_files)} val / {len(test_files)} test patients")

    def build_arrays(files, split_name):
        X_parts, y_parts, rows = [], [], []
        for fp in files:
            if not os.path.exists(fp):
                print(f"  [SKIP] {fp} not found")
                continue
            try:
                ax, ay, az, label = process_file(fp)
            except Exception as e:
                print(f"  [ERR]  {os.path.basename(fp)}: {e}")
                continue

            X_w, y_w = window_xyz(ax, ay, az, label)
            X_parts.append(X_w)
            y_parts.append(y_w)
            rows.append(pd.DataFrame({"x": ax, "y": ay, "z": az, "tremor": label}))

        X      = np.concatenate(X_parts, axis=0)
        y      = np.concatenate(y_parts, axis=0)
        df_out = pd.concat(rows, ignore_index=True)
        print(f"  {split_name}: windows={len(X)}  tremor={y.sum()}  control={(y==0).sum()}")
        return X, y, df_out

    print("\nProcessing train…")
    X_train, y_train, df_train = build_arrays(train_files, "train")
    print("Processing val…")
    X_val,   y_val,   df_val   = build_arrays(val_files,   "val")
    print("Processing test…")
    X_test,  y_test,  df_test  = build_arrays(test_files,  "test")

    for name, arr in [
        ("X_train", X_train), ("y_train", y_train),
        ("X_val",   X_val),   ("y_val",   y_val),
        ("X_test",  X_test),  ("y_test",  y_test),
    ]:
        np.save(os.path.join(out_dir, f"{name}.npy"), arr)

    df_train["split"] = "train"
    df_val["split"]   = "val"
    df_test["split"]  = "test"
    df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)
    df_all.to_csv(os.path.join(out_dir, "formatted_data_continuous.csv"), index=False)

    print(f"\nDone. Saved to: {out_dir}")
    print(f"  X_train {X_train.shape}  X_val {X_val.shape}  X_test {X_test.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--id_csv",   required=True)
    parser.add_argument("--out_dir",  default="artifacts")
    args = parser.parse_args()
    main(args.data_dir, args.id_csv, args.out_dir)