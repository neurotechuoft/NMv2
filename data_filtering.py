"""
data_filtering.py
-----------------
All raw-data loading, column standardisation, signal preprocessing, and
windowing live here.  Nothing in this file trains a model or touches PyTorch.

Public API used by train_all_datasets.py and main.py:
  - dataset_artifact_dir(name)       -> str path, created on demand
  - format_dataset_for_cnn_rf(...)   -> tidy DataFrame  (x, y, z, tremor, [source])
  - preprocess_and_window(df, ...)   -> (X: float32[N,T,3], y: int64[N])
  - drift_remove_ma(x, fs, seconds)  -> ndarray
  - bandpass_fir(x, fs, lo, hi)      -> ndarray
  - window_xyz(ax, ay, az, y, ...)   -> X  or  (X, y)
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.signal import filtfilt, firwin

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FS_TARGET: int = 100          # Hz — assumed sample rate for all datasets
WIN_SEC:   float = 3.0        # window length in seconds
OVERLAP:   float = 0.5        # fractional overlap between consecutive windows

STANDARD_ACCEL_COLS = ("x", "y", "z")

# Column-name aliases accepted for each axis.
ACCEL_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "x": ("x", "X", "WRIST.x", "wrist.x", "acc_x", "ax"),
    "y": ("y", "Y", "WRIST.y", "wrist.y", "acc_y", "ay"),
    "z": ("z", "Z", "WRIST.z", "wrist.z", "acc_z", "az"),
}

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def dataset_artifact_dir(dataset_name: str, base_dir: str = "artifacts") -> str:
    """Return (and create) the per-dataset artifact directory."""
    path = Path(base_dir) / dataset_name
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def _find_first_existing(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    colset = set(columns)
    for c in candidates:
        if c in colset:
            return c
    return None

# ---------------------------------------------------------------------------
# Column normalisation
# ---------------------------------------------------------------------------

def _standardize_accel_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename aliased axis columns to canonical 'x', 'y', 'z'."""
    rename_map: dict[str, str] = {}
    for dst in STANDARD_ACCEL_COLS:
        src = _find_first_existing(df.columns, ACCEL_COLUMN_ALIASES[dst])
        if src is None:
            raise ValueError(
                f"Could not find an accelerometer column for axis '{dst}'. "
                f"Available columns: {list(df.columns)[:20]}"
            )
        rename_map[src] = dst

    out = df.rename(columns=rename_map)
    out = out.loc[:, ~out.columns.duplicated()]
    return out


def _ensure_tremor_column(df: pd.DataFrame, tremor_value: int | None = None) -> pd.DataFrame:
    """Add a constant 'tremor' column when the file does not already have one."""
    out = df.copy()
    if "tremor" not in out.columns:
        if tremor_value is None:
            raise ValueError(
                "No 'tremor' column found in this file and no fallback tremor_value "
                "was provided.  Either add a 'tremor' column or pass tremor_value."
            )
        out["tremor"] = int(tremor_value)
    out["tremor"] = out["tremor"].astype(int)
    return out


def _standardize_single_file(
    csv_path: str | Path,
    tremor_value: int | None = None,
    source_name: str | None = None,
) -> pd.DataFrame:
    """
    Load one raw CSV, normalise column names, and guarantee x/y/z/tremor columns.

    Raises ValueError if the file looks pre-windowed (wide x_0, x_1 … format).
    """
    df = pd.read_csv(csv_path)

    if any(c.startswith("x_") for c in df.columns):
        raise ValueError(
            f"{csv_path} looks like pre-windowed wide data (x_0 … x_N columns). "
            "Pass the continuous/raw CSV with one row per sample instead."
        )

    df = _standardize_accel_columns(df)
    df = _ensure_tremor_column(df, tremor_value=tremor_value)
    df = df[["x", "y", "z", "tremor"]].copy()

    if source_name is not None:
        df["source"] = source_name

    return df

# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------


def load_dataset_for_cnn_rf(
    dataset: str,
    root_dir: str = ".",
    infer_parkinsons_tremor_from_filename: bool = False,
) -> pd.DataFrame:
    """
    Load and standardise one of the supported datasets.

    Returns a DataFrame with columns: x, y, z, tremor, [source].

    Supported datasets
    ------------------
    'jiehu-rima'
        Jiehu Set (Rima)/df_all_timesteps.csv
        Must contain a 'tremor' column.

    'kaggle-arushna'
        kaggle-arushna/formatted_data_continuous.csv
        Must contain a 'tremor' column.

    'kaggle-yunji'
        Kaggle Set (Yunji)/tremorPatientID.csv  +
        Kaggle Set (Yunji)/patient_tremor_data/*.csv
        Labels are taken from tremorPatientID.csv.

    'parkinsons-home'
        parkinsons@home/*.csv
        If infer_parkinsons_tremor_from_filename=True the loader guesses
        tremor=1 for "_OFF_" files and tremor=0 for "_ON_" files.
        Files without a 'tremor' column and without a filename match are
        skipped with a warning rather than crashing.
    """
    ds = dataset.strip().lower()
    root = Path(root_dir)

    # ------------------------------------------------------------------
    if ds == "jiehu-rima":
        path = root / "Jiehu Set (Rima)" / "df_all_timesteps.csv"
        return _standardize_single_file(path, source_name="jiehu-rima")

    # ------------------------------------------------------------------
    if ds == "kaggle-arushna":
        path = root / "kaggle-arushna" / "formatted_data_continuous.csv"
        return _standardize_single_file(path, source_name="kaggle-arushna")

    # ------------------------------------------------------------------
    if ds == "kaggle-yunji":
        map_path = root / "Kaggle Set (Yunji)" / "tremorPatientID.csv"
        raw_dir  = root / "Kaggle Set (Yunji)" / "patient_tremor_data"

        mapping = pd.read_csv(map_path)
        required = {"data_file_name", "tremor"}
        if not required.issubset(mapping.columns):
            raise ValueError(
                f"{map_path} must contain columns {required}; "
                f"found: {set(mapping.columns)}"
            )

        parts: list[pd.DataFrame] = []
        for row in mapping.itertuples(index=False):
            csv_path = raw_dir / str(row.data_file_name)
            if not csv_path.exists():
                print(f"  [kaggle-yunji] Missing file, skipping: {csv_path.name!r}")
                continue
            part = _standardize_single_file(
                csv_path,
                tremor_value=int(row.tremor),
                source_name=csv_path.name,
            )
            parts.append(part)

        if not parts:
            raise ValueError(
                "No Yunji files could be loaded.  Check file paths and tremorPatientID.csv."
            )
        return pd.concat(parts, ignore_index=True)

    # ------------------------------------------------------------------
    if ds == "parkinsons-home":
        data_dir  = root / "parkinsons@home"
        csv_files = sorted(data_dir.glob("*.csv"))
        if not csv_files:
            raise ValueError(f"No CSV files found in {data_dir}")

        parts   : list[pd.DataFrame] = []
        skipped : list[str]          = []

        for csv_path in csv_files:
            tremor_value: int | None = None
            if infer_parkinsons_tremor_from_filename:
                name_upper = csv_path.name.upper()
                if "_OFF_" in name_upper:
                    tremor_value = 1
                elif "_ON_" in name_upper:
                    tremor_value = 0

            try:
                part = _standardize_single_file(
                    csv_path,
                    tremor_value=tremor_value,
                    source_name=csv_path.name,
                )
                parts.append(part)
            except (ValueError, KeyError) as exc:
                print(f"  [parkinsons-home] Skipping {csv_path.name!r}: {exc}")
                skipped.append(csv_path.name)

        if skipped:
            print(f"  [parkinsons-home] Skipped {len(skipped)} file(s) total.")
        if not parts:
            raise ValueError(
                f"No valid CSV files loaded from {data_dir}.  "
                "Ensure files have x/y/z and tremor columns, or enable "
                "infer_parkinsons_tremor_from_filename=True for OFF/ON filenames."
            )
        return pd.concat(parts, ignore_index=True)

    # ------------------------------------------------------------------
    raise ValueError(
        "Unsupported dataset.  Choose one of: "
        "'jiehu-rima', 'kaggle-arushna', 'kaggle-yunji', 'parkinsons-home'."
    )


def format_dataset_for_cnn_rf(
    dataset: str,
    root_dir: str = ".",
    infer_parkinsons_tremor_from_filename: bool = False,
    save_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Load → standardise → clean → (optionally save) one dataset.

    Guaranteed output columns: x, y, z, tremor  (and 'source' when available).
    All NaN/Inf rows and non-numeric values are dropped.
    """
    df = load_dataset_for_cnn_rf(
        dataset,
        root_dir=root_dir,
        infer_parkinsons_tremor_from_filename=infer_parkinsons_tremor_from_filename,
    ).copy()

    keep = [c for c in ["x", "y", "z", "tremor", "source"] if c in df.columns]
    df = df[keep].copy()

    df[["x", "y", "z"]] = df[["x", "y", "z"]].apply(pd.to_numeric, errors="coerce")
    df["tremor"]         = pd.to_numeric(df["tremor"], errors="coerce")

    df = (
        df.replace([np.inf, -np.inf], np.nan)
          .dropna(subset=["x", "y", "z", "tremor"])
          .copy()
    )
    df["tremor"] = df["tremor"].astype(int)
    df = df.reset_index(drop=True)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)

    return df

# ---------------------------------------------------------------------------
# Signal preprocessing
# ---------------------------------------------------------------------------

def drift_remove_ma(
    x: np.ndarray,
    fs: int = FS_TARGET,
    seconds: float = 5.0,
) -> np.ndarray:
    """
    Remove slow drift (gravity component + DC bias) via a moving-average high-pass.
    """
    L = max(int(fs * seconds), 3)
    # MAGIC LINE 1: Dynamically shrink the filter for short CSVs
    L = min(L, max(3, (len(x) - 1) // 3))   
    
    ma    = np.ones(L, dtype=float) / L
    trend = filtfilt(ma, [1.0], x)
    return x - trend


def bandpass_fir(
    x: np.ndarray,
    fs: int = FS_TARGET,
    lo: float = 1.0,
    hi: float = 30.0,
    numtaps: int = 201,
) -> np.ndarray:
    """
    Zero-phase FIR bandpass filter.
    """
    # MAGIC LINE 2: Dynamically shrink the tap size for short CSVs
    numtaps = min(numtaps, max(3, (len(x) - 1) // 3))
    if numtaps % 2 == 0:
        numtaps -= 1
    numtaps = max(numtaps, 3)
    
    bp = firwin(numtaps, [lo, hi], pass_zero=False, fs=fs)
    return filtfilt(bp, [1.0], x)

# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

def window_xyz(
    ax: np.ndarray,
    ay: np.ndarray,
    az: np.ndarray,
    y: np.ndarray | None = None,
    fs: int = FS_TARGET,
    win_sec: float = WIN_SEC,
    overlap: float = OVERLAP,
    meta: np.ndarray | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Slice three synchronised accelerometer channels into overlapping windows.

    Each window is multiplied by a Hamming taper before stacking.

    Parameters
    ----------
    ax, ay, az : 1-D arrays of the same length
    y          : optional per-sample label array (same length as ax)
    fs         : sample rate in Hz
    win_sec    : window length in seconds
    overlap    : fractional overlap  (0 = no overlap, 0.5 = 50 % overlap)

    Returns
    -------
    X          : float32 array of shape (N_windows, win_samples, 3)
    y_windows  : int64 array of shape (N_windows,) — only when y is not None.
                 Label per window = majority vote of per-sample labels.
    """
    win = int(fs * win_sec)
    hop = max(int(win * (1.0 - overlap)), 1)
    ham = np.hamming(win).astype(np.float32)

    X:     list[np.ndarray] = []
    y_out: list[int]        = [] if y is not None else None  # type: ignore[assignment]
    meta_out: list[str]     = [] if (meta is not None and y is not None) else None  # type: ignore[assignment]

    for i in range(0, len(ax) - win + 1, hop):
        seg = np.stack(
            [ax[i : i + win], ay[i : i + win], az[i : i + win]],
            axis=1,
        ).astype(np.float32)
        seg *= ham[:, None]
        X.append(seg)

        if y is not None:
            lab = y[i : i + win]
            y_out.append(int(np.bincount(lab).argmax()))  # type: ignore[index]

        if meta is not None:
            window_meta = meta[i : i + win]
            if len(window_meta) == 0:
                meta_out.append("unknown")
            else:
                vals, cnts = np.unique(window_meta, return_counts=True)
                meta_out.append(str(vals[np.argmax(cnts)]))

    X_arr = np.asarray(X, dtype=np.float32)

    if y is None:
        return X_arr
    if meta is None:
        return X_arr, np.asarray(y_out, dtype=np.int64)
    return X_arr, np.asarray(y_out, dtype=np.int64), np.asarray(meta_out, dtype=object)


def preprocess_and_window(
    df: pd.DataFrame,
    accel_cols: tuple[str, str, str] = ("x", "y", "z"),
    label_col: str = "tremor",
    fs: int = FS_TARGET,
    win_sec: float = WIN_SEC,
    overlap: float = OVERLAP,
    return_meta: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Full preprocessing pipeline for a DataFrame split.

    Steps (applied per axis):
      1. Moving-average drift removal  (removes gravity / DC)
      2. FIR bandpass  1–30 Hz        (keeps tremor band, kills noise)
      3. Hamming-windowed segmentation with majority-vote labels

    Returns
    -------
    X : float32[N, win_samples, 3]
    y : int64[N]
    """
    ax = df[accel_cols[0]].to_numpy(dtype=float)
    ay = df[accel_cols[1]].to_numpy(dtype=float)
    az = df[accel_cols[2]].to_numpy(dtype=float)
    y  = df[label_col].to_numpy(dtype=int)

    ax = bandpass_fir(drift_remove_ma(ax, fs=fs), fs=fs)
    ay = bandpass_fir(drift_remove_ma(ay, fs=fs), fs=fs)
    az = bandpass_fir(drift_remove_ma(az, fs=fs), fs=fs)

    meta_arr = None
    if return_meta:
        if "source" in df.columns:
            meta_arr = df["source"].to_numpy(dtype=object)
        else:
            meta_arr = np.full(len(ax), "unknown", dtype=object)

    if return_meta:
        return window_xyz(ax, ay, az, y=y, fs=fs, win_sec=win_sec, overlap=overlap, meta=meta_arr)
    return window_xyz(ax, ay, az, y=y, fs=fs, win_sec=win_sec, overlap=overlap)