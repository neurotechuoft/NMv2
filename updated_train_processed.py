import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    matthews_corrcoef, precision_recall_curve, auc,
    roc_auc_score, confusion_matrix, 
)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import joblib
from data_filtering import (
    format_dataset_for_cnn_rf,
    preprocess_and_window,
    FS_TARGET, WIN_SEC, OVERLAP,
)

# -------------------------------------------------------------------------
# FOCAL LOSS (for kaggle-yunji class imbalance)
# -------------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce  = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        p_t  = inputs * targets + (1 - inputs) * (1 - targets)
        loss = bce * ((1 - p_t) ** self.gamma)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * loss).mean()


# -------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------
EPOCHS        = 50
BATCH_SIZE    = 256
LEARNING_RATE = 1e-3

DATASET_CONFIGS = {
    "jiehu-rima": {
        "dir":                        "Jiehu Set (Rima)",
        "accel_cols":                 ["x", "y", "z"],
        "label_col":                  "tremor",
        "cnn_arch":                   "shallow",
        "rf_estimators":              50,
        "train_undersample_target":   None,
        # Folder where Jiehu_preprocess.py saved its .npy files
        "npy_dir":                    os.path.join("artifacts", "jiehu-rima"),
    },
    "kaggle-arushna": {
        "dir":                        "kaggle-arushna",
        "accel_cols":                 ["x", "y", "z"],
        "label_col":                  "tremor",
        "cnn_arch":                   "standard",
        "rf_estimators":              100,
        "train_undersample_target":   None,
        # No preprocessor for arushna — loaded directly from CSV
        "npy_dir":                    None,
    },
    "kaggle-yunji": {
        "dir":                        os.path.join("Kaggle Set (Yunji)", "patient_tremor_data"),
        "accel_cols":                 ["x", "y", "z"],
        "label_col":                  "tremor",
        "cnn_arch":                   "standard",
        "rf_estimators":              100,
        "train_undersample_target":   None,
        # Folder where kaggle_preprocess.py saved its .npy files
        "npy_dir":                    os.path.join("artifacts", "kaggle-yunji"),
    },
    "parkinsons-home": {
        "dir":                        "parkinsons@home",
        "accel_cols":                 ["x", "y", "z"],
        "label_col":                  "tremor",
        "cnn_arch":                   "deep_wide",
        "rf_estimators":              200,
        "train_undersample_target":   0.5,
        # Folder where parkinsons_at_home_preprocess.py saved its .npy files
        "npy_dir":                    os.path.join("artifacts", "parkinsons-home"),
    },
}


# -------------------------------------------------------------------------
# CNN ARCHITECTURES  — all use AdaptiveAvgPool1d so input length is flexible
# -------------------------------------------------------------------------
class StandardCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1  = nn.Conv1d(3,  32, kernel_size=5, stride=2, padding=2)
        self.bn1    = nn.BatchNorm1d(32)
        self.conv2  = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2    = nn.BatchNorm1d(64)
        self.pool   = nn.MaxPool1d(2)
        self.gpool  = nn.AdaptiveAvgPool1d(1)
        self.fc_emb = nn.Linear(64, 64)
        self.fc_out = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x   = torch.relu(self.bn1(self.conv1(x)))
        x   = self.pool(x)
        x   = torch.relu(self.bn2(self.conv2(x)))
        x   = self.pool(x)
        x   = self.gpool(x).view(x.size(0), -1)
        emb = torch.relu(self.fc_emb(x))
        out = self.sigmoid(self.fc_out(emb))
        return out, emb


class DeepWideCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_wide   = nn.Conv1d(3, 32, kernel_size=31, stride=2, padding=15)
        self.bn1_wide     = nn.BatchNorm1d(32)
        self.conv1_narrow = nn.Conv1d(3, 32, kernel_size=5,  stride=2, padding=2)
        self.bn1_narrow   = nn.BatchNorm1d(32)
        self.pool   = nn.MaxPool1d(2)
        self.conv2  = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2    = nn.BatchNorm1d(128)
        self.gpool  = nn.AdaptiveAvgPool1d(1)
        self.fc_emb = nn.Linear(128, 128)
        self.fc_out = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xw  = torch.relu(self.bn1_wide(self.conv1_wide(x)))
        xw  = self.pool(xw)
        xn  = torch.relu(self.bn1_narrow(self.conv1_narrow(x)))
        xn  = self.pool(xn)
        x   = torch.relu(self.bn2(self.conv2(torch.cat([xw, xn], dim=1))))
        x   = self.pool(x)
        x   = self.gpool(x).view(x.size(0), -1)
        emb = torch.relu(self.fc_emb(x))
        out = self.sigmoid(self.fc_out(emb))
        return out, emb


class ShallowCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1  = nn.Conv1d(3,  16, kernel_size=9, stride=2, padding=4)
        self.bn1    = nn.BatchNorm1d(16)
        self.conv2  = nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2)
        self.bn2    = nn.BatchNorm1d(32)
        self.pool   = nn.MaxPool1d(2)
        self.gpool  = nn.AdaptiveAvgPool1d(1)
        self.fc_emb = nn.Linear(32, 32)
        self.fc_out = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x   = torch.relu(self.bn1(self.conv1(x)))
        x   = self.pool(x)
        x   = torch.relu(self.bn2(self.conv2(x)))
        x   = self.pool(x)
        x   = self.gpool(x).view(x.size(0), -1)
        emb = torch.relu(self.fc_emb(x))
        out = self.sigmoid(self.fc_out(emb))
        return out, emb


_ARCH_MAP = {
    "shallow":   ShallowCNN,
    "standard":  StandardCNN,
    "deep_wide": DeepWideCNN,
}


# -------------------------------------------------------------------------
# HANDCRAFTED FEATURES
# -------------------------------------------------------------------------
def handcrafted_features(X: np.ndarray, fs: float = float(FS_TARGET)) -> np.ndarray:
    X_twc = X.transpose(0, 2, 1)
    n     = X_twc.shape[1]
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    feats = []
    for axis in range(3):
        sig  = X_twc[:, :, axis]
        rms  = np.sqrt(np.mean(sig ** 2, axis=1, keepdims=True))
        var  = np.var(sig, axis=1, keepdims=True)
        mags = np.abs(np.fft.rfft(sig, axis=1))
        mags[:, 0] = 0.0
        peak_freq   = freqs[np.argmax(mags, axis=1)][:, None]
        spec_energy = np.sum(mags ** 2, axis=1, keepdims=True)
        feats.extend([rms, var, peak_freq, spec_energy])

    return np.concatenate(feats, axis=1).astype(np.float32)


# -------------------------------------------------------------------------
# DATA LOADING
# Two paths:
#   1. npy_dir is set → load X_train/val/test.npy from the preprocessor output
#   2. npy_dir is None → load raw CSV via data_filtering (kaggle-arushna only)
#
# Preprocessors save shape (N, T, 3) — time-last.
# CNNs expect (N, 3, T)  — channels-first.
# We transpose on load.
# -------------------------------------------------------------------------
def load_npy_splits(npy_dir: str, dataset_name: str):
    """
    Load the 6 .npy files saved by a preprocessor script and return
    (X_train, y_train, X_val, y_val, X_test, y_test) all channels-first.
    """
    def load(name):
        path = os.path.join(npy_dir, f"{name}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[{dataset_name}] Missing {path}\n"
                f"  → Run the preprocessor first:\n"
                f"    python kaggle_preprocess.py / Jiehu_preprocess.py / "
                f"parkinsons_at_home_preprocess.py"
            )
        return np.load(path)

    X_train = load("X_train").transpose(0, 2, 1).astype(np.float32)  # (N,T,3)→(N,3,T)
    y_train = load("y_train").astype(np.int64)
    X_val   = load("X_val").transpose(0, 2, 1).astype(np.float32)
    y_val   = load("y_val").astype(np.int64)
    X_test  = load("X_test").transpose(0, 2, 1).astype(np.float32)
    y_test  = load("y_test").astype(np.int64)

    print(f"  [npy] X_train {X_train.shape}  X_val {X_val.shape}  X_test {X_test.shape}")
    print(f"  [npy] y_train tremor={y_train.sum()}  y_val tremor={y_val.sum()}  y_test tremor={y_test.sum()}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def load_csv_splits(dataset_name: str, config: dict):
    """
    Load via data_filtering (used for kaggle-arushna which has no preprocessor).
    Returns (X_train, y_train, X_val, y_val, X_test, y_test).
    """
    try:
        infer = (dataset_name == "parkinsons-home")
        df = format_dataset_for_cnn_rf(
            dataset_name, root_dir=".",
            infer_parkinsons_tremor_from_filename=infer,
        )
    except Exception as exc:
        raise RuntimeError(f"Error loading {dataset_name}: {exc}")

    all_X, all_y = [], []

    groups = (
        df.groupby("source") if "source" in df.columns
        else [("__all__", df)]
    )

    min_samples = int(FS_TARGET * WIN_SEC)

    for src, g in groups:
        g = g.reset_index(drop=True)
        if len(g) < min_samples:
            continue
        try:
            X_w, y_w = preprocess_and_window(
                g,
                accel_cols=("x", "y", "z"),
                label_col="tremor",
                fs=FS_TARGET,
                win_sec=WIN_SEC,
                overlap=OVERLAP,
            )
        except Exception as e:
            print(f"  [{dataset_name}] Error on source '{src}': {e}")
            continue

        if len(X_w) == 0:
            continue

        all_X.append(X_w.transpose(0, 2, 1).astype(np.float32))
        all_y.append(y_w)

    if not all_X:
        raise RuntimeError(f"No windows loaded for {dataset_name}")

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    # Dummy meta for train_test_split compatibility
    meta = pd.DataFrame({"source_file": ["csv"] * len(y)})

    X_tmp, X_test, y_tmp, y_test, meta_tmp, _ = train_test_split(
        X, y, meta, test_size=0.15, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val, _, _ = train_test_split(
        X_tmp, y_tmp, meta_tmp,
        test_size=0.15 / 0.85, stratify=y_tmp, random_state=42,
    )

    print(f"  [csv] X_train {X_train.shape}  X_val {X_val.shape}  X_test {X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def load_dataset_splits(dataset_name: str, config: dict):
    """
    Route to npy or csv loading based on whether npy_dir is set.
    Returns (X_train, y_train, X_val, y_val, X_test, y_test).
    """
    npy_dir = config.get("npy_dir")
    if npy_dir is not None:
        print(f"[{dataset_name}] Loading from preprocessor .npy files in {npy_dir}/")
        return load_npy_splits(npy_dir, dataset_name)
    else:
        print(f"[{dataset_name}] Loading from raw CSV via data_filtering")
        return load_csv_splits(dataset_name, config)


def dataset_artifact_dir(dataset_name: str) -> str:
    path = os.path.join("artifacts", dataset_name)
    os.makedirs(path, exist_ok=True)
    return path


# -------------------------------------------------------------------------
# TRAINING LOOP  (unchanged from original)
# -------------------------------------------------------------------------
def train_cnn_rf(
    dataset_name, X_train, y_train, X_val, y_val, X_test, y_test,
    config, art_dir, device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{dataset_name}] Using device: {device}")

    arch  = config["cnn_arch"]
    model = _ARCH_MAP[arch]().to(device)

    pos_cases  = int(np.sum(y_train == 1))
    neg_cases  = int(np.sum(y_train == 0))
    weight_pos = (neg_cases / pos_cases) if pos_cases > 0 else 1.0

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    #change
    '''focal_criterion = (
        FocalLoss(alpha=0.75, gamma=2.0).to(device)
        if dataset_name == "kaggle-yunji" else None
    )'''
    focal_criterion = None

    X_tr_t  = torch.tensor(X_train, dtype=torch.float32)
    y_tr_t  = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    train_loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t),
        batch_size=BATCH_SIZE, shuffle=True,
    )

    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)

    best_val_f1     = 0.0
    best_model_path = os.path.join(art_dir, "best_cnn_model.pth")

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        preds_tr, trues_tr = [], []

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out, _ = model(bx)

            if focal_criterion is not None:
                loss = focal_criterion(out, by)
            else:
                batch_w = torch.where(
                    by == 1,
                    torch.full_like(by, weight_pos),
                    torch.ones_like(by),
                )
                loss = nn.functional.binary_cross_entropy(out, by, weight=batch_w)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            preds_tr.extend((out.detach().cpu().numpy() > 0.5).astype(int).tolist())
            trues_tr.extend(by.cpu().numpy().tolist())

        tr_acc = accuracy_score(trues_tr, preds_tr)

        model.eval()
        with torch.no_grad():
            val_out, _ = model(X_val_t)
            val_loss   = nn.functional.binary_cross_entropy(val_out, y_val_t).item()
            val_preds  = (val_out.cpu().numpy() > 0.5).astype(int)

        val_acc = accuracy_score(y_val, val_preds)
        val_f1  = f1_score(y_val, val_preds, average='macro', zero_division=0)

        print(
            f"[{dataset_name}] epoch {epoch+1}/{EPOCHS} | "
            f"train loss {epoch_loss/len(train_loader):.4f} acc {tr_acc:.3f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.3f} f1 {val_f1:.3f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)

    # Extract embeddings
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        _, emb_train = model(X_tr_t.to(device))
        _, emb_val   = model(X_val_t)
        _, emb_test  = model(torch.tensor(X_test, dtype=torch.float32).to(device))

    emb_train_np = emb_train.cpu().numpy()
    emb_val_np   = emb_val.cpu().numpy()
    emb_test_np  = emb_test.cpu().numpy()

    hc_train = handcrafted_features(X_train)
    hc_val   = handcrafted_features(X_val)
    hc_test  = handcrafted_features(X_test)

    X_train_rf = np.concatenate([emb_train_np, hc_train], axis=1)
    X_val_rf   = np.concatenate([emb_val_np,   hc_val],   axis=1)
    X_test_rf  = np.concatenate([emb_test_np,  hc_test],  axis=1)

    emb_dim = X_train_rf.shape[1]

    print(f"[{dataset_name}] Training Random Forest ({config['rf_estimators']} trees)...")
    rf = RandomForestClassifier(
        n_estimators=min(config['rf_estimators'], 50),
        max_depth=12,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(
        np.vstack([X_train_rf, X_val_rf]),
        np.concatenate([y_train, y_val]),
    )

    rf_path = os.path.join(art_dir, "best_rf_model.pkl")
    joblib.dump(rf, rf_path)

    rf_probs = rf.predict_proba(X_test_rf)[:, 1]
    rf_preds = rf.predict(X_test_rf)

    if "yunji" in dataset_name:
        rf_preds = (rf_probs >= 0.30).astype(int)
        print(f"[{dataset_name}] Applied custom threshold 0.30")

    test_acc = accuracy_score(y_test, rf_preds)
    test_f1  = f1_score(y_test, rf_preds, average='macro', zero_division=0)

    report      = classification_report(y_test, rf_preds, zero_division=0)
    report_dict = classification_report(y_test, rf_preds, zero_division=0, output_dict=True)

    cm = confusion_matrix(y_test, rf_preds)
    np.savetxt(os.path.join(art_dir, "confusion_matrix.txt"), cm, fmt="%d")

    mcc = matthews_corrcoef(y_test, rf_preds)
    try:
        roc_auc = roc_auc_score(y_test, rf_probs)
    except ValueError:
        roc_auc = 0.0

    prec_c, rec_c, _ = precision_recall_curve(y_test, rf_probs)
    pr_auc = auc(rec_c, prec_c)

    metrics = {
        "test_acc":      test_acc,
        "test_f1_macro": test_f1,
        "val_f1_cnn":    best_val_f1,
        "precision_pos": report_dict.get("1", {}).get("precision", 0),
        "recall_pos":    report_dict.get("1", {}).get("recall",    0),
        "f1_pos":        report_dict.get("1", {}).get("f1-score",  0),
        "mcc":           mcc,
        "roc_auc":       roc_auc,
        "pr_auc":        pr_auc,
    }

    pd.DataFrame([metrics]).to_csv(
        os.path.join(art_dir, "evaluation_metrics.csv"), index=False
    )
    with open(os.path.join(art_dir, "classification_report.txt"), "w") as fh:
        fh.write(report)

    print(f"[{dataset_name}] RF Test Acc: {test_acc:.4f} | F1 Macro: {test_f1:.4f}")

    return {
        "dataset":          dataset_name,
        "art_dir":          art_dir,
        "val_f1_cnn":       best_val_f1,
        "test_acc_rf":      test_acc,
        "test_f1_macro_rf": test_f1,
        "emb_dim":          emb_dim,
    }


# -------------------------------------------------------------------------
# TRAIN ONE DATASET  (called by run_parkinsons.py)
# -------------------------------------------------------------------------
def train_one_dataset(dataset_name: str, device: torch.device | None = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dconfig = DATASET_CONFIGS.get(dataset_name)
    if dconfig is None:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Choose from: {list(DATASET_CONFIGS)}")

    print(f"\n[train_one_dataset] Processing {dataset_name}...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset_splits(dataset_name, dconfig)

    # Optional majority-class undersampling (parkinsons-home)
    target_minority = dconfig.get("train_undersample_target")
    if target_minority is not None:
        pos_idx = np.where(y_train == 1)[0]
        neg_idx = np.where(y_train == 0)[0]
        n_pos, n_neg = len(pos_idx), len(neg_idx)
        if n_pos > 0 and n_neg > 0:
            n_neg_target = int(n_pos * (1.0 - target_minority) / target_minority)
            if n_neg_target < n_neg:
                np.random.seed(42)
                keep = np.concatenate([
                    pos_idx,
                    np.random.choice(neg_idx, size=n_neg_target, replace=False),
                ])
                np.random.shuffle(keep)
                X_train = X_train[keep]
                y_train = y_train[keep]
                print(f"[{dataset_name}] Undersampled → {len(X_train)} windows ({n_pos} pos / {n_neg_target} neg)")

    art_dir = dataset_artifact_dir(dataset_name)
    res = train_cnn_rf(
        dataset_name=dataset_name,
        X_train=X_train, y_train=y_train,
        X_val=X_val,     y_val=y_val,
        X_test=X_test,   y_test=y_test,
        config=dconfig,
        art_dir=art_dir,
        device=device,
    )

    pd.DataFrame([res]).to_csv(os.path.join(art_dir, "train_summary.csv"), index=False)
    print(f"[{dataset_name}] Summary → {art_dir}/train_summary.csv")
    return res


# -------------------------------------------------------------------------
# MAIN  — train all datasets in sequence
# -------------------------------------------------------------------------
def main():
    print("=" * 60)
    print(" Starting Training for All Datasets")
    print("=" * 60)

    results = []

    for dname, dconfig in DATASET_CONFIGS.items():
        print(f"\nProcessing {dname}...")
        try:
            X_train, y_train, X_val, y_val, X_test, y_test = load_dataset_splits(dname, dconfig)
        except Exception as e:
            print(f"Skipping {dname}: {e}")
            continue

        print(f"  Windows → train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")

        art_dir = dataset_artifact_dir(dname)

        target_minority = dconfig.get("train_undersample_target")
        if target_minority is not None:
            pos_idx = np.where(y_train == 1)[0]
            neg_idx = np.where(y_train == 0)[0]
            n_pos, n_neg = len(pos_idx), len(neg_idx)
            if n_pos > 0 and n_neg > 0:
                n_neg_target = int(n_pos * (1.0 - target_minority) / target_minority)
                if n_neg_target < n_neg:
                    np.random.seed(42)
                    keep = np.concatenate([
                        pos_idx,
                        np.random.choice(neg_idx, size=n_neg_target, replace=False),
                    ])
                    np.random.shuffle(keep)
                    X_train = X_train[keep]
                    y_train = y_train[keep]
                    print(f"  [{dname}] Undersampled → {len(X_train)} windows ({n_pos} pos / {n_neg_target} neg)")

        try:
            res = train_cnn_rf(
                dataset_name=dname,
                X_train=X_train, y_train=y_train,
                X_val=X_val,     y_val=y_val,
                X_test=X_test,   y_test=y_test,
                config=dconfig,
                art_dir=art_dir,
            )
            results.append(res)
        except Exception as e:
            print(f"  Error training {dname}: {e}")
            continue

    print("\n" + "=" * 60)
    print(" Training summary")
    print("=" * 60)
    if results:
        df_res   = pd.DataFrame(results)
        out_path = os.path.join("artifacts", "all_datasets_training_summary.csv")
        df_res.to_csv(out_path, index=False)
        print(df_res.to_string(index=False))
        print(f"\nSaved → {out_path}")
    else:
        print("No datasets were successfully processed.")


if __name__ == "__main__":
    main()
