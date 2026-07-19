"""
evaluate_ensemble.py
────────────────────
Evaluates the full weighted CNN+RF ensemble across all four datasets,
with automatic threshold tuning on the validation set.

Pipeline:
  1. Load all 4 CNN+RF model pairs from artifacts/
  2. For each dataset, compute ensemble probabilities on the VAL set
  3. Sweep thresholds 0.05–0.95 to find the one that maximises val F1-macro
  4. Apply that best threshold to the TEST set and report all metrics
  5. Also evaluate a pooled test set across all datasets

Metrics: test_acc, test_f1_macro, val_f1_cnn, precision_pos, recall_pos,
         f1_pos, mcc, roc_auc, pr_auc

Usage:  python evaluate_ensemble.py
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    matthews_corrcoef, precision_recall_curve, auc,
    roc_auc_score, confusion_matrix,
)

sys.path.insert(0, os.path.dirname(__file__))

from updated_train_processed import (
    load_dataset_splits,
    handcrafted_features,
    DATASET_CONFIGS,
    ShallowCNN, StandardCNN, DeepWideCNN,
)

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_NAMES    = ["jiehu-rima", "kaggle-arushna", "kaggle-yunji"]
ENSEMBLE_WEIGHTS = np.array([0.45, 0.40, 0.15])

_ARCH_MAP = {
    "jiehu-rima":      ShallowCNN,
    "kaggle-arushna":  StandardCNN,
    "kaggle-yunji":    StandardCNN,
    "parkinsons-home": DeepWideCNN,
}

THRESHOLD_SWEEP = np.round(np.arange(0.05, 0.96, 0.05), 2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model(dataset_name: str):
    art_dir  = os.path.join("artifacts", dataset_name)
    cnn_path = os.path.join(art_dir, "best_cnn_model.pth")
    rf_path  = os.path.join(art_dir, "best_rf_model.pkl")

    if not os.path.exists(cnn_path) or not os.path.exists(rf_path):
        print(f"  [WARN] No artifacts for {dataset_name} — skipping.")
        return None, None

    cnn = _ARCH_MAP[dataset_name]()
    cnn.load_state_dict(torch.load(cnn_path, map_location=device))
    cnn.to(device).eval()

    rf = joblib.load(rf_path)
    return cnn, rf


# ── Inference ─────────────────────────────────────────────────────────────────
def member_proba(cnn, rf, X: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    """Run one CNN+RF member on X (N,3,T), return P(tremor=1) per window using batching."""
    cnn.eval()
    emb_list = []
    
    # 1. Process CNN in batches to save GPU VRAM
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch_x = X[i : i + batch_size]
            tensor = torch.tensor(batch_x, dtype=torch.float32).to(device)
            _, emb = cnn(tensor)
            emb_list.append(emb.cpu().numpy())
            
    emb_np = np.vstack(emb_list)
    
    # 2. Compute handcrafted features on CPU
    hc = handcrafted_features(X)
    
    # 3. Combine and pass to Random Forest
    rf_in = np.concatenate([emb_np, hc], axis=1)
    return rf.predict_proba(rf_in)[:, 1]


def ensemble_proba(models: dict, loaded_names: list, X: np.ndarray) -> np.ndarray:
    """Weighted average of all member probabilities."""
    raw_w = np.array([ENSEMBLE_WEIGHTS[DATASET_NAMES.index(n)] for n in loaded_names])
    w     = raw_w / raw_w.sum()
    probs = np.stack(
        [member_proba(models[n]["cnn"], models[n]["rf"], X) for n in loaded_names],
        axis=0,
    )
    return np.average(probs, axis=0, weights=w)


# ── Threshold tuning ──────────────────────────────────────────────────────────
def tune_threshold(y_val: np.ndarray, val_prob: np.ndarray) -> tuple[float, float]:
    """
    Sweep thresholds on the VAL set and return (best_threshold, best_f1_macro).
    Optimises for F1-macro so both classes matter equally.
    """
    best_thr, best_f1 = 0.5, 0.0
    print("  Threshold sweep (val set):")
    for thr in THRESHOLD_SWEEP:
        preds = (val_prob >= thr).astype(int)
        f1    = f1_score(y_val, preds, average="macro", zero_division=0)
        # Show the sweep so you can see what's happening
        bar = "█" * int(f1 * 20)
        print(f"    thr={thr:.2f}  val_f1_macro={f1:.4f}  {bar}")
        if f1 > best_f1:
            best_f1  = f1
            best_thr = thr
    print(f"  → Best threshold: {best_thr:.2f}  (val_f1_macro={best_f1:.4f})")
    return best_thr, best_f1


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_prob, label="") -> dict:
    acc    = accuracy_score(y_true, y_pred)
    f1_mac = f1_score(y_true, y_pred, average="macro", zero_division=0)
    rep    = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    prec   = rep.get("1", {}).get("precision", 0.0)
    rec    = rep.get("1", {}).get("recall",    0.0)
    f1p    = rep.get("1", {}).get("f1-score",  0.0)
    mcc    = matthews_corrcoef(y_true, y_pred)

    try:
        roc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc = float("nan")

    prec_c, rec_c, _ = precision_recall_curve(y_true, y_prob)
    pr  = auc(rec_c, prec_c)
    cm  = confusion_matrix(y_true, y_pred)

    return dict(
        split         = label,
        test_acc      = round(acc,   4),
        test_f1_macro = round(f1_mac,4),
        precision_pos = round(prec,  4),
        recall_pos    = round(rec,   4),
        f1_pos        = round(f1p,   4),
        mcc           = round(mcc,   4),
        roc_auc       = round(roc,   4),
        pr_auc        = round(pr,    4),
        n_samples     = int(len(y_true)),
        n_tremor      = int(y_true.sum()),
        confusion_matrix = cm.tolist(),
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print(" Ensemble Evaluation with Threshold Tuning")
    print(f" Weights: " + "  ".join(
        f"{n}={w:.0%}" for n, w in zip(DATASET_NAMES, ENSEMBLE_WEIGHTS)
    ))
    print("=" * 65)

    # 1. Load models
    models, loaded_names = {}, []
    for dname in DATASET_NAMES:
        print(f"\nLoading {dname}...")
        cnn, rf = load_model(dname)
        if cnn is not None:
            models[dname] = {"cnn": cnn, "rf": rf}
            loaded_names.append(dname)

    if not loaded_names:
        print("ERROR: No models loaded. Run updated_train.py first.")
        sys.exit(1)

    print(f"\nLoaded {len(loaded_names)} / {len(DATASET_NAMES)} models")

    # 2. Per-dataset: load splits, tune threshold on val, evaluate on test
    rows = []
    all_y_true, all_y_prob = [], []
    best_thresholds = {}

    for dname in loaded_names:
        config = DATASET_CONFIGS[dname]
        print(f"\n{'─'*65}")
        print(f" {dname}")
        print(f"{'─'*65}")

        X_train, y_train, X_val, y_val, X_test, y_test = load_dataset_splits(dname, config)

        print(f"  val  windows: {len(X_val)}   (tremor={int(y_val.sum())}  control={int((y_val==0).sum())})")
        print(f"  test windows: {len(X_test)}  (tremor={int(y_test.sum())}  control={int((y_test==0).sum())})")

        # Ensemble probs on val → tune threshold
        print(f"\n  Computing ensemble probabilities on val set...")
        val_prob  = ensemble_proba(models, loaded_names, X_val)

        # Show probability distribution so we understand what's happening
        print(f"  Val prob distribution:")
        print(f"    min={val_prob.min():.3f}  max={val_prob.max():.3f}  "
              f"mean={val_prob.mean():.3f}  median={np.median(val_prob):.3f}")
        print(f"    % above 0.5: {(val_prob >= 0.5).mean()*100:.1f}%")

        best_thr, best_val_f1 = tune_threshold(y_val, val_prob)
        best_thresholds[dname] = best_thr

        # Apply best threshold to test set
        print(f"\n  Evaluating test set at threshold={best_thr:.2f}...")
        test_prob = ensemble_proba(models, loaded_names, X_test)
        test_pred = (test_prob >= best_thr).astype(int)

        metrics = compute_metrics(y_test, test_pred, test_prob, label=dname)
        metrics["val_f1_cnn"]        = float("nan")
        metrics["best_threshold"]    = best_thr
        metrics["val_f1_macro_tuned"]= round(best_val_f1, 4)

        # Load val_f1_cnn from training artifacts
        saved_csv = os.path.join("artifacts", dname, "evaluation_metrics.csv")
        if os.path.exists(saved_csv):
            saved = pd.read_csv(saved_csv)
            metrics["val_f1_cnn"] = round(float(saved["val_f1_cnn"].iloc[0]), 4)

        cm = np.array(metrics["confusion_matrix"])
        print(f"  Results:")
        print(f"    acc={metrics['test_acc']:.4f}  f1_macro={metrics['test_f1_macro']:.4f}  "
              f"f1_pos={metrics['f1_pos']:.4f}  recall={metrics['recall_pos']:.4f}")
        print(f"    mcc={metrics['mcc']:.4f}  roc_auc={metrics['roc_auc']:.4f}  pr_auc={metrics['pr_auc']:.4f}")
        print(f"    Confusion matrix:  TN={cm[0,0]}  FP={cm[0,1]}  FN={cm[1,0]}  TP={cm[1,1]}")

        rows.append(metrics)
        all_y_true.append(y_test)
        all_y_prob.append(test_prob)

    # 3. Pooled evaluation
    if len(all_y_true) > 1:
        print(f"\n{'─'*65}")
        print(" Pooled evaluation (all datasets combined)")
        print(f"{'─'*65}")

        y_pool    = np.concatenate(all_y_true)
        prob_pool = np.concatenate(all_y_prob)

        # Tune threshold on pooled val set
        all_y_val, all_val_prob = [], []
        for dname in loaded_names:
            config   = DATASET_CONFIGS[dname]
            _, _, X_val, y_val, _, _ = load_dataset_splits(dname, config)
            vp = ensemble_proba(models, loaded_names, X_val)
            all_y_val.append(y_val)
            all_val_prob.append(vp)

        y_val_pool   = np.concatenate(all_y_val)
        prob_val_pool = np.concatenate(all_val_prob)

        print("  Tuning pooled threshold on combined val set...")
        best_thr_pool, _ = tune_threshold(y_val_pool, prob_val_pool)

        pred_pool = (prob_pool >= best_thr_pool).astype(int)
        pooled    = compute_metrics(y_pool, pred_pool, prob_pool, label="POOLED")
        pooled["val_f1_cnn"]         = float("nan")
        pooled["best_threshold"]     = best_thr_pool
        pooled["val_f1_macro_tuned"] = float("nan")

        cm = np.array(pooled["confusion_matrix"])
        print(f"  Total: {pooled['n_samples']} windows  "
              f"(tremor={pooled['n_tremor']}  control={pooled['n_samples']-pooled['n_tremor']})")
        print(f"  acc={pooled['test_acc']:.4f}  f1_macro={pooled['test_f1_macro']:.4f}  "
              f"f1_pos={pooled['f1_pos']:.4f}  recall={pooled['recall_pos']:.4f}")
        print(f"  TN={cm[0,0]}  FP={cm[0,1]}  FN={cm[1,0]}  TP={cm[1,1]}")
        rows.append(pooled)

    # 4. Save
    out_cols = [
        "split", "best_threshold", "n_samples", "n_tremor",
        "test_acc", "test_f1_macro", "val_f1_cnn", "val_f1_macro_tuned",
        "precision_pos", "recall_pos", "f1_pos",
        "mcc", "roc_auc", "pr_auc",
    ]
    df_out = pd.DataFrame(rows)
    for c in out_cols:
        if c not in df_out.columns:
            df_out[c] = float("nan")
    df_out = df_out[out_cols]

    out_path = os.path.join("artifacts", "ensemble_evaluation.csv")
    df_out.to_csv(out_path, index=False)

    cm_path = os.path.join("artifacts", "ensemble_confusion_matrices.txt")
    with open(cm_path, "w") as fh:
        for row in rows:
            fh.write(f"\n{row['split']}  (threshold={row.get('best_threshold', 0.5):.2f})\n")
            cm = np.array(row["confusion_matrix"])
            fh.write(f"  TN={cm[0,0]}  FP={cm[0,1]}\n")
            fh.write(f"  FN={cm[1,0]}  TP={cm[1,1]}\n")

    # Save best thresholds so updated_main.py can load them at inference time
    thr_path = os.path.join("artifacts", "best_thresholds.csv")
    pd.DataFrame([
        {"dataset": k, "threshold": v} for k, v in best_thresholds.items()
    ]).to_csv(thr_path, index=False)

    print("\n" + "=" * 65)
    print(" Summary")
    print("=" * 65)
    display_cols = [
        "split", "best_threshold", "test_acc", "test_f1_macro",
        "precision_pos", "recall_pos", "f1_pos", "mcc", "roc_auc", "pr_auc",
    ]
    print(df_out[display_cols].to_string(index=False))
    print(f"\nSaved → {out_path}")
    print(f"Saved → {cm_path}")
    print(f"Saved → {thr_path}  ← use these thresholds in production inference")

    print("\n" + "=" * 65)
    print(" How to improve the model — next steps")
    print("=" * 65)
    for row in rows:
        if row["split"] == "POOLED":
            continue
        dname = row["split"]
        rec   = row["recall_pos"]
        prec  = row["precision_pos"]
        f1    = row["f1_pos"]
        thr   = row.get("best_threshold", 0.5)
        print(f"\n  {dname}  (best_thr={thr:.2f}  recall={rec:.3f}  precision={prec:.3f}  f1={f1:.3f})")
        if rec < 0.4:
            print("    ⚠ Low recall — model misses most tremors")
            print("    → Lower the threshold further or increase class weight in training")
            print("    → Check class balance in the preprocessor output")
        if prec < 0.4:
            print("    ⚠ Low precision — too many false alarms")
            print("    → Raise the threshold or add more negative training examples")
        if f1 > 0.6:
            print("    ✓ This member is performing well")


if __name__ == "__main__":
    main()