"""
Evaluate winning feature config (brief_full_windowed) with full metrics.

Uses stratified 5-fold CV; reports ROC-AUC, EER, confusion matrix, and
optional comparison against a baseline run JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.preprocessing import LabelEncoder

FEATURES_DIR = Path(__file__).resolve().parents[1] / "features"
sys.path.insert(0, str(FEATURES_DIR))

from dac_token_features import (  # noqa: E402
    METADATA_COLUMNS,
    feature_columns_for_set,
    infer_n_codebooks,
    latent_feature_columns,
    recon_feature_columns,
    window_feature_columns,
)

ROOT = Path(__file__).resolve().parents[2]
WINNING_CONFIG = "8_brief_full_windowed"
CV_FOLDS = 5
RANDOM_STATE = 42


def compute_eer(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[idx] + fnr[idx]) / 2)


def winning_feature_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c not in METADATA_COLUMNS]
    n_cb = infer_n_codebooks(cols)
    token_full = feature_columns_for_set("token_full", n_cb)
    recon = recon_feature_columns(cols)
    latent = latent_feature_columns(cols)
    window = window_feature_columns(cols)
    return recon + latent + token_full + window


def evaluate(
    df: pd.DataFrame,
    feature_cols: list[str],
    dataset_name: str,
    output_dir: Path,
) -> dict:
    X = df[feature_cols].values
    le = LabelEncoder()
    y = le.fit_transform(df["label"])
    class_names = list(le.classes_)

    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    acc_scores = cross_val_score(clf, X, y, cv=skf, scoring="accuracy")
    y_pred = cross_val_predict(clf, X, y, cv=skf, method="predict")
    y_proba = cross_val_predict(clf, X, y, cv=skf, method="predict_proba")[:, 1]

    auc = float(roc_auc_score(y, y_proba))
    eer = compute_eer(y, y_proba)
    cm = confusion_matrix(y, y_pred)

    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "dataset": dataset_name,
        "n_samples": int(len(df)),
        "n_features": len(feature_cols),
        "config": WINNING_CONFIG,
        "cv_folds": CV_FOLDS,
        "cv_random_state": RANDOM_STATE,
        "class_names": class_names,
        "accuracy_mean": float(acc_scores.mean()),
        "accuracy_std": float(acc_scores.std()),
        "accuracy_per_fold": [float(s) for s in acc_scores],
        "roc_auc": auc,
        "eer": eer,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": {
            "rows_true": class_names,
            "cols_pred": class_names,
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "spoof_as_positive_class_index": int(np.where(le.classes_ == "spoof")[0][0])
        if "spoof" in le.classes_
        else 1,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = dataset_name.replace(" ", "_")

    report_path = output_dir / f"{prefix}_metrics.json"
    report_path.write_text(json.dumps(metrics, indent=2))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, cmap="Blues")
    ax.set_title(f"{dataset_name} — {WINNING_CONFIG}\nAUC={auc:.3f} EER={eer:.3f}")
    fig.tight_layout()
    cm_path = output_dir / f"{prefix}_confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y, y_proba)
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax2.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax2.set_xlabel("False positive rate")
    ax2.set_ylabel("True positive rate")
    ax2.set_title(f"{dataset_name} ROC")
    ax2.legend()
    fig2.tight_layout()
    roc_path = output_dir / f"{prefix}_roc_curve.png"
    fig2.savefig(roc_path, dpi=150)
    plt.close(fig2)

    metrics["artifacts"] = {
        "metrics_json": str(report_path),
        "confusion_matrix_png": str(cm_path),
        "roc_curve_png": str(roc_path),
    }
    return metrics


def compare_runs(current: dict, baseline: dict | None) -> dict:
    if baseline is None:
        return {"note": "no baseline provided"}
    return {
        "baseline_dataset": baseline.get("dataset"),
        "current_dataset": current.get("dataset"),
        "roc_auc_delta": current["roc_auc"] - baseline["roc_auc"],
        "eer_delta": current["eer"] - baseline["eer"],
        "accuracy_delta": current["accuracy_mean"] - baseline["accuracy_mean"],
        "baseline": {
            "roc_auc": baseline["roc_auc"],
            "eer": baseline["eer"],
            "accuracy_mean": baseline["accuracy_mean"],
        },
        "current": {
            "roc_auc": current["roc_auc"],
            "eer": current["eer"],
            "accuracy_mean": current["accuracy_mean"],
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features-csv",
        required=True,
    )
    parser.add_argument(
        "--dataset-name",
        default="asvspoof_2k",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "reports" / "evaluation"),
    )
    parser.add_argument(
        "--compare-json",
        default="",
        help="Prior metrics JSON (e.g. 400-clip run) for delta table",
    )
    parser.add_argument(
        "--comparison-out",
        default="",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.features_csv)
    cols = winning_feature_columns(df)
    out_dir = Path(args.output_dir)

    print(f"Evaluating {args.dataset_name}: n={len(df)}, features={len(cols)}")
    metrics = evaluate(df, cols, args.dataset_name, out_dir)

    print(f"\n{args.dataset_name} ({WINNING_CONFIG})")
    print(f"  Accuracy: {metrics['accuracy_mean']:.4f} (+/- {metrics['accuracy_std']:.4f})")
    print(f"  ROC-AUC:  {metrics['roc_auc']:.4f}")
    print(f"  EER:      {metrics['eer']:.4f}")
    print(f"  Confusion matrix (rows=true, cols=pred):")
    print(f"    {metrics['class_names']}")
    for row in metrics["confusion_matrix"]:
        print(f"    {row}")

    baseline = None
    if args.compare_json and Path(args.compare_json).exists():
        baseline = json.loads(Path(args.compare_json).read_text())

    comparison = compare_runs(metrics, baseline)
    comp_path = Path(args.comparison_out or out_dir / f"{args.dataset_name}_vs_baseline.json")
    comp_path.write_text(json.dumps(comparison, indent=2))
    print(f"\nComparison written to {comp_path}")
    if "roc_auc_delta" in comparison:
        print(
            f"  AUC delta (2k - 400): {comparison['roc_auc_delta']:+.4f}\n"
            f"  EER delta (2k - 400): {comparison['eer_delta']:+.4f} "
            f"(positive = worse on 2k)"
        )


if __name__ == "__main__":
    main()
