"""
Validate project-brief hypothesis on ASVspoof subset.

Hypothesis: real vs spoof speech is separable using DAC reconstruction error
and codec token/latent structure (dual-stream), without a large anti-spoofing backbone.

Pass thresholds (400-clip subset, 5-fold CV) are intentionally modest — they gate
whether building the deployed system is justified, not whether we beat ASVspoof SOTA.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.preprocessing import LabelEncoder

FEATURES_DIR = Path(__file__).resolve().parents[1] / "features"
sys.path.insert(0, str(FEATURES_DIR))

from dac_token_features import (  # noqa: E402
    feature_columns_for_set,
    infer_n_codebooks,
    latent_feature_columns,
    recon_feature_columns,
    token_feature_columns,
    window_feature_columns,
)

ROOT = Path(__file__).resolve().parents[2]

# Modest gates for 400-sample exploration (not full ASVspoof leaderboard).
PASS_MEAN_AUC = 0.72
PASS_MEAN_EER = 0.28


def compute_eer(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[idx] + fnr[idx]) / 2)


def evaluate_split(
    df: pd.DataFrame,
    feature_cols: list[str],
    name: str,
) -> dict:
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name}: missing columns {missing[:5]}... ({len(missing)} total)")

    X = df[feature_cols]
    y = LabelEncoder().fit_transform(df["label"])

    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    acc_scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    proba = cross_val_predict(clf, X, y, cv=5, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, proba)
    eer = compute_eer(y, proba)

    passed = auc >= PASS_MEAN_AUC and eer <= PASS_MEAN_EER

    print(f"\n{name}")
    print(f"  features: {len(feature_cols)}")
    print(f"  accuracy: {acc_scores.mean():.4f} (+/- {acc_scores.std():.4f})")
    print(f"  ROC-AUC:  {auc:.4f}")
    print(f"  EER:      {eer:.4f}  (lower is better)")
    print(f"  gate:     AUC>={PASS_MEAN_AUC}, EER<={PASS_MEAN_EER} -> {'PASS' if passed else 'FAIL'}")

    return {
        "name": name,
        "n_features": len(feature_cols),
        "accuracy_mean": float(acc_scores.mean()),
        "accuracy_std": float(acc_scores.std()),
        "roc_auc": float(auc),
        "eer": float(eer),
        "passed": passed,
    }


def build_feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    n_cb = infer_n_codebooks(df.columns)
    token_codec = feature_columns_for_set("token_codec", n_cb)
    token_full = feature_columns_for_set("token_full", n_cb)
    recon = recon_feature_columns(df.columns)
    latent = latent_feature_columns(df.columns)
    window = window_feature_columns(df.columns)

    dual = recon + latent + token_codec
    dual_full = recon + latent + token_full
    brief = dual_full + window

    return {
        "1_baseline_global_tokens": ["entropy", "repeat_rate"],
        "2_token_codec_enriched": token_codec,
        "3_token_full_histogram": token_full,
        "4_reconstruction_only": recon,
        "5_latent_only": latent,
        "6_dual_stream": dual,
        "7_dual_stream_full": dual_full,
        "8_brief_full_windowed": brief,
        "9_windowed_only": window,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features-csv",
        default=str(ROOT / "asvspoof_features_full.csv"),
    )
    parser.add_argument(
        "--report-json",
        default=str(ROOT / "hypothesis_validation_report.json"),
    )
    args = parser.parse_args()

    df = pd.read_csv(args.features_csv)
    print("=== Hypothesis validation (ASVspoof 400-clip subset, 5-fold CV) ===")
    print(f"Samples: {len(df)} | labels: {dict(df['label'].value_counts())}")

    results = []
    for name, cols in build_feature_sets(df).items():
        if not cols:
            print(f"\n{name}: skipped (no columns)")
            continue
        results.append(evaluate_split(df, cols, name))

    best = max(results, key=lambda r: r["roc_auc"])
    any_pass = any(r["passed"] for r in results)

    print("\n" + "=" * 60)
    print(f"Best ROC-AUC: {best['roc_auc']:.4f} ({best['name']})")
    print(f"Hypothesis gates passed by any split: {any_pass}")
    print(
        "Recommendation:",
        "PROCEED with deployed build"
        if any_pass
        else "Signal present but weak — consider learned head on tokens before full UI",
    )

    report = {
        "thresholds": {"min_roc_auc": PASS_MEAN_AUC, "max_eer": PASS_MEAN_EER},
        "results": results,
        "best": best,
        "hypothesis_supported": any_pass,
    }
    Path(args.report_json).write_text(json.dumps(report, indent=2))
    print(f"\nWrote {args.report_json}")


if __name__ == "__main__":
    main()
