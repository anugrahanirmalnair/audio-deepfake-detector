"""Speaker-disjoint CV (GroupKFold) — stricter leakage check."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold, cross_val_predict, cross_val_score
from sklearn.preprocessing import LabelEncoder

FEATURES_DIR = Path(__file__).resolve().parents[1] / "features"
sys.path.insert(0, str(FEATURES_DIR))

from evaluate_winning_pipeline import compute_eer, winning_feature_columns  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-csv", required=True)
    parser.add_argument("--sample-csv", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    feats = pd.read_csv(args.features_csv)
    meta = pd.read_csv(args.sample_csv)
    if "speaker_id" in feats.columns:
        df = feats
    else:
        df = feats.merge(meta[["path", "speaker_id"]], on="path")

    cols = winning_feature_columns(df)
    X = df[cols].values
    groups = df["speaker_id"].values
    le = LabelEncoder()
    y = le.fit_transform(df["label"])

    n_groups = len(np.unique(groups))
    n_splits = min(5, n_groups)
    gkf = GroupKFold(n_splits=n_splits)

    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    acc = cross_val_score(clf, X, y, cv=gkf, groups=groups, scoring="accuracy")
    proba = cross_val_predict(
        clf, X, y, cv=gkf, groups=groups, method="predict_proba"
    )[:, 1]
    auc = float(roc_auc_score(y, proba))
    eer = compute_eer(y, proba)

    report = {
        "method": "GroupKFold",
        "n_splits": n_splits,
        "split_by": "speaker_id",
        "n_speakers": int(n_groups),
        "accuracy_mean": float(acc.mean()),
        "roc_auc": auc,
        "eer": eer,
        "note": "No speaker appears in both train and test within a fold.",
    }
    Path(args.output_json).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
