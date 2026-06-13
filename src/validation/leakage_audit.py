"""
Leakage and split integrity audit for DAC feature + RF evaluation.

Run after feature CSV exists; uses same sample metadata as training list.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

FEATURES_DIR = Path(__file__).resolve().parents[1] / "features"
sys.path.insert(0, str(FEATURES_DIR))

from dac_token_features import infer_n_codebooks  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CV_FOLDS = 5
RANDOM_STATE = 42


def fold_assignments(n_samples: int, labels: np.ndarray) -> np.ndarray:
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_id = np.full(n_samples, -1, dtype=int)
    for k, (_, test_idx) in enumerate(skf.split(np.zeros(n_samples), labels)):
        fold_id[test_idx] = k
    return fold_id


def speaker_leakage_stats(meta: pd.DataFrame, fold_id: np.ndarray) -> dict:
    """Speakers appearing in more than one fold (should be zero for GroupKFold)."""
    meta = meta.copy()
    meta["fold"] = fold_id
    leaked_speakers = []
    for speaker, grp in meta.groupby("speaker_id"):
        if grp["fold"].nunique() > 1:
            leaked_speakers.append(speaker)
    return {
        "n_speakers": int(meta["speaker_id"].nunique()),
        "speakers_in_multiple_folds": len(leaked_speakers),
        "speaker_overlap_across_folds": len(leaked_speakers) > 0,
        "example_leaked_speakers": leaked_speakers[:10],
    }


def attack_leakage_stats(meta: pd.DataFrame, fold_id: np.ndarray) -> dict:
    spoof = meta[meta["label"] == "spoof"].copy()
    spoof["fold"] = fold_id[spoof.index]
    per_fold_attacks = {}
    for k in range(CV_FOLDS):
        sub = spoof[spoof["fold"] == k]
        per_fold_attacks[str(k)] = sub["attack_id"].value_counts().to_dict()
    all_attacks = set(spoof["attack_id"].unique())
    folds_per_attack = {}
    for attack in sorted(all_attacks):
        folds_per_attack[attack] = int(spoof[spoof["attack_id"] == attack]["fold"].nunique())
    return {
        "n_attack_types": len(all_attacks),
        "attack_types_per_fold": per_fold_attacks,
        "each_attack_in_multiple_folds": folds_per_attack,
    }


def overlap_with_prior_sample(
    meta: pd.DataFrame,
    prior_paths: set[str],
) -> dict:
    current_paths = set(meta["path"])
    overlap = current_paths & prior_paths
    return {
        "n_current": len(current_paths),
        "n_prior": len(prior_paths),
        "n_overlap": len(overlap),
        "pct_current_in_prior": len(overlap) / len(current_paths) if current_paths else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-csv", required=True)
    parser.add_argument(
        "--prior-sample-csv",
        default=str(ROOT / "asvspoof_sample.csv"),
    )
    parser.add_argument(
        "--output-json",
        default=str(ROOT / "reports" / "leakage_audit.json"),
    )
    args = parser.parse_args()

    meta = pd.read_csv(args.sample_csv)
    if "speaker_id" not in meta.columns:
        raise SystemExit("sample CSV must include speaker_id (use build_asvspoof_sample.py)")

    le_map = {"bonafide": 0, "spoof": 1}
    y = meta["label"].map(le_map).values
    fold_id = fold_assignments(len(meta), y)

    prior_paths = set()
    prior_path = Path(args.prior_sample_csv)
    if prior_path.exists():
        prior_paths = set(pd.read_csv(prior_path)["path"])

    audit = {
        "cv_protocol": {
            "method": "StratifiedKFold",
            "n_splits": CV_FOLDS,
            "shuffle": True,
            "random_state": RANDOM_STATE,
            "description": (
                "sklearn.model_selection.cross_val_predict / cross_val_score "
                "refit a fresh RandomForest on each train fold and predict the "
                "held-out fold. No sample appears in both train and test within a fold."
            ),
        },
        "feature_extraction": {
            "uses_labels": False,
            "uses_test_fold": False,
            "per_clip_only": True,
            "dac_pretrained": "fixed public weights (not trained on this subset)",
            "normalization": "none — raw handcrafted scalars fed to RF",
            "global_statistics": "none computed across dataset before CV",
        },
        "classifier_training": {
            "model": "RandomForestClassifier",
            "normalization_in_pipeline": False,
            "feature_selection_on_full_data": False,
        },
        "speaker_leakage_stratified_kfold": speaker_leakage_stats(meta, fold_id),
        "spoof_attack_coverage": attack_leakage_stats(meta, fold_id),
        "overlap_with_400_clip_sample": overlap_with_prior_sample(meta, prior_paths),
        "duplicate_paths_in_sample": int(meta["path"].duplicated().sum()),
        "risks": [
            (
                "Speaker leakage: StratifiedKFold splits by clip label, not speaker. "
                "The same speaker can appear in train and test folds. This can inflate "
                "scores vs speaker-disjoint evaluation."
            ),
            (
                "Attack-type leakage: All spoof attack families (A01–A19) appear in "
                "every fold because folds are stratified by bonafide/spoof only. "
                "The classifier may learn attack-specific codec artifacts rather than "
                "general spoof cues."
            ),
            (
                "Subset overlap: If the 2k sample uses the same random_state=42 as the "
                "400 sample, the 400 clips are a subset of the 2k; metrics are not "
                "independent across experiments."
            ),
        ],
    }

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
