# Leakage audit — DAC dual-stream evaluation

This documents how train/test splits are built, what can leak, and what we measured on the **2k ASVspoof LA** subset.

## 1. How folds are created (primary evaluation)

| Setting | Value |
|---------|--------|
| Method | `StratifiedKFold` (sklearn) |
| Folds | 5 |
| Shuffle | Yes |
| `random_state` | 42 |
| Stratify on | Clip label (`bonafide` vs `spoof`) |

For each fold, sklearn **refits a new `RandomForestClassifier`** on the training folds only and predicts the held-out fold. **No clip appears in both train and test within the same fold.**

Implementation: `src/validation/evaluate_winning_pipeline.py` uses `cross_val_predict` / `cross_val_score` with the same `StratifiedKFold` definition.

## 2. Feature extraction — test-fold leakage?

**No label leakage in extraction.**

- Each FLAC is processed **alone**: DAC encode → decode → handcrafted stats.
- Labels (`bonafide` / `spoof`) are **not** passed into `extract_all_features()`.
- DAC weights are **fixed pretrained** (24 kHz, 8 kbps); not fine-tuned on this subset.
- Features are **not** normalized or scaled using dataset-wide statistics before CV.
- No PCA / feature selection fit on the full 2k before CV.

Pipeline: `src/features/extract_hypothesis_features.py` → `dac_pipeline.py` + recon / token / window modules.

**Important:** Extraction runs on **all clips once** and saves a CSV. That is **not** leakage by itself — leakage would occur only if the **classifier** trained on test clips. CV retrains per fold and only uses train-fold rows for fitting.

## 3. Normalization inside folds?

**None.**

- Raw scalar features go directly into Random Forest.
- No `StandardScaler`, no per-fold normalization pipeline.
- `class_weight="balanced"` only adjusts RF class weights — uses fold train labels, which is standard for imbalanced CV.

## 4. Duplicate / subset overlap

| Check | Result |
|-------|--------|
| Duplicate `path` in `asvspoof_sample_2k.csv` | **0** |
| Overlap with original `asvspoof_sample.csv` (400 clips) | **0 paths** (disjoint draws) |

The 400-clip and 2k experiments are **independent samples** (both `random_state=42` but different `n_per_class` → different draws).

## 5. Speaker leakage (real risk)

ASVspoof LA train has **20 speakers**. Our 2k subset uses all 20 (`speaker_id` in protocol).

**StratifiedKFold splits by clip, not speaker.** Audit result (`src/validation/leakage_audit.py`):

- **All 20 speakers appear in multiple folds** (train and test for the same speaker).

That can **inflate** accuracy vs a deployment scenario where new speakers are unseen.

### Stricter check: speaker-disjoint CV

`GroupKFold(n_splits=5)` grouped by `speaker_id` — no speaker in both train and test within a fold.

| Metric | StratifiedKFold (2k) | GroupKFold speaker-disjoint (2k) |
|--------|----------------------|----------------------------------|
| Accuracy | 81.5% | 77.3% |
| ROC-AUC | 0.897 | 0.862 |
| EER | 19.1% | 22.8% |

Report: `reports/evaluation/asvspoof_2k_speaker_disjoint.json`

**Interpretation:** Hypothesis still holds under speaker-disjoint eval (AUC ~0.86, EER ~23%), but scores are **~3.5 pts AUC lower** than stratified CV. Report both in papers/resume.

## 6. Spoof attack-family leakage

Spoof clips include attack IDs **A01–A06** in our 2k sample (6 types in train partition).

- Every attack type appears in **all 5 folds** (stratified by bonafide/spoof only).
- The model may learn **attack-specific** codec artifacts, not only “generic spoof.”

For LA Logical Access this is partially acceptable (same task as the benchmark), but **eval on held-out attack types** would be needed for “unseen attack” claims.

## 7. Metadata columns in feature CSV

2k extraction stores `speaker_id`, `file_id`, `attack_id` in the CSV for auditing. The evaluator **excludes** them via `METADATA_COLUMNS` — they are **not** used as RF inputs.

## 8. Reproduce audits

```bash
./venv/bin/python src/validation/leakage_audit.py \
  --sample-csv asvspoof_sample_2k.csv \
  --output-json reports/leakage_audit_2k.json

./venv/bin/python src/validation/evaluate_speaker_disjoint.py \
  --features-csv asvspoof_features_2k.csv \
  --sample-csv asvspoof_sample_2k.csv \
  --output-json reports/evaluation/asvspoof_2k_speaker_disjoint.json
```
