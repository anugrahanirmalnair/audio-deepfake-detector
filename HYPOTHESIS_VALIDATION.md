# Hypothesis validation — ASVspoof LA

## Question

Can real vs spoof speech be separated using **DAC reconstruction error + codec token/latent structure** (project-brief dual stream), without Wav2Vec2/AASIST?

**Winning config:** `8_brief_full_windowed` (295 features: recon + latent + per-codebook tokens/transitions/histograms + 500 ms windows).

---

## Results summary

### 400 clips (200 bonafide + 200 spoof)

| Metric | Value |
|--------|-------|
| Accuracy (5-fold CV) | **79.5%** ± 0.6% |
| ROC-AUC | **0.845** |
| EER | **21.5%** |

**Confusion matrix** (rows = true, cols = predicted):

|  | pred bonafide | pred spoof |
|--|---------------|------------|
| true bonafide | 172 | 28 |
| true spoof | 54 | 146 |

Artifacts: `reports/evaluation/asvspoof_400_metrics.json`, `asvspoof_400_confusion_matrix.png`, `asvspoof_400_roc_curve.png`

---

### 2000 clips (1000 bonafide + 1000 spoof)

| Metric | Value |
|--------|-------|
| Accuracy (5-fold CV) | **81.5%** ± 1.4% |
| ROC-AUC | **0.897** |
| EER | **19.1%** |

**Confusion matrix:**

|  | pred bonafide | pred spoof |
|--|---------------|------------|
| true bonafide | 875 | 125 |
| true spoof | 245 | 755 |

Artifacts: `reports/evaluation/asvspoof_2k_metrics.json`, `asvspoof_2k_confusion_matrix.png`, `asvspoof_2k_roc_curve.png`

---

### 400 vs 2k comparison

| Metric | 400 clips | 2k clips | Δ (2k − 400) |
|--------|-----------|----------|----------------|
| ROC-AUC | 0.845 | 0.897 | **+0.052** |
| EER | 21.5% | 19.1% | **−2.4 pts** (better) |
| Accuracy | 79.5% | 81.5% | **+2.0 pts** |

Comparison JSON: `reports/evaluation/asvspoof_2k_vs_400.json`

**Takeaway:** Scaling to 2k **improves** metrics slightly — the effect is **stable**, not a small-sample fluke.

---

### Speaker-disjoint CV (2k, stricter)

| Metric | StratifiedKFold | GroupKFold (by speaker) |
|--------|-----------------|-------------------------|
| ROC-AUC | 0.897 | **0.862** |
| EER | 19.1% | **22.8%** |
| Accuracy | 81.5% | **77.3%** |

See [LEAKAGE_AUDIT.md](LEAKAGE_AUDIT.md) for fold/leakage details.

---

## Conclusion

**Hypothesis supported** on 400 and 2k clips. Dual-stream DAC features generalize modestly better at 2k. Use speaker-disjoint numbers when claiming robustness to new speakers.

**Not** ASVspoof leaderboard performance — prototype gate before building FastAPI + UI.

---

## Data & code artifacts

| File | Description |
|------|-------------|
| `asvspoof_sample.csv` | Original 400-clip list |
| `asvspoof_sample_2k.csv` | 2k list + `speaker_id`, `attack_id` |
| `asvspoof_features_full.csv` | Features for 400 clips |
| `asvspoof_features_2k.csv` | Features for 2k clips (~41 min extraction) |
| `src/data/build_asvspoof_sample.py` | Build balanced subsets |
| `src/features/extract_hypothesis_features.py` | Dual-stream extraction (`--resume`) |
| `src/validation/evaluate_winning_pipeline.py` | AUC, EER, confusion matrix, plots |
| `src/validation/leakage_audit.py` | Speaker/attack overlap audit |
| `src/validation/evaluate_speaker_disjoint.py` | GroupKFold by speaker |

## Reproduce

```bash
# 2k sample (if needed)
./venv/bin/python src/data/build_asvspoof_sample.py --n-per-class 1000

# Features (~40 min for 2k on CPU)
./venv/bin/python src/features/extract_hypothesis_features.py \
  --sample-csv asvspoof_sample_2k.csv \
  --output-csv asvspoof_features_2k.csv --resume

# Metrics + plots
./venv/bin/python src/validation/evaluate_winning_pipeline.py \
  --features-csv asvspoof_features_2k.csv \
  --dataset-name asvspoof_2k \
  --compare-json reports/evaluation/asvspoof_400_metrics.json
```
