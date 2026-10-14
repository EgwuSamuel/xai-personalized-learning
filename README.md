# Explainable dropout prediction for personalised intervention: reproducible code

This repository holds the code for the experiments in the manuscript
*"Evaluating the trustworthiness of explainable AI for student dropout prediction"*. It rebuilds, and then
extends, the experiment of the MSc dissertation *Development of Explainable Artificial
Intelligence (XAI) Models for the Enhancement of Personalized Learning in Academic Environment*
(L. E. Longwul, Veritas University Abuja, 2025). That dissertation's original code was not available.

Every number and figure in the paper is produced by the scripts below. Nothing is typed in by hand.

## What the experiments do

| Script | Experiment |
|---|---|
| `01_kaggle_replication_and_audit.py` | **E1** replicates the dissertation's pipeline on the Kaggle data (one-hot encoding, 80/20 split, Random Forest with 100 trees). **E2** audits whether the data contains any signal: a label-permutation test, uniformity tests, univariate association tests, and a tuned model zoo. |
| `01b_kaggle_resampling_leakage.py` | **E2b** shows how oversampling (SMOTE or random) *before* the train/test split inflates performance on data that has no signal. |
| `02_uci_predictive_study.py` | **E3** predicts dropout on UCI #697 at three decision points (enrolment, end of semester 1, end of semester 2). Seven models are run under 5×2 repeated stratified CV with nested random search and inner-fold threshold tuning, followed by Friedman/Nemenyi tests and an imbalance-strategy ablation. |
| `03_oulad_early_warning.py` | **E4** gives early warning of withdrawal or failure in OULAD, using only data up to day 14 … 140. Validation is temporal: train on 2013, test on 2014, with bootstrap CIs. |
| `04_explanations_fairness_counterfactuals.py` | **E5** evaluates the explanations themselves: TreeSHAP and LIME agreement, stability across bootstrap refits and LIME seeds, and a deletion-based faithfulness test. It also generates actionable counterfactuals. **E6** is a group-fairness audit. |
| `05_figures_and_tables.py` | Builds all paper figures (`results/figures`) and formatted tables (`results/tables/paper_*.csv`). |

## Key results (from `results/tables`)

| Finding | Value |
|---|---|
| Thesis pipeline re-run (seed 42) | Classification report reproduced exactly; dropout recall 0, test ROC-AUC 0.531 |
| Kaggle label-permutation test (5-fold CV ROC-AUC) | 0.498 vs null 0.500 ± 0.008, *p* = 0.595 |
| Kaggle, oversampling inside training split vs before splitting | ROC-AUC 0.50 vs 0.94 (SMOTE) / 0.985 (random oversampling) |
| UCI dropout, best ROC-AUC at enrolment / end of sem. 1 / end of sem. 2 | 0.827 / 0.907 / 0.926 (5×2 nested CV) |
| OULAD at-risk, test ROC-AUC (train 2013 → test 2014), day 28 / day 140 | 0.774 / 0.895 (XGBoost) |
| SHAP global-ranking stability, Kaggle (no signal) vs UCI | Spearman 0.88 vs 0.93 |
| SHAP importance vs feature cardinality, Kaggle vs UCI vs OULAD | Spearman 0.95 vs 0.30 vs −0.11 |

## Data

The data is **not** redistributed here. `download_all()` fetches each file from its original
public source and checks the SHA-256 where the file is fixed.

| Dataset | Source | Licence |
|---|---|---|
| Personalized Learning and Adaptive Education Dataset, **version 1** (10,000 rows) | Kaggle, `adilshamim8/personalized-learning-and-adaptive-education-dataset` | CC BY 4.0 |
| Predict Students' Dropout and Academic Success (4,424 students) | UCI ML Repository #697; Realinho et al. (2022), *Data* 7(11):146 | CC BY 4.0 |
| Open University Learning Analytics Dataset (32,593 students) | UCI ML Repository #349; Kuzilek et al. (2017), *Scientific Data* 4:170171 | CC BY 4.0 |

**Note on the Kaggle dataset.** On 4 September 2025 the Kaggle page above was overwritten with an
unrelated dataset (secondary-school grades). The file used in the dissertation is still served
as **version 1** (25 February 2025). The loader pins that version and its checksum
(`931a36cb…7230cdc`).

## Reproduce

Requires Python 3.13 (tested on Windows 10 and Linux). The full run takes several hours on a
4-core CPU. Most of that time is the nested CV in E3.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q                          # fast tests on synthetic data, no download needed
bash run_all.sh                    # Windows PowerShell: .\run_all.ps1
```

All randomness is seeded (`SEED = 42` in `src/xaiedu/config.py`), and running twice gives identical tables.
`tests/test_pipeline.py` checks determinism, the leakage-free feature sets, SHAP/LIME plumbing,
and the counterfactual constraints.

## Design choices that matter for validity

* **No leakage.** Encoding, scaling and SMOTE sit inside the pipeline, so they are fitted on
  training folds only. Decision thresholds are tuned on inner folds. UCI feature sets contain only
  information available at each decision point. OULAD features use only clicks and assessments
  up to the cutoff day, and students who had already withdrawn by then are excluded.
* **Explanations on original features.** SHAP values of one-hot columns are summed back to
  their source feature (SHAP is additive). LIME runs in the original feature space.
* **Counterfactuals** may change only actionable features (e.g. approved or evaluated units,
  tuition status, scholarship), within the observed range and in the plausible direction. They
  must also satisfy domain constraints (approved ≤ enrolled and ≤ evaluated units; a non-zero
  semester grade requires at least one approved unit).

## Layout

```
src/xaiedu/   data.py models.py evaluate.py stats.py signal_tests.py explain.py
              counterfactuals.py fairness.py config.py
scripts/      01 … 05 experiment scripts
tests/        test_pipeline.py
results/      tables/ figures/ logs/   (generated)
```

## Licence

Code: MIT (see `LICENSE`). The datasets keep their own licences (CC BY 4.0).
