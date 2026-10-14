#!/usr/bin/env bash
# Reproduce every table and figure. Usage: bash run_all.sh  (from the repo root, venv active)
set -euo pipefail
PY=${PYTHON:-python}
mkdir -p results/logs
$PY -c "import sys; sys.path.insert(0, 'src'); from xaiedu.data import download_all; download_all()"
for s in 01_kaggle_baseline_and_audit 01b_kaggle_resampling_leakage 02_uci_predictive_study 03_oulad_early_warning \
         04_explanations_fairness_counterfactuals 04b_lime_binary_default_comparison 05_figures_and_tables; do
  echo "=== $s"
  $PY scripts/$s.py 2>&1 | tee results/logs/$s.log
done
