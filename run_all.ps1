# Reproduce every table and figure on Windows. Usage: .\run_all.ps1 (from the repo root, venv active)
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force results\logs | Out-Null
python -c "import sys; sys.path.insert(0, 'src'); from xaiedu.data import download_all; download_all()"
foreach ($s in "01_kaggle_replication_and_audit", "01b_kaggle_resampling_leakage", "02_uci_predictive_study", "03_oulad_early_warning",
               "04_explanations_fairness_counterfactuals", "04b_lime_binary_default_comparison", "05_figures_and_tables") {
    Write-Host "=== $s"
    python "scripts\$s.py" 2>&1 | Tee-Object -FilePath "results\logs\$s.log"
    if ($LASTEXITCODE -ne 0) { throw "$s failed" }
}
