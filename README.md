# Personalized Learning

Code for evaluating explainable AI (SHAP, LIME and counterfactual explanations) of student dropout models.
Requires Python 3.13. Install the pinned dependencies with `pip install -r requirements.txt`.
Run `bash run_all.sh` (on Windows, `.\run_all.ps1`) to download the public datasets and regenerate every table and figure in `results/`.
Run `pytest -q` for the quick tests.

Scripts 06-09 add negative controls for the explanation diagnostics: models trained on permuted labels with XGBoost, logistic regression and a main-effects EBM (06), synthetic data that separate signal strength from feature cardinality (07), sensitivity to the hyperparameter configuration (08), and their figures (09). Script 06 caches each run in `results/cache/e7/`, so it can be stopped and resumed.
