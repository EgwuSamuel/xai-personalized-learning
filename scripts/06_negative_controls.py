"""E7: label-permutation negative controls and other model families.

For each dataset (Kaggle, UCI end of semester 1, OULAD day 56) and model family
(XGBoost, logistic regression, EBM), the explanation diagnostics of E5 are computed
for the model trained on the real labels and for models trained on randomly
permuted training labels (data randomization test, Adebayo et al., 2018). Test
labels are never permuted, so test ROC-AUC shows whether a model learned anything.

Usage: python scripts/06_negative_controls.py [--datasets ...] [--models ...] [--perms N]
Each run is cached in results/cache/e7/, so the script can be stopped and resumed.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from xaiedu.config import CACHE, SEED, TABLES
from xaiedu.controls import diagnostics
from xaiedu.data import load_kaggle, load_oulad, load_uci
from xaiedu.evaluate import all_metrics, fit_model
from xaiedu.explain import LimeWrapper, base_pipeline

ap = argparse.ArgumentParser()
ap.add_argument("--datasets", nargs="+", default=["kaggle", "uci_sem1", "oulad_d56"])
ap.add_argument("--models", nargs="+", default=["XGB", "LR", "EBM_main"])
ap.add_argument("--perms", type=int, default=3)
ap.add_argument("--only", nargs="+", default=None, help="run only these conditions, e.g. perm2 perm1")
args = ap.parse_args()

OUT = CACHE / "e7"
OUT.mkdir(parents=True, exist_ok=True)
LOADERS = {"kaggle": load_kaggle, "uci_sem1": lambda: load_uci("sem1"), "oulad_d56": lambda: load_oulad(56)}


def split(ds):
    if ds.groups is not None:  # OULAD: temporal split, as in E4/E5
        tr = ds.groups.str.startswith("2013").to_numpy()
        return ds.X[tr], ds.X[~tr], ds.y[tr], ds.y[~tr]
    return train_test_split(ds.X, ds.y, test_size=0.2, stratify=ds.y, random_state=SEED)


for key in args.datasets:
    ds = LOADERS[key]()
    Xtr, Xte, ytr, yte = split(ds)
    conditions = [("real", None)] + [(f"perm{r}", r) for r in range(args.perms)]
    if args.only:
        conditions = [c for name in args.only for c in conditions if c[0] == name]
    for model_name in args.models:
        for cond, r in conditions:
            path = OUT / f"{key}__{model_name}__{cond}.json"
            if path.exists():
                continue
            t0 = time.time()
            if r is None:
                y_fit = ytr
            else:  # permute training labels only
                y_fit = pd.Series(np.random.default_rng(1000 + r).permutation(ytr.to_numpy()),
                                  index=ytr.index, name=ytr.name)
            model, params, thr = fit_model(model_name, ds.categorical, ds.numeric, Xtr, y_fit,
                                           "weight+threshold")
            proba, pred = model.predict_proba(Xte)[:, 1], model.predict(Xte)
            perf = all_metrics(yte, proba, pred)
            lime = LimeWrapper(model, Xtr, ds.categorical, seed=SEED) if model_name == "XGB" else None
            d = diagnostics(model_name, model, base_pipeline(model), Xtr, y_fit, Xte, yte,
                            ds.categorical, SEED, lime=lime)
            d.update({"dataset": key, "model": model_name, "condition": cond,
                      "test_roc_auc": perf["roc_auc"], "test_pr_auc": perf["pr_auc"],
                      "prevalence": perf["prevalence"], "threshold": thr,
                      "best_params": json.dumps({k: str(v) for k, v in params.items()}),
                      "features": list(ds.X.columns), "seconds": time.time() - t0})
            d["global_importance"] = np.asarray(d["global_importance"]).tolist()
            path.write_text(json.dumps(d, default=float))
            print(f"{key} {model_name} {cond}: AUC {perf['roc_auc']:.3f} "
                  f"stab {d['global_stability_rho']:.3f} card {d['cardinality_rho']:.3f} "
                  f"heldout {d['heldout_importance_sum']:.4f} ({time.time() - t0:.0f}s)", flush=True)

# --------------------------------------------------------------- summary table
rows = []
for f in sorted(OUT.glob("*.json")):
    d = json.loads(f.read_text())
    rows.append({k: v for k, v in d.items() if not isinstance(v, list)} | {
        "deletion_attr_k5": d["deletion_attr"][-1], "deletion_random_k5": d["deletion_random"][-1],
        "deletion_lime_k5": d.get("deletion_lime", [np.nan])[-1]})
if rows:
    df = pd.DataFrame(rows)
    # similarity of each model's global importance to the real-label model of the same family
    sims = []
    for _, row in df.iterrows():
        real = OUT / f"{row['dataset']}__{row['model']}__real.json"
        mine = OUT / f"{row['dataset']}__{row['model']}__{row['condition']}.json"
        if real.exists():
            a = json.loads(real.read_text())["global_importance"]
            b = json.loads(mine.read_text())["global_importance"]
            from scipy.stats import spearmanr
            sims.append(float(spearmanr(a, b).statistic))
        else:
            sims.append(np.nan)
    df["rho_vs_real_model"] = sims
    df.to_csv(TABLES / "e7_negative_controls.csv", index=False)
    print(df.drop(columns=["features"], errors="ignore").round(3).to_string())
