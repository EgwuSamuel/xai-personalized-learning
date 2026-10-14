"""E4: early warning of withdrawal/failure in online courses (OULAD).

Features use only data up to each cutoff day. Validation is temporal: models
are trained and tuned on the 2013 presentations and tested once on the 2014
presentations, with bootstrap 95% CIs on the test set.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from xaiedu.config import SEED, TABLES
from xaiedu.data import load_oulad
from xaiedu.evaluate import bootstrap_ci, fit_model

CUTOFFS = [14, 28, 56, 84, 112, 140]
MODELS = ["LR", "RF", "XGB", "LGBM"]

rows = []
for day in CUTOFFS:
    ds = load_oulad(day)
    train = ds.groups.str.startswith("2013").to_numpy()
    Xtr, ytr, Xte, yte = ds.X[train], ds.y[train], ds.X[~train], ds.y[~train]
    print(f"day {day}: train {len(ytr)} (risk {ytr.mean():.3f}), test {len(yte)} (risk {yte.mean():.3f})")
    for name in MODELS:
        model, params, thr = fit_model(name, ds.categorical, ds.numeric, Xtr, ytr,
                                       "weight+threshold", seed=SEED)
        proba, pred = model.predict_proba(Xte)[:, 1], model.predict(Xte)
        ci = bootstrap_ci(yte, proba, pred, n_boot=500)
        for metric, r in ci.iterrows():
            rows.append({"cutoff_day": day, "model": name, "metric": metric, "threshold": thr,
                         "n_train": len(ytr), "n_test": len(yte), **r.to_dict()})
        print(f"  {name:5s} AUC={ci.loc['roc_auc', 'estimate']:.3f} "
              f"[{ci.loc['roc_auc', 'ci_low']:.3f}, {ci.loc['roc_auc', 'ci_high']:.3f}] "
              f"PR-AUC={ci.loc['pr_auc', 'estimate']:.3f} recall={ci.loc['recall_dropout', 'estimate']:.3f}")
    pd.DataFrame(rows).to_csv(TABLES / "e4_oulad_temporal_test.csv", index=False)
