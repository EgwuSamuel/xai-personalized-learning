"""E9: sensitivity of explanation diagnostics to the XGBoost hyper-parameter configuration.

The randomised search of the main pipeline evaluates ten candidate configurations.
On data without signal all candidates score alike, so which one is selected is
arbitrary. Here the diagnostics are computed for every one of the ten candidates
(the same candidates the search draws), on the Kaggle data, on UCI with permuted
training labels and on UCI with the real labels.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterSampler, train_test_split

from xaiedu.config import N_ITER_SEARCH, SEED, TABLES
from xaiedu.controls import attributions, cardinality, heldout_importance
from xaiedu.data import load_kaggle, load_uci, sample_ids
from xaiedu.explain import baseline_row, deletion_curve, ranking_stability
from xaiedu.models import PARAM_SPACE, make_pipeline

rows = []
settings = [("kaggle", load_kaggle(), False), ("uci_sem1", load_uci("sem1"), True), ("uci_sem1", load_uci("sem1"), False)]
for key, ds, permute in settings:
    Xtr, Xte, ytr, yte = train_test_split(ds.X, ds.y, test_size=0.2, stratify=ds.y, random_state=SEED)
    if permute:
        ytr = pd.Series(np.random.default_rng(1000).permutation(ytr.to_numpy()), index=ytr.index, name=ytr.name)
    cond = "permuted labels" if permute else ("no signal" if key == "kaggle" else "real labels")
    bg = Xtr.iloc[sample_ids(len(Xtr), 100, SEED)]
    ex = Xte.iloc[sample_ids(len(Xte), 200, SEED + 1)]
    base = baseline_row(Xtr, ds.categorical)
    card = cardinality(Xtr)
    candidates = list(ParameterSampler(PARAM_SPACE["XGB"], n_iter=N_ITER_SEARCH, random_state=SEED))
    for c, params in enumerate(candidates):
        pipe = make_pipeline("XGB", ds.categorical, ds.numeric, "weight", y=ytr, seed=SEED).set_params(**params)
        model = clone(pipe).fit(Xtr, ytr)
        auc = roc_auc_score(yte, model.predict_proba(Xte)[:, 1])
        S = attributions("XGB", model, bg, ex)
        glob = np.abs(S).mean(0)
        rng = np.random.default_rng(SEED)
        imps = []
        for b in range(10):
            i = rng.integers(0, len(Xtr), len(Xtr))
            m = clone(pipe).set_params(clf__random_state=SEED + b).fit(Xtr.iloc[i], ytr.iloc[i])
            imps.append(np.abs(attributions("XGB", m, bg, ex)).mean(0))
        st = ranking_stability(imps, k=5)
        d_attr = deletion_curve(model, ex, S, base)[-1]
        d_rand = deletion_curve(model, ex, np.random.default_rng(SEED).normal(size=S.shape), base)[-1]
        hi = heldout_importance(model, Xte, yte, SEED)
        rows.append({"dataset": key, "condition": cond, "candidate": c,
                     "params": json.dumps({k: (float(v) if not isinstance(v, (int, np.integer)) else int(v))
                                           for k, v in params.items()}),
                     "max_depth": int(params["clf__max_depth"]), "test_roc_auc": auc,
                     "global_stability_rho": st["mean_pairwise_spearman"],
                     "global_stability_top5": st["mean_top5_overlap"],
                     "deletion_attr_k5": d_attr, "deletion_random_k5": d_rand,
                     "cardinality_rho": spearmanr(glob, card).statistic,
                     "heldout_importance_sum": float(hi["mean"].clip(lower=0).sum())})
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items() if k != "params"}, flush=True)

df = pd.DataFrame(rows)
df.to_csv(TABLES / "e9_hyperparameter_sensitivity.csv", index=False)
summ = df.groupby(["dataset", "condition"])[["test_roc_auc", "global_stability_rho", "global_stability_top5",
                                              "deletion_attr_k5", "deletion_random_k5", "cardinality_rho",
                                              "heldout_importance_sum"]].agg(["min", "median", "max"]).round(3)
summ.to_csv(TABLES / "e9_hyperparameter_sensitivity_summary.csv")
print(summ.to_string())
