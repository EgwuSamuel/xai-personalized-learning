"""E8: synthetic data in which signal strength and feature cardinality are controlled.

Ten features with known cardinality: three binary, two with 5 levels, two with 20
levels and three continuous. Only one binary and one 5-level feature carry signal,
with strength s (s = 0 is a null dataset); all high-cardinality features are noise.
For each s and seed we fit XGBoost with default hyper-parameters (max_depth 6),
a shallow XGBoost (max_depth 2), XGBoost tuned as in the main pipeline, and logistic
regression (all with class weighting), and compute the explanation diagnostics of E5/E7.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone

from xaiedu.config import SEED, TABLES
from xaiedu.controls import attributions, cardinality, heldout_importance
from xaiedu.evaluate import fit_model
from xaiedu.explain import base_pipeline, baseline_row, deletion_curve, ranking_stability
from xaiedu.models import make_pipeline

N_TRAIN, N_TEST = 4000, 2000
STRENGTHS = [0.0, 0.1, 0.25, 0.5, 1.0]
SEEDS = range(5)
SIGNAL = ["bin_1", "lvl5_1"]


def simulate(s: float, seed: int, n: int) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "bin_1": rng.integers(0, 2, n), "bin_2": rng.integers(0, 2, n), "bin_3": rng.integers(0, 2, n),
        "lvl5_1": rng.integers(0, 5, n), "lvl5_2": rng.integers(0, 5, n),
        "lvl20_1": rng.integers(0, 20, n), "lvl20_2": rng.integers(0, 20, n),
        "cont_1": rng.uniform(0, 1, n), "cont_2": rng.uniform(0, 1, n), "cont_3": rng.uniform(0, 1, n),
    }).astype(float)
    z = (2 * X["bin_1"] - 1) + (X["lvl5_1"] - 2) / np.sqrt(2)  # both standardised
    logit = -1.45 + s * z  # prevalence close to 0.19, as in the Kaggle data
    y = pd.Series((rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int), name="y")
    return X, y


rows = []
for s in STRENGTHS:
    for seed in SEEDS:
        Xtr, ytr = simulate(s, 10_000 + seed, N_TRAIN)
        Xte, yte = simulate(s, 20_000 + seed, N_TEST)
        feats = list(Xtr.columns)
        card = cardinality(Xtr)
        for variant, name, extra in [("XGB_depth6", "XGB", {}), ("XGB_depth2", "XGB", {"clf__max_depth": 2}),
                                     ("XGB_tuned", "XGB", None), ("LR", "LR", {})]:
            if extra is None:
                model, params, _ = fit_model(name, [], feats, Xtr, ytr, "weight", seed=SEED + seed, tune=True)
            else:
                model = make_pipeline(name, [], feats, "weight", y=ytr, seed=SEED + seed).set_params(**extra)
                model.fit(Xtr, ytr)
                params = extra
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(yte, model.predict_proba(Xte)[:, 1])
            bg, ex = Xtr.iloc[:100], Xte.iloc[:200]
            S = attributions(name, model, bg, ex)
            glob = np.abs(S).mean(0)
            rng = np.random.default_rng(seed)
            imps = []
            for b in range(5):
                i = rng.integers(0, len(Xtr), len(Xtr))
                m = clone(base_pipeline(model))
                if "clf__random_state" in m.get_params():
                    m.set_params(clf__random_state=SEED + b)
                m.fit(Xtr.iloc[i], ytr.iloc[i])
                imps.append(np.abs(attributions(name, m, bg, ex)).mean(0))
            stab = ranking_stability(imps, k=3)
            base = baseline_row(Xtr, [])
            d_attr = deletion_curve(model, ex, S, base, kmax=3)[-1]
            d_rand = deletion_curve(model, ex, np.random.default_rng(seed).normal(size=S.shape), base, kmax=3)[-1]
            hi = heldout_importance(model, Xte, yte, seed)
            top2 = [feats[j] for j in np.argsort(-glob)[:2]]
            rows.append({
                "strength": s, "seed": seed, "model": variant, "params": str(params), "test_roc_auc": auc,
                "global_stability_rho": stab["mean_pairwise_spearman"],
                "global_stability_top3": stab["mean_top3_overlap"],
                "deletion_attr_k3": d_attr, "deletion_random_k3": d_rand,
                "cardinality_rho": spearmanr(glob, card).statistic,
                "signal_in_top2": len(set(top2) & set(SIGNAL)) / 2,
                "top1_feature": feats[int(np.argmax(glob))],
                "heldout_importance_sum": float(hi["mean"].clip(lower=0).sum()),
                "heldout_n_features_gt_2sd": int(((hi["mean"] - 2 * hi["sd"]) > 0).sum()),
                "heldout_signal_detected": int(all(((hi.set_index("feature").loc[f, "mean"]
                                                     - 2 * hi.set_index("feature").loc[f, "sd"]) > 0)
                                                   for f in SIGNAL)),
            })
            print(rows[-1], flush=True)

df = pd.DataFrame(rows)
df.to_csv(TABLES / "e8_synthetic_runs.csv", index=False)
summ = (df.drop(columns=["seed", "top1_feature", "params"]).groupby(["model", "strength"])
        .agg(["mean", "std"]).round(3))
summ.columns = ["_".join(c) for c in summ.columns]
summ.to_csv(TABLES / "e8_synthetic_summary.csv")
print(summ.to_string())
