"""Negative controls for explanation diagnostics.

Label-permutation controls follow the data randomization test of Adebayo et al.
(2018): a model is trained on randomly permuted training labels, so it can learn
nothing that generalises, and the usual explanation diagnostics are computed for it
exactly as for the model trained on the real labels.

Attributions are exact for every model family and reported on the original features:
interventional TreeSHAP on P(at risk) for XGBoost, interventional linear SHAP (log-odds)
for logistic regression, and the additive term contributions (log-odds) of the EBM, with
each pairwise interaction term split equally between its two features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.inspection import permutation_importance

from .explain import (_group_index, agreement, base_pipeline, baseline_row, deletion_curve,
                      ranking_stability, shap_values)


def _aggregate(pipe, features, contrib) -> np.ndarray:
    g = _group_index(pipe, features)
    out = np.zeros((contrib.shape[0], len(features)))
    for j, i in enumerate(g):
        out[:, i] += contrib[:, j]
    return out


def attributions(model_name: str, model, X_background: pd.DataFrame, X_explain: pd.DataFrame) -> np.ndarray:
    """Exact local attributions for XGB, LR and EBM, aggregated to the original features."""
    if model_name == "XGB":
        return shap_values(model, X_background, X_explain)
    pipe = base_pipeline(model)
    pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
    features = list(X_explain.columns)
    Zb, Ze = pre.transform(X_background), pre.transform(X_explain)
    if model_name == "LR":
        sv = shap.LinearExplainer(clf, Zb).shap_values(Ze)
        return _aggregate(pipe, features, np.asarray(sv))
    if model_name in ("EBM", "EBM_main"):
        terms = clf.eval_terms(Ze)
        # centre on the background so that contributions are relative to the same reference as SHAP
        terms = terms - clf.eval_terms(Zb).mean(axis=0)
        contrib = np.zeros((terms.shape[0], Ze.shape[1]))
        for t, feats in enumerate(clf.term_features_):
            for f in feats:
                contrib[:, f] += terms[:, t] / len(feats)
        return _aggregate(pipe, features, contrib)
    raise ValueError(model_name)


def cardinality(X: pd.DataFrame) -> np.ndarray:
    return X.nunique().to_numpy(dtype=float)


def heldout_importance(model, X_test, y_test, seed: int, n_repeats: int = 10) -> pd.DataFrame:
    """Permutation importance measured as the drop in held-out ROC-AUC."""
    r = permutation_importance(model, X_test, y_test, scoring="roc_auc", n_repeats=n_repeats,
                               random_state=seed, n_jobs=1)
    return pd.DataFrame({"feature": X_test.columns, "mean": r.importances_mean, "sd": r.importances_std})


def diagnostics(model_name, model, refit_pipe, Xtr, ytr_fit, Xte, yte, categorical, seed,
                n_explain=200, n_background=100, n_boot=10, lime=None, n_lime_stab=30) -> dict:
    """All explanation diagnostics for one fitted model (same protocol as E5)."""
    from .data import sample_ids
    bg = Xtr.iloc[sample_ids(len(Xtr), n_background, seed)]
    ex = Xte.iloc[sample_ids(len(Xte), n_explain, seed + 1)]
    S = attributions(model_name, model, bg, ex)
    glob = np.abs(S).mean(0)
    out = {"global_importance": glob}

    # global stability over bootstrap refits with the same hyper-parameters
    rng = np.random.default_rng(seed)
    imps = []
    for b in range(n_boot):
        i = rng.integers(0, len(Xtr), len(Xtr))
        m = clone(refit_pipe)
        if "clf__random_state" in m.get_params():
            m.set_params(clf__random_state=seed + b)
        m.fit(Xtr.iloc[i], ytr_fit.iloc[i])
        imps.append(np.abs(attributions(model_name, m, bg, ex)).mean(0))
    st = ranking_stability(imps, k=5)
    out["global_stability_rho"] = st["mean_pairwise_spearman"]
    out["global_stability_top5"] = st["mean_top5_overlap"]

    # deletion faithfulness on the students the model flags (all explained students if < 10 flagged)
    flag = model.predict(ex) == 1
    idx = np.where(flag)[0] if flag.sum() >= 10 else np.arange(len(ex))
    base = baseline_row(Xtr, categorical)
    sub = ex.iloc[idx]
    out["n_deletion"] = int(len(idx))
    out["deletion_attr"] = deletion_curve(model, sub, S[idx], base).tolist()
    rnd = np.random.default_rng(seed).normal(size=S[idx].shape)
    out["deletion_random"] = deletion_curve(model, sub, rnd, base).tolist()

    # cardinality
    rho, p = spearmanr(glob, cardinality(Xtr))
    out["cardinality_rho"], out["cardinality_p"] = float(rho), float(p)

    # LIME agreement and local stability (XGBoost only, as in E5)
    if lime is not None:
        L = lime.explain(ex)
        agr = agreement(S, L, k=5).mean()
        out["agreement_spearman"] = float(agr["spearman_abs"])
        out["agreement_top5"] = float(agr["top5_overlap"])
        out["deletion_lime"] = deletion_curve(model, sub, L[idx], base).tolist()
        runs = [lime.explain(ex.iloc[:n_lime_stab], seed=seed + s) for s in range(5)]
        ls = pd.DataFrame([ranking_stability([np.abs(r[j]) for r in runs], k=5)
                           for j in range(n_lime_stab)]).mean()
        out["lime_stability_top5"] = float(ls["mean_top5_overlap"])

    # performance-anchored check: held-out permutation importance
    hi = heldout_importance(model, Xte, yte, seed)
    out["heldout_importance_sum"] = float(hi["mean"].clip(lower=0).sum())
    out["heldout_n_features_gt_2sd"] = int(((hi["mean"] - 2 * hi["sd"]) > 0).sum())
    out["heldout_max"] = float(hi["mean"].max())
    return out
