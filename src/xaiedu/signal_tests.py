"""Checks for whether a dataset contains any learnable signal before explaining it."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, permutation_test_score
from sklearn.pipeline import Pipeline
from statsmodels.stats.multitest import multipletests

from .models import make_preprocessor


def label_permutation_test(ds, n_permutations=199, seed=42, n_jobs=-1) -> dict:
    """Is cross-validated ROC-AUC higher than with randomly permuted labels?"""
    pipe = Pipeline([("pre", make_preprocessor(ds.categorical, ds.numeric, scale=False)),
                     ("clf", RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1))])
    score, perm_scores, p = permutation_test_score(
        pipe, ds.X, ds.y, scoring="roc_auc", cv=StratifiedKFold(5, shuffle=True, random_state=seed),
        n_permutations=n_permutations, random_state=seed, n_jobs=n_jobs)
    return {"observed_auc": float(score), "perm_mean": float(perm_scores.mean()),
            "perm_sd": float(perm_scores.std(ddof=1)), "p_value": float(p),
            "n_permutations": n_permutations, "perm_scores": perm_scores}


def uniformity_tests(ds) -> pd.DataFrame:
    """Chi-square goodness of fit of each integer feature to a discrete uniform on [min, max]."""
    rows = []
    for c in ds.numeric:
        v = ds.X[c].to_numpy()
        if not np.allclose(v, np.round(v)):
            continue
        lo, hi = int(v.min()), int(v.max())
        observed = np.bincount(v.astype(int) - lo, minlength=hi - lo + 1)
        chi2, p = stats.chisquare(observed)
        sd_uniform = np.sqrt(((hi - lo + 1) ** 2 - 1) / 12)
        rows.append({"feature": c, "min": lo, "max": hi, "levels": hi - lo + 1,
                     "sd_observed": v.std(ddof=1), "sd_if_uniform": sd_uniform,
                     "chi2": chi2, "p_value": p})
    for c in ds.categorical:
        observed = ds.X[c].value_counts().to_numpy()
        chi2, p = stats.chisquare(observed)
        rows.append({"feature": c, "levels": len(observed), "chi2": chi2, "p_value": p})
    return pd.DataFrame(rows)


def target_association(ds) -> pd.DataFrame:
    """Univariate association of each feature with the target, Holm-corrected."""
    rows = []
    y = ds.y.to_numpy()
    for c in ds.numeric:
        a, b = ds.X.loc[y == 1, c], ds.X.loc[y == 0, c]
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        rows.append({"feature": c, "test": "Mann-Whitney U", "effect": u / (len(a) * len(b)),
                     "effect_name": "AUC (U/n1n2)", "p_value": p})
    for c in ds.categorical:
        tab = pd.crosstab(ds.X[c], y)
        chi2, p, dof, _ = stats.chi2_contingency(tab)
        v = np.sqrt(chi2 / (tab.to_numpy().sum() * (min(tab.shape) - 1)))
        rows.append({"feature": c, "test": "chi-square", "effect": v,
                     "effect_name": "Cramer's V", "p_value": p})
    df = pd.DataFrame(rows)
    df["p_holm"] = multipletests(df["p_value"], method="holm")[1]
    return df
