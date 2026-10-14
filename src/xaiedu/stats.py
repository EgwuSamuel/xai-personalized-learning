"""Statistical comparison of models evaluated on the same CV folds."""
from __future__ import annotations

import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy import stats


def friedman_nemenyi(cv_df: pd.DataFrame, metric: str) -> tuple[dict, pd.DataFrame, pd.Series]:
    """Friedman test across models (blocks = folds) and Nemenyi post-hoc p-values."""
    wide = cv_df.pivot_table(index="fold", columns="model", values=metric)
    stat, p = stats.friedmanchisquare(*[wide[c] for c in wide.columns])
    ranks = wide.rank(axis=1, ascending=False).mean().sort_values()
    nemenyi = sp.posthoc_nemenyi_friedman(wide.to_numpy())
    nemenyi.index = nemenyi.columns = wide.columns
    return {"metric": metric, "friedman_chi2": float(stat), "p_value": float(p),
            "n_folds": len(wide), "n_models": wide.shape[1]}, nemenyi, ranks


def corrected_resampled_ttest(a, b, n_train: int, n_test: int) -> tuple[float, float]:
    """Nadeau & Bengio (2003) corrected resampled t-test for paired CV scores."""
    d = np.asarray(a) - np.asarray(b)
    j = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = d.mean() / np.sqrt((1 / j + n_test / n_train) * var)
    p = 2 * stats.t.sf(abs(t), df=j - 1)
    return float(t), float(p)


def compare_to_best(cv_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    wide = cv_df.pivot_table(index="fold", columns="model", values=metric)
    best = wide.mean().idxmax()
    n_train, n_test = int(cv_df["n_train"].iloc[0]), int(cv_df["n_test"].iloc[0])
    rows = []
    for m in wide.columns:
        t, p = corrected_resampled_ttest(wide[best], wide[m], n_train, n_test) if m != best else (np.nan, np.nan)
        rows.append({"metric": metric, "best": best, "model": m, "mean": wide[m].mean(),
                     "diff_vs_best": wide[m].mean() - wide[best].mean(), "t": t, "p_value": p})
    return pd.DataFrame(rows)
