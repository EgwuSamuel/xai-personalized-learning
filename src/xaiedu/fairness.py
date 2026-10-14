"""Group-wise error rates of the early-warning decisions (a fairness audit)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def group_metrics(y, proba, pred, groups: pd.Series, min_n: int = 30) -> pd.DataFrame:
    y, proba, pred = map(np.asarray, (y, proba, pred))
    rows = []
    for g in sorted(pd.unique(groups.astype(str))):
        m = groups.astype(str).to_numpy() == g
        if m.sum() < min_n:
            continue
        yt, pt, pr = y[m], proba[m], pred[m]
        pos, neg = yt == 1, yt == 0
        rows.append({
            "attribute": groups.name, "group": g, "n": int(m.sum()), "base_rate": yt.mean(),
            "selection_rate": pr.mean(),
            "tpr": pr[pos].mean() if pos.any() else np.nan,
            "fpr": pr[neg].mean() if neg.any() else np.nan,
            "precision": yt[pr == 1].mean() if (pr == 1).any() else np.nan,
            "auc": roc_auc_score(yt, pt) if 0 < yt.mean() < 1 else np.nan,
        })
    return pd.DataFrame(rows)


def gaps(gm: pd.DataFrame) -> pd.DataFrame:
    """Max-minus-min differences across groups (0 = parity)."""
    out = []
    for attr, d in gm.groupby("attribute"):
        out.append({"attribute": attr, "n_groups": len(d),
                    "demographic_parity_diff": d["selection_rate"].max() - d["selection_rate"].min(),
                    "equal_opportunity_diff": d["tpr"].max() - d["tpr"].min(),
                    "fpr_diff": d["fpr"].max() - d["fpr"].min(),
                    "auc_diff": d["auc"].max() - d["auc"].min()})
    return pd.DataFrame(out)
