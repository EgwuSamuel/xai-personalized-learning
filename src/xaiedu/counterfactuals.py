"""Actionable counterfactual explanations for students flagged as at risk.

A counterfactual answers "what is the smallest change to the things a student
or institution can act on that would move this student below the risk
threshold?". Only features listed as actionable may change, numeric changes
must stay within the observed training range, and features that can only
increase over time (e.g. approved units) are not allowed to decrease.

The search is a seeded random local search in the spirit of DiCE's
model-agnostic "random" method (Mothilal et al., 2020): candidates are drawn
by perturbing a random subset of actionable features, valid candidates are
kept, and the sparsest/closest ones are returned.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def counterfactuals(model, x: pd.Series, X_train: pd.DataFrame, actionable: dict, threshold: float,
                    n_candidates: int = 4000, n_return: int = 3, max_changes: int = 3,
                    seed: int = 42, feasible=None) -> pd.DataFrame:
    """Return up to ``n_return`` diverse counterfactuals for one student.

    ``actionable`` maps feature -> "increase", "decrease", "any" or a list of
    allowed categorical values. ``feasible`` is an optional function that takes
    the candidate frame and returns a boolean mask of plausible rows.
    """
    rng = np.random.default_rng(seed)
    feats = list(actionable)
    span = {f: (X_train[f].min(), X_train[f].max()) for f in feats
            if not isinstance(actionable[f], list)}
    mad = {f: max(float((X_train[f] - X_train[f].median()).abs().median()), 1e-6) for f in span}
    cand = pd.DataFrame([x] * n_candidates).reset_index(drop=True)
    n_change = rng.integers(1, max_changes + 1, n_candidates)
    for r in range(n_candidates):
        for f in rng.choice(feats, size=min(n_change[r], len(feats)), replace=False):
            rule, cur = actionable[f], x[f]
            if isinstance(rule, list):
                cand.at[r, f] = rng.choice(rule)
                continue
            lo, hi = span[f]
            if rule == "increase":
                lo = cur
            elif rule == "decrease":
                hi = cur
            if hi <= lo:
                continue
            v = rng.uniform(lo, hi)
            cand.at[r, f] = np.round(v) if float(X_train[f].dropna().mod(1).eq(0).all()) else v
    cand = cand.astype(X_train.dtypes.to_dict())
    if feasible is not None:  # domain constraints, e.g. approved units <= enrolled units
        cand = cand[feasible(cand)].reset_index(drop=True)
        if cand.empty:
            return cand
    p = model.predict_proba(cand[X_train.columns])[:, 1]
    valid = cand[p < threshold].copy()
    valid["p_dropout"] = p[p < threshold]
    if valid.empty:
        return valid
    changed = np.zeros(len(valid), dtype=int)
    dist = np.zeros(len(valid))
    for f in feats:
        diff = valid[f].to_numpy() != x[f]
        changed += diff
        if f in mad:
            dist += np.abs(valid[f].to_numpy(dtype=float) - float(x[f])) / mad[f]
        else:
            dist += diff
    valid["n_changed"], valid["distance"] = changed, dist
    valid = valid.sort_values(["n_changed", "distance"]).drop_duplicates(subset=feats)
    # Greedy diversity: do not return two counterfactuals that change the same feature set.
    picked, seen = [], set()
    for i, row in valid.iterrows():
        key = tuple(f for f in feats if row[f] != x[f])
        if key in seen:
            continue
        seen.add(key)
        picked.append(i)
        if len(picked) == n_return:
            break
    return valid.loc[picked]


def _fmt(v) -> str:
    if isinstance(v, (int, float, np.number)):
        return f"{float(v):.1f}" if not float(v).is_integer() else f"{int(v)}"
    return str(v)


def describe(x: pd.Series, cf: pd.DataFrame, feats) -> list[str]:
    """Human-readable change lists, e.g. 'approved units 1st sem: 2 -> 5'."""
    out = []
    for _, row in cf.iterrows():
        parts = [f"{f}: {_fmt(x[f])} -> {_fmt(row[f])}" for f in feats if row[f] != x[f]]
        out.append("; ".join(parts) + f"  (P(dropout) = {row['p_dropout']:.2f})")
    return out
