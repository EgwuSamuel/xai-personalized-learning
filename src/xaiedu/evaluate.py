"""Nested, repeated cross-validation and the metrics reported in the paper."""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss,
                             f1_score, matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import (RandomizedSearchCV, RepeatedStratifiedKFold,
                                     StratifiedKFold, TunedThresholdClassifierCV)

from .config import CV_FOLDS, CV_REPEATS, INNER_FOLDS, N_ITER_SEARCH, SEED
from .models import PARAM_SPACE, make_pipeline

warnings.filterwarnings("ignore", category=UserWarning)


def threshold_metrics(y, pred) -> dict:
    return {
        "f1_dropout": f1_score(y, pred, zero_division=0),
        "recall_dropout": recall_score(y, pred, zero_division=0),
        "precision_dropout": precision_score(y, pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y, pred),
        "accuracy": float(np.mean(np.asarray(y) == np.asarray(pred))),
    }


def all_metrics(y, proba, pred) -> dict:
    out = {"roc_auc": roc_auc_score(y, proba), "pr_auc": average_precision_score(y, proba),
           "brier": brier_score_loss(y, proba), "prevalence": float(np.mean(y))}
    out.update(threshold_metrics(y, pred))
    return out


def fit_model(name, ds_cat, ds_num, X, y, strategy, seed=SEED, tune=True):
    """Fit one model on (X, y): inner random search, then optional threshold tuning."""
    pipe = make_pipeline(name, ds_cat, ds_num, strategy, y=y, seed=seed)
    space = PARAM_SPACE[name]
    if tune and space:
        search = RandomizedSearchCV(pipe, space, n_iter=N_ITER_SEARCH, scoring="average_precision",
                                    cv=StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=seed),
                                    random_state=seed, n_jobs=-1, refit=True)
        search.fit(X, y)
        model, params = search.best_estimator_, search.best_params_
    else:
        model, params = pipe.fit(X, y), {}
    if strategy == "weight+threshold":
        tuned = TunedThresholdClassifierCV(
            clone(model), scoring="f1",
            cv=StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=seed),
            random_state=seed, n_jobs=-1)
        tuned.fit(X, y)
        return tuned, params, float(tuned.best_threshold_)
    return model, params, 0.5


def predict_proba1(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def cross_validate(ds, models, strategies, folds=CV_FOLDS, repeats=CV_REPEATS, seed=SEED,
                   tune=True, log=print) -> pd.DataFrame:
    """Outer repeated stratified CV; returns one row per (model, strategy, fold)."""
    cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    splits = list(cv.split(ds.X, ds.y))
    rows = []
    for name in models:
        for strategy in strategies:
            t0 = time.time()
            for k, (tr, te) in enumerate(splits):
                Xtr, Xte = ds.X.iloc[tr], ds.X.iloc[te]
                ytr, yte = ds.y.iloc[tr], ds.y.iloc[te]
                model, params, thr = fit_model(name, ds.categorical, ds.numeric, Xtr, ytr,
                                               strategy, seed=seed + k, tune=tune)
                proba = predict_proba1(model, Xte)
                pred = model.predict(Xte)
                row = {"dataset": ds.name, "model": name, "strategy": strategy, "fold": k,
                       "threshold": thr, "n_train": len(tr), "n_test": len(te)}
                row.update(all_metrics(yte, proba, pred))
                rows.append(row)
            log(f"  {ds.name} {name:5s} {strategy:17s} "
                f"AUC={np.mean([r['roc_auc'] for r in rows[-len(splits):]]):.3f} "
                f"PR-AUC={np.mean([r['pr_auc'] for r in rows[-len(splits):]]):.3f} "
                f"({time.time() - t0:.0f}s)")
    return pd.DataFrame(rows)


def summarise(cv_df: pd.DataFrame, metrics=("roc_auc", "pr_auc", "balanced_accuracy",
                                           "f1_dropout", "recall_dropout", "precision_dropout",
                                           "mcc", "brier")) -> pd.DataFrame:
    """Mean ± SD over folds, plus a 95% percentile interval of the fold scores."""
    g = cv_df.groupby(["dataset", "model", "strategy"])
    out = []
    for key, d in g:
        row = dict(zip(["dataset", "model", "strategy"], key))
        for m in metrics:
            v = d[m].to_numpy()
            row[f"{m}_mean"] = v.mean()
            row[f"{m}_sd"] = v.std(ddof=1)
        out.append(row)
    return pd.DataFrame(out)


def bootstrap_ci(y, proba, pred, n_boot=1000, seed=SEED) -> pd.DataFrame:
    """Percentile bootstrap CIs on a fixed test set (used for the OULAD temporal split)."""
    rng = np.random.default_rng(seed)
    y, proba, pred = map(np.asarray, (y, proba, pred))
    stats = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        stats.append(all_metrics(y[i], proba[i], pred[i]))
    s = pd.DataFrame(stats)
    point = all_metrics(y, proba, pred)
    return pd.DataFrame({"estimate": point, "ci_low": s.quantile(0.025), "ci_high": s.quantile(0.975)})
