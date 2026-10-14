"""SHAP and LIME explanations, and quantitative checks of explanation quality.

Attributions are always reported on the *original* features: SHAP values of
one-hot columns are summed back to their source feature (valid because SHAP
values are additive), and LIME is run directly on the original feature space.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from scipy.stats import spearmanr
from sklearn.model_selection import TunedThresholdClassifierCV


def base_pipeline(model):
    """Return the fitted imblearn/sklearn pipeline under a threshold wrapper."""
    return model.estimator_ if isinstance(model, TunedThresholdClassifierCV) else model


def decision_threshold(model) -> float:
    return float(model.best_threshold_) if isinstance(model, TunedThresholdClassifierCV) else 0.5


def _group_index(pipe, features) -> np.ndarray:
    """Map each transformed column to the index of its original feature."""
    names = pipe.named_steps["pre"].get_feature_names_out()
    order = sorted(range(len(features)), key=lambda i: -len(features[i]))
    idx = np.empty(len(names), dtype=int)
    for j, n in enumerate(names):
        for i in order:
            f = features[i]
            if n == f or n.startswith(f + "_"):
                idx[j] = i
                break
        else:
            raise KeyError(n)
    return idx


def shap_values(model, X_background: pd.DataFrame, X_explain: pd.DataFrame) -> np.ndarray:
    """Interventional TreeSHAP on P(dropout), aggregated to original features."""
    pipe = base_pipeline(model)
    pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
    features = list(X_explain.columns)
    Zb, Ze = pre.transform(X_background), pre.transform(X_explain)
    explainer = shap.TreeExplainer(clf, data=Zb, feature_perturbation="interventional",
                                   model_output="probability")
    sv = explainer.shap_values(Ze, check_additivity=False)
    if isinstance(sv, list):
        sv = sv[1]
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    g = _group_index(pipe, features)
    out = np.zeros((sv.shape[0], len(features)))
    for j, i in enumerate(g):
        out[:, i] += sv[:, j]
    return out


class LimeWrapper:
    """LIME on the original features; categoricals are integer-coded for LIME only.

    Binary numeric features are also passed to LIME as categorical: LIME's default
    quartile discretiser collapses a mostly-constant 0/1 column into one bin, which
    silently gives it zero weight.
    """

    def __init__(self, model, X_train: pd.DataFrame, categorical, seed=42, num_samples=2000,
                 binary_as_categorical=True):
        self.model, self.features = model, list(X_train.columns)
        binary = [c for c in self.features if c not in categorical and X_train[c].nunique() <= 2
                  and binary_as_categorical]
        categorical = list(categorical) + binary
        self.cat_idx = [self.features.index(c) for c in categorical]
        self.levels = {c: sorted(X_train[c].unique()) for c in categorical}
        self.num_samples, self.seed = num_samples, seed
        self.train = self._encode(X_train)
        self.cat_names = {self.features.index(c): [str(v) for v in lv] for c, lv in self.levels.items()}

    def _encode(self, X):
        Z = X.copy()
        for c, lv in self.levels.items():
            Z[c] = pd.Categorical(Z[c], categories=lv).codes
        return Z.to_numpy(dtype=float)

    def _decode(self, Z):
        df = pd.DataFrame(Z, columns=self.features)
        for c, lv in self.levels.items():
            df[c] = np.asarray(lv)[np.clip(np.round(df[c]).astype(int), 0, len(lv) - 1)]
        return df

    def predict_fn(self, Z):
        return self.model.predict_proba(self._decode(Z))

    def explain(self, X: pd.DataFrame, seed: int | None = None) -> np.ndarray:
        explainer = LimeTabularExplainer(
            self.train, feature_names=self.features, categorical_features=self.cat_idx,
            categorical_names=self.cat_names, class_names=["no", "dropout"],
            discretize_continuous=True, random_state=self.seed if seed is None else seed)
        Z = self._encode(X)
        out = np.zeros((len(Z), len(self.features)))
        for r, z in enumerate(Z):
            e = explainer.explain_instance(z, self.predict_fn, labels=(1,),
                                           num_features=len(self.features),
                                           num_samples=self.num_samples)
            for i, w in e.as_map()[1]:
                out[r, i] = w
        return out


# ------------------------------------------------------------------ agreement
def topk(a: np.ndarray, k: int) -> set:
    return set(np.argsort(-np.abs(a))[:k])


def agreement(A: np.ndarray, B: np.ndarray, k: int = 5) -> pd.DataFrame:
    """Per-instance agreement of two local attribution matrices (rows = instances)."""
    rows = []
    for a, b in zip(A, B):
        ta, tb = topk(a, k), topk(b, k)
        inter = ta & tb
        rho = spearmanr(np.abs(a), np.abs(b)).statistic
        sign = np.mean([np.sign(a[i]) == np.sign(b[i]) for i in inter]) if inter else np.nan
        rows.append({"spearman_abs": rho, f"top{k}_overlap": len(inter) / k, "sign_agreement": sign})
    return pd.DataFrame(rows)


def ranking_stability(importances: list[np.ndarray], k: int = 5) -> dict:
    """Mean pairwise Spearman and top-k overlap of global importance vectors."""
    rho, ov = [], []
    for a, b in itertools.combinations(importances, 2):
        rho.append(spearmanr(a, b).statistic)
        ov.append(len(topk(a, k) & topk(b, k)) / k)
    return {"mean_pairwise_spearman": float(np.mean(rho)), "sd_pairwise_spearman": float(np.std(rho)),
            f"mean_top{k}_overlap": float(np.mean(ov)), "n_models": len(importances)}


# --------------------------------------------------------------- faithfulness
def baseline_row(X_train: pd.DataFrame, categorical) -> pd.Series:
    base = X_train.median(numeric_only=True)
    for c in categorical:
        base[c] = X_train[c].mode().iloc[0]
    return base[X_train.columns]


def deletion_curve(model, X: pd.DataFrame, attributions: np.ndarray, baseline: pd.Series,
                   kmax: int = 5) -> np.ndarray:
    """Mean drop in P(dropout) after replacing the k most risk-increasing features by baseline."""
    p0 = model.predict_proba(X)[:, 1]
    drops = np.zeros((len(X), kmax))
    for r in range(len(X)):
        order = np.argsort(-attributions[r])  # most positive (risk-increasing) first
        rows = []
        x = X.iloc[[r]].copy()
        for k in range(kmax):
            col = X.columns[order[k]]
            x[col] = baseline[col]
            rows.append(x.copy())
        p = model.predict_proba(pd.concat(rows))[:, 1]
        drops[r] = p0[r] - p
    return drops.mean(axis=0)
