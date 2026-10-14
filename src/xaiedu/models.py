"""Model zoo, preprocessing and imbalance-handling strategies.

All preprocessing lives inside the pipeline so that it is fitted on training
folds only (no information from test folds leaks into scaling or encoding).
"""
from __future__ import annotations

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

try:  # glass-box baseline; optional dependency
    from interpret.glassbox import ExplainableBoostingClassifier
except ImportError:  # pragma: no cover
    ExplainableBoostingClassifier = None

MODELS = ["LR", "DT", "RF", "XGB", "LGBM", "MLP", "EBM"]
STRATEGIES = ["none", "weight", "smote", "weight+threshold"]


def make_preprocessor(categorical, numeric, scale=True) -> ColumnTransformer:
    num = StandardScaler() if scale else "passthrough"
    return ColumnTransformer(
        [("num", num, list(numeric)),
         ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20,
                               sparse_output=False), list(categorical))],
        verbose_feature_names_out=False,
    )


def _estimator(name: str, weighted: bool, pos_weight: float, seed: int):
    cw = "balanced" if weighted else None
    if name == "LR":
        return LogisticRegression(max_iter=3000, class_weight=cw, random_state=seed)
    if name == "DT":
        return DecisionTreeClassifier(class_weight=cw, random_state=seed)
    if name == "RF":
        return RandomForestClassifier(n_estimators=300, class_weight=cw, n_jobs=1, random_state=seed)
    if name == "XGB":
        return XGBClassifier(n_estimators=300, tree_method="hist", eval_metric="logloss",
                             scale_pos_weight=pos_weight if weighted else 1.0,
                             n_jobs=1, random_state=seed, verbosity=0)
    if name == "LGBM":
        return LGBMClassifier(n_estimators=300, class_weight=cw, n_jobs=1,
                              random_state=seed, verbose=-1)
    if name == "MLP":  # no class-weight support in scikit-learn's MLP
        return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, early_stopping=True,
                             random_state=seed)
    if name == "EBM":
        if ExplainableBoostingClassifier is None:
            raise ImportError("interpret is not installed")
        # 4 outer bags (default 14) keeps the nested CV tractable on a 4-core machine
        return ExplainableBoostingClassifier(outer_bags=4, random_state=seed, n_jobs=1)
    raise ValueError(name)


PARAM_SPACE = {
    "LR": {"clf__C": loguniform(1e-3, 1e2)},
    "DT": {"clf__max_depth": randint(2, 12), "clf__min_samples_leaf": randint(5, 100)},
    "RF": {"clf__max_depth": [None, 6, 10, 16], "clf__min_samples_leaf": randint(1, 20),
           "clf__max_features": ["sqrt", 0.3, 0.5]},
    "XGB": {"clf__max_depth": randint(2, 8), "clf__learning_rate": loguniform(0.01, 0.3),
            "clf__subsample": uniform(0.6, 0.4), "clf__colsample_bytree": uniform(0.5, 0.5),
            "clf__min_child_weight": randint(1, 10)},
    "LGBM": {"clf__num_leaves": randint(8, 64), "clf__learning_rate": loguniform(0.01, 0.3),
             "clf__min_child_samples": randint(10, 100), "clf__subsample": uniform(0.6, 0.4),
             "clf__subsample_freq": [1], "clf__colsample_bytree": uniform(0.5, 0.5)},
    "MLP": {"clf__alpha": loguniform(1e-5, 1e-1),
            "clf__hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64)]},
    "EBM": {},  # default EBM settings; it is not tuned (costly, robust defaults)
}


def make_pipeline(name: str, categorical, numeric, strategy: str, y=None, seed: int = 42):
    """Build ``preprocess -> [SMOTE] -> classifier`` for one model and strategy.

    ``weight+threshold`` returns the weighted pipeline; threshold tuning is
    applied by the evaluation code on top of it.
    """
    weighted = strategy in ("weight", "weight+threshold") and name not in ("MLP", "EBM")
    pos_weight = 1.0
    if y is not None:
        y = np.asarray(y)
        pos_weight = float((y == 0).sum() / max((y == 1).sum(), 1))
    steps = [("pre", make_preprocessor(categorical, numeric, scale=name in ("LR", "MLP")))]
    if strategy == "smote":
        steps.append(("smote", SMOTE(random_state=seed)))
    steps.append(("clf", _estimator(name, weighted, pos_weight, seed)))
    return Pipeline(steps)
