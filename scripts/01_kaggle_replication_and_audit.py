"""E1 + E2: replicate the dissertation's experiment on the Kaggle data, then audit its signal.

E1 reproduces the original pipeline (one-hot encoding, 80/20 split,
RandomForest with 100 trees) with a fixed seed.
E2 asks whether *any* model can beat chance on this data: a label-permutation
test, uniformity and univariate association tests, and the full tuned model
zoo with imbalance handling under 5-fold CV.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from xaiedu.config import SEED, TABLES
from xaiedu.data import load_kaggle
from xaiedu.evaluate import all_metrics, cross_validate, summarise
from xaiedu.models import make_preprocessor
from xaiedu.signal_tests import label_permutation_test, target_association, uniformity_tests

ds = load_kaggle()
print(f"Kaggle v1: {ds.X.shape}, dropout prevalence = {ds.y.mean():.4f}")

# ---------------- E1: faithful replication of the original pipeline ----------------
Xtr, Xte, ytr, yte = train_test_split(ds.X, ds.y, test_size=0.2, random_state=SEED)
rf = Pipeline([("pre", make_preprocessor(ds.categorical, ds.numeric, scale=False)),
               ("clf", RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1))])
rf.fit(Xtr, ytr)
proba, pred = rf.predict_proba(Xte)[:, 1], rf.predict(Xte)
e1 = all_metrics(yte, proba, pred)
e1["confusion_matrix"] = confusion_matrix(yte, pred).tolist()
e1["test_support"] = {"no_dropout": int((yte == 0).sum()), "dropout": int((yte == 1).sum())}
print(classification_report(yte, pred, digits=3, zero_division=0))
print("E1", {k: v for k, v in e1.items() if k != "confusion_matrix"}, e1["confusion_matrix"])
(TABLES / "e1_kaggle_replication.json").write_text(json.dumps(e1, indent=2))

# Split-to-split variability of the same pipeline (explains differing supports in the thesis)
var = []
for s in range(20):
    a, b, c, d = train_test_split(ds.X, ds.y, test_size=0.2, random_state=s)
    m = rf.set_params(clf__random_state=s).fit(a, c)
    var.append({"split_seed": s, "dropout_in_test": int(d.sum()),
                **all_metrics(d, m.predict_proba(b)[:, 1], m.predict(b))})
pd.DataFrame(var).to_csv(TABLES / "e1_kaggle_split_variability.csv", index=False)

# ---------------- E2: signal audit ----------------
perm = label_permutation_test(ds, n_permutations=199, seed=SEED)
np.save(TABLES / "e2_kaggle_permutation_scores.npy", perm.pop("perm_scores"))
print("E2 permutation test", perm)
(TABLES / "e2_kaggle_permutation_test.json").write_text(json.dumps(perm, indent=2))

uni = uniformity_tests(ds)
uni.to_csv(TABLES / "e2_kaggle_uniformity.csv", index=False)
print(uni.round(4).to_string())
assoc = target_association(ds)
assoc.to_csv(TABLES / "e2_kaggle_target_association.csv", index=False)
print(assoc.round(4).to_string())

# Best-effort modelling: every model x imbalance strategy, tuned, 5-fold CV (1 repeat).
cv = cross_validate(ds, ["LR", "RF", "XGB", "LGBM", "EBM"], ["weight", "smote", "weight+threshold"],
                    folds=5, repeats=1)
cv.to_csv(TABLES / "e2_kaggle_cv_folds.csv", index=False)
summarise(cv).to_csv(TABLES / "e2_kaggle_cv_summary.csv", index=False)
print(summarise(cv)[["model", "strategy", "roc_auc_mean", "roc_auc_sd", "pr_auc_mean",
                     "balanced_accuracy_mean"]].round(3).to_string())
