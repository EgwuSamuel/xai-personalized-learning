"""E2b: how oversampling *before* the train/test split inflates results on no-signal data.

Correct protocol: SMOTE (or random oversampling) is fitted on the training
split only. Leaky protocol: the whole dataset is oversampled first and then
split, so synthetic or duplicated minority rows derived from test-set
neighbours end up in training. Repeated over 10 random splits.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from xaiedu.config import TABLES
from xaiedu.data import load_kaggle
from xaiedu.evaluate import all_metrics
from xaiedu.models import make_preprocessor

ds = load_kaggle()
Z = make_preprocessor(ds.categorical, ds.numeric, scale=True).fit_transform(ds.X)  # unsupervised; same for both
y = ds.y.to_numpy()

rows = []
for seed in range(10):
    for sampler_name, sampler in (("SMOTE", SMOTE(random_state=seed)),
                                  ("RandomOverSampler", RandomOverSampler(random_state=seed))):
        # correct: split first, oversample training data only
        Ztr, Zte, ytr, yte = train_test_split(Z, y, test_size=0.2, stratify=y, random_state=seed)
        Zr, yr = sampler.fit_resample(Ztr, ytr)
        rf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1).fit(Zr, yr)
        rows.append({"seed": seed, "sampler": sampler_name, "protocol": "oversample train only",
                     **all_metrics(yte, rf.predict_proba(Zte)[:, 1], rf.predict(Zte))})
        # leaky: oversample everything, then split
        Zall, yall = sampler.fit_resample(Z, y)
        Ztr, Zte, ytr, yte = train_test_split(Zall, yall, test_size=0.2, stratify=yall, random_state=seed)
        rf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1).fit(Ztr, ytr)
        rows.append({"seed": seed, "sampler": sampler_name, "protocol": "oversample before split (leaky)",
                     **all_metrics(yte, rf.predict_proba(Zte)[:, 1], rf.predict(Zte))})

df = pd.DataFrame(rows)
df.to_csv(TABLES / "e2b_kaggle_leakage_runs.csv", index=False)
summary = (df.groupby(["sampler", "protocol"])[["roc_auc", "accuracy", "f1_dropout", "recall_dropout"]]
           .agg(["mean", "std"]).round(3))
summary.to_csv(TABLES / "e2b_kaggle_leakage_summary.csv")
print(summary.to_string())
