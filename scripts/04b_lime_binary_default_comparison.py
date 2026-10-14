"""E5b: effect of LIME's default handling of binary features (UCI, end of semester 1).

Re-creates the setting of script 04 exactly (same split, model, students and SHAP values) and
compares LIME with binary features passed as categorical (used throughout the paper) against
LIME's default, which discretises them as continuous variables.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from xaiedu.config import SEED, TABLES
from xaiedu.data import load_uci, sample_ids
from xaiedu.evaluate import fit_model
from xaiedu.explain import LimeWrapper, agreement, shap_values

ds = load_uci("sem1")
Xtr, Xte, ytr, yte = train_test_split(ds.X, ds.y, test_size=0.2, stratify=ds.y, random_state=SEED)
model, _, _ = fit_model("XGB", ds.categorical, ds.numeric, Xtr, ytr, "weight+threshold")
bg = Xtr.iloc[sample_ids(len(Xtr), 100, SEED)]
ex = Xte.iloc[sample_ids(len(Xte), 200, SEED + 1)]
S = shap_values(model, bg, ex)
out, imp = {}, {"feature": list(ds.X.columns), "mean_abs_shap": np.abs(S).mean(0)}
for name, flag in (("binary_as_categorical", True), ("lime_default", False)):
    L = LimeWrapper(model, Xtr, ds.categorical, seed=SEED, binary_as_categorical=flag).explain(ex)
    out[name] = agreement(S, L, k=5).mean().to_dict()
    imp[f"mean_abs_lime_{name}"] = np.abs(L).mean(0)
    print(name, {k: round(v, 3) for k, v in out[name].items()})
(TABLES / "e5b_uci_lime_binary_agreement.json").write_text(json.dumps(out, indent=2))
pd.DataFrame(imp).sort_values("mean_abs_shap", ascending=False).to_csv(
    TABLES / "e5b_uci_lime_binary_importance.csv", index=False)
print(pd.DataFrame(imp).sort_values("mean_abs_shap", ascending=False).head(8).round(4).to_string())
