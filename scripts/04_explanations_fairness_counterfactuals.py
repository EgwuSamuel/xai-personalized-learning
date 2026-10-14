"""E5 + E6: evaluate explanations, audit fairness and generate counterfactuals.

The same protocol is applied to the Kaggle data (negative control), UCI at the
end of semester 1, and OULAD at day 56. The explained model is XGBoost with
class weighting and a tuned threshold (TreeSHAP is exact for tree ensembles).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from xaiedu.config import FIGURES, SEED, TABLES
from xaiedu.counterfactuals import counterfactuals, describe
from xaiedu.data import load_kaggle, load_oulad, load_uci, sample_ids
from xaiedu.evaluate import all_metrics, fit_model
from xaiedu.explain import (LimeWrapper, agreement, base_pipeline, baseline_row, decision_threshold,
                            deletion_curve, ranking_stability, shap_values)
from xaiedu.fairness import gaps, group_metrics

N_EXPLAIN, N_BACKGROUND, N_BOOT_MODELS, N_LIME_STAB = 200, 100, 10, 30


def split(ds):
    if ds.groups is not None:  # OULAD: temporal split
        tr = ds.groups.str.startswith("2013").to_numpy()
        return ds.X[tr], ds.X[~tr], ds.y[tr], ds.y[~tr]
    return train_test_split(ds.X, ds.y, test_size=0.2, stratify=ds.y, random_state=SEED)


datasets = {"kaggle": load_kaggle(), "uci_sem1": load_uci("sem1"), "oulad_d56": load_oulad(56)}
summary, global_tables = {}, []
for key, ds in datasets.items():
    print(f"=== {key}")
    Xtr, Xte, ytr, yte = split(ds)
    model, params, thr = fit_model("XGB", ds.categorical, ds.numeric, Xtr, ytr, "weight+threshold")
    proba, pred = model.predict_proba(Xte)[:, 1], model.predict(Xte)
    perf = all_metrics(yte, proba, pred)
    print("  test", {k: round(v, 3) for k, v in perf.items()})

    feats = list(ds.X.columns)
    bg = Xtr.iloc[sample_ids(len(Xtr), N_BACKGROUND, SEED)]
    ex = Xte.iloc[sample_ids(len(Xte), N_EXPLAIN, SEED + 1)]
    S = shap_values(model, bg, ex)
    lime = LimeWrapper(model, Xtr, ds.categorical, seed=SEED)
    L = lime.explain(ex)
    np.save(TABLES / f"e5_{key}_shap.npy", S)
    np.save(TABLES / f"e5_{key}_lime.npy", L)

    agr = agreement(S, L, k=5)
    glob = pd.DataFrame({"dataset": key, "feature": feats, "mean_abs_shap": np.abs(S).mean(0),
                         "mean_abs_lime": np.abs(L).mean(0)}).sort_values("mean_abs_shap", ascending=False)
    global_tables.append(glob)

    # Global stability: refit with the tuned hyper-parameters on bootstrap resamples.
    pipe = base_pipeline(model)
    imps = []
    rng = np.random.default_rng(SEED)
    for b in range(N_BOOT_MODELS):
        i = rng.integers(0, len(Xtr), len(Xtr))
        m = clone(pipe).set_params(clf__random_state=SEED + b).fit(Xtr.iloc[i], ytr.iloc[i])
        imps.append(np.abs(shap_values(m, bg, ex)).mean(0))
    stab_global = ranking_stability(imps, k=5)

    # Local stability of LIME under different sampling seeds.
    sub = ex.iloc[:N_LIME_STAB]
    runs = [lime.explain(sub, seed=SEED + s) for s in range(5)]
    lime_stab = []
    for r in range(len(sub)):
        lime_stab.append(ranking_stability([np.abs(run[r]) for run in runs], k=5))
    lime_stab = pd.DataFrame(lime_stab).mean().to_dict()

    # Faithfulness: deletion of the top-k risk-increasing features, on students flagged at risk.
    flagged = ex[model.predict(ex) == 1]
    fi = np.where(model.predict(ex) == 1)[0]
    base = baseline_row(Xtr, ds.categorical)
    faith = {}
    if len(fi) >= 10:
        faith["shap"] = deletion_curve(model, flagged, S[fi], base).tolist()
        faith["lime"] = deletion_curve(model, flagged, L[fi], base).tolist()
        rnd = np.random.default_rng(SEED).normal(size=S[fi].shape)
        faith["random"] = deletion_curve(model, flagged, rnd, base).tolist()

    summary[key] = {
        "test_metrics": perf, "threshold": thr, "best_params": {k: str(v) for k, v in params.items()},
        "shap_lime_agreement": agr.mean().to_dict(), "shap_lime_agreement_sd": agr.std().to_dict(),
        "global_shap_stability": stab_global, "lime_local_stability": lime_stab,
        "deletion_drop_by_k": faith, "n_flagged_explained": int(len(fi)),
    }
    print("  agreement", {k: round(v, 3) for k, v in agr.mean().items()})
    print("  global stability", stab_global)
    print("  LIME local stability", {k: round(v, 3) for k, v in lime_stab.items()})
    print("  deletion", {k: np.round(v, 3).tolist() for k, v in faith.items()})

    # Figures: SHAP beeswarm on original features (one-hot SHAP values summed back).
    X_disp = ex.copy()
    for c in ds.categorical:
        X_disp[c] = pd.Categorical(X_disp[c]).codes
    expl = shap.Explanation(values=S, data=X_disp.to_numpy(dtype=float), feature_names=feats)
    plt.figure()
    shap.plots.beeswarm(expl, max_display=12, show=False)
    plt.title(f"SHAP values for P(dropout) - {key}")
    plt.tight_layout()
    plt.savefig(FIGURES / f"fig_shap_beeswarm_{key}.png", dpi=600, bbox_inches="tight")
    plt.close("all")

    # E6: fairness audit on the test set.
    if key != "kaggle":
        Xg = Xte.reset_index(drop=True)
        if key.startswith("uci"):
            attrs = {"Gender": Xg["Gender"].map({1: "male", 0: "female"}),
                     "Age band": pd.cut(Xg["Age at enrollment"], [0, 20, 25, 100],
                                        labels=["<=20", "21-25", ">25"]),
                     "Scholarship": Xg["Scholarship holder"].map({1: "holder", 0: "non-holder"})}
        else:
            attrs = {"Gender": Xg["gender"], "Disability": Xg["disability"],
                     "IMD band": Xg["imd_band"], "Age band": Xg["age_band"]}
        gm = pd.concat([group_metrics(yte, proba, pred, s.rename(a).astype(str)) for a, s in attrs.items()])
        gm.to_csv(TABLES / f"e6_{key}_group_metrics.csv", index=False)
        gaps(gm).to_csv(TABLES / f"e6_{key}_gaps.csv", index=False)
        print(gaps(gm).round(3).to_string())

    # Counterfactuals for the highest-risk UCI students.
    if key == "uci_sem1":
        act = {"Curricular units 1st sem (approved)": "increase",
               "Curricular units 1st sem (evaluations)": "increase",
               "Curricular units 1st sem (grade)": "increase",
               "Tuition fees up to date": [1], "Debtor": [0], "Scholarship holder": [1]}

        def feasible(c):
            approved = c["Curricular units 1st sem (approved)"]
            return ((approved <= c["Curricular units 1st sem (enrolled)"])
                    & (approved <= c["Curricular units 1st sem (evaluations)"])
                    & ((c["Curricular units 1st sem (grade)"] == 0) | (approved > 0)))  # a grade needs an approved unit

        risky = Xte.assign(p=proba).sort_values("p", ascending=False).head(20)
        cf_rows, n_valid = [], 0
        for sid, (idx, row) in enumerate(risky.iterrows()):
            x = row.drop("p")
            cf = counterfactuals(model, x, Xtr, act, threshold=decision_threshold(model),
                                 seed=SEED + sid, feasible=feasible)
            n_valid += int(len(cf) > 0)
            for j, text in enumerate(describe(x, cf, list(act))):
                cf_rows.append({"student": sid, "p_dropout": row["p"], "cf": j,
                                "n_changed": int(cf.iloc[j]["n_changed"]), "changes": text})
        cfdf = pd.DataFrame(cf_rows)
        cfdf.to_csv(TABLES / "e5_uci_counterfactuals.csv", index=False)
        summary[key]["counterfactuals"] = {
            "students": len(risky), "with_valid_cf": n_valid,
            "mean_features_changed": float(cfdf["n_changed"].mean()) if len(cfdf) else None}
        print(cfdf.head(9).to_string())

pd.concat(global_tables).to_csv(TABLES / "e5_global_importance.csv", index=False)
(TABLES / "e5_summary.json").write_text(json.dumps(summary, indent=2, default=float))
