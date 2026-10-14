"""E3: dropout prediction on UCI #697 at three decision points.

Main comparison: seven models, each with class weighting and a decision
threshold tuned on inner folds, under 5x2 repeated stratified CV with nested
random search. Ablation: four imbalance strategies for LR, RF and XGB at the
end-of-semester-1 decision point.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from xaiedu.data import UCI_SCENARIOS, load_uci
from xaiedu.evaluate import cross_validate, summarise
from xaiedu.config import TABLES
from xaiedu.models import MODELS, STRATEGIES
from xaiedu.stats import compare_to_best, friedman_nemenyi

all_cv = []
for scenario in UCI_SCENARIOS:
    ds = load_uci(scenario)
    print(f"UCI scenario={scenario}: X={ds.X.shape}, dropout prevalence={ds.y.mean():.3f}")
    cv = cross_validate(ds, MODELS, ["weight+threshold"])
    all_cv.append(cv)
    pd.concat(all_cv).to_csv(TABLES / "e3_uci_cv_folds.csv", index=False)

cv = pd.concat(all_cv)
summary = summarise(cv)
summary.to_csv(TABLES / "e3_uci_cv_summary.csv", index=False)
print(summary[["dataset", "model", "roc_auc_mean", "pr_auc_mean", "balanced_accuracy_mean",
               "recall_dropout_mean", "f1_dropout_mean", "mcc_mean"]].round(3).to_string())

tests, ranks_all, ttests = [], [], []
for name, d in cv.groupby("dataset"):
    for metric in ("pr_auc", "roc_auc", "mcc"):
        res, nemenyi, ranks = friedman_nemenyi(d, metric)
        res["dataset"] = name
        tests.append(res)
        nemenyi.to_csv(TABLES / f"e3_nemenyi_{name}_{metric}.csv")
        ranks_all.append(ranks.rename(f"{name}|{metric}"))
        t = compare_to_best(d, metric)
        t["dataset"] = name
        ttests.append(t)
pd.DataFrame(tests).to_csv(TABLES / "e3_friedman.csv", index=False)
pd.concat(ranks_all, axis=1).to_csv(TABLES / "e3_mean_ranks.csv")
pd.concat(ttests).to_csv(TABLES / "e3_corrected_ttests.csv", index=False)
print(pd.DataFrame(tests).round(4).to_string())

# Imbalance-strategy ablation (end of semester 1)
ds = load_uci("sem1")
abl = cross_validate(ds, ["LR", "RF", "XGB"], STRATEGIES)
abl.to_csv(TABLES / "e3_uci_imbalance_ablation_folds.csv", index=False)
summarise(abl).to_csv(TABLES / "e3_uci_imbalance_ablation_summary.csv", index=False)
print(summarise(abl)[["model", "strategy", "pr_auc_mean", "recall_dropout_mean",
                      "precision_dropout_mean", "balanced_accuracy_mean", "f1_dropout_mean"]]
      .round(3).to_string())
(TABLES / "e3_done.json").write_text(json.dumps({"ok": True}))
