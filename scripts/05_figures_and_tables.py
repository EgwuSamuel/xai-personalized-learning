"""Build the paper's figures and formatted tables from the result files of scripts 01-04."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from xaiedu.config import FIGURES, TABLES

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3"]
NICE = {"kaggle": "Kaggle (v1)", "uci_sem1": "UCI, end of sem. 1", "oulad_d56": "OULAD, day 56",
        "uci_enrolment": "At enrolment", "uci_sem1_s": "End of sem. 1", "uci_sem2": "End of sem. 2"}


def pm(m, s, d=3):
    return f"{m:.{d}f} ± {s:.{d}f}"


# ------------------------------------------------------------- study design
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(7.2, 3.3))
ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")


def box(x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=fc, ec="#555555", lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7)


def arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8))


box(0.1, 4.3, 2.6, 1.3, "Kaggle v1 (n = 10,000)\nno documentation\n→ negative control", "#F4D6D6")
box(0.1, 2.4, 2.6, 1.3, "UCI #697 (n = 4,424)\nenrolment / sem. 1 / sem. 2", "#D6E4F4")
box(0.1, 0.5, 2.6, 1.3, "OULAD (32,593 registrations)\ncutoffs day 14 … 140", "#D6F0DC")
box(3.4, 4.3, 2.8, 1.3, "E1 replication\nE2 signal audit\nE2b resampling order", "#FFFFFF")
box(3.4, 2.4, 2.8, 1.3, "E3 nested 5×2 CV\n7 models, imbalance ablation\nFriedman / Nemenyi", "#FFFFFF")
box(3.4, 0.5, 2.8, 1.3, "E4 temporal validation\ntrain 2013 → test 2014\nbootstrap CIs", "#FFFFFF")
box(6.9, 2.6, 3.0, 2.9, "E5 explanation quality\nSHAP vs LIME agreement\nstability, deletion test\n\nCounterfactuals (UCI)\n\nE6 fairness audit", "#FFF4D6")
for y in (4.95, 3.05, 1.15):
    arrow(2.75, y, 3.35, y)
for y in (4.95, 3.05, 1.15):
    arrow(6.25, y, 6.85, min(max(y, 2.8), 5.3))
fig.savefig(FIGURES / "fig1_study_design.png")
plt.close(fig)

# ---------------------------------------------------------------- Kaggle audit
perm = json.loads((TABLES / "e2_kaggle_permutation_test.json").read_text())
null = np.load(TABLES / "e2_kaggle_permutation_scores.npy")
e1 = json.loads((TABLES / "e1_kaggle_replication.json").read_text())
leak = pd.read_csv(TABLES / "e2b_kaggle_leakage_runs.csv")
kcv = pd.read_csv(TABLES / "e2_kaggle_cv_summary.csv")

fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
ax[0].hist(null, bins=25, color="#BBBBBB", edgecolor="white", label="permuted labels")
ax[0].axvline(perm["observed_auc"], color="#C44E52", lw=2, label=f"observed ({perm['observed_auc']:.3f})")
ax[0].set(xlabel="5-fold CV ROC-AUC", ylabel="count", title=f"(a) Label-permutation test, p = {perm['p_value']:.2f}")
ax[0].legend(frameon=False, fontsize=7)
g = leak.groupby(["sampler", "protocol"])["roc_auc"].agg(["mean", "std"]).reset_index()
labels = [f"{s}\n{'train only' if 'train' in p else 'before split'}" for s, p in zip(g["sampler"], g["protocol"])]
colors = ["#C44E52" if "leaky" in p else "#4C72B0" for p in g["protocol"]]
ax[1].bar(range(len(g)), g["mean"], yerr=g["std"], color=colors, capsize=3)
ax[1].axhline(0.5, ls="--", color="grey", lw=1)
ax[1].set_xticks(range(len(g)), labels, fontsize=7)
ax[1].set(ylabel="test ROC-AUC", ylim=(0, 1.05), title="(b) Oversampling before vs after splitting")
fig.tight_layout()
fig.savefig(FIGURES / "fig2_kaggle_audit.png")
plt.close(fig)

uni = pd.read_csv(TABLES / "e2_kaggle_uniformity.csv")
assoc = pd.read_csv(TABLES / "e2_kaggle_target_association.csv")
t_k = kcv[["model", "strategy", "roc_auc_mean", "roc_auc_sd", "pr_auc_mean", "balanced_accuracy_mean",
           "recall_dropout_mean"]].copy()
t_k.to_csv(TABLES / "paper_table_kaggle_models.csv", index=False)

# ------------------------------------------------------------------------ UCI
ucv = pd.read_csv(TABLES / "e3_uci_cv_summary.csv")
order = ["LR", "DT", "RF", "XGB", "LGBM", "MLP", "EBM"]
scen = ["uci_enrolment", "uci_sem1", "uci_sem2"]
rows = []
for s in scen:
    for m in order:
        r = ucv[(ucv.dataset == s) & (ucv.model == m)]
        if r.empty:
            continue
        r = r.iloc[0]
        rows.append({"Decision point": {"uci_enrolment": "Enrolment", "uci_sem1": "End of sem. 1",
                                        "uci_sem2": "End of sem. 2"}[s], "Model": m,
                     "ROC-AUC": pm(r.roc_auc_mean, r.roc_auc_sd), "PR-AUC": pm(r.pr_auc_mean, r.pr_auc_sd),
                     "Balanced acc.": pm(r.balanced_accuracy_mean, r.balanced_accuracy_sd),
                     "Recall (dropout)": pm(r.recall_dropout_mean, r.recall_dropout_sd),
                     "Precision (dropout)": pm(r.precision_dropout_mean, r.precision_dropout_sd),
                     "MCC": pm(r.mcc_mean, r.mcc_sd), "Brier": pm(r.brier_mean, r.brier_sd)})
pd.DataFrame(rows).to_csv(TABLES / "paper_table_uci_main.csv", index=False)

fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=False)
for k, metric in enumerate(["roc_auc", "pr_auc"]):
    for j, s in enumerate(scen):
        d = ucv[ucv.dataset == s].set_index("model").reindex(order)
        x = np.arange(len(order)) + (j - 1) * 0.25
        ax[k].errorbar(x, d[f"{metric}_mean"], yerr=d[f"{metric}_sd"], fmt="o", ms=4, capsize=2,
                       color=PALETTE[j], label=["Enrolment", "End of sem. 1", "End of sem. 2"][j])
    ax[k].set_xticks(range(len(order)), order)
    ax[k].set(ylabel={"roc_auc": "ROC-AUC", "pr_auc": "PR-AUC"}[metric],
              title=f"({'ab'[k]}) {'ROC-AUC' if metric == 'roc_auc' else 'PR-AUC'} (mean ± SD, 5×2 CV)")
ax[0].legend(frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(FIGURES / "fig3_uci_models.png")
plt.close(fig)

abl = pd.read_csv(TABLES / "e3_uci_imbalance_ablation_summary.csv")
abl_t = abl[["model", "strategy", "pr_auc_mean", "pr_auc_sd", "recall_dropout_mean", "recall_dropout_sd",
             "precision_dropout_mean", "precision_dropout_sd", "f1_dropout_mean", "f1_dropout_sd",
             "balanced_accuracy_mean", "balanced_accuracy_sd"]]
abl_t.to_csv(TABLES / "paper_table_uci_ablation.csv", index=False)

# ---------------------------------------------------------------------- OULAD
ou = pd.read_csv(TABLES / "e4_oulad_temporal_test.csv")
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
for k, metric in enumerate(["roc_auc", "recall_dropout"]):
    for j, m in enumerate(["LR", "RF", "XGB", "LGBM"]):
        d = ou[(ou.model == m) & (ou.metric == metric)].sort_values("cutoff_day")
        ax[k].plot(d.cutoff_day, d.estimate, "-o", ms=3, color=PALETTE[j], label=m)
        ax[k].fill_between(d.cutoff_day, d.ci_low, d.ci_high, color=PALETTE[j], alpha=0.15)
    ax[k].set(xlabel="cutoff (days since module start)",
              ylabel={"roc_auc": "ROC-AUC", "recall_dropout": "recall (at risk)"}[metric],
              title=f"({'ab'[k]}) {'ROC-AUC' if metric == 'roc_auc' else 'Recall at tuned threshold'} on 2014 presentations")
ax[0].legend(frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(FIGURES / "fig4_oulad_early_warning.png")
plt.close(fig)
wide = ou[ou.metric.isin(["roc_auc", "pr_auc", "recall_dropout", "precision_dropout", "balanced_accuracy"])]
wide = wide.assign(val=[f"{e:.3f} [{lo:.3f}, {hi:.3f}]" for e, lo, hi in zip(wide.estimate, wide.ci_low, wide.ci_high)])
wide.pivot_table(index=["cutoff_day", "model"], columns="metric", values="val", aggfunc="first") \
    .reset_index().to_csv(TABLES / "paper_table_oulad.csv", index=False)

# --------------------------------------------------------- explanation quality
summ = json.loads((TABLES / "e5_summary.json").read_text())
keys = [k for k in ["kaggle", "uci_sem1", "oulad_d56"] if k in summ]
xrows = []
for k in keys:
    s = summ[k]
    xrows.append({"Dataset": NICE[k], "Test ROC-AUC": s["test_metrics"]["roc_auc"],
                  "SHAP-LIME Spearman (|attr.|)": s["shap_lime_agreement"]["spearman_abs"],
                  "SHAP-LIME top-5 overlap": s["shap_lime_agreement"]["top5_overlap"],
                  "SHAP-LIME sign agreement": s["shap_lime_agreement"]["sign_agreement"],
                  "Global SHAP stability (Spearman)": s["global_shap_stability"]["mean_pairwise_spearman"],
                  "Global SHAP top-5 stability": s["global_shap_stability"]["mean_top5_overlap"],
                  "LIME local stability (Spearman)": s["lime_local_stability"]["mean_pairwise_spearman"],
                  "LIME local top-5 stability": s["lime_local_stability"]["mean_top5_overlap"]})
xt = pd.DataFrame(xrows)
xt.round(3).to_csv(TABLES / "paper_table_xai_quality.csv", index=False)

fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
cols = ["SHAP-LIME top-5 overlap", "Global SHAP top-5 stability", "LIME local top-5 stability"]
w = 0.25
for j, c in enumerate(cols):
    ax[0].bar(np.arange(len(xt)) + (j - 1) * w, xt[c], w, color=PALETTE[j], label=c)
ax[0].set_xticks(range(len(xt)), xt["Dataset"], fontsize=7)
ax[0].set(ylim=(0, 1.05), ylabel="overlap (0-1)", title="(a) Agreement and stability of top-5 features")
ax[0].legend(frameon=False, fontsize=6.5, loc="upper left")
for j, k in enumerate(keys):
    d = summ[k]["deletion_drop_by_k"]
    if not d:
        continue
    for meth, ls in (("shap", "-"), ("lime", "--"), ("random", ":")):
        ax[1].plot(range(1, 6), d[meth], ls, color=PALETTE[j], marker="o", ms=2.5,
                   label=f"{NICE[k]} - {meth.upper() if meth != 'random' else 'random'}")
ax[1].set(xlabel="k features replaced by baseline", ylabel="mean drop in P(at risk)",
          title="(b) Deletion test on flagged students")
ax[1].legend(frameon=False, fontsize=5.5)
fig.tight_layout()
fig.savefig(FIGURES / "fig6_explanation_quality.png")
plt.close(fig)

gi = pd.read_csv(TABLES / "e5_global_importance.csv")
gi.groupby("dataset").head(10).round(4).to_csv(TABLES / "paper_table_top10_features.csv", index=False)

# Does global importance merely track how many distinct values a feature has?
# (Tree models split more often on high-cardinality features even when they carry no signal.)
from scipy.stats import spearmanr
from xaiedu.data import load_kaggle, load_oulad, load_uci
loaders = {"kaggle": load_kaggle, "uci_sem1": lambda: load_uci("sem1"), "oulad_d56": lambda: load_oulad(56)}
card_rows = []
for k, load in loaders.items():
    X = load().X
    d = gi[gi.dataset == k].copy()
    d["n_unique"] = [X[f].nunique() for f in d.feature]
    rho, p = spearmanr(d["mean_abs_shap"], d["n_unique"])
    card_rows.append({"dataset": NICE[k], "spearman_importance_vs_cardinality": rho, "p_value": p,
                      "n_features": len(d)})
pd.DataFrame(card_rows).round(4).to_csv(TABLES / "paper_table_importance_cardinality.csv", index=False)
print(pd.DataFrame(card_rows).round(3).to_string())

# --------------------------------------------------------------------- fairness
frows = []
for k in ["uci_sem1", "oulad_d56"]:
    p = TABLES / f"e6_{k}_group_metrics.csv"
    if p.exists():
        frows.append(pd.read_csv(p).assign(dataset=NICE[k]))
if frows:
    fm = pd.concat(frows)
    fm.round(3).to_csv(TABLES / "paper_table_fairness.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for a, (dname, d) in zip(axes, fm.groupby("dataset", sort=False)):
        lab = [f"{r.attribute}: {r.group}" for r in d.itertuples()]
        y = np.arange(len(d))
        a.barh(y - 0.2, d["tpr"], 0.4, color=PALETTE[0], label="TPR (recall)")
        a.barh(y + 0.2, d["fpr"], 0.4, color=PALETTE[3], label="FPR")
        a.set_yticks(y, lab, fontsize=6)
        a.invert_yaxis()
        a.set(xlim=(0, 1), title=dname)
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig7_fairness.png")
    plt.close(fig)
    gaps = pd.concat([pd.read_csv(TABLES / f"e6_{k}_gaps.csv").assign(dataset=NICE[k])
                      for k in ["uci_sem1", "oulad_d56"] if (TABLES / f"e6_{k}_gaps.csv").exists()])
    gaps.round(3).to_csv(TABLES / "paper_table_fairness_gaps.csv", index=False)

print("figures:", sorted(p.name for p in FIGURES.glob("*.png")))
print("tables:", sorted(p.name for p in TABLES.glob("paper_*.csv")))
