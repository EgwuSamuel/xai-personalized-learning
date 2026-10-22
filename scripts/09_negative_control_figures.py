"""Figures and tables for E7-E9 (negative controls, synthetic data, hyperparameter sensitivity)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from xaiedu.config import FIGURES, TABLES

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 600, "savefig.bbox": "tight"})
MODEL_COL = {"XGB": "#4C72B0", "LR": "#DD8452", "EBM": "#55A868", "EBM_main": "#55A868",
             "XGB_depth6": "#4C72B0", "XGB_depth2": "#8172B3", "XGB_tuned": "#64B5CD"}
DS_MARK = {"kaggle": "s", "uci_sem1": "o", "oulad_d56": "^"}
DS_NAME = {"kaggle": "Kaggle", "uci_sem1": "UCI", "oulad_d56": "OULAD"}

# ------------------------------------------------------------------ study design (updated)
fig, ax = plt.subplots(figsize=(7.2, 4.0))
ax.set_xlim(0, 10); ax.set_ylim(0, 7.3); ax.axis("off")


def box(x, y, w, h, text, fc, fs=6.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=fc, ec="#555555", lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(x0, y0, x1, y1, ls="-"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8, linestyle=ls))


box(0.1, 5.9, 2.5, 1.2, "Kaggle v1 (n = 10,000)\nno documentation\n→ natural negative control", "#F4D6D6")
box(0.1, 4.3, 2.5, 1.2, "UCI #697 (n = 4,424)\nenrolment / sem. 1 / sem. 2", "#D6E4F4")
box(0.1, 2.7, 2.5, 1.2, "OULAD (32,593 registrations)\ncutoffs day 14 … 140", "#D6F0DC")
box(0.1, 0.35, 2.5, 1.2, "Synthetic data\nsignal strength × cardinality", "#EEEEEE")
box(3.25, 5.9, 2.7, 1.2, "E1 baseline\nE2 signal audit\nE2b resampling order", "#FFFFFF")
box(3.25, 4.3, 2.7, 1.2, "E3 nested 5×2 CV\n7 models, imbalance ablation\nFriedman / Nemenyi", "#FFFFFF")
box(3.25, 2.7, 2.7, 1.2, "E4 temporal validation\ntrain 2013 → test 2014\nbootstrap CIs", "#FFFFFF")
box(6.6, 2.7, 3.3, 4.4, "E5 explanation diagnostics\n(XGBoost; SHAP and LIME)\nagreement, stability,\ndeletion faithfulness\n\nAppendix A: counterfactuals (UCI)\nAppendix B: fairness audit (E6)", "#FFF4D6")
box(3.25, 0.35, 6.65, 1.2, "E7 label-permutation controls on Kaggle, UCI and OULAD (XGBoost, LR, EBM)\n"
    "E8 synthetic signal × cardinality      E9 hyperparameter sensitivity\n"
    "E5 diagnostics + cardinality dependence + held-out checks", "#FCE8D5")
for y in (6.5, 4.9, 3.3):
    arrow(2.65, y, 3.2, y)
    arrow(6.0, y, 6.55, y)
arrow(2.65, 0.95, 3.2, 0.95)
arrow(8.25, 2.65, 8.25, 1.6, ls="--")
ax.text(8.35, 2.12, "same protocol,\nall datasets", fontsize=5.5, va="center", color="#555555")
fig.savefig(FIGURES / "fig1_study_design_v2.png")
plt.close(fig)

# ------------------------------------------------------------------ E7 negative controls
e7 = pd.read_csv(TABLES / "e7_negative_controls.csv")
e7["null"] = (e7["condition"] != "real") | (e7["dataset"] == "kaggle")
e7["deletion_gain"] = e7["deletion_attr_k5"] - e7["deletion_random_k5"]
panels = [("global_stability_rho", "(a) Global stability\n(mean pairwise Spearman)"),
          ("deletion_gain", "(b) Deletion gain, k = 5\n(attribution − random order)"),
          ("agreement_top5", "(c) SHAP–LIME top-5 overlap\n(XGBoost only)"),
          ("cardinality_rho", "(d) Spearman(importance,\nfeature cardinality)")]
fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.45))
for ax, (col, lab) in zip(axes, panels):
    for _, r in e7.iterrows():
        if pd.isna(r[col]):
            continue
        ax.scatter(r["test_roc_auc"], r[col], marker=DS_MARK[r["dataset"]], s=22,
                   facecolors=MODEL_COL[r["model"]] if not r["null"] else "none",
                   edgecolors=MODEL_COL[r["model"]], linewidths=0.9)
    ax.axvline(0.5, color="#999999", lw=0.6, ls="--")
    ax.set_xlabel("test ROC-AUC"); ax.set_title(lab, fontsize=7)
    ax.set_xlim(0.40, 0.97)
from matplotlib.lines import Line2D
handles = [Line2D([], [], marker="o", ls="", color=MODEL_COL[m], label=m.replace("EBM_main", "EBM")) for m in e7["model"].unique()]
handles += [Line2D([], [], marker=DS_MARK[d], ls="", color="#555555", mfc="none", label=DS_NAME[d]) for d in DS_MARK]
handles += [Line2D([], [], marker="o", ls="", color="#555555", label="real labels"),
            Line2D([], [], marker="o", ls="", color="#555555", mfc="none", label="no signal / permuted")]
fig.legend(handles=handles, loc="lower center", ncol=8, fontsize=6, frameon=False, bbox_to_anchor=(0.5, -0.08))
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(FIGURES / "fig8_negative_controls.png")
plt.close(fig)

# ------------------------------------------------------------------ E8 synthetic
e8 = pd.read_csv(TABLES / "e8_synthetic_runs.csv")
panels = [("test_roc_auc", "(a) Test ROC-AUC"), ("global_stability_rho", "(b) Global stability\n(mean pairwise Spearman)"),
          ("cardinality_rho", "(c) Spearman(importance,\nfeature cardinality)"),
          ("signal_in_top2", "(d) Share of signal\nfeatures in top 2"),
          ("heldout_importance_sum", "(e) Held-out permutation\nimportance (sum)")]
fig, axes = plt.subplots(1, 5, figsize=(7.4, 2.1))
order = ["XGB_depth6", "XGB_depth2", "XGB_tuned", "LR"]
for ax, (col, lab) in zip(axes, panels):
    for m in order:
        g = e8[e8["model"] == m].groupby("strength")[col].agg(["mean", "std"])
        ax.errorbar(g.index, g["mean"], yerr=g["std"], marker="o", ms=3, lw=1, capsize=1.5,
                    color=MODEL_COL[m], label=m.replace("_", " "))
    ax.set_xlabel("signal strength s"); ax.set_title(lab, fontsize=6.5)
    ax.set_xticks([0, 0.25, 0.5, 1.0]); ax.set_xticklabels(["0", ".25", ".5", "1"])
axes[0].legend(fontsize=5.5, frameon=False)
fig.tight_layout()
fig.savefig(FIGURES / "fig9_synthetic.png")
plt.close(fig)
print("figures written")
