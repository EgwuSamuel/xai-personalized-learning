"""Fast checks: leakage-safe feature sets, determinism, and explanation plumbing.

Uses synthetic data shaped like the real datasets, so the tests run without
downloading anything.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

from xaiedu.counterfactuals import counterfactuals
from xaiedu.data import UCI_SCENARIOS, UCI_SEM1, UCI_SEM2, Dataset
from xaiedu.evaluate import cross_validate, fit_model
from xaiedu.explain import LimeWrapper, agreement, shap_values
from xaiedu.fairness import gaps, group_metrics


@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(0)
    n = 500
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n),
                      "units": rng.integers(0, 8, n).astype(float),
                      "c": rng.choice(["x", "y", "z"], n)})
    logit = 1.5 * X["a"] - 1.0 * X["units"] / 4 + (X["c"] == "x") * 1.0
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int))
    return Dataset("toy", X, y, ["c"], ["a", "b", "units"])


def test_uci_scenarios_are_nested_and_leakage_free():
    assert UCI_SCENARIOS["enrolment"] == []
    assert not set(UCI_SCENARIOS["sem1"]) & set(UCI_SEM2)
    assert set(UCI_SCENARIOS["sem2"]) == set(UCI_SEM1) | set(UCI_SEM2)


def test_cv_is_deterministic(toy):
    a = cross_validate(toy, ["LR", "XGB"], ["weight+threshold"], folds=3, repeats=1, log=lambda *_: None)
    b = cross_validate(toy, ["LR", "XGB"], ["weight+threshold"], folds=3, repeats=1, log=lambda *_: None)
    pd.testing.assert_frame_equal(a, b)
    assert (a["roc_auc"] > 0.6).all()


def test_shap_aggregation_and_lime(toy):
    model, _, thr = fit_model("XGB", toy.categorical, toy.numeric, toy.X, toy.y, "weight+threshold")
    S = shap_values(model, toy.X.iloc[:50], toy.X.iloc[:20])
    assert S.shape == (20, toy.X.shape[1])
    assert np.isfinite(S).all()
    # the informative feature must get larger attributions than pure noise
    assert np.abs(S[:, toy.X.columns.get_loc("a")]).mean() > np.abs(S[:, toy.X.columns.get_loc("b")]).mean()
    L = LimeWrapper(model, toy.X, toy.categorical, num_samples=500).explain(toy.X.iloc[:5])
    assert L.shape == (5, toy.X.shape[1])
    ag = agreement(S[:5], L, k=2)
    assert ag["top2_overlap"].between(0, 1).all()


def test_lime_weights_binary_features():
    """A predictive, mostly-1 binary column must not get zero LIME weight."""
    rng = np.random.default_rng(1)
    n = 600
    X = pd.DataFrame({"fees_ok": (rng.random(n) < 0.85).astype(int), "z": rng.normal(size=n),
                      "c": rng.choice(["p", "q"], n)})
    y = pd.Series(((X["fees_ok"] == 0) | (rng.random(n) < 0.15)).astype(int))
    model, _, _ = fit_model("LR", ["c"], ["fees_ok", "z"], X, y, "weight", tune=False)
    L = LimeWrapper(model, X, ["c"], num_samples=500).explain(X.iloc[:5])
    assert np.abs(L[:, 0]).mean() > 0.05


def test_counterfactuals_respect_constraints(toy):
    model, _, thr = fit_model("LR", toy.categorical, toy.numeric, toy.X, toy.y, "weight+threshold")
    p = model.predict_proba(toy.X)[:, 1]
    # a flagged student close to the threshold, so an actionable change can flip the decision
    above = np.where(p >= thr)[0]
    x = toy.X.iloc[above[np.argmin(p[above])]]
    cf = counterfactuals(model, x, toy.X, {"units": "increase", "c": ["y", "z"]}, threshold=thr, seed=1)
    assert len(cf) > 0
    assert (cf["units"] >= x["units"]).all()
    assert (cf["p_dropout"] < thr).all()
    assert (cf["a"] == x["a"]).all()  # non-actionable feature untouched


def test_fairness_gaps():
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0] * 10)
    pred = np.array([1, 0, 1, 0, 0, 0, 0, 0] * 10)
    g = pd.Series(["m", "m", "m", "m", "f", "f", "f", "f"] * 10, name="sex")
    gm = group_metrics(y, pred.astype(float), pred, g, min_n=5)
    assert gaps(gm)["equal_opportunity_diff"].iloc[0] == pytest.approx(1.0)


def test_exact_attributions_for_lr_and_ebm_are_additive(toy):
    """Linear SHAP (LR) and EBM term contributions sum to the log-odds minus the background mean."""
    from xaiedu.controls import attributions
    from xaiedu.explain import base_pipeline
    bg, ex = toy.X.iloc[:60], toy.X.iloc[100:110]
    for name in ("LR", "EBM"):
        model, _, _ = fit_model(name, toy.categorical, toy.numeric, toy.X, toy.y, "weight", tune=False)
        pipe = base_pipeline(model)
        S = attributions(name, model, bg, ex)
        assert S.shape == (10, toy.X.shape[1])
        clf, pre = pipe.named_steps["clf"], pipe.named_steps["pre"]
        f_ex = clf.decision_function(pre.transform(ex))
        f_bg = clf.decision_function(pre.transform(bg)).mean()
        np.testing.assert_allclose(S.sum(1), f_ex - f_bg, atol=1e-6)
        # the informative feature outranks pure noise
        a, b = toy.X.columns.get_loc("a"), toy.X.columns.get_loc("b")
        assert np.abs(S[:, a]).mean() > np.abs(S[:, b]).mean()


def test_heldout_importance_is_near_zero_without_signal():
    from xaiedu.controls import heldout_importance
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"u": rng.integers(0, 50, 2000).astype(float), "v": rng.normal(size=2000)})
    y = pd.Series(rng.integers(0, 2, 2000))
    model, _, _ = fit_model("XGB", [], ["u", "v"], X.iloc[:1500], y.iloc[:1500], "weight", tune=False)
    hi = heldout_importance(model, X.iloc[1500:], y.iloc[1500:], seed=0)
    assert (hi["mean"].abs() < 0.05).all()
