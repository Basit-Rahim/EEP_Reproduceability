"""
Performance metrics and statistical tests.

Metrics per held-out test set: AUC-ROC (also survey-weighted), accuracy,
precision / recall / F1 / PR-AUC for the out-of-school class, and F1 for the
attending class.

Tests
  * corrected resampled t-test (Nadeau & Bengio 2003) for means over repeated
    cross-validation: 95% CIs and ML-vs-LR differences
  * DeLong test (DeLong et al. 1988; fast algorithm of Sun & Xu 2014) comparing
    two models' AUCs on the same children, using out-of-fold predictions
  * household-cluster bootstrap of the AUC difference: resamples whole households,
    so the dependence between siblings is reflected (DeLong assumes independence)
  * Holm correction across the nine districts
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)


def _weighted_auc(y, p, w):
    """Survey-weighted AUC over the children that have a sampling weight."""
    if w is None:
        return np.nan
    ok = np.isfinite(w)
    if len(np.unique(y[ok])) < 2:
        return np.nan
    return roc_auc_score(y[ok], p[ok], sample_weight=w[ok])


def fold_metrics(y, p, w=None, threshold=0.5):
    """y: 1 = attending, 0 = out of school; p = predicted probability of attending."""
    pred = (p >= threshold).astype(int)
    out_true, out_pred = (y == 0), (pred == 0)
    return {
        "auc": roc_auc_score(y, p),
        "auc_weighted": _weighted_auc(y, p, w),
        "accuracy": accuracy_score(y, pred),
        "precision_out": precision_score(out_true, out_pred, zero_division=0),
        "recall_out": recall_score(out_true, out_pred, zero_division=0),
        "f1_out": f1_score(out_true, out_pred, zero_division=0),
        "pr_auc_out": average_precision_score(out_true, 1 - p),
        "f1_attending": f1_score(y, pred, zero_division=0),
        "prevalence_out": float(out_true.mean()),
    }


def corrected_ttest(values, n_train, n_test, alpha=0.05):
    """Mean, 95% CI and two-sided p (H0: mean = 0) with the Nadeau-Bengio correction
    for overlapping training sets in repeated cross-validation."""
    x = np.asarray(values, dtype=float)
    J = len(x)
    mean = float(x.mean())
    se = np.sqrt((1 / J + n_test / n_train) * x.var(ddof=1))
    t = mean / se if se > 0 else np.inf
    crit = stats.t.ppf(1 - alpha / 2, J - 1)
    return {"mean": mean, "ci_lo": mean - crit * se, "ci_hi": mean + crit * se,
            "p": float(2 * stats.t.sf(abs(t), J - 1))}


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def _midranks(x):
    order = np.argsort(x)
    xs = x[order]
    ranks = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j < len(x) and xs[j] == xs[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(len(x))
    out[order] = ranks
    return out


def delong(y, p1, p2):
    """DeLong test for two correlated AUCs on the same children.
    Returns AUC1, AUC2, difference, 95% CI of the difference and two-sided p."""
    y = np.asarray(y).astype(bool)
    preds = np.vstack([p1, p2])
    pos, neg = preds[:, y], preds[:, ~y]
    m, n = pos.shape[1], neg.shape[1]
    tx = np.array([_midranks(r) for r in pos])
    ty = np.array([_midranks(r) for r in neg])
    tz = np.array([_midranks(r) for r in np.hstack([pos, neg])])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1) / (2 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    se = np.sqrt(var)
    z = diff / se if se > 0 else np.inf
    return {"auc_1": aucs[0], "auc_2": aucs[1], "diff": diff,
            "ci_lo": diff - 1.96 * se, "ci_hi": diff + 1.96 * se,
            "p": float(2 * stats.norm.sf(abs(z)))}


def cluster_bootstrap_auc_diff(y, p1, p2, groups, n_boot=2000, seed=42):
    """AUC(p1) - AUC(p2) with a household-cluster bootstrap: whole households are
    resampled with replacement, so siblings stay together and the dependence between
    them is reflected in the uncertainty (DeLong assumes independent children).
    Returns the observed difference, a 95 % percentile CI and a two-sided p-value
    (twice the share of bootstrap differences on the other side of zero)."""
    y, p1, p2, groups = map(np.asarray, (y, p1, p2, groups))
    observed = roc_auc_score(y, p1) - roc_auc_score(y, p2)
    _, codes = np.unique(groups, return_inverse=True)
    members = np.split(np.argsort(codes, kind="stable"), np.cumsum(np.bincount(codes))[:-1])
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(members), len(members))
        idx = np.concatenate([members[i] for i in pick])
        if y[idx].min() == y[idx].max():
            continue
        diffs.append(roc_auc_score(y[idx], p1[idx]) - roc_auc_score(y[idx], p2[idx]))
    diffs = np.asarray(diffs)
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"diff": observed, "ci_lo": float(np.percentile(diffs, 2.5)),
            "ci_hi": float(np.percentile(diffs, 97.5)),
            "p": float(max(p, 1 / len(diffs)))}   # resolution limit of the bootstrap


def summarise(results, df, target, weights="weights"):
    """Per-fold metrics for every model, from the output of modeling.run_cv."""
    y = df[target].to_numpy()
    w = df[weights].to_numpy() if weights in df else None
    rows = []
    n = len(y)
    for res in results:
        te = res["test"]
        for model, p in res["proba"].items():
            m = fold_metrics(y[te], p, w[te] if w is not None else None)
            rows.append({"repeat": res["repeat"], "fold": res["fold"], "model": model,
                         "n_train": n - len(te), "n_test": len(te), **m})
    return pd.DataFrame(rows)


def out_of_fold(results, n, repeat=0):
    """Out-of-fold predictions of one repeat: every child predicted once."""
    oof = {}
    for res in results:
        if res["repeat"] == repeat:
            for model, p in res["proba"].items():
                oof.setdefault(model, np.full(n, np.nan))[res["test"]] = p
    return pd.DataFrame(oof)
