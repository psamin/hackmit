"""Small, honest statistics: only what the analysis needs, each piece tested against a known answer.

Everything here is numpy/scipy. The two things that matter most for not fooling yourself:

  bh()              adjusts p-values for testing many things at once (Benjamini-Hochberg, false discovery rate)
  logit_cluster()   a logistic regression whose standard errors allow for each patient's days being related to
                    each other (clustered by patient). Without this, the same patient counted 360 times looks like
                    360 independent people, and everything looks significant.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as sps


def bh(p) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values): flag q < 0.05 to keep the share of false discoveries near 5%."""
    p = np.asarray(p, dtype=float)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out


def trend(y) -> tuple[float, float, float]:
    """(Kendall tau, p-value, Theil-Sen slope per step) of a series against time. Robust to outliers; NaNs are skipped."""
    y = np.asarray(y, dtype=float)
    t = np.arange(y.size, dtype=float)
    ok = ~np.isnan(y)
    if ok.sum() < 8 or np.ptp(y[ok]) == 0:
        return 0.0, 1.0, 0.0
    tau, p = sps.kendalltau(t[ok], y[ok])
    slope = sps.theilslopes(y[ok], t[ok])[0]
    return float(tau), float(p), float(slope)


@dataclass
class Fit:
    names: list[str]
    coef: np.ndarray
    se: np.ndarray
    p: np.ndarray
    n: int
    groups: int

    def ci(self, name: str, level: float = 0.95) -> tuple[float, float]:
        i = self.names.index(name)
        z = sps.norm.ppf(0.5 + level / 2)
        return float(self.coef[i] - z * self.se[i]), float(self.coef[i] + z * self.se[i])

    def get(self, name: str) -> dict:
        i = self.names.index(name)
        lo, hi = self.ci(name)
        return {"coef": float(self.coef[i]), "se": float(self.se[i]), "p": float(self.p[i]), "ci95": (lo, hi)}


def logit_cluster(X: np.ndarray, y: np.ndarray, groups: np.ndarray, names: list[str], ridge: float = 1e-8, iters: int = 40) -> Fit:
    """Logistic regression by IRLS with standard errors clustered by `groups` (the patient).

    X must already contain an intercept column if one is wanted. Returns coefficients on the log-odds scale."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = X.shape
    b = np.zeros(k)
    eye = np.eye(k) * ridge
    for _ in range(iters):
        eta = np.clip(X @ b, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        w = np.maximum(p * (1 - p), 1e-9)
        z = eta + (y - p) / w
        xtw = X.T * w
        b_new = np.linalg.solve(xtw @ X + eye, xtw @ z)
        done = np.max(np.abs(b_new - b)) < 1e-9
        b = b_new
        if done:
            break
    eta = np.clip(X @ b, -30, 30)
    p = 1.0 / (1.0 + np.exp(-eta))
    w = np.maximum(p * (1 - p), 1e-9)
    bread = np.linalg.inv((X.T * w) @ X + eye)
    _, inv = np.unique(groups, return_inverse=True)
    g = inv.max() + 1
    scores = np.zeros((g, k))
    np.add.at(scores, inv, X * (y - p)[:, None])
    meat = scores.T @ scores * (g / max(g - 1, 1))
    cov = bread @ meat @ bread
    se = np.sqrt(np.maximum(np.diag(cov), 1e-30))
    pvals = 2 * sps.norm.sf(np.abs(b / se))
    return Fit(list(names), b, se, pvals, n, g)


def crosscorr(x: np.ndarray, y: np.ndarray, lags: range) -> np.ndarray:
    """Per-row correlation of x[t] with y[t + lag] for each lag (rows are patients, columns are days).

    lag > 0 means x comes first: x LEADS y by `lag` days. Result shape (rows, len(lags)). Rows must have no NaNs."""
    rows, d = x.shape
    out = np.empty((rows, len(lags)))
    for j, lag in enumerate(lags):
        a, b = (x[:, : d - lag], y[:, lag:]) if lag >= 0 else (x[:, -lag:], y[:, : d + lag])
        a = a - a.mean(axis=1, keepdims=True)
        b = b - b.mean(axis=1, keepdims=True)
        den = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
        out[:, j] = np.where(den > 0, (a * b).sum(axis=1) / np.where(den > 0, den, 1), 0.0)
    return out


def fill_gaps(a: np.ndarray) -> np.ndarray:
    """Straight-line fill of missing days (NaN) within each row. Rows with nothing to go on become all zeros."""
    out = a.copy()
    t = np.arange(a.shape[1])
    for i in range(a.shape[0]):
        ok = ~np.isnan(a[i])
        out[i] = np.interp(t, t[ok], a[i][ok]) if ok.sum() >= 2 else 0.0
    return out


def smooth(a: np.ndarray, width: int = 7) -> np.ndarray:
    """Trailing moving average along days (rows are patients). The first days use what there is."""
    c = np.cumsum(np.insert(a, 0, 0.0, axis=1), axis=1)
    idx = np.arange(1, a.shape[1] + 1)
    lo = np.maximum(idx - width, 0)
    return (c[:, idx] - c[:, lo]) / (idx - lo)


def detrend(a: np.ndarray) -> np.ndarray:
    """Remove each row's straight-line trend."""
    t = np.arange(a.shape[1], dtype=float)
    t = t - t.mean()
    slope = ((a - a.mean(axis=1, keepdims=True)) * t).sum(axis=1) / (t * t).sum()
    return a - a.mean(axis=1, keepdims=True) - slope[:, None] * t
