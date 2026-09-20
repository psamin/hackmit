"""The blind analysis: hand it a Dataset and it says what it found. It is never given the answer key.

    result = analyze(data)
    result.findings["night_wakes"]            # an estimate, a 95% interval, a corrected p-value, and the caveat
    result.rising, result.drops               # which patients, per method

Five questions, each with the method chosen to stop it fooling itself:

  1. WHO IS ON A STEADY SLIDE?     Kendall trend on each patient's weekly questions, corrected for testing 240
                                   patients (Benjamini-Hochberg), AND the change has to be big enough to matter.
                                   Bad weeks make a patient look like they are sliding; the size gate is what
                                   stops that.
  2. WHAT LEADS WHAT?              cross-correlation of questions against doses, per patient, averaged. The
                                   interval comes from resampling patients; the p-value from shifting each patient's
                                   questions by a random number of days (which keeps their rhythms, and breaks any
                                   real link).
  3. WHO HAD AN ABRUPT DROP?       best single step in each patient's on-time rate, compared with what the same
                                   patient's own shuffled series can do by luck, and with a straight-line fit (a slide
                                   is not a step). Dated to the day.
  4. WHAT DRIVES A DAY'S DOSES?    ONE logistic regression with every candidate at once, weekday and each patient's
                                   own baseline included, and standard errors that treat each patient as one unit.
                                   Fitting them together is what removes the confounded decoy (social minutes) and
                                   the mediated one (sleep).
  5. THE NAIVE VERSION             the same candidates, one correlation at a time, every patient-day counted as an
                                   independent observation. Kept only to show what it would have got wrong.

Settings live in Config and were fixed using development seeds (DEV_SEEDS) that are NOT the seeds the scorecard is
reported on. See README.md: choosing thresholds by looking at the answer you are then graded on is how people fool
themselves.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats as sps

from . import stats
from .synth import Dataset

DEV_SEEDS = (9001, 9002, 9003, 9004, 9005)          # thresholds were set on these; never scored on these


@dataclass(frozen=True)
class Config:
    alpha: float = 0.05                    # false discovery rate for the patient-level and covariate-level flags
    min_weekly_change: float = 2.0         # questions per day, across the whole period, before a rise is "big enough"
    lags: tuple[int, int] = (-35, 35)      # days; positive = questions come first
    smooth_days: int = 7
    n_boot: int = 500
    n_perm: int = 300
    step_margin: int = 21                  # a step must be at least this many days from either end
    step_min_drop: float = 0.08            # in on-time share, before a drop is "big enough"
    step_perms: int = 40                   # shuffled copies of every patient's series, to learn what luck alone produces
    step_bic_gate: float = 0.0             # how much better than a straight line a step must fit (0: at least as good)
    seed: int = 0                          # for the resampling and shuffling inside the analysis (not the data)


@dataclass
class Finding:
    id: str
    estimate: float | None
    ci95: tuple[float, float] | None
    q: float | None                        # corrected p-value where one applies
    n: int
    method: str
    caveat: str
    detail: dict = field(default_factory=dict)


@dataclass
class Analysis:
    findings: dict[str, Finding]
    rising: np.ndarray                     # (n,) bool
    rising_change: np.ndarray              # (n,) estimated change in questions/day over the period
    drops: np.ndarray                      # (n,) bool: an abrupt, lasting drop
    drop_day: np.ndarray                   # (n,) day of the best step (only meaningful where drops is True)
    drop_size: np.ndarray                  # (n,) change in on-time share across the step
    screen: list[dict]                     # every candidate, fitted together
    naive: list[dict]                      # every candidate, one at a time
    leadlag_lags: np.ndarray
    leadlag_curve: np.ndarray              # mean correlation at each lag
    config: Config


# --------------------------------------------------------------------------------------------------------------
def _weekly(a: np.ndarray) -> np.ndarray:
    n, d = a.shape
    w = d // 7
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(a[:, : w * 7].reshape(n, w, 7), axis=2)


def rising_questions(data: Dataset, cfg: Config) -> tuple[np.ndarray, np.ndarray, Finding]:
    weekly = _weekly(data.metrics["repeated_questions"])
    weeks = weekly.shape[1]
    res = np.array([stats.trend(weekly[i]) for i in range(data.n)])
    tau, p, slope = res[:, 0], res[:, 1], res[:, 2]
    q = stats.bh(p)
    change_per_day = slope * (weeks - 1)                       # slope is questions/day per week: times the weeks = total change
    flagged = (q < cfg.alpha) & (tau > 0) & (change_per_day >= cfg.min_weekly_change)
    f = Finding("rising_questions", float(np.median(change_per_day[flagged])) if flagged.any() else None, None,
                float(np.min(q)) if data.n else None, data.n,
                "Kendall trend per patient on weekly means, Benjamini-Hochberg across patients, plus a minimum size",
                "A patient can drift up for a few weeks and come back; this flags sustained rises only, and it is a "
                "signal to look at, not a diagnosis.",
                {"flagged": int(flagged.sum()), "of": data.n, "min_change_per_day": cfg.min_weekly_change})
    return flagged, change_per_day, f


def lead_lag(data: Dataset, cfg: Config) -> tuple[Finding, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    lags = range(cfg.lags[0], cfg.lags[1] + 1)
    lag_arr = np.array(list(lags))
    adh = data.adherence
    q = data.metrics["repeated_questions"]
    keep = (np.isnan(q).mean(axis=1) < 0.3) & (np.isnan(adh).mean(axis=1) < 0.3)
    x = stats.detrend(stats.smooth(stats.fill_gaps(q[keep]), cfg.smooth_days))
    y = stats.detrend(stats.smooth(stats.fill_gaps(adh[keep]), cfg.smooth_days))
    n, d = x.shape
    curve = stats.crosscorr(x, y, lags)                       # (n, lags)
    mean = curve.mean(axis=0)
    best = int(lag_arr[np.argmin(mean)])                      # doses fall when questions rise: the most negative point
    boots = np.array([lag_arr[np.argmin(curve[rng.integers(0, n, n)].mean(axis=0))] for _ in range(cfg.n_boot)])
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    observed = mean.min()
    idx = np.arange(d)[None, :]
    null = np.empty(cfg.n_perm)
    for b in range(cfg.n_perm):
        s = rng.integers(45, d - 45, n)
        shifted = np.take_along_axis(x, (idx - s[:, None]) % d, axis=1)
        null[b] = stats.crosscorr(shifted, y, lags).mean(axis=0).min()
    p = float((1 + np.sum(null <= observed)) / (1 + cfg.n_perm))
    f = Finding("lead_lag", float(best), (lo, hi), p, int(n),
                "Per-patient cross-correlation of smoothed, detrended questions and on-time doses; interval by resampling "
                "patients; p-value by circularly shifting each patient's questions",
                "Timing, not cause: questions moving first is what the data shows, and it does not say questions cause "
                "the change in doses.",
                {"peak_correlation": float(observed), "positive_lag_means": "questions come first"})
    return f, lag_arr, mean


def _without_questions_effect(a: np.ndarray, q: np.ndarray, lag: int, width: int) -> np.ndarray:
    """Take out of each patient's on-time share the part that their own questions, `lag` days earlier, explain.

    What is left is what the questions do NOT explain, which is where an abrupt change would show. The lag is the one
    the lead-lag analysis found in this same data."""
    x = stats.smooth(stats.fill_gaps(q), width)
    d = x.shape[1]
    shifted = np.concatenate([np.repeat(x[:, :1], lag, axis=1), x[:, : d - lag]], axis=1) if lag > 0 else x
    xc = shifted - shifted.mean(axis=1, keepdims=True)
    beta = (xc * (a - a.mean(axis=1, keepdims=True))).sum(axis=1) / np.maximum((xc * xc).sum(axis=1), 1e-9)
    return a - beta[:, None] * xc


def abrupt_drops(data: Dataset, cfg: Config, lag: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, Finding]:
    rng = np.random.default_rng(cfg.seed + 1)
    a = stats.fill_gaps(data.adherence)
    if lag is not None and lag > 0:
        a = _without_questions_effect(a, data.metrics["repeated_questions"], lag, cfg.smooth_days)
    n, d = a.shape
    lo, hi = cfg.step_margin, d - cfg.step_margin

    def best_step(a):
        cs = np.cumsum(a, axis=1)
        tot = cs[:, -1:]
        taus = np.arange(lo, hi)
        left, right = cs[:, taus - 1], tot - cs[:, taus - 1]
        gain = left ** 2 / taus[None, :] + right ** 2 / (d - taus)[None, :]
        j = np.argmax(gain, axis=1)
        tau = taus[j]
        rows = np.arange(a.shape[0])
        mL, mR = left[rows, j] / tau, right[rows, j] / (d - tau)
        sse = (a ** 2).sum(axis=1) - gain[rows, j]
        return tau, mR - mL, np.maximum(sse, 1e-9)

    tau, delta, sse_step = best_step(a)
    t = np.arange(d, dtype=float)
    t = t - t.mean()
    slope = ((a - a.mean(axis=1, keepdims=True)) * t).sum(axis=1) / (t * t).sum()
    sse_lin = ((a - a.mean(axis=1, keepdims=True) - slope[:, None] * t) ** 2).sum(axis=1)
    sd = np.sqrt(sse_step / (d - 3))
    tstat = delta / (sd * np.sqrt(1 / tau + 1 / (d - tau)))

    # what a patient's own series produces by luck: shuffle it in blocks (keeps its habits, breaks any step)
    block = 14
    nb = d // block
    null_t = []
    for _ in range(cfg.step_perms):
        pieces = a[:, : nb * block].reshape(n, nb, block)
        order = np.argsort(rng.random((n, nb)), axis=1)
        shuffled = np.take_along_axis(pieces, order[:, :, None], axis=1).reshape(n, nb * block)
        pad = a[:, nb * block:]
        sh = np.concatenate([shuffled, pad], axis=1)
        tt, dd, ss = best_step(sh)
        null_t.append(dd / (np.sqrt(ss / (d - 3)) * np.sqrt(1 / tt + 1 / (d - tt))))
    null_sorted = np.sort(np.concatenate(null_t))
    # one-sided p: how often does luck alone give a drop at least this sharp?
    pvals = (1 + np.searchsorted(null_sorted, tstat, side="right")) / (1 + null_sorted.size)
    q = stats.bh(pvals)
    bic_gain = d * np.log(sse_lin / sse_step) - np.log(d)
    flagged = (q < cfg.alpha) & (delta <= -cfg.step_min_drop) & (bic_gain >= cfg.step_bic_gate)
    f = Finding("abrupt_drop", float(np.median(delta[flagged])) if flagged.any() else None, None, None, n,
                "Best single step in each patient's on-time share; must beat the same patient's shuffled series and a "
                "straight-line fit, and be big enough to matter",
                "Dates are approximate (a step fitted to noisy daily data). A flagged drop is a reason to ask what "
                "changed that week: a new medicine, an illness, a fall.",
                {"flagged": int(flagged.sum()), "of": n})
    return flagged, tau, delta, f


# --------------------------------------------------------------------------------------------------------------
CANDIDATES_EXCLUDED = {"repeated_questions"}         # analysed on its own, as a leading indicator


def _candidate_names(data: Dataset) -> list[str]:
    return [m for m in data.metrics if m not in CANDIDATES_EXCLUDED]


def screen(data: Dataset, cfg: Config) -> tuple[list[dict], dict[str, Finding]]:
    """Every candidate at once, with weekday and each patient's own baseline in the model."""
    n, d = data.n, data.days
    weekend = (data.dow >= 5).astype(float)
    names = _candidate_names(data)

    complete = np.ones((n, d), dtype=bool)                  # only days where EVERY candidate was recorded are used:
    for m in names:                                          # filling gaps with averages would let a candidate that is
        complete &= ~np.isnan(data.metrics[m])               # recorded (sleep) stand in for one that is not (wakes)

    def centred(a):
        c = a - np.nanmean(a, axis=1, keepdims=True)
        return np.where(np.isnan(c), 0.0, c)

    cols, labels = {}, []
    for m in names:
        c = centred(data.metrics[m])
        if m != "night_wakes":
            c = c / (c.std() or 1.0)                         # everything else per standard deviation; wakes stay per wake
        cols[m] = c
    rate = np.nanmean(np.stack([data.dose_am, data.dose_pm]), axis=(0, 2))
    base = np.log(np.clip(rate, 0.02, 0.98) / (1 - np.clip(rate, 0.02, 0.98)))

    rows_X, rows_y, rows_g = [], [], []
    for k, dose in enumerate((data.dose_am, data.dose_pm)):
        ok = ~np.isnan(dose) & complete
        pi, ti = np.nonzero(ok)
        feats = [np.ones(len(pi)), np.full(len(pi), float(k)), weekend[ti], base[pi]] + [cols[m][pi, ti] for m in names]
        rows_X.append(np.column_stack(feats))
        rows_y.append(dose[pi, ti])
        rows_g.append(pi)
    X, y, g = np.vstack(rows_X), np.concatenate(rows_y), np.concatenate(rows_g)
    labels = ["intercept", "evening_dose", "weekend", "patient_baseline"] + names
    fit = stats.logit_cluster(X, y, g, labels)
    testable = [i for i, nm in enumerate(labels) if nm not in ("intercept", "patient_baseline")]
    q = stats.bh(fit.p[testable])
    table = []
    for j, i in enumerate(testable):
        nm = labels[i]
        lo, hi = fit.ci(nm)
        table.append({"name": nm, "coef": float(fit.coef[i]), "ci95": (lo, hi), "p": float(fit.p[i]), "q": float(q[j]),
                      "flagged": bool(q[j] < cfg.alpha)})
    by = {r["name"]: r for r in table}
    caveat = ("A link, adjusted for the other measures, the weekday and the person's own baseline. It shows what goes with "
              "what; it does not prove one thing causes the other.")
    findings = {}
    for fid, nm in (("night_wakes", "night_wakes"), ("evening", "evening_dose"), ("weekend", "weekend")):
        r = by[nm]
        findings[fid] = Finding(fid, r["coef"], r["ci95"], r["q"], int(fit.n),
                                "Logistic regression of each dose's on-time answer on every candidate at once; standard "
                                "errors clustered by patient; Benjamini-Hochberg across candidates", caveat,
                                {"odds_ratio_on_time": float(np.exp(r["coef"]))})
    return table, findings


def naive_screen(data: Dataset, cfg: Config) -> list[dict]:
    """One correlation at a time, every patient-day treated as independent. Here to be beaten, not believed."""
    adh = data.adherence
    rows, ps = [], []
    for m in _candidate_names(data):
        x = data.metrics[m]
        ok = ~np.isnan(x) & ~np.isnan(adh)
        r, p = sps.pearsonr(x[ok], adh[ok])
        rows.append({"name": m, "r": float(r), "p": float(p)})
        ps.append(p)
    q = stats.bh(ps)
    for row, qq in zip(rows, q):
        row["q"] = float(qq)
        row["flagged"] = bool(qq < cfg.alpha)
    return rows


# --------------------------------------------------------------------------------------------------------------
def analyze(data: Dataset, cfg: Config | None = None) -> Analysis:
    """Everything, from the data alone."""
    cfg = cfg or Config()
    if not isinstance(data, Dataset):
        raise TypeError("analyze() takes a Dataset, and only a Dataset")
    rising, change, f_rise = rising_questions(data, cfg)
    f_lag, lags, curve = lead_lag(data, cfg)
    lagged = f_lag.estimate is not None and f_lag.q < cfg.alpha and f_lag.ci95[0] > 0
    drops, day, size, f_drop = abrupt_drops(data, cfg, int(f_lag.estimate) if lagged else None)
    table, f_screen = screen(data, cfg)
    findings = {"rising_questions": f_rise, "lead_lag": f_lag, "abrupt_drop": f_drop, **f_screen}
    return Analysis(findings=findings, rising=rising, rising_change=change, drops=drops, drop_day=day, drop_size=size,
                    screen=table, naive=naive_screen(data, cfg), leadlag_lags=lags, leadlag_curve=curve, config=cfg)
