"""From statistics to sentences a caregiver can read, without adding a single number the analysis did not compute.

    cohort_insights(result, data)        what holds across the group, with its caveat
    patient_signals(result, data, i)     what stands out for one person, worth raising with their doctor

Every sentence is built from a Finding or from the data itself; nothing is written by hand around a number. Words that
sound like a diagnosis are not used (a test scans for them), and every list ends with DISCLAIMER.

THE PEOPLE IN THIS DATA DO NOT EXIST. Whoever shows these sentences must say so (SYNTHETIC_BADGE).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .findings import Analysis
from .stats import fill_gaps
from .synth import Dataset

DISCLAIMER = ("These are patterns in the records, not a diagnosis. They are worth mentioning to the doctor; they do not "
              "say what is wrong, or why.")
SYNTHETIC_BADGE = "Synthetic data: invented people, generated for a demonstration. Nobody real is described."

# Words that would turn a pattern into a claim about the person's health. None may appear in an insight or a signal
# (except in the phrase "not a diagnosis", which is the point).
NOT_ALLOWED = ("diagnos", "dementia", "alzheimer", "worsening", "deteriorat", "disease", "cognitive decline", "you have")


def problem_words(text: str) -> list[str]:
    """Which words in `text` would turn a pattern into a claim about someone's health. "not a diagnosis" is fine."""
    plain = text.lower().replace("not a diagnosis", "")
    return [w for w in NOT_ALLOWED if w in plain]


@dataclass
class Insight:
    id: str
    headline: str
    text: str
    evidence: dict
    caveat: str


@dataclass
class Signal:
    kind: str
    text: str
    evidence: dict = field(default_factory=dict)


def _odds_words(coef: float) -> str:
    """0.72 -> "about 28% lower odds"; 1.4 -> "about 40% higher odds"."""
    ratio = float(np.exp(coef))
    return f"about {abs(1 - ratio):.0%} {'lower' if ratio < 1 else 'higher'} odds"


def _rate(a: np.ndarray) -> float:
    return float(np.nanmean(a))


def cohort_insights(result: Analysis, data: Dataset) -> list[Insight]:
    f = result.findings
    out: list[Insight] = []
    lag = f["lead_lag"]
    if lag.q is not None and lag.q < 0.05 and lag.ci95[0] > 0:
        out.append(Insight("lead_lag", "Questions change first",
                           f"Across {lag.n} people, repeated questions tend to change about {lag.estimate:.0f} days before on-time doses "
                           f"do (95% interval {lag.ci95[0]:.0f} to {lag.ci95[1]:.0f} days).",
                           {"days": lag.estimate, "ci95": lag.ci95, "corrected_p": lag.q, "people": lag.n}, lag.caveat))
    nw = f["night_wakes"]
    if nw.q < 0.05:
        out.append(Insight("night_wakes", "Nights and the next day's doses",
                           f"Each extra time someone is up in the night goes with {_odds_words(nw.estimate)} of answering that day's "
                           f"dose on time (odds ratio {np.exp(nw.estimate):.2f}, 95% interval {np.exp(nw.ci95[0]):.2f} to "
                           f"{np.exp(nw.ci95[1]):.2f}).",
                           {"odds_ratio": float(np.exp(nw.estimate)), "ci95_or": (float(np.exp(nw.ci95[0])), float(np.exp(nw.ci95[1]))),
                            "corrected_p": nw.q}, nw.caveat))
    ev = f["evening"]
    if ev.q < 0.05:
        am, pm = _rate(data.dose_am), _rate(data.dose_pm)
        out.append(Insight("evening", "Evening doses",
                           f"Evening doses are answered on time less often than morning ones: {pm:.0%} against {am:.0%} of recorded doses "
                           f"(odds ratio {np.exp(ev.estimate):.2f}, after adjusting for everything else measured).",
                           {"evening": pm, "morning": am, "odds_ratio": float(np.exp(ev.estimate)), "corrected_p": ev.q}, ev.caveat))
    we = f["weekend"]
    if we.q < 0.05:
        wk = (data.dow >= 5)
        end = _rate(np.stack([data.dose_am[:, wk], data.dose_pm[:, wk]]))
        mid = _rate(np.stack([data.dose_am[:, ~wk], data.dose_pm[:, ~wk]]))
        out.append(Insight("weekend", "Weekends",
                           f"Weekend doses are answered on time a little less often: {end:.0%} against {mid:.0%} on weekdays "
                           f"(odds ratio {np.exp(we.estimate):.2f}).",
                           {"weekend": end, "weekday": mid, "odds_ratio": float(np.exp(we.estimate)), "corrected_p": we.q}, we.caveat))
    k = int(result.rising.sum())
    if k:
        weeks = data.days // 7
        med = float(np.median(result.rising_change[result.rising]))
        out.append(Insight("rising_questions", "Steady rises in repeated questions",
                           f"{k} of {data.n} people ({k / data.n:.0%}) show a steady rise in repeated questions: a median of {med:+.1f} "
                           f"more a day than {weeks} weeks earlier.",
                           {"people": k, "of": data.n, "median_change_per_day": med}, f["rising_questions"].caveat))
    d = int(result.drops.sum())
    if d:
        out.append(Insight("abrupt_drop", "Abrupt, lasting drops in on-time doses",
                           f"{d} of {data.n} people show an abrupt, lasting drop in on-time doses (a median of "
                           f"{abs(float(np.median(result.drop_size[result.drops]))):.0%} of doses). This finds only the clearest ones; "
                           f"a noisy record hides many more.",
                           {"people": d, "of": data.n}, f["abrupt_drop"].caveat))
    # what looked linked one at a time, and did not hold up
    careful = {r["name"] for r in result.screen if r["flagged"]}
    fooled = [r["name"] for r in result.naive if r["flagged"] and r["name"] not in careful]
    if fooled:
        shown = ", ".join(n.replace("_", " ") for n in fooled[:4]) + (f" and {len(fooled) - 4} more" if len(fooled) > 4 else "")
        out.append(Insight("did_not_hold_up", "What did not hold up",
                           f"Checked one at a time, {len(fooled)} measures looked linked to missed doses ({shown}). None held up once "
                           f"the weekday, the other measures and each person's own habits were taken into account.",
                           {"looked_linked": fooled}, "Looking at measures one at a time, and counting every day as a separate "
                                                       "person, finds links that are not there."))
    return out


def patient_signals(result: Analysis, data: Dataset, i: int) -> list[Signal]:
    """What stands out for person `i`. Only things the analysis flagged, plus the recent-weeks comparison."""
    out: list[Signal] = []
    if result.rising[i]:
        out.append(Signal("rising_questions", f"Repeated questions have been rising steadily: about {result.rising_change[i]:+.1f} more a "
                                                f"day than {data.days // 7} weeks ago.", {"change_per_day": float(result.rising_change[i])}))
    if result.drops[i]:
        day = int(result.drop_day[i])
        a = fill_gaps(data.adherence[i:i + 1])[0]
        before, after = float(a[:day].mean()), float(a[day:].mean())
        out.append(Signal("abrupt_drop", f"On-time doses dropped abruptly around day {day}: {before:.0%} before, {after:.0%} since. "
                                          f"Worth asking what changed that week.", {"day": day, "before": before, "after": after}))
    a = data.adherence[i]
    recent, prior = a[-28:], a[-56:-28]
    if np.isfinite(recent).sum() >= 20 and np.isfinite(prior).sum() >= 20:
        r, p = float(np.nanmean(recent)), float(np.nanmean(prior))
        if p - r >= 0.15:
            out.append(Signal("recent_change", f"On-time doses over the last four weeks: {r:.0%}, against {p:.0%} in the four weeks before.",
                              {"recent": r, "prior": p}))
    return out
