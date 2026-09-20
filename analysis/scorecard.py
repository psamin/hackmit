"""Grade the analysis against the answer key.

    score(result, key)                        one dataset: which planted patterns came back, which decoys were fooled
    evaluate(seeds, world)                    many datasets: how often, with honest uncertainty on those rates
    false_alarms(result)                      for the NULL world, where nothing is real: everything flagged is wrong

"Recovered" means what the key says it means (see synth.py: Pattern.tolerance). It is the same yardstick for every
seed. Nothing here changes how the analysis runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .findings import Analysis, Config, analyze
from .synth import AnswerKey, World, generate


# The seeds the reported numbers come from. Settings were chosen on findings.DEV_SEEDS. Seeds 1-20 were used once to
# look at results, a weakness was found (see README, "what we changed and why"), and the code was changed; so the
# numbers we report use these FRESH seeds, which nothing was tuned on.
EVAL_SEEDS = tuple(range(201, 221))
NULL_SEEDS = tuple(range(301, 321))


@dataclass
class Check:
    id: str
    what: str
    recovered: bool
    detail: str
    numbers: dict = field(default_factory=dict)


@dataclass
class Score:
    seed: int
    checks: list[Check]
    decoys_flagged: list[str]              # by the careful analysis
    naive_decoys_flagged: list[str]        # by the naive one
    n_decoys: int

    @property
    def recovered(self) -> int:
        return sum(c.recovered for c in self.checks)


def _pr(flagged: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    tp = int((flagged & truth).sum())
    precision = tp / flagged.sum() if flagged.sum() else 1.0        # flagging nothing is never "wrong", only unhelpful
    recall = tp / truth.sum() if truth.sum() else 1.0
    return float(precision), float(recall)


def _rate(finding, truth: float, tol: float = 0.35) -> tuple[bool, str]:
    close = abs(finding.estimate - truth) <= tol * abs(truth)          # within 35% of the truth, which also means the right direction
    sig = finding.q is not None and finding.q < 0.05
    ok = bool(close and sig)
    return ok, f"estimate {finding.estimate:+.2f} vs truth {truth:+.2f}, corrected p {finding.q:.3g}"


def score(result: Analysis, key: AnswerKey, seed: int = -1) -> Score:
    checks: list[Check] = []
    f = result.findings
    for p in key.patterns:
        if p.id == "rising_questions":
            prec, rec = _pr(result.rising, key.decliner)
            checks.append(Check(p.id, p.what, prec >= 0.8 and rec >= 0.6,
                                f"flagged {int(result.rising.sum())} patients; precision {prec:.0%}, recall {rec:.0%}",
                                {"precision": prec, "recall": rec}))
        elif p.id == "lead_lag":
            lo, hi = f["lead_lag"].ci95
            est = f["lead_lag"].estimate
            ok = bool(lo <= p.truth <= hi and lo > 0 and f["lead_lag"].q < 0.05)
            checks.append(Check(p.id, p.what, ok, f"questions lead by {est:.0f} days (95% interval {lo:.0f} to {hi:.0f}); truth {p.truth}",
                                {"estimate": est, "lo": lo, "hi": hi}))
        elif p.id in ("night_wakes", "evening", "weekend"):
            ok, detail = _rate(f[p.id], p.truth)
            checks.append(Check(p.id, p.what, ok, detail, {"estimate": f[p.id].estimate}))
        elif p.id == "abrupt_drop":
            truth = key.step_day >= 0
            prec, rec = _pr(result.drops, truth)
            both = result.drops & truth
            err = float(np.median(np.abs(result.drop_day[both] - key.step_day[both]))) if both.any() else float("inf")
            ok = prec >= 0.8 and rec >= 0.2 and err <= 7
            checks.append(Check(p.id, p.what, ok, f"flagged {int(result.drops.sum())}; precision {prec:.0%}, recall {rec:.0%}, "
                                                  f"median date error {err:.0f} days", {"precision": prec, "recall": rec, "date_error": err}))
    decoys = set(key.decoys)
    return Score(seed=seed, checks=checks,
                 decoys_flagged=[r["name"] for r in result.screen if r["flagged"] and r["name"] in decoys],
                 naive_decoys_flagged=[r["name"] for r in result.naive if r["flagged"] and r["name"] in decoys],
                 n_decoys=len(decoys))


def false_alarms(result: Analysis) -> list[str]:
    """For a world where NOTHING is real: everything the analysis calls a finding is a false alarm."""
    out = [f"covariate:{r['name']}" for r in result.screen if r["flagged"]]
    out += [f"naive:{r['name']}" for r in result.naive if r["flagged"]]
    if result.rising.any():
        out.append(f"rising:{int(result.rising.sum())} patients")
    if result.drops.any():
        out.append(f"drop:{int(result.drops.sum())} patients")
    lag = result.findings["lead_lag"]
    if lag.q is not None and lag.q < 0.05 and lag.ci95[0] > 0:
        out.append("lead_lag")
    return out


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval for a share, so 'recovered 19 of 20' is not reported as if it were exact."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


@dataclass
class Evaluation:
    seeds: list[int]
    scores: list[Score]
    null_alarms: dict[int, list[str]]      # seed -> false alarms in the null world

    def recovery(self) -> dict[str, tuple[int, int]]:
        out: dict[str, list[int]] = {}
        for s in self.scores:
            for c in s.checks:
                out.setdefault(c.id, []).append(int(c.recovered))
        return {k: (sum(v), len(v)) for k, v in out.items()}

    def decoy_rates(self) -> dict[str, float]:
        n = max(len(self.scores), 1)
        d = self.scores[0].n_decoys if self.scores else 0
        return {"careful_false_flags_per_dataset": sum(len(s.decoys_flagged) for s in self.scores) / n,
                "naive_false_flags_per_dataset": sum(len(s.naive_decoys_flagged) for s in self.scores) / n,
                "decoys_per_dataset": d}

    def null_rate(self, naive: bool = False) -> tuple[int, int]:
        """Null datasets with at least one false alarm (from the careful analysis, or the naive one), out of all of them."""
        hit = lambda v: any(a.startswith("naive:") == naive for a in v)
        return sum(hit(v) for v in self.null_alarms.values()), len(self.null_alarms)


def evaluate(seeds, null_seeds=(), cfg: Config | None = None, world: World | None = None) -> Evaluation:
    """Generate, analyse blind, and grade, once per seed. `null_seeds` run in the world where nothing is real."""
    scores = []
    for sd in seeds:
        data, key = generate(sd, world)
        scores.append(score(analyze(data, cfg), key, sd))
    nulls = {}
    for sd in null_seeds:
        data, _ = generate(sd, World.null())
        nulls[sd] = false_alarms(analyze(data, cfg))
    return Evaluation(list(seeds), scores, nulls)
