"""Run it from the repo root:

    python -m analysis.report                      one synthetic cohort: what the analysis finds, in plain words
    python -m analysis.report --seed 12            a different cohort
    python -m analysis.report --evaluate           the scorecard: many cohorts, graded against the answer key
    python -m analysis.report --evaluate --write-results     ...and save it as analysis/RESULTS.md

Everything it writes goes to analysis/out/ (not committed), and every file says it is synthetic.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import date
from pathlib import Path

import numpy as np

from . import insights
from .findings import DEV_SEEDS, Config, analyze
from .scorecard import EVAL_SEEDS, NULL_SEEDS, Evaluation, evaluate, false_alarms, score, wilson
from .synth import SYNTHETIC_NOTICE, generate

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"


def cohort_report(seed: int, out: Path = OUT) -> str:
    data, key = generate(seed)
    result = analyze(data)
    sc = score(result, key, seed)
    lines = [f"# What the analysis found in one synthetic cohort (seed {seed})", "", f"**{insights.SYNTHETIC_BADGE}**", ""]
    for ins in insights.cohort_insights(result, data):
        lines += [f"## {ins.headline}", ins.text, f"*Caveat: {ins.caveat}*", ""]
    lines += ["## Checked against the answer key", ""]
    for c in sc.checks:
        lines.append(f"- {'recovered' if c.recovered else 'MISSED'}: {c.what}. {c.detail}")
    lines += ["", f"Decoys wrongly called real: careful analysis {len(sc.decoys_flagged)} of {sc.n_decoys}, "
                  f"naive analysis {len(sc.naive_decoys_flagged)} of {sc.n_decoys}.", ""]
    strongest = int(np.argmax(result.rising_change * result.rising + result.drops * 10))
    sig = insights.patient_signals(result, data, strongest)
    lines += [f"## An example person ({data.patient_ids[strongest]}, invented)", ""]
    lines += [f"- {s.text}" for s in sig] or ["- Nothing stands out."]
    lines += ["", f"_{insights.DISCLAIMER}_", ""]
    out.mkdir(parents=True, exist_ok=True)
    data.to_csv(out / "synthetic_data.csv")
    (out / "answer_key.json").write_text(json.dumps({"notice": SYNTHETIC_NOTICE, **key.to_dict()}, indent=2), encoding="utf-8")
    (out / "findings.json").write_text(json.dumps({"notice": SYNTHETIC_NOTICE, "findings": {
        k: {"estimate": v.estimate, "ci95": v.ci95, "corrected_p": v.q, "n": v.n, "method": v.method, "caveat": v.caveat}
        for k, v in result.findings.items()}, "covariates_fitted_together": result.screen, "one_at_a_time": result.naive},
        indent=2, default=float), encoding="utf-8")
    text = "\n".join(lines)
    (out / "report.md").write_text(text, encoding="utf-8")
    return text


def scoreboard(ev: Evaluation) -> str:
    n = len(ev.scores)
    rec = ev.recovery()
    rows = ["| Pattern planted in the data | Recovered | 95% interval |", "|---|---|---|"]
    for pid, (a, b) in rec.items():
        lo, hi = wilson(a, b)
        rows.append(f"| {pid.replace('_', ' ')} | {a} of {b} | {lo:.0%} to {hi:.0%} |")
    dr = ev.decoy_rates()
    a, b = ev.null_rate()
    na, nb = ev.null_rate(naive=True)
    lines = [f"Graded on {n} fresh datasets (seeds {ev.seeds[0]}-{ev.seeds[-1]}), each with {int(dr['decoys_per_dataset'])} decoy measures; "
             f"and {b} datasets where nothing is real (seeds {min(ev.null_alarms)}-{max(ev.null_alarms)}).", "", *rows, "",
             f"- Decoys wrongly called real, per dataset: **{dr['careful_false_flags_per_dataset']:.2f}** for the careful analysis, "
             f"**{dr['naive_false_flags_per_dataset']:.2f}** for the naive one (one measure at a time, every day counted as a person).",
             f"- Datasets where nothing is real but the analysis still reported something: **{a} of {b}** careful, **{na} of {nb}** naive."]
    return "\n".join(lines)


def results_markdown(ev: Evaluation) -> str:
    return "\n".join([
        "# Results", "", f"_Generated {date.today().isoformat()} by `python -m analysis.report --evaluate --write-results`. "
        f"{SYNTHETIC_NOTICE}_", "", scoreboard(ev), "",
        "Settings were chosen on seeds " + ", ".join(map(str, DEV_SEEDS)) + ", which are not used above. See README.md for what was "
        "changed after the first look and why.", ""])


def main(argv=None) -> int:
    warnings.simplefilter("ignore", RuntimeWarning)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--evaluate", action="store_true", help="grade the analysis on many datasets against the answer key")
    ap.add_argument("--write-results", action="store_true", help="with --evaluate: save analysis/RESULTS.md")
    args = ap.parse_args(argv)
    if args.evaluate:
        ev = evaluate(EVAL_SEEDS, null_seeds=NULL_SEEDS)
        print(scoreboard(ev))
        if args.write_results:
            (HERE / "RESULTS.md").write_text(results_markdown(ev), encoding="utf-8")
            print(f"\nwrote {HERE / 'RESULTS.md'}")
        return 0
    print(cohort_report(args.seed))
    print(f"(files written to {OUT})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
