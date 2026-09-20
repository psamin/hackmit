"""Tests for the synthetic-data analysis.

    python analysis/test_analysis.py            (from the repo root)

Three kinds of test:

  - the statistics are checked against answers we know (a hand-worked p-value adjustment, a regression on data
    whose coefficients we chose, a cross-correlation of a series with a shifted copy of itself)
  - the setup is fair: the analysis is only ever handed `data`, the key is separate, and nothing in `data` gives the
    answer away
  - the analysis behaves on fixed seeds: it finds what was planted and stays quiet where nothing is. These are smoke
    tests on ONE small world; the real rates, over many datasets, are in RESULTS.md and come from `--evaluate`.
"""
import dataclasses
import json
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis import findings, insights, report, scorecard, stats, synth  # noqa: E402

warnings.simplefilter("ignore", RuntimeWarning)
FAST = findings.Config(n_boot=80, n_perm=80, step_perms=12)
SMALL = synth.World(n_patients=160, days=180)


def planted(seed=3):
    data, key = synth.generate(seed, SMALL)
    return data, key, findings.analyze(data, FAST)


# ==================================================================================================================
class Statistics(unittest.TestCase):
    def test_benjamini_hochberg_matches_a_hand_worked_example(self):
        # p = .01 .02 .03 .5, m = 4: raw .04 .04 .04 .5, then made non-decreasing from the right
        np.testing.assert_allclose(stats.bh([0.01, 0.02, 0.03, 0.5]), [0.04, 0.04, 0.04, 0.5])

    def test_bh_never_lowers_a_p_value_or_exceeds_one_and_keeps_the_order(self):
        rng = np.random.default_rng(0)
        p = rng.random(200) ** 3
        q = stats.bh(p)
        self.assertTrue(np.all(q >= p - 1e-12) and np.all(q <= 1))
        order = np.argsort(p)
        self.assertTrue(np.all(np.diff(q[order]) >= -1e-12))
        self.assertEqual(stats.bh([]).size, 0)

    def test_bh_on_pure_noise_flags_almost_nothing(self):
        flagged = [(stats.bh(np.random.default_rng(s).random(300)) < 0.05).any() for s in range(200)]
        self.assertLess(np.mean(flagged), 0.12)                                 # about 5%, with room for 200 tries

    def test_trend_finds_a_rise_and_a_flat_line_and_survives_gaps(self):
        t = np.arange(40.0)
        tau, p, slope = stats.trend(t * 0.5 + np.random.default_rng(0).normal(0, 1, 40))
        self.assertGreater(tau, 0.8)
        self.assertLess(p, 1e-6)
        self.assertAlmostEqual(slope, 0.5, delta=0.1)
        self.assertEqual(stats.trend(np.ones(40)), (0.0, 1.0, 0.0))
        gappy = t.copy()
        gappy[::3] = np.nan
        self.assertGreater(stats.trend(gappy)[0], 0.9)
        self.assertEqual(stats.trend(np.array([1.0, 2.0]))[1], 1.0)              # too little to say anything

    def test_the_regression_recovers_coefficients_we_chose(self):
        rng = np.random.default_rng(1)
        g = np.repeat(np.arange(300), 40)
        x1, x2 = rng.normal(size=g.size), rng.normal(size=g.size)
        eta = 0.4 + 0.8 * x1 - 0.5 * x2
        y = (rng.random(g.size) < 1 / (1 + np.exp(-eta))).astype(float)
        fit = stats.logit_cluster(np.column_stack([np.ones(g.size), x1, x2]), y, g, ["c", "x1", "x2"])
        for name, truth in (("c", 0.4), ("x1", 0.8), ("x2", -0.5)):
            lo, hi = fit.ci(name)
            self.assertTrue(lo < truth < hi, (name, truth, lo, hi))
        self.assertEqual((fit.n, fit.groups), (g.size, 300))

    def test_a_regression_agrees_with_a_known_closed_form(self):
        # one binary predictor: the coefficient is exactly the log odds ratio of the 2x2 table
        rng = np.random.default_rng(2)
        x = rng.integers(0, 2, 4000).astype(float)
        y = (rng.random(4000) < np.where(x == 1, 0.7, 0.4)).astype(float)
        a, b = y[x == 1].sum(), (1 - y[x == 1]).sum()
        c, d = y[x == 0].sum(), (1 - y[x == 0]).sum()
        fit = stats.logit_cluster(np.column_stack([np.ones(4000), x]), y, np.arange(4000), ["c", "x"])
        self.assertAlmostEqual(fit.coef[1], np.log((a / b) / (c / d)), places=6)

    def test_clustered_errors_are_wider_when_each_patient_repeats_themselves(self):
        # 200 patients with 60 days each. Each has their own habit (how often they answer) and their own level of x, and
        # the two are unrelated. Counting every day as a separate person makes that look far more certain than it is.
        rng = np.random.default_rng(3)
        g = np.repeat(np.arange(200), 60)
        x = np.repeat(rng.normal(size=200), 60) + rng.normal(0, 0.3, g.size)
        y = (rng.random(g.size) < 1 / (1 + np.exp(-np.repeat(rng.normal(0, 1.5, 200), 60)))).astype(float)
        X = np.column_stack([np.ones(g.size), x])
        clustered = stats.logit_cluster(X, y, g, ["c", "x"]).se[1]
        as_if_independent = stats.logit_cluster(X, y, np.arange(g.size), ["c", "x"]).se[1]
        self.assertGreater(clustered, 2 * as_if_independent)

    def test_crosscorr_finds_a_known_shift_and_its_direction(self):
        rng = np.random.default_rng(4)
        x = stats.smooth(rng.normal(size=(50, 200)), 5)
        y = np.roll(x, 9, axis=1)                                                # y is x, 9 days later
        lags = range(-20, 21)
        best = list(lags)[int(np.argmax(stats.crosscorr(x, y, lags).mean(axis=0)))]
        self.assertEqual(best, 9)                                                # positive: x comes first

    def test_smoothing_detrending_and_gap_filling(self):
        a = np.arange(20.0)[None, :] * 2 + 5
        d = stats.detrend(a)
        np.testing.assert_allclose(d, 0, atol=1e-9)                              # a straight line has nothing left
        self.assertEqual(stats.smooth(np.ones((1, 10)), 4).round(6).tolist(), [[1.0] * 10])
        gappy = np.array([[1.0, np.nan, 3.0, np.nan, 5.0]])
        np.testing.assert_allclose(stats.fill_gaps(gappy), [[1, 2, 3, 4, 5]])
        np.testing.assert_allclose(stats.fill_gaps(np.array([[np.nan, np.nan]])), [[0, 0]])


# ==================================================================================================================
class TheWorldIsFairAndSynthetic(unittest.TestCase):
    def test_the_same_seed_gives_the_same_world_and_another_seed_a_different_one(self):
        a, ka = synth.generate(5, SMALL)
        b, kb = synth.generate(5, SMALL)
        c, _ = synth.generate(6, SMALL)
        np.testing.assert_array_equal(np.nan_to_num(a.dose_pm), np.nan_to_num(b.dose_pm))
        np.testing.assert_array_equal(ka.step_day, kb.step_day)
        self.assertFalse(np.array_equal(np.nan_to_num(a.dose_pm), np.nan_to_num(c.dose_pm)))

    def test_the_data_says_it_is_synthetic_everywhere_it_goes(self):
        data, key = synth.generate(1, SMALL)
        self.assertTrue(data.synthetic)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "d.csv"
            data.to_csv(path)
            self.assertTrue(path.read_text(encoding="utf-8").startswith("# SYNTHETIC DATA"))
            import pandas as pd
            df = pd.read_csv(path, comment="#")
            self.assertEqual(len(df), data.n * data.days)
            self.assertIn("dose_pm", df.columns)
        self.assertIn("SYNTHETIC", synth.SYNTHETIC_NOTICE)
        self.assertIn("Synthetic", insights.SYNTHETIC_BADGE)

    def test_nothing_in_the_data_object_gives_the_answer_away(self):
        fields = {f.name for f in dataclasses.fields(synth.Dataset)}
        for giveaway in ("decliner", "ramp", "step", "truth", "key", "cohort", "planted", "world"):
            self.assertFalse(any(giveaway in f for f in fields), (giveaway, fields))
        data, key = synth.generate(1, SMALL)
        for name in data.metrics:                                                # the metrics are named for what they are,
            self.assertNotIn("truth", name)                                      # and the decoys are not marked as decoys
            self.assertNotIn("decoy", name)

    def test_the_analysis_takes_a_dataset_and_nothing_else(self):
        data, key = synth.generate(1, SMALL)
        for wrong in (key, (data, key), {"data": data}, None):
            with self.assertRaises(TypeError):
                findings.analyze(wrong, FAST)

    def test_the_analysis_gives_the_same_answer_whatever_the_key_says(self):
        data, key = synth.generate(2, SMALL)
        first = findings.analyze(data, FAST)
        key.decliner[:] = ~key.decliner                                          # scramble the key: the analysis cannot care
        key.step_day[:] = -1
        second = findings.analyze(data, FAST)
        np.testing.assert_array_equal(first.rising, second.rising)
        self.assertEqual(first.findings["lead_lag"].estimate, second.findings["lead_lag"].estimate)

    def test_the_planted_patterns_are_really_in_the_data_before_any_analysis(self):
        # checked from the data directly, with no help from the analysis: the planted effects exist
        data, key = synth.generate(4, SMALL)
        wk = data.dow >= 5
        weekend = np.nanmean(np.stack([data.dose_am[:, wk], data.dose_pm[:, wk]]))
        weekday = np.nanmean(np.stack([data.dose_am[:, ~wk], data.dose_pm[:, ~wk]]))
        self.assertLess(weekend, weekday)
        self.assertLess(np.nanmean(data.dose_pm), np.nanmean(data.dose_am))
        q = data.metrics["repeated_questions"]
        late = np.nanmean(q[key.decliner][:, -28:])
        early = np.nanmean(q[key.decliner][:, :28])
        self.assertGreater(late, early + 1.0)                                    # the sliders' questions really rise
        self.assertAlmostEqual(np.nanmean(q[~key.decliner][:, -28:]), np.nanmean(q[~key.decliner][:, :28]), delta=0.8)

    def test_the_confounded_decoy_really_is_correlated_with_adherence_only_through_the_weekend(self):
        # The link is faint (a few points of adherence between weekdays and weekends), so use 600 patients and five worlds.
        raw, partial = [], []
        for seed in range(4, 9):
            data, _ = synth.generate(seed, synth.World(n_patients=600))
            s, a = data.metrics["social_minutes"], data.adherence
            wk = np.broadcast_to((data.dow >= 5)[None, :], s.shape)
            ok = ~np.isnan(s) & ~np.isnan(a)

            def without_weekday(v):
                v, w = v[ok], wk[ok]
                return v - np.where(w, v[w].mean(), v[~w].mean())
            raw.append(np.corrcoef(s[ok], a[ok])[0, 1])
            partial.append(np.corrcoef(without_weekday(s), without_weekday(a))[0, 1])
        self.assertLess(np.mean(raw), -0.012)                                    # a real-looking link on its face ...
        self.assertLess(abs(np.mean(partial)), 0.4 * abs(np.mean(raw)))          # ... that is mostly gone once weekday is removed

    def test_the_null_world_has_no_patterns_and_no_slides_or_steps(self):
        data, key = synth.generate(4, synth.World.null(n_patients=100))
        self.assertEqual((key.patterns, int(key.decliner.sum()), int((key.step_day >= 0).sum())), ([], 0, 0))
        self.assertFalse(key.world.planted)
        self.assertTrue(synth.World().planted)

    def test_the_answer_key_describes_every_decoy_and_serialises(self):
        _, key = synth.generate(1, SMALL)
        self.assertEqual(len(key.decoys), 3 + SMALL.n_noise)
        json.dumps(key.to_dict())
        self.assertEqual([p.id for p in key.patterns],
                         ["rising_questions", "lead_lag", "night_wakes", "evening", "weekend", "abrupt_drop"])


# ==================================================================================================================
class TheAnalysisOnOneSmallWorld(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, cls.key, cls.result = planted(3)
        cls.score = scorecard.score(cls.result, cls.key, 3)

    def test_it_finds_that_questions_lead_doses_by_about_two_weeks(self):
        f = self.result.findings["lead_lag"]
        self.assertAlmostEqual(f.estimate, 14, delta=3)                          # the estimate itself, not only its interval
        self.assertTrue(f.ci95[0] <= 14 <= f.ci95[1], f.ci95)
        self.assertGreater(f.ci95[0], 0)                                         # positive: questions come first
        self.assertLess(f.q, 0.05)
        self.assertLess(f.detail["peak_correlation"], 0)                         # questions up, doses down

    def test_it_recovers_the_three_effects_on_a_days_doses_in_the_right_direction_and_size(self):
        for pid in ("night_wakes", "evening", "weekend"):
            truth = next(p.truth for p in self.key.patterns if p.id == pid)
            f = self.result.findings[pid]
            self.assertLess(f.q, 0.05, pid)
            self.assertEqual(np.sign(f.estimate), np.sign(truth), pid)
            self.assertLess(abs(f.estimate - truth), 0.35 * abs(truth), (pid, f.estimate, truth))
            self.assertLess(f.ci95[1], 0, pid)                                   # the whole interval is on the right side of zero

    def test_the_scorecard_agrees_it_recovered_these_and_the_slides(self):
        got = {c.id: c.recovered for c in self.score.checks}
        for pid in ("lead_lag", "night_wakes", "evening", "weekend", "rising_questions"):
            self.assertTrue(got[pid], (pid, [c.detail for c in self.score.checks if c.id == pid]))

    def test_what_it_flags_as_a_drop_or_a_slide_is_mostly_right(self):
        drops_truth = self.key.step_day >= 0
        self.assertGreaterEqual((self.result.drops & drops_truth).sum(), 0.8 * max(self.result.drops.sum(), 1))
        self.assertGreaterEqual((self.result.rising & self.key.decliner).sum(), 0.8 * self.result.rising.sum())
        both = self.result.drops & drops_truth
        if both.any():
            self.assertLessEqual(np.median(np.abs(self.result.drop_day[both] - self.key.step_day[both])), 10)

    def test_fitting_everything_together_beats_one_at_a_time_on_the_decoys(self):
        careful, naive = len(self.score.decoys_flagged), len(self.score.naive_decoys_flagged)
        self.assertLessEqual(careful, 2)
        self.assertGreaterEqual(naive, careful + 2)                              # the naive version is fooled a lot more

    def test_the_confounded_and_the_mediated_decoys_are_not_called_real(self):
        by = {r["name"]: r for r in self.result.screen}
        self.assertFalse(by["social_minutes"]["flagged"])
        self.assertFalse(by["sleep_hours"]["flagged"])
        naive = {r["name"]: r for r in self.result.naive}
        self.assertTrue(naive["social_minutes"]["flagged"] or naive["sleep_hours"]["flagged"])   # ... though one at a time they look real

    def test_only_days_where_every_measure_was_recorded_are_used_in_the_fit(self):
        # filling a gap with an average lets a measure that IS recorded (sleep) stand in for one that is not (wakes)
        complete = np.ones((self.data.n, self.data.days), dtype=bool)
        for name, values in self.data.metrics.items():
            if name != "repeated_questions":
                complete &= ~np.isnan(values)
        expected = int(((~np.isnan(self.data.dose_am)) & complete).sum() + ((~np.isnan(self.data.dose_pm)) & complete).sum())
        self.assertEqual(self.result.findings["night_wakes"].n, expected)
        self.assertLess(expected, int((~np.isnan(self.data.dose_am)).sum() + (~np.isnan(self.data.dose_pm)).sum()))

    def test_every_finding_carries_a_method_and_a_caveat(self):
        for f in self.result.findings.values():
            self.assertTrue(f.method and f.caveat, f.id)
        self.assertEqual(set(self.result.findings), {"rising_questions", "lead_lag", "abrupt_drop", "night_wakes", "evening", "weekend"})

    def test_it_is_repeatable(self):
        again = findings.analyze(self.data, FAST)
        self.assertEqual(again.findings["lead_lag"].estimate, self.result.findings["lead_lag"].estimate)
        np.testing.assert_array_equal(again.drops, self.result.drops)


class WhereNothingIsReal(unittest.TestCase):
    def test_it_finds_next_to_nothing(self):
        # fixed seeds, so this is repeatable; the honest false-alarm rate over many datasets is in RESULTS.md
        alarms = []
        for seed in (11, 12, 13):
            data, _ = synth.generate(seed, synth.World.null(n_patients=160))
            alarms.append([a for a in scorecard.false_alarms(findings.analyze(data, FAST)) if not a.startswith("naive:")])
        self.assertLessEqual(sum(bool(a) for a in alarms), 1, alarms)

    def test_a_lead_lag_is_not_reported_as_significant_when_there_is_none(self):
        for seed in (11, 12, 13, 14, 15):                                        # p-values here were .59 .16 .85 .67 1.0
            data, _ = synth.generate(seed, synth.World.null(n_patients=160))
            self.assertGreater(findings.lead_lag(data, FAST)[0].q, 0.05, seed)

    def test_the_naive_analysis_is_fooled_by_the_same_data(self):
        data, _ = synth.generate(11, synth.World.null(n_patients=160))
        naive = [a for a in scorecard.false_alarms(findings.analyze(data, FAST)) if a.startswith("naive:")]
        self.assertGreaterEqual(len(naive), 1)


# ==================================================================================================================
class Grading(unittest.TestCase):
    """The scorecard's own logic, on results we make up, so we know what it should say."""

    def result(self, key, **over):
        n = len(key.decliner)
        F = findings.Finding
        base = dict(
            findings={"rising_questions": F("rising_questions", 3.0, None, 0.01, n, "m", "c"),
                      "lead_lag": F("lead_lag", 14.0, (12.0, 16.0), 0.001, n, "m", "c"),
                      "abrupt_drop": F("abrupt_drop", -0.2, None, None, n, "m", "c"),
                      "night_wakes": F("night_wakes", -0.35, (-0.4, -0.3), 0.001, 10, "m", "c"),
                      "evening": F("evening", -0.45, (-0.5, -0.4), 0.001, 10, "m", "c"),
                      "weekend": F("weekend", -0.25, (-0.3, -0.2), 0.001, 10, "m", "c")},
            rising=key.decliner.copy(), rising_change=np.zeros(n), drops=key.step_day >= 0, drop_day=key.step_day.copy(),
            drop_size=np.zeros(n), screen=[], naive=[], leadlag_lags=np.array([0]), leadlag_curve=np.array([0.0]), config=FAST)
        base.update(over)
        return findings.Analysis(**base)

    def setUp(self):
        _, self.key = synth.generate(1, SMALL)

    def test_a_perfect_analysis_recovers_everything(self):
        s = scorecard.score(self.result(self.key), self.key)
        self.assertTrue(all(c.recovered for c in s.checks), [c.detail for c in s.checks if not c.recovered])

    def test_each_kind_of_failure_is_marked_as_a_miss(self):
        F = findings.Finding
        cases = {
            "night_wakes": F("night_wakes", +0.35, (0.3, 0.4), 0.001, 10, "m", "c"),           # wrong direction
            "evening": F("evening", -0.05, (-0.1, 0.0), 0.001, 10, "m", "c"),                  # far too small
            "weekend": F("weekend", -0.25, (-0.3, -0.2), 0.2, 10, "m", "c"),                   # not significant
            "lead_lag": F("lead_lag", 3.0, (1.0, 5.0), 0.001, 10, "m", "c"),                   # interval misses 14
        }
        for pid, finding in cases.items():
            res = self.result(self.key)
            res.findings[pid] = finding
            got = {c.id: c.recovered for c in scorecard.score(res, self.key).checks}
            self.assertFalse(got[pid], pid)
            self.assertTrue(got["night_wakes" if pid != "night_wakes" else "evening"], pid)   # and only that one

    def test_a_lag_interval_that_includes_zero_does_not_count(self):
        res = self.result(self.key)
        res.findings["lead_lag"] = findings.Finding("lead_lag", 14.0, (-3.0, 16.0), 0.001, 10, "m", "c")
        self.assertFalse({c.id: c.recovered for c in scorecard.score(res, self.key).checks}["lead_lag"])

    def test_flagging_the_wrong_people_or_almost_nobody_is_not_recovery(self):
        n = len(self.key.decliner)
        nobody = self.result(self.key, rising=np.zeros(n, bool), drops=np.zeros(n, bool))
        got = {c.id: c.recovered for c in scorecard.score(nobody, self.key).checks}
        self.assertFalse(got["rising_questions"])                                # zero recall
        self.assertFalse(got["abrupt_drop"])
        everyone = self.result(self.key, rising=np.ones(n, bool))
        self.assertFalse({c.id: c.recovered for c in scorecard.score(everyone, self.key).checks}["rising_questions"])   # poor precision

    def test_a_drop_dated_far_from_the_truth_is_a_miss(self):
        res = self.result(self.key, drop_day=self.key.step_day + 30)
        self.assertFalse({c.id: c.recovered for c in scorecard.score(res, self.key).checks}["abrupt_drop"])

    def test_decoys_flagged_are_counted_and_real_drivers_are_not(self):
        screen = [{"name": "noise_03", "flagged": True}, {"name": "night_wakes", "flagged": True}, {"name": "steps_k", "flagged": False}]
        naive = [{"name": "noise_03", "flagged": True}, {"name": "social_minutes", "flagged": True}]
        s = scorecard.score(self.result(self.key, screen=screen, naive=naive), self.key)
        self.assertEqual(s.decoys_flagged, ["noise_03"])
        self.assertEqual(s.naive_decoys_flagged, ["noise_03", "social_minutes"])
        self.assertEqual(s.n_decoys, len(self.key.decoys))

    def test_in_a_world_where_nothing_is_real_everything_flagged_is_a_false_alarm(self):
        res = self.result(self.key, screen=[{"name": "x", "flagged": True}], naive=[{"name": "y", "flagged": True}])
        alarms = scorecard.false_alarms(res)
        self.assertIn("covariate:x", alarms)
        self.assertIn("naive:y", alarms)
        self.assertTrue(any(a.startswith("rising") for a in alarms))
        quiet = self.result(self.key, rising=np.zeros(len(self.key.decliner), bool), drops=np.zeros(len(self.key.decliner), bool))
        quiet.findings["lead_lag"] = findings.Finding("lead_lag", 5.0, (-2.0, 9.0), 0.4, 10, "m", "c")
        self.assertEqual(scorecard.false_alarms(quiet), [])

    def test_wilson_intervals_are_wide_for_small_counts_and_never_leave_zero_to_one(self):
        lo, hi = scorecard.wilson(20, 20)
        self.assertAlmostEqual(hi, 1.0, places=6)
        self.assertLess(lo, 0.9)                                                 # 20 of 20 is not "certain"
        self.assertAlmostEqual(sum(scorecard.wilson(10, 20)) / 2, 0.5, delta=0.02)
        self.assertEqual(scorecard.wilson(0, 0), (0.0, 1.0))
        for k in range(0, 21):
            lo, hi = scorecard.wilson(k, 20)
            self.assertTrue(0 <= lo <= k / 20 <= hi <= 1)

    def test_evaluating_many_seeds_counts_them_up(self):
        ev = scorecard.evaluate([21, 22], null_seeds=[31], cfg=FAST, world=SMALL)
        self.assertEqual(len(ev.scores), 2)
        self.assertEqual(ev.recovery()["lead_lag"][1], 2)
        self.assertEqual(ev.null_rate()[1], 1)
        self.assertEqual(ev.decoy_rates()["decoys_per_dataset"], 3 + SMALL.n_noise)

    def test_the_seeds_the_settings_were_chosen_on_are_never_the_seeds_we_report(self):
        self.assertFalse(set(findings.DEV_SEEDS) & set(scorecard.EVAL_SEEDS))
        self.assertFalse(set(findings.DEV_SEEDS) & set(scorecard.NULL_SEEDS))
        self.assertFalse(set(scorecard.EVAL_SEEDS) & set(scorecard.NULL_SEEDS))
        self.assertFalse(set(scorecard.EVAL_SEEDS) & set(range(1, 21)))          # the first look, before the change (see README)


# ==================================================================================================================
class PlainWords(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, cls.key, cls.result = planted(3)
        cls.cohort = insights.cohort_insights(cls.result, cls.data)
        cls.signals = {i: insights.patient_signals(cls.result, cls.data, i) for i in range(cls.data.n)}

    def texts(self):
        out = [f"{i.headline} {i.text} {i.caveat}" for i in self.cohort]
        return out + [s.text for sigs in self.signals.values() for s in sigs]

    def test_it_says_what_it_found_with_the_numbers_it_computed(self):
        by = {i.id: i for i in self.cohort}
        lag = self.result.findings["lead_lag"]
        self.assertIn(f"about {lag.estimate:.0f} days", by["lead_lag"].text)
        self.assertIn(f"{lag.ci95[0]:.0f} to {lag.ci95[1]:.0f}", by["lead_lag"].text)
        self.assertIn("odds ratio", by["night_wakes"].text)
        self.assertIn("morning", by["evening"].text)
        self.assertIn(f"{int(self.result.rising.sum())} of {self.data.n}", by["rising_questions"].text)

    def test_every_number_in_the_night_wakes_sentence_is_the_analysis_own(self):
        nw = self.result.findings["night_wakes"]
        text = next(i.text for i in self.cohort if i.id == "night_wakes")
        self.assertIn(f"{np.exp(nw.estimate):.2f}", text)
        self.assertIn(f"{np.exp(nw.ci95[0]):.2f}", text)
        self.assertIn(f"{np.exp(nw.ci95[1]):.2f}", text)

    def test_the_word_scan_catches_what_it_should_and_lets_the_disclaimer_through(self):
        self.assertEqual(insights.problem_words("This suggests a diagnosis of something."), ["diagnos"])
        self.assertEqual(insights.problem_words("Signs of dementia and worsening memory"), ["dementia", "worsening"])
        self.assertEqual(insights.problem_words("This is not a diagnosis."), [])
        self.assertEqual(insights.problem_words(insights.DISCLAIMER), [])

    def test_no_sentence_sounds_like_a_diagnosis(self):
        # saying "this is not a diagnosis" is the point; claiming or hinting at one is what is not allowed
        for text in self.texts():
            self.assertEqual(insights.problem_words(text), [], text)

    def test_every_insight_has_evidence_and_a_caveat(self):
        for i in self.cohort:
            self.assertTrue(i.evidence and i.caveat and i.headline, i.id)

    def test_what_did_not_hold_up_is_reported(self):
        by = {i.id: i for i in self.cohort}
        self.assertIn("did_not_hold_up", by)
        self.assertIn("one at a time", by["did_not_hold_up"].text)

    def test_the_disclaimer_and_the_synthetic_badge_say_what_they_must(self):
        self.assertIn("not a diagnosis", insights.DISCLAIMER)
        self.assertIn("mentioning to the doctor", insights.DISCLAIMER)
        self.assertIn("Nobody real", insights.SYNTHETIC_BADGE)

    def test_a_person_on_a_slide_gets_a_signal_and_an_ordinary_person_usually_does_not(self):
        sliders = [i for i in range(self.data.n) if self.key.decliner[i] and self.result.rising[i]]
        self.assertTrue(sliders)
        for i in sliders[:5]:
            self.assertTrue(any(s.kind == "rising_questions" for s in self.signals[i]))
        steady = [i for i in range(self.data.n) if not self.key.decliner[i] and self.key.step_day[i] < 0]
        quiet = sum(1 for i in steady if not self.signals[i])
        self.assertGreater(quiet / len(steady), 0.7)

    def test_a_dated_drop_names_the_day_and_the_before_and_after(self):
        both = np.nonzero(self.result.drops)[0]
        if both.size:
            sig = next(s for s in self.signals[int(both[0])] if s.kind == "abrupt_drop")
            self.assertIn(f"around day {int(self.result.drop_day[both[0]])}", sig.text)
            self.assertGreater(sig.evidence["before"], sig.evidence["after"])

    def test_no_person_is_ever_named_and_ids_are_plainly_invented(self):
        self.assertTrue(all(p.startswith("S") and p[1:].isdigit() for p in self.data.patient_ids))


class TheReport(unittest.TestCase):
    def test_it_writes_its_files_and_every_one_says_synthetic(self):
        with tempfile.TemporaryDirectory() as tmp:
            text = report.cohort_report(3, Path(tmp))
            self.assertIn("Synthetic data", text)
            self.assertIn("not a diagnosis", text)
            names = {p.name for p in Path(tmp).iterdir()}
            self.assertEqual(names, {"synthetic_data.csv", "answer_key.json", "findings.json", "report.md"})
            for n in ("answer_key.json", "findings.json"):
                self.assertIn("SYNTHETIC", json.loads((Path(tmp) / n).read_text())["notice"])
            self.assertTrue((Path(tmp) / "synthetic_data.csv").read_text().startswith("# SYNTHETIC"))

    def test_the_scoreboard_prints_every_planted_pattern_with_an_interval(self):
        ev = scorecard.evaluate([21], null_seeds=[31], cfg=FAST, world=SMALL)
        board = report.scoreboard(ev)
        for pid in ("rising questions", "lead lag", "night wakes", "evening", "weekend", "abrupt drop"):
            self.assertIn(pid, board)
        self.assertIn("95% interval", board)
        self.assertIn("nothing is real", board)


if __name__ == "__main__":
    unittest.main(verbosity=1)
