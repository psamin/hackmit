"""The streak inside the running system: taps over several days, late taps, the safety guards that must count a
late tap, who may see the streak, the caregiver's numbers, and how Pam is told about it.

    python server/test_streak.py

The pure rules are in test_adherence.py. This file checks they are wired to real taps: a late tap never erases
the streak and never adds to it; silence resets it only after the grace period; a late tap counts as TAKEN for
the min-gap and daily-maximum guards; and the caregiver can hide the streak from the patient.
"""
import json
import os
import subprocess
import sys
import time
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adherence  # noqa: E402
import caregiver  # noqa: E402
import caregiver_schedule as cs  # noqa: E402
import doses as d  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from test_dose_schedule import DAY, Files, at  # noqa: E402

SERVER = Path(__file__).resolve().parent
ONE = {"name": "Metformin", "times": ["08:00"], "note": "500mg with food"}
PIN = "correct-horse-7"


def day_at(n, h, m=0):
    return at(h, m, DAY + timedelta(days=n))


def group(n, hhmm="08:00"):
    return f"{(DAY + timedelta(days=n)):%Y-%m-%d}T{hhmm}"


class Streaks(Files):
    def setUp(self):
        super().setUp()
        for target, value in (("STREAK_KILL_FILE", self.tmp / ".streak_off"),):
            p = mock.patch.object(d, target, value)
            p.start()
            self.patches.append(p)
        os.environ.pop("PAM_STREAK", None)
        self.save(ONE)

    def good_day(self, n):
        self.tick(day_at(n, 8, 0))
        return d.confirm(None, "yes", day_at(n, 8, 5), group=group(n))

    def missed_window(self, n):
        """Day n: prompted, nobody taps, the window closes (and the engine notices)."""
        self.tick(day_at(n, 8, 0))
        self.tick(day_at(n, 9, 45))

    def summary_at(self, now):
        return d.adherence_summary(now)


class OnTimeTaps(Streaks):
    def test_finishing_the_first_day_says_so(self):
        self.assertEqual(self.good_day(0)["say"], "Thank you. I've noted that you took your Metformin at 8:05 AM. "
                                                  "That's your first day in a row.")

    def test_the_second_day_counts_up(self):
        self.good_day(0)
        self.assertTrue(self.good_day(1)["say"].endswith("That's 2 days in a row."))

    def test_with_two_doses_a_day_the_line_comes_only_after_the_last(self):
        self.save({**ONE, "times": ["08:00", "18:00"]})
        self.tick(day_at(0, 8, 0))
        morning = d.confirm(None, "yes", day_at(0, 8, 5), group=group(0))
        self.assertNotIn("in a row", morning["say"])                              # the day is not finished
        self.tick(day_at(0, 18, 0))
        evening = d.confirm(None, "yes", day_at(0, 18, 5), group=group(0, "18:00"))
        self.assertTrue(evening["say"].endswith("That's your first day in a row."))

    def test_tapping_twice_says_the_same_thing_and_counts_once(self):
        first = self.good_day(0)
        again = d.confirm(None, "yes", day_at(0, 8, 20), group=group(0))
        self.assertEqual(again["say"], first["say"])
        self.assertEqual(self.summary_at(day_at(0, 9)).streak, 1)

    def test_not_yet_never_mentions_a_streak(self):
        self.good_day(0)
        self.tick(day_at(1, 8, 0))
        self.assertNotIn("row", d.confirm(None, "not_yet", day_at(1, 8, 5), group=group(1))["say"])


class LateTaps(Streaks):
    def build_streak_of_three_then_a_missed_window(self):
        for n in range(3):
            self.good_day(n)
        self.missed_window(3)

    def test_a_late_tap_is_recorded_and_says_the_streak_is_untouched(self):
        self.build_streak_of_three_then_a_missed_window()
        r = d.confirm(None, "yes", day_at(3, 9, 45), group=group(3))
        self.assertEqual(r["say"], "Thank you. I've noted that you took your Metformin at 9:45 AM. Your streak is still 3 days.")
        self.assertTrue(r["recorded"])

    def test_a_late_tap_before_the_engine_noticed_the_window_closed_is_not_also_logged_unconfirmed(self):
        for n in range(3):
            self.good_day(n)
        self.tick(day_at(3, 8, 0))                                                # prompted; nobody taps; no tick since 9:30
        d.confirm(None, "yes", day_at(3, 9, 45), group=group(3))                  # a late tap arrives first
        self.tick(day_at(3, 9, 50))                                               # now the engine looks
        kinds_ = [e["type"] for e in self.events() if str(e.get("dose", e.get("doses"))).find("2026-09-24") >= 0]
        self.assertIn("confirmed_late", kinds_)
        self.assertNotIn("unconfirmed", kinds_)                                   # the log never contradicts itself
        self.assertEqual(self.summary_at(day_at(3, 12)).streak, 3)

    def test_pam_never_says_the_tap_was_late(self):
        self.build_streak_of_three_then_a_missed_window()
        say = d.confirm(None, "yes", day_at(3, 9, 45), group=group(3))["say"]
        for word in ("late", "miss", "behind", "should"):
            self.assertNotIn(word, say.lower())

    def test_a_late_tap_does_not_erase_the_streak(self):
        self.build_streak_of_three_then_a_missed_window()
        d.confirm(None, "yes", day_at(3, 9, 45), group=group(3))
        self.assertEqual(self.summary_at(day_at(3, 12)).streak, 3)
        self.assertEqual(self.summary_at(day_at(9, 12)).streak, 3)                # nor does time erase it

    def test_and_does_not_add_to_it(self):
        self.build_streak_of_three_then_a_missed_window()
        d.confirm(None, "yes", day_at(3, 9, 45), group=group(3))
        self.assertEqual(self.summary_at(day_at(3, 12)).streak, 3)                # 3, not 4
        self.assertFalse(self.summary_at(day_at(3, 12)).today_complete)

    def test_the_next_on_time_day_then_adds_one(self):
        self.build_streak_of_three_then_a_missed_window()
        d.confirm(None, "yes", day_at(3, 9, 45), group=group(3))
        self.assertTrue(self.good_day(4)["say"].endswith("That's 4 days in a row."))

    def test_silence_resets_it_only_after_the_grace_period(self):
        self.build_streak_of_three_then_a_missed_window()
        self.assertEqual(self.summary_at(day_at(3, 11)).streak, 3)                # window closed at 9:30; a late tap could still come
        after = day_at(3, 9, 30) + adherence.LATE_TAP_GRACE_S + 60
        self.assertEqual(self.summary_at(after).streak, 0)
        self.assertEqual(self.summary_at(after).best, 3)                          # the best run is never lost

    def test_a_tap_that_comes_too_late_is_refused_and_the_reset_stands(self):
        self.build_streak_of_three_then_a_missed_window()
        after = day_at(3, 9, 30) + adherence.LATE_TAP_GRACE_S + 60
        r = d.confirm(None, "yes", after, group=group(3))
        self.assertFalse(r["ok"])
        self.assertEqual(self.summary_at(after).streak, 0)

    def test_a_late_tap_inside_the_grace_period_undoes_what_silence_would_have_done(self):
        self.build_streak_of_three_then_a_missed_window()
        near_the_end = day_at(3, 9, 30) + adherence.LATE_TAP_GRACE_S - 60
        d.confirm(None, "yes", near_the_end, group=group(3))
        self.assertEqual(self.summary_at(day_at(4, 12)).streak, 3)


class LateTapsAreTapsForSafety(Streaks):
    """A late tap means someone already took it. Everything that guards against a second dose must know."""

    def test_the_minimum_gap_guard_counts_a_late_tap(self):
        self.save({"name": "Warfarin", "times": ["08:00", "12:00"], "min_gap_hours": 3})
        self.tick(day_at(0, 8, 0))
        self.tick(day_at(0, 9, 45))                                               # nobody tapped in the window
        d.confirm(None, "yes", day_at(0, 10, 30), group=group(0))                 # ...but they took it, and said so at 10:30
        self.pushed.clear()
        self.tick(day_at(0, 12, 0))                                               # 90 minutes later: inside the 3 hour gap
        self.assertEqual(self.pushed, [])                                         # no "it's time" for a second dose
        skipped = [e for e in self.events() if e["type"] == "skipped"]
        self.assertEqual([e["reason"] for e in skipped], ["taken recently"])

    def test_without_that_tap_the_next_dose_is_prompted_as_normal(self):
        self.save({"name": "Warfarin", "times": ["08:00", "12:00"], "min_gap_hours": 3})
        self.tick(day_at(0, 8, 0))
        self.tick(day_at(0, 9, 45))
        self.pushed.clear()
        self.tick(day_at(0, 12, 0))
        self.assertTrue(self.pushed)

    def test_the_daily_maximum_guard_counts_a_late_tap(self):
        self.save({"name": "Warfarin", "times": ["08:00", "12:00"], "max_per_day": 2})
        self.tick(day_at(0, 8, 0))
        self.tick(day_at(0, 9, 45))
        d.confirm(None, "yes", day_at(0, 10, 0), group=group(0))
        self.pushed.clear()
        self.tick(day_at(0, 12, 0))
        self.assertTrue(self.pushed)                                              # 1 taken of 2 allowed: still prompted

    def test_the_answer_to_did_i_take_my_pills_counts_a_late_tap(self):
        self.tick(day_at(0, 8, 0))
        self.tick(day_at(0, 9, 45))
        d.confirm(None, "yes", day_at(0, 10, 30), group=group(0))
        r = d.status_response(day_at(0, 11, 0))
        self.assertIn("You marked your Metformin as taken at 10:30 AM.", r["say"])
        self.assertNotIn("I don't have a record", r["say"])
        self.assertTrue(r["recorded"])


class Visibility(Streaks):
    def test_how_am_i_doing_when_visible(self):
        for n in range(3):
            self.good_day(n)
        r = d.streak_response(day_at(2, 12))
        self.assertEqual(r["say"], "You've marked all of your medication on time 3 days in a row.")
        self.assertEqual((r["streak"], r["best"]), (3, 3))
        self.assertEqual(r["card"]["title"], "Your streak")

    def test_after_a_reset_the_answer_is_kind(self):
        for n in range(3):
            self.good_day(n)
        self.missed_window(3)
        after = day_at(3, 9, 30) + adherence.LATE_TAP_GRACE_S + 60
        self.assertEqual(d.streak_response(after)["say"], "Let's start a new streak today. Your best so far is 3 days.")

    def test_with_no_history_it_is_encouraging(self):
        self.store.path.unlink()
        self.assertEqual(d.streak_response(day_at(0, 9))["say"], "I'll start counting once you've marked your medication.")

    def test_the_env_switch_hides_it_from_the_patient(self):
        for n in range(3):
            self.good_day(n)
        os.environ["PAM_STREAK"] = "off"
        self.assertFalse(d.streak_visible())
        r = d.streak_response(day_at(2, 12))
        self.assertEqual((r["say"], r["hidden"]), ("I can't share that right now.", True))
        self.assertNotIn("streak", d.confirm(None, "yes", day_at(2, 8, 20), group=group(2))["say"].lower())
        self.assertNotIn("in a row", d.confirm(None, "yes", day_at(2, 8, 20), group=group(2))["say"])

    def test_the_kill_file_hides_it_at_once_and_deleting_it_shows_it_again(self):
        self.good_day(0)
        (self.tmp / ".streak_off").touch()
        self.assertTrue(d.streak_response(day_at(0, 12))["hidden"])
        (self.tmp / ".streak_off").unlink()
        self.assertNotIn("hidden", d.streak_response(day_at(0, 12)))

    def test_when_the_medication_check_is_off_everything_defers_to_the_caregiver(self):
        os.environ["PAM_DOSE_CHECK"] = "off"
        self.assertFalse(d.streak_visible())
        r = d.streak_response(day_at(0, 12))
        self.assertFalse(r["enabled"])
        self.assertIn("Please ask Mike", r["say"])

    def test_hiding_it_from_the_patient_does_not_hide_the_numbers_from_the_caregiver(self):
        for n in range(2):
            self.good_day(n)
        os.environ["PAM_STREAK"] = "off"
        self.assertEqual(d.adherence_summary(day_at(1, 12)).streak, 2)

    def test_doses_and_taps_are_recorded_the_same_whether_or_not_the_streak_is_shown(self):
        os.environ["PAM_STREAK"] = "off"
        self.good_day(0)
        self.assertEqual(len([e for e in self.events() if e["type"] == "confirmed"]), 1)


class CaregiverRoute(Streaks):
    def setUp(self):
        super().setUp()
        for target, value in (("get_store", lambda: self.store),):
            p = mock.patch.object(cs, target, value)
            p.start()
            self.patches.append(p)
        p = mock.patch.dict(os.environ, {"CAREGIVER_PIN": PIN})
        p.start()
        self.patches.append(p)
        p = mock.patch.object(caregiver, "_log", lambda m: None)
        p.start()
        self.patches.append(p)
        caregiver._sessions.clear()
        caregiver._failures.clear()
        app = FastAPI()
        app.include_router(caregiver.router)
        app.include_router(cs.router)
        self.client = TestClient(app)

    def login(self):
        self.assertEqual(self.client.post("/api/caregiver/login", json={"pin": PIN}).status_code, 200)

    def test_it_needs_the_caregiver_to_be_signed_in(self):
        self.assertEqual(self.client.get("/api/caregiver/adherence").status_code, 401)

    def test_it_is_off_when_the_dashboard_is_off(self):
        os.environ.pop("CAREGIVER_PIN")
        self.assertEqual(self.client.get("/api/caregiver/adherence").status_code, 404)

    def test_it_shows_each_dose_and_how_it_was_tapped(self):
        self.good_day(0)
        self.missed_window(1)
        d.confirm(None, "yes", day_at(1, 9, 45), group=group(1))
        self.tick(day_at(2, 8, 0))
        self.login()
        # only the dose module's clock moves: patching time.time globally would also age the test client's cookie
        with mock.patch.object(d, "time", SimpleNamespace(time=lambda: day_at(2, 8, 30), monotonic=time.monotonic)):
            body = self.client.get("/api/caregiver/adherence").json()
        self.assertEqual([x["state"] for x in body["days"]], ["complete", "late", "pending"])
        self.assertEqual(body["days"][0]["doses"][0], {"name": "Metformin", "due": "08:00", "state": "on_time", "tapped_at": "08:05"})
        self.assertEqual(body["days"][1]["doses"][0]["state"], "late")
        self.assertEqual(body["days"][1]["doses"][0]["tapped_at"], "09:45")
        self.assertEqual((body["streak"], body["best"]), (1, 1))

    def test_it_calls_them_taps_not_intake(self):
        self.good_day(0)
        self.login()
        text = json.dumps(self.client.get("/api/caregiver/adherence").json()).lower()
        for word in ("took", "swallow", "ingest"):
            self.assertNotIn(word, text)

    def test_it_is_never_cached(self):
        self.login()
        self.assertEqual(self.client.get("/api/caregiver/adherence").headers["cache-control"], "no-store")

    def test_the_number_of_days_is_limited_and_bad_input_is_a_clean_error(self):
        self.login()
        for query, ok in (("?days=0", 200), ("?days=1", 200), ("?days=9999", 200), ("?days=-5", 200), ("?days=abc", 422)):
            self.assertEqual(self.client.get("/api/caregiver/adherence" + query).status_code, ok, query)

    def test_with_a_schedule_but_no_taps_today_shows_the_dose_as_still_to_come(self):
        self.login()
        with mock.patch.object(d, "time", SimpleNamespace(time=lambda: day_at(0, 6), monotonic=time.monotonic)):
            body = self.client.get("/api/caregiver/adherence").json()
        self.assertEqual([(x["state"], [y["state"] for y in x["doses"]]) for x in body["days"]], [("pending", ["pending"])])
        self.assertEqual(body["streak"], 0)

    def test_it_works_with_no_history(self):
        self.store.path.unlink()                                                    # no schedule, so nothing is even due
        self.login()
        body = self.client.get("/api/caregiver/adherence").json()
        self.assertEqual((body["days"], body["streak"], body["best"], body["on_time_rate_7d"]), ([], 0, 0, None))

    def test_it_still_works_when_the_streak_is_hidden_from_the_patient(self):
        self.good_day(0)
        os.environ["PAM_STREAK"] = "off"
        self.login()
        self.assertEqual(self.client.get("/api/caregiver/adherence").status_code, 200)


class HowPamIsTold(unittest.TestCase):
    """The function and its rule are registered only when the streak may be shown."""

    @staticmethod
    def app_state(**env):
        code = ("import json, app; c = app.agent_config(); "
                "print(json.dumps({'fn': 'check_streak' in [f['name'] for f in c['think']['functions']], "
                "'rule': 'call check_streak' in c['think']['prompt']}))")
        full = {k: v for k, v in os.environ.items() if not k.startswith("PAM_")}
        full.update(env)
        out = subprocess.run([sys.executable, "-c", code], cwd=SERVER, env=full, capture_output=True, text=True, timeout=120)
        return json.loads(out.stdout.strip().splitlines()[-1])

    def test_registered_by_default(self):
        self.assertEqual(self.app_state(), {"fn": True, "rule": True})

    def test_not_registered_when_the_streak_is_hidden(self):
        self.assertEqual(self.app_state(PAM_STREAK="off"), {"fn": False, "rule": False})

    def test_not_registered_when_the_medication_check_is_off(self):
        self.assertEqual(self.app_state(PAM_DOSE_CHECK="off"), {"fn": False, "rule": False})

    def test_the_rule_forbids_blaming(self):
        for phrase in ("call check_streak", "read the answer aloud as", "Never say they failed", "never compare"):
            self.assertIn(phrase, d.STREAK_RULE.replace("\n  ", " "))
        self.assertIn("days in a row", d.STREAK_FUNCTION_DESCRIPTION)

    def test_the_page_can_call_it(self):
        html = (SERVER.parent / "phone" / "agent.html").read_text(encoding="utf-8")
        self.assertIn("check_streak: () => api(`/api/streak`)", html)                # the plumbing Pam's function call needs
        self.assertNotIn('["check_streak"', html)                                    # but nothing new for the patient to see


if __name__ == "__main__":
    unittest.main(verbosity=1)
