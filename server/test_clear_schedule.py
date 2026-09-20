"""Clearing the medication schedule from Pam oversight.

    python server/test_clear_schedule.py

Clearing is a NEW version of the schedule with no medications in it, never a deletion. What this pins down:

  - it takes an explicit confirmation on the server, and a second step on the page
  - earlier versions stay in the history; nothing is erased
  - reminders stop at once, and a dose that was open and still inside its window is withdrawn, so it is not nagged
    about and not held against anyone. A dose whose window had already closed keeps its record as it was
  - "no schedule" and "an empty schedule" are different things, and the caregiver can tell which one Pam is in
"""
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adherence  # noqa: E402
import caregiver_schedule as cs  # noqa: E402
import doses as d  # noqa: E402
import schedule as sched  # noqa: E402
import test_streak as ts  # noqa: E402
from test_dose_schedule import at  # noqa: E402

CLEAR = "/api/caregiver/schedule/clear"
SERVER = Path(__file__).resolve().parent


class TheStore(ts.Streaks):
    def test_clearing_saves_an_empty_schedule_as_the_next_version(self):
        before = self.store.load()
        entry = self.store.clear()
        self.assertEqual((entry.version, entry.action, entry.schedule.medications), (before.version + 1, "clear", []))
        self.assertEqual(entry.changes, ["Removed: Metformin"])
        self.assertEqual(self.store.load().version, entry.version)

    def test_nothing_is_erased_and_the_old_schedule_can_be_brought_back(self):
        self.store.clear()
        history = self.store.history()
        self.assertEqual([e.action for e in history], ["clear", "save"])
        self.assertEqual([m.name for m in history[1].schedule.medications], ["Metformin"])   # still there, word for word
        back = self.store.restore(history[1].version)
        self.assertEqual([m.name for m in back.schedule.medications], ["Metformin"])

    def test_clearing_twice_or_with_nothing_saved_is_refused_and_writes_nothing(self):
        self.store.clear()
        n = len(self.store.history())
        with self.assertRaises(sched.ScheduleError):
            self.store.clear()
        self.assertEqual(len(self.store.history()), n)
        empty = sched.Store(self.tmp / "never.jsonl", clock=lambda: at(7))
        with self.assertRaises(sched.ScheduleError):
            empty.clear()
        self.assertFalse((self.tmp / "never.jsonl").exists())


class TheRoute(ts.CaregiverRoute):
    def clear(self, body=None):
        return self.client.post(CLEAR, json={"confirm": True} if body is None else body)

    def test_it_needs_the_caregiver_to_be_signed_in_and_the_page_to_be_on(self):
        self.assertEqual(self.client.post(CLEAR, json={"confirm": True}).status_code, 401)
        os.environ.pop("CAREGIVER_PIN")
        self.assertEqual(self.client.post(CLEAR, json={"confirm": True}).status_code, 404)
        self.assertEqual(len(self.store.history()), 1)

    def test_it_needs_an_explicit_confirmation_and_nothing_else_will_do(self):
        self.login()
        for body in ({}, {"confirm": False}, {"confirm": "yes"}, {"confirm": 1}, {"confirm": "true"}, {"confirm": None}):
            r = self.client.post(CLEAR, json=body)
            self.assertEqual(r.status_code, 400, body)
        self.assertEqual(len(self.store.history()), 1)                          # nothing was cleared

    def test_clearing_works_and_the_page_can_see_an_empty_schedule(self):
        self.login()
        r = self.clear()
        self.assertEqual((r.status_code, r.json()["ok"], r.json()["version"]), (200, True, 2))
        now = self.client.get("/api/caregiver/schedule").json()
        self.assertEqual((now["exists"], now["lines"], now["version"]), (True, [], 2))
        self.assertEqual(now["history"][0]["action"], "clear")
        self.assertEqual(now["history"][0]["changes"], ["Removed: Metformin"])

    def test_the_old_instructions_are_not_offered_back_after_a_clear(self):
        self.login()
        self.store.save({"medications": [ts.ONE]}, source_text="Metformin 500mg at 8am with food")
        self.assertEqual(self.client.get("/api/caregiver/schedule").json()["instructions"], "Metformin 500mg at 8am with food")
        self.clear()
        self.assertEqual(self.client.get("/api/caregiver/schedule").json()["instructions"], "")
        self.store.save({"medications": [ts.ONE]}, source_text="Metformin 500mg at 9am")           # a new save after the clear is offered again
        self.assertEqual(self.client.get("/api/caregiver/schedule").json()["instructions"], "Metformin 500mg at 9am")

    def test_clearing_an_already_empty_schedule_is_a_polite_409(self):
        self.login()
        self.clear()
        r = self.clear()
        self.assertEqual(r.status_code, 409)
        self.assertIn("no schedule to clear", r.json()["error"])
        self.assertEqual(len(self.store.history()), 3 - 1)                      # the save and the one clear

    def test_clearing_when_nothing_was_ever_saved_is_a_409_too(self):
        self.login()
        self.store.path.unlink()
        self.assertEqual(self.clear().status_code, 409)

    def test_a_reading_made_before_the_clear_can_no_longer_be_saved(self):
        # the guard that already existed for any change: a draft is only good for the version it was read against
        import schedule_parse as sp
        draft = mock.Mock(errors=[], questions=[], schedule=self.store.load().schedule, unsupported=[],
                          base_version=self.store.load().version, source_text="x")
        drafts = mock.Mock(get=lambda _id: draft)
        self.login()
        self.clear()
        with self.assertRaises(sp.SaveRefused):
            sp.save_draft(drafts, self.store, "id")

    def test_it_says_how_many_reminders_it_stopped(self):
        self.tick(ts.day_at(0, 8, 0))
        with mock.patch.object(d.time, "time", lambda: ts.day_at(0, 8, 10)):     # the clock is global: sign in inside it too
            self.login()
            self.assertEqual(self.clear().json()["reminders_stopped"], 1)


class WhatHappensToPam(ts.Streaks):
    def test_a_dose_still_inside_its_window_is_withdrawn_and_never_nagged_about_again(self):
        self.tick(ts.day_at(0, 8, 0))
        self.store.clear()
        self.assertEqual(d.withdraw_open_doses("schedule cleared", ts.day_at(0, 8, 10)), 1)
        skipped = [e for e in self.events() if e["type"] == "skipped"]
        self.assertEqual(skipped[0]["reason"], "schedule cleared")
        n = len(self.pushed)
        for h, m in ((8, 20), (8, 40), (9, 0), (9, 40)):
            self.tick(ts.day_at(0, h, m))
        self.assertEqual(len(self.pushed), n)                                    # not one more word about it
        self.assertEqual([e["type"] for e in self.events()].count("nudge"), 0)

    def test_withdrawing_does_not_hurt_a_streak(self):
        self.tick(ts.day_at(0, 8, 0))
        d.confirm(None, "yes", ts.day_at(0, 8, 5), group=ts.group(0))
        self.tick(ts.day_at(1, 8, 0))                                            # day 1 is prompted ...
        self.store.clear()
        d.withdraw_open_doses("schedule cleared", ts.day_at(1, 8, 10))           # ... then the schedule is cleared
        after = ts.day_at(1, 9, 30) + adherence.LATE_TAP_GRACE_S + 3600
        summary = d.adherence_summary(after)
        self.assertEqual(summary.streak, 1)                                      # not reset by a dose that no longer exists
        self.assertEqual([day.state for day in summary.days], [adherence.COMPLETE])

    def test_a_dose_whose_window_had_closed_keeps_its_record(self):
        self.tick(ts.day_at(0, 8, 0))
        self.tick(ts.day_at(0, 9, 45))                                           # the window closed with no answer
        self.store.clear()
        self.assertEqual(d.withdraw_open_doses("schedule cleared", ts.day_at(0, 10, 0)), 0)
        self.assertEqual([e for e in self.events() if e["type"] == "skipped"], [])

    def test_a_window_that_closed_before_the_engine_noticed_is_not_rewritten_either(self):
        self.tick(ts.day_at(0, 8, 0))                                            # no tick after that: the dose still looks open
        self.assertEqual(d.withdraw_open_doses("schedule cleared", ts.day_at(0, 9, 45)), 0)
        self.assertEqual([e for e in self.events() if e["type"] == "skipped"], [])

    def test_a_dose_that_was_answered_is_left_alone(self):
        self.tick(ts.day_at(0, 8, 0))
        d.confirm(None, "yes", ts.day_at(0, 8, 5), group=ts.group(0))
        self.assertEqual(d.withdraw_open_doses("schedule cleared", ts.day_at(0, 8, 10)), 0)
        self.assertEqual(d.adherence_summary(ts.day_at(0, 9)).days[-1].state, adherence.COMPLETE)

    def test_reminders_that_were_not_from_the_schedule_are_not_touched(self):
        reminder = {"id": "r1", "text": "take your medication", "fired": True, "due_ts": ts.day_at(0, 8)}
        d.tick([reminder], self.memfile, self.push, ts.day_at(0, 8, 11))
        self.assertEqual(d.withdraw_open_doses("schedule cleared", ts.day_at(0, 8, 12)), 0)

    def test_after_a_clear_no_scheduled_reminder_is_ever_sent(self):
        self.store.clear()
        for h in (8, 12, 18):
            self.tick(ts.day_at(1, h, 0))
        self.assertEqual(self.pushed, [])
        self.assertEqual(d.schedule_health()["state"], "empty")

    def test_no_schedule_and_an_empty_one_are_told_apart(self):
        self.tick(ts.day_at(0, 7, 0))
        self.assertEqual(d.schedule_health()["state"], "active")
        self.store.clear()
        self.tick(ts.day_at(0, 7, 5))
        self.assertEqual(d.schedule_health()["state"], "empty")
        self.store.path.unlink()
        self.tick(ts.day_at(0, 7, 10))
        self.assertEqual(d.schedule_health()["state"], "none")

    def test_a_new_schedule_after_a_clear_starts_reminding_again(self):
        self.store.clear()
        self.store.clock = lambda: at(7)
        self.store.save({"medications": [ts.ONE]})
        self.tick(ts.day_at(0, 8, 0))
        self.assertTrue(any("Metformin" in m.get("text", "") for m in self.pushed))


class ThePage(unittest.TestCase):
    def setUp(self):
        self.text = (SERVER / "caregiver.html").read_text(encoding="utf-8")

    def test_the_button_only_asks_and_a_second_step_does_the_clearing(self):
        self.assertIn('id="clear-btn"', self.text)
        first = self.text.split('$("clear-btn").addEventListener')[1].split('$("clear-no")')[0]
        self.assertNotIn("post(", first)                                         # one click never clears anything
        self.assertIn("Yes, clear it", self.text)
        self.assertIn("Keep it", self.text)
        self.assertIn('post("/api/caregiver/schedule/clear", { confirm: true })', self.text)

    def test_the_confirmation_says_what_will_and_will_not_happen(self):
        for words in ("stop reminding about every medication", "Doses already on record stay on record",
                      "earlier versions stay under Recent changes"):
            self.assertIn(words, self.text)

    def test_there_is_no_button_when_there_is_nothing_to_clear(self):
        self.assertIn('$("clear-area").hidden = !(r.ok && r.body.exists && r.body.lines.length)', self.text)

    def test_an_empty_schedule_and_the_status_say_so_plainly(self):
        self.assertIn("The schedule is empty. Pam is not sending medication reminders.", self.text)
        self.assertIn("Schedule is empty: no reminders are being sent", self.text)

    def test_the_words_of_the_old_schedule_are_cleared_from_the_box_and_the_session_end_is_handled(self):
        after = self.text.split('$("clear-yes").addEventListener')[1].split("// ---- signing in")[0]
        self.assertIn('$("instructions").value = ""', after)
        self.assertIn("r.status === 401", after)
        self.assertNotIn("innerHTML", self.text)


if __name__ == "__main__":
    unittest.main(verbosity=1)
