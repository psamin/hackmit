"""Saying it out loud: Pam records a spoken "yes, I took them" / "not yet", and asks after the arm hands over the pills.

    python server/test_voice_confirm.py

A spoken answer is worth exactly what a tap on the card is worth, and no more. The things this file pins down:

  - the voice model never chooses which dose an answer is for; the server does, and asks when it cannot tell
  - only a dose Pam actually asked about, and that can still be answered, takes a spoken answer
  - the question after the handoff is neutral (both answers offered), asked once, later, and never instructs
  - a handoff with no dose waiting is logged and nothing else happens
  - `via` records how each answer arrived, so the caregiver can tell a spoken one from a tap
  - PAM_VOICE_CONFIRM / .voice_confirm_off stop all of it, and the card keeps working
"""
import asyncio
import os
import re
import subprocess
import sys
import unittest
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import adherence  # noqa: E402
import doses as d  # noqa: E402
import handoff  # noqa: E402
from test_dose_schedule import DAY, GROUP_8, LIS, MET, VITD, Files, at  # noqa: E402

SERVER = Path(__file__).resolve().parent
ONE = {"name": "Metformin", "times": ["08:00"], "note": "500mg with food"}
EARLY = {"name": "Metformin", "times": ["08:00"]}
LATER = {"name": "Vitamin D", "times": ["08:30"]}
QUESTION = "Have you taken your Metformin yet, or not yet? If you're not sure, please check with Mike first."


def day_at(n, h, m=0):
    return at(h, m, DAY + timedelta(days=n))


def group(n, hhmm="08:00"):
    return f"{(DAY + timedelta(days=n)):%Y-%m-%d}T{hhmm}"


class Base(Files):
    def setUp(self):
        super().setUp()
        p = mock.patch.object(d, "VOICE_KILL_FILE", self.tmp / ".voice_confirm_off")
        p.start()
        self.patches.append(p)
        p = mock.patch.object(d, "STREAK_KILL_FILE", self.tmp / ".streak_off")
        p.start()
        self.patches.append(p)
        for var in ("PAM_VOICE_CONFIRM", "PAM_STREAK"):
            os.environ.pop(var, None)
        self.save(ONE)

    def kinds(self, *types):
        return [e for e in self.events() if e["type"] in types]

    def spoken(self):
        return [m["text"] for m in self.pushed if m["type"] == "speak"]

    def open_dose(self, h=8, m=0):
        self.tick(at(h, m))


# ==================================================================================================
class AfterTheHandoff(Base):
    def test_a_little_after_the_handoff_pam_asks_once_and_the_words_are_exact(self):
        self.open_dose()
        self.assertEqual(d.record_handoff("pill bottle", at(8, 2))["reason"], "will_ask")
        self.tick(at(8, 5))
        self.assertEqual(self.spoken()[-1], QUESTION)
        self.assertEqual(len(self.kinds("handoff_asked")), 1)
        for minute in (6, 7, 8, 9):
            self.tick(at(8, minute))
        self.assertEqual(len(self.kinds("handoff_asked")), 1)               # once, however long it stays unanswered

    def test_the_question_carries_the_same_yes_and_not_yet_buttons_a_tap_uses(self):
        self.open_dose()
        d.record_handoff("pill bottle", at(8, 2))
        self.tick(at(8, 5))
        card = [m for m in self.pushed if m["type"] == "card"][-1]["card"]
        self.assertEqual(card["action"]["body"], {"group": GROUP_8, "answer": "yes"})
        self.assertEqual(card["action2"]["body"], {"group": GROUP_8, "answer": "not_yet"})
        self.assertIn("Metformin: 500mg with food", card["body"])             # the note is shown, never spoken
        self.assertNotIn("500", self.spoken()[-1])

    def test_it_waits_before_asking_because_they_need_time_to_take_them(self):
        self.open_dose()
        with mock.patch.object(d, "timing", lambda: replace(d.Timing(), min_gap_s=0)):
            d.record_handoff("pill bottle", at(8, 2))
            self.tick(at(8, 3) + 59)
            self.assertEqual(self.kinds("handoff_asked"), [])
            self.tick(at(8, 4))
            self.assertEqual(len(self.kinds("handoff_asked")), 1)

    def test_an_answer_before_the_question_makes_the_question_unnecessary(self):
        self.open_dose()
        d.record_handoff("pill bottle", at(8, 2))
        d.confirm(None, "yes", at(8, 3), group=GROUP_8)
        self.tick(at(8, 6))
        self.assertEqual(self.kinds("handoff_asked"), [])

    def test_a_spoken_answer_before_the_question_does_too(self):
        self.open_dose()
        d.record_handoff("pill bottle", at(8, 2))
        d.voice_confirm("yes", None, at(8, 3))
        self.tick(at(8, 6))
        self.assertEqual(self.kinds("handoff_asked"), [])

    def test_the_question_is_neutral_and_never_tells_anyone_to_take_anything(self):
        self.save(MET, LIS)
        self.open_dose()
        d.record_handoff("my medicine", at(8, 2))
        self.tick(at(8, 5))
        said = self.spoken()[-1]
        self.assertIn("yet, or not yet?", said)                               # both answers, so it does not lead
        self.assertIn("check with Mike", said)
        self.assertEqual(said, "Have you taken your Lisinopril and Metformin yet, or not yet? "
                               "If you're not sure, please check with Mike first.")
        for bad in (r"\btake\b", r"\bshould\b", r"\bmust\b", r"\d", r"\bmissed\b", r"\bforgot", r"\bdidn't\b"):
            self.assertIsNone(re.search(bad, said, re.I), bad)

    def test_asking_counts_as_speaking_so_a_nudge_does_not_follow_at_once(self):
        self.open_dose()
        d.record_handoff("pill bottle", at(8, 6))
        self.tick(at(8, 9))
        self.assertEqual(len(self.kinds("handoff_asked")), 1)
        self.tick(at(8, 10))                                                    # the first nudge was due now, but Pam just spoke
        self.assertEqual(self.kinds("nudge"), [])
        self.tick(at(8, 15))
        self.assertEqual(len(self.kinds("nudge")), 1)

    def test_replaying_two_handoffs_keeps_the_first_time(self):
        events = [{"type": "due", "dose": "a", "text": "x", "due_ts": 1, "source": "schedule", "name": "X", "closes_ts": 999},
                  {"type": "handoff", "doses": ["a"], "ts": 10}, {"type": "handoff", "doses": ["a"], "ts": 500}]
        self.assertEqual(d.replay(events)["a"].handoff_at, 10)

    def test_a_not_yet_leaves_the_normal_reminders_running(self):
        self.open_dose()
        d.record_handoff("pill bottle", at(8, 2))
        self.tick(at(8, 5))
        d.voice_confirm("not_yet", None, at(8, 6))
        self.tick(at(8, 10))
        self.assertEqual(len(self.kinds("nudge")), 1)
        self.assertEqual(self.kinds("confirmed"), [])

    def test_it_is_not_logged_as_asked_when_nobody_was_there_to_hear_it(self):
        self.open_dose()
        d.record_handoff("pill bottle", at(8, 2))
        self.tick(at(8, 5), deliver=False)
        self.assertEqual(self.kinds("handoff_asked"), [])
        self.tick(at(8, 6))
        self.assertEqual(len(self.kinds("handoff_asked")), 1)

    def test_it_never_speaks_twice_in_one_pass_and_still_gets_asked(self):
        self.tick(at(8, 0), deliver=False)                                    # the dose opens; nobody heard the reminder
        d.record_handoff("pill bottle", at(8, 1))
        for minute in range(10, 60, 2):
            before = len(self.spoken())
            self.tick(at(8, minute))
            self.assertLessEqual(len(self.spoken()) - before, 1, minute)
        self.assertEqual(len(self.kinds("handoff_asked")), 1)

    def test_when_the_window_and_the_grace_period_are_over_there_is_nothing_left_to_ask(self):
        self.open_dose()
        d.record_handoff("pill bottle", at(8, 2))
        late = at(8, 0) + 3 * 3600 + adherence.LATE_TAP_GRACE_S
        self.tick(late)
        self.assertEqual(self.kinds("handoff_asked"), [])

    def test_a_handoff_after_the_window_still_asks_and_the_answer_is_a_late_one(self):
        self.open_dose()
        self.tick(at(9, 45))
        self.assertEqual(d.record_handoff("pill bottle", at(10, 0))["reason"], "will_ask")
        self.tick(at(10, 5))
        self.assertEqual(self.spoken()[-1], QUESTION)
        d.voice_confirm("yes", None, at(10, 6))
        self.assertEqual(len(self.kinds("confirmed_late")), 1)
        self.assertEqual(self.kinds("confirmed_late")[0]["via"], "voice")

    def test_demo_timing_asks_within_seconds(self):
        os.environ["PAM_DOSE_DEMO"] = "1"
        self.open_dose()
        d.record_handoff("pill bottle", at(8) + 20)
        for sec in range(30, 91, 15):                                          # the fast demo nudges speak first ...
            self.tick(at(8) + sec)
        self.assertIn(QUESTION, self.spoken())                                 # ... and the question follows within seconds
        self.assertLess(d.DEMO.handoff_ask_s, d.Timing().handoff_ask_s)

    def test_a_medication_without_a_schedule_is_asked_about_as_pills(self):
        reminder = {"id": "r1", "text": "take your medication", "fired": True, "due_ts": at(8)}
        d.tick([reminder], self.memfile, self.push, at(8, 11))               # the ladder has asked (nudge 1)
        d.record_handoff("pill bottle", at(8, 12))
        d.tick([reminder], self.memfile, self.push, at(8, 17))
        self.assertEqual(self.spoken()[-1], "Have you taken your pills yet, or not yet? "
                                            "If you're not sure, please check with Mike first.")
        card = [m for m in self.pushed if m["type"] == "card"][-1]["card"]
        self.assertEqual(card["action"]["body"], {"dose": "r1", "answer": "yes"})


class RecordingAHandoff(Base):
    def test_with_no_dose_waiting_it_is_only_logged(self):
        r = d.record_handoff("pill bottle", at(7, 0))
        self.assertEqual((r["ok"], r["logged"], r["reason"]), (True, True, "no_open_dose"))
        self.assertEqual(self.kinds("handoff")[0]["doses"], [])
        for minute in range(0, 600, 20):
            self.tick(at(8, 0) + minute * 60)
        self.assertEqual(self.kinds("handoff_asked"), [])

    def test_a_dose_already_marked_taken_is_not_asked_about_again(self):
        self.open_dose()
        d.confirm(None, "yes", at(8, 3), group=GROUP_8)
        r = d.record_handoff("pill bottle", at(8, 4))
        self.assertEqual(r["reason"], "no_open_dose")
        self.tick(at(8, 20))
        self.assertEqual(self.kinds("handoff_asked"), [])

    def test_only_a_medication_fetched_counts(self):
        self.open_dose()
        for item in ("glasses", "my keys", "", "the remote"):
            self.assertEqual(d.record_handoff(item, at(8, 2))["reason"], "not_medication", item)
        self.assertEqual(self.kinds("handoff"), [])
        for item in ("pill bottle", "my medicine", "the tablets", "prescription", "meds"):
            self.assertTrue(d.is_pill_item(item), item)
        self.assertFalse(d.is_pill_item("glasses"))

    def test_fetching_again_does_not_mean_asking_again(self):
        self.open_dose()
        self.assertTrue(d.record_handoff("pill bottle", at(8, 2))["logged"])
        self.assertEqual(d.record_handoff("pill bottle", at(8, 3))["reason"], "already_recorded")
        self.assertEqual(len(self.kinds("handoff")), 1)
        self.tick(at(8, 6))                                                    # asked
        self.assertEqual(d.record_handoff("pill bottle", at(8, 8))["reason"], "already_recorded")
        self.tick(at(8, 20))
        self.assertEqual(len(self.kinds("handoff_asked")), 1)

    def test_it_asks_about_the_newest_dose_not_a_stale_one(self):
        self.save(EARLY, VITD)
        self.open_dose()
        self.tick(at(9, 45))                                                    # morning window closes, untapped
        self.tick(at(12, 0))                                                    # noon dose opens
        r = d.record_handoff("pill bottle", at(12, 1))
        self.assertEqual(r["asks"], ["Vitamin D"])
        morning = [x for x in d.replay(self.events()).values() if x.name == "Metformin"][0]
        self.assertIsNone(morning.handoff_at)

    def test_the_handoff_names_only_the_group_it_is_for(self):
        self.save(MET, LIS)
        self.open_dose()
        self.assertEqual(sorted(d.record_handoff("pill bottle", at(8, 2))["asks"]), ["Lisinopril", "Metformin"])
        self.assertEqual(len(self.kinds("handoff")[0]["doses"]), 2)

    def test_a_handoff_is_a_log_entry_not_a_dose_outcome(self):
        self.open_dose()
        d.record_handoff("pill bottle", at(8, 2))
        dose = list(d.replay(self.events()).values())[0]
        self.assertIsNone(dose.confirmed_ts)
        self.assertEqual(adherence.dose_state(dose, at(8, 3)), adherence.PENDING)

    def test_the_item_is_stored_short(self):
        self.open_dose()
        d.record_handoff("pill bottle " + "x" * 500, at(8, 2))
        self.assertLessEqual(len(self.kinds("handoff")[0]["item"]), 80)


class TheSwitch(Base):
    def test_the_env_switch_stops_recording_and_asking(self):
        self.open_dose()
        os.environ["PAM_VOICE_CONFIRM"] = "off"
        self.assertFalse(d.voice_confirm_enabled())
        self.assertEqual(d.record_handoff("pill bottle", at(8, 2))["reason"], "off")
        self.assertEqual(self.kinds("handoff"), [])
        r = d.voice_confirm("yes", None, at(8, 3))
        self.assertEqual((r["ok"], r["enabled"]), (False, False))
        self.assertEqual(self.kinds("confirmed"), [])

    def test_the_kill_file_does_it_at_once_and_deleting_it_turns_it_back_on(self):
        self.open_dose()
        d.record_handoff("pill bottle", at(8, 2))                               # logged while it was on
        (self.tmp / ".voice_confirm_off").touch()
        self.tick(at(8, 6))
        self.assertEqual(self.kinds("handoff_asked"), [])                       # the question is held back
        self.assertFalse(d.voice_confirm("yes", None, at(8, 7))["ok"])
        (self.tmp / ".voice_confirm_off").unlink()
        self.tick(at(8, 8))
        self.assertEqual(len(self.kinds("handoff_asked")), 1)

    def test_the_card_keeps_working_with_voice_off(self):
        self.open_dose()
        os.environ["PAM_VOICE_CONFIRM"] = "off"
        r = d.confirm(None, "yes", at(8, 5), group=GROUP_8)
        self.assertTrue(r["ok"])
        self.assertEqual(self.kinds("confirmed")[0]["via"], "tap")

    def test_the_whole_medication_check_off_turns_voice_off_with_it(self):
        os.environ["PAM_DOSE_CHECK"] = "off"
        self.assertFalse(d.voice_confirm_enabled())
        r = d.voice_confirm("yes", None, at(8, 5))
        self.assertFalse(r["enabled"])
        self.assertIn("Please ask Mike", r["say"])


# ==================================================================================================
class SpokenAnswers(Base):
    def test_yes_is_recorded_like_a_tap_and_marked_as_spoken(self):
        self.open_dose()
        r = d.voice_confirm("yes", None, at(8, 5))
        self.assertEqual((r["ok"], r["recorded"]), (True, True))
        self.assertTrue(r["say"].startswith("Thank you. I've noted that you took your Metformin at 8:05 AM."))
        self.assertEqual(self.kinds("confirmed")[0]["via"], "voice")
        self.assertEqual(list(d.replay(self.events()).values())[0].via, "voice")

    def test_a_tap_is_marked_as_a_tap(self):
        self.open_dose()
        d.confirm(None, "yes", at(8, 5), group=GROUP_8)
        self.assertEqual(list(d.replay(self.events()).values())[0].via, "tap")

    def test_old_log_lines_with_no_via_read_as_taps(self):
        events = [{"type": "due", "dose": "a", "text": "x", "due_ts": 1, "source": "schedule", "name": "X", "closes_ts": 99},
                  {"type": "confirmed", "dose": "a", "ts": 5}]
        self.assertEqual(d.replay(events)["a"].via, "tap")

    def test_one_spoken_yes_answers_for_the_whole_group(self):
        self.save(MET, LIS)
        self.open_dose()
        r = d.voice_confirm("yes", None, at(8, 5))
        self.assertEqual(r["say"], "Thank you. I've noted that you took your Lisinopril and Metformin at 8:05 AM.")
        self.assertEqual(len(self.kinds("confirmed")), 1)
        self.assertEqual(len(self.kinds("confirmed")[0]["doses"]), 2)

    def test_saying_it_twice_records_it_once(self):
        self.open_dose()
        d.voice_confirm("yes", None, at(8, 5))
        again = d.voice_confirm("yes", None, at(8, 9))
        self.assertEqual(len(self.kinds("confirmed")), 1)
        # nothing is waiting any more, so they are told what is already recorded (they may just be repeating themselves)
        self.assertEqual((again["already"], again["say"]), (True, "You marked your Metformin as taken at 8:05 AM."))

    def test_repeating_a_not_yet_or_saying_yes_with_nothing_recorded_says_nothing_new(self):
        self.open_dose()
        self.tick(at(9, 45))
        d.voice_confirm("yes", None, at(8, 5) + 86400 * 3)                      # days later: nothing recorded, nothing waiting
        self.assertEqual(self.kinds("confirmed", "confirmed_late"), [])
        self.assertNotIn("already", d.voice_confirm("yes", None, at(8, 5) + 86400 * 3))

    def test_a_tap_then_a_spoken_yes_records_once_and_keeps_the_tap(self):
        self.open_dose()
        d.confirm(None, "yes", at(8, 5), group=GROUP_8)
        d.voice_confirm("yes", None, at(8, 6))
        self.assertEqual([e["via"] for e in self.kinds("confirmed")], ["tap"])

    def test_not_yet_records_the_answer_and_nothing_taken(self):
        self.open_dose()
        r = d.voice_confirm("not_yet", None, at(8, 5))
        self.assertTrue(r["ok"])
        self.assertNotIn("recorded", r)
        self.assertIn("check with Mike", r["say"])
        self.assertEqual(self.kinds("not_yet")[0]["via"], "voice")
        self.assertEqual(self.kinds("confirmed"), [])
        self.assertTrue(list(d.replay(self.events()).values())[0].is_open)
        self.assertTrue(d.voice_confirm("yes", None, at(8, 7))["recorded"])    # they can say yes later

    def test_a_dose_pam_never_asked_about_takes_no_spoken_answer(self):
        self.tick(at(8, 0), deliver=False)                                      # opened, but never announced
        r = d.voice_confirm("yes", None, at(8, 5))
        self.assertFalse(r["ok"])
        self.assertIn("check with Mike", r["say"])
        self.assertEqual(self.kinds("confirmed", "confirmed_late"), [])

    def test_an_answer_that_is_neither_is_said_not_to_be_understood_even_with_nothing_open(self):
        self.assertEqual(d.voice_confirm("maybe", None, at(7, 0))["say"], "I didn't understand that answer.")

    def test_with_nothing_open_nothing_is_recorded(self):
        for answer in ("yes", "not_yet"):
            r = d.voice_confirm(answer, None, at(7, 0))
            self.assertFalse(r["ok"])
        self.assertEqual(self.events(), [])

    def test_only_yes_and_not_yet_are_answers(self):
        self.open_dose()
        for answer in ("maybe", "", "yes please", "no", "YES", "I think so", None):
            self.assertFalse(d.voice_confirm(answer, None, at(8, 5))["ok"], answer)
        self.assertEqual(self.kinds("confirmed", "not_yet"), [])

    def test_a_late_spoken_yes_is_recorded_and_neutral_for_the_streak(self):
        self.open_dose()
        self.tick(at(9, 45))
        r = d.voice_confirm("yes", None, at(10, 0))
        self.assertTrue(r["ok"])
        self.assertEqual(self.kinds("confirmed_late")[0]["via"], "voice")
        self.assertEqual(self.kinds("confirmed"), [])
        summary = d.adherence_summary(at(10, 5))
        self.assertEqual(summary.streak, 0)                                     # neither added ...
        self.assertEqual(summary.days[-1].state, adherence.LATE)               # ... nor erased, exactly as a late tap

    def test_a_spoken_yes_past_the_grace_period_is_refused(self):
        self.open_dose()
        self.tick(at(9, 45))
        r = d.voice_confirm("yes", None, at(9, 30) + adherence.LATE_TAP_GRACE_S + 60)
        self.assertFalse(r["ok"])
        self.assertEqual(self.kinds("confirmed", "confirmed_late"), [])

    def test_a_spoken_yes_counts_toward_the_streak_exactly_like_a_tap(self):
        self.tick(day_at(0, 8, 0))
        d.confirm(None, "yes", day_at(0, 8, 5), group=group(0))
        self.tick(day_at(1, 8, 0))
        r = d.voice_confirm("yes", None, day_at(1, 8, 5))
        self.assertTrue(r["say"].endswith("That's 2 days in a row."))
        self.assertEqual(d.adherence_summary(day_at(1, 9)).streak, 2)

    def test_a_spoken_yes_counts_as_taken_for_the_double_dose_guard(self):
        self.save({"name": "Metformin", "times": ["08:00", "12:00"], "min_gap_hours": 4})
        self.open_dose()
        d.voice_confirm("yes", None, at(8, 30))
        self.tick(at(12, 0))
        self.assertTrue(any(e["type"] == "skipped" for e in self.events()))

    def test_the_spoken_answer_reaches_the_caregivers_numbers_marked_as_spoken(self):
        self.open_dose()
        d.voice_confirm("yes", None, at(8, 5))
        pub = adherence.public(d.adherence_summary(at(8, 30)))
        dose = pub["days"][-1]["doses"][0]
        self.assertEqual((dose["state"], dose["via"], dose["tapped_at"]), ("on_time", "voice", "08:05"))

    def test_an_unanswered_dose_shows_no_via(self):
        self.open_dose()
        pub = adherence.public(d.adherence_summary(at(8, 30)))
        self.assertIsNone(pub["days"][-1]["doses"][0]["via"])

    def test_a_reminder_based_dose_can_be_answered_by_voice_too(self):
        reminder = {"id": "r1", "text": "take your medication", "fired": True, "due_ts": at(8)}
        d.tick([reminder], self.memfile, self.push, at(8, 11))
        r = d.voice_confirm("yes", None, at(8, 12))
        self.assertEqual(r["say"], "Thank you. I've noted that you took your pills at 8:12 AM.")
        self.assertEqual(self.kinds("confirmed")[0]["via"], "voice")


class WhichDose(Base):
    """Pam does not guess. With two candidates a wrong guess records a dose nobody took."""

    def setUp(self):
        super().setUp()
        self.save(EARLY, LATER)
        for h, m in ((8, 0), (8, 10), (8, 20), (8, 30)):      # Metformin: asked, nudged, escalated. Vitamin D: announced at 8:30
            self.tick(at(h, m))

    def test_a_named_medication_settles_it(self):
        r = d.voice_confirm("yes", "vitamin d", at(8, 50))
        self.assertTrue(r["ok"])
        self.assertEqual([x["name"] for x in self.recorded()], ["Vitamin D"])

    def recorded(self):
        return [{"name": x.name} for x in d.replay(self.events()).values() if x.confirmed_ts]

    def test_the_one_asked_about_in_the_last_minutes_wins_when_only_one_is_recent(self):
        r = d.voice_confirm("yes", None, at(8, 33))                              # Vitamin D was announced at 8:30
        self.assertEqual([x["name"] for x in self.recorded()], ["Vitamin D"])
        self.assertIn("Vitamin D", r["say"])

    def test_when_neither_is_recent_pam_asks_which_and_records_nothing(self):
        r = d.voice_confirm("yes", None, at(8, 50))
        self.assertEqual((r["ok"], r["ambiguous"]), (False, True))
        self.assertEqual(r["say"], "Which one do you mean: your Metformin at 8:00 AM or your Vitamin D at 8:30 AM?")
        self.assertEqual(self.kinds("confirmed", "not_yet"), [])

    def test_when_both_are_recent_pam_asks_which(self):
        with mock.patch.object(d, "RECENT_QUESTION_S", 3600):
            r = d.voice_confirm("yes", None, at(8, 40))
        self.assertTrue(r["ambiguous"])
        self.assertEqual(self.kinds("confirmed"), [])

    def test_a_medication_that_is_not_waiting_is_not_guessed_at(self):
        r = d.voice_confirm("yes", "aspirin", at(8, 33))
        self.assertFalse(r["ok"])
        self.assertEqual(self.kinds("confirmed"), [])

    def test_a_generic_word_does_not_narrow_anything(self):
        r = d.voice_confirm("yes", "pills", at(8, 50))
        self.assertTrue(r["ambiguous"])                                          # "pills" says nothing, so it still asks which
        self.assertEqual(self.kinds("confirmed"), [])

    def test_a_generic_word_still_lets_the_recent_question_win(self):
        self.assertTrue(d.voice_confirm("yes", "pills", at(8, 33))["ok"])


# ==================================================================================================
class Watching(unittest.IsolatedAsyncioTestCase):
    """handoff.watch: notice the arm stopping cleanly. It only looks."""

    def run_watch(self, statuses, **kw):
        calls, seen = [], []
        it = iter(statuses)

        def status_fn():
            calls.append(1)
            s = next(it)
            if isinstance(s, Exception):
                raise s
            return s

        clock = [0.0]

        async def sleep(s):
            clock[0] += s

        async def go():
            return await handoff.watch("pill bottle", status_fn, seen.append, sleep=sleep, clock=lambda: clock[0], **kw)

        return asyncio.run(go()), seen, calls

    def test_active_then_stopped_cleanly_is_a_handoff_reported_once(self):
        outcome, seen, calls = self.run_watch([{"active": True}, {"active": True}, {"active": False, "last_error": None}])
        self.assertEqual((outcome, seen, len(calls)), ("handed_over", ["pill bottle"], 3))

    def test_stopping_with_an_error_is_not_a_handoff(self):
        outcome, seen, _ = self.run_watch([{"active": True}, {"active": False, "last_error": "trajectory rejected"}])
        self.assertEqual((outcome, seen), ("failed", []))

    def test_an_arm_that_stops_answering_is_not_a_handoff(self):
        boom = ConnectionError("down")
        outcome, seen, _ = self.run_watch([{"active": True}, boom, boom, boom])
        self.assertEqual((outcome, seen), ("lost", []))

    def test_a_blip_is_forgiven(self):
        boom = ConnectionError("down")
        outcome, seen, _ = self.run_watch([{"active": True}, boom, boom, {"active": True}, boom, boom, {"active": False}])
        self.assertEqual((outcome, seen), ("handed_over", ["pill bottle"]))

    def test_an_arm_that_never_stops_is_given_up_on(self):
        outcome, seen, calls = self.run_watch([{"active": True}] * 1000, timeout_s=20, poll_s=2)
        self.assertEqual((outcome, seen), ("timeout", []))
        self.assertLess(len(calls), 20)

    def test_it_only_asks_for_status(self):
        # the watcher is handed one function and it is a status reader: there is nothing here that could drive the arm
        import inspect
        params = list(inspect.signature(handoff.watch).parameters)
        self.assertNotIn("start", params)
        self.assertNotIn("stop", params)
        self.assertNotIn("arm", "".join(params))


# ==================================================================================================
class Routes(Base):
    @classmethod
    def setUpClass(cls):
        import app
        from fastapi.testclient import TestClient
        cls.app = app
        cls.client = TestClient(app.app)

    def now(self, h, m=0):
        return mock.patch.object(d.time, "time", lambda: at(h, m))

    def test_pill_answer_records_a_spoken_yes(self):
        self.open_dose()
        with self.now(8, 5):
            r = self.client.post("/api/pill-answer", json={"answer": "yes"})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["ok"])
        self.assertEqual(self.kinds("confirmed")[0]["via"], "voice")

    def test_pill_answer_ignores_a_medication_that_is_not_text(self):
        self.open_dose()
        with self.now(8, 5):
            r = self.client.post("/api/pill-answer", json={"answer": "yes", "medication": {"a": 1}})
        self.assertTrue(r.json()["ok"])

    def test_pill_answer_with_no_answer_records_nothing(self):
        self.open_dose()
        self.assertFalse(self.client.post("/api/pill-answer", json={}).json()["ok"])
        self.assertEqual(self.kinds("confirmed", "not_yet"), [])

    def test_handoff_defaults_to_the_pill_bottle(self):
        self.open_dose()
        with self.now(8, 2):
            r = self.client.post("/api/handoff", json={})
        self.assertEqual(r.json()["reason"], "will_ask")

    def test_handoff_for_something_else_does_nothing(self):
        self.open_dose()
        r = self.client.post("/api/handoff", json={"item": "glasses"})
        self.assertEqual(r.json()["reason"], "not_medication")
        self.assertEqual(self.kinds("handoff"), [])

    def fetch(self, item, active=True, voice=True, watcher_raises=False):
        import vla.arm_client as arm_client
        started = []
        if not voice:
            os.environ["PAM_VOICE_CONFIRM"] = "off"

        def fake_watch(item, fn):
            if watcher_raises:
                raise RuntimeError("no loop")
            started.append(item)
        with mock.patch.object(arm_client, "fetch", lambda url: {"active": active}), \
                mock.patch.object(self.app, "_watch_handoff", fake_watch):
            r = self.client.post("/api/fetch", json={"item": item})
        return r.json(), started

    def test_fetching_the_pills_starts_watching_for_the_handoff(self):
        r, started = self.fetch("pill bottle")
        self.assertEqual(started, ["pill bottle"])
        self.assertIn("on its way", r["say"])

    def test_fetching_anything_else_does_not(self):
        self.assertEqual(self.fetch("glasses")[1], [])

    def test_an_arm_that_did_not_start_is_not_watched(self):
        r, started = self.fetch("pill bottle", active=False)
        self.assertEqual(started, [])
        self.assertIn("can't start the arm", r["say"])

    def test_with_voice_off_nothing_is_watched(self):
        self.assertEqual(self.fetch("pill bottle", voice=False)[1], [])

    def test_not_being_able_to_watch_never_makes_the_fetch_look_like_it_failed(self):
        r, _ = self.fetch("pill bottle", watcher_raises=True)
        self.assertIn("on its way", r["say"])

    def test_the_server_side_watcher_reports_the_handoff_when_the_arm_stops_cleanly(self):
        recorded = mock.Mock()

        async def go():
            self.app._watch_handoff("pill bottle", lambda: {"active": False, "last_error": None})
            self.assertEqual(len(self.app._handoff_tasks), 1)               # held, so it cannot be collected mid-flight
            while self.app._handoff_tasks:
                await asyncio.sleep(0.01)
        with mock.patch.object(self.app.doses, "record_handoff", recorded):
            asyncio.run(go())
        recorded.assert_called_once_with("pill bottle")

    def test_the_server_side_watcher_reports_nothing_when_the_arm_stops_with_an_error(self):
        recorded = mock.Mock()

        async def go():
            self.app._watch_handoff("pill bottle", lambda: {"active": False, "last_error": "boom"})
            while self.app._handoff_tasks:
                await asyncio.sleep(0.01)
        with mock.patch.object(self.app.doses, "record_handoff", recorded):
            asyncio.run(go())
        recorded.assert_not_called()

    def test_the_page_can_send_the_answer_and_it_is_not_a_patient_screen_feature(self):
        html = (SERVER.parent / "phone" / "agent.html").read_text(encoding="utf-8")
        self.assertIn("confirm_pills: (a) =>", html)
        self.assertIn("/api/pill-answer", html)
        self.assertNotIn('["confirm_pills"', html)                               # not in the Features dialog


# ==================================================================================================
class HowPamIsTold(unittest.TestCase):
    def config(self, **env):
        code = ("import app, json; print(json.dumps({'fns': {f['name']: f for f in app.FUNCTIONS}, "
                "'rule': 'confirm_pills' in app.SYSTEM_PROMPT}))")
        e = {k: v for k, v in os.environ.items() if not k.startswith("PAM_")}
        e.update(env, PYTHONPATH=str(SERVER.parent))
        out = subprocess.run([sys.executable, "-c", code], cwd=SERVER, env=e, capture_output=True, text=True, timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr[-800:])
        import json
        return json.loads(out.stdout.strip().splitlines()[-1])

    def test_on_by_default_deferred_and_limited_to_two_answers(self):
        cfg = self.config()
        fn = cfg["fns"]["confirm_pills"]
        self.assertTrue(fn["defer_until_eot"])                                   # it writes to the record: only after the turn ends
        self.assertEqual(fn["parameters"]["properties"]["answer"]["enum"], ["yes", "not_yet"])
        self.assertEqual(fn["parameters"]["required"], ["answer"])
        self.assertTrue(cfg["rule"])

    def test_with_voice_off_pam_is_not_told_the_function_exists(self):
        cfg = self.config(PAM_VOICE_CONFIRM="off")
        self.assertNotIn("confirm_pills", cfg["fns"])
        self.assertFalse(cfg["rule"])
        self.assertIn("check_pills_taken", cfg["fns"])                           # the rest of the feature is untouched

    def test_with_the_medication_check_off_neither_is_offered(self):
        cfg = self.config(PAM_DOSE_CHECK="off")
        self.assertNotIn("confirm_pills", cfg["fns"])
        self.assertFalse(cfg["rule"])

    def test_the_rule_tells_pam_never_to_assume_yes(self):
        rule = d.VOICE_RULE
        self.assertIn("Never assume yes", rule)
        self.assertIn("do not call it", rule)
        self.assertIn("check", rule)
        self.assertNotIn("take your", rule.lower().replace("took", ""))


if __name__ == "__main__":
    unittest.main()
