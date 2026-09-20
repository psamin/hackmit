"""Tests for adherence.py: the streak and the wording around it. Pure functions; no files, network or key.

    python server/test_adherence.py

The rules that matter most, straight from the design:
  - a LATE tap neither erases the streak nor adds to it
  - a dose nobody tapped resets it, but only after the grace period (a late tap could still arrive)
  - an unfinished day, a day with no doses, and a skipped dose never hurt anyone
  - Pam never says "broke", "lost", "missed", "forgot" or "late" to the patient
"""
import json
import sys
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adherence as a  # noqa: E402
import doses as d  # noqa: E402

BASE = date(2026, 9, 14)                      # a Monday


def ts(day_offset, h=8, m=0):
    day = BASE + timedelta(days=day_offset)
    return datetime(day.year, day.month, day.day, h, m).timestamp()


def dose(day, state, h=8, name="Metformin", source="schedule"):
    """A scheduled dose on day BASE+day, in the given final state."""
    due = ts(day, h)
    x = d.Dose(id=f"{name}|{day}|{h}", text=name, due_ts=due, source=source, group=f"g{day}-{h}", med_id=name.lower(),
               name=name, closes_ts=due + 90 * 60)
    if state == "on_time":
        x.confirmed_ts = due + 300
    elif state == "late":
        x.late_ts = due + 90 * 60 + 600
    elif state == "skipped":
        x.skipped = True
    return x


def days_of(*states, h=8):
    return [dose(i, s, h) for i, s in enumerate(states)]


NOW_AFTER = ts(20)                            # long after every day used below: unresolved doses are final


def summary(doses, now=NOW_AFTER):
    return a.summarize(doses, now)


# ================================================================================
# One dose, one day
# ================================================================================
class DoseStates(unittest.TestCase):
    def test_each_state(self):
        due = ts(0)
        closes = due + 90 * 60
        self.assertEqual(a.dose_state(dose(0, "on_time"), NOW_AFTER), "on_time")
        self.assertEqual(a.dose_state(dose(0, "late"), NOW_AFTER), "late")
        self.assertEqual(a.dose_state(dose(0, "skipped"), NOW_AFTER), "skipped")
        self.assertEqual(a.dose_state(dose(0, "none"), closes - 1), "pending")                      # inside the window
        self.assertEqual(a.dose_state(dose(0, "none"), NOW_AFTER), "unconfirmed")                  # long after everything

    def test_the_grace_period_edges(self):
        x = dose(0, "none")
        edge = x.closes_ts + a.LATE_TAP_GRACE_S
        self.assertEqual(a.dose_state(x, x.closes_ts), "pending")
        self.assertEqual(a.dose_state(x, x.closes_ts + 1), "pending")               # window over, but a late tap could still come
        self.assertEqual(a.dose_state(x, edge), "pending")                          # exactly at the end: still accepted
        self.assertEqual(a.dose_state(x, edge + 1), "unconfirmed")

    def test_a_tap_beats_everything_else(self):
        x = dose(0, "on_time")
        x.unconfirmed = True
        self.assertEqual(a.dose_state(x, NOW_AFTER), "on_time")
        y = dose(0, "late")
        y.unconfirmed = True                       # the window-closed event was logged, then a late tap arrived
        self.assertEqual(a.dose_state(y, NOW_AFTER), "late")

    def test_a_dose_with_no_window_is_never_held_against_anyone(self):
        x = dose(0, "none")
        x.closes_ts = None
        self.assertEqual(a.dose_state(x, NOW_AFTER), "pending")


class DayStates(unittest.TestCase):
    def test_days_from_their_doses(self):
        cases = [([], "none"), (["skipped"], "none"), (["on_time"], "complete"), (["on_time", "on_time"], "complete"),
                 (["on_time", "late"], "late"), (["late"], "late"), (["on_time", "pending"], "pending"),
                 (["late", "pending"], "pending"), (["on_time", "unconfirmed"], "broken"),
                 (["late", "unconfirmed"], "broken"), (["pending", "unconfirmed"], "broken"), (["on_time", "skipped"], "complete"),
                 (["skipped", "skipped"], "none")]
        for states, want in cases:
            self.assertEqual(a.day_state(states), want, states)


# ================================================================================
# THE STREAK
# ================================================================================
class Streak(unittest.TestCase):
    def test_complete_days_in_a_row_count(self):
        s = summary(days_of(*["on_time"] * 5))
        self.assertEqual((s.streak, s.best), (5, 5))

    def test_a_late_tap_does_not_erase_the_streak(self):
        s = summary(days_of(*["on_time"] * 5 + ["late"]))
        self.assertEqual(s.streak, 5)

    def test_and_does_not_add_to_it_either(self):
        with_late = summary(days_of(*["on_time"] * 5 + ["late"])).streak
        without = summary(days_of(*["on_time"] * 5)).streak
        self.assertEqual(with_late, without)
        self.assertEqual(summary(days_of(*["on_time"] * 5 + ["late", "on_time"])).streak, 6)      # the next good day adds one

    def test_a_late_day_in_the_middle_is_neutral(self):
        self.assertEqual(summary(days_of("on_time", "on_time", "on_time", "late", "on_time", "on_time")).streak, 5)

    def test_many_late_days_in_a_row_neither_erase_nor_add(self):
        self.assertEqual(summary(days_of("on_time", "on_time", "late", "late", "late", "late")).streak, 2)

    def test_a_dose_nobody_tapped_resets_it_once_the_grace_period_is_over(self):
        s = summary(days_of(*["on_time"] * 5 + ["none"]))
        self.assertEqual((s.streak, s.best), (0, 5))

    def test_and_after_the_reset_it_starts_again_and_the_best_is_kept(self):
        s = summary(days_of("on_time", "on_time", "on_time", "none", "on_time", "on_time"))
        self.assertEqual((s.streak, s.best), (2, 3))

    def test_an_untapped_dose_inside_the_grace_period_does_not_reset_yet(self):
        doses = days_of(*["on_time"] * 5 + ["none"])
        last = doses[-1]
        inside = last.closes_ts + 3600                                            # an hour after its window: a late tap could still come
        self.assertEqual(a.summarize(doses, inside).streak, 5)
        self.assertEqual(a.summarize(doses, last.closes_ts + a.LATE_TAP_GRACE_S + 1).streak, 0)

    def test_a_late_tap_that_arrives_keeps_the_streak_that_silence_would_have_ended(self):
        doses = days_of(*["on_time"] * 5 + ["none"])
        self.assertEqual(a.summarize(doses, NOW_AFTER).streak, 0)                 # no tap: reset
        doses[-1].late_ts = doses[-1].closes_ts + 1800                            # ...but the person tapped late, in time
        self.assertEqual(a.summarize(doses, NOW_AFTER).streak, 5)                 # so nothing was erased

    def test_today_unfinished_does_not_hurt_and_finished_adds_one(self):
        past = days_of(*["on_time"] * 3)
        today_dose = dose(3, "none")
        s = a.summarize(past + [today_dose], today_dose.due_ts + 60)
        self.assertEqual((s.streak, s.today_complete), (3, False))
        today_dose.confirmed_ts = today_dose.due_ts + 300
        s = a.summarize(past + [today_dose], today_dose.due_ts + 600)
        self.assertEqual((s.streak, s.today_complete), (4, True))

    def test_days_with_no_doses_say_nothing(self):
        # a Monday-to-Friday medication: nothing on the weekend, and the streak simply continues
        doses = [dose(i, "on_time") for i in (0, 1, 2, 3, 4)] + [dose(7, "on_time"), dose(8, "on_time")]
        self.assertEqual(summary(doses).streak, 7)

    def test_a_skipped_dose_is_ignored_entirely(self):
        doses = days_of("on_time", "on_time") + [dose(2, "skipped")] + [dose(3, "on_time")]
        self.assertEqual(summary(doses).streak, 3)
        self.assertEqual(len(summary(doses).days), 3)                              # the skipped-only day is not a day

    def test_one_off_reminders_are_not_part_of_a_streak(self):
        doses = days_of("on_time", "on_time") + [dose(2, "none", source="reminder")]
        self.assertEqual(summary(doses).streak, 2)

    def test_two_medications_the_same_day(self):
        both = [dose(0, "on_time"), dose(0, "on_time", name="Lisinopril")]
        self.assertEqual(summary(both).streak, 1)
        one_late = [dose(0, "on_time"), dose(0, "late", name="Lisinopril")]
        self.assertEqual(summary(one_late).streak, 0)                              # neutral: not added
        self.assertEqual(summary(days_of("on_time") + [dose(1, "on_time"), dose(1, "late", name="Lisinopril")]).streak, 1)
        one_missing = days_of("on_time", "on_time") + [dose(2, "on_time"), dose(2, "none", name="Lisinopril")]
        self.assertEqual(summary(one_missing).streak, 0)

    def test_a_morning_and_an_evening_dose_must_both_be_on_time(self):
        doses = [dose(0, "on_time", h=8), dose(0, "on_time", h=18), dose(1, "on_time", h=8), dose(1, "late", h=18)]
        self.assertEqual(summary(doses).streak, 1)

    def test_input_order_does_not_matter(self):
        doses = days_of(*["on_time"] * 4 + ["none"] + ["on_time"] * 2)
        self.assertEqual(summary(list(reversed(doses))).streak, summary(doses).streak)

    def test_a_window_that_runs_past_midnight_belongs_to_the_day_it_was_due(self):
        x = dose(0, "on_time", h=23)
        x.closes_ts = ts(1, 0, 30)
        y = dose(1, "on_time", h=8)
        s = summary([x, y])
        self.assertEqual([day.day for day in s.days], [BASE, BASE + timedelta(days=1)])
        self.assertEqual(s.streak, 2)

    def test_no_doses_at_all(self):
        s = summary([])
        self.assertEqual((s.streak, s.best, s.days, s.has_history, s.today_complete), (0, 0, [], False, False))

    def test_the_best_run_is_the_best_ever_not_the_last(self):
        s = summary(days_of(*["on_time"] * 6 + ["none"] + ["on_time"] * 2 + ["none"] + ["on_time"]))
        self.assertEqual((s.streak, s.best), (1, 6))

    def test_today_complete_only_when_the_last_day_is_today_and_complete(self):
        doses = days_of("on_time", "on_time")
        self.assertTrue(a.summarize(doses, ts(1, 12)).today_complete)
        self.assertFalse(a.summarize(doses, ts(2, 12)).today_complete)             # the last day was yesterday


class Details(unittest.TestCase):
    def test_a_day_lists_each_dose_with_when_it_was_tapped(self):
        s = summary([dose(0, "on_time"), dose(0, "late", name="Lisinopril"), dose(0, "none", name="Aricept"), dose(0, "skipped", name="Z")])
        day = s.days[0]
        self.assertEqual(day.state, "broken")
        self.assertEqual({x.name: x.state for x in day.doses}, {"Metformin": "on_time", "Lisinopril": "late", "Aricept": "unconfirmed"})
        by = {x.name: x for x in day.doses}
        self.assertIsNotNone(by["Metformin"].confirmed_ts)
        self.assertIsNotNone(by["Lisinopril"].confirmed_ts)                          # the late tap time is kept
        self.assertIsNone(by["Aricept"].confirmed_ts)

    def test_on_time_rate(self):
        doses = days_of("on_time", "on_time", "late", "none")
        self.assertEqual(a.on_time_rate(summary(doses), days=30), 0.5)
        self.assertIsNone(a.on_time_rate(summary([]), days=7))

    def test_pending_doses_do_not_count_against_the_rate(self):
        x = dose(0, "none")
        s = a.summarize([dose(0, "on_time", name="A"), x], x.due_ts + 60)
        self.assertEqual(a.on_time_rate(s, days=7), 1.0)

    def test_the_rate_looks_only_at_recent_days(self):
        doses = [dose(0, "none")] + [dose(i, "on_time") for i in range(15, 20)]
        self.assertEqual(a.on_time_rate(summary(doses), days=7), 1.0)


# ================================================================================
# What Pam says
# ================================================================================
BANNED = ("broke", "broken", "lost", "lose", "fail", "miss", "forgot", "forget", "late", "unconfirmed", "behind", "disappoint",
          "should have", "only ", "unfortunately", "sadly", "worse", "bad")


class Words(unittest.TestCase):
    def sentences(self):
        out = []
        for states in ([], ["on_time"], ["on_time"] * 2, ["on_time"] * 5, ["none"], ["on_time"] * 4 + ["none"],
                       ["on_time"] * 4 + ["none"] + ["on_time"], ["on_time"] * 3 + ["late"], ["late"], ["on_time"] * 8 + ["none", "on_time", "on_time"]):
            s = summary(days_of(*states))
            out += [a.how_am_i_doing(s), a.streak_line(s), a.still_line(s)]
            today = dose(len(states), "on_time")
            out += [a.streak_line(a.summarize(days_of(*states) + [today], today.due_ts + 900))]
        return [x for x in out if x]

    def test_pam_never_blames_and_never_mentions_late_or_unrecorded_doses(self):
        sentences = self.sentences()
        self.assertGreater(len(sentences), 12)
        for text in sentences:
            for word in BANNED:
                self.assertNotIn(word, text.lower(), text)

    def test_it_says_marked_never_took(self):
        for text in self.sentences():
            self.assertNotIn("took", text.lower(), text)             # the camera and a tap cannot know that

    def test_the_streak_line_only_appears_when_the_day_is_finished_and_there_is_something_to_say(self):
        done = a.summarize(days_of("on_time", "on_time"), ts(1, 12))
        self.assertEqual(a.streak_line(done), "That's 2 days in a row.")
        self.assertEqual(a.streak_line(a.summarize(days_of("on_time"), ts(0, 12))), "That's your first day in a row.")
        self.assertEqual(a.streak_line(a.summarize(days_of("on_time", "on_time"), ts(2, 12))), "")      # last day was yesterday
        self.assertEqual(a.streak_line(a.summarize(days_of("late"), ts(0, 12))), "")
        self.assertEqual(a.streak_line(a.summarize([], ts(0, 12))), "")

    def test_after_a_late_tap_pam_only_says_the_streak_is_still_there(self):
        s = summary(days_of("on_time", "on_time", "on_time", "late"))
        self.assertEqual(a.still_line(s), "Your streak is still 3 days.")
        self.assertEqual(a.still_line(summary(days_of("on_time", "late"))), "Your streak is still 1 day.")
        self.assertEqual(a.still_line(summary(days_of("late"))), "")                   # nothing to protect yet

    def test_how_am_i_doing_in_every_situation(self):
        self.assertEqual(a.how_am_i_doing(summary([])), "I'll start counting once you've marked your medication.")
        self.assertEqual(a.how_am_i_doing(summary(days_of(*["on_time"] * 3))),
                         "You've marked all of your medication on time 3 days in a row.")
        self.assertEqual(a.how_am_i_doing(summary(days_of("on_time"))), "You've marked all of your medication on time 1 day in a row.")
        self.assertEqual(a.how_am_i_doing(summary(days_of(*["on_time"] * 5 + ["none"] + ["on_time"] * 2))),
                         "You've marked all of your medication on time 2 days in a row. Your best is 5 days.")
        self.assertEqual(a.how_am_i_doing(summary(days_of(*["on_time"] * 4 + ["none"]))),
                         "Let's start a new streak today. Your best so far is 4 days.")
        self.assertEqual(a.how_am_i_doing(summary(days_of("none"))),
                         "Let's start a streak today. Each day you mark all of your medication on time adds one.")

    def test_a_streak_equal_to_the_best_does_not_repeat_it(self):
        self.assertNotIn("best", a.how_am_i_doing(summary(days_of(*["on_time"] * 4))))

    def test_the_module_itself_never_uses_the_forbidden_words_in_what_pam_says(self):
        source = (Path(__file__).resolve().parent / "adherence.py").read_text(encoding="utf-8")
        strings = [line.split('"')[1] for line in source.splitlines() if 'return "' in line or ' text = f"' in line]
        for text in strings:
            for word in ("broke", "lost", "fail", "miss", "forgot"):
                self.assertNotIn(word, text.lower(), text)


class PublicView(unittest.TestCase):
    def test_the_caregiver_view_is_plain_data(self):
        pub = a.public(summary(days_of("on_time", "late", "none"), ts(3, 12)))
        json.dumps(pub)                                                              # must serialise
        self.assertEqual((pub["streak"], pub["best"]), (0, 1))
        self.assertEqual([x["state"] for x in pub["days"]], ["complete", "late", "broken"])
        first = pub["days"][0]["doses"][0]
        self.assertEqual((first["name"], first["due"], first["state"], first["tapped_at"]), ("Metformin", "08:00", "on_time", "08:05"))
        self.assertIsNone(pub["days"][2]["doses"][0]["tapped_at"])

    def test_it_is_limited_to_the_requested_number_of_days(self):
        doses = days_of(*["on_time"] * 20)
        pub = a.public(summary(doses, ts(19, 20)), days=7)                             # "today" is the 20th day
        self.assertEqual(len(pub["days"]), 7)
        self.assertEqual(pub["streak"], 20)                                            # the streak still counts the whole history

    def test_the_rate_is_rounded_and_absent_without_data(self):
        self.assertEqual(a.public(summary(days_of("on_time", "on_time", "none"), ts(3, 12)))["on_time_rate_7d"], 0.667)
        self.assertIsNone(a.public(summary([]))["on_time_rate_7d"])


if __name__ == "__main__":
    unittest.main(verbosity=1)
