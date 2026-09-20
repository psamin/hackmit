"""Tests for schedule.py: the medication schedule model, checks, storage and audit history.
No network, browser or API key needed.

    python server/test_schedule.py

The ones that matter most: a bad schedule is refused with a message a caregiver can act on and
nothing is written; the history is append-only; a damaged file never looks like "no schedule";
and a crash halfway through a save never loses or garbles the previous schedule.
"""
import io
import json
import os
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import schedule as s  # noqa: E402

TODAY = date(2026, 9, 21)                       # a Monday
NOW = datetime(2026, 9, 21, 9, 30).timestamp()


def med(**kw):
    return {**{"name": "Metformin", "times": ["08:00", "18:00"]}, **kw}


def sched(*meds, **kw):
    return {"medications": list(meds) or [med()], **kw}


def errors(data):
    return s.check(data, TODAY).errors


class Valid(unittest.TestCase):
    def test_a_minimal_medication_is_valid_with_sensible_defaults(self):
        r = s.check(sched(med()), TODAY)
        self.assertTrue(r.ok, r.errors)
        m = r.schedule.medications[0]
        self.assertEqual((m.early_minutes, m.late_minutes, m.active, m.as_needed), (30, 90, True, False))
        self.assertEqual(m.days, list(s.WEEKDAYS))
        self.assertIsNone(m.id)                                   # ids are assigned on save, not on check

    def test_an_as_needed_medication_needs_no_times(self):
        self.assertTrue(s.check(sched({"name": "Ibuprofen", "as_needed": True, "max_per_day": 3}), TODAY).ok)

    def test_an_empty_schedule_is_valid_but_warned_about(self):
        r = s.check({"medications": []}, TODAY)
        self.assertTrue(r.ok)
        self.assertTrue(any("no medications" in w for w in r.warnings))

    def test_a_schedule_object_can_be_checked_too(self):
        self.assertTrue(s.check(s.check(sched(), TODAY).schedule, TODAY).ok)

    def test_the_json_schema_builds_so_a_model_can_be_asked_to_fill_it(self):
        self.assertIn("medications", s.Schedule.model_json_schema()["properties"])


class Times(unittest.TestCase):
    def test_bad_times_are_refused_with_an_example(self):
        for bad in ("8:00", "25:00", "08:60", "8am", "", "08:00:00", "0800", " 08:00"):
            errs = errors(sched(med(times=[bad])))
            self.assertEqual(len(errs), 1, bad)
            self.assertIn("isn't a valid time", errs[0])
            self.assertIn("for example 08:00", errs[0])

    def test_a_non_string_time_is_refused(self):
        self.assertTrue(errors(sched(med(times=[800]))))

    def test_times_are_sorted(self):
        m = s.check(sched(med(times=["20:00", "08:00", "13:00"])), TODAY).schedule.medications[0]
        self.assertEqual(m.times, ["08:00", "13:00", "20:00"])

    def test_a_repeated_time_and_too_many_times_are_refused(self):
        self.assertIn("listed twice", errors(sched(med(times=["08:00", "08:00"])))[0])
        many = [f"{h:02d}:00" for h in range(0, 18, 2)]
        self.assertIn("at most 8", errors(sched(med(times=many, late_minutes=15, early_minutes=0)))[0])

    def test_midnight_and_the_last_minute_are_valid_times(self):
        # on Mondays only there is no dose the next day, so 23:59 does not run into the next 00:00
        self.assertTrue(s.check(sched(med(times=["00:00", "23:59"], days=["mon"], late_minutes=15, early_minutes=0)),
                                TODAY).ok)

    def test_but_a_minute_before_midnight_and_midnight_every_day_overlap(self):
        e = errors(sched(med(times=["00:00", "23:59"], late_minutes=15, early_minutes=0)))
        self.assertIn("11:59 PM and 12:00 AM overlap", e[0])


class Days(unittest.TestCase):
    def test_days_are_sorted_into_week_order(self):
        m = s.check(sched(med(days=["fri", "mon", "wed"])), TODAY).schedule.medications[0]
        self.assertEqual(m.days, ["mon", "wed", "fri"])

    def test_no_days_a_repeated_day_and_a_bad_name_are_refused(self):
        self.assertIn("at least one day", errors(sched(med(days=[])))[0])
        self.assertIn("listed twice", errors(sched(med(days=["mon", "mon"])))[0])
        self.assertTrue(errors(sched(med(days=["monday"]))))


class Text(unittest.TestCase):
    def test_a_name_is_required_trimmed_and_whitespace_collapsed(self):
        self.assertTrue(errors(sched(med(name=""))))
        self.assertTrue(errors(sched(med(name="   "))))
        m = s.check(sched(med(name="  Blood   pressure\n pill ")), TODAY).schedule.medications[0]
        self.assertEqual(m.name, "Blood pressure pill")

    def test_length_limits(self):
        self.assertIn("too long", errors(sched(med(name="x" * 81)))[0])
        self.assertIn("too long", errors(sched(med(note="x" * 301)))[0])
        self.assertTrue(s.check(sched(med(name="x" * 80, note="x" * 300)), TODAY).ok)

    def test_control_characters_are_refused(self):
        self.assertIn("control characters", errors(sched(med(name="Met\x00formin")))[0])
        self.assertIn("control characters", errors(sched(med(note="a\x07b")))[0])

    def test_a_note_may_be_empty_and_is_kept_as_written(self):
        self.assertEqual(s.check(sched(med(note="")), TODAY).schedule.medications[0].note, "")
        m = s.check(sched(med(note="1 tablet with food <b>&")), TODAY).schedule.medications[0]
        self.assertEqual(m.note, "1 tablet with food <b>&")        # escaping is the page's job, not ours

    def test_unicode_is_fine(self):
        self.assertTrue(s.check(sched(med(name="Ácido fólico", note="con el desayuno")), TODAY).ok)

    def test_the_patient_name_is_optional_and_checked(self):
        self.assertTrue(s.check(sched(patient_name="Alex"), TODAY).ok)
        self.assertTrue(s.check(sched(patient_name=None), TODAY).ok)
        self.assertTrue(errors(sched(patient_name="")))


class Numbers(unittest.TestCase):
    def test_ranges(self):
        for field, bad in (("early_minutes", -1), ("early_minutes", 121), ("late_minutes", 14), ("late_minutes", 361),
                           ("min_gap_hours", 0), ("min_gap_hours", 73), ("max_per_day", 0), ("max_per_day", 25)):
            self.assertTrue(errors(sched(med(**{field: bad}))), (field, bad))

    def test_true_and_false_are_not_numbers(self):
        for field in ("early_minutes", "late_minutes", "max_per_day", "min_gap_hours"):
            errs = errors(sched(med(**{field: True})))
            self.assertTrue(errs, field)

    def test_text_that_is_not_a_number_is_refused(self):
        self.assertTrue(errors(sched(med(late_minutes="soon"))))


class UnknownSettings(unittest.TestCase):
    def test_an_unknown_setting_is_refused_and_points_at_the_note(self):
        errs = errors(sched(med(dose="500 mg")))
        self.assertEqual(len(errs), 1)
        self.assertIn("Metformin: 'dose' isn't a setting this schedule understands", errs[0])
        self.assertIn("note", errs[0])

    def test_unknown_top_level_setting_and_wrong_shapes(self):
        self.assertTrue(errors({"medications": [], "doctor": "Dr Patel"}))
        self.assertEqual(errors([1, 2]), ["The schedule isn't in the expected form."])
        self.assertTrue(errors({"medications": "aspirin"}))
        self.assertTrue(errors({"medications": ["aspirin"]}))
        self.assertTrue(errors(None))


class Consistency(unittest.TestCase):
    def test_as_needed_with_times_and_scheduled_without_times(self):
        self.assertIn("'as needed', so it can't also have set times", errors(sched(med(as_needed=True)))[0])
        self.assertIn("needs at least one time", errors(sched(med(times=[])))[0])

    def test_end_before_start(self):
        e = errors(sched(med(start_date="2026-09-20", end_date="2026-09-19")))
        self.assertIn("end date is before the start date", e[0])
        self.assertTrue(s.check(sched(med(start_date="2026-09-20", end_date="2026-09-20")), TODAY).ok)

    def test_daily_maximum_below_the_number_of_times(self):
        self.assertIn("2 times a day but a daily maximum of 1", errors(sched(med(max_per_day=1)))[0])
        self.assertTrue(s.check(sched(med(max_per_day=2)), TODAY).ok)

    def test_minimum_gap_that_contradicts_the_times(self):
        e = errors(sched(med(times=["08:00", "12:00"], min_gap_hours=6)))
        self.assertIn("as close as 4 hours", e[0])
        self.assertIn("minimum gap is 6 hours", e[0])
        self.assertTrue(s.check(sched(med(times=["08:00", "14:00"], min_gap_hours=6, late_minutes=60)), TODAY).ok)

    def test_the_gap_across_midnight_counts_only_when_doses_fall_on_consecutive_days(self):
        # once a day, every day: doses are 24 h apart, so a 30 h minimum is impossible
        self.assertIn("24 hours", errors(sched(med(times=["08:00"], min_gap_hours=30)))[0])
        # Mondays only: no dose the next day, so the same minimum is fine
        self.assertTrue(s.check(sched(med(times=["08:00"], days=["mon"], min_gap_hours=30)), TODAY).ok)
        # Sunday and Monday are consecutive across the week boundary
        self.assertTrue(errors(sched(med(times=["08:00"], days=["sun", "mon"], min_gap_hours=30))))

    def test_overlapping_windows_are_refused_because_a_tap_must_belong_to_one_dose(self):
        e = errors(sched(med(times=["08:00", "09:00"])))
        self.assertIn("8:00 AM and 9:00 AM overlap", e[0])
        self.assertTrue(s.check(sched(med(times=["08:00", "12:00"])), TODAY).ok)

    def test_windows_that_only_touch_still_overlap(self):
        # 08:00 + 90 min = 09:30; 10:00 - 30 min = 09:30: a tap at 09:30 would belong to both
        self.assertTrue(errors(sched(med(times=["08:00", "10:00"]))))
        self.assertTrue(s.check(sched(med(times=["08:00", "10:00"], late_minutes=89)), TODAY).ok)

    def test_windows_overlapping_across_midnight(self):
        self.assertTrue(errors(sched(med(times=["23:00", "00:30"]))))
        self.assertTrue(s.check(sched(med(times=["22:00", "02:00"])), TODAY).ok)

    def test_a_single_dose_a_day_can_never_overlap_itself(self):
        self.assertTrue(s.check(sched(med(times=["08:00"], late_minutes=360, early_minutes=120)), TODAY).ok)


class WholeSchedule(unittest.TestCase):
    def test_two_medications_may_not_share_a_name_in_any_case(self):
        e = errors(sched(med(name="Aspirin"), med(name="aspirin", times=["09:00"])))
        self.assertIn("Two medications are named", e[0])

    def test_duplicate_names_are_caught_in_either_order_and_any_case(self):
        for a, b in (("Aspirin", "aspirin"), ("aspirin", "ASPIRIN"), ("ASPIRIN", "Aspirin")):
            e = errors(sched(med(name=a), med(name=b, times=["12:00"])))
            self.assertTrue(e and "Two medications are named" in e[0], (a, b))

    def test_shared_ids_are_refused(self):
        e = errors(sched(med(name="A", id="x"), med(name="B", id="x")))
        self.assertIn("share the same id", e[0])

    def test_a_bad_id_is_refused(self):
        self.assertTrue(errors(sched(med(id="Bad Id!"))))

    def test_at_most_30_medications(self):
        many = [med(name=f"Med {i}") for i in range(31)]
        self.assertTrue(errors({"medications": many}))
        self.assertTrue(s.check({"medications": many[:30]}, TODAY).ok)

    def test_errors_name_the_medication_and_the_field(self):
        e = errors(sched(med(name="Lisinopril"), med(name="Metformin", times=["25:00"])))
        self.assertEqual(len(e), 1)
        self.assertTrue(e[0].startswith("Metformin: times: '25:00'"))

    def test_an_unnamed_medication_is_referred_to_by_position(self):
        e = errors(sched(med(), {"times": ["08:00"]}))
        self.assertTrue(any(x.startswith("Medication 2:") for x in e), e)

    def test_no_library_jargon_reaches_the_caregiver(self):
        jargon = ("Input should", "Field required", "Value error", "greater than", "less than",
                  "Extra inputs", "type=", "pydantic", "unexpected keyword")
        cases = [med(late_minutes=5), med(late_minutes=999), med(early_minutes=-1), med(min_gap_hours=0),
                 med(max_per_day=0), med(max_per_day="lots"), med(late_minutes=2.5), med(min_gap_hours="x"),
                 med(start_date="tomorrow"), med(end_date=20260921), med(days=["funday"]), med(days="mon"),
                 med(times="08:00"), med(as_needed="maybe"), med(active="perhaps"), med(name=None), med(note=5),
                 med(patient=1), {"times": ["08:00"]}]
        for bad in cases:
            for e in errors(sched(bad)):
                for word in jargon:
                    self.assertNotIn(word, e, (bad, e))
        for e in errors({"medications": [med()] * 2 + [med(name=f"M{i}") for i in range(31)]}):
            for word in jargon:
                self.assertNotIn(word, e, e)

    def test_range_errors_say_the_range_and_the_unit(self):
        self.assertEqual(errors(sched(med(late_minutes=5))), ["Metformin: minutes late: must be between 15 and 360 minutes."])
        self.assertEqual(errors(sched(med(max_per_day=0))), ["Metformin: daily maximum: must be between 1 and 24."])
        self.assertIn("more than 0 and at most 72 hours", errors(sched(med(min_gap_hours=0)))[0])

    def test_a_missing_name_says_so_and_a_bad_date_shows_the_format(self):
        self.assertEqual(errors(sched({"times": ["08:00"]})), ["Medication 1: name: is required."])
        self.assertIn("Use YYYY-MM-DD", errors(sched(med(start_date="tomorrow")))[0])
        self.assertIn("Use mon, tue", errors(sched(med(days=["funday"])))[0])

    def test_every_error_is_a_plain_sentence(self):
        bad = sched(med(times=["25:00"], late_minutes=5, days=[]), med(name="", note="x" * 400), {"name": "Y"})
        found = errors(bad)
        self.assertGreaterEqual(len(found), 4)
        for e in found:
            self.assertNotIn("Value error", e)      # no library jargon leaking through
            self.assertGreater(len(e), 10)


class Warnings(unittest.TestCase):
    def test_night_time_doses_are_flagged_not_blocked(self):
        for t in ("22:00", "03:00", "05:59"):
            r = s.check(sched(med(times=[t])), TODAY)
            self.assertTrue(r.ok)
            self.assertTrue(any("middle of the night" in w for w in r.warnings), t)
        for t in ("06:00", "21:59", "12:00"):
            self.assertFalse(s.check(sched(med(times=[t])), TODAY).warnings, t)

    def test_as_needed_without_a_limit_and_an_ended_course(self):
        r = s.check(sched({"name": "Ibuprofen", "as_needed": True}), TODAY)
        self.assertTrue(any("no daily limit" in w for w in r.warnings))
        r = s.check(sched(med(end_date="2026-09-01")), TODAY)
        self.assertTrue(any("course ended on Sep 1" in w for w in r.warnings))
        self.assertFalse(s.check(sched(med(end_date="2026-09-21")), TODAY).warnings)   # today is the last day

    def test_a_schedule_with_errors_reports_no_warnings(self):
        self.assertEqual(s.check(sched(med(times=["03:00", "bad"])), TODAY).warnings, [])


class Words(unittest.TestCase):
    def test_times_read_in_twelve_hour_form(self):
        self.assertEqual([s.fmt_time(t) for t in ("00:05", "12:00", "12:30", "13:00", "23:59", "08:00")],
                         ["12:05 AM", "12:00 PM", "12:30 PM", "1:00 PM", "11:59 PM", "8:00 AM"])

    def test_durations(self):
        self.assertEqual([s.fmt_duration(m) for m in (0, 1, 30, 60, 90, 120, 121, 360)],
                         ["0 minutes", "1 minute", "30 minutes", "1 hour", "1 hour 30 minutes", "2 hours",
                          "2 hours 1 minute", "6 hours"])

    def one(self, **kw):
        return s.describe_one(s.check(sched(med(**kw)), TODAY).schedule.medications[0])

    def test_a_plain_daily_medication(self):
        self.assertEqual(self.one(), "Metformin: every day at 8:00 AM and 6:00 PM "
                                     "(counts as on time from 30 minutes before to 1 hour 30 minutes after)")

    def test_day_phrases(self):
        self.assertIn("on weekdays at", self.one(days=["mon", "tue", "wed", "thu", "fri"]))
        self.assertIn("on weekends at", self.one(days=["sat", "sun"]))
        self.assertIn("on Monday, Wednesday and Friday at", self.one(days=["fri", "mon", "wed"]))
        self.assertIn("on Tuesday and Thursday at", self.one(days=["thu", "tue"]))
        self.assertIn("every Monday at", self.one(days=["mon"]))

    def test_three_times_read_with_and(self):
        self.assertIn("at 8:00 AM, 1:00 PM and 6:00 PM", self.one(times=["08:00", "13:00", "18:00"], late_minutes=60))

    def test_gap_maximum_dates_pause_and_note(self):
        text = self.one(min_gap_hours=6, max_per_day=2, start_date="2026-09-20", end_date="2026-09-27",
                        active=False, note="With food")
        self.assertIn("at least 6 hours apart", text)
        self.assertIn("no more than 2 a day", text)
        self.assertIn("from Sep 20 to Sep 27", text)
        self.assertIn("[paused: no reminders]", text)
        self.assertTrue(text.endswith('Note from the caregiver: "With food"'))

    def test_one_sided_dates(self):
        self.assertIn(", starting Oct 1", self.one(start_date="2026-10-01"))
        self.assertIn(", until Oct 9", self.one(end_date="2026-10-09"))

    def test_as_needed(self):
        m = s.check(sched({"name": "Ibuprofen", "as_needed": True, "max_per_day": 3}), TODAY).schedule.medications[0]
        self.assertEqual(s.describe_one(m), "Ibuprofen: only when needed, with no reminders, at most 3 a day")

    def test_describe_lists_one_line_per_medication(self):
        sc = s.check(sched(med(), med(name="Lisinopril", times=["08:30"])), TODAY).schedule
        self.assertEqual(len(s.describe(sc)), 2)

    def test_the_note_is_shown_as_the_caregiver_wrote_it_not_reworded(self):
        self.assertIn('"1 tablet with breakfast"', self.one(note="1 tablet with breakfast"))


class Diff(unittest.TestCase):
    def sc(self, *meds):
        return s.check(sched(*meds), TODAY).schedule

    def test_first_save_lists_everything_as_added(self):
        self.assertEqual([c.split(":")[0] for c in s.diff(None, self.sc(med(), med(name="B", times=["12:00"])))],
                         ["Added", "Added"])

    def test_no_change_is_no_lines(self):
        self.assertEqual(s.diff(self.sc(med()), self.sc(med())), [])

    def test_added_removed_and_changed(self):
        old = self.sc(med(), med(name="Lisinopril", times=["09:00"]))
        new = self.sc(med(times=["08:00", "18:00", "22:00"]), med(name="Aspirin", times=["10:00"]))
        lines = s.diff(old, new)
        self.assertEqual(len(lines), 3)
        self.assertTrue(any(l.startswith("Changed: Metformin") and "->" in l for l in lines))
        self.assertIn("Added: Aspirin: every day at 10:00 AM (counts as on time from 30 minutes before to "
                      "1 hour 30 minutes after)", lines)
        self.assertIn("Removed: Lisinopril", lines)

    def test_the_patient_name_change_is_reported(self):
        self.assertIn("Patient name: none -> Alex", s.diff(self.sc(med()), s.check(sched(med(), patient_name="Alex"), TODAY).schedule))

    def test_order_alone_is_not_a_change(self):
        a, b = med(), med(name="Lisinopril", times=["12:00"])
        self.assertEqual(s.diff(self.sc(a, b), self.sc(b, a)), [])

    def test_pausing_is_a_change(self):
        self.assertTrue(s.diff(self.sc(med()), self.sc(med(active=False))))


class Ids(unittest.TestCase):
    def test_slugs(self):
        self.assertEqual(s.slugify("Vitamin D3 (1000 IU)"), "vitamin-d3-1000-iu")
        self.assertEqual(s.slugify("!!!"), "medication")
        self.assertEqual(s.slugify("Ácido"), "cido")                        # documented: non-ASCII letters drop out
        self.assertLessEqual(len(s.slugify("x" * 100)), 40)

    def test_new_medications_get_slug_ids_and_collisions_get_a_suffix(self):
        sc = s.check(sched(med(name="Vitamin D"), med(name="Vitamin-D!", times=["12:00"])), TODAY).schedule
        self.assertEqual([m.id for m in s.assign_ids(sc, None).medications], ["vitamin-d", "vitamin-d-2"])

    def test_a_kept_name_keeps_its_id_through_an_edit(self):
        first = s.assign_ids(s.check(sched(med()), TODAY).schedule, None)
        edited = s.check(sched(med(name="METFORMIN", times=["07:00", "19:00"])), TODAY).schedule
        self.assertEqual(s.assign_ids(edited, first).medications[0].id, first.medications[0].id)

    def test_a_custom_id_is_kept_when_the_medication_is_saved_again_without_one(self):
        # the slug of "Metformin" is "metformin"; the id here is deliberately different, so only
        # matching on the name (not re-deriving the slug) can keep it
        first = s.assign_ids(s.check(sched(med(id="diabetes-am")), TODAY).schedule, None)
        again = s.check(sched(med(times=["07:00", "19:00"])), TODAY).schedule
        self.assertEqual(s.assign_ids(again, first).medications[0].id, "diabetes-am")

    def test_the_store_keeps_that_id_across_saves(self):
        path = Path(tempfile.mkdtemp()) / "m.jsonl"
        store = s.Store(path, clock=lambda: NOW)
        store.save(sched(med(id="diabetes-am")))
        store.save(sched(med(times=["07:00", "19:00"])))
        self.assertEqual(store.load().schedule.medications[0].id, "diabetes-am")

    def test_an_explicit_id_is_kept_and_others_avoid_it(self):
        sc = s.check(sched(med(name="A", id="metformin"), med()), TODAY).schedule
        ids = [m.id for m in s.assign_ids(sc, None).medications]
        self.assertEqual(ids[0], "metformin")
        self.assertNotEqual(ids[1], "metformin")


class Slots(unittest.TestCase):
    def build(self, *meds):
        return s.assign_ids(s.check(sched(*meds), TODAY).schedule, None)

    def test_a_days_doses_in_order_with_windows(self):
        sl = s.slots_for_day(self.build(med(), med(name="Lisinopril", times=["07:00"])), date(2026, 9, 21))
        self.assertEqual([(x.name, x.due.strftime("%H:%M")) for x in sl],
                         [("Lisinopril", "07:00"), ("Metformin", "08:00"), ("Metformin", "18:00")])
        m = sl[1]
        self.assertEqual((m.opens.strftime("%H:%M"), m.closes.strftime("%H:%M")), ("07:30", "09:30"))
        self.assertEqual(m.med_id, "metformin")

    def test_only_the_scheduled_weekdays(self):
        sc = self.build(med(days=["mon", "wed"]))
        self.assertEqual(len(s.slots_for_day(sc, date(2026, 9, 21))), 2)      # Monday
        self.assertEqual(s.slots_for_day(sc, date(2026, 9, 22)), [])          # Tuesday

    def test_a_window_may_run_past_midnight(self):
        sl = s.slots_for_day(self.build(med(times=["23:30"])), date(2026, 9, 21))[0]
        self.assertEqual(sl.closes, datetime(2026, 9, 22, 1, 0))

    def test_courses_start_and_end_inclusively(self):
        sc = self.build(med(start_date="2026-09-21", end_date="2026-09-22"))
        self.assertEqual(len(s.slots_for_day(sc, date(2026, 9, 20))), 0)
        self.assertEqual(len(s.slots_for_day(sc, date(2026, 9, 21))), 2)
        self.assertEqual(len(s.slots_for_day(sc, date(2026, 9, 22))), 2)
        self.assertEqual(len(s.slots_for_day(sc, date(2026, 9, 23))), 0)

    def test_paused_and_as_needed_medications_are_never_due(self):
        sc = self.build(med(active=False), {"name": "Ibuprofen", "as_needed": True})
        self.assertEqual(s.slots_for_day(sc, date(2026, 9, 21)), [])

    def test_the_caregivers_note_travels_with_the_slot(self):
        self.assertEqual(s.slots_for_day(self.build(med(note="With food")), date(2026, 9, 21))[0].note, "With food")

    def test_an_unsaved_schedule_still_gets_ids(self):
        sc = s.check(sched(med()), TODAY).schedule                            # no assign_ids
        self.assertEqual(s.slots_for_day(sc, date(2026, 9, 21))[0].med_id, "metformin")


class StoreBase(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.path = self.dir / "medications.jsonl"
        self.t = NOW
        self.store = s.Store(self.path, clock=lambda: self.t)

    def lines(self):
        return [ln for ln in self.path.read_text(encoding="utf-8").splitlines() if ln.strip()]


class Saving(StoreBase):
    def test_nothing_saved_yet_is_none_not_an_error(self):
        self.assertIsNone(self.store.load())
        self.assertEqual(self.store.history(), [])

    def test_save_then_load(self):
        e = self.store.save(sched(med()), source_text="Metformin twice a day")
        got = self.store.load()
        self.assertEqual((got.version, got.actor, got.action, got.source_text), (1, "caregiver", "save", "Metformin twice a day"))
        self.assertEqual(got.ts, "2026-09-21T09:30:00")
        self.assertEqual(got.schedule.medications[0].id, "metformin")
        self.assertEqual(e.schedule, got.schedule)
        self.assertEqual(got.changes[0][:6], "Added:")

    def test_versions_count_up_and_history_is_newest_first(self):
        for n in range(3):
            self.t += 60
            self.store.save(sched(med(times=["08:00", f"{10 + n}:00"], late_minutes=60)))
        self.assertEqual([e.version for e in self.store.history()], [3, 2, 1])
        self.assertEqual(self.store.load().version, 3)
        self.assertEqual([e.version for e in self.store.history(limit=2)], [3, 2])

    def test_an_invalid_schedule_is_refused_and_nothing_is_written(self):
        with self.assertRaises(s.ScheduleError) as ctx:
            self.store.save(sched(med(times=["25:00"])))
        self.assertIn("isn't a valid time", ctx.exception.errors[0])
        self.assertFalse(self.path.exists())

    def test_an_invalid_edit_leaves_the_current_schedule_in_force(self):
        self.store.save(sched(med()))
        before = self.path.read_bytes()
        with self.assertRaises(s.ScheduleError):
            self.store.save(sched(med(times=[])))
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(self.store.load().version, 1)

    def test_history_is_append_only(self):
        self.store.save(sched(med()))
        first = self.path.read_bytes()
        self.store.save(sched(med(times=["07:00", "19:00"])))
        self.assertTrue(self.path.read_bytes().startswith(first))              # earlier bytes are untouched
        self.assertEqual(len(self.lines()), 2)

    def test_each_entry_says_what_changed_in_words(self):
        self.store.save(sched(med()))
        self.store.save(sched(med(times=["07:00", "19:00"])))
        self.assertTrue(self.store.load().changes[0].startswith("Changed: Metformin"))

    def test_ids_stay_stable_across_edits(self):
        self.store.save(sched(med()))
        self.store.save(sched(med(times=["07:00", "19:00"])))
        self.assertEqual(self.store.load().schedule.medications[0].id, "metformin")

    def test_the_file_is_private_to_the_owner(self):
        self.store.save(sched(med()))
        self.assertEqual(os.stat(self.path).st_mode & 0o777, 0o600)

    def test_unicode_survives_the_round_trip(self):
        self.store.save(sched(med(note="con el desayuno, café ☕")))
        self.assertEqual(self.store.load().schedule.medications[0].note, "con el desayuno, café ☕")
        self.assertIn("café ☕", self.path.read_text(encoding="utf-8"))          # readable, not \u-escaped

    def test_a_custom_actor_is_recorded(self):
        self.store.save(sched(med()), actor="someone-else")
        self.assertEqual(self.store.load().actor, "someone-else")

    def test_saving_an_identical_schedule_makes_a_new_version_with_no_changes(self):
        self.store.save(sched(med()))
        e = self.store.save(sched(med()))
        self.assertEqual((e.version, e.changes), (2, []))

    def test_concurrent_saves_get_distinct_versions_and_whole_lines(self):
        def go(i):
            self.store.save(sched(med(times=["08:00", "18:00"], note=f"n{i}")))
        threads = [threading.Thread(target=go, args=(i,)) for i in range(25)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(sorted(e.version for e in self.store.history()), list(range(1, 26)))
        self.assertEqual(len(self.lines()), 25)


class Restoring(StoreBase):
    def test_restore_saves_the_old_schedule_as_a_new_version(self):
        self.store.save(sched(med(note="original")))
        self.store.save(sched(med(note="changed")))
        e = self.store.restore(1)
        self.assertEqual((e.version, e.action, e.source_text), (3, "restore", "Restored version 1"))
        self.assertEqual(self.store.load().schedule.medications[0].note, "original")
        self.assertEqual(len(self.store.history()), 3)                          # nothing was rewritten

    def test_an_unknown_version_is_refused(self):
        self.store.save(sched(med()))
        with self.assertRaises(s.ScheduleError):
            self.store.restore(7)


class Crashes(StoreBase):
    """What happens to the schedule when the file is not in a perfect state."""

    def test_a_torn_last_line_is_ignored_and_the_previous_schedule_stays_in_force(self):
        self.store.save(sched(med(note="v1")))
        with open(self.path, "ab") as f:
            f.write(b'{"version": 2, "ts": "2026-09-21T10:00:00", "actor": "care')     # power cut mid-write
        self.assertEqual(self.store.load().version, 1)
        self.assertEqual(self.store.load().schedule.medications[0].note, "v1")

    def test_the_next_save_after_a_torn_line_is_not_glued_onto_it(self):
        self.store.save(sched(med(note="v1")))
        with open(self.path, "ab") as f:
            f.write(b'{"version": 2, "ts"')
        e = self.store.save(sched(med(note="v2")))
        self.assertEqual(e.version, 2)
        self.assertEqual(self.store.load().schedule.medications[0].note, "v2")
        self.assertEqual([x.version for x in self.store.history()], [2, 1])

    def test_a_line_that_is_json_but_not_a_valid_schedule_is_skipped(self):
        self.store.save(sched(med(note="v1")))
        bad = {"version": 2, "ts": "x", "actor": "a", "action": "save", "changes": [],
               "schedule": {"medications": [{"name": "X", "times": ["99:99"]}]}}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(bad) + "\n")
        self.assertEqual(self.store.load().version, 1)

    def test_a_file_of_only_junk_is_corrupt_not_empty(self):
        self.path.write_text("this is not json\n{also not\n")
        with self.assertRaises(s.ScheduleCorrupt):
            self.store.load()
        with self.assertRaises(s.ScheduleCorrupt):
            self.store.history()

    def test_saving_over_a_corrupt_file_is_refused_rather_than_guessing(self):
        self.path.write_text("garbage\n")
        with self.assertRaises(s.ScheduleCorrupt):
            self.store.save(sched(med()))
        self.assertEqual(self.path.read_text(), "garbage\n")

    def test_an_empty_or_blank_file_means_no_schedule_yet(self):
        self.path.write_text("")
        self.assertIsNone(self.store.load())
        self.path.write_text("\n\n  \n")
        self.assertIsNone(self.store.load())

    def test_saving_into_an_empty_file_starts_at_version_one(self):
        self.path.write_text("")
        self.assertEqual(self.store.save(sched(med())).version, 1)

    def test_a_junk_line_in_the_middle_does_not_hide_later_entries(self):
        self.store.save(sched(med(note="v1")))
        with open(self.path, "a", encoding="utf-8") as f:
            f.write("junk\n")
        self.store.save(sched(med(note="v2")))
        self.assertEqual(self.store.load().schedule.medications[0].note, "v2")


class Example(unittest.TestCase):
    def test_the_shipped_example_is_valid_and_covers_every_kind_of_medication(self):
        path = Path(__file__).resolve().parent / "medications.example.json"
        r = s.check(json.loads(path.read_text(encoding="utf-8")), date(2026, 9, 21))
        self.assertTrue(r.ok, r.errors)
        meds = r.schedule.medications
        self.assertTrue(any(m.as_needed for m in meds))                        # as needed
        self.assertTrue(any(m.start_date and m.end_date for m in meds))        # a course
        self.assertTrue(any(len(m.days) < 7 for m in meds))                    # some days only
        self.assertTrue(any(len(m.times) > 1 for m in meds))                   # more than once a day
        self.assertFalse(r.warnings)


class Cli(unittest.TestCase):
    def run_cli(self, *argv):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = s._cli(["schedule.py", *argv])
        return code, buf.getvalue()

    def write(self, data):
        p = Path(tempfile.mkdtemp()) / "schedule.json"
        p.write_text(json.dumps(data))
        return str(p)

    def test_check_explains_a_good_schedule_and_saves_nothing(self):
        code, out = self.run_cli("check", self.write(sched(med())))
        self.assertEqual(code, 0)
        self.assertIn("Metformin: every day at 8:00 AM and 6:00 PM", out)
        self.assertIn("nothing was saved", out)

    def test_check_lists_problems_and_exits_nonzero(self):
        code, out = self.run_cli("check", self.write(sched(med(times=["25:00"]))))
        self.assertEqual(code, 1)
        self.assertIn("ERROR    Metformin: times:", out)

    def test_check_shows_warnings(self):
        code, out = self.run_cli("check", self.write(sched(med(times=["03:00"]))))
        self.assertEqual(code, 0)
        self.assertIn("WARNING", out)

    def test_a_missing_or_unreadable_file(self):
        self.assertEqual(self.run_cli("check", "/no/such/file.json")[0], 2)
        p = Path(tempfile.mkdtemp()) / "x.json"
        p.write_text("{not json")
        self.assertEqual(self.run_cli("check", str(p))[0], 2)

    def test_usage(self):
        self.assertEqual(self.run_cli()[0], 2)


if __name__ == "__main__":
    unittest.main(verbosity=1)
