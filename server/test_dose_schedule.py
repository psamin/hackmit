"""Tests for the SCHEDULE-driven part of doses.py: doses that open from the caregiver's saved schedule.
No phone, camera, network or key needed. The reminder-based flow is covered by test_doses.py.

    python server/test_dose_schedule.py

The ones that matter most: medications due together share ONE prompt; a late server start never says
"it's time" for a dose that may already have been taken; a prompt is never spoken with an amount; a dose
whose window closes is `unconfirmed`, never "missed"; the same medication is not prompted twice inside its
minimum gap; and an unreadable schedule is reported, never mistaken for "no reminders".
"""
import json
import os
import sys
import tempfile
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import doses as d  # noqa: E402
import schedule as sched  # noqa: E402

DAY = date(2026, 9, 21)                                   # a Monday
T = d.Timing()                                            # 10 / 20 minute nudges, 15 minute announce grace
CG = {"name": "Mike", "phone": None}
GROUP_8 = "2026-09-21T08:00"


def at(h, m=0, day=DAY):
    return datetime(day.year, day.month, day.day, h, m).timestamp()


MET = {"name": "Metformin", "times": ["08:00", "18:00"], "note": "500mg with food"}
LIS = {"name": "Lisinopril", "times": ["08:00"], "note": "10mg"}
VITD = {"name": "Vitamin D", "times": ["12:00"]}


def entry_for(*meds, saved=at(7), path=None):
    """A saved schedule (with ids) as the store would hand it over."""
    store = sched.Store(path or Path(tempfile.mkdtemp()) / "m.jsonl", clock=lambda: saved)
    return store.save({"medications": list(meds) or [MET]})


def slots_of(entry, day=DAY):
    return sched.slots_for_day(entry.schedule, day)


def meds_of(entry):
    return {m.id: m for m in entry.schedule.medications}


class Sim:
    """Runs plan() over an in-memory log, delivering (or not) whatever it asks to say."""

    def __init__(self, entry, day=DAY, t=T):
        self.slots, self.meds, self.t, self.events = slots_of(entry, day), meds_of(entry), t, []

    def step(self, now, memories=(), deliver=True):
        out = d.plan(d.replay(self.events), [], list(memories), now, self.t, CG, slots=self.slots, meds=self.meds)
        said = []
        for event, msgs in out:
            if msgs and not deliver:
                continue
            self.events.append({**event, "ts": now})
            said += msgs
        return out, said

    def doses(self):
        return d.replay(self.events)


def kinds(out):
    return [e["type"] for e, _ in out]


def speech(msgs):
    return " ".join(m["text"] for m in msgs if m["type"] == "speak")


def card_of(msgs):
    return next(m["card"] for m in msgs if m["type"] == "card")


def memory(h, m, conf=0.9, obj="pill bottle"):
    return {"event": "placed", "object": obj, "confidence": conf, "logged_at": datetime(2026, 9, 21, h, m).isoformat(timespec="seconds"),
            "frames": ["a/after.jpg"]}


# ================================================================================
# Opening a dose and the first prompt
# ================================================================================
class Opening(unittest.TestCase):
    def test_nothing_happens_before_the_scheduled_time(self):
        out, said = Sim(entry_for()).step(at(7, 59))
        self.assertEqual((out, said), ([], []))

    def test_at_the_scheduled_time_a_dose_opens_and_the_person_is_told(self):
        sim = Sim(entry_for())
        out, said = sim.step(at(8, 0))
        self.assertEqual(kinds(out), ["due", "announced"])
        self.assertEqual(speech(said), "It's time for your Metformin. If you're not sure whether you already took it, "
                                       "please check with Mike first.")

    def test_the_dose_keeps_a_snapshot_of_the_medication(self):
        sim = Sim(entry_for())
        sim.step(at(8, 0))
        dose = next(iter(sim.doses().values()))
        self.assertEqual((dose.name, dose.note, dose.source, dose.group), ("Metformin", "500mg with food", "schedule", GROUP_8))
        self.assertEqual((dose.due_ts, dose.closes_ts), (at(8, 0), at(9, 30)))          # on time until 90 minutes after
        self.assertEqual(dose.id, "metformin|2026-09-21|08:00")

    def test_the_card_shows_the_caregivers_note_exactly_and_asks_with_two_buttons(self):
        _, said = Sim(entry_for()).step(at(8, 0))
        card = card_of(said)
        self.assertEqual(card["title"], "Time for your medication")
        self.assertEqual(card["body"], "Metformin: 500mg with food")
        self.assertEqual(card["action"], {"label": "Yes, I took it", "post": "/api/dose/confirm",
                                          "body": {"group": GROUP_8, "answer": "yes"}})
        self.assertEqual(card["action2"]["body"], {"group": GROUP_8, "answer": "not_yet"})

    def test_an_amount_or_note_is_never_spoken(self):
        _, said = Sim(entry_for({"name": "Metformin", "times": ["08:00"], "note": "500mg, 2 tablets with food"})).step(at(8, 0))
        for word in ("500", "mg", "tablet", "food", "2 "):
            self.assertNotIn(word, speech(said))
        self.assertIn("500mg, 2 tablets with food", card_of(said)["body"])                 # ...but it is on the card

    def test_a_medication_with_no_note_shows_just_its_name(self):
        self.assertEqual(card_of(Sim(entry_for(VITD)).step(at(12, 0))[1])["body"], "Vitamin D")

    def test_nothing_is_created_twice(self):
        sim = Sim(entry_for())
        sim.step(at(8, 0))
        out, said = sim.step(at(8, 0, ) + 2)
        self.assertEqual((out, said), ([], []))
        self.assertEqual(len(sim.doses()), 1)

    def test_only_the_scheduled_days_and_only_active_medications(self):
        entry = entry_for({**MET, "days": ["tue"]}, {"name": "Paused", "times": ["08:00"], "active": False},
                          {"name": "Ibuprofen", "as_needed": True, "max_per_day": 3})
        self.assertEqual(Sim(entry).step(at(8, 0)), ([], []))                              # Monday: nothing is due


class Groups(unittest.TestCase):
    def test_medications_due_the_same_minute_share_one_prompt(self):
        sim = Sim(entry_for(MET, LIS))
        out, said = sim.step(at(8, 0))
        self.assertEqual(kinds(out).count("announced"), 1)
        self.assertEqual(len([m for m in said if m["type"] == "speak"]), 1)
        self.assertEqual(speech(said), "It's time for your Lisinopril and Metformin. If you're not sure whether you "
                                       "already took them, please check with Mike first.")
        self.assertEqual(set(card_of(said)["body"].splitlines()), {"Metformin: 500mg with food", "Lisinopril: 10mg"})
        self.assertEqual(card_of(said)["title"], "Time for your medications")
        self.assertEqual(card_of(said)["action"]["label"], "Yes, I took them")

    def test_three_medications_read_with_and(self):
        _, said = Sim(entry_for(MET, LIS, {"name": "Aricept", "times": ["08:00"]})).step(at(8, 0))
        self.assertIn("Aricept, Lisinopril and Metformin", speech(said))

    def test_one_event_names_every_dose_in_the_group(self):
        sim = Sim(entry_for(MET, LIS))
        out, _ = sim.step(at(8, 0))
        announced = next(e for e, _ in out if e["type"] == "announced")
        self.assertEqual(len(announced["doses"]), 2)
        self.assertTrue(all(x.announced for x in sim.doses().values()))

    def test_medications_due_at_different_times_are_prompted_separately(self):
        sim = Sim(entry_for(MET, {"name": "Lisinopril", "times": ["08:05"], "note": "10mg"}))
        _, first = sim.step(at(8, 0))
        _, second = sim.step(at(8, 5))
        self.assertIn("Metformin", speech(first))
        self.assertNotIn("Lisinopril", speech(first))
        self.assertIn("Lisinopril", speech(second))

    def test_the_evening_dose_is_a_separate_prompt_the_same_day(self):
        sim = Sim(entry_for())
        sim.step(at(8, 0))
        _, evening = sim.step(at(18, 0))
        self.assertIn("It's time for your Metformin", speech(evening))
        self.assertEqual(len(sim.doses()), 2)


# ================================================================================
# A server that was not running, or nobody listening
# ================================================================================
class LateStarts(unittest.TestCase):
    def test_a_start_within_the_grace_period_still_announces(self):
        _, said = Sim(entry_for()).step(at(8, 10))
        self.assertIn("It's time for your Metformin", speech(said))

    def test_a_late_start_asks_instead_of_announcing(self):
        # the person may well have taken it at 8: "it's time" would invite a second dose
        sim = Sim(entry_for())
        out, said = sim.step(at(8, 20))
        self.assertEqual(speech(said), "I haven't recorded your Metformin yet. Did you take it? If you're not sure, "
                                       "please check with Mike before taking any.")
        self.assertNotIn("It's time", speech(said))
        self.assertTrue(next(e for e, _ in out if e["type"] == "announced")["late"])
        self.assertEqual(card_of(said)["action"]["body"]["group"], GROUP_8)

    def test_the_grace_period_is_exactly_the_configured_length(self):
        for offset, expect_time in ((T.announce_grace_s, True), (T.announce_grace_s + 1, False)):
            _, said = Sim(entry_for()).step(at(8) + offset)
            self.assertEqual("It's time" in speech(said), expect_time, offset)

    def test_a_window_that_passed_entirely_is_recorded_not_announced(self):
        sim = Sim(entry_for())
        out, said = sim.step(at(9, 45))                                  # window closed at 9:30
        self.assertEqual((kinds(out), said), (["due", "unconfirmed"], []))
        dose = next(iter(sim.doses().values()))
        self.assertTrue(dose.unconfirmed)
        self.assertEqual(next(e for e, _ in out if e["type"] == "unconfirmed")["reason"], "not_running")

    def test_a_prompt_nobody_heard_is_retried_and_then_asked_not_announced(self):
        sim = Sim(entry_for())
        sim.step(at(8, 0), deliver=False)                                # no page open
        self.assertFalse(any(x.announced for x in sim.doses().values()))
        _, said = sim.step(at(8, 30))                                    # a page connects, half an hour late
        self.assertIn("Did you take it?", speech(said))
        self.assertNotIn("It's time", speech(said))

    def test_the_dose_opens_even_when_the_prompt_could_not_be_delivered(self):
        sim = Sim(entry_for())
        sim.step(at(8, 0), deliver=False)
        self.assertEqual(len(sim.doses()), 1)                            # the dose exists (opening says nothing aloud)...
        self.assertFalse(any(x.announced for x in sim.doses().values()))  # ...but was not announced: nobody heard it
        sim.step(at(8, 1))
        self.assertTrue(all(x.announced for x in sim.doses().values()))  # so it is announced as soon as someone can hear


# ================================================================================
# The ladder: announce, ask, nudge, escalate, then quiet
# ================================================================================
class Ladder(unittest.TestCase):
    def test_after_the_announcement_nothing_is_said_until_the_first_nudge(self):
        sim = Sim(entry_for())
        sim.step(at(8, 0))
        for minute in (1, 5, 9):
            self.assertEqual(sim.step(at(8, minute))[1], [], minute)

    def test_the_first_nudge_asks_a_question_and_does_not_instruct(self):
        sim = Sim(entry_for())
        sim.step(at(8, 0))
        out, said = sim.step(at(8) + T.nudge1_s)
        self.assertEqual(kinds(out), ["nudge"])
        self.assertEqual(speech(said), "I haven't recorded your Metformin yet. Did you take it? If you're not sure, "
                                       "please check with Mike before taking any.")

    def test_the_second_step_asks_them_to_check_with_the_caregiver_with_no_phone_action(self):
        sim = Sim(entry_for())
        sim.step(at(8, 0))
        sim.step(at(8) + T.nudge1_s)
        out, said = sim.step(at(8) + T.nudge2_s)
        self.assertEqual(kinds(out), ["escalated"])
        self.assertEqual(speech(said), "I still don't have a record of your Metformin. Please check with Mike before taking any pills.")
        card = card_of(said)
        self.assertNotIn("href", json.dumps(card))
        self.assertNotIn("tel:", json.dumps(card))
        self.assertEqual(card["action2"]["body"]["group"], GROUP_8)

    def test_then_it_stops_and_the_window_closes_as_unconfirmed(self):
        sim = Sim(entry_for())
        for now in (at(8, 0), at(8) + T.nudge1_s, at(8) + T.nudge2_s):
            sim.step(now)
        self.assertEqual(sim.step(at(9, 0))[1], [])
        self.assertEqual(sim.step(at(9, 29))[1], [])
        out, said = sim.step(at(9, 31))
        self.assertEqual((kinds(out), said), (["unconfirmed"], []))
        self.assertTrue(next(iter(sim.doses().values())).unconfirmed)
        self.assertEqual(sim.step(at(12, 0))[1], [])                     # and it never speaks about it again

    def test_prompts_are_never_closer_than_the_minimum_gap(self):
        sim = Sim(entry_for(), t=d.Timing(nudge1_s=60, nudge2_s=120, min_gap_s=300))
        sim.step(at(8, 0))
        self.assertEqual(sim.step(at(8, 2))[1], [])                      # nudge1 is due but the gap has not passed
        self.assertNotEqual(sim.step(at(8, 5))[1], [])

    def test_a_confirmed_dose_is_left_alone(self):
        sim = Sim(entry_for())
        sim.step(at(8, 0))
        sim.events.append({"type": "confirmed", "doses": list(sim.doses()), "ts": at(8, 5)})
        self.assertEqual(sim.step(at(8) + T.nudge2_s + 60), ([], []))

    def test_the_group_is_nudged_together_once(self):
        sim = Sim(entry_for(MET, LIS))
        sim.step(at(8, 0))
        out, said = sim.step(at(8) + T.nudge1_s)
        self.assertEqual(kinds(out), ["nudge"])
        self.assertIn("Lisinopril and Metformin", speech(said))
        self.assertIn("Did you take them?", speech(said))


class Evidence(unittest.TestCase):
    def test_a_moved_bottle_after_the_announcement_prompts_one_question(self):
        sim = Sim(entry_for())
        sim.step(at(8, 0))
        out, said = sim.step(at(8, 6), memories=[memory(8, 3)])
        self.assertEqual(kinds(out), ["evidence", "asked"])
        self.assertEqual(speech(said), "I saw your pill bottle move. Did you take your Metformin? If you're not sure, "
                                       "please check with Mike before taking any.")
        sim.step(at(8, 8), memories=[memory(8, 3)])
        sim.step(at(8, 9), memories=[memory(8, 3)])
        self.assertEqual(len([e for e in sim.events if e["type"] == "asked"]), 1)        # asked only once

    def test_evidence_covers_every_medication_in_the_group(self):
        sim = Sim(entry_for(MET, LIS))
        sim.step(at(8, 0))
        sim.step(at(8, 6), memories=[memory(8, 3)])
        self.assertTrue(all(x.evidence and x.asked for x in sim.doses().values()))

    def test_the_question_waits_for_the_minimum_gap_after_the_announcement(self):
        sim = Sim(entry_for())
        sim.step(at(8, 0))
        self.assertEqual(sim.step(at(8, 1), memories=[memory(8, 0, )])[1], [])          # only a minute after speaking
        self.assertNotEqual(sim.step(at(8, 6), memories=[memory(8, 0)])[1], [])

    def test_only_a_bottle_moved_inside_the_window_counts(self):
        sim = Sim(entry_for())
        sim.step(at(8, 0))
        sim.step(at(8, 10), memories=[memory(7, 50), memory(9, 45), memory(8, 12, conf=0.2), memory(8, 12, obj="water bottle")])
        self.assertTrue(all(x.evidence is None for x in sim.doses().values()))

    def test_evidence_never_confirms_anything(self):
        sim = Sim(entry_for())
        sim.step(at(8, 0))
        for minute in (6, 12, 30):
            sim.step(at(8, minute), memories=[memory(8, 3)])
        self.assertTrue(all(x.confirmed_ts is None for x in sim.doses().values()))
        self.assertNotIn("confirmed", [e["type"] for e in sim.events])


# ================================================================================
# Not prompting when it would be unsafe
# ================================================================================
class Guards(unittest.TestCase):
    def setup(self, **med):
        entry = entry_for({"name": "Warfarin", "times": ["08:00"], **med}, path=None)
        return Sim(entry)

    def confirmed(self, sim, when, med_id="warfarin", day_key="2026-09-21|06:00"):
        sim.events += [{"type": "due", "dose": f"{med_id}|{day_key}", "text": "Warfarin", "due_ts": when, "source": "schedule",
                        "group": "g0", "med_id": med_id, "name": "Warfarin", "note": "", "closes_ts": when + 5400, "ts": when},
                       {"type": "confirmed", "dose": f"{med_id}|{day_key}", "ts": when}]

    def test_the_same_medication_taken_recently_is_not_prompted(self):
        sim = self.setup(min_gap_hours=6)
        self.confirmed(sim, at(5, 30))                                    # taken at 5:30; the 8:00 dose is inside 6 hours
        out, said = sim.step(at(8, 0))
        self.assertEqual(said, [])
        self.assertEqual(next(e for e, _ in out if e["type"] == "skipped")["reason"], "taken recently")
        self.assertTrue(sim.doses()["warfarin|2026-09-21|08:00"].skipped)

    def test_outside_the_gap_it_is_prompted_as_normal(self):
        sim = self.setup(min_gap_hours=2)
        self.confirmed(sim, at(5, 30))
        self.assertIn("It's time for your Warfarin", speech(sim.step(at(8, 0))[1]))

    def test_the_daily_maximum_stops_further_prompts(self):
        sim = self.setup(max_per_day=1)
        self.confirmed(sim, at(6, 0))
        out, said = sim.step(at(8, 0))
        self.assertEqual(said, [])
        self.assertEqual(next(e for e, _ in out if e["type"] == "skipped")["reason"], "daily maximum reached")

    def test_below_the_daily_maximum_it_is_prompted(self):
        sim = self.setup(max_per_day=2)
        self.confirmed(sim, at(6, 0))
        self.assertIn("It's time", speech(sim.step(at(8, 0))[1]))

    def test_another_medications_recent_dose_does_not_block_this_one(self):
        sim = self.setup(min_gap_hours=6)
        self.confirmed(sim, at(5, 30), med_id="something-else")
        self.assertIn("It's time", speech(sim.step(at(8, 0))[1]))

    def test_a_skipped_dose_is_final_and_leaves_the_rest_of_the_group_prompted(self):
        entry = entry_for({"name": "Warfarin", "times": ["08:00"], "min_gap_hours": 6}, LIS)
        sim = Sim(entry)
        self.confirmed(sim, at(5, 30))
        _, said = sim.step(at(8, 0))
        self.assertIn("Lisinopril", speech(said))
        self.assertNotIn("Warfarin", speech(said))
        for minute in (20, 40):
            self.assertNotIn("Warfarin", speech(sim.step(at(8, minute))[1]))       # later prompts never mention it either


class SkipsAreFinal(unittest.TestCase):
    def test_a_skipped_dose_stays_skipped_after_the_gap_passes(self):
        # taken at 5:30 with a 3 hour minimum gap: the 8:00 dose is skipped. At 8:35 the gap is over,
        # but this dose was already judged, and must not start prompting halfway through its window.
        entry = entry_for({"name": "Warfarin", "times": ["08:00"], "min_gap_hours": 3})
        sim = Sim(entry)
        sim.events += [{"type": "due", "dose": "warfarin|2026-09-21|05:30", "text": "Warfarin", "due_ts": at(5, 30), "source": "schedule",
                        "group": "g0", "med_id": "warfarin", "name": "Warfarin", "note": "", "closes_ts": at(7), "ts": at(5, 30)},
                       {"type": "confirmed", "dose": "warfarin|2026-09-21|05:30", "ts": at(5, 30)}]
        self.assertEqual(sim.step(at(8, 0))[1], [])
        for minute in (20, 35, 50):
            self.assertEqual(sim.step(at(8, minute))[1], [], minute)
        self.assertEqual(len([e for e in sim.events if e["type"] == "skipped"]), 1)      # judged once, not re-judged
        self.assertTrue(sim.doses()["warfarin|2026-09-21|08:00"].skipped)

    def test_a_second_confirmation_in_the_log_never_moves_the_first_time(self):
        events = [{"type": "due", "dose": "x|d|08:00", "text": "X", "due_ts": 1000.0, "source": "schedule", "group": "g",
                   "med_id": "x", "name": "X", "note": "", "closes_ts": 9000.0, "ts": 1000.0},
                  {"type": "confirmed", "doses": ["x|d|08:00"], "ts": 1500.0},
                  {"type": "confirmed", "doses": ["x|d|08:00"], "ts": 2500.0}]
        self.assertEqual(d.replay(events)["x|d|08:00"].confirmed_ts, 1500.0)


# ================================================================================
# The tap
# ================================================================================
class Files(unittest.TestCase):
    """Point the module at temporary files and a clean environment."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        contacts = self.tmp / "contacts.json"
        contacts.write_text(json.dumps({"caregiver": {"name": "Mike", "phone": "+15557654321"}}))
        self.store = sched.Store(self.tmp / "medications.jsonl", clock=lambda: at(7))
        self.patches = [mock.patch.object(d, "LOG", self.tmp / "doses.jsonl"), mock.patch.object(d, "CONTACTS", contacts),
                        mock.patch.object(d, "KILL_FILE", self.tmp / ".dose_check_off"),
                        mock.patch.object(d, "SCHEDULE_KILL_FILE", self.tmp / ".schedule_reminders_off"),
                        mock.patch.object(d, "schedule_store", lambda: self.store), mock.patch.dict(os.environ, {}, clear=False)]
        for p in self.patches:
            p.start()
        for var in ("PAM_DOSE_CHECK", "PAM_DOSE_DEMO", "PAM_SCHEDULE_REMINDERS"):
            os.environ.pop(var, None)
        d._HEALTH.update(state="none", version=None, at=None)
        self.memfile = self.tmp / "memory.jsonl"
        self.pushed = []

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def push(self, msg):
        self.pushed.append(msg)
        return 1

    def save(self, *meds, saved=at(7)):
        self.store.clock = lambda: saved
        return self.store.save({"medications": list(meds) or [MET]})

    def tick(self, now, deliver=True):
        return d.tick([], self.memfile, self.push if deliver else (lambda m: 0), now, schedule_fn=lambda: self.store.load())

    def events(self):
        return d.read_events()

    def last_said(self):
        return [m["text"] for m in self.pushed if m["type"] == "speak"][-1]


class Confirming(Files):
    def open(self, *meds):
        self.save(*meds)
        self.tick(at(8, 0))

    def test_yes_confirms_every_open_dose_in_the_group_with_one_event(self):
        self.open(MET, LIS)
        r = d.confirm(None, "yes", at(8, 5), group=GROUP_8)
        self.assertEqual((r["ok"], r["recorded"]), (True, True))
        self.assertEqual(r["say"], "Thank you. I've noted that you took your Lisinopril and Metformin at 8:05 AM.")
        confirmed = [e for e in self.events() if e["type"] == "confirmed"]
        self.assertEqual(len(confirmed), 1)
        self.assertEqual(len(confirmed[0]["doses"]), 2)
        self.assertTrue(all(x.confirmed_ts == at(8, 5) for x in d.replay(self.events()).values()))

    def test_tapping_twice_records_once_and_keeps_the_first_time(self):
        self.open()
        first = d.confirm(None, "yes", at(8, 5), group=GROUP_8)
        second = d.confirm(None, "yes", at(8, 20), group=GROUP_8)
        self.assertEqual(second["say"], first["say"])
        self.assertEqual(len([e for e in self.events() if e["type"] == "confirmed"]), 1)

    def test_not_yet_records_nothing_but_the_answer(self):
        self.open()
        r = d.confirm(None, "not_yet", at(8, 5), group=GROUP_8)
        self.assertEqual(r["say"], "Okay. If you're not sure, please check with Mike before taking any.")
        self.assertTrue(all(x.confirmed_ts is None for x in d.replay(self.events()).values()))

    def test_a_single_dose_can_be_confirmed_by_its_id(self):
        self.open(MET, LIS)
        r = d.confirm("metformin|2026-09-21|08:00", "yes", at(8, 5))
        self.assertEqual(r["say"], "Thank you. I've noted that you took your Metformin at 8:05 AM.")
        doses = d.replay(self.events())
        self.assertIsNotNone(doses["metformin|2026-09-21|08:00"].confirmed_ts)
        self.assertIsNone(doses["lisinopril|2026-09-21|08:00"].confirmed_ts)

    def test_a_tap_after_the_window_does_not_count_and_says_so(self):
        self.open()
        r = d.confirm(None, "yes", at(9, 45), group=GROUP_8)                   # the window closed at 9:30
        self.assertFalse(r["ok"])
        self.assertIn("That reminder has passed. Please check with Mike about your Metformin.", r["say"])
        self.assertNotIn("confirmed", [e["type"] for e in self.events()])

    def test_a_tap_on_the_last_minute_of_the_window_counts(self):
        self.open()
        self.assertTrue(d.confirm(None, "yes", at(9, 30), group=GROUP_8)["ok"])

    def test_only_the_doses_still_in_their_window_are_confirmed(self):
        self.open({**MET, "times": ["08:00"], "late_minutes": 120}, {**LIS, "late_minutes": 60})
        r = d.confirm(None, "yes", at(9, 30), group=GROUP_8)                   # Lisinopril closed at 9:00, Metformin at 10:00
        self.assertEqual(r["say"], "Thank you. I've noted that you took your Metformin at 9:30 AM.")
        doses = d.replay(self.events())
        self.assertIsNone(doses["lisinopril|2026-09-21|08:00"].confirmed_ts)

    def test_an_unknown_group_or_dose_is_refused_and_writes_nothing(self):
        self.open()
        before = self.events()
        for kwargs in ({"group": "2099-01-01T08:00"}, {"group": "nope"}):
            self.assertFalse(d.confirm(None, "yes", at(8, 5), **kwargs)["ok"])
        self.assertFalse(d.confirm("nope|x|y", "yes", at(8, 5))["ok"])
        self.assertEqual(self.events(), before)

    def test_malformed_ids_are_refused_not_crashed_on(self):
        self.open()
        for dose_id, group in ((["x"], None), ({"a": 1}, None), (None, ["x"]), (None, 123), (1.5, None)):
            self.assertFalse(d.confirm(dose_id, "yes", at(8, 5), group=group)["ok"], (dose_id, group))

    def test_an_answer_that_is_not_yes_or_not_yet_is_refused(self):
        self.open()
        r = d.confirm(None, "maybe", at(8, 5), group=GROUP_8)
        self.assertEqual((r["ok"], r["say"]), (False, "I didn't understand that answer."))

    def test_the_kill_switch_stops_taps_from_recording_anything(self):
        self.open()
        (self.tmp / ".dose_check_off").touch()
        r = d.confirm(None, "yes", at(8, 5), group=GROUP_8)
        self.assertFalse(r["enabled"])
        self.assertNotIn("confirmed", [e["type"] for e in self.events()])


# ================================================================================
# What Pam says when asked
# ================================================================================
class Status(Files):
    def setUp(self):
        super().setUp()
        self.save(MET, LIS, VITD)

    def ask(self, now):
        return d.status_response(now)

    def test_a_confirmed_medication_is_read_back_and_the_others_have_no_record_yet(self):
        self.tick(at(8, 0))
        d.confirm("metformin|2026-09-21|08:00", "yes", at(8, 5))
        r = self.ask(at(8, 30))
        self.assertIn("You marked your Metformin as taken at 8:05 AM.", r["say"])
        self.assertIn("I don't have a record for your Lisinopril at 8:00 AM.", r["say"])
        self.assertIn("I can't see inside the bottle. Please check with Mike or your pill organiser before taking any pills.", r["say"])
        self.assertIn("Your next one is your Vitamin D at 12:00 PM.", r["say"])
        self.assertFalse(r["recorded"])

    def test_when_everything_due_is_confirmed_there_is_no_caution_and_it_is_recorded(self):
        self.tick(at(8, 0))
        d.confirm(None, "yes", at(8, 5), group=GROUP_8)
        r = self.ask(at(8, 30))
        self.assertTrue(r["recorded"])
        self.assertNotIn("can't see inside", r["say"])
        self.assertIn("You marked your Lisinopril and Metformin as taken at 8:05 AM.", r["say"])     # one sentence, not two

    def test_medications_with_the_same_times_share_a_sentence_and_different_times_do_not(self):
        self.tick(at(8, 0))
        d.confirm("metformin|2026-09-21|08:00", "yes", at(8, 5))
        d.confirm("lisinopril|2026-09-21|08:00", "yes", at(8, 20))
        say = self.ask(at(8, 30))["say"]
        self.assertIn("You marked your Metformin as taken at 8:05 AM.", say)
        self.assertIn("You marked your Lisinopril as taken at 8:20 AM.", say)
        self.assertEqual(say.count("You marked"), 2)

    def test_the_answer_never_repeats_a_sentence_for_medications_due_together(self):
        self.tick(at(8, 0))
        say = self.ask(at(8, 30))["say"]
        self.assertEqual(say.count("I don't have a record for your"), 1)
        d.confirm(None, "yes", at(8, 35), group=GROUP_8)
        self.assertEqual(self.ask(at(8, 40))["say"].count("You marked"), 1)

    def test_a_medication_taken_twice_lists_both_times(self):
        self.tick(at(8, 0))
        d.confirm(None, "yes", at(8, 5), group=GROUP_8)
        self.tick(at(18, 0))
        d.confirm("metformin|2026-09-21|18:00", "yes", at(18, 10))
        self.assertIn("You marked your Metformin as taken at 8:05 AM and 6:10 PM.", self.ask(at(18, 30))["say"])

    def test_an_unconfirmed_dose_is_never_called_missed_or_denied(self):
        self.tick(at(8, 0))
        self.tick(at(9, 45))                                              # window closed, nobody tapped
        say = self.ask(at(10, 0))["say"]
        self.assertIn("I don't have a record for your Lisinopril and Metformin at 8:00 AM.", say)
        for banned in ("missed", "forgot", "haven't taken", "have not taken", "didn't take", "not taken", "you haven't"):
            self.assertNotIn(banned, say.lower())

    def test_yesterdays_doses_do_not_appear_today(self):
        self.tick(at(8, 0))
        d.confirm(None, "yes", at(8, 5), group=GROUP_8)
        tomorrow = at(7, 0, DAY + timedelta(days=1))
        say = self.ask(tomorrow)["say"]
        self.assertNotIn("You marked", say)

    def test_a_paused_or_skipped_dose_is_left_out(self):
        self.tick(at(8, 0))
        self.events()
        d.append({"type": "skipped", "dose": "vitamin-d|2026-09-21|12:00", "reason": "taken recently"}, at(11, 0))
        self.assertNotIn("Vitamin D", self.ask(at(11, 30))["say"].replace("Your next one is your Vitamin D", ""))

    def test_before_the_first_dose_of_the_day_it_is_cautious_and_says_what_comes_next(self):
        r = self.ask(at(6, 0))
        self.assertFalse(r["recorded"])
        self.assertIn("I don't have a record that you took your pills.", r["say"])
        self.assertTrue(r["say"].endswith("Your next one is your Lisinopril at 8:00 AM."), r["say"])
        for banned in ("missed", "forgot", "haven't taken", "you haven't"):
            self.assertNotIn(banned, r["say"].lower())

    def test_with_only_one_off_reminders_the_original_answer_is_kept_plus_the_next_scheduled_one(self):
        d.append({"type": "due", "dose": 1, "text": "take your pills", "due_ts": at(8, 0)}, at(8, 0))
        d.append({"type": "confirmed", "dose": 1}, at(8, 5))
        self.assertEqual(d.status_response(at(9, 0))["say"],
                         "You marked your pills as taken at 8:05 AM. Your next one is your Vitamin D at 12:00 PM.")

    def test_without_any_schedule_the_original_answers_are_unchanged(self):
        self.store.path.unlink()
        d.append({"type": "due", "dose": 1, "text": "take your pills", "due_ts": at(8, 0)}, at(8, 0))
        d.append({"type": "confirmed", "dose": 1}, at(8, 5))
        self.assertEqual(d.status_response(at(9, 0))["say"], "You marked your pills as taken at 8:05 AM.")

    def test_a_one_off_confirmation_alongside_scheduled_doses_is_mentioned(self):
        self.tick(at(8, 0))
        d.append({"type": "due", "dose": 5, "text": "take your pills", "due_ts": at(9, 0)}, at(9, 0))
        d.append({"type": "confirmed", "dose": 5}, at(9, 5))
        self.assertIn("You also marked your pills as taken at 9:05 AM.", self.ask(at(9, 10))["say"])

    def test_the_answer_works_without_a_schedule_file(self):
        self.store.path.unlink()
        self.assertIn("I don't have a record", d.status_response(at(8, 0))["say"])

    def test_the_answer_survives_a_damaged_schedule_file(self):
        self.tick(at(8, 0))
        self.store.path.write_text("garbage\n")
        self.assertIn("I don't have a record for your Lisinopril and Metformin at 8:00 AM.", self.ask(at(8, 30))["say"])   # no crash

    def test_the_kill_switch_defers_to_the_caregiver(self):
        (self.tmp / ".dose_check_off").touch()
        r = self.ask(at(8, 30))
        self.assertFalse(r["enabled"])
        self.assertIn("Please ask Mike about your pills", r["say"])


# ================================================================================
# The rules about what Pam may say, checked over every scheduled sentence
# ================================================================================
class Safety(Files):
    DENIALS = ("haven't taken", "have not taken", "didn't take", "did not take", "not taken", "you haven't", "no, you",
               "you forgot", "you missed", "missed")
    INSTRUCTIONS = ("take your pills now", "you should take", "go ahead and take", "take another", "take your medication",
                    "skip", "double", "you need to take", "please take", "take it now")

    def every_sentence(self):
        entry = entry_for(MET, LIS, VITD)
        sim = Sim(entry)
        said = []
        for now in (at(8, 0), at(8, 6), at(8, 16), at(8, 30)):        # announce, ask, nudge, escalate
            said += sim.step(now, memories=[memory(8, 3)])[1]
        late = Sim(entry_for()).step(at(8, 30))[1]
        return [m["text"] for m in said + late if m["type"] == "speak"]

    def test_no_scheduled_sentence_denies_or_instructs(self):
        sentences = self.every_sentence()
        self.assertGreaterEqual(len(sentences), 5)
        for text in sentences:
            for banned in self.DENIALS + self.INSTRUCTIONS:
                self.assertNotIn(banned, text.lower(), text)

    def test_every_scheduled_sentence_points_at_the_caregiver(self):
        for text in self.every_sentence():
            self.assertIn("check with Mike", text)

    def test_no_scheduled_sentence_contains_an_amount(self):
        for text in self.every_sentence():
            for word in ("500", "10mg", "mg", "tablet"):
                self.assertNotIn(word, text)

    def test_a_note_that_looks_like_an_instruction_is_shown_not_spoken(self):
        sim = Sim(entry_for({"name": "Metformin", "times": ["08:00"], "note": "Take 2 tablets, do not skip"}))
        _, said = sim.step(at(8, 0))
        self.assertNotIn("skip", speech(said))
        self.assertIn("Take 2 tablets, do not skip", card_of(said)["body"])

    def test_the_camera_alone_never_records_a_dose(self):
        sim = Sim(entry_for(MET, LIS))
        for now in (at(8, 0), at(8, 6), at(8, 12), at(8, 25), at(9, 0)):
            sim.step(now, memories=[memory(8, 3)])
        self.assertNotIn("confirmed", [e["type"] for e in sim.events])


# ================================================================================
# Running it: tick(), reading the schedule, and what happens when it is wrong
# ================================================================================
class Ticking(Files):
    def test_the_whole_day_end_to_end(self):
        self.save(MET, LIS)
        self.tick(at(8, 0))
        self.assertEqual([m["type"] for m in self.pushed], ["speak", "card"])
        d.confirm(None, "yes", at(8, 5), group=GROUP_8)
        self.pushed.clear()
        self.tick(at(8) + T.nudge2_s + 60)
        self.assertEqual(self.pushed, [])                                    # answered: no more prompts
        self.tick(at(18, 0))
        self.assertIn("It's time for your Metformin", self.last_said())

    def test_a_prompt_with_nobody_listening_is_not_logged_and_is_retried(self):
        self.save()
        self.tick(at(8, 0), deliver=False)
        self.assertNotIn("announced", [e["type"] for e in self.events()])
        self.tick(at(8, 1))
        self.assertIn("announced", [e["type"] for e in self.events()])

    def test_a_dose_never_opens_for_a_window_that_closed_before_the_schedule_existed(self):
        self.save(MET, saved=at(10, 0))                                      # saved at 10:00: the 8:00 window (to 9:30) is history
        self.tick(at(10, 5))
        self.assertEqual(self.events(), [])

    def test_a_schedule_saved_inside_a_window_prompts_for_that_dose(self):
        self.save(MET, saved=at(8, 40))
        self.tick(at(8, 45))
        self.assertIn("Did you take it?", self.last_said())                 # 45 min after 8:00: asked, not announced

    def test_yesterdays_window_that_runs_past_midnight_is_still_watched(self):
        self.save({"name": "Melatonin", "times": ["23:30"], "late_minutes": 90})
        self.tick(at(0, 30, DAY + timedelta(days=1)))                       # 12:30 AM the next day
        doses = d.replay(self.events())
        self.assertEqual(list(doses), ["melatonin|2026-09-21|23:30"])
        self.assertIn("Did you take it?", self.last_said())                 # an hour late: a question

    def test_editing_the_schedule_later_never_rewrites_a_dose_already_logged(self):
        self.save(MET)
        self.tick(at(8, 0))
        self.save({"name": "Metformin XR", "times": ["08:00"], "note": "a different note"}, saved=at(8, 30))
        self.tick(at(8, 31))
        dose = d.replay(self.events())["metformin|2026-09-21|08:00"]
        self.assertEqual((dose.name, dose.note), ("Metformin", "500mg with food"))

    def test_a_medication_that_keeps_its_name_keeps_its_dose_history_through_an_edit(self):
        self.save(MET)
        self.tick(at(8, 0))
        d.confirm(None, "yes", at(8, 5), group=GROUP_8)
        self.save({**MET, "times": ["08:00", "18:00"], "note": "with dinner too"}, saved=at(8, 30))
        self.tick(at(8, 31))
        self.assertEqual(len([e for e in self.events() if e["type"] == "due"]), 1)     # no second 8:00 dose


class ScheduleHealth(Files):
    def test_no_schedule_yet(self):
        self.tick(at(8, 0))
        self.assertEqual(d.schedule_health()["state"], "none")

    def test_an_active_schedule_reports_its_version_and_that_it_is_running(self):
        self.save()
        self.tick(at(8, 0))
        h = d.schedule_health()
        self.assertEqual((h["state"], h["version"], h["running"]), ("active", 1, True))

    def test_never_ticked_means_not_running(self):
        self.assertFalse(d.schedule_health()["running"])

    def test_a_damaged_schedule_is_reported_and_sends_nothing_rather_than_looking_fine(self):
        self.save()
        self.store.path.write_text("garbage\n")
        with mock.patch.object(d, "_report") as report:
            self.tick(at(8, 0))
        self.assertEqual(d.schedule_health()["state"], "damaged")
        self.assertEqual(self.pushed, [])
        self.assertIn("damaged", report.call_args[0][0])
        self.assertIn("NO scheduled reminders", report.call_args[0][0])

    def test_it_recovers_when_the_file_is_fixed(self):
        self.save()
        good = self.store.path.read_text()
        self.store.path.write_text("garbage\n")
        self.tick(at(8, 0))
        self.store.path.write_text(good)
        self.tick(at(8, 1))
        self.assertEqual(d.schedule_health()["state"], "active")
        self.assertTrue(self.pushed)

    def test_any_other_failure_reading_the_schedule_is_contained(self):
        def boom():
            raise RuntimeError("disk on fire")
        with mock.patch.object(d, "_report"):
            d.tick([], self.memfile, self.push, at(8, 0), schedule_fn=boom)
        self.assertEqual(d.schedule_health()["state"], "error")

    def test_the_watcher_loop_survives_a_failing_tick(self):
        import asyncio

        async def go():
            calls = []
            def reminders():
                calls.append(1)
                if len(calls) == 1:
                    raise RuntimeError("first tick fails")
                return []
            task = asyncio.create_task(d.watch(reminders, self.memfile, self.push, interval=0.01,
                                               schedule_fn=lambda: self.store.load()))
            await asyncio.sleep(0.1)
            task.cancel()
            return len(calls)
        self.assertGreaterEqual(asyncio.run(go()), 2)

    def test_the_report_is_rate_limited(self):
        d._last_report[0] = float("-inf")
        with mock.patch("builtins.print") as pr:
            d._report("one")
            d._report("two")
        self.assertEqual(pr.call_count, 1)
        d._last_report[0] = float("-inf")


class KillSwitches(Files):
    def test_the_schedule_switch_stops_scheduled_reminders_only(self):
        self.save()
        os.environ["PAM_SCHEDULE_REMINDERS"] = "off"
        d.append({"type": "due", "dose": 7, "text": "take your pills", "due_ts": at(8, 0)}, at(8, 0))
        reminders = [{"id": 7, "text": "take your pills", "due_ts": at(8, 0), "fired": True}]
        d.tick(reminders, self.memfile, self.push, at(8) + T.nudge1_s, schedule_fn=lambda: self.store.load())
        kinds_ = [m["type"] for m in self.pushed]
        self.assertEqual(kinds_, ["speak", "card"])                          # the reminder-based nudge still worked
        self.assertNotIn("metformin", json.dumps(self.pushed).lower())        # and nothing came from the schedule
        self.assertEqual(d.schedule_health()["state"], "off")

    def test_the_runtime_file_stops_scheduled_reminders_immediately_and_deleting_it_resumes(self):
        self.save()
        (self.tmp / ".schedule_reminders_off").touch()
        self.tick(at(8, 0))
        self.assertEqual((self.pushed, self.events()), ([], []))
        (self.tmp / ".schedule_reminders_off").unlink()
        self.tick(at(8, 1))
        self.assertTrue(self.pushed)

    def test_the_whole_medication_check_off_stops_everything_including_the_schedule(self):
        self.save()
        os.environ["PAM_DOSE_CHECK"] = "off"
        self.assertFalse(d.schedule_enabled())
        self.assertEqual(self.tick(at(8, 0)), 0)
        self.assertEqual((self.pushed, self.events()), ([], []))

    def test_off_values_are_understood(self):
        for value in ("0", "off", "FALSE", "no", "disabled", " Off "):
            os.environ["PAM_SCHEDULE_REMINDERS"] = value
            self.assertFalse(d.schedule_enabled(), value)
        os.environ["PAM_SCHEDULE_REMINDERS"] = "on"
        self.assertTrue(d.schedule_enabled())

    def test_a_dose_that_was_open_when_the_switch_was_pulled_gets_no_more_prompts(self):
        self.save()
        self.tick(at(8, 0))
        self.pushed.clear()
        (self.tmp / ".schedule_reminders_off").touch()
        self.tick(at(8) + T.nudge1_s)
        self.assertEqual(self.pushed, [])


class DemoHook(Files):
    def test_simulated_evidence_covers_the_whole_scheduled_group(self):
        os.environ["PAM_DOSE_DEMO"] = "1"
        self.save(MET, LIS)
        self.tick(at(8, 0))
        self.assertTrue(d.simulate_evidence(at(8, 2))["ok"])
        doses = d.replay(self.events())
        self.assertTrue(all(x.evidence and x.evidence["simulated"] for x in doses.values()))
        self.assertEqual(len(doses), 2)


class ThePage(unittest.TestCase):
    HTML = (Path(__file__).resolve().parent / "caregiver.html").read_text(encoding="utf-8")

    def test_the_dashboard_says_plainly_when_reminders_are_not_being_sent(self):
        for phrase in ("Scheduled reminders", "Not running", "Schedule file damaged: no reminders are being sent",
                       "No schedule saved yet"):
            self.assertIn(phrase, self.HTML)

    def test_the_page_still_never_inserts_text_as_html(self):
        for banned in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write"):
            self.assertNotIn(banned, self.HTML)


if __name__ == "__main__":
    unittest.main(verbosity=1)
