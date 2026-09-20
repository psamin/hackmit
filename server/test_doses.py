"""Tests for doses.py: the medication check. No phone, camera, network or key needed.

    python server/test_doses.py

The ones that matter most are in Safety: Pam must never say the user has NOT taken their pills,
never tell them to take one, and the kill switch must really stop everything.
"""
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import doses as d  # noqa: E402
import schedule as sched  # noqa: E402

T = d.Timing()                               # the real timings: 10 / 20 minutes
CG = {"name": "Mike", "phone": "+15557654321"}
DUE = datetime(2026, 9, 19, 8, 0, 0).timestamp()   # a Saturday morning, local time
REMINDER = {"id": 1, "text": "take your medication", "due_ts": DUE, "fired": True}


def mem(minutes_after_due, obj="pill bottle", conf=0.9, event="placed"):
    when = datetime.fromtimestamp(DUE) + timedelta(minutes=minutes_after_due)
    return {"event": event, "object": obj, "confidence": conf, "logged_at": when.isoformat(timespec="seconds"),
            "frames": ["runs/live/events/001/before.jpg", "runs/live/events/001/after.jpg"]}


def run(events, reminders, memories, now, t=T):
    """plan() from a list of already-logged events."""
    return d.plan(d.replay(events), reminders, memories, now, t, CG)


def kinds(result):
    return [e["type"] for e, _ in result]


def spoken(result):
    return " ".join(m["text"] for _, msgs in result for m in msgs if m["type"] == "speak")


class Isolated(unittest.TestCase):
    """Point the module at temp files and a clean environment, whatever is on this machine."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        contacts = self.tmp / "contacts.json"
        contacts.write_text(json.dumps({"caregiver": {"name": "Mike", "phone": "+15557654321"}}))
        self.patches = [mock.patch.object(d, "LOG", self.tmp / "doses.jsonl"),
                        mock.patch.object(d, "KILL_FILE", self.tmp / ".dose_check_off"),
                        mock.patch.object(d, "CONTACTS", contacts),
                        # nothing on this machine may leak in: not the schedule someone saved while trying Pam out, and
                        # not a switch someone left turned off
                        mock.patch.object(d, "schedule_store", lambda: sched.Store(self.tmp / "medications.jsonl")),
                        mock.patch.object(d, "SCHEDULE_KILL_FILE", self.tmp / ".schedule_reminders_off"),
                        mock.patch.object(d, "STREAK_KILL_FILE", self.tmp / ".streak_off"),
                        mock.patch.object(d, "VOICE_KILL_FILE", self.tmp / ".voice_confirm_off"),
                        mock.patch.dict(os.environ, {}, clear=False)]
        for p in self.patches:
            p.start()
        os.environ.pop("PAM_DOSE_CHECK", None)
        os.environ.pop("PAM_DOSE_DEMO", None)
        self.memfile = self.tmp / "memory.jsonl"

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def write_memories(self, *ms):
        self.memfile.write_text("".join(json.dumps(m) + "\n" for m in ms))


class Detecting(unittest.TestCase):
    def test_medication_words(self):
        for text in ("take your medication", "Take my pills", "morning meds", "one tablet", "prescription refill"):
            self.assertTrue(d.is_medication(text), text)
        for text in ("call Sarah", "water the plants", "lunch", "medium coffee"):
            self.assertFalse(d.is_medication(text), text)

    def test_fired_medication_reminder_opens_a_dose(self):
        r = run([], [REMINDER], [], DUE + 1)
        self.assertEqual(kinds(r), ["due"])

    def test_unfired_non_medication_and_stale_reminders_do_not(self):
        self.assertEqual(run([], [{**REMINDER, "fired": False}], [], DUE + 1), [])
        self.assertEqual(run([], [{**REMINDER, "text": "call Sarah"}], [], DUE + 1), [])
        # fired days ago (e.g. the samples in reminders.jsonl): not a dose we are waiting on
        self.assertEqual(run([], [REMINDER], [], DUE + T.expire_s + 1), [])

    def test_a_dose_is_opened_once(self):
        first = run([], [REMINDER], [], DUE + 1)
        logged = [{**e, "ts": DUE + 1} for e, _ in first]
        self.assertEqual(kinds(run(logged, [REMINDER], [], DUE + 2)), [])


class Evidence(unittest.TestCase):
    def test_pill_bottle_placed_after_the_reminder_is_evidence(self):
        ev = d.find_evidence([mem(3)], DUE, T)
        self.assertEqual(ev["confidence"], 0.9)
        self.assertFalse(ev["simulated"])
        self.assertTrue(ev["frame"].endswith("after.jpg"))

    def test_things_that_are_not_evidence(self):
        for m in (mem(3, obj="water bottle"), mem(3, obj="other"), mem(3, conf=0.39),
                  mem(3, event="picked_up"), mem(-5), mem(T.evidence_window_s / 60 + 1)):
            self.assertIsNone(d.find_evidence([m], DUE, T), m)

    def test_the_earliest_evidence_wins(self):
        ev = d.find_evidence([mem(20), mem(5), mem(9)], DUE, T)
        self.assertEqual(ev["logged_at"], mem(5)["logged_at"])

    def test_junk_memories_are_ignored_not_fatal(self):
        self.assertIsNone(d.find_evidence([{"event": "placed", "object": "pill bottle", "confidence": "x"},
                                           {"event": "placed", "object": "pill bottle", "confidence": 0.9,
                                            "logged_at": "not a time"}], DUE, T))


class Ladder(unittest.TestCase):
    def logged(self, *events, at=DUE):
        return [{**e, "ts": at} for e in events]

    def test_evidence_asks_once_with_yes_and_not_yet(self):
        opened = self.logged({"type": "due", "dose": 1, "text": "pills", "due_ts": DUE})
        r = run(opened, [REMINDER], [mem(3)], DUE + 200)
        self.assertEqual(kinds(r), ["evidence", "asked"])
        card = next(m["card"] for _, msgs in r for m in msgs if m["type"] == "card")
        self.assertEqual(card["action"]["body"], {"dose": 1, "answer": "yes"})
        self.assertEqual(card["action2"]["body"], {"dose": 1, "answer": "not_yet"})
        self.assertIn("I saw your pill bottle move", spoken(r))

    def test_asked_only_once(self):
        ev = self.logged({"type": "due", "dose": 1, "text": "pills", "due_ts": DUE},
                         {"type": "evidence", "dose": 1, "logged_at": "x", "confidence": 1, "frame": None,
                          "simulated": False}, {"type": "asked", "dose": 1}, at=DUE + 200)
        self.assertEqual(kinds(run(ev, [REMINDER], [mem(3)], DUE + 210)), [])

    def test_no_evidence_no_nudge_before_the_first_delay(self):
        opened = self.logged({"type": "due", "dose": 1, "text": "pills", "due_ts": DUE})
        self.assertEqual(kinds(run(opened, [REMINDER], [], DUE + T.nudge1_s - 1)), [])

    def test_first_nudge_asks_again_without_claiming_to_have_seen_anything(self):
        opened = self.logged({"type": "due", "dose": 1, "text": "pills", "due_ts": DUE})
        r = run(opened, [REMINDER], [], DUE + T.nudge1_s)
        self.assertEqual(kinds(r), ["nudge"])
        self.assertIn("I haven't recorded your pills yet", spoken(r))
        self.assertNotIn("I saw", spoken(r))

    def test_second_step_keeps_caregiver_support_without_calling(self):
        ev = self.logged({"type": "due", "dose": 1, "text": "pills", "due_ts": DUE},
                         {"type": "nudge", "dose": 1, "n": 1}, at=DUE + T.nudge1_s)
        r = run(ev, [REMINDER], [], DUE + T.nudge2_s)
        self.assertEqual(kinds(r), ["escalated"])
        card = next(m["card"] for _, msgs in r for m in msgs if m["type"] == "card")
        self.assertEqual(card["action2"]["post"], "/api/dose/confirm")  # a tap, never auto
        self.assertNotIn("action", card)
        self.assertNotIn("href", json.dumps(card))
        self.assertIn("Please check with Mike", spoken(r))
        self.assertNotIn("call", spoken(r))

    def test_no_phone_number_means_no_call_button_not_a_broken_one(self):
        r = d.escalate_messages(d.Dose(1, "pills", DUE), {"name": "your caregiver", "phone": None})
        self.assertNotIn("href", json.dumps(r))

    def test_it_stops_after_two_prompts(self):
        ev = self.logged({"type": "due", "dose": 1, "text": "pills", "due_ts": DUE},
                         {"type": "nudge", "dose": 1, "n": 1}, {"type": "escalated", "dose": 1}, at=DUE + T.nudge2_s)
        self.assertEqual(kinds(run(ev, [REMINDER], [], DUE + T.nudge2_s + 300)), [])

    def test_prompts_are_spaced_out(self):
        # evidence arrives late, right at the first nudge time: the ask and the nudge must not stack
        ev = self.logged({"type": "due", "dose": 1, "text": "pills", "due_ts": DUE},
                         {"type": "asked", "dose": 1}, at=DUE + T.nudge1_s)
        self.assertEqual(kinds(run(ev, [REMINDER], [], DUE + T.nudge1_s + 5)), [])
        self.assertEqual(kinds(run(ev, [REMINDER], [], DUE + T.nudge1_s + T.min_gap_s)), ["nudge"])

    def test_confirmed_dose_is_left_alone(self):
        ev = self.logged({"type": "due", "dose": 1, "text": "pills", "due_ts": DUE}, {"type": "confirmed", "dose": 1})
        self.assertEqual(kinds(run(ev, [REMINDER], [mem(3)], DUE + T.nudge2_s + 10)), [])

    def test_dose_expires_and_goes_quiet(self):
        ev = self.logged({"type": "due", "dose": 1, "text": "pills", "due_ts": DUE})
        self.assertEqual(kinds(run(ev, [REMINDER], [], DUE + T.expire_s + 1)), ["expired"])
        ev += self.logged({"type": "expired", "dose": 1})
        self.assertEqual(kinds(run(ev, [REMINDER], [], DUE + T.expire_s + 60)), [])

    def test_demo_timings_are_short(self):
        opened = self.logged({"type": "due", "dose": 1, "text": "pills", "due_ts": DUE})
        self.assertEqual(kinds(run(opened, [REMINDER], [], DUE + 21, d.DEMO)), ["nudge"])


class Answers(Isolated):
    def dose(self, **kw):
        return d.Dose(1, "pills", DUE, **kw)

    def test_recorded_dose_is_read_back_from_the_tap(self):
        r = d.status({1: self.dose(confirmed_ts=DUE + 300)}, DUE + 3600, CG)
        self.assertEqual(r["say"], "You marked your pills as taken at 8:05 AM.")
        self.assertTrue(r["recorded"])

    def test_two_recorded_doses(self):
        second = d.Dose(2, "pills", DUE + 6 * 3600, confirmed_ts=DUE + 6 * 3600 + 60)
        r = d.status({1: self.dose(confirmed_ts=DUE + 300), 2: second}, DUE + 8 * 3600, CG)
        self.assertEqual(r["say"], "You marked your pills as taken at 8:05 AM and 2:01 PM.")

    def test_yesterdays_tap_does_not_count_today(self):
        yesterday = d.Dose(1, "pills", DUE - 86400, confirmed_ts=DUE - 86400 + 60)
        self.assertFalse(d.status({1: yesterday}, DUE + 60, CG)["recorded"])

    def test_unrecorded_with_evidence_says_what_was_seen_and_what_was_not(self):
        ev = {"logged_at": mem(3)["logged_at"], "confidence": 0.9, "frame": None, "simulated": False}
        say = d.status({1: self.dose(evidence=ev)}, DUE + 600, CG)["say"]
        self.assertIn("No, you haven't taken your pills yet", say)
        self.assertIn("I did see your pill bottle move at 8:03 AM, but I can't tell whether you took any", say)
        self.assertIn("I can't see inside the bottle", say)
        self.assertIn("check with Mike or your pill organiser before taking any pills", say)

    def test_unrecorded_without_evidence_does_not_say_nothing_happened(self):
        say = d.status({}, DUE + 600, CG)["say"]
        self.assertIn("I haven't seen your pill bottle move, but I can't see everything", say)

    def test_unrecorded_answer_offers_the_caregiver(self):
        result = d.status({}, DUE, CG)
        self.assertIn("Please check with Mike", result["say"])
        self.assertNotIn("action", result["card"])
        self.assertNotIn("href", json.dumps(d._off(CG)))


class Safety(Isolated):
    """The rules that exist because a wrong answer here can cause a second dose."""

    # The denial phrases used to be banned here: Pam answered "I don't have a record"
    # rather than "no". That was reversed as a product decision -- the hedge tested as
    # vague to the person it is for -- so a plain "no, you haven't taken your pills yet"
    # is now the intended answer and this list no longer bans it.
    #
    # What the denial ban was PROTECTING against has not gone away: a memory-impaired
    # user told "no" may take a second dose. That risk is now carried entirely by the
    # caregiver referral, which is why it is asserted separately and must not be dropped.
    BANNED_DENIALS = ("you forgot", "you missed")
    BANNED_INSTRUCTIONS = ("take your pills now", "you should take", "go ahead and take", "take another",
                           "take your medication", "skip", "double")

    def every_sentence_pam_can_say(self):
        cg = CG
        dose = d.Dose(1, "pills", DUE)
        out = [d.status({}, DUE, cg)["say"],
               d.status({1: d.Dose(1, "pills", DUE, evidence={"logged_at": mem(3)["logged_at"], "confidence": 1,
                                                              "frame": None, "simulated": False})}, DUE, cg)["say"]]
        for saw in (True, False):
            out += [m["text"] for m in d.ask_messages(dose, cg, saw) if m["type"] == "speak"]
        out += [m["text"] for m in d.escalate_messages(dose, cg) if m["type"] == "speak"]
        d.append({"type": "due", "dose": 1, "text": "pills", "due_ts": DUE}, DUE)
        out += [d.confirm(1, "not_yet", DUE)["say"], d.OFF_ANSWER.format(name="Mike")]
        return out

    def test_pam_never_tells_the_user_they_have_not_taken_their_pills(self):
        for text in self.every_sentence_pam_can_say():
            for banned in self.BANNED_DENIALS:
                self.assertNotIn(banned, text.lower(), text)

    def test_pam_never_tells_the_user_to_take_a_dose(self):
        for text in self.every_sentence_pam_can_say():
            for banned in self.BANNED_INSTRUCTIONS:
                self.assertNotIn(banned, text.lower(), text)

    def test_every_prompt_that_asks_also_says_to_check_with_the_caregiver_if_unsure(self):
        dose = d.Dose(1, "pills", DUE)
        for saw in (True, False):
            say = next(m["text"] for m in d.ask_messages(dose, CG, saw) if m["type"] == "speak")
            self.assertIn("If you're not sure, please check with Mike", say)

    def test_the_camera_alone_never_records_a_dose(self):
        # evidence, asking, nudging, escalating: none of them logs a confirmation
        ev = [{"type": "due", "dose": 1, "text": "pills", "due_ts": DUE, "ts": DUE}]
        for now in (DUE + 60, DUE + T.nudge1_s, DUE + T.nudge2_s):
            for event, _ in run(ev, [REMINDER], [mem(2)], now):
                self.assertNotEqual(event["type"], "confirmed")

    def test_simulated_evidence_is_marked_simulated_in_the_log(self):
        os.environ["PAM_DOSE_DEMO"] = "1"
        d.append({"type": "due", "dose": 1, "text": "pills", "due_ts": DUE}, DUE)
        d.simulate_evidence(DUE + 5)
        self.assertTrue(d.replay(d.read_events())[1].evidence["simulated"])

    def test_simulation_is_refused_outside_demo_mode(self):
        d.append({"type": "due", "dose": 1, "text": "pills", "due_ts": DUE}, DUE)
        self.assertFalse(d.simulate_evidence(DUE + 5)["ok"])
        self.assertIsNone(d.replay(d.read_events())[1].evidence)


class Confirming(Isolated):
    def open_dose(self):
        d.append({"type": "due", "dose": 1, "text": "pills", "due_ts": DUE}, DUE)

    def test_yes_records_the_dose_once_even_if_tapped_twice(self):
        self.open_dose()
        first, second = d.confirm(1, "yes", DUE + 300), d.confirm(1, "yes", DUE + 400)
        self.assertEqual(first["say"], "Thank you. I've noted that you took your pills at 8:05 AM.")
        self.assertEqual(second["say"], first["say"])   # still reports the FIRST time
        self.assertEqual(sum(e["type"] == "confirmed" for e in d.read_events()), 1)

    def test_not_yet_records_no_dose(self):
        self.open_dose()
        d.confirm(1, "not_yet", DUE + 300)
        self.assertIsNone(d.replay(d.read_events())[1].confirmed_ts)

    def test_unknown_dose_and_unknown_answer_are_refused(self):
        self.open_dose()
        self.assertFalse(d.confirm(99, "yes")["ok"])
        self.assertFalse(d.confirm(1, "maybe")["ok"])
        self.assertEqual(sum(e["type"] == "confirmed" for e in d.read_events()), 0)


class Ticking(Isolated):
    def test_full_flow_reminder_to_evidence_to_ask_to_tap_to_answer(self):
        sent = []
        push = lambda m: sent.append(m) or 1   # one connected page
        reminders = [REMINDER]
        self.write_memories()
        d.tick(reminders, self.memfile, push, DUE + 1)                       # opens the dose
        self.write_memories(mem(3))
        d.tick(reminders, self.memfile, push, DUE + 200)                     # bottle moved -> asks
        self.assertEqual([m["type"] for m in sent], ["speak", "card"])
        self.assertFalse(d.status_response(DUE + 250)["recorded"])           # not confirmed by the camera
        d.confirm(1, "yes", DUE + 300)
        self.assertEqual(d.status_response(DUE + 400)["say"], "You marked your pills as taken at 8:05 AM.")
        sent.clear()
        d.tick(reminders, self.memfile, push, DUE + T.nudge2_s + 60)         # confirmed: no more prompts
        self.assertEqual(sent, [])

    def test_a_prompt_nobody_heard_is_not_used_up(self):
        reminders = [REMINDER]
        self.write_memories(mem(3))
        d.tick(reminders, self.memfile, lambda m: 0, DUE + 1)                # opens dose; nobody connected
        d.tick(reminders, self.memfile, lambda m: 0, DUE + 200)              # would ask: no page, not logged
        self.assertFalse(d.replay(d.read_events())[1].asked)
        got = []
        d.tick(reminders, self.memfile, lambda m: got.append(m) or 1, DUE + 202)   # a page connects
        self.assertTrue(d.replay(d.read_events())[1].asked)
        self.assertEqual([m["type"] for m in got], ["speak", "card"])

    def test_a_corrupt_log_line_is_skipped(self):
        d.LOG.write_text('{"type": "due", "dose": 1, "text": "p", "due_ts": 1}\n{oops\n')
        self.assertEqual(len(d.read_events()), 1)


class KillSwitch(Isolated):
    def test_env_off_disables_everything_including_registration(self):
        for value in ("off", "0", "false", "NO", "disabled", " Off "):
            os.environ["PAM_DOSE_CHECK"] = value
            self.assertFalse(d.env_enabled(), value)
            self.assertFalse(d.enabled(), value)

    def test_on_by_default_and_for_other_values(self):
        self.assertTrue(d.enabled())
        os.environ["PAM_DOSE_CHECK"] = "on"
        self.assertTrue(d.enabled())

    def test_kill_file_disables_immediately_without_a_restart(self):
        self.assertTrue(d.enabled())
        d.KILL_FILE.touch()
        self.assertFalse(d.enabled())
        self.assertTrue(d.env_enabled())        # the env switch alone did not change
        d.KILL_FILE.unlink()
        self.assertTrue(d.enabled())            # and deleting the file turns it back on

    def test_when_off_nothing_is_planned_pushed_or_logged(self):
        d.KILL_FILE.touch()
        pushed = []
        self.write_memories(mem(3))
        n = d.tick([REMINDER], self.memfile, lambda m: pushed.append(m) or 1, DUE + T.nudge2_s)
        self.assertEqual((n, pushed), (0, []))
        self.assertFalse(d.LOG.exists())

    def test_when_off_answers_defer_to_the_caregiver_and_record_nothing(self):
        d.append({"type": "due", "dose": 1, "text": "pills", "due_ts": DUE}, DUE)
        d.KILL_FILE.touch()
        for r in (d.status_response(DUE + 60), d.confirm(1, "yes", DUE + 60), d.simulate_evidence(DUE + 60)):
            self.assertFalse(r["enabled"])
            self.assertIn("Please ask Mike about your pills", r["say"])
        self.assertEqual(sum(e["type"] == "confirmed" for e in d.read_events()), 0)

    def test_turning_it_back_on_resumes_from_the_log(self):
        d.KILL_FILE.touch()
        d.KILL_FILE.unlink()
        self.assertEqual(d.tick([REMINDER], self.memfile, lambda m: 1, DUE + 1), 1)


if __name__ == "__main__":
    unittest.main(verbosity=1)
