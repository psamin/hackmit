"""Tests for schedule_parse.py and caregiver_schedule.py: reading instructions into a draft, and
saving a draft only after a person has confirmed it. A fake model stands in for the AI service, so
nothing here touches the network or needs a key.

    python server/test_schedule_parse.py

The ones that matter most: an invented medication or a misquoted phrase can never pass; vague times
become QUESTIONS, not guesses; a medication with something Pam cannot track gets NO reminders; nothing
is saved without the caregiver, and only exactly what they were shown.
"""
import asyncio
import json
import os
import sys
import tempfile
import threading
import unittest
from datetime import date, datetime
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import anthropic  # noqa: E402
import caregiver  # noqa: E402
import caregiver_schedule as cs  # noqa: E402
import httpx  # noqa: E402
import schedule as sched  # noqa: E402
import schedule_parse as sp  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

NOW = datetime(2026, 9, 21, 9, 30).timestamp()      # a Monday
PIN = "correct-horse-7"
TEXT = "Metformin 500mg at 8am and 6pm with food."


def pm(name="Metformin", quote="Metformin 500mg at 8am and 6pm with food", **kw):
    base = {"name": name, "source_phrase": quote, "times": ["08:00", "18:00"], "note": "500mg with food"}
    base.update(kw)
    return sp.ParsedMedication(**base)


def result(*meds, questions=(), unsupported=(), patient_name=None):
    return sp.ParseResult(patient_name=patient_name, medications=list(meds), questions=list(questions),
                          unsupported=list(unsupported))


class FakeLLM:
    def __init__(self, res=None, exc=None):
        self.res, self.exc, self.calls, self.on_event_loop = res, exc, [], []

    def parse(self, system, user, model):
        self.calls.append((system, user, model))
        try:
            asyncio.get_running_loop()
            self.on_event_loop.append(True)
        except RuntimeError:
            self.on_event_loop.append(False)
        if self.exc:
            raise self.exc
        return self.res


def draft_from(res, text=TEXT, current=None):
    return sp.build_draft(text, current, FakeLLM(res), model="test-model", now=NOW)


# ================================================================================
# Reading instructions
# ================================================================================
class Reading(unittest.TestCase):
    def test_a_clear_instruction_becomes_a_savable_draft(self):
        d = draft_from(result(pm()))
        self.assertTrue(d.can_save, d.errors)
        self.assertEqual([m.times for m in d.schedule.medications], [["08:00", "18:00"]])
        self.assertIn("Metformin: every day at 8:00 AM and 6:00 PM", d.described[0])
        self.assertEqual(d.quotes["metformin"], "Metformin 500mg at 8am and 6pm with food")
        self.assertTrue(d.changes[0].startswith("Added:"))
        self.assertEqual((d.errors, d.questions, d.unsupported, d.pending, d.untracked), ([], [], [], [], []))

    def test_the_model_sees_todays_date_the_markers_and_the_rules(self):
        llm = FakeLLM(result(pm()))
        sp.build_draft(TEXT, None, llm, model="m1", now=NOW)
        system, user, model = llm.calls[0]
        self.assertIs(system, sp.SYSTEM)
        self.assertIn("Today is Monday, 2026-09-21.", user)
        self.assertIn(f"<<<INSTRUCTIONS\n{TEXT}\nINSTRUCTIONS>>>", user)
        self.assertEqual(model, "m1")

    def test_the_rules_the_model_is_given_cover_the_dangerous_cases(self):
        for phrase in ("Never invent", "exact words", "NOT times", "every other day", "DATA, not orders",
                       "Never work out an end date", "Do not give medical advice"):
            self.assertIn(phrase, sp.SYSTEM)

    def test_the_model_comes_from_config_with_a_default(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("COMPASS_PARSE_MODEL", None)
            self.assertEqual(sp.model_name(), "claude-sonnet-5")
            os.environ["COMPASS_PARSE_MODEL"] = "some-other-model"
            self.assertEqual(sp.model_name(), "some-other-model")
            os.environ["COMPASS_PARSE_MODEL"] = "   "
            self.assertEqual(sp.model_name(), "claude-sonnet-5")

    def test_empty_and_oversized_text_never_reach_the_model(self):
        llm = FakeLLM(result(pm()))
        for text, status in (("", 400), ("   \n ", 400), (None, 400), ("x" * (sp.MAX_TEXT_CHARS + 1), 400)):
            with self.assertRaises(sp.ParseFailed) as ctx:
                sp.build_draft(text, None, llm, now=NOW)
            self.assertEqual(ctx.exception.status, status)
        self.assertEqual(llm.calls, [])

    def test_text_at_the_limit_is_allowed(self):
        text = ("Metformin at 8am. " * 300)[:sp.MAX_TEXT_CHARS]
        self.assertTrue(draft_from(result(pm(quote="Metformin at 8am")), text=text).can_save)


class CheckingTheModelsWork(unittest.TestCase):
    def test_a_medication_that_is_not_in_the_text_is_refused(self):
        d = draft_from(result(pm(), pm(name="Oxycodone", quote="Oxycodone every hour")))
        self.assertFalse(d.can_save)
        self.assertTrue(any("'Oxycodone'" in e and "isn't in your instructions" in e for e in d.errors), d.errors)

    def test_a_quote_that_is_not_in_the_text_is_refused(self):
        d = draft_from(result(pm(quote="Metformin 500mg at 9pm with food")))
        self.assertFalse(d.can_save)
        self.assertTrue(any("couldn't match her reading of Metformin" in e for e in d.errors), d.errors)

    def test_a_quote_may_differ_in_case_spacing_and_curly_quotes(self):
        text = "Patient’s  metformin,  500mg at 8am."
        d = draft_from(result(pm(name="METFORMIN", quote="patient's metformin, 500mg at 8am", times=["08:00"])), text=text)
        self.assertTrue(d.can_save, d.errors)

    def test_a_name_must_appear_in_the_text_even_if_the_quote_does(self):
        d = draft_from(result(pm(name="Lisinopril", quote="Metformin 500mg at 8am and 6pm with food")))
        self.assertFalse(d.can_save)

    def test_a_medication_with_no_name_or_quote_is_refused(self):
        self.assertFalse(draft_from(result(pm(name="  "))).can_save)
        self.assertFalse(draft_from(result(pm(quote=""))).can_save)

    def test_no_medications_and_no_questions_is_an_error(self):
        d = draft_from(result())
        self.assertFalse(d.can_save)
        self.assertIn("didn't find any medications", d.errors[0])

    def test_the_result_goes_through_the_same_checks_as_a_hand_written_schedule(self):
        d = draft_from(result(pm(times=["8am"])))
        self.assertFalse(d.can_save)
        self.assertTrue(any("isn't a valid time" in e for e in d.errors), d.errors)
        d = draft_from(result(pm(times=["08:00", "09:00"])))
        self.assertTrue(any("overlap" in e for e in d.errors), d.errors)
        d = draft_from(result(pm(as_needed=True)))
        self.assertTrue(any("'as needed'" in e for e in d.errors), d.errors)

    def test_two_medications_with_one_name_are_refused(self):
        d = draft_from(result(pm(), pm(times=["12:00"])))
        self.assertTrue(any("Two medications are named" in e for e in d.errors), d.errors)

    def test_the_patient_name_is_kept_only_if_it_is_in_the_text(self):
        kept = draft_from(result(pm(), patient_name="Alex"), text="For Alex: " + TEXT)
        self.assertEqual(kept.schedule.patient_name, "Alex")
        dropped = draft_from(result(pm(), patient_name="Robert"))
        self.assertIsNone(dropped.schedule.patient_name)


class Questions(unittest.TestCase):
    ASK = sp.Question(medication="Lisinopril", question="What time of day?", suggestion="8:00 AM")

    def lis(self, **kw):
        return pm(name="Lisinopril", quote="Lisinopril once daily", times=[], note="10mg", **kw)

    TEXT = "Lisinopril once daily in the morning."

    def test_a_vague_time_becomes_a_question_and_blocks_saving(self):
        d = draft_from(result(self.lis(), questions=[self.ASK]), text="Lisinopril once daily in the morning.")
        self.assertFalse(d.can_save)
        self.assertEqual(d.pending, ["Lisinopril"])
        self.assertEqual(d.questions[0].suggestion, "8:00 AM")

    def test_a_pending_medication_does_not_also_show_a_confusing_needs_a_time_error(self):
        d = draft_from(result(self.lis(), questions=[self.ASK]), text="Lisinopril once daily in the morning.")
        self.assertEqual(d.errors, [])
        self.assertNotIn("needs at least one time", " ".join(d.errors))

    def test_other_medications_in_the_same_text_are_still_read(self):
        text = "Metformin 500mg at 8am and 6pm with food. Lisinopril once daily."
        d = draft_from(result(pm(), self.lis(), questions=[self.ASK]), text=text)
        self.assertEqual([m.name for m in d.schedule.medications], ["Metformin"])
        self.assertFalse(d.can_save)

    def test_a_question_blocks_saving_even_with_no_medication_named(self):
        d = draft_from(result(pm(), questions=[sp.Question(question="Is that every day?")]))
        self.assertFalse(d.can_save)

    def test_a_question_blocks_saving_even_if_the_model_also_filled_in_times(self):
        d = draft_from(result(pm(), questions=[sp.Question(medication="Metformin", question="Which times exactly?")]))
        self.assertFalse(d.can_save)
        self.assertEqual(d.schedule.medications[0].times, ["08:00", "18:00"])

    def test_a_question_with_no_medications_is_the_not_found_case_not_an_error(self):
        d = draft_from(result(questions=[sp.Question(question="I didn't see a medication. What should Pam remind about?")]),
                       text="Please water the plants.")
        self.assertEqual(d.errors, [])
        self.assertFalse(d.can_save)


class NotTracked(unittest.TestCase):
    def test_a_medication_with_something_untrackable_gets_no_reminders_at_all(self):
        # the dangerous case: the model lists "aspirin every other day" as a daily medication AND flags
        # "every other day" as unsupported. Leaving the daily reminder in would send reminders that
        # should not exist while the caregiver believes Pam is not tracking it.
        text = "Aspirin at 9am every other day. Metformin 500mg at 8am and 6pm with food."
        d = draft_from(result(pm(name="Aspirin", quote="Aspirin at 9am every other day", times=["09:00"], note=""),
                              pm(quote="Metformin 500mg at 8am and 6pm with food"),
                              unsupported=[sp.Unsupported(medication="Aspirin", phrase="every other day",
                                                          reason="alternate days can't be scheduled")]), text=text)
        self.assertEqual([m.name for m in d.schedule.medications], ["Metformin"])
        self.assertEqual(d.untracked, ["Aspirin"])
        self.assertNotIn("Aspirin", " ".join(d.described))

    def test_untracked_is_matched_in_any_case(self):
        text = "aspirin at 9am every other day."
        d = draft_from(result(pm(name="Aspirin", quote="aspirin at 9am every other day", times=["09:00"], note=""),
                              unsupported=[sp.Unsupported(medication="ASPIRIN", phrase="every other day", reason="x")]), text=text)
        self.assertEqual(d.untracked, ["Aspirin"])

    def test_an_unsupported_item_that_names_no_medication_is_kept_and_needs_acknowledging(self):
        d = draft_from(result(pm(), unsupported=[sp.Unsupported(phrase="with food", reason="not a time")]))
        self.assertEqual(len(d.unsupported), 1)
        self.assertEqual(len(d.schedule.medications), 1)

    def test_an_unsupported_phrase_that_is_not_in_the_text_is_kept_not_hidden_and_flagged(self):
        d = draft_from(result(pm(), unsupported=[sp.Unsupported(phrase="every third day", reason="not supported")]))
        self.assertEqual(len(d.unsupported), 1)
        self.assertIn("couldn't find these exact words", d.unsupported[0].reason)


class Dates(unittest.TestCase):
    def course(self, **kw):
        text = "Amoxicillin at 9am and 9pm for 7 days starting 2026-09-21."
        return draft_from(result(pm(name="Amoxicillin", quote="Amoxicillin at 9am and 9pm for 7 days",
                                    times=["09:00", "21:00"], note="", **kw)), text=text)

    def test_the_code_works_out_the_end_date_not_the_model(self):
        d = self.course(start_date="2026-09-21", duration_days=7)
        self.assertEqual(str(d.schedule.medications[0].end_date), "2026-09-27")     # 7 days, inclusive
        self.assertEqual(str(d.schedule.medications[0].start_date), "2026-09-21")

    def test_a_one_day_course_ends_the_same_day(self):
        self.assertEqual(str(self.course(start_date="2026-09-21", duration_days=1).schedule.medications[0].end_date), "2026-09-21")

    def test_month_and_year_boundaries(self):
        d = self.course(start_date="2026-12-28", duration_days=7)
        self.assertEqual(str(d.schedule.medications[0].end_date), "2027-01-03")

    def test_an_explicit_end_date_is_kept_over_a_duration(self):
        d = self.course(start_date="2026-09-21", duration_days=7, end_date="2026-09-30")
        self.assertEqual(str(d.schedule.medications[0].end_date), "2026-09-30")

    def test_a_duration_with_no_start_date_gives_no_invented_end_date(self):
        d = self.course(duration_days=7)
        self.assertIsNone(d.schedule.medications[0].end_date)

    def test_an_impossible_duration_is_refused(self):
        for days in (0, -3, 400):
            d = self.course(start_date="2026-09-21", duration_days=days)
            self.assertFalse(d.can_save, days)
            self.assertTrue(any("isn't supported" in e for e in d.errors), (days, d.errors))

    def test_a_bad_date_is_refused_with_the_format(self):
        d = self.course(start_date="next week")
        self.assertTrue(any("YYYY-MM-DD" in e for e in d.errors), d.errors)

    def test_an_end_before_the_start_is_refused(self):
        d = self.course(start_date="2026-09-21", end_date="2026-09-01")
        self.assertTrue(any("end date is before" in e for e in d.errors), d.errors)


class TheDraftAndItsChanges(unittest.TestCase):
    def test_a_first_draft_is_read_against_version_zero(self):
        self.assertEqual(draft_from(result(pm())).base_version, 0)

    def test_a_later_draft_is_read_against_the_current_version_and_diffed_with_it(self):
        store = sched.Store(Path(tempfile.mkdtemp()) / "m.jsonl", clock=lambda: NOW)
        store.save({"medications": [{"name": "Metformin", "times": ["08:00"]}]})
        d = draft_from(result(pm()), current=store.load())
        self.assertEqual(d.base_version, 1)
        self.assertTrue(d.changes[0].startswith("Changed: Metformin"), d.changes)

    def test_the_public_view_has_the_quote_and_no_raw_model_output(self):
        pub = draft_from(result(pm())).public(now=NOW + 60)
        self.assertEqual(pub["understood"][0]["quote"], "Metformin 500mg at 8am and 6pm with food")
        self.assertTrue(pub["can_save"])
        self.assertEqual(pub["expires_in_s"], sp.DRAFT_TTL_S - 60)
        self.assertEqual(set(pub), {"draft_id", "can_save", "model", "base_version", "expires_in_s", "understood", "pending",
                                    "untracked", "questions", "unsupported", "errors", "warnings", "changes"})
        self.assertNotIn("source_phrase", json.dumps(pub))                          # the raw model output stays server-side

    def test_draft_ids_are_long_and_unique(self):
        ids = {draft_from(result(pm())).id for _ in range(20)}
        self.assertEqual(len(ids), 20)
        self.assertTrue(all(len(i) >= 16 for i in ids))

    def test_a_note_with_markup_passes_through_untouched_the_page_is_what_escapes_it(self):
        d = draft_from(result(pm(note="<img src=x onerror=alert(1)>")))
        self.assertEqual(d.schedule.medications[0].note, "<img src=x onerror=alert(1)>")


# ================================================================================
# Saving: only what the caregiver was shown, only after they confirmed
# ================================================================================
class Saving(unittest.TestCase):
    def setUp(self):
        self.t = NOW
        self.store = sched.Store(Path(tempfile.mkdtemp()) / "m.jsonl", clock=lambda: self.t)
        self.drafts = sp.Drafts(clock=lambda: self.t)

    def add(self, res, text=TEXT):
        d = sp.build_draft(text, self.store.load(), FakeLLM(res), model="m", now=self.t)
        self.drafts.add(d)
        return d

    def test_a_confirmed_draft_is_saved_exactly_as_previewed(self):
        d = self.add(result(pm()))
        e = sp.save_draft(self.drafts, self.store, d.id)
        self.assertEqual(e.schedule, self.store.load().schedule)
        self.assertEqual(e.source_text, TEXT)
        self.assertEqual(e.actor, "caregiver")
        self.assertEqual(e.schedule.medications[0].times, ["08:00", "18:00"])

    def test_a_draft_can_only_be_saved_once(self):
        d = self.add(result(pm()))
        sp.save_draft(self.drafts, self.store, d.id)
        with self.assertRaises(sp.SaveRefused):
            sp.save_draft(self.drafts, self.store, d.id)
        self.assertEqual(self.store.load().version, 1)

    def test_a_used_draft_is_gone_not_merely_out_of_date(self):
        d = self.add(result(pm()))
        sp.save_draft(self.drafts, self.store, d.id)
        self.assertIsNone(self.drafts.get(d.id))          # not relying on the version check to refuse a second save

    def test_a_draft_that_somehow_has_both_a_schedule_and_errors_is_still_refused(self):
        # build_draft never produces one, but the guard must hold on its own, not by luck of another
        d = self.add(result(pm()))
        d.errors.append("something the caregiver must fix")
        with self.assertRaises(sp.SaveRefused) as ctx:
            sp.save_draft(self.drafts, self.store, d.id)
        self.assertIn("problems", ctx.exception.message)
        self.assertIsNone(self.store.load())

    def test_unknown_or_malformed_draft_ids_save_nothing(self):
        for bad in ("nope", "", None, 123, ["x"], {"a": 1}):
            with self.assertRaises(sp.SaveRefused):
                sp.save_draft(self.drafts, self.store, bad)
        self.assertIsNone(self.store.load())

    def test_a_draft_with_problems_cannot_be_saved(self):
        d = self.add(result(pm(name="Ghost", quote="Ghost at 3am")))
        with self.assertRaises(sp.SaveRefused):
            sp.save_draft(self.drafts, self.store, d.id)
        self.assertIsNone(self.store.load())

    def test_a_draft_with_open_questions_cannot_be_saved(self):
        d = self.add(result(pm(), questions=[sp.Question(medication="Metformin", question="Which times?")]))
        with self.assertRaises(sp.SaveRefused) as ctx:
            sp.save_draft(self.drafts, self.store, d.id)
        self.assertIn("questions", ctx.exception.message)
        self.assertIsNone(self.store.load())

    def not_tracked(self):
        return self.add(result(pm(), unsupported=[sp.Unsupported(phrase="with food", reason="not a time")]))

    def test_unsupported_items_must_be_acknowledged(self):
        d = self.not_tracked()
        for ack in (None, [], ["something else"], [None, 5], "with food"):
            with self.assertRaises(sp.SaveRefused) as ctx:
                sp.save_draft(self.drafts, self.store, d.id, ack)
            self.assertIn("with food", ctx.exception.message)
        self.assertIsNone(self.store.load())

    def test_acknowledging_them_saves_and_records_the_acknowledgement(self):
        d = self.not_tracked()
        e = sp.save_draft(self.drafts, self.store, d.id, ["  WITH   food "])
        self.assertIn("does not track: with food", e.source_text)
        self.assertTrue(e.source_text.startswith(TEXT))

    def test_every_unsupported_item_needs_its_own_acknowledgement(self):
        d = self.add(result(pm(), unsupported=[sp.Unsupported(phrase="with food", reason="x"),
                                               sp.Unsupported(phrase="500mg", reason="y")]))
        with self.assertRaises(sp.SaveRefused):
            sp.save_draft(self.drafts, self.store, d.id, ["with food"])
        self.assertEqual(sp.save_draft(self.drafts, self.store, d.id, ["with food", "500mg"]).version, 1)

    def test_a_schedule_that_changed_since_the_preview_is_refused(self):
        d = self.add(result(pm()))
        self.store.save({"medications": [{"name": "Other", "times": ["12:00"]}]})      # someone saved in between
        with self.assertRaises(sp.SaveRefused) as ctx:
            sp.save_draft(self.drafts, self.store, d.id)
        self.assertIn("changed after you read", ctx.exception.message)
        self.assertEqual(self.store.load().version, 1)                                # only their save exists

    def test_a_stale_preview_expires(self):
        d = self.add(result(pm()))
        self.t += sp.DRAFT_TTL_S + 1
        with self.assertRaises(sp.SaveRefused) as ctx:
            sp.save_draft(self.drafts, self.store, d.id)
        self.assertIn("expired", ctx.exception.message)
        self.assertIsNone(self.store.load())

    def test_a_preview_still_valid_just_before_it_expires(self):
        d = self.add(result(pm()))
        self.t += sp.DRAFT_TTL_S - 1
        self.assertEqual(sp.save_draft(self.drafts, self.store, d.id).version, 1)

    def test_the_oldest_drafts_are_dropped_beyond_the_cap(self):
        ids = []
        for i in range(sp.MAX_DRAFTS + 3):
            self.t += 1
            ids.append(self.add(result(pm())).id)
        self.assertIsNone(self.drafts.get(ids[0]))
        self.assertIsNotNone(self.drafts.get(ids[-1]))

    def test_a_second_edit_saves_against_the_new_current_version(self):
        sp.save_draft(self.drafts, self.store, self.add(result(pm())).id)
        d2 = self.add(result(pm(times=["07:00", "19:00"], quote="Metformin 500mg at 8am and 6pm with food")))
        self.assertEqual(d2.base_version, 1)
        self.assertEqual(sp.save_draft(self.drafts, self.store, d2.id).version, 2)

    def test_a_store_failure_becomes_a_refusal_not_a_crash(self):
        d = self.add(result(pm()))
        with mock.patch.object(self.store, "save", side_effect=sched.ScheduleError(["nope"])):
            with self.assertRaises(sp.SaveRefused):
                sp.save_draft(self.drafts, self.store, d.id)
        self.assertIsNotNone(self.drafts.get(d.id))                                   # still there to retry


# ================================================================================
# The real model client, with the network replaced
# ================================================================================
def _req():
    return httpx.Request("POST", "https://api.anthropic.com/v1/messages")


class FakeMessages:
    def __init__(self, resp=None, exc=None):
        self.resp, self.exc = resp, exc

    def parse(self, **kw):
        self.kw = kw
        if self.exc:
            raise self.exc
        return self.resp


class FakeAnthropic:
    def __init__(self, messages):
        self.messages = messages


class Resp:
    def __init__(self, stop_reason="end_turn", parsed=None):
        self.stop_reason, self.parsed_output = stop_reason, parsed


class RealClient(unittest.TestCase):
    def call(self, exc=None, resp=None):
        fake = FakeMessages(resp=resp, exc=exc)
        with mock.patch.object(anthropic, "Anthropic", lambda **kw: FakeAnthropic(fake)):
            return sp.AnthropicLLM().parse("sys", "user", "model-x"), fake

    def fails(self, exc=None, resp=None):
        with self.assertRaises(sp.ParseFailed) as ctx:
            self.call(exc, resp)
        return ctx.exception.message

    def test_a_good_answer_is_returned_and_asked_for_in_the_expected_shape(self):
        parsed = result(pm())
        got, fake = self.call(resp=Resp(parsed=parsed))
        self.assertIs(got, parsed)
        self.assertIs(fake.kw["output_format"], sp.ParseResult)
        self.assertEqual((fake.kw["model"], fake.kw["system"]), ("model-x", "sys"))
        self.assertEqual(fake.kw["messages"], [{"role": "user", "content": "user"}])
        self.assertLessEqual(fake.kw["max_tokens"], 8192)

    def test_each_kind_of_failure_gets_its_own_plain_message(self):
        r401 = httpx.Response(401, request=_req())
        cases = [(anthropic.AuthenticationError("bad key LEAKED-MARKER-12345", response=r401, body=None), "rejected the API key"),
                 (anthropic.APITimeoutError(request=_req()), "took too long"),
                 (anthropic.APIConnectionError(request=_req()), "Couldn't reach"),
                 (anthropic.RateLimitError("slow down", response=httpx.Response(429, request=_req()), body=None), "busy"),
                 (anthropic.InternalServerError("boom", response=httpx.Response(500, request=_req()), body=None), "returned an error"),
                 (anthropic.AnthropicError("no api key"), "isn't set up"),
                 (ValueError("schema mismatch"), "couldn't understand")]
        for exc, expect in cases:
            self.assertIn(expect, self.fails(exc=exc), type(exc).__name__)

    def test_error_text_from_the_service_never_reaches_the_caregiver(self):
        r401 = httpx.Response(401, request=_req())
        msg = self.fails(exc=anthropic.AuthenticationError("Invalid key LEAKED-MARKER-12345", response=r401, body=None))
        self.assertNotIn("LEAKED-MARKER", msg)                   # whatever the service said, the caregiver never sees it

    def test_a_refusal_a_cut_off_answer_and_an_empty_answer_are_failures(self):
        self.assertIn("declined", self.fails(resp=Resp(stop_reason="refusal")))
        self.assertIn("too long or complex", self.fails(resp=Resp(stop_reason="max_tokens", parsed=result(pm()))))
        self.assertIn("too long or complex", self.fails(resp=Resp(parsed=None)))

    def test_a_failure_produces_no_draft(self):
        with self.assertRaises(sp.ParseFailed):
            sp.build_draft(TEXT, None, FakeLLM(exc=sp.ParseFailed("down")), now=NOW)


# ================================================================================
# The routes
# ================================================================================
def make_app():
    app = FastAPI()
    app.include_router(caregiver.router)
    app.include_router(cs.router)
    return app


class Routes(unittest.TestCase):
    def setUp(self):
        self.t = NOW
        self.store = sched.Store(Path(tempfile.mkdtemp()) / "m.jsonl", clock=lambda: self.t)
        self.llm = FakeLLM(result(pm()))
        self.patches = [mock.patch.dict(os.environ, {"CAREGIVER_PIN": PIN}), mock.patch.object(caregiver, "_log", lambda m: None),
                        mock.patch.object(cs, "get_store", lambda: self.store), mock.patch.object(cs, "get_llm", lambda: self.llm),
                        mock.patch.object(cs, "_now", lambda: self.t)]
        for p in self.patches:
            p.start()
        caregiver._sessions.clear()
        caregiver._failures.clear()
        cs._parse_times.clear()
        cs.DRAFTS = sp.Drafts(clock=lambda: self.t)
        self.client = TestClient(make_app())

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def login(self, client=None):
        r = (client or self.client).post("/api/caregiver/login", json={"pin": PIN})
        self.assertEqual(r.status_code, 200)

    def parse(self, text=TEXT):
        return self.client.post("/api/caregiver/schedule/parse", json={"text": text})


class Access(Routes):
    ROUTES = (("get", "/api/caregiver/schedule"), ("post", "/api/caregiver/schedule/parse"),
              ("post", "/api/caregiver/schedule/save"))

    def hit(self, method, path):
        return getattr(self.client, method)(path, **({"json": {"text": TEXT, "draft_id": "x"}} if method == "post" else {}))

    def test_every_route_needs_the_caregiver_to_be_signed_in(self):
        for method, path in self.ROUTES:
            self.assertEqual(self.hit(method, path).status_code, 401, path)
        self.assertEqual(self.llm.calls, [])                               # and no paid call was made

    def test_every_route_is_off_when_no_pin_is_configured(self):
        os.environ.pop("CAREGIVER_PIN")
        for method, path in self.ROUTES:
            self.assertEqual(self.hit(method, path).status_code, 404, path)

    def test_a_signed_out_caregiver_cannot_use_a_draft_made_earlier(self):
        self.login()
        draft_id = self.parse().json()["draft_id"]
        self.client.post("/api/caregiver/logout")
        r = self.client.post("/api/caregiver/schedule/save", json={"draft_id": draft_id})
        self.assertEqual(r.status_code, 401)
        self.assertIsNone(self.store.load())

    def test_responses_are_marked_do_not_cache(self):
        self.login()
        draft_id = self.parse().json()["draft_id"]
        for r in (self.client.get("/api/caregiver/schedule"), self.parse(),
                  self.client.post("/api/caregiver/schedule/save", json={"draft_id": draft_id}),
                  self.client.get("/api/caregiver/status"), self.client.get("/api/caregiver/me"),
                  self.client.get("/caregiver"), self.client.post("/api/caregiver/login", json={"pin": PIN})):
            self.assertEqual(r.headers.get("cache-control"), "no-store", r.request.url)


class ParseRoute(Routes):
    def setUp(self):
        super().setUp()
        self.login()

    def test_reading_returns_a_draft_and_saves_nothing(self):
        r = self.parse()
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertTrue(body["can_save"])
        self.assertIn("Metformin: every day at 8:00 AM and 6:00 PM", body["understood"][0]["line"])
        self.assertIsNone(self.store.load())

    def test_bad_input_is_refused_before_any_paid_call(self):
        for payload in ({}, {"text": None}, {"text": 5}, {"text": ""}, {"text": "  "}, {"text": "x" * 4001}):
            r = self.client.post("/api/caregiver/schedule/parse", json=payload)
            self.assertEqual(r.status_code, 400, payload)
            self.assertTrue(r.json()["error"])
        self.assertEqual(self.llm.calls, [])

    def test_a_failed_reading_says_why_and_leaves_no_draft(self):
        self.llm.exc = sp.ParseFailed("The AI service took too long to answer. Please try again.")
        r = self.parse()
        self.assertEqual((r.status_code, r.json()["error"]), (502, "The AI service took too long to answer. Please try again."))
        self.assertEqual(cs.DRAFTS._d, {})

    def test_the_model_call_runs_off_the_event_loop_so_pam_keeps_answering(self):
        self.parse()
        self.assertEqual(self.llm.on_event_loop, [False])

    def test_reading_is_limited_because_it_costs_money(self):
        for _ in range(cs.PARSE_LIMIT):
            self.assertEqual(self.parse().status_code, 200)
        r = self.parse()
        self.assertEqual(r.status_code, 429)
        self.assertEqual(len(self.llm.calls), cs.PARSE_LIMIT)              # the 11th never reached the model
        self.t += cs.PARSE_WINDOW_S + 1
        self.assertEqual(self.parse().status_code, 200)                    # and it recovers

    def test_failed_readings_use_up_the_allowance_but_empty_input_does_not(self):
        for _ in range(20):
            self.client.post("/api/caregiver/schedule/parse", json={"text": ""})
        self.llm.exc = sp.ParseFailed("down")
        for _ in range(cs.PARSE_LIMIT):
            self.assertEqual(self.parse().status_code, 502)
        self.assertEqual(self.parse().status_code, 429)

    def test_a_corrupt_store_is_reported_not_crashed_on(self):
        self.store.path.write_text("garbage\n")
        self.assertEqual(self.parse().status_code, 500)
        self.assertEqual(self.llm.calls, [])

    def test_the_draft_shows_questions_and_untracked_items(self):
        self.llm.res = result(pm(name="Aspirin", quote="Aspirin at 9am every other day", times=["09:00"], note=""),
                              questions=[sp.Question(medication="Aspirin", question="Which days?")],
                              unsupported=[sp.Unsupported(medication="Aspirin", phrase="every other day", reason="x")])
        body = self.parse("Aspirin at 9am every other day.").json()
        self.assertFalse(body["can_save"])
        self.assertEqual(body["untracked"], ["Aspirin"])
        self.assertEqual(body["questions"][0]["question"], "Which days?")
        self.assertEqual(body["unsupported"][0]["phrase"], "every other day")


class SaveRoute(Routes):
    def setUp(self):
        super().setUp()
        self.login()

    def save(self, draft_id, **kw):
        return self.client.post("/api/caregiver/schedule/save", json={"draft_id": draft_id, **kw})

    def test_previewing_then_saving_stores_the_schedule(self):
        d = self.parse().json()
        r = self.save(d["draft_id"])
        self.assertEqual((r.status_code, r.json()["ok"], r.json()["version"]), (200, True, 1))
        self.assertEqual(self.store.load().schedule.medications[0].name, "Metformin")
        self.assertEqual(self.store.load().source_text, TEXT)

    def test_saving_ignores_anything_the_browser_adds_only_the_previewed_draft_is_saved(self):
        d = self.parse().json()
        forged = {"medications": [{"name": "Evil", "times": ["03:00"]}]}
        r = self.save(d["draft_id"], schedule=forged, medications=forged["medications"], text="Evil at 3am")
        self.assertEqual(r.status_code, 200)
        saved = self.store.load()
        self.assertEqual([m.name for m in saved.schedule.medications], ["Metformin"])
        self.assertEqual(saved.source_text, TEXT)

    def test_a_refused_save_says_why_and_saves_nothing(self):
        self.llm.res = result(pm(), questions=[sp.Question(medication="Metformin", question="Which times?")])
        d = self.parse().json()
        r = self.save(d["draft_id"])
        self.assertEqual(r.status_code, 409)
        self.assertIn("questions", r.json()["error"])
        self.assertIsNone(self.store.load())

    def test_unknown_missing_and_reused_drafts_are_refused(self):
        self.assertEqual(self.save("nope").status_code, 409)
        self.assertEqual(self.client.post("/api/caregiver/schedule/save", json={}).status_code, 409)
        d = self.parse().json()
        self.assertEqual(self.save(d["draft_id"]).status_code, 200)
        self.assertEqual(self.save(d["draft_id"]).status_code, 409)
        self.assertEqual(self.store.load().version, 1)

    def test_unsupported_items_must_be_ticked(self):
        self.llm.res = result(pm(), unsupported=[sp.Unsupported(phrase="with food", reason="x")])
        d = self.parse().json()
        self.assertEqual(self.save(d["draft_id"]).status_code, 409)
        self.assertEqual(self.save(d["draft_id"], acknowledged="with food").status_code, 409)     # not a list
        self.assertEqual(self.save(d["draft_id"], acknowledged=["with food"]).status_code, 200)

    def test_saving_over_a_changed_schedule_is_refused(self):
        d = self.parse().json()
        self.store.save({"medications": [{"name": "Other", "times": ["12:00"]}]})
        self.assertEqual(self.save(d["draft_id"]).status_code, 409)

    def test_a_corrupt_store_is_reported_on_save(self):
        d = self.parse().json()
        self.store.path.write_text("garbage\n")
        self.assertEqual(self.save(d["draft_id"]).status_code, 500)


class CurrentRoute(Routes):
    def setUp(self):
        super().setUp()
        self.login()

    def test_nothing_saved_yet(self):
        self.assertEqual(self.client.get("/api/caregiver/schedule").json(), {"exists": False})

    def test_the_saved_schedule_history_and_a_prefill_for_editing(self):
        draft_id = self.parse().json()["draft_id"]
        self.client.post("/api/caregiver/schedule/save", json={"draft_id": draft_id})
        body = self.client.get("/api/caregiver/schedule").json()
        self.assertEqual((body["exists"], body["version"], body["instructions"]), (True, 1, TEXT))
        self.assertIn("Metformin: every day at 8:00 AM and 6:00 PM", body["lines"][0])
        self.assertEqual(body["history"][0]["version"], 1)
        self.assertTrue(body["history"][0]["changes"][0].startswith("Added:"))

    def test_the_prefill_leaves_out_the_acknowledgement_note(self):
        self.llm.res = result(pm(), unsupported=[sp.Unsupported(phrase="with food", reason="x")])
        d = self.parse().json()
        self.client.post("/api/caregiver/schedule/save", json={"draft_id": d["draft_id"], "acknowledged": ["with food"]})
        self.assertEqual(self.client.get("/api/caregiver/schedule").json()["instructions"], TEXT)

    def test_the_prefill_skips_a_restore_entry_and_uses_the_last_real_instructions(self):
        draft_id = self.parse().json()["draft_id"]
        self.client.post("/api/caregiver/schedule/save", json={"draft_id": draft_id})
        self.store.restore(1)
        body = self.client.get("/api/caregiver/schedule").json()
        self.assertEqual((body["version"], body["instructions"]), (2, TEXT))

    def test_history_is_limited_to_recent_entries(self):
        for i in range(9):
            self.store.save({"medications": [{"name": "M", "times": ["08:00"], "note": f"v{i}"}]})
        self.assertEqual(len(self.client.get("/api/caregiver/schedule").json()["history"]), 6)

    def test_a_corrupt_store_gives_an_error_not_an_empty_schedule(self):
        self.store.path.write_text("garbage\n")
        r = self.client.get("/api/caregiver/schedule")
        self.assertEqual(r.status_code, 500)
        self.assertIn("no readable entry", r.json()["error"])


class ThePage(unittest.TestCase):
    """The page shows caregiver-written text. It must never be able to run as HTML."""

    HTML = (Path(__file__).resolve().parent / "caregiver.html").read_text(encoding="utf-8")

    def test_the_page_never_inserts_text_as_html(self):
        for banned in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function"):
            self.assertNotIn(banned, self.HTML, banned)

    def test_the_page_builds_its_text_with_textcontent(self):
        self.assertIn("textContent", self.HTML)

    def test_every_element_id_the_script_uses_exists_on_the_page(self):
        import re
        used = set(re.findall(r'\$\("([a-z-]+)"\)', self.HTML))
        self.assertTrue(used)
        for i in used:
            self.assertIn(f'id="{i}"', self.HTML, i)

    def test_the_page_warns_that_the_words_are_sent_to_an_ai_service_and_to_check_the_prescription(self):
        self.assertIn("sent to an AI service", self.HTML)
        self.assertIn("check this against the prescription", self.HTML)


if __name__ == "__main__":
    unittest.main(verbosity=1)
