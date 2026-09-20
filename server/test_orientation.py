"""Tests for orientation.py. No phone, network, or API key needed.

    python server/test_orientation.py
"""
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import orientation as o  # noqa: E402

EDT = timezone(timedelta(hours=-4))
SAT_2PM = datetime(2026, 9, 19, 14, 37, tzinfo=EDT)   # a Saturday afternoon
HOME = {"place": "home", "source": "known"}


def ics(*events):
    body = "".join(f"BEGIN:VEVENT\nUID:{i}@t\nDTSTART{dt}\nSUMMARY:{title}\nEND:VEVENT\n"
                   for i, (dt, title) in enumerate(events))
    return f"BEGIN:VCALENDAR\nVERSION:2.0\nPRODID:-//t//EN\n{body}END:VCALENDAR\n"


class Wording(unittest.TestCase):
    def test_ordinals(self):
        got = [o.ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 21, 22, 23, 30, 31)]
        self.assertEqual(got, ["1st", "2nd", "3rd", "4th", "11th", "12th", "13th",
                               "21st", "22nd", "23rd", "30th", "31st"])

    def test_day_part_boundaries(self):
        self.assertEqual([o.day_part(h) for h in (4, 5, 11, 12, 16, 17, 20, 21, 0)],
                         ["at night", "in the morning", "in the morning", "in the afternoon",
                          "in the afternoon", "in the evening", "in the evening", "at night", "at night"])

    def test_clock_uses_twelve_hour_time(self):
        self.assertEqual(o.clock(datetime(2026, 9, 19, 0, 5)), "12:05")
        self.assertEqual(o.clock(datetime(2026, 9, 19, 12, 0)), "12:00")
        self.assertEqual(o.clock(datetime(2026, 9, 19, 18, 30), with_period=True), "6:30 PM")
        self.assertEqual(o.clock(datetime(2026, 9, 19, 9, 5), with_period=True), "9:05 AM")

    def test_says_day_date_and_time(self):
        say = o.describe(SAT_2PM, HOME, [])["say"]
        self.assertIn("It's Saturday, September 19th, and it's 2:37 in the afternoon.", say)

    def test_never_refers_to_earlier_questions(self):
        # Asked five times in ten minutes must sound the same as the first time.
        for events in ([], None, [(SAT_2PM + timedelta(hours=2), "Dinner")]):
            say = o.describe(SAT_2PM, HOME, events)["say"].lower()
            for banned in ("already", "again", "before", "as i said", "remember"):
                self.assertNotIn(banned, say)


class Place(unittest.TestCase):
    def test_known_place_is_stated(self):
        r = o.describe(SAT_2PM, HOME, [])
        self.assertIn("You're at home.", r["say"])
        self.assertEqual(r["place"], "home")

    def test_google_guess_is_hedged(self):
        r = o.describe(SAT_2PM, {"place": "Boston Common", "source": "google"}, [])
        self.assertIn("You're near Boston Common.", r["say"])

    def test_no_usable_fix_admits_it(self):
        for fix in (None, {}, {"place": None, "source": "no_fix"}, {"place": None, "source": "unknown"},
                    {"place": None, "source": "too_inaccurate"}):
            r = o.describe(SAT_2PM, fix, [])
            self.assertIn("I can't tell exactly where you are", r["say"], fix)
            self.assertNotIn("You're at", r["say"], fix)
            self.assertIsNone(r["place"])

    def test_stale_fix_is_not_reported_as_current(self):
        r = o.describe(SAT_2PM, HOME, [], fix_age_s=o.FIX_MAX_AGE_S + 1)
        self.assertIn("I can't tell exactly where you are", r["say"])
        self.assertNotIn("home", r["say"])

    def test_fresh_fix_is_reported(self):
        self.assertIn("You're at home.", o.describe(SAT_2PM, HOME, [], fix_age_s=60)["say"])


class Calendar(unittest.TestCase):
    def test_utc_event_is_spoken_in_local_time(self):
        # 22:30 UTC is 6:30 PM in EDT. Printing the UTC hour would say 10:30 PM.
        events = o.parse_events(ics((":20260919T223000Z", "Sarah is visiting for dinner")), SAT_2PM)
        say = o.describe(SAT_2PM, HOME, events)["say"]
        self.assertIn("Coming up today: Sarah is visiting for dinner, at 6:30 PM.", say)
        self.assertNotIn("10:30", say)

    def test_next_event_skips_past_ones(self):
        events = o.parse_events(ics((":20260919T130000Z", "Morning walk"),       # 9:00 AM local
                                    (":20260919T200000Z", "Afternoon medication"),  # 4:00 PM local
                                    (":20260919T223000Z", "Dinner")), SAT_2PM)     # 6:30 PM local
        r = o.describe(SAT_2PM, HOME, events)
        self.assertEqual(r["next_event"], {"title": "Afternoon medication", "time": "4:00 PM"})

    def test_events_on_other_days_are_ignored(self):
        events = o.parse_events(ics((":20260918T200000Z", "Yesterday"), (":20260920T200000Z", "Tomorrow")), SAT_2PM)
        self.assertEqual(events, [])

    def test_utc_late_evening_is_still_today_locally(self):
        # 02:00 UTC on the 20th is 10 PM on the 19th in EDT: today, not tomorrow.
        events = o.parse_events(ics((":20260920T020000Z", "Late show")), SAT_2PM)
        self.assertEqual([t for _, t in events], ["Late show"])

    def test_all_day_events_have_no_time_to_speak(self):
        self.assertEqual(o.parse_events(ics((";VALUE=DATE:20260919", "Birthday")), SAT_2PM), [])

    def test_floating_time_is_read_as_local(self):
        events = o.parse_events(ics((":20260919T160000", "Physio")), SAT_2PM)
        self.assertEqual(o.describe(SAT_2PM, HOME, events)["next_event"]["time"], "4:00 PM")

    def test_nothing_left_today(self):
        events = o.parse_events(ics((":20260919T130000Z", "Morning walk")), SAT_2PM)
        self.assertIn("You have nothing else planned today.", o.describe(SAT_2PM, HOME, events)["say"])

    def test_unreadable_calendar_is_not_reported_as_empty(self):
        # None (could not read) and [] (read, nothing there) are different facts.
        unread = o.describe(SAT_2PM, HOME, None)["say"]
        empty = o.describe(SAT_2PM, HOME, [])["say"]
        self.assertIn("I couldn't check your calendar", unread)
        self.assertNotIn("nothing else planned", unread)
        self.assertIn("nothing else planned", empty)

    def test_event_starting_exactly_now_counts_as_next(self):
        r = o.describe(SAT_2PM, HOME, [(SAT_2PM, "Pills")])
        self.assertEqual(r["next_event"]["title"], "Pills")


class Shape(unittest.TestCase):
    def test_result_has_a_card_and_structured_fields(self):
        r = o.describe(SAT_2PM, HOME, [(SAT_2PM + timedelta(hours=4), "Dinner")])
        self.assertEqual(r["card"]["title"], "Right now")
        self.assertEqual(r["card"]["body"].splitlines(),
                         ["Saturday, September 19", "2:37 PM", "At home", "Dinner at 6:37 PM"])
        self.assertEqual((r["day"], r["date"], r["time"]), ("Saturday", "2026-09-19", "2:37 PM"))

    def test_midnight_hours_read_as_night(self):
        night = datetime(2026, 9, 19, 0, 20, tzinfo=EDT)
        self.assertIn("it's 12:20 at night.", o.describe(night, HOME, [])["say"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
