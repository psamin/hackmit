"""get_time_and_place: "what day is it? where am I?" -> one short, calm spoken answer.

    from orientation import time_and_place
    await time_and_place(fix)     # fix = the phone's last location, as server/app.py keeps it

--------------------------------------------------------------------------------
WHAT IT SAYS, AND WHERE EACH PART COMES FROM
--------------------------------------------------------------------------------
    "It's Saturday, September 20th, and it's 2:37 in the afternoon. You're at home.
     Coming up today: Sarah is visiting for dinner, at 6:30 PM."

  day, date, time   the laptop clock, in its own timezone
  place             the phone's fix through places.py. Known places ("home") are stated;
                    a Google guess is hedged ("near ..."); no usable fix says so
  what's next       the first timed calendar event today that has not started yet

Every part is left out or hedged rather than guessed. Pam's prompt already says times and
places come from function results only, so a made-up "you're at home" would be spoken as
fact to someone who cannot check it.

The wording never refers to earlier questions. The same person may ask this five times in
ten minutes, and "as I said before" is the one thing this feature must not say.

--------------------------------------------------------------------------------
TIMEZONES
--------------------------------------------------------------------------------
Calendar feeds store times in UTC ("20260919T200000Z"). Printing that as-is says 8 PM for
a 4 PM dinner, so every event is converted to the timezone of `now` before it is compared
or spoken. Floating times (no zone in the feed) are read as local. All-day events are not
timed, so they are not offered as "next".

KNOWN LIMIT: an event already in progress is not reported as next. Calendar data comes
from the shared calendar integration, including Google's recurring-event expansion.
"""
from __future__ import annotations

from datetime import datetime

try:  # imported as a top-level module when server/app.py runs
    from places import phrase as place_phrase
except ImportError:  # pragma: no cover - only when imported from elsewhere
    place_phrase = lambda loc: ""

# A fix older than this is not "where you are" any more. The caller supplies the age; the
# server does not record one yet, in which case None means "assume it is current".
FIX_MAX_AGE_S = 15 * 60


def ordinal(n: int) -> str:
    """1 -> '1st', 22 -> '22nd', 13 -> '13th'."""
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def day_part(hour: int) -> str:
    """The phrase that follows a clock time, so it reads 'it's 2:37 in the afternoon'."""
    if 5 <= hour < 12:
        return "in the morning"
    if 12 <= hour < 17:
        return "in the afternoon"
    if 17 <= hour < 21:
        return "in the evening"
    return "at night"


def clock(dt: datetime, with_period: bool = False) -> str:
    """'2:37', or '6:30 PM' when the period is wanted."""
    h = dt.hour % 12 or 12
    text = f"{h}:{dt.minute:02d}"
    return f"{text} {'AM' if dt.hour < 12 else 'PM'}" if with_period else text


def parse_events(ics_text: str, now: datetime) -> list[tuple[datetime, str]]:
    """Today's timed events as (start in now's timezone, title), earliest first."""
    from icalendar import Calendar

    out = []
    for ev in Calendar.from_ical(ics_text).walk("VEVENT"):
        start = ev.get("DTSTART").dt
        if not isinstance(start, datetime):  # a plain date: all-day, no time to speak
            continue
        start = start.replace(tzinfo=now.tzinfo) if start.tzinfo is None else start.astimezone(now.tzinfo)
        if start.date() == now.date():
            out.append((start, str(ev.get("SUMMARY", "event"))))
    return sorted(out)


def describe(now: datetime, fix: dict | None, events: list[tuple[datetime, str]] | None,
             fix_age_s: float | None = None) -> dict:
    """Pure function of its inputs, so every branch can be tested without a phone or a feed.

    `events` is None when the calendar could not be read (say so) and [] when it was read
    and today is empty (also say so). Those are different facts.
    """
    parts = [f"It's {now:%A}, {now:%B} {ordinal(now.day)}, and it's {clock(now)} {day_part(now.hour)}."]

    fix = fix or {}
    stale = fix_age_s is not None and fix_age_s > FIX_MAX_AGE_S
    where = "" if stale else place_phrase(fix)
    if where:
        parts.append(f"You're {where}.")
        place_line = where[0].upper() + where[1:]
    else:
        parts.append("I can't tell exactly where you are right now.")
        place_line = "Location not known"

    if events is None:
        parts.append("I couldn't check your calendar just now.")
        next_line, upcoming = "Calendar unavailable", None
    else:
        upcoming = next(((s, t) for s, t in events if s >= now), None)
        if upcoming:
            start, title = upcoming
            parts.append(f"Coming up today: {title}, at {clock(start, with_period=True)}.")
            next_line = f"{title} at {clock(start, with_period=True)}"
        else:
            parts.append("You have nothing else planned today.")
            next_line = "Nothing else planned today"

    return {
        "say": " ".join(parts),
        "card": {"title": "Right now",
                 "body": "\n".join([f"{now:%A}, {now:%B} {now.day}", f"{clock(now, True)}", place_line, next_line])},
        "day": f"{now:%A}", "date": f"{now:%Y-%m-%d}", "time": clock(now, True),
        "place": fix.get("place") if where else None,
        "next_event": None if not upcoming else {"title": upcoming[1], "time": clock(upcoming[0], True)},
    }


async def time_and_place(fix: dict | None, fix_age_s: float | None = None, calendar_reader=None) -> dict:
    now = datetime.now().astimezone()
    events = None
    try:
        if calendar_reader is None:
            from app import calendar as calendar_reader
        result = await calendar_reader()
        if result.get("status") == "connected":
            events = []
            for event in result["events"]:
                if event.get("all_day"):
                    continue
                start = datetime.fromisoformat(event["_dt"].replace("Z", "+00:00"))
                start = start.replace(tzinfo=now.tzinfo) if start.tzinfo is None else start.astimezone(now.tzinfo)
                if start.date() == now.date():
                    events.append((start, event["title"]))
            events.sort()
    except Exception:
        events = None
    return describe(now, fix, events, fix_age_s)


if __name__ == "__main__":  # quick look, no phone: python server/orientation.py
    import asyncio
    print(asyncio.run(time_and_place(None))["say"])
