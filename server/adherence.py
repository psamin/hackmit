"""Adherence and the streak, worked out from the dose log. Pure functions: no files, no clock of their own.

    summary = summarize(doses, now)            # doses: what doses.replay() returns, values()
    summary.streak, summary.best               # days in a row, and the best run so far
    streak_line(summary)                       # what Pam adds after a tap, or ""
    how_am_i_doing(summary)                    # her answer to "how am I doing?"

--------------------------------------------------------------------------------
WHAT COUNTS, AND WHAT DOES NOT
--------------------------------------------------------------------------------
This measures ANSWERS, never intake: a tap on the card or a spoken yes to Pam (recorded as `via`). A streak is
"days in a row you told Pam you took all of your medication on time". The camera does not count, and nothing here can know a pill was swallowed. That is why the
words are always "marked" and never "took" when Pam speaks about it.

Each scheduled dose is in one of these states:

  on_time      confirmed inside its window
  late         confirmed after its window, within LATE_TAP_GRACE_S. Recorded, but it counts for NOTHING
  pending      still inside its window, or past it but inside the grace period, so a late tap could
               still arrive. Not held against anyone yet
  unconfirmed  the window and the grace period are both over and there was no tap at all
  skipped      never prompted (taken recently, or daily maximum reached). Ignored entirely

A DAY is judged from its doses:

  broken    any dose unconfirmed
  pending   any dose pending
  late      any dose late (and none unconfirmed or pending)
  complete  every dose on time
  none      no doses that day (a weekend medication, a paused one): says nothing

THE STREAK
  A complete day adds one. A broken day resets it to zero. Every other kind of day, including a late one,
  is NEUTRAL: it neither adds nor erases. So a late tap never costs anyone their streak, and never
  earns them a day either. Today, unfinished, is simply pending: the streak stands as it was.

  The one thing that resets it is a dose with no tap at all, once its grace period has passed. That is a
  policy, and it is one line to change (see _step). Doses that were never prompted do not count either way.

WHY A GRACE PERIOD
  Right after a window closes with no tap, nobody knows whether the dose was taken and the tap forgotten.
  Calling the day broken at that instant would reset a streak that a late tap a few minutes later would
  have kept. So an untapped dose stays pending for LATE_TAP_GRACE_S first.

WORDS
  Pam never says a streak was "broken", "lost", "failed" or "missed", never mentions a late or unconfirmed
  dose to the patient, and never compares them with anyone. After a reset she says "let's start a new
  one today". A test scans every sentence in this file for those words.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

LATE_TAP_GRACE_S = 6 * 3600         # a tap this long after a window closes is still accepted, and still neutral

ON_TIME, LATE, PENDING, UNCONFIRMED, SKIPPED = "on_time", "late", "pending", "unconfirmed", "skipped"
COMPLETE, BROKEN, NONE = "complete", "broken", "none"      # (a day can also be LATE or PENDING)


class _Upcoming:
    """A dose the schedule has but that has not opened yet. Nothing can have happened to it."""
    skipped, confirmed_ts, late_ts, closes_ts, source = False, None, None, None, "schedule"

    def __init__(self, name: str, due_ts: float):
        self.name, self.due_ts = name, due_ts


def dose_state(d, now: float) -> str:
    """The state of one SCHEDULED dose at time `now`."""
    if d.skipped:
        return SKIPPED
    if d.confirmed_ts is not None:
        return ON_TIME
    if d.late_ts is not None:
        return LATE
    if d.closes_ts is None or now <= d.closes_ts + LATE_TAP_GRACE_S:
        return PENDING
    return UNCONFIRMED


def day_state(states: list[str]) -> str:
    states = [s for s in states if s != SKIPPED]
    if not states:
        return NONE
    if UNCONFIRMED in states:
        return BROKEN
    if PENDING in states:
        return PENDING
    if LATE in states:
        return LATE
    return COMPLETE


def _step(run: int, state: str) -> int:
    """How one day moves the streak. The whole policy is here.

    complete: +1.  broken: back to zero.  late, pending, none: unchanged (neutral)."""
    if state == COMPLETE:
        return run + 1
    if state == BROKEN:
        return 0
    return run


@dataclass
class DoseSummary:
    name: str
    due_ts: float
    state: str
    confirmed_ts: float | None = None       # when the person tapped or said it, on time or late
    via: str | None = None                  # "tap" or "voice"


@dataclass
class DaySummary:
    day: date
    state: str
    doses: list[DoseSummary] = field(default_factory=list)


@dataclass
class Summary:
    days: list[DaySummary]                  # oldest first, only days that had scheduled doses
    streak: int
    best: int
    today: date
    today_complete: bool

    @property
    def has_history(self) -> bool:
        return bool(self.days)


def summarize(doses, now: float, upcoming=()) -> Summary:
    """Every scheduled dose, by the local day it was due, and the streak they add up to.

    `upcoming` is [(name, due_ts)] for doses the schedule has but that have not opened yet (this evening's).
    They count as PENDING, or the morning dose alone would make today look finished."""
    today = datetime.fromtimestamp(now).date()
    by_day: dict[date, list] = {}
    for d in doses:
        if getattr(d, "source", None) != "schedule":
            continue                          # one-off reminders have no window and no place in a streak
        by_day.setdefault(datetime.fromtimestamp(d.due_ts).date(), []).append(d)
    for name, due_ts in upcoming:
        by_day.setdefault(datetime.fromtimestamp(due_ts).date(), []).append(_Upcoming(name, due_ts))

    days, run, best = [], 0, 0
    for day in sorted(by_day):
        members = sorted(by_day[day], key=lambda x: (x.due_ts, x.name or ""))
        states = [dose_state(m, now) for m in members]
        state = day_state(states)
        if state == NONE:
            continue                          # every dose that day was skipped: nothing happened, so it is not a day
        run = _step(run, state)
        best = max(best, run)
        days.append(DaySummary(day, state, [DoseSummary(m.name or "medication", m.due_ts, s, m.confirmed_ts or m.late_ts, getattr(m, "via", None))
                                            for m, s in zip(members, states) if s != SKIPPED]))
    today_complete = bool(days) and days[-1].day == today and days[-1].state == COMPLETE
    return Summary(days=days, streak=run, best=best, today=today, today_complete=today_complete)


def on_time_rate(summary: Summary, days: int = 7) -> float | None:
    """Of the doses whose outcome is known in the last `days` days, the share confirmed on time (None if none)."""
    since = summary.today - timedelta(days=days - 1)
    settled = [d.state for day in summary.days if day.day >= since for d in day.doses if d.state != PENDING]
    return None if not settled else sum(s == ON_TIME for s in settled) / len(settled)


# --------------------------------------------------------------------------------
# What Pam says to the patient. Encouraging, honest, never blaming.
# --------------------------------------------------------------------------------
def _days(n: int) -> str:
    return f"{n} day" if n == 1 else f"{n} days"


def streak_line(summary: Summary) -> str:
    """Added to Pam's reply when a tap finishes today's medication. "" when there is nothing to celebrate."""
    if not summary.today_complete or summary.streak < 1:
        return ""
    if summary.streak == 1:
        return "That's your first day in a row."
    return f"That's {summary.streak} days in a row."


def still_line(summary: Summary) -> str:
    """After a LATE tap: say the streak is untouched, and say nothing about the tap being late."""
    return f"Your streak is still {_days(summary.streak)}." if summary.streak >= 1 else ""


def how_am_i_doing(summary: Summary) -> str:
    """The answer to "how am I doing?" or "what's my streak?"."""
    if not summary.has_history:
        return "I'll start counting once you've marked your medication."
    if summary.streak >= 1:
        text = f"You've marked all of your medication on time {_days(summary.streak)} in a row."
        if summary.best > summary.streak:
            text += f" Your best is {_days(summary.best)}."
        return text
    if summary.best >= 1:
        return f"Let's start a new streak today. Your best so far is {_days(summary.best)}."
    return "Let's start a streak today. Each day you mark all of your medication on time adds one."


def public(summary: Summary, days: int = 14) -> dict:
    """For Pam oversight (the caregiver's page): the last `days` days, exactly as recorded, honestly labelled."""
    since = summary.today - timedelta(days=days - 1)
    rate = on_time_rate(summary)
    return {
        "streak": summary.streak, "best": summary.best, "today_complete": summary.today_complete,
        "on_time_rate_7d": None if rate is None else round(rate, 3),
        "days": [{"date": day.day.isoformat(), "state": day.state,
                  "doses": [{"name": d.name, "due": datetime.fromtimestamp(d.due_ts).strftime("%H:%M"), "state": d.state,
                             "tapped_at": None if d.confirmed_ts is None else datetime.fromtimestamp(d.confirmed_ts).strftime("%H:%M"),
                             "via": d.via if d.confirmed_ts is not None else None}
                            for d in day.doses]}
                 for day in summary.days if day.day >= since],
    }
