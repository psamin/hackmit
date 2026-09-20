"""The medication schedule: what the caregiver has set, checked, stored and explained.

    from schedule import Store, check, describe
    result = check(data)            # errors a person can act on, plus warnings; saves nothing
    entry = Store().save(data)      # validate, version, log, then it is the current schedule
    Store().load().schedule         # the current schedule, or None if none was ever saved
    python server/schedule.py check schedule.json     # look at a file without saving it

--------------------------------------------------------------------------------
WHAT THIS IS, AND IS NOT
--------------------------------------------------------------------------------
This holds a schedule a CAREGIVER wrote (or that a language model drafted and the caregiver
approved). It checks that the schedule is well formed and consistent with itself. It does NOT
check that it is medically right: nothing here knows what a drug is, what a safe dose is, or
whether two medications interact. It gives no dosing advice and adds none. A pharmacist or
clinician should review a real patient's schedule.

Nothing reads this yet. Reminders, doses and streaks arrive in later changes; until then it
cannot affect anything that is running.

--------------------------------------------------------------------------------
HOW IT IS STORED, AND WHY
--------------------------------------------------------------------------------
One append-only file, medications.jsonl (gitignored: it is patient data). Every save adds one
line holding the whole schedule at that moment, who saved it, when, the caregiver's original
words if there were any, and a plain-English list of what changed. The CURRENT schedule is
simply the last complete line.

  - There is no second "current" file that could disagree with the history. Nothing can be
    changed without leaving a line, and a line cannot be edited without it showing.
  - A save that crashed halfway leaves a torn last line, which is ignored, so the previous
    schedule stays in force. That is the right outcome: the save did not finish.
  - "No schedule yet" (no file) and "the file is damaged" (lines exist, none readable) are
    different things and raise different errors. A damaged file must never quietly look like
    an empty schedule, because that would mean no reminders and no one knowing why.

Times are the patient's local wall-clock times and are stored as "HH:MM" strings.

--------------------------------------------------------------------------------
THE ON-TIME WINDOW
--------------------------------------------------------------------------------
A dose due at 08:00 with early_minutes=30 and late_minutes=90 counts as on time from 07:30 to
09:30. Windows for one medication may not overlap (a tap must belong to exactly one dose), and
min_gap_hours / max_per_day may not contradict the times themselves. These are stored now so
the format is stable; they are used by later changes.
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

HERE = Path(__file__).resolve().parent
STORE_PATH = HERE / "medications.jsonl"

WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
DAY_NAMES = dict(zip(WEEKDAYS, ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")))
Weekday = Literal["mon", "tue", "wed", "thu", "fri", "sat", "sun"]

MAX_MEDICATIONS = 30
MAX_TIMES = 8
NAME_MAX, NOTE_MAX, PATIENT_MAX = 80, 300, 80
TIME_RE = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")
ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")
NIGHT_START, NIGHT_END = 22 * 60, 6 * 60   # 22:00 up to 05:59 draws a warning


class ScheduleError(Exception):
    """The schedule was rejected. `errors` are sentences a caregiver can act on."""

    def __init__(self, errors: list[str], warnings: list[str] | None = None):
        super().__init__("; ".join(errors))
        self.errors, self.warnings = errors, warnings or []


class ScheduleCorrupt(Exception):
    """The store has content but no readable entry. Not the same as "no schedule yet"."""


# --------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------
def _minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def fmt_time(hhmm: str) -> str:
    """'20:30' -> '8:30 PM'."""
    mins = _minutes(hhmm)
    h, m = divmod(mins, 60)
    return f"{h % 12 or 12}:{m:02d} {'AM' if h < 12 else 'PM'}"


def fmt_duration(minutes: float) -> str:
    """90 -> '1 hour 30 minutes'."""
    minutes = int(round(minutes))
    h, m = divmod(minutes, 60)
    parts = ([f"{h} hour{'s' if h != 1 else ''}"] if h else []) + ([f"{m} minute{'s' if m != 1 else ''}"] if m or not h else [])
    return " ".join(parts)


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.casefold()).strip("-")[:40].strip("-")
    return slug or "medication"


def _text(v, what: str, max_len: int, allow_empty: bool = False) -> str:
    if not isinstance(v, str):
        raise ValueError(f"The {what} must be text.")
    v = " ".join(v.split())               # collapse runs of whitespace, line breaks and tabs
    if not v and not allow_empty:
        raise ValueError(f"The {what} can't be empty.")
    if len(v) > max_len:
        raise ValueError(f"The {what} is too long ({len(v)} characters; the limit is {max_len}).")
    if any(ord(c) < 32 or ord(c) == 127 for c in v):
        raise ValueError(f"The {what} can't contain control characters.")
    return v


# --------------------------------------------------------------------------------
# The model. Strict about what it accepts, so a mistake is caught here, not at 8 AM.
# --------------------------------------------------------------------------------
class Medication(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str | None = None                 # assigned on save, then stable so history follows the medication
    name: str
    times: list[str] = Field(default_factory=list)
    days: list[Weekday] = Field(default_factory=lambda: list(WEEKDAYS))
    early_minutes: int = Field(30, ge=0, le=120)
    late_minutes: int = Field(90, ge=15, le=360)
    min_gap_hours: float | None = Field(None, gt=0, le=72)
    max_per_day: int | None = Field(None, ge=1, le=24)
    as_needed: bool = False
    start_date: date | None = None
    end_date: date | None = None
    note: str = ""                        # the caregiver's words, shown as written, never spoken or reworded
    active: bool = True

    @field_validator("early_minutes", "late_minutes", "max_per_day", "min_gap_hours", mode="before")
    @classmethod
    def _not_a_bool(cls, v):
        if isinstance(v, bool):
            raise ValueError("must be a number, not true or false.")
        return v

    @field_validator("id")
    @classmethod
    def _id(cls, v):
        if v is not None and not ID_RE.match(v):
            raise ValueError("ids use lowercase letters, digits and hyphens only.")
        return v

    @field_validator("name")
    @classmethod
    def _name(cls, v):
        return _text(v, "medication name", NAME_MAX)

    @field_validator("note")
    @classmethod
    def _note(cls, v):
        return _text(v, "note", NOTE_MAX, allow_empty=True)

    @field_validator("times")
    @classmethod
    def _times(cls, v):
        for t in v:
            if not isinstance(t, str) or not TIME_RE.match(t):
                raise ValueError(f"'{t}' isn't a valid time. Use 24-hour HH:MM, for example 08:00 or 20:30.")
        if len(set(v)) != len(v):
            raise ValueError("the same time is listed twice.")
        if len(v) > MAX_TIMES:
            raise ValueError(f"at most {MAX_TIMES} times a day are supported.")
        return sorted(v)

    @field_validator("days")
    @classmethod
    def _days(cls, v):
        if not v:
            raise ValueError("choose at least one day.")
        if len(set(v)) != len(v):
            raise ValueError("a day is listed twice.")
        return sorted(v, key=WEEKDAYS.index)

    @model_validator(mode="after")
    def _consistent(self):
        label = self.name
        if self.as_needed and self.times:
            raise ValueError(f"{label} is 'as needed', so it can't also have set times.")
        if not self.as_needed and not self.times:
            raise ValueError(f"{label} needs at least one time, or mark it 'as needed'.")
        if self.start_date and self.end_date and self.end_date < self.start_date:
            raise ValueError(f"{label}: the end date is before the start date.")
        if not self.times:
            return self
        if self.max_per_day is not None and self.max_per_day < len(self.times):
            raise ValueError(f"{label} has {len(self.times)} times a day but a daily maximum of {self.max_per_day}.")

        mins = [_minutes(t) for t in self.times]
        idx = {WEEKDAYS.index(d) for d in self.days}
        runs_into_next_day = any((i + 1) % 7 in idx for i in idx)      # a dose one day, another the next
        pairs = list(zip(mins, mins[1:])) + ([(mins[-1], mins[0] + 1440)] if runs_into_next_day else [])
        if self.min_gap_hours is not None and pairs:
            tightest = min(b - a for a, b in pairs)
            if tightest < self.min_gap_hours * 60:
                raise ValueError(f"{label}: the doses can be as close as {fmt_duration(tightest)}, "
                                 f"but the minimum gap is {fmt_duration(self.min_gap_hours * 60)}.")
        for a, b in pairs:
            if a + self.late_minutes >= b - self.early_minutes:
                raise ValueError(f"{label}: the on-time windows for {fmt_time(f'{(a // 60) % 24:02d}:{a % 60:02d}')} and "
                                 f"{fmt_time(f'{(b // 60) % 24:02d}:{b % 60:02d}')} overlap, so a tap couldn't be matched "
                                 f"to one dose. Move the times apart or shorten the window.")
        return self


class Schedule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patient_name: str | None = None
    medications: list[Medication] = Field(default_factory=list, max_length=MAX_MEDICATIONS)

    @field_validator("patient_name")
    @classmethod
    def _patient(cls, v):
        return None if v is None else _text(v, "patient name", PATIENT_MAX)

    @model_validator(mode="after")
    def _unique(self):
        seen: dict[str, str] = {}
        for m in self.medications:
            if m.name.casefold() in seen:
                raise ValueError(f"Two medications are named '{m.name}'. Give each a distinct name.")
            seen[m.name.casefold()] = m.name
        ids = [m.id for m in self.medications if m.id]
        if len(set(ids)) != len(ids):
            raise ValueError("Two medications share the same id.")
        return self


# --------------------------------------------------------------------------------
# Checking: plain-English errors and warnings
# --------------------------------------------------------------------------------
FIELD_WORDS = {"name": "name", "times": "times", "days": "days", "early_minutes": "minutes early",
               "late_minutes": "minutes late", "min_gap_hours": "minimum hours between doses",
               "max_per_day": "daily maximum", "start_date": "start date", "end_date": "end date",
               "note": "note", "as_needed": "as-needed setting", "active": "active setting", "id": "id"}


@dataclass
class Checked:
    schedule: Schedule | None
    errors: list[str]
    warnings: list[str]

    @property
    def ok(self) -> bool:
        return self.schedule is not None and not self.errors


RANGE_TEXT = {"early_minutes": "between 0 and 120 minutes", "late_minutes": "between 15 and 360 minutes",
              "min_gap_hours": "more than 0 and at most 72 hours", "max_per_day": "between 1 and 24"}


def _plain(err: dict, field: str) -> str:
    """The library's message says 'Input should be...'; a caregiver needs 'must be between...'."""
    kind, msg = err["type"], err["msg"].removeprefix("Value error, ")
    if kind in ("greater_than_equal", "less_than_equal", "greater_than", "less_than") and field in RANGE_TEXT:
        return f"must be {RANGE_TEXT[field]}."
    if kind in ("int_parsing", "int_from_float", "int_type"):
        return "must be a whole number."
    if kind in ("float_parsing", "float_type"):
        return "must be a number."
    if kind in ("date_parsing", "date_from_datetime_parsing", "date_type", "date_from_datetime_inexact"):
        return "isn't a valid date. Use YYYY-MM-DD, for example 2026-09-21."
    if kind == "literal_error":
        return "isn't a day. Use mon, tue, wed, thu, fri, sat or sun."
    if kind in ("bool_parsing", "bool_type"):
        return "must be true or false."
    if kind == "string_type":
        return "must be text."
    if kind == "list_type":
        return "must be a list."
    if kind == "missing":
        return "is required."
    if kind == "too_long":
        return f"has too many entries (at most {MAX_MEDICATIONS} medications)."
    return msg


def _problems(exc: ValidationError, data) -> list[str]:
    out = []
    meds = data.get("medications") if isinstance(data, dict) else None
    for err in exc.errors():
        loc = err["loc"]
        msg = _plain(err, str(loc[2]) if len(loc) > 2 else str(loc[-1]) if loc else "")
        if err["type"] == "extra_forbidden":
            msg = f"'{loc[-1]}' isn't a setting this schedule understands. Put dosing instructions in the note."
        if len(loc) >= 2 and loc[0] == "medications" and isinstance(loc[1], int):
            name = None
            if isinstance(meds, list) and loc[1] < len(meds) and isinstance(meds[loc[1]], dict):
                name = meds[loc[1]].get("name")
            name = name if isinstance(name, str) and name.strip() else f"Medication {loc[1] + 1}"
            if len(loc) == 2:
                out.append(msg)                                   # a whole-medication rule; it names the medication itself
            elif err["type"] == "extra_forbidden":
                out.append(f"{name}: {msg}")                      # the message already names the setting
            else:
                out.append(f"{name}: {FIELD_WORDS.get(str(loc[2]), str(loc[2]))}: {msg}")
        elif loc and loc[0] == "medications":
            out.append(f"Medications: {msg}")
        elif loc:
            out.append(f"{FIELD_WORDS.get(str(loc[0]), str(loc[0]).replace('_', ' ').capitalize())}: {msg}")
        else:
            out.append(msg if "valid dictionary" not in msg else "The schedule isn't in the expected form.")
    return out


def check(data, today: date | None = None) -> Checked:
    """Validate without saving. Never raises for bad input; look at `.errors`."""
    if isinstance(data, Schedule):
        data = data.model_dump(mode="json")
    try:
        schedule = Schedule.model_validate(data)
    except ValidationError as exc:
        return Checked(None, _problems(exc, data), [])
    return Checked(schedule, [], _warnings(schedule, today or date.today()))


def _warnings(s: Schedule, today: date) -> list[str]:
    out = []
    if not s.medications:
        out.append("The schedule has no medications, so Pam won't remind about anything.")
    for m in s.medications:
        for t in m.times:
            if _minutes(t) >= NIGHT_START or _minutes(t) < NIGHT_END:
                out.append(f"{m.name} is set for {fmt_time(t)}, in the middle of the night. Check that's intended.")
        if m.as_needed and m.max_per_day is None:
            out.append(f"{m.name} is 'as needed' with no daily limit set.")
        if m.end_date and m.end_date < today:
            out.append(f"{m.name}'s course ended on {m.end_date:%b} {m.end_date.day}, so it will never be due.")
    return out


# --------------------------------------------------------------------------------
# Explaining it back, in words. This is what the caregiver reviews before saving.
# --------------------------------------------------------------------------------
def _days_phrase(days: list[str]) -> str:
    s = set(days)
    if len(s) == 7:
        return "every day"
    if s == {"mon", "tue", "wed", "thu", "fri"}:
        return "on weekdays"
    if s == {"sat", "sun"}:
        return "on weekends"
    if len(s) == 1:
        return f"every {DAY_NAMES[days[0]]}"
    return "on " + _join([DAY_NAMES[d] for d in days])


def describe_one(m: Medication) -> str:
    parts = [m.name + ": "]
    if m.as_needed:
        parts.append("only when needed, with no reminders")
        if m.max_per_day:
            parts.append(f", at most {m.max_per_day} a day")
    else:
        parts.append(f"{_days_phrase(m.days)} at {_join([fmt_time(t) for t in m.times])}")
        parts.append(f" (counts as on time from {fmt_duration(m.early_minutes)} before to {fmt_duration(m.late_minutes)} after)")
        if m.min_gap_hours:
            parts.append(f", at least {fmt_duration(m.min_gap_hours * 60)} apart")
        if m.max_per_day:
            parts.append(f", no more than {m.max_per_day} a day")
    if m.start_date and m.end_date:
        parts.append(f", from {m.start_date:%b} {m.start_date.day} to {m.end_date:%b} {m.end_date.day}")
    elif m.start_date:
        parts.append(f", starting {m.start_date:%b} {m.start_date.day}")
    elif m.end_date:
        parts.append(f", until {m.end_date:%b} {m.end_date.day}")
    if not m.active:
        parts.append(" [paused: no reminders]")
    if m.note:
        parts.append(f'. Note from the caregiver: "{m.note}"')
    return "".join(parts)


def describe(s: Schedule) -> list[str]:
    return [describe_one(m) for m in s.medications]


def diff(old: Schedule | None, new: Schedule) -> list[str]:
    """What changed, in words. Matches medications by id; falls back to name for unsaved drafts."""
    lines: list[str] = []
    if (old.patient_name if old else None) != new.patient_name:
        lines.append(f"Patient name: {(old.patient_name if old else None) or 'none'} -> {new.patient_name or 'none'}")
    key = lambda m: m.id or m.name.casefold()  # noqa: E731
    before = {key(m): m for m in (old.medications if old else [])}
    after = {key(m): m for m in new.medications}
    for k, m in after.items():
        if k not in before:
            lines.append(f"Added: {describe_one(m)}")
        elif m.model_dump(exclude={"id"}) != before[k].model_dump(exclude={"id"}):
            lines.append(f"Changed: {describe_one(before[k])}  ->  {describe_one(m)}")
    for k, m in before.items():
        if k not in after:
            lines.append(f"Removed: {m.name}")
    return lines


# --------------------------------------------------------------------------------
# Ids and dose slots
# --------------------------------------------------------------------------------
def assign_ids(schedule: Schedule, previous: Schedule | None) -> Schedule:
    """Give every medication a stable id. A medication that keeps its name keeps its id, so its
    history (and later its streak) follows it through edits; a new one gets a slug of its name."""
    prev = {m.name.casefold(): m.id for m in (previous.medications if previous else []) if m.id}
    taken = {m.id for m in schedule.medications if m.id}
    out = []
    for m in schedule.medications:
        if m.id:
            out.append(m)
            continue
        base = prev.get(m.name.casefold()) or slugify(m.name)
        cand, n = base, 2
        while cand in taken:
            cand, n = f"{base[:36].rstrip('-')}-{n}", n + 1
        taken.add(cand)
        out.append(m.model_copy(update={"id": cand}))
    return schedule.model_copy(update={"medications": out})


@dataclass(frozen=True)
class Slot:
    """One scheduled dose. `due`, `opens`, `closes` are naive local datetimes."""
    med_id: str
    name: str
    due: datetime
    opens: datetime
    closes: datetime
    note: str


def slots_for_day(schedule: Schedule, day: date) -> list[Slot]:
    """The doses due on `day`, earliest first. Paused, as-needed and out-of-course medications
    contribute none. A window can run past midnight; the times are datetimes, not clock times."""
    out = []
    for m in schedule.medications:
        if not m.active or m.as_needed:
            continue
        if (m.start_date and day < m.start_date) or (m.end_date and day > m.end_date):
            continue
        if WEEKDAYS[day.weekday()] not in m.days:
            continue
        for t in m.times:
            due = datetime.combine(day, datetime.min.time()) + timedelta(minutes=_minutes(t))
            out.append(Slot(m.id or slugify(m.name), m.name, due, due - timedelta(minutes=m.early_minutes),
                            due + timedelta(minutes=m.late_minutes), m.note))
    return sorted(out, key=lambda s: (s.due, s.name))


# --------------------------------------------------------------------------------
# The store
# --------------------------------------------------------------------------------
@dataclass(frozen=True)
class Entry:
    version: int
    ts: str
    actor: str
    action: str                       # "save" or "restore"
    source_text: str | None           # the caregiver's original words, when a draft came from them
    changes: list[str]
    schedule: Schedule = field(compare=False)

    def to_json(self) -> str:
        return json.dumps({"version": self.version, "ts": self.ts, "actor": self.actor, "action": self.action,
                           "source_text": self.source_text, "changes": self.changes,
                           "schedule": self.schedule.model_dump(mode="json")}, ensure_ascii=False)

    @classmethod
    def from_line(cls, line: str) -> "Entry | None":
        """None for anything that is not a complete, valid entry (a torn line, junk, a bad schedule)."""
        try:
            d = json.loads(line)
            return cls(int(d["version"]), str(d["ts"]), str(d["actor"]), str(d["action"]), d.get("source_text"),
                       [str(c) for c in d.get("changes", [])], Schedule.model_validate(d["schedule"]))
        except (ValueError, KeyError, TypeError, ValidationError):
            return None


@lru_cache(maxsize=1)
def _windows_file_api():
    import ctypes as ct
    from ctypes import wintypes as wt

    class SecurityAttributes(ct.Structure):
        _fields_ = [("length", wt.DWORD), ("descriptor", ct.c_void_p), ("inherit", wt.BOOL)]

    class TokenUser(ct.Structure):
        _fields_ = [("sid", ct.c_void_p), ("attributes", wt.DWORD)]

    kernel = ct.WinDLL("kernel32", use_last_error=True)
    security = ct.WinDLL("advapi32", use_last_error=True)
    ptr, out_ptr = ct.c_void_p, ct.POINTER(ct.c_void_p)
    signatures = [
        (kernel.GetCurrentProcess, [], wt.HANDLE),
        (kernel.CloseHandle, [wt.HANDLE], wt.BOOL),
        (kernel.LocalFree, [ptr], ptr),
        (kernel.CreateFileW, [wt.LPCWSTR, wt.DWORD, wt.DWORD, ct.POINTER(SecurityAttributes), wt.DWORD, wt.DWORD, wt.HANDLE], wt.HANDLE),
        (security.OpenProcessToken, [wt.HANDLE, wt.DWORD, ct.POINTER(wt.HANDLE)], wt.BOOL),
        (security.GetTokenInformation, [wt.HANDLE, ct.c_int, ptr, wt.DWORD, ct.POINTER(wt.DWORD)], wt.BOOL),
        (security.ConvertSidToStringSidW, [ptr, ct.POINTER(wt.LPWSTR)], wt.BOOL),
        (security.ConvertStringSecurityDescriptorToSecurityDescriptorW, [wt.LPCWSTR, wt.DWORD, out_ptr, ct.POINTER(wt.DWORD)], wt.BOOL),
        (security.GetSecurityDescriptorDacl, [ptr, ct.POINTER(wt.BOOL), out_ptr, ct.POINTER(wt.BOOL)], wt.BOOL),
        (security.GetSecurityInfo, [wt.HANDLE, ct.c_int, wt.DWORD, out_ptr, out_ptr, out_ptr, out_ptr, out_ptr], wt.DWORD),
        (security.SetSecurityInfo, [wt.HANDLE, ct.c_int, wt.DWORD, ptr, ptr, ptr, ptr], wt.DWORD),
        (security.EqualSid, [ptr, ptr], wt.BOOL),
    ]
    for function, args, result in signatures:
        function.argtypes, function.restype = args, result
    return kernel, security, SecurityAttributes, TokenUser


def private_append_fd(path: Path) -> int:
    """Open `path` for appending, readable only by the user running Pam. Also used by
    setup.py for server/.env, which holds API keys."""
    if os.name != "nt":
        fd = os.open(path, os.O_RDWR | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            if os.fstat(fd).st_uid != os.geteuid():
                raise PermissionError("Medication history must be owned by the user running Pam.")
            os.fchmod(fd, 0o600)
            return fd
        except BaseException:
            os.close(fd)
            raise
    import ctypes as ct
    import msvcrt
    from ctypes import wintypes as wt

    kernel, security, SecurityAttributes, TokenUser = _windows_file_api()
    token, sid_text = wt.HANDLE(), wt.LPWSTR()
    descriptor, owner_descriptor, owner = ct.c_void_p(), ct.c_void_p(), ct.c_void_p()
    handle = None
    try:
        if not security.OpenProcessToken(kernel.GetCurrentProcess(), 0x0008, ct.byref(token)):
            raise ct.WinError(ct.get_last_error())
        needed = wt.DWORD()
        security.GetTokenInformation(token, 1, None, 0, ct.byref(needed))
        if not needed.value:
            raise ct.WinError(ct.get_last_error())
        user_buffer = ct.create_string_buffer(needed.value)
        if not security.GetTokenInformation(token, 1, user_buffer, needed.value, ct.byref(needed)):
            raise ct.WinError(ct.get_last_error())
        user_sid = ct.cast(user_buffer, ct.POINTER(TokenUser)).contents.sid
        if not security.ConvertSidToStringSidW(user_sid, ct.byref(sid_text)):
            raise ct.WinError(ct.get_last_error())
        sddl = f"O:{sid_text.value}D:P(A;;FA;;;{sid_text.value})"
        if not security.ConvertStringSecurityDescriptorToSecurityDescriptorW(sddl, 1, ct.byref(descriptor), None):
            raise ct.WinError(ct.get_last_error())
        present, defaulted, dacl = wt.BOOL(), wt.BOOL(), ct.c_void_p()
        if not security.GetSecurityDescriptorDacl(descriptor, ct.byref(present), ct.byref(dacl), ct.byref(defaulted)):
            raise ct.WinError(ct.get_last_error())
        if not present.value or not dacl.value:
            raise PermissionError("Windows did not provide an owner-only access list.")
        attributes = SecurityAttributes(ct.sizeof(SecurityAttributes), descriptor, False)
        access = 0x80000000 | 0x40000000 | 0x00040000
        handle = kernel.CreateFileW(str(path), access, 3, ct.byref(attributes), 4, 0x80, None)
        if handle == ct.c_void_p(-1).value:
            handle = None
            raise ct.WinError(ct.get_last_error())
        error = security.GetSecurityInfo(handle, 1, 1, ct.byref(owner), None, None, None, ct.byref(owner_descriptor))
        if error:
            raise ct.WinError(error)
        if not owner.value or not security.EqualSid(owner, user_sid):
            raise PermissionError("Medication history must be owned by the Windows user running Pam.")
        error = security.SetSecurityInfo(handle, 1, 0x80000004, None, None, dacl, None)
        if error:
            raise ct.WinError(error)
        fd = msvcrt.open_osfhandle(handle, os.O_RDWR | os.O_APPEND | os.O_BINARY | os.O_NOINHERIT)
        handle = None
        return fd
    finally:
        if handle is not None:
            kernel.CloseHandle(handle)
        if owner_descriptor.value:
            kernel.LocalFree(owner_descriptor)
        if descriptor.value:
            kernel.LocalFree(descriptor)
        if sid_text:
            kernel.LocalFree(ct.cast(sid_text, ct.c_void_p))
        if token.value:
            kernel.CloseHandle(token)


class Store:
    def __init__(self, path: Path | str | None = None, clock: Callable[[], float] = time.time):
        self.path = Path(path) if path else STORE_PATH
        self.clock = clock
        self._lock = threading.Lock()

    def _entries(self) -> list[Entry]:
        """Oldest first. Raises ScheduleCorrupt only when there is content and none of it is readable."""
        if not self.path.exists():
            return []
        lines = [ln for ln in self.path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        entries = [e for e in (Entry.from_line(ln) for ln in lines) if e]
        if lines and not entries:
            raise ScheduleCorrupt(f"{self.path.name} has content but no readable entry. "
                                  "Restore it from a backup, or move it aside to start again.")
        return entries

    def load(self) -> Entry | None:
        """The current entry, or None if no schedule was ever saved."""
        entries = self._entries()
        return entries[-1] if entries else None

    def history(self, limit: int | None = None) -> list[Entry]:
        """Newest first."""
        entries = self._entries()[::-1]
        return entries[:limit] if limit else entries

    def save(self, data, *, actor: str = "caregiver", source_text: str | None = None, action: str = "save") -> Entry:
        """Validate, version and log. Raises ScheduleError (nothing written) if it is not valid."""
        with self._lock:
            now = self.clock()
            checked = check(data, date.fromtimestamp(now))
            if not checked.ok:
                raise ScheduleError(checked.errors, checked.warnings)
            prev = self.load()                                     # may raise ScheduleCorrupt: do not guess
            schedule = assign_ids(checked.schedule, prev.schedule if prev else None)
            entry = Entry(version=(prev.version + 1) if prev else 1,
                          ts=datetime.fromtimestamp(now).isoformat(timespec="seconds"), actor=actor, action=action,
                          source_text=source_text, changes=diff(prev.schedule if prev else None, schedule),
                          schedule=schedule)
            self._append(entry.to_json() + "\n")
            return entry

    def restore(self, version: int, *, actor: str = "caregiver") -> Entry:
        """Save an old version again as a NEW version. History is never rewritten."""
        match = next((e for e in self._entries() if e.version == version), None)
        if match is None:
            raise ScheduleError([f"There is no version {version} to restore."])
        return self.save(match.schedule, actor=actor, action="restore", source_text=f"Restored version {version}")

    def _append(self, line: str) -> None:
        """One write, then fsync: a crash leaves either the whole line or a torn last line, which is ignored.

        If the file already ends mid-line (a torn write), start on a fresh line. Otherwise the new
        entry would be glued onto the fragment and both would be unreadable."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = private_append_fd(self.path)      # patient data: owner only
        try:
            size = os.fstat(fd).st_size
            if size:
                os.lseek(fd, -1, os.SEEK_END)
                if os.read(fd, 1) != b"\n":
                    line = "\n" + line
            os.write(fd, line.encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)


# --------------------------------------------------------------------------------
# A way to look at a file without saving anything:  python server/schedule.py check schedule.json
# --------------------------------------------------------------------------------
def _cli(argv: list[str]) -> int:
    if len(argv) == 3 and argv[1] == "check":
        try:
            data = json.loads(Path(argv[2]).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"Can't read {argv[2]}: {exc}")
            return 2
        result = check(data)
        for e in result.errors:
            print(f"ERROR    {e}")
        for w in result.warnings:
            print(f"WARNING  {w}")
        if result.ok:
            print("\nWhat this schedule says:")
            for line in describe(result.schedule):
                print(f"  - {line}")
            print("\n(nothing was saved)")
        return 0 if result.ok else 1
    if len(argv) == 2 and argv[1] == "show":
        try:
            entry = Store().load()
        except ScheduleCorrupt as exc:
            print(exc)
            return 2
        if entry is None:
            print("No schedule has been saved.")
            return 0
        print(f"Version {entry.version}, saved {entry.ts} ({entry.action}):")
        for line in describe(entry.schedule):
            print(f"  - {line}")
        return 0
    print("usage: python server/schedule.py check <file.json>   |   python server/schedule.py show")
    return 2


if __name__ == "__main__":
    sys.exit(_cli(sys.argv))
