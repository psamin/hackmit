"""Medication check: "did I take my pills?" without ever claiming to know.

    PAM_DOSE_CHECK=off             turn the whole feature off (server/.env or the environment)
    touch server/.dose_check_off   turn it off NOW, no restart; delete the file to turn it back on
    PAM_DOSE_DEMO=1                nudges after 20 s / 40 s instead of 10 / 20 min, for a live demo
    PAM_SCHEDULE_REMINDERS=off     stop only the SCHEDULE-driven reminders (the reminder-based flow keeps working)
    touch server/.schedule_reminders_off    the same, instantly, no restart
    PAM_VOICE_CONFIRM=off          Pam stops recording spoken yes / not-yet answers and stops asking after a handoff
    touch server/.voice_confirm_off         the same, instantly, no restart (taps on the card keep working)

--------------------------------------------------------------------------------
THE PRINCIPLE: the camera is evidence, a person's tap is the record
--------------------------------------------------------------------------------
Seeing a pill bottle move is not seeing anyone take a pill, and not seeing it move is not
seeing that nobody did (the pills may come from a pillbox; the detector may have missed it).
So the camera only ever *prompts a question*. A dose counts as taken only when the person,
or their caregiver, taps "Yes, I took them". Nothing here decides that on the camera's word.

Pam answers the question plainly. Until a dose is confirmed the answer is "no, not yet":

    recorded    "You marked your pills as taken at 8:05 AM."
    unrecorded  "No, you haven't taken your pills yet. ... Please check with Mike or your
                 pill organiser before taking any."

This was a deliberate product decision, taken over the earlier wording ("I don't have a
record that you took your pills"), which tested as vague to the person it is for. Know
what it costs: the state tracked here is a RECORD, not the inside of the bottle, so a
person who took their pills without tapping will be told "no". For a memory-impaired user
that can prompt a second dose, which is why every unconfirmed answer still ends by
referring them to their caregiver or pill organiser before taking anything. Do not remove
that sentence -- it is the only thing standing between a clear answer and a double dose.

For the same reason no prompt ever tells the user to take a pill. Prompts ask a question
and, in the same breath, say to check with the caregiver if unsure.

--------------------------------------------------------------------------------
HOW IT WORKS
--------------------------------------------------------------------------------
A medication reminder (any reminder whose text mentions pills/medication/meds/...) that has
fired opens a *dose*. Everything that then happens is appended to doses.jsonl, one event per
line, and a dose's state is rebuilt by replaying that log, so it survives a restart and can
be audited: due, evidence, asked, confirmed / not_yet, nudge, escalated, expired.

  1. A `placed` pill-bottle memory (confidence >= 0.4) after the reminder is EVIDENCE.
     Pam asks once, out loud and with a Yes / Not yet card.
  2. No answer after nudge1_s: ask again. After nudge2_s: ask the person to check with
     their caregiver, without offering a phone action. Then stop. Two prompts, no more.
  3. After expire_s the dose is closed and Pam goes quiet about it.

SCHEDULED DOSES (from the caregiver's saved schedule, see schedule.py) work the same way, with these
differences:

  - They open by themselves at each scheduled time; nobody has to set a reminder.
  - Medications due in the same minute share ONE prompt ("It's time for your Metformin and
    Lisinopril"), so a patient with several morning pills is not asked three times. One tap
    answers for the group.
  - Pam says only the medication's NAME, never an amount. The caregiver's note is shown on the
    card, exactly as written, and is never spoken or reworded.
  - Every prompt also says to check with the caregiver if unsure whether it was already taken.
  - If the server was not running at the scheduled time, Pam does not announce "it's time" late
    (the person may well have taken it). After announce_grace_s the first prompt is a QUESTION
    ("I haven't recorded your Metformin yet. Did you take it?") instead of a statement.
  - A dose is not prompted if the same medication was confirmed within its minimum gap, or its
    daily maximum is already reached (the schedule's min_gap_hours and max_per_day).
  - When the on-time window closes with no tap the dose becomes `unconfirmed`. Not "missed":
    the system only knows about taps, so all it can honestly say is that none was recorded.
  - A dose snapshots the medication's name and note when it opens, so editing the schedule later
    never rewrites what happened. Only doses due after the schedule was saved are created.

plan() is a pure function of (doses, reminders, slots, memories, now), so every branch is tested
with a fake clock. tick() runs it and performs the result; watch() is the loop that calls
tick() from the server.

A prompt is only logged once it was actually delivered to a connected page. If nobody has the
page open, the next tick tries again, so a nudge is not silently spent on an empty room.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from pathlib import Path

import adherence
import schedule as sched
from orientation import clock

HERE = Path(__file__).resolve().parent
LOG = HERE / "doses.jsonl"
KILL_FILE = HERE / ".dose_check_off"
SCHEDULE_KILL_FILE = HERE / ".schedule_reminders_off"
STREAK_KILL_FILE = HERE / ".streak_off"
VOICE_KILL_FILE = HERE / ".voice_confirm_off"
CONTACTS = HERE / "contacts.json"

MEDICATION = re.compile(r"\b(medic\w*|meds?|pills?|tablets?|capsules?|prescriptions?|doses?)\b", re.I)
PILL_OBJECT = re.compile(r"pill|medic|prescription|tablet|capsule", re.I)


# --------------------------------------------------------------------------------
# Kill switch and timing
# --------------------------------------------------------------------------------
def env_enabled() -> bool:
    """False when PAM_DOSE_CHECK says off. Read at startup to decide whether Pam is even told
    the function exists; a stopped feature must not be callable by the voice model."""
    return os.environ.get("PAM_DOSE_CHECK", "on").strip().lower() not in {"0", "off", "false", "no", "disabled"}


def enabled() -> bool:
    """The live check: the env switch AND the kill file. Every entry point asks this."""
    return env_enabled() and not KILL_FILE.exists()


def schedule_enabled() -> bool:
    """Scheduled reminders on? The whole medication check must be on, and neither schedule switch pulled."""
    off = os.environ.get("PAM_SCHEDULE_REMINDERS", "on").strip().lower() in {"0", "off", "false", "no", "disabled"}
    return enabled() and not off and not SCHEDULE_KILL_FILE.exists()


def streak_visible() -> bool:
    """May Pam show the patient their streak? On by default. `PAM_STREAK=off` or `touch server/.streak_off`
    hides it from the patient; the caregiver still sees the numbers in Pam oversight."""
    off = os.environ.get("PAM_STREAK", "on").strip().lower() in {"0", "off", "false", "no", "disabled"}
    return enabled() and not off and not STREAK_KILL_FILE.exists()


def voice_confirm_enabled() -> bool:
    """May Pam record a SPOKEN answer, and ask the question after the arm hands over the pills? On by default."""
    off = os.environ.get("PAM_VOICE_CONFIRM", "on").strip().lower() in {"0", "off", "false", "no", "disabled"}
    return enabled() and not off and not VOICE_KILL_FILE.exists()


def demo_mode() -> bool:
    return os.environ.get("PAM_DOSE_DEMO", "").strip().lower() in {"1", "true", "on", "yes"}


@dataclass(frozen=True)
class Timing:
    nudge1_s: float = 600         # ask again after this long with no answer
    nudge2_s: float = 1200        # then offer the caregiver
    min_gap_s: float = 300        # never speak two prompts about a dose closer than this
    expire_s: float = 2 * 3600    # close the dose; Pam stops raising it
    evidence_window_s: float = 3600  # bottle moves this long after the reminder count for it
    min_confidence: float = 0.4   # the same bar server/app.py uses to call a memory certain
    announce_grace_s: float = 900  # after this, a scheduled dose is asked about, not announced
    handoff_ask_s: float = 120    # after the arm hands over the pills, wait this long before asking (they need time to take them)


DEMO = Timing(nudge1_s=20, nudge2_s=40, min_gap_s=10, expire_s=300, evidence_window_s=300, announce_grace_s=60, handoff_ask_s=8)


def timing() -> Timing:
    return DEMO if demo_mode() else Timing()


def caregiver() -> dict:
    try:
        c = json.loads(CONTACTS.read_text(encoding="utf-8")).get("caregiver") or {}
    except (OSError, json.JSONDecodeError):
        c = {}
    return {"name": c.get("name") or "your caregiver", "phone": c.get("phone")}


# --------------------------------------------------------------------------------
# The log, and dose state rebuilt from it
# --------------------------------------------------------------------------------
def append(event: dict, now: float | None = None) -> None:
    event = {"ts": round(now if now is not None else time.time(), 2), **event}
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def read_events() -> list[dict]:
    if not LOG.exists():
        return []
    out = []
    for line in LOG.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a half-written last line is not an error
    return out


@dataclass
class Dose:
    id: int | str
    text: str
    due_ts: float
    evidence: dict | None = None
    asked: bool = False                  # the "I saw your bottle move, did you take it?" question was asked
    confirmed_ts: float | None = None    # tapped ON TIME (inside the window)
    late_ts: float | None = None         # tapped LATE: recorded, counts as taken, but not toward the streak
    nudges: int = 0
    escalated: bool = False
    expired: bool = False
    last_spoken: float = 0.0
    # scheduled doses only
    source: str = "reminder"             # "reminder" (a one-off reminder fired) or "schedule"
    group: str | None = None             # doses due the same minute share one prompt
    med_id: str | None = None
    name: str | None = None              # the medication's name, as the caregiver wrote it
    note: str = ""                       # the caregiver's note, shown as written, never spoken
    closes_ts: float | None = None       # the on-time window ends here
    announced: bool = False              # the "it's time for..." prompt (or its late question) was given
    unconfirmed: bool = False            # the window closed and no tap was recorded
    skipped: bool = False                # not prompted: same medication taken recently, or daily maximum reached
    via: str | None = None               # how the answer came in: "tap" or "voice"
    handoff_at: float | None = None      # the arm handed over the pills; the question is due handoff_ask_s later
    handoff_asked: bool = False          # ...and it has been asked

    @property
    def is_open(self) -> bool:
        return (self.confirmed_ts is None and self.late_ts is None and not self.expired
                and not self.unconfirmed and not self.skipped)


def _how(e: dict) -> str:
    """How an answer arrived: "voice" (said to Pam), "robot" (the arm handed the pills over) or "tap". Old lines: a tap."""
    return e.get("via") or ("robot" if e.get("by") == "robot" else "tap")


def replay(events: list[dict]) -> dict:
    """Rebuild every dose from the log. An event names one dose (`dose`) or a whole group at once (`doses`)."""
    doses: dict = {}
    for e in events:
        kind, ts = e.get("type"), float(e.get("ts", 0))
        if kind == "due":
            did = e.get("dose")
            doses[did] = Dose(id=did, text=e.get("text", ""), due_ts=float(e.get("due_ts", ts)),
                              source=e.get("source", "reminder"), group=e.get("group"), med_id=e.get("med_id"),
                              name=e.get("name"), note=e.get("note") or "", closes_ts=e.get("closes_ts"))
            continue
        ids = e.get("doses") if isinstance(e.get("doses"), list) else [e.get("dose")]
        for did in ids:
            d = doses.get(did)
            if d is None:
                continue
            if kind == "evidence":
                d.evidence = {k: e.get(k) for k in ("logged_at", "confidence", "frame", "simulated")}
            elif kind == "announced":
                d.announced, d.last_spoken = True, ts
            elif kind == "asked":
                d.asked, d.last_spoken = True, ts
            elif kind == "confirmed":
                if d.confirmed_ts is None:
                    d.confirmed_ts, d.via = ts, _how(e)
            elif kind == "confirmed_late":
                if d.late_ts is None:
                    d.late_ts, d.via = ts, _how(e)
            elif kind == "handoff":
                d.handoff_at = d.handoff_at or ts
            elif kind == "handoff_asked":
                d.handoff_asked, d.last_spoken = True, ts
            elif kind == "nudge":
                d.nudges, d.last_spoken = max(d.nudges, int(e.get("n", 1))), ts
            elif kind == "escalated":
                d.escalated, d.last_spoken = True, ts
            elif kind == "expired":
                d.expired = True
            elif kind == "unconfirmed":
                d.unconfirmed = True
            elif kind == "skipped":
                d.skipped = True
    return doses


def read_memories(path: Path) -> list[dict]:
    """The pipeline's memory.jsonl. Tolerant: the file is being appended to as we read it."""
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def is_medication(text: str) -> bool:
    return bool(MEDICATION.search(text or ""))


def is_pill_item(text: str) -> bool:
    """Is what the arm was sent for the medication? "pill bottle", "my medicine", "the tablets"."""
    return bool(is_medication(text) or PILL_OBJECT.search(text or ""))


def find_evidence(memories: list[dict], since: float, t: Timing, until: float | None = None) -> dict | None:
    """The earliest 'pill bottle was set down' memory after `since`, inside the window."""
    hi = since + t.evidence_window_s if until is None else until
    best = None
    for m in memories:
        if m.get("event") != "placed" or not PILL_OBJECT.search(str(m.get("object", ""))):
            continue
        try:
            if float(m.get("confidence") or 0) < t.min_confidence:
                continue
            logged = datetime.fromisoformat(str(m.get("logged_at"))).timestamp()  # naive = local
        except (TypeError, ValueError):
            continue
        if since <= logged <= hi and (best is None or logged < best[0]):
            frames = m.get("frames") or []
            best = (logged, {"logged_at": m.get("logged_at"), "confidence": m.get("confidence"),
                             "frame": m.get("after_frame") or (frames[-1] if frames else None), "simulated": False})
    return best[1] if best else None


# --------------------------------------------------------------------------------
# What Pam says. Every prompt asks a question and points at the caregiver; none says "take".
# --------------------------------------------------------------------------------
def _answer_actions(dose_id: int) -> dict:
    post = "/api/dose/confirm"
    return {"action": {"label": "Yes, I took them", "post": post, "body": {"dose": dose_id, "answer": "yes"}},
            "action2": {"label": "Not yet", "post": post, "body": {"dose": dose_id, "answer": "not_yet"}}}


def ask_messages(d: Dose, cg: dict, saw_bottle: bool) -> list[dict]:
    lead = "I saw your pill bottle move. " if saw_bottle else "I haven't recorded your pills yet. "
    say = f"{lead}Did you take your pills? If you're not sure, please check with {cg['name']} before taking any."
    card = {"title": "Your pills",
            "body": f"Did you take your pills? If you're not sure, check with {cg['name']} first.",
            **_answer_actions(d.id)}
    return [{"type": "speak", "text": say}, {"type": "card", "card": card}]


def escalate_messages(d: Dose, cg: dict) -> list[dict]:
    say = f"You still haven't taken your pills. Please check with {cg['name']} before taking any pills."
    card = {"title": f"Please check with {cg['name']}",
            "body": f"You haven't taken your pills yet. {cg['name']} can help you check.",
            "action2": _answer_actions(d.id)["action"]}
    return [{"type": "speak", "text": say}, {"type": "card", "card": card}]


def _list(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _names(members: list[Dose]) -> str:
    return _list([m.name or "medication" for m in members])


def _group_actions(group: str, n: int) -> dict:
    post = "/api/dose/confirm"
    yes = "Yes, I took it" if n == 1 else "Yes, I took them"
    return {"action": {"label": yes, "post": post, "body": {"group": group, "answer": "yes"}},
            "action2": {"label": "Not yet", "post": post, "body": {"group": group, "answer": "not_yet"}}}


def _card_lines(members: list[Dose]) -> str:
    """Each medication and the caregiver's note, exactly as written. Shown, never spoken."""
    return "\n".join(m.name + (f": {m.note}" if m.note else "") for m in members)


def announce_group_messages(members: list[Dose], cg: dict) -> list[dict]:
    """The scheduled reminder. It states the time and names the medication: no amounts, no instructions."""
    it = "it" if len(members) == 1 else "them"
    say = (f"It's time for your {_names(members)}. If you're not sure whether you already took {it}, "
           f"please check with {cg['name']} first.")
    card = {"title": "Time for your medication" if len(members) == 1 else "Time for your medications",
            "body": _card_lines(members), **_group_actions(members[0].group, len(members))}
    return [{"type": "speak", "text": say}, {"type": "card", "card": card}]


def ask_group_messages(members: list[Dose], cg: dict, saw_bottle: bool) -> list[dict]:
    it = "it" if len(members) == 1 else "them"
    lead = "I saw your pill bottle move. " if saw_bottle else f"I haven't recorded your {_names(members)} yet. "
    tail = f"Did you take your {_names(members)}?" if saw_bottle else f"Did you take {it}?"
    say = f"{lead}{tail} If you're not sure, please check with {cg['name']} before taking any."
    card = {"title": "Your medication", "body": _card_lines(members) + f"\n\nIf you're not sure, check with {cg['name']} first.",
            **_group_actions(members[0].group, len(members))}
    return [{"type": "speak", "text": say}, {"type": "card", "card": card}]


def escalate_group_messages(members: list[Dose], cg: dict) -> list[dict]:
    say = f"You still haven't taken your {_names(members)}. Please check with {cg['name']} before taking any pills."
    card = {"title": f"Please check with {cg['name']}",
            "body": f"You haven't taken your {_names(members)} yet. {cg['name']} can help you check.",
            "action2": _group_actions(members[0].group, len(members))["action"]}
    return [{"type": "speak", "text": say}, {"type": "card", "card": card}]


def handoff_messages(members: list[Dose], cg: dict) -> list[dict]:
    """The question after the arm hands over the pills. Neutral on purpose: a person with memory loss tends to
    agree with whatever they are asked, so it offers both answers, and it never says to take anything."""
    what = _names(members) if members[0].source == "schedule" else "pills"
    say = f"Have you taken your {what} yet, or not yet? If you're not sure, please check with {cg['name']} first."
    body = (_card_lines(members) if members[0].source == "schedule" else "Your pills") + f"\n\nIf you're not sure, check with {cg['name']} first."
    actions = _group_actions(members[0].group, len(members)) if members[0].source == "schedule" else _answer_actions(members[0].id)
    return [{"type": "speak", "text": say}, {"type": "card", "card": {"title": "Your pills", "body": body, **actions}}]


# --------------------------------------------------------------------------------
# The decision: pure, so the whole ladder is tested with a fake clock
# --------------------------------------------------------------------------------
def slot_key(s) -> str:
    """One scheduled dose: this medication, on this date, at this time."""
    return f"{s.med_id}|{s.due:%Y-%m-%d}|{s.due:%H:%M}"


def slot_group(s) -> str:
    """Doses due the same minute share a prompt."""
    return f"{s.due:%Y-%m-%dT%H:%M}"


def _due_event(d: Dose) -> dict:
    return {"type": "due", "dose": d.id, "text": d.name or "", "due_ts": d.due_ts, "source": "schedule", "group": d.group,
            "med_id": d.med_id, "name": d.name, "note": d.note, "closes_ts": d.closes_ts}


def _guard_reason(d: Dose, doses: dict, med, now: float) -> str | None:
    """Why NOT to prompt for this dose: the same medication was taken recently, or its daily maximum is reached."""
    if med is None:
        return None
    # A late tap counts as taken here: it is exactly the case where someone already took it.
    others = [(o, o.confirmed_ts or o.late_ts) for o in doses.values()
              if o is not d and o.med_id == d.med_id and (o.confirmed_ts or o.late_ts)]
    if med.min_gap_hours and any(0 <= now - taken < med.min_gap_hours * 3600 for _, taken in others):
        return "taken recently"
    if med.max_per_day and sum(_same_day(taken, d.due_ts) for _, taken in others) >= med.max_per_day:
        return "daily maximum reached"
    return None


def _plan_scheduled(doses: dict, slots: list, meds: dict, memories: list[dict], now: float, t: Timing, cg: dict,
                    out: list) -> None:
    # 1. open a dose for every slot that is due and not yet in the log
    for s in slots:
        key = slot_key(s)
        due_ts = s.due.timestamp()
        if key in doses or now < due_ts:
            continue
        d = Dose(id=key, text=s.name, due_ts=due_ts, source="schedule", group=slot_group(s), med_id=s.med_id, name=s.name,
                 note=s.note, closes_ts=s.closes.timestamp())
        doses[key] = d
        out.append((_due_event(d), []))
        if now > d.closes_ts:      # the whole window went by with nothing running to ask; nothing to prompt now
            d.unconfirmed = True
            out.append(({"type": "unconfirmed", "dose": key, "reason": "not_running"}, []))

    # 2. per group: close what is out of time, hold back what should not be prompted, then prompt once
    groups: dict[str, list[Dose]] = {}
    for d in doses.values():
        if d.source == "schedule" and d.is_open:
            groups.setdefault(d.group, []).append(d)
    for gid in sorted(groups, key=lambda g: min(m.due_ts for m in groups[g])):
        members = []
        for m in groups[gid]:
            if m.closes_ts is not None and now > m.closes_ts:
                m.unconfirmed = True
                out.append(({"type": "unconfirmed", "dose": m.id, "reason": "no_tap"}, []))
                continue
            if not m.announced:
                why = _guard_reason(m, doses, meds.get(m.med_id), now)
                if why:
                    m.skipped = True
                    out.append(({"type": "skipped", "dose": m.id, "reason": why}, []))
                    continue
            members.append(m)
        if not members:
            continue
        ids, due = [m.id for m in members], min(m.due_ts for m in members)
        age = now - due

        if any(m.evidence is None for m in members):
            found = find_evidence(memories, due, t, until=max(m.closes_ts or due + t.evidence_window_s for m in members))
            if found:
                for m in members:
                    m.evidence = found
                out.append(({"type": "evidence", "doses": ids, **found}, []))

        if not all(m.announced for m in members):
            for m in members:
                m.announced = True
            late = age > t.announce_grace_s     # too late to say "it's time": the person may already have taken it
            msgs = ask_group_messages(members, cg, saw_bottle=False) if late else announce_group_messages(members, cg)
            out.append(({"type": "announced", "doses": ids, "late": late}, msgs))
            continue
        last = max(m.last_spoken for m in members)
        if now - last < t.min_gap_s:
            continue
        if any(m.evidence for m in members) and not all(m.asked for m in members):
            for m in members:
                m.asked = True
            out.append(({"type": "asked", "doses": ids}, ask_group_messages(members, cg, saw_bottle=True)))
            continue
        nudges = min(m.nudges for m in members)
        if nudges == 0 and age >= t.nudge1_s:
            out.append(({"type": "nudge", "doses": ids, "n": 1},
                        ask_group_messages(members, cg, saw_bottle=any(m.evidence for m in members))))
        elif nudges == 1 and not any(m.escalated for m in members) and age >= t.nudge2_s:
            out.append(({"type": "escalated", "doses": ids}, escalate_group_messages(members, cg)))


def _on_time(d: Dose, now: float) -> bool:
    """A tap now would be recorded on time."""
    return d.is_open and (d.closes_ts is None or now <= d.closes_ts)


def _late_ok(d: Dose, now: float) -> bool:
    """A tap now would be recorded as late: after the window, inside the grace period, and nothing recorded yet."""
    return (d.confirmed_ts is None and d.late_ts is None and not d.skipped and d.closes_ts is not None
            and d.closes_ts < now <= d.closes_ts + adherence.LATE_TAP_GRACE_S)


def _answerable(d: Dose, now: float) -> bool:
    return _on_time(d, now) or _late_ok(d, now)


def _prompted(d: Dose) -> bool:
    """Pam has actually asked about this dose. A spoken "yes" is only taken as an answer to a question she asked."""
    return d.announced or d.asked or d.nudges > 0 or d.escalated or d.handoff_asked


def _group_key(d: Dose):
    return d.group if d.source == "schedule" and d.group is not None else d.id


def _handoff_group(doses: dict, now: float) -> list[Dose]:
    """The dose the pills were fetched for: the newest one that can still be answered. Empty if there is none."""
    live = [d for d in doses.values() if _answerable(d, now) and d.due_ts <= now]
    if not live:
        return []
    key = _group_key(max(live, key=lambda d: (d.due_ts, str(d.id))))
    return sorted((d for d in live if _group_key(d) == key), key=lambda d: (d.due_ts, d.name or ""))


def _plan_handoff(doses: dict, now: float, t: Timing, cg: dict, out: list) -> None:
    """Some time after the arm handed over the pills, ask once whether they were taken."""
    spoke = {i for ev, msgs in out if msgs for i in (ev.get("doses") if isinstance(ev.get("doses"), list) else [ev.get("dose")])}
    due: dict = {}
    for d in doses.values():
        if d.handoff_at is not None and not d.handoff_asked and d.id not in spoke and _answerable(d, now) \
                and now - d.handoff_at >= t.handoff_ask_s:
            due.setdefault(_group_key(d), []).append(d)
    for members in sorted(due.values(), key=lambda g: min(m.due_ts for m in g)):
        if now - max(m.last_spoken for m in members) < t.min_gap_s:
            continue                       # something was just said about this dose; ask on a later tick
        members.sort(key=lambda d: (d.due_ts, d.name or ""))
        for m in members:
            m.handoff_asked = True
        key = "doses" if len(members) > 1 or members[0].source == "schedule" else "dose"
        ids = [m.id for m in members]
        out.append(({"type": "handoff_asked", key: ids if key == "doses" else ids[0]}, handoff_messages(members, cg)))


def plan(doses: dict, reminders: list[dict], memories: list[dict], now: float, t: Timing, cg: dict | None = None,
         slots: list | None = None, meds: dict | None = None) -> list[tuple[dict, list[dict]]]:
    """(event to log, messages to push) pairs. Does not touch the log or the network."""
    cg = cg or caregiver()
    doses = {k: replace(v) for k, v in doses.items()}
    out: list[tuple[dict, list[dict]]] = []

    for r in reminders:
        rid = r.get("id")
        if rid is None or rid in doses or not r.get("fired") or not is_medication(r.get("text", "")):
            continue
        due = float(r["due_ts"])
        if now - due > t.expire_s:
            continue  # an old reminder that fired long ago is not a dose we are waiting on
        doses[rid] = Dose(id=rid, text=r["text"], due_ts=due)
        out.append(({"type": "due", "dose": rid, "text": r["text"], "due_ts": due}, []))

    for d in sorted(doses.values(), key=lambda x: x.due_ts):
        if not d.is_open or d.source != "reminder":
            continue
        age = now - d.due_ts
        if age > t.expire_s:
            d.expired = True
            out.append(({"type": "expired", "dose": d.id}, []))
            continue
        if d.evidence is None:
            found = find_evidence(memories, d.due_ts, t)
            if found:
                d.evidence = found
                out.append(({"type": "evidence", "dose": d.id, **found}, []))
        if d.evidence and not d.asked:
            d.asked = True
            out.append(({"type": "asked", "dose": d.id}, ask_messages(d, cg, saw_bottle=True)))
            continue
        if now - d.last_spoken < t.min_gap_s:
            continue
        if d.nudges == 0 and age >= t.nudge1_s:
            out.append(({"type": "nudge", "dose": d.id, "n": 1}, ask_messages(d, cg, saw_bottle=bool(d.evidence))))
        elif d.nudges == 1 and not d.escalated and age >= t.nudge2_s:
            out.append(({"type": "escalated", "dose": d.id}, escalate_messages(d, cg)))
    if slots:
        _plan_scheduled(doses, slots, meds or {}, memories, now, t, cg, out)
    if voice_confirm_enabled():
        _plan_handoff(doses, now, t, cg, out)
    return out


def schedule_store() -> "sched.Store":
    """A seam: tests point this at a temporary file."""
    return sched.Store()


_HEALTH = {"state": "none", "version": None, "at": None}
_last_report = [float("-inf")]


def schedule_health() -> dict:
    """For Pam oversight: the schedule's state, its version, and whether the scheduler has ticked lately.

    state: "none" | "active" | "damaged" | "error" | "off". "damaged" matters: an unreadable schedule
    file must never look like "no reminders, all fine". `running` is False if nothing has ticked in 30 s
    (say, the server was started without the watcher), so the state shown is not trusted blindly."""
    at = _HEALTH["at"]
    return {"state": _HEALTH["state"], "version": _HEALTH["version"], "running": at is not None and time.time() - at < 30}


def _report(message: str) -> None:
    if time.monotonic() - _last_report[0] >= 600:           # say it, but not every two seconds
        _last_report[0] = time.monotonic()
        print(f"{datetime.now():%H:%M:%S} [DOSE  ] {message}", flush=True)


def _schedule_inputs(schedule_fn, now: float) -> tuple[list, dict]:
    """(slots that may be due now, medications by id). Never raises: a bad schedule is reported, not fatal."""
    if not schedule_enabled():
        _HEALTH.update(state="off", version=None)
        return [], {}
    try:
        entry = schedule_fn()
    except sched.ScheduleCorrupt as exc:
        _HEALTH.update(state="damaged", version=None)
        _report(f"the medication schedule file is damaged, so NO scheduled reminders are being sent: {exc}")
        return [], {}
    except Exception as exc:  # noqa: BLE001
        _HEALTH.update(state="error", version=None)
        _report(f"couldn't read the medication schedule: {type(exc).__name__}: {exc}")
        return [], {}
    if entry is None:
        _HEALTH.update(state="none", version=None)
        return [], {}
    _HEALTH.update(state="active", version=entry.version)
    saved = datetime.fromisoformat(entry.ts).timestamp()
    today = datetime.fromtimestamp(now).date()
    # yesterday too: a window can run past midnight. And nothing whose window closed before this
    # schedule existed: that would invent an "unconfirmed" dose for a time nobody had asked about yet.
    slots = [x for day in (today - timedelta(days=1), today) for x in sched.slots_for_day(entry.schedule, day)
             if x.closes.timestamp() >= saved]
    return slots, {m.id: m for m in entry.schedule.medications if m.id}


def tick(reminders: list[dict], memories_path: Path, push, now: float | None = None, schedule_fn=None) -> int:
    """One pass: plan, deliver, log. `push(msg)` returns how many pages received it.

    An event that carries a prompt is logged only if the prompt was delivered somewhere, so
    a nudge is not used up while no page is open. Returns the number of events logged."""
    if not enabled():
        return 0
    now = time.time() if now is None else now
    logged = 0
    _HEALTH["at"] = time.time()
    slots, meds = _schedule_inputs(schedule_fn, now) if schedule_fn is not None else ([], {})
    for event, msgs in plan(replay(read_events()), reminders, read_memories(memories_path), now, timing(),
                            slots=slots, meds=meds):
        if msgs and not any([push(m) for m in msgs]):  # push every message, then ask if any arrived
            continue  # nobody was there to hear it; try again next tick
        append(event, now)
        logged += 1
    return logged


async def watch(reminders_fn, memories_path: Path, push, interval: float = 2.0, schedule_fn=None) -> None:
    """The server's background loop. It must never die: a bad tick is skipped, not fatal."""
    while True:
        try:
            tick(reminders_fn(), memories_path, push, schedule_fn=schedule_fn)
        except Exception as exc:  # noqa: BLE001 - a medication watcher that crashes silently is worse than a noisy one
            print(f"{datetime.now():%H:%M:%S} [DOSE  ] tick failed: {type(exc).__name__}: {exc}", flush=True)
        await asyncio.sleep(interval)


# --------------------------------------------------------------------------------
# Answers to the page and to Pam
# --------------------------------------------------------------------------------
OFF_ANSWER = "I can't check that right now. Please ask {name} about your pills."


def _off(cg: dict) -> dict:
    card = {"title": "Your pills", "body": OFF_ANSWER.format(name=cg["name"])}
    return {"say": OFF_ANSWER.format(name=cg["name"]), "card": card, "enabled": False}


def _same_day(a: float, b: float) -> bool:
    return datetime.fromtimestamp(a).date() == datetime.fromtimestamp(b).date()


def _times(stamps: list[float]) -> str:
    return _list([clock(datetime.fromtimestamp(ts), with_period=True) for ts in sorted(stamps)])


def _next_line(upcoming: list | None) -> str:
    """" Your next one is your Vitamin D at 12:00 PM." from the schedule, or nothing."""
    if not upcoming:
        return ""
    name, due_ts = upcoming[0]
    return f" Your next one is your {name} at {_times([due_ts])}."


def _scheduled_status(today: list[Dose], scheduled: list[Dose], cg: dict, upcoming: list) -> dict:
    """Per medication: what was marked, what has no record, and what is next. Never "you didn't"."""
    by_med: dict[str, list[Dose]] = {}
    for d in scheduled:
        by_med.setdefault(d.name or "medication", []).append(d)
    # Medications with the same times share one sentence: "your Lisinopril and Metformin at 8:05 AM",
    # not one near-identical sentence each. Pam says this aloud, so it should not sound like a list.
    taken_groups: dict[tuple, list[str]] = {}
    waiting_groups: dict[tuple, list[str]] = {}
    for name, ds in sorted(by_med.items(), key=lambda kv: min(d.due_ts for d in kv[1])):
        taken = tuple(sorted(d.confirmed_ts or d.late_ts for d in ds if d.confirmed_ts or d.late_ts))
        waiting = tuple(sorted(d.due_ts for d in ds if not (d.confirmed_ts or d.late_ts)))
        if taken:
            taken_groups.setdefault(taken, []).append(name)
        if waiting:
            waiting_groups.setdefault(waiting, []).append(name)
    parts = [f"You marked your {_list(names)} as taken at {_times(list(stamps))}." for stamps, names in taken_groups.items()]
    parts += [f"You haven't taken your {_list(names)} from {_times(list(stamps))} yet." for stamps, names in waiting_groups.items()]
    any_taken, unrecorded = bool(taken_groups), bool(waiting_groups)
    extra = [d.confirmed_ts for d in today if d.source == "reminder" and d.confirmed_ts]
    if extra:
        any_taken = True
        parts.append(f"You also marked your pills as taken at {_times(extra)}.")
    if unrecorded:
        parts.append(f"I can't see inside the bottle. Please check with {cg['name']} or your pill organiser before taking any pills.")
    say = " ".join(parts) + _next_line(upcoming)
    return {"say": say, "card": {"title": "Your medication", "body": say}, "recorded": any_taken and not unrecorded}


def status(doses: dict, now: float, cg: dict | None = None, upcoming: list | None = None) -> dict:
    """The answer to "did I take my pills?". Only a person's tap counts as taken."""
    cg = cg or caregiver()
    today = [d for d in doses.values() if _same_day(d.due_ts, now)]
    scheduled = [d for d in today if d.source == "schedule" and not d.skipped]
    if scheduled:
        return _scheduled_status(today, scheduled, cg, upcoming or [])
    taken = sorted(d.confirmed_ts for d in today if d.confirmed_ts)
    if taken:
        times = [clock(datetime.fromtimestamp(ts), with_period=True) for ts in taken]
        when = times[0] if len(times) == 1 else ", ".join(times[:-1]) + " and " + times[-1]
        say = f"You marked your pills as taken at {when}." + _next_line(upcoming)
        return {"say": say, "card": {"title": "Your pills", "body": say}, "recorded": True}

    seen = sorted((d.evidence for d in today if d.evidence and d.evidence.get("logged_at")),
                  key=lambda e: str(e["logged_at"]))
    if seen:
        at = clock(datetime.fromisoformat(str(seen[0]["logged_at"])), with_period=True)
        middle = f"I did see your pill bottle move at {at}, but I can't tell whether you took any."
    else:
        middle = "I haven't seen your pill bottle move, but I can't see everything."
    say = (f"No, you haven't taken your pills yet. {middle} I can't see inside the bottle. "
           f"Please check with {cg['name']} or your pill organiser before taking any pills." + _next_line(upcoming))
    card = {"title": "Your pills", "body": say}
    return {"say": say, "card": card, "recorded": False}


def status_response(now: float | None = None) -> dict:
    cg = caregiver()
    if not enabled():
        return _off(cg)
    now = time.time() if now is None else now
    return status(replay(read_events()), now, cg, upcoming=_upcoming(now))


def _upcoming(now: float) -> list:
    """(name, due) for today's scheduled doses that are not due yet, soonest first."""
    if not schedule_enabled():
        return []
    try:
        entry = schedule_store().load()
    except Exception:  # noqa: BLE001 - the answer must still be given without the "next one" line
        return []
    if entry is None:
        return []
    slots = sched.slots_for_day(entry.schedule, datetime.fromtimestamp(now).date())
    return [(x.name, x.due.timestamp()) for x in slots if x.due.timestamp() > now]


def _confirm_group(doses: dict, dose_id, group, answer: str, now: float, cg: dict, via: str = "tap") -> dict:
    members = [d for d in doses.values() if d.source == "schedule" and (d.group == group if group is not None else d.id == dose_id)]
    if not members:
        return {"say": "I don't have a pill question waiting for an answer.", "ok": False}
    if answer == "not_yet":
        append({"type": "not_yet", "doses": [d.id for d in members], "group": members[0].group, "via": via}, now)
        return {"say": f"Okay. If you're not sure, please check with {cg['name']} before taking any.", "ok": True}
    if answer != "yes":
        return {"say": "I didn't understand that answer.", "ok": False}
    on_time = [d for d in members if _on_time(d, now)]
    # A tap after the window is still a tap: recorded, and it counts as taken for safety, but it neither adds to
    # the streak nor erases it. Only for a while: a tap days later would mostly be confusion.
    late = [d for d in members if _late_ok(d, now)]
    if on_time:
        append({"type": "confirmed", "doses": [d.id for d in on_time], "group": members[0].group, "via": via}, now)
    if late:
        append({"type": "confirmed_late", "doses": [d.id for d in late], "group": members[0].group, "via": via}, now)
    done = [d for d in members if d.confirmed_ts or d.late_ts] + [d for d in on_time + late if not (d.confirmed_ts or d.late_ts)]
    if not done:            # every dose in the group is out of time, or was not prompted
        return {"say": f"That reminder has passed. Please check with {cg['name']} about your {_names(members)}.", "ok": False}
    done.sort(key=lambda d: (d.due_ts, d.name or ""))
    at = min(d.confirmed_ts or d.late_ts or now for d in done)
    say = f"Thank you. I've noted that you took your {_names(done)} at {clock(datetime.fromtimestamp(at), with_period=True)}."
    if streak_visible():
        summary = adherence_summary(now)
        extra = adherence.streak_line(summary) or (adherence.still_line(summary) if late or any(d.late_ts for d in done) else "")
        say = f"{say} {extra}" if extra else say
    return {"say": say, "ok": True, "recorded": True}


def confirm(dose_id, answer: str, now: float | None = None, *, group: str | None = None, via: str = "tap") -> dict:
    """The tap. 'yes' records the dose (once); 'not_yet' records nothing but the question.

    A scheduled reminder is answered for its whole group at once (`group`)."""
    cg = caregiver()
    if not enabled():
        return _off(cg)
    now = time.time() if now is None else now
    if not isinstance(dose_id, (int, str, type(None))) or not isinstance(group, (str, type(None))):
        return {"say": "I don't have a pill question waiting for an answer.", "ok": False}
    doses = replay(read_events())
    if group is not None or (dose_id in doses and doses[dose_id].source == "schedule"):
        return _confirm_group(doses, dose_id, group, answer, now, cg, via)
    d = doses.get(dose_id)
    if d is None:
        return {"say": "I don't have a pill question waiting for an answer.", "ok": False}
    if answer == "yes":
        if d.confirmed_ts is None:
            append({"type": "confirmed", "dose": d.id, "via": via}, now)
        at = clock(datetime.fromtimestamp(d.confirmed_ts or now), with_period=True)
        return {"say": f"Thank you. I've noted that you took your pills at {at}.", "ok": True, "recorded": True}
    if answer == "not_yet":
        append({"type": "not_yet", "dose": d.id, "via": via}, now)
        return {"say": f"Okay. If you're not sure, please check with {cg['name']} before taking any.", "ok": True}
    return {"say": "I didn't understand that answer.", "ok": False}


def confirm_by_robot(now: float | None = None, what: str = "pills") -> dict:
    """The arm handed the medication over: record the dose as taken.

    This is a PRODUCT DECISION that overrides the principle at the top of this file.
    Everywhere else, the camera only ever produces `evidence` and a person's tap is the
    only thing that records a dose -- because a bottle moving is not a pill being
    swallowed. A robot hand-over is the same class of signal: it proves the bottle
    travelled, not that anything was taken. It is recorded as `confirmed` anyway,
    because the arm delivering the medication is the moment this product treats as the
    dose being taken.

    What that costs: if the arm fetches the bottle and the person does not take one,
    Pam will say they did. The event is written with by="robot" so the log can always
    tell an arm hand-over apart from a human tap, which is the difference an audit or a
    caregiver would care about.

    Confirms every dose open today. If none is open -- nothing was scheduled, or it was
    already answered -- one is opened and immediately confirmed, so "did I take my
    pills?" answers yes rather than falling back to "no, not yet".
    """
    cg = caregiver()
    if not enabled():
        return _off(cg)
    now = time.time() if now is None else now
    doses = replay(read_events())
    open_today = [d for d in doses.values() if _same_day(d.due_ts, now) and d.is_open]
    if open_today:
        for d in open_today:
            append({"type": "confirmed", "dose": d.id, "by": "robot"}, now)
        names = _list([d.name or "pills" for d in open_today])
    else:
        did = f"robot-{int(now)}"
        append({"type": "due", "dose": did, "text": what, "due_ts": now, "source": "robot"}, now)
        append({"type": "confirmed", "dose": did, "by": "robot"}, now)
        names = what
    at = clock(datetime.fromtimestamp(now), with_period=True)
    return {"ok": True, "recorded": True, "doses": len(open_today) or 1,
            "say": f"I've noted that you took your {names} at {at}."}


def simulate_evidence(now: float | None = None) -> dict:
    """DEMO ONLY (PAM_DOSE_DEMO=1): stand in for the camera seeing the bottle move.
    Recorded as simulated, so the log never passes it off as an observation."""
    if not enabled():
        return _off(caregiver())
    if not demo_mode():
        return {"say": "Simulation is only available in demo mode.", "ok": False}
    now = time.time() if now is None else now
    open_doses = [d for d in replay(read_events()).values() if d.is_open and d.evidence is None]
    if not open_doses:
        return {"say": "No pill reminder is waiting for the bottle to move.", "ok": False}
    d = max(open_doses, key=lambda x: x.due_ts)
    same = [x for x in open_doses if d.group is not None and x.group == d.group] or [d]     # the whole group, if scheduled
    append({"type": "evidence", "doses": [x.id for x in same],
            "logged_at": datetime.fromtimestamp(now).isoformat(timespec="seconds"),
            "confidence": 1.0, "frame": None, "simulated": True}, now)
    return {"say": "Simulated: the pill bottle was moved.", "ok": True}


RECENT_QUESTION_S = 600     # a bare "yes" answers the question Pam asked in the last ten minutes


def record_handoff(item: str = "", now: float | None = None) -> dict:
    """The arm has handed over `item`. If that was the pills and a dose can still be answered, Pam asks about it
    handoff_ask_s later (see _plan_handoff). Nothing is spoken now. If no dose is waiting the handoff is only logged:
    a bottle fetched outside a dose time is not Pam's cue to ask a question that has no answer to record."""
    if not voice_confirm_enabled():
        return {"ok": False, "logged": False, "reason": "off"}
    item = str(item or "")[:80]
    if not is_pill_item(item):
        return {"ok": True, "logged": False, "reason": "not_medication"}
    now = time.time() if now is None else now
    members = _handoff_group(replay(read_events()), now)
    if not members:
        append({"type": "handoff", "doses": [], "item": item, "reason": "no_open_dose"}, now)
        return {"ok": True, "logged": True, "asks": [], "reason": "no_open_dose"}
    if any(m.handoff_at is not None for m in members):
        return {"ok": True, "logged": False, "reason": "already_recorded"}     # one question per dose, however often it is fetched
    append({"type": "handoff", "doses": [m.id for m in members], "group": members[0].group, "item": item}, now)
    return {"ok": True, "logged": True, "asks": [m.name or "pills" for m in members], "reason": "will_ask"}


def _clarify(groups: list[list[Dose]], cg: dict) -> dict:
    parts = [f"your {_names(g) if g[0].source == 'schedule' else 'pills'} at {_times([min(m.due_ts for m in g)])}" for g in groups]
    say = f"Which one do you mean: {' or '.join(parts)}?"
    return {"say": say, "ok": False, "ambiguous": True}


def voice_confirm(answer: str, medication: str | None = None, now: float | None = None) -> dict:
    """The patient SAID it, to Pam: the same as tapping "Yes, I took them" or "Not yet", recorded as via "voice".

    Which dose is decided here, not by the voice model. Only doses Pam has asked about, and that can still be
    answered, are candidates. One candidate: that one. Several: the medication they named, else the one Pam asked
    about in the last ten minutes if that is only one, else Pam asks which. A wrong guess would record a dose
    nobody took, so when unsure she asks."""
    cg = caregiver()
    if not enabled():
        return _off(cg)
    if not voice_confirm_enabled():
        return {"say": f"I can't note that down right now. Please let {cg['name']} know.", "ok": False, "enabled": False}
    if answer not in {"yes", "not_yet"}:
        return {"say": "I didn't understand that answer.", "ok": False}
    now = time.time() if now is None else now
    everything = replay(read_events())
    groups: dict = {}
    for d in everything.values():
        if _prompted(d) and _answerable(d, now):
            groups.setdefault(_group_key(d), []).append(d)
    cands = list(groups.values())
    if medication:
        want = str(medication).strip().casefold()
        named = [g for g in cands if any(want and want in (m.name or "").casefold() for m in g)]
        cands = named or (cands if want in {"pills", "medication", "medicine", "meds", "pill"} else [])
    if not cands:
        if answer == "yes":         # they may be saying it a second time: tell them what is already recorded, do not record again
            have = status(everything, now, cg, upcoming=_upcoming(now))
            if have["recorded"]:
                return {"say": have["say"], "ok": True, "recorded": True, "already": True}
        return {"say": f"I don't have a pill question waiting for an answer. If you're not sure, please check with {cg['name']}.", "ok": False}
    if len(cands) > 1:
        recent = [g for g in cands if now - max(m.last_spoken for m in g) <= RECENT_QUESTION_S]
        if len(recent) != 1:
            return _clarify(sorted(cands, key=lambda g: min(m.due_ts for m in g)), cg)
        cands = recent
    members = cands[0]
    if members[0].source == "schedule":
        return confirm(None, answer, now, group=members[0].group, via="voice")
    return confirm(members[0].id, answer, now, via="voice")


def adherence_summary(now: float | None = None) -> "adherence.Summary":
    """The streak as of `now`, counting this evening's doses (not yet opened) as still to come."""
    now = time.time() if now is None else now
    return adherence.summarize(replay(read_events()).values(), now, upcoming=_upcoming(now))


def streak_response(now: float | None = None) -> dict:
    """Pam's answer to "how am I doing?". Off when the medication check is off, or the caregiver hid the streak."""
    cg = caregiver()
    if not enabled():
        return _off(cg)
    if not streak_visible():
        say = "I can't share that right now."
        return {"say": say, "card": {"title": "Your streak", "body": say}, "hidden": True}
    summary = adherence_summary(now)
    say = adherence.how_am_i_doing(summary)
    return {"say": say, "card": {"title": "Your streak", "body": say}, "streak": summary.streak, "best": summary.best}


# What Pam is told, when the feature is on. See the module docstring for why.
FUNCTION_DESCRIPTION = ("Answer whether the user has taken their pills or medication. Use whenever they ask if they "
                        "took, or already took, their pills, medicine or medication. Read the whole result aloud.")
PROMPT_RULE = """
- Pills: you cannot see inside the pill bottle and you do not know whether the user has taken their medication.
  Never say that they have, or have not, taken it. When they ask, call check_pills_taken and read its answer
  aloud in full, without softening it or adding to it. Never tell them to take, skip or double a dose; when in
  doubt, send them to their caregiver.
"""

VOICE_FUNCTION_DESCRIPTION = ("Record the user's spoken answer to whether they took their pills or medication. Call it only when "
                              "they clearly say they took them (yes) or clearly say they have not (not_yet).")
VOICE_RULE = """
- Pill answers: when you have asked whether they took their pills, listen for a clear answer. Only when they
  clearly say they took them, call confirm_pills with answer yes; only when they clearly say they have not, or
  not yet, call it with not_yet. Never assume yes. If the reply is unclear ("okay", "hmm", "I think so", silence),
  do not call it: ask once more, "Have you taken them, or not yet?", and if it is still unclear tell them to check
  with their caregiver. A yes to any other question is not an answer about pills. Read the result aloud.
"""

STREAK_FUNCTION_DESCRIPTION = ("Tell the user how they are doing with their medication: how many days in a row they have marked "
                               "all of it on time. Use when they ask about their streak, their progress, or how they are doing. "
                               "Read the whole result aloud.")
STREAK_RULE = """
- Streak: when they ask how they are doing, or about their streak, call check_streak and read the answer aloud as
  written. Never say they failed, missed, forgot, lost or broke anything, never mention late or unrecorded doses, and
  never compare them with anyone else. If there is no streak yet, be encouraging and say nothing about why.
"""
