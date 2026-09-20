"""Medication check: "did I take my pills?" without ever claiming to know.

    PAM_DOSE_CHECK=off             turn the whole feature off (server/.env or the environment)
    touch server/.dose_check_off   turn it off NOW, no restart; delete the file to turn it back on
    PAM_DOSE_DEMO=1                nudges after 20 s / 40 s instead of 10 / 20 min, for a live demo

--------------------------------------------------------------------------------
THE PRINCIPLE: the camera is evidence, a person's tap is the record
--------------------------------------------------------------------------------
Seeing a pill bottle move is not seeing anyone take a pill, and not seeing it move is not
seeing that nobody did (the pills may come from a pillbox; the detector may have missed it).
So the camera only ever *prompts a question*. A dose counts as taken only when the person,
or their caregiver, taps "Yes, I took them". Nothing here decides that on the camera's word.

The dangerous case is a memory-impaired user asking "did I take my pills?" and being told
"no". That answer can cause a second dose. So Pam never says the user has NOT taken them:

    recorded    "You marked your pills as taken at 8:05 AM."
    unrecorded  "I don't have a record that you took your pills. I can't see inside the bottle.
                 Please check with Mike or your pill organiser before taking any pills."

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
  2. No answer after nudge1_s: ask again. After nudge2_s: offer to call the caregiver (a
     button; the tap is the consent, nothing dials itself). Then stop. Two prompts, no more.
  3. After expire_s the dose is closed and Pam goes quiet about it.

plan() is a pure function of (doses, reminders, memories, now), so every branch is tested
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
from datetime import datetime
from pathlib import Path

from orientation import clock

HERE = Path(__file__).resolve().parent
LOG = HERE / "doses.jsonl"
KILL_FILE = HERE / ".dose_check_off"
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


DEMO = Timing(nudge1_s=20, nudge2_s=40, min_gap_s=10, expire_s=300, evidence_window_s=300)


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
    id: int
    text: str
    due_ts: float
    evidence: dict | None = None
    asked: bool = False
    confirmed_ts: float | None = None
    nudges: int = 0
    escalated: bool = False
    expired: bool = False
    last_spoken: float = 0.0

    @property
    def is_open(self) -> bool:
        return self.confirmed_ts is None and not self.expired


def replay(events: list[dict]) -> dict[int, Dose]:
    doses: dict[int, Dose] = {}
    for e in events:
        did, kind, ts = e.get("dose"), e.get("type"), float(e.get("ts", 0))
        if kind == "due":
            doses[did] = Dose(id=did, text=e.get("text", ""), due_ts=float(e.get("due_ts", ts)))
            continue
        d = doses.get(did)
        if d is None:
            continue
        if kind == "evidence":
            d.evidence = {k: e.get(k) for k in ("logged_at", "confidence", "frame", "simulated")}
        elif kind == "asked":
            d.asked, d.last_spoken = True, ts
        elif kind == "confirmed":
            d.confirmed_ts = ts
        elif kind == "nudge":
            d.nudges, d.last_spoken = max(d.nudges, int(e.get("n", 1))), ts
        elif kind == "escalated":
            d.escalated, d.last_spoken = True, ts
        elif kind == "expired":
            d.expired = True
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


def find_evidence(memories: list[dict], since: float, t: Timing) -> dict | None:
    """The earliest 'pill bottle was set down' memory after `since`, inside the window."""
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
        if since <= logged <= since + t.evidence_window_s and (best is None or logged < best[0]):
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
    say = (f"I still don't have a record of your pills. Please check with {cg['name']}. "
           f"I can call {cg['name']} for you. Tap the button on your screen.")
    card = {"title": f"Call {cg['name']}?",
            "body": f"I don't have a record that you took your pills. {cg['name']} can help you check.",
            "action2": _answer_actions(d.id)["action"]}
    if cg.get("phone"):
        card["action"] = {"label": f"Call {cg['name']}", "href": f"tel:{cg['phone']}"}
    return [{"type": "speak", "text": say}, {"type": "card", "card": card}]


# --------------------------------------------------------------------------------
# The decision: pure, so the whole ladder is tested with a fake clock
# --------------------------------------------------------------------------------
def plan(doses: dict[int, Dose], reminders: list[dict], memories: list[dict], now: float,
         t: Timing, cg: dict | None = None) -> list[tuple[dict, list[dict]]]:
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
        if not d.is_open:
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
    return out


def tick(reminders: list[dict], memories_path: Path, push, now: float | None = None) -> int:
    """One pass: plan, deliver, log. `push(msg)` returns how many pages received it.

    An event that carries a prompt is logged only if the prompt was delivered somewhere, so
    a nudge is not used up while no page is open. Returns the number of events logged."""
    if not enabled():
        return 0
    now = time.time() if now is None else now
    logged = 0
    for event, msgs in plan(replay(read_events()), reminders, read_memories(memories_path), now, timing()):
        if msgs and not any([push(m) for m in msgs]):  # push every message, then ask if any arrived
            continue  # nobody was there to hear it; try again next tick
        append(event, now)
        logged += 1
    return logged


async def watch(reminders_fn, memories_path: Path, push, interval: float = 2.0) -> None:
    """The server's background loop. It must never die: a bad tick is skipped, not fatal."""
    while True:
        try:
            tick(reminders_fn(), memories_path, push)
        except Exception as exc:  # noqa: BLE001 - a medication watcher that crashes silently is worse than a noisy one
            print(f"{datetime.now():%H:%M:%S} [DOSE  ] tick failed: {type(exc).__name__}: {exc}", flush=True)
        await asyncio.sleep(interval)


# --------------------------------------------------------------------------------
# Answers to the page and to Pam
# --------------------------------------------------------------------------------
OFF_ANSWER = "I can't check that right now. Please ask {name} about your pills."


def _off(cg: dict) -> dict:
    card = {"title": "Your pills", "body": OFF_ANSWER.format(name=cg["name"])}
    if cg.get("phone"):
        card["action"] = {"label": f"Call {cg['name']}", "href": f"tel:{cg['phone']}"}
    return {"say": OFF_ANSWER.format(name=cg["name"]), "card": card, "enabled": False}


def _same_day(a: float, b: float) -> bool:
    return datetime.fromtimestamp(a).date() == datetime.fromtimestamp(b).date()


def status(doses: dict[int, Dose], now: float, cg: dict | None = None) -> dict:
    """The answer to "did I take my pills?". Only a person's tap counts as taken."""
    cg = cg or caregiver()
    today = [d for d in doses.values() if _same_day(d.due_ts, now)]
    taken = sorted(d.confirmed_ts for d in today if d.confirmed_ts)
    if taken:
        times = [clock(datetime.fromtimestamp(ts), with_period=True) for ts in taken]
        when = times[0] if len(times) == 1 else ", ".join(times[:-1]) + " and " + times[-1]
        say = f"You marked your pills as taken at {when}."
        return {"say": say, "card": {"title": "Your pills", "body": say}, "recorded": True}

    seen = sorted((d.evidence for d in today if d.evidence and d.evidence.get("logged_at")),
                  key=lambda e: str(e["logged_at"]))
    if seen:
        at = clock(datetime.fromisoformat(str(seen[0]["logged_at"])), with_period=True)
        middle = f"I did see your pill bottle move at {at}, but I can't tell whether you took any."
    else:
        middle = "I haven't seen your pill bottle move, but I can't see everything."
    say = (f"I don't have a record that you took your pills. {middle} I can't see inside the bottle. "
           f"Please check with {cg['name']} or your pill organiser before taking any pills.")
    card = {"title": "Your pills", "body": say}
    if cg.get("phone"):
        card["action"] = {"label": f"Call {cg['name']}", "href": f"tel:{cg['phone']}"}
    return {"say": say, "card": card, "recorded": False}


def status_response(now: float | None = None) -> dict:
    cg = caregiver()
    if not enabled():
        return _off(cg)
    return status(replay(read_events()), time.time() if now is None else now, cg)


def confirm(dose_id, answer: str, now: float | None = None) -> dict:
    """The tap. 'yes' records the dose (once); 'not_yet' records nothing but the question."""
    cg = caregiver()
    if not enabled():
        return _off(cg)
    now = time.time() if now is None else now
    d = replay(read_events()).get(dose_id)
    if d is None:
        return {"say": "I don't have a pill question waiting for an answer.", "ok": False}
    if answer == "yes":
        if d.confirmed_ts is None:
            append({"type": "confirmed", "dose": d.id}, now)
        at = clock(datetime.fromtimestamp(d.confirmed_ts or now), with_period=True)
        return {"say": f"Thank you. I've noted that you took your pills at {at}.", "ok": True, "recorded": True}
    if answer == "not_yet":
        append({"type": "not_yet", "dose": d.id}, now)
        return {"say": f"Okay. If you're not sure, please check with {cg['name']} before taking any.", "ok": True}
    return {"say": "I didn't understand that answer.", "ok": False}


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
    append({"type": "evidence", "dose": d.id, "logged_at": datetime.fromtimestamp(now).isoformat(timespec="seconds"),
            "confidence": 1.0, "frame": None, "simulated": True}, now)
    return {"say": "Simulated: the pill bottle was moved.", "ok": True}


# What Pam is told, when the feature is on. See the module docstring for why.
FUNCTION_DESCRIPTION = ("Answer whether the user has taken their pills or medication. Use whenever they ask if they "
                        "took, or already took, their pills, medicine or medication. Read the whole result aloud.")
PROMPT_RULE = """
- Pills: you cannot see inside the pill bottle and you do not know whether the user has taken their medication.
  Never say that they have, or have not, taken it. When they ask, call check_pills_taken and read its answer
  aloud in full, without softening it or adding to it. Never tell them to take, skip or double a dose; when in
  doubt, send them to their caregiver.
"""
