"""Caregiver routes for the medication schedule: see the current one, read new instructions into a
draft, and save a draft once a person has looked at it.

    GET  /api/caregiver/schedule          the current schedule in words, and recent history
    POST /api/caregiver/schedule/parse    {"text": "..."} -> a DRAFT to review (saves nothing)
    POST /api/caregiver/schedule/save     {"draft_id": "...", "acknowledged": [...]} -> saves exactly that draft

Every route sits behind the caregiver PIN (see caregiver.py). Nothing here can change a schedule
without a person clicking Save on a draft they were shown; the rules for that live in schedule_parse.py.

Parsing calls a paid AI service, so it is limited to PARSE_LIMIT reads per PARSE_WINDOW_S.
"""
from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

import adherence
import caregiver
import doses
import schedule as sched
import schedule_parse as sp

router = APIRouter(route_class=caregiver.NoStoreRoute, dependencies=[Depends(caregiver.require_caregiver)])

DRAFTS = sp.Drafts()
PARSE_LIMIT, PARSE_WINDOW_S = 10, 600
_parse_times: list[float] = []
_now = time.time                      # a seam, so tests can move the clock


def get_store() -> sched.Store:      # seams, so tests use a temporary file and a fake model
    return sched.Store()


def get_llm() -> sp.LLM:
    return sp.AnthropicLLM()


def _error(message: str, status: int = 400) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


def _instructions_only(text: str | None) -> str:
    """The caregiver's words without the acknowledgement note added at save time."""
    return (text or "").split(sp.ACK_MARK)[0].strip()


@router.get("/api/caregiver/schedule")
async def current_schedule():
    store = get_store()
    try:
        entries = store.history()
    except sched.ScheduleCorrupt as exc:
        return _error(str(exc), 500)
    if not entries:
        return {"exists": False}
    latest = entries[0]
    prefill = next((_instructions_only(e.source_text) for e in entries
                    if e.action == "save" and _instructions_only(e.source_text)), "")
    return {"exists": True, "version": latest.version, "saved_at": latest.ts, "lines": sched.describe(latest.schedule),
            "instructions": prefill,
            "history": [{"version": e.version, "ts": e.ts, "action": e.action, "changes": e.changes} for e in entries[:6]]}


@router.get("/api/caregiver/adherence")
async def adherence_view(days: int = 14):
    """What Pam oversight shows: each recent day, each dose and whether it was tapped on time, late or not at all.
    Labelled as recorded, never as "taken": these are taps. The caregiver sees this even if the patient's
    streak is hidden."""
    days = max(1, min(int(days), 90))
    return adherence.public(doses.adherence_summary(), days)


@router.post("/api/caregiver/schedule/parse")
async def parse(body: dict):
    text = body.get("text")
    if not isinstance(text, str):
        return _error("Please type the instructions first.")
    now = _now()
    _parse_times[:] = [t for t in _parse_times if now - t < PARSE_WINDOW_S]
    if len(_parse_times) >= PARSE_LIMIT:
        return _error("That's a lot of readings in a short time. Please wait a few minutes.", 429)
    store = get_store()
    try:
        current = store.load()
    except sched.ScheduleCorrupt as exc:
        return _error(str(exc), 500)
    if text.strip() and len(text.strip()) <= sp.MAX_TEXT_CHARS:
        _parse_times.append(now)                                     # only real attempts use up the allowance
    try:
        draft = await asyncio.to_thread(sp.build_draft, text, current, get_llm(), now=now)   # blocking call: keep it off the event loop
    except sp.ParseFailed as exc:
        return _error(exc.message, exc.status)
    DRAFTS.add(draft)
    return draft.public()


@router.post("/api/caregiver/schedule/save")
async def save(body: dict):
    acknowledged = body.get("acknowledged")
    try:
        entry = sp.save_draft(DRAFTS, get_store(), body.get("draft_id"),
                              acknowledged if isinstance(acknowledged, list) else [])
    except sp.SaveRefused as exc:
        return _error(exc.message, 409)
    except sched.ScheduleCorrupt as exc:
        return _error(str(exc), 500)
    return {"ok": True, "version": entry.version, "lines": sched.describe(entry.schedule)}
