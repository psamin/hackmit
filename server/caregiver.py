"""Caregiver access: one shared PIN, no accounts. Everything under /caregiver and
/api/caregiver/ sits behind it.

    CAREGIVER_PIN=<6 or more characters>   in server/.env, then restart the server

--------------------------------------------------------------------------------
WHAT THIS IS, AND IS NOT
--------------------------------------------------------------------------------
There is no sign-up, no password, no reset and no per-person identity. One caregiver PIN
opens the dashboard. That is enough to keep the medication data away from anyone else on the
same network. It is not enough for a real product, which would use real accounts or a
sign-in provider so that a change could be attributed to a person. The audit log will say
"the caregiver", not who.

It protects only the routes in this file. The patient-side endpoints elsewhere in the server
(message, call, ride, fetch, confirm a dose) have never had a login, and this does not add one.

--------------------------------------------------------------------------------
THE RULES
--------------------------------------------------------------------------------
  fails closed     no PIN, or a PIN shorter than MIN_PIN_LEN, means the feature is OFF: every
                   protected route answers 404 and the page says the dashboard is not set up.
                   Forgetting to configure it can never leave the data open.
  PIN stays put    it is compared in constant time, sent only in a JSON body (never a URL), and
                   never logged, echoed, or stored. What the browser keeps is a random session
                   token in an HttpOnly, SameSite=Strict cookie (Secure over https).
  lockout          MAX_FAILURES wrong PINs within LOCKOUT_WINDOW_S locks logins out, even the
                   right PIN, until the oldest failure ages out. The lock is global, not per
                   client: behind localhost every client looks the same. The cost is that
                   someone can lock the caregiver out for a few minutes; that is the right side
                   to err on for a prototype.
  sessions         live in memory and last SESSION_TTL_S. A restart logs everyone out.
"""
from __future__ import annotations

import hmac
import os
import secrets
import time
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.routing import APIRoute

HERE = Path(__file__).resolve().parent
PAGE = HERE / "caregiver.html"

MIN_PIN_LEN = 6
SESSION_TTL_S = 8 * 3600
MAX_FAILURES = 5
LOCKOUT_WINDOW_S = 300
COOKIE = "pam_caregiver"

class NoStoreRoute(APIRoute):
    """Health data must not linger in a browser cache: after sign-out, Back should not show it."""

    def get_route_handler(self):
        original = super().get_route_handler()

        async def handler(request: Request):
            response = await original(request)
            response.headers["Cache-Control"] = "no-store"
            return response
        return handler


router = APIRouter(route_class=NoStoreRoute)
_sessions: dict[str, float] = {}   # token -> expiry
_failures: list[float] = []        # times of recent wrong PINs
_now = time.time                   # a seam, so tests can move the clock


def _log(msg: str) -> None:
    print(f"{datetime.now():%H:%M:%S} [CARE  ] {msg}", flush=True)   # never includes the PIN


def _pin() -> str | None:
    """The configured PIN, or None when the feature is off (unset, or too short to trust)."""
    pin = os.environ.get("CAREGIVER_PIN", "")
    return pin if len(pin) >= MIN_PIN_LEN else None


def enabled() -> bool:
    return _pin() is not None


def _lockout_remaining(now: float) -> int:
    """Seconds until logins are allowed again; 0 when they are allowed now."""
    _failures[:] = [t for t in _failures if now - t < LOCKOUT_WINDOW_S]
    if len(_failures) < MAX_FAILURES:
        return 0
    return int(_failures[0] + LOCKOUT_WINDOW_S - now) + 1


def _valid_session(token: str | None, now: float) -> bool:
    for t in [t for t, exp in _sessions.items() if exp <= now]:
        del _sessions[t]                                   # sweep the expired ones
    return bool(token) and token in _sessions


def require_caregiver(request: Request) -> None:
    """Dependency for every protected route. 404 when the feature is off, 401 when not signed in."""
    if not enabled():
        raise HTTPException(status_code=404)
    if not _valid_session(request.cookies.get(COOKIE), _now()):
        raise HTTPException(status_code=401, detail="Please sign in.")


@router.get("/caregiver")
async def caregiver_page():
    """The page itself is public; it holds no data and asks /api/caregiver/me who is looking."""
    return FileResponse(PAGE)


@router.get("/api/caregiver/me")
async def me(request: Request):
    on = enabled()
    return {"enabled": on, "authenticated": on and _valid_session(request.cookies.get(COOKIE), _now())}


@router.post("/api/caregiver/login")
async def login(request: Request, body: dict):
    pin = _pin()
    if pin is None:
        raise HTTPException(status_code=404)
    now = _now()
    wait = _lockout_remaining(now)
    if wait:
        return JSONResponse({"error": "Too many tries. Please wait a few minutes.", "retry_after_s": wait},
                            status_code=429, headers={"Retry-After": str(wait)})
    if not hmac.compare_digest(str(body.get("pin", "")).encode(), pin.encode()):
        _failures.append(now)
        _log(f"wrong PIN ({len(_failures)} of {MAX_FAILURES} before lockout)")
        return JSONResponse({"error": "That PIN isn't right."}, status_code=401)
    _failures.clear()
    token = secrets.token_urlsafe(32)
    _sessions[token] = now + SESSION_TTL_S
    _log("caregiver signed in")
    resp = JSONResponse({"ok": True})
    resp.set_cookie(COOKIE, token, max_age=SESSION_TTL_S, httponly=True, samesite="strict",
                    secure=request.url.scheme == "https", path="/")
    return resp


@router.post("/api/caregiver/logout")
async def logout(request: Request):
    _sessions.pop(request.cookies.get(COOKIE, ""), None)
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(COOKIE, path="/")
    return resp


@router.get("/api/caregiver/status", dependencies=[Depends(require_caregiver)])
async def status():
    """What the dashboard shell shows for now: which parts of the system are switched on."""
    import doses
    health = doses.schedule_health()
    return {"medication_check": doses.enabled(), "demo_timings": doses.demo_mode(),
            "schedule_reminders": doses.schedule_enabled(), "schedule": health["state"],
            "schedule_version": health["version"], "scheduler_running": health["running"],
            "elasticsearch": bool(os.environ.get("ELASTICSEARCH_URL"))}
