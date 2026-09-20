"""First run: collect the keys Pam needs, from the laptop, at startup.

    http://127.0.0.1:8000/setup

Two integrations need a credential the user must create themselves, because both are tied
to their own account and neither can ship inside the repo:

    Google Calendar   an OAuth client (ID + secret) for a read-only calendar scope
    Flight search     a SerpApi key, which is what returns Google Flights results

This page takes those values, writes them to the gitignored server/.env with the same
owner-only protection the medication history uses, applies them to the running process so
no restart is needed, and then sends the browser straight into Google's consent screen.
After that first approval the stored refresh token keeps the calendar working by itself.

RULES
  laptop only     every route here refuses anything that is not the loopback address, the
                  same gate the OAuth routes use. A phone on the LAN cannot reach setup.
  never echoed    a saved secret is never returned, rendered, or logged. The status route
                  answers with booleans, not values.
  checked first   a flight key is verified against the provider before it is written, so a
                  typo fails here rather than silently during a conversation.
  nothing partial a refused value leaves .env exactly as it was.
"""
from __future__ import annotations

import os
import re
import tempfile
from datetime import datetime
from pathlib import Path

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse

import google_calendar as gc
from schedule import private_append_fd

HERE = Path(__file__).resolve().parent
ENV_PATH = HERE / ".env"
PAGE = HERE / "setup.html"
BASE_URL = "http://127.0.0.1:8000"
SETUP_PATH = "/setup"

GOOGLE_CLIENT_ID = re.compile(r"^[A-Za-z0-9-]+\.apps\.googleusercontent\.com$")
SECRET = re.compile(r"^\S{8,200}$")          # no whitespace: a pasted line break would corrupt .env
FLIGHT_KEY = re.compile(r"^[A-Za-z0-9]{32,128}$")

import caregiver

router = APIRouter(route_class=caregiver.NoStoreRoute)   # keys and setup state never sit in a cache


def _log(message: str) -> None:
    print(f"{datetime.now():%H:%M:%S} [SETUP ] {message}", flush=True)   # never includes a secret


def app_for_tests():
    """A bare app with just these routes, so the laptop-only gate can be tested off-loopback."""
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    return app


def _refuse_remote(request: Request):
    if gc.local_setup_request(request):
        return None
    return JSONResponse({"error": "Set Pam up on the laptop at http://127.0.0.1:8000/setup ."}, 403,
                        headers={"Cache-Control": "no-store"})


def _read_env_text() -> str:
    if not ENV_PATH.exists():
        return ""
    raw = ENV_PATH.read_bytes()
    for bom, encoding in ((b"\xff\xfe", "utf-16"), (b"\xfe\xff", "utf-16"), (b"\xef\xbb\xbf", "utf-8-sig")):
        if raw.startswith(bom):
            return raw.decode(encoding)
    return raw.decode("utf-8", errors="replace")


def write_env(updates: dict[str, str]) -> None:
    """Replace these keys in server/.env, keeping every other line and comment as it was.

    Written to a private temporary file and moved into place, so a crash cannot leave the
    file half-written and a reader never sees a torn .env."""
    lines = _read_env_text().splitlines()
    for key, value in updates.items():
        assigned = f"{key}={value}"
        for index, line in enumerate(lines):
            if line.split("=", 1)[0].strip() == key and not line.lstrip().startswith("#"):
                lines[index] = assigned
                break
        else:
            lines.append(assigned)
    text = "\n".join(lines).strip("\n") + "\n"
    ENV_PATH.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=ENV_PATH.parent, prefix=".env.", suffix=".tmp")
    os.close(handle)
    temporary = Path(temporary)
    try:
        fd = private_append_fd(temporary)
        try:
            os.write(fd, text.encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)
        temporary.replace(ENV_PATH)
    finally:
        if temporary.exists():
            temporary.unlink()


def status() -> dict:
    calendar = gc.calendar_service.status()
    return {"calendar": {"configured": calendar["configured"], "connected": calendar["connected"]},
            "flights": {"configured": bool((os.environ.get("SERPAPI_KEY") or "").strip())},
            "connect_path": gc.CONNECT_PATH, "redirect_uri": gc.REDIRECT_URI}


@router.get(SETUP_PATH)
async def setup_page(request: Request):
    return _refuse_remote(request) or FileResponse(PAGE, headers={"Cache-Control": "no-store"})


@router.get("/api/setup/status")
async def setup_status(request: Request):
    return _refuse_remote(request) or JSONResponse(status(), headers={"Cache-Control": "no-store"})


@router.post("/api/setup/google")
async def save_google(request: Request, body: dict):
    refused = _refuse_remote(request)
    if refused:
        return refused
    client_id = str(body.get("client_id", "")).strip()
    client_secret = str(body.get("client_secret", "")).strip()
    if not GOOGLE_CLIENT_ID.match(client_id) or not SECRET.match(client_secret):
        return JSONResponse({"error": "That doesn't look like a Google OAuth client. The ID ends in "
                                      ".apps.googleusercontent.com, and the secret has no spaces."}, 400)
    write_env({"GOOGLE_CALENDAR_CLIENT_ID": client_id, "GOOGLE_CALENDAR_CLIENT_SECRET": client_secret})
    os.environ["GOOGLE_CALENDAR_CLIENT_ID"] = client_id
    os.environ["GOOGLE_CALENDAR_CLIENT_SECRET"] = client_secret
    gc.calendar_service.reset()
    _log("Google OAuth client saved; sending the browser to Google's consent screen")
    return {"ok": True, "connect_path": gc.CONNECT_PATH}


@router.post("/api/setup/flights")
async def save_flights(request: Request, body: dict):
    refused = _refuse_remote(request)
    if refused:
        return refused
    api_key = str(body.get("api_key", "")).strip()
    if not FLIGHT_KEY.match(api_key):
        return JSONResponse({"error": "That doesn't look like a SerpApi key. Copy the whole key from your dashboard."}, 400)
    try:  # /account is free and does not spend a search, so a typo costs nothing
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get("https://serpapi.com/account", params={"api_key": api_key})
    except httpx.HTTPError:
        return JSONResponse({"error": "I couldn't reach the flight service to check that key. Try again in a moment."}, 503)
    if response.status_code != 200:
        return JSONResponse({"error": "The flight service did not accept that key. Check it and paste it again."}, 400)
    write_env({"SERPAPI_KEY": api_key})
    os.environ["SERPAPI_KEY"] = api_key
    _log("flight-search key saved and accepted by the provider")
    return {"ok": True}


def start(open_browser=None, announce=print) -> bool:
    """Called once at startup. Opens whatever the user still has to do, and nothing if they
    are done. Never raises: Pam must start with or without a browser."""
    import webbrowser

    open_browser = open_browser or webbrowser.open
    state = status()
    calendar, flights = state["calendar"], state["flights"]
    if calendar["connected"] and flights["configured"]:
        announce("  Setup: Google Calendar connected, flight search connected.")
        return False
    if not calendar["configured"] or not flights["configured"]:
        url, what = BASE_URL + SETUP_PATH, "finish setup"
    else:
        url, what = BASE_URL + gc.CONNECT_PATH, "approve Google Calendar"
    try:
        opened = open_browser(url) is not False
    except Exception:
        opened = False
    announce(f"  Setup: opening {url} so you can {what}." if opened
             else f"  Setup: open {url} on this laptop to {what}.")
    return opened
