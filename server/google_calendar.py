import asyncio
import ctypes
import ipaddress
import json
import logging
import os
import secrets
import tempfile
import time
from datetime import datetime, time as day_time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
from google.auth.transport.requests import Request as GoogleRequest
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow

SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]
REDIRECT_URI = "http://127.0.0.1:8000/api/calendar/google/callback"
CONNECT_PATH = "/api/calendar/google/connect"
CALLBACK_PATH = "/api/calendar/google/callback"
COOKIE = "pam_calendar_oauth"


class CalendarError(Exception):
    pass


class CalendarNotConnected(CalendarError):
    pass


class RedactCalendarCallback(logging.Filter):
    def filter(self, record):
        if isinstance(record.args, tuple) and len(record.args) == 5:
            args = list(record.args)
            if isinstance(args[2], str) and args[2].split("?", 1)[0] == CALLBACK_PATH:
                args[2] = CALLBACK_PATH
                record.args = tuple(args)
        return True


def install_log_filter():
    logger = logging.getLogger("uvicorn.access")
    if not any(isinstance(f, RedactCalendarCallback) for f in logger.filters):
        logger.addFilter(RedactCalendarCallback())


def local_setup_request(request):
    try:
        loopback = ipaddress.ip_address(request.client.host).is_loopback
    except (ValueError, AttributeError):
        return False
    return loopback and request.headers.get("host") == "127.0.0.1:8000" and request.url.scheme == "http"


def token_path():
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path.home() / ".local" / "share"
    return base / "Pam" / "google-calendar.dat"


def protect(data, decrypt=False):
    if os.name != "nt":
        return data
    from ctypes import wintypes

    class Blob(ctypes.Structure):
        _fields_ = [("size", wintypes.DWORD), ("data", ctypes.c_void_p)]

    buffer = ctypes.create_string_buffer(data)
    source = Blob(len(data), ctypes.cast(buffer, ctypes.c_void_p))
    result = Blob()
    function = ctypes.windll.crypt32.CryptUnprotectData if decrypt else ctypes.windll.crypt32.CryptProtectData
    function.argtypes = [ctypes.POINTER(Blob), ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                         ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(Blob)]
    function.restype = wintypes.BOOL
    if not function(ctypes.byref(source), None, None, None, None, 1, ctypes.byref(result)):
        raise OSError("Windows could not protect the calendar connection.")
    try:
        return ctypes.string_at(result.data, result.size)
    finally:
        free = ctypes.windll.kernel32.LocalFree
        free.argtypes = [ctypes.c_void_p]
        free.restype = ctypes.c_void_p
        free(result.data)


class CalendarService:
    def __init__(self, path=None):
        self.path = path or token_path()
        self.credentials = None
        self.loaded = False
        self.pending = {}
        self.lock = asyncio.Lock()

    def configured(self):
        return bool(os.environ.get("GOOGLE_CALENDAR_CLIENT_ID") and os.environ.get("GOOGLE_CALENDAR_CLIENT_SECRET"))

    def load(self):
        if self.loaded or not self.configured():
            return
        self.loaded = True
        if not self.path.exists():
            return
        try:
            payload = json.loads(protect(self.path.read_bytes(), decrypt=True).decode("utf-8"))
            if payload.get("client_id") != os.environ.get("GOOGLE_CALENDAR_CLIENT_ID"):
                return
            payload["client_secret"] = os.environ["GOOGLE_CALENDAR_CLIENT_SECRET"]
            self.credentials = Credentials.from_authorized_user_info(payload, scopes=SCOPES)
        except (OSError, ValueError, KeyError):
            self.credentials = None

    def reset(self):
        """Re-read the client from the environment: setup.py can change it without a restart."""
        self.loaded, self.credentials = False, None
        self.pending.clear()

    def status(self):
        self.load()
        return {"configured": self.configured(), "connected": bool(self.configured() and self.credentials),
                "read_only": True, "source": "google_calendar"}

    def save(self, credentials):
        payload = protect(credentials.to_json(strip=["client_secret"]).encode("utf-8"))
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.path.parent, delete=False) as f:
                temporary = Path(f.name)
                if os.name != "nt":
                    os.chmod(temporary, 0o600)
                f.write(payload)
            temporary.replace(self.path)
        finally:
            if temporary and temporary.exists():
                temporary.unlink()
        self.credentials = credentials
        self.loaded = True

    def begin(self):
        if not self.configured():
            raise CalendarNotConnected("Google Calendar OAuth credentials have not been configured on the laptop.")
        self.pending = {state: value for state, value in self.pending.items() if value[0] > time.monotonic()}
        if len(self.pending) >= 16:
            raise CalendarError("Too many calendar sign-in attempts. Wait a few minutes and try again.")
        flow = Flow.from_client_config({"web": {
            "client_id": os.environ["GOOGLE_CALENDAR_CLIENT_ID"],
            "client_secret": os.environ["GOOGLE_CALENDAR_CLIENT_SECRET"],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [REDIRECT_URI],
        }}, scopes=SCOPES, autogenerate_code_verifier=True)
        flow.redirect_uri = REDIRECT_URI
        url, state = flow.authorization_url(access_type="offline", prompt="consent")
        self.pending[state] = (time.monotonic() + 600, flow)
        return url, state

    async def finish(self, state, cookie, code, denied=False):
        if not state or not cookie or not secrets.compare_digest(state.encode(), cookie.encode()):
            raise CalendarError("Calendar sign-in could not be verified. Start again from Pam on the laptop.")
        pending = self.pending.pop(state, None)
        if not pending or pending[0] <= time.monotonic():
            raise CalendarError("Calendar sign-in expired. Start again from Pam on the laptop.")
        if denied or not code:
            raise CalendarNotConnected("Google Calendar access was not granted. No calendar was connected.")
        flow = pending[1]
        async with self.lock:
            try:
                await asyncio.to_thread(flow.fetch_token, code=code, timeout=12)
                credentials = flow.credentials
                granted = credentials.granted_scopes
                if granted is not None and not set(SCOPES).issubset(granted):
                    raise CalendarError("Read-only Calendar permission was not granted. Please reconnect and allow it.")
                if not credentials.refresh_token:
                    raise CalendarError("Google did not grant offline access. Please reconnect your calendar.")
                self.save(credentials)
            except CalendarError:
                raise
            except Exception:
                raise CalendarError("Google Calendar sign-in could not finish. Check the OAuth client and redirect URI, then try again.") from None

    async def access_token(self):
        self.load()
        if not self.configured() or not self.credentials:
            raise CalendarNotConnected("Your Google Calendar is not connected yet. Ask your helper to connect it on the laptop.")
        async with self.lock:
            if not self.credentials.valid:
                try:
                    await asyncio.to_thread(self.credentials.refresh, GoogleRequest())
                    self.save(self.credentials)
                except Exception:
                    self.credentials = None
                    raise CalendarNotConnected("Google Calendar needs to be reconnected on the laptop.") from None
            return self.credentials.token

    async def today(self, now=None):
        token = await self.access_token()
        async with httpx.AsyncClient(timeout=12) as client:
            async def get(url, params=None):
                response = await client.get(url, params=params, headers={"Authorization": f"Bearer {token}"})
                if response.status_code == 401:
                    self.credentials = None
                    raise CalendarNotConnected("Google Calendar needs to be reconnected on the laptop.")
                if response.status_code == 403:
                    raise CalendarError("Google Calendar access was refused. Check that the Calendar API is enabled and reconnect with read-only permission.")
                response.raise_for_status()
                return response.json()
            try:
                metadata = await get("https://www.googleapis.com/calendar/v3/calendars/primary", {"fields": "timeZone"})
                zone = ZoneInfo(metadata["timeZone"])
                today = (now or datetime.now(timezone.utc)).astimezone(zone).date()
                start = datetime.combine(today, day_time.min, tzinfo=zone)
                params = {"timeMin": start.isoformat(), "timeMax": (start + timedelta(days=1)).isoformat(),
                          "singleEvents": "true", "orderBy": "startTime", "showDeleted": "false", "maxResults": 2500,
                          "fields": "items(summary,status,start,end,location),nextPageToken"}
                events, seen = [], set()
                while True:
                    page = await get("https://www.googleapis.com/calendar/v3/calendars/primary/events", params)
                    for event in page.get("items", []):
                        if event.get("status") == "cancelled":
                            continue
                        beginning = event.get("start", {})
                        all_day = "date" in beginning
                        if all_day:
                            when, sort_key = "all day", beginning["date"]
                        elif beginning.get("dateTime"):
                            dt = datetime.fromisoformat(beginning["dateTime"].replace("Z", "+00:00")).astimezone(zone)
                            when, sort_key = dt.strftime("%I:%M %p").lstrip("0"), dt.isoformat()
                            if dt.date() < today:
                                when = f"continuing from {dt.strftime('%b %d')} at {when}"
                        else:
                            continue
                        events.append({"title": event.get("summary") or "Untitled event", "time": when,
                                       "all_day": all_day, "location": event.get("location", ""), "_dt": sort_key})
                    page_token = page.get("nextPageToken")
                    if not page_token:
                        break
                    if page_token in seen or len(seen) >= 100:
                        raise CalendarError("Google Calendar returned an incomplete event list. Please try again.")
                    seen.add(page_token)
                    params["pageToken"] = page_token
                events.sort(key=lambda event: (not event["all_day"], event["_dt"]))
                return {"events": events, "timezone": str(zone), "date": today.isoformat()}
            except CalendarError:
                raise
            except (httpx.HTTPError, ValueError, KeyError, TypeError):
                raise CalendarError("I couldn't read Google Calendar just now. Please try again; I can't tell whether your day is free.") from None


calendar_service = CalendarService()
