import asyncio
import json
import os
import sys
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from types import SimpleNamespace
import logging
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient
import httpx

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app
import google_calendar as gc
from google.oauth2.credentials import Credentials


CONTACTS = {
    "user": "Test user", "home_airport": "BOS",
    "caregiver": {"name": "Sam", "phone": "+15550100002", "relation": "caregiver"},
    "contacts": [{"name": "Sarah", "phone": "+15550100001", "relation": "granddaughter"},
                 {"name": "Sam", "phone": "+15550100002", "relation": "caregiver"}],
    "places": {"airport": {"lat": 42.36, "lon": -71.01, "address": "Test airport"}},
}


class CapabilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "repo"
        self.here = self.root / "server"
        self.here.mkdir(parents=True)
        self.memory = self.root / "perception" / "runs" / "audit" / "memory.jsonl"
        self.memory.parent.mkdir(parents=True)
        self.memory.write_text("", encoding="utf-8")
        (self.here / "photos").mkdir()
        self.reminders = self.here / "reminders.jsonl"
        self.enterContext(patch.multiple(app, ROOT=self.root, HERE=self.here, MEMORY_JSONL=self.memory,
                                        REMINDERS=self.reminders, CONTACTS=CONTACTS, _fired=set(), _subscribers=set()))
        self.enterContext(patch.dict(os.environ, {key: "" for key in (
            "CALENDAR_ICS_URL", "GOOGLE_CALENDAR_CLIENT_ID", "GOOGLE_CALENDAR_CLIENT_SECRET",
            "TWILIO_SID", "TWILIO_TOKEN", "TWILIO_FROM", "USER_PHONE", "AMADEUS_KEY", "AMADEUS_SECRET")}))
        self.enterContext(patch.object(gc, "calendar_service", gc.CalendarService(Path(self.temp.name) / "credentials.dat")))
        self.client = TestClient(app.app)
        self.addCleanup(self.client.close)

    def test_calendar_does_not_silently_use_demo(self):
        (self.here / "demo.ics").write_text("BEGIN:VCALENDAR\r\nVERSION:2.0\r\nEND:VCALENDAR\r\n", encoding="utf-8")
        result = self.client.get("/api/calendar").json()
        self.assertEqual(result.get("status"), "not_connected")
        self.assertIn("not connected", result["say"])
        self.assertNotIn("free day", result["say"])

    def test_tomorrow_reminder_means_tomorrow(self):
        result = self.client.post("/api/reminders", json={"text": "test", "at": "tomorrow 9am"})
        self.assertEqual(result.status_code, 200)
        due = datetime.fromtimestamp(app._read_reminders()[0]["due_ts"])
        self.assertEqual(due.date(), (datetime.now() + timedelta(days=1)).date())
        self.assertEqual((due.hour, due.minute), (9, 0))

    def test_invalid_reminder_is_not_persisted(self):
        for body in ({"text": "test", "in_minutes": -1}, {"in_minutes": 5}, {"text": "test", "at": "not a time"}):
            response = self.client.post("/api/reminders", json=body)
            self.assertEqual(response.status_code, 400)
        self.assertFalse(self.reminders.exists())

    def test_reminders_round_trip_and_firing_persists(self):
        self.client.post("/api/reminders", json={"text": "a future reminder", "in_minutes": 60})
        records = app._read_reminders()
        records.append({"id": "due-test", "text": "a due reminder", "due_ts": time.time() - 1, "fired": False})
        self.reminders.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
        queue = asyncio.Queue()
        app._subscribers.add(queue)
        async def one_tick(_):
            if one_tick.called:
                raise asyncio.CancelledError
            one_tick.called = True
        one_tick.called = False
        with patch.object(app.asyncio, "sleep", one_tick):
            with self.assertRaises(asyncio.CancelledError):
                asyncio.run(app.reminder_loop())
        self.assertEqual(queue.get_nowait()["type"], "say")
        self.assertTrue(app._read_reminders()[1]["fired"])
        active = self.client.get("/api/reminders").json()["reminders"]
        self.assertEqual([r["text"] for r in active], ["a future reminder"])

    def test_contacts_reject_missing_and_ambiguous_names(self):
        self.assertIsNone(app.contact(""))
        self.assertIsNone(app.contact("S"))
        self.assertEqual(app.contact("Sarah")["name"], "Sarah")
        self.assertEqual(app.contact("__caregiver__")["name"], "Sam")

    def test_message_call_and_caregiver_cards(self):
        sms = self.client.post("/api/message", json={"to": "Sarah", "message": "Hello & goodbye"}).json()
        self.assertTrue(sms["card"]["action"]["href"].startswith("sms:"))
        self.assertIn("Hello%20%26%20goodbye", sms["card"]["action"]["href"])
        for name in ("Sarah", "__caregiver__"):
            result = self.client.post("/api/call", json={"name": name}).json()
            self.assertTrue(result["card"]["action"]["href"].startswith("tel:"))
        self.assertIn("confirm", self.client.get("/api/caregiver-card").json()["card"]["body"])

    def test_twilio_paths_use_only_mock_provider(self):
        provider = MagicMock()
        with patch.object(app, "twilio", return_value=(provider, "+15550100003")), patch.dict(os.environ, {"USER_PHONE": "+15550100004"}):
            result = self.client.post("/api/message", json={"to": "Sarah", "message": "Test"}).json()
            self.assertIn("texted", result["say"])
            provider.messages.create.assert_called_once()
            result = self.client.post("/api/call", json={"name": "Sarah"}).json()
            self.assertIn("ring", result["say"])
            provider.calls.create.assert_called_once()

    def test_ride_requires_a_destination_and_only_creates_a_link(self):
        result = self.client.post("/api/ride", json={"destination": ""}).json()
        self.assertNotIn("card", result)
        result = self.client.post("/api/ride", json={"destination": "airport"}).json()
        self.assertTrue(result["card"]["action"]["href"].startswith("https://m.uber.com/"))

    def test_flights_fallback_is_a_search_not_a_booking(self):
        result = self.client.get("/api/flights", params={"destination": "New York", "date": "next Friday"}).json()
        self.assertIn("flight search", result["say"])
        self.assertTrue(result["card"]["action"]["href"].startswith("https://www.google.com/travel/flights"))

    def test_photo_and_missing_photo(self):
        (self.here / "photos" / "Sarah.jpg").write_bytes(b"test-photo")
        result = self.client.get("/api/photo-info", params={"name": "Sarah"}).json()
        self.assertEqual(self.client.get(result["photo"]["url"]).content, b"test-photo")
        self.assertNotIn("photo", self.client.get("/api/photo-info", params={"name": "missing"}).json())

    def test_find_returns_each_place_and_frame(self):
        records = [{"object": "pill bottle", "location_description": "On a chair", "logged_at": datetime.now().isoformat(),
                    "frames": ["perception/runs/audit/events/1/after.jpg"]},
                   {"object": "pill bottle", "location_description": "On a table", "logged_at": datetime.now().isoformat()}]
        with patch("es.search_all", return_value=(records, "fixture")):
            result = self.client.get("/api/find", params={"q": "medicine"}).json()
        self.assertEqual(len(result["places"]), 2)
        self.assertIn("chair", result["say"])
        self.assertIn("table", result["say"])
        self.assertIn("image", result["card"])
        with patch("es.search_all", return_value=([], "fixture")):
            self.assertNotIn("card", self.client.get("/api/find", params={"q": "missing"}).json())

    def test_frames_do_not_serve_private_files(self):
        (self.here / "private.txt").write_text("synthetic-private-data", encoding="utf-8")
        self.assertEqual(self.client.get("/frames/server/private.txt").status_code, 404)
        jpg = self.memory.parent / "events" / "one" / "after.jpg"
        jpg.parent.mkdir(parents=True)
        jpg.write_bytes(b"test-image")
        self.assertEqual(self.client.get("/frames/perception/runs/audit/events/one/after.jpg").status_code, 200)

    def test_weather_success_and_provider_failure(self):
        original = httpx.AsyncClient
        def transport(request):
            return httpx.Response(200, json={"current": {"temperature_2m": 65.4, "weather_code": 0}})
        with patch.object(app.httpx, "AsyncClient", side_effect=lambda: original(transport=httpx.MockTransport(transport))):
            self.assertIn("65 degrees and clear", self.client.get("/api/weather").json()["say"])
        def unavailable(request):
            raise httpx.ConnectError("offline")
        with patch.object(app.httpx, "AsyncClient", side_effect=lambda: original(transport=httpx.MockTransport(unavailable))):
            self.assertIn("couldn't reach", self.client.get("/api/weather").json()["say"])

    def test_fetch_wrapper_never_moves_real_robot(self):
        with patch("vla.arm_client.fetch", return_value={"active": True}) as fetch:
            self.assertIn("on its way", self.client.post("/api/fetch", json={"item": "pill bottle"}).json()["say"])
            fetch.assert_called_once()
        with patch("vla.arm_client.fetch", side_effect=OSError("offline")):
            self.assertIn("can't reach", self.client.post("/api/fetch", json={"item": "pill bottle"}).json()["say"])

    def test_audited_model_functions_have_expected_contracts(self):
        names = {f["name"] for f in app.FUNCTIONS}
        audited = {"find_object", "get_schedule", "get_reminders", "set_reminder", "get_weather", "show_photo", "search_flights", "send_message", "call_contact", "call_caregiver", "request_ride", "fetch_object"}
        self.assertTrue(audited.issubset(names))
        self.assertEqual(len(names), len(app.FUNCTIONS))
        self.assertIn("exactly ONE step", app.SYSTEM_PROMPT)
        for f in app.FUNCTIONS:
            if f["name"] in {"send_message", "call_contact", "call_caregiver", "set_reminder", "request_ride", "fetch_object"}:
                self.assertTrue(f.get("defer_until_eot"))


class GoogleCalendarTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "calendar.dat"
        self.service = gc.CalendarService(self.path)
        self.enterContext(patch.dict(os.environ, {"GOOGLE_CALENDAR_CLIENT_ID": "test.apps.googleusercontent.com", "GOOGLE_CALENDAR_CLIENT_SECRET": "fixture-secret"}))
        self.enterContext(patch.object(gc, "calendar_service", self.service))

    def credentials(self, expired=False):
        return Credentials("fixture-access", refresh_token="fixture-refresh", token_uri="https://oauth2.googleapis.com/token",
                           client_id="test.apps.googleusercontent.com", client_secret="fixture-secret", scopes=gc.SCOPES,
                           expiry=datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=-1 if expired else 1))

    async def test_state_pkce_scope_and_callback_replay(self):
        url, state = self.service.begin()
        params = parse_qs(urlsplit(url).query)
        self.assertEqual(params["scope"], gc.SCOPES)
        self.assertEqual(params["code_challenge_method"], ["S256"])
        self.assertEqual(params["redirect_uri"], [gc.REDIRECT_URI])
        self.assertEqual(params["access_type"], ["offline"])
        with self.assertRaises(gc.CalendarError):
            await self.service.finish(state, "wrong-cookie", "fixture-code")
        credentials = self.credentials()
        flow = SimpleNamespace(fetch_token=MagicMock(), credentials=credentials)
        self.service.pending[state] = (time.monotonic() + 600, flow)
        await self.service.finish(state, state, "fixture-code")
        self.assertTrue(self.service.status()["connected"])
        flow.fetch_token.assert_called_once_with(code="fixture-code", timeout=12)
        with self.assertRaises(gc.CalendarError):
            await self.service.finish(state, state, "fixture-code")

    async def test_denial_expiry_and_missing_scope(self):
        _, state = self.service.begin()
        with self.assertRaises(gc.CalendarNotConnected):
            await self.service.finish(state, state, "", denied=True)
        self.assertFalse(self.path.exists())
        _, state = self.service.begin()
        expiry, flow = self.service.pending[state]
        self.service.pending[state] = (time.monotonic() - 1, flow)
        with self.assertRaises(gc.CalendarError):
            await self.service.finish(state, state, "fixture-code")
        _, state = self.service.begin()
        denied = SimpleNamespace(granted_scopes=["openid"])
        self.service.pending[state] = (time.monotonic() + 600, SimpleNamespace(fetch_token=MagicMock(), credentials=denied))
        with self.assertRaises(gc.CalendarError):
            await self.service.finish(state, state, "fixture-code")
        self.assertFalse(self.path.exists())

    async def test_credentials_survive_restart_and_are_not_returned(self):
        self.service.save(self.credentials())
        if os.name == "nt":
            self.assertNotIn(b"fixture-refresh", self.path.read_bytes())
        restored = gc.CalendarService(self.path)
        status = restored.status()
        self.assertTrue(status["connected"])
        self.assertNotIn("fixture", json.dumps(status))
        self.assertEqual(await restored.access_token(), "fixture-access")
        with patch.dict(os.environ, {"GOOGLE_CALENDAR_CLIENT_ID": "different-client"}):
            self.assertFalse(gc.CalendarService(self.path).status()["connected"])

    async def test_expired_access_token_refreshes_without_user_prompt(self):
        credentials = self.credentials(expired=True)
        self.service.credentials = credentials
        def refresh(request):
            credentials.token = "refreshed-access"
            credentials.expiry = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=1)
        with patch.object(credentials, "refresh", side_effect=refresh) as refresh_mock:
            self.assertEqual(await self.service.access_token(), "refreshed-access")
            refresh_mock.assert_called_once()
        self.assertTrue(self.path.exists())

    async def test_google_expands_recurrences_and_uses_calendar_timezone(self):
        self.service.credentials = self.credentials()
        requests = []
        def respond(request):
            requests.append(request)
            if not request.url.path.endswith("/events"):
                return httpx.Response(200, json={"timeZone": "America/New_York"})
            if request.url.params.get("pageToken"):
                return httpx.Response(200, json={"items": [{"summary": "Cancelled", "status": "cancelled"},
                    {"summary": "Evening appointment", "start": {"dateTime": "2026-09-20T00:00:00Z"}}]})
            return httpx.Response(200, json={"nextPageToken": "second-page", "items": [
                {"summary": "All-day event", "start": {"date": "2026-09-19"}},
                {"summary": "Recurring appointment", "recurringEventId": "series", "start": {"dateTime": "2026-09-19T13:00:00Z"}}]})
        original = httpx.AsyncClient
        with patch.object(gc.httpx, "AsyncClient", side_effect=lambda **kw: original(transport=httpx.MockTransport(respond), **kw)):
            result = await self.service.today(datetime(2026, 9, 20, 2, tzinfo=timezone.utc))
        self.assertEqual(result["date"], "2026-09-19")
        self.assertEqual([e["time"] for e in result["events"]], ["all day", "9:00 AM", "8:00 PM"])
        self.assertEqual(requests[1].url.params["singleEvents"], "true")
        self.assertEqual(requests[1].url.params["timeMin"], "2026-09-19T00:00:00-04:00")
        self.assertEqual(requests[1].url.params["timeMax"], "2026-09-20T00:00:00-04:00")
        self.assertEqual(requests[2].url.params["pageToken"], "second-page")

    async def test_google_errors_are_not_empty_days_or_secret_leaks(self):
        self.service.credentials = self.credentials()
        original = httpx.AsyncClient
        for status in (401, 403, 500):
            self.service.credentials = self.credentials()
            with patch.object(gc.httpx, "AsyncClient", side_effect=lambda **kw: original(transport=httpx.MockTransport(lambda request: httpx.Response(status, text="fixture-secret")), **kw)):
                with self.assertRaises(gc.CalendarError) as raised:
                    await self.service.today()
            self.assertNotIn("fixture-secret", str(raised.exception))
            self.assertNotIn("free day", str(raised.exception))

    async def test_oauth_routes_are_laptop_only_and_no_store(self):
        with TestClient(app.app, base_url="http://127.0.0.1:8000", client=("127.0.0.1", 55555)) as client:
            response = client.get(gc.CONNECT_PATH, follow_redirects=False)
            self.assertEqual(response.status_code, 303)
            self.assertIn("HttpOnly", response.headers["set-cookie"])
            self.assertIn("samesite=lax", response.headers["set-cookie"].lower())
            self.assertEqual(response.headers["cache-control"], "no-store")
            response = client.get(gc.CALLBACK_PATH, params={"state": "wrong", "code": "fixture-code"})
            self.assertEqual(response.status_code, 400)
            self.assertNotIn("fixture-code", response.text)
        with TestClient(app.app, base_url="http://127.0.0.1:8000", client=("10.0.0.4", 55555)) as client:
            self.assertEqual(client.get(gc.CONNECT_PATH).status_code, 403)

    async def test_callback_codes_are_redacted_from_access_logs(self):
        record = logging.LogRecord("uvicorn.access", logging.INFO, "", 1, "%s %s %s %s %s",
                                   ("127.0.0.1", "GET", gc.CALLBACK_PATH + "?code=fixture-code&state=private-state", "1.1", 303), None)
        gc.RedactCalendarCallback().filter(record)
        self.assertNotIn("fixture-code", record.getMessage())
        self.assertNotIn("private-state", record.getMessage())


async def audit_voice_routing():
    from websockets.asyncio.client import connect
    key = os.environ.get("DEEPGRAM_API_KEY")
    if not key:
        raise SystemExit("Deepgram is not configured.")
    name = app.CONTACTS["contacts"][0]["name"]
    cases = [
        ("find_object", "Where did I leave my pill bottle?"),
        ("get_schedule", "What events are on my calendar today?"),
        ("get_reminders", "Read my current reminders."),
        ("set_reminder", "Remind me to drink water in ten minutes."),
        ("get_weather", "What is the weather at home right now?"),
        ("show_photo", f"Show me a photo of {name}."),
        ("search_flights", "Find flight options to New York next Friday."),
        ("send_message", f"Text {name} the message: I am home."),
        ("call_contact", f"Please call {name}."),
        ("call_caregiver", "Please call my caregiver."),
        ("request_ride", "Help me get an Uber to the airport."),
        ("fetch_object", "Ask the robot arm to fetch my pill bottle."),
        ("guide_me", "Guide me through writing a grocery list, one step at a time."),
    ]
    if "--only" in sys.argv:
        selected = sys.argv[sys.argv.index("--only") + 1]
        cases = [case for case in cases if case[0] == selected]
        if not cases:
            raise SystemExit("Unknown capability")
    results = []
    definitions = {f["name"]: f for f in app.FUNCTIONS}
    for expected, question in cases:
        config = app.agent_config()
        config.pop("greeting", None)
        matched, observed = False, []
        try:
            async with connect("wss://agent.deepgram.com/v1/agent/converse", subprotocols=["token", key], open_timeout=15, close_timeout=2) as ws:
                await ws.send(json.dumps({"type": "Settings", "audio": {"input": {"encoding": "linear16", "sample_rate": 16000}, "output": {"encoding": "linear16", "sample_rate": 24000}}, "agent": config}))
                deadline = time.monotonic() + 45
                while time.monotonic() < deadline:
                    raw = await asyncio.wait_for(ws.recv(), 20)
                    if isinstance(raw, bytes):
                        continue
                    message = json.loads(raw)
                    if message.get("type") == "SettingsApplied":
                        await ws.send(json.dumps({"type": "InjectUserMessage", "content": question}))
                    elif message.get("type") == "Error":
                        observed.append("provider error")
                        break
                    elif message.get("type") == "FunctionCallRequest":
                        for call in message.get("functions", []):
                            observed.append(call["name"])
                            args = json.loads(call.get("arguments") or "{}")
                            required = definitions.get(expected, {}).get("parameters", {}).get("required", [])
                            matched = call["name"] == expected and all(arg in args for arg in required)
                            if matched:
                                break
                        break
                    elif expected == "guide_me" and message.get("type") == "ConversationText" and message.get("role") == "assistant":
                        observed.append(message.get("content", "")[:220])
                    elif expected == "guide_me" and message.get("type") == "AgentAudioDone":
                        text = " ".join(observed).lower()
                        if "?" in text and any(word in text for word in ("first", "paper", "pen", "start")):
                            matched = True
                            break
        except Exception as error:
            observed.append(type(error).__name__)
        results.append(matched)
        print(f"{'PASS' if matched else 'FAIL'} {expected}: {', '.join(observed)}", flush=True)
    print(f"{sum(results)}/{len(results)} live routing checks passed. No function handlers were executed.")
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    if "--live-routing" in sys.argv:
        asyncio.run(audit_voice_routing())
    else:
        unittest.main()
