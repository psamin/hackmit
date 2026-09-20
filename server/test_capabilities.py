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
            "AMADEUS_KEY", "AMADEUS_SECRET")}))
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
        self.assertIsNone(app.contact("__caregiver__"))

    def test_calling_and_texting_are_removed(self):
        removed = {"send_message", "call_contact", "call_caregiver"}
        self.assertTrue(removed.isdisjoint(f["name"] for f in app.FUNCTIONS))
        self.assertEqual(self.client.post("/api/message", json={"to": "Sarah", "message": "Test"}).status_code, 404)
        self.assertEqual(self.client.post("/api/call", json={"name": "Sarah"}).status_code, 404)
        self.assertEqual(self.client.get("/api/caregiver-card").status_code, 404)
        self.assertNotIn("twilio", self.client.get("/api/health").json())

    def test_ride_requires_a_destination_and_only_creates_a_link(self):
        result = self.client.post("/api/ride", json={"destination": ""}).json()
        self.assertNotIn("card", result)
        result = self.client.post("/api/ride", json={"destination": "airport"}).json()
        self.assertTrue(result["card"]["action"]["href"].startswith("https://m.uber.com/"))

    def test_flights_without_provider_are_voice_only_and_honest(self):
        result = self.client.get("/api/flights", params={"destination": "New York", "date": "next Friday"}).json()
        self.assertEqual(result["status"], "not_configured")
        self.assertEqual(result["offers"], [])
        self.assertIn("not connected", result["say"])
        self.assertNotRegex(result["say"].lower(), r"\b(screen|tap|click|link)\b")
        self.assertNotIn("card", result)

    def flight_fixture(self, provider_status=200, empty=False, payload=None, destination="New York"):
        travel_date = (datetime.now() + timedelta(days=7)).date().isoformat()
        payload = payload if payload is not None else {"best_flights": [] if empty else [{
            "flights": [{"departure_airport": {"id": "BOS", "time": travel_date + " 08:30"},
                         "arrival_airport": {"id": "JFK", "time": travel_date + " 10:00"},
                         "airline": "Example Air", "flight_number": "XA 100"}],
            "layovers": [], "price": 125, "type": "One way"}], "other_flights": []}
        requests = []
        def respond(request):
            requests.append(request)
            return httpx.Response(provider_status, json=payload if provider_status == 200 else {"error": "fixture-secret rejected"})
        original = httpx.AsyncClient
        with patch.dict(os.environ, {"SERPAPI_KEY": "fixture-secret"}), \
             patch.object(app.httpx, "AsyncClient", side_effect=lambda **kw: original(transport=httpx.MockTransport(respond), **kw)):
            output = self.client.get("/api/flights", params={"destination": destination, "date": travel_date}).json()
        return output, requests

    def test_flight_options_are_complete_spoken_answers(self):
        result, requests = self.flight_fixture()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(result["offers"]), 1)
        for value in ("Example Air", "8:30 AM", "10:00 AM", "125", "US dollars", "nonstop", "BOS", "JFK"):
            self.assertIn(value, result["say"])
        self.assertNotRegex(result["say"].lower(), r"\b(screen|tap|click|link)\b")
        self.assertNotIn("card", result)
        self.assertEqual(requests[-1].url.host, "serpapi.com")
        query = requests[-1].url.params
        self.assertEqual(query["engine"], "google_flights")
        self.assertEqual((query["departure_id"], query["arrival_id"], query["type"], query["adults"]), ("BOS", "JFK", "2", "1"))

    def test_spoken_connections_keep_the_returned_currency(self):
        result = app.spoken_flight_offer({"price": 280, "flights": [
            {"departure_airport": {"id": "BOS", "time": "2027-05-08 08:00"}, "arrival_airport": {"id": "ORD", "time": "2027-05-08 09:00"}, "airline": "Example Air"},
            {"departure_airport": {"id": "ORD", "time": "2027-05-08 10:00"}, "arrival_airport": {"id": "SFO", "time": "2027-05-08 13:00"}, "airline": "Sample Air"}]}, "CAD")
        self.assertEqual(result["stops"], 1)
        self.assertEqual(result["currency"], "CAD")
        self.assertIn("Example Air and Sample Air", result["summary"])
        self.assertIn("Canadian dollars", result["summary"])
        self.assertIn("1 stop", result["summary"])

    def test_flight_provider_error_is_not_no_results(self):
        result, _ = self.flight_fixture(provider_status=401)
        self.assertEqual(result["status"], "unavailable")
        self.assertNotIn("fixture-secret", json.dumps(result))
        self.assertNotRegex(result["say"].lower(), r"\b(screen|tap|click|link)\b")
        result, _ = self.flight_fixture(payload={"error": "Google Flights hasn't returned any results"})
        self.assertEqual(result["status"], "no_offers")
        result, _ = self.flight_fixture(empty=True)
        self.assertEqual(result["status"], "no_offers")
        self.assertEqual(result["offers"], [])

    def test_flight_search_key_is_never_exposed(self):
        result, requests = self.flight_fixture()
        self.assertEqual(requests[-1].url.params["api_key"], "fixture-secret")
        self.assertNotIn("fixture-secret", json.dumps(result))
        with self.assertLogs() if False else patch.object(app, "log") as logged:
            self.client.get("/api/flights", params={"destination": "New York", "date": "next Friday"})
        self.assertNotIn("api_key", json.dumps([str(call) for call in logged.call_args_list]))

    def test_flight_date_is_requested_instead_of_guessed(self):
        with patch.dict(os.environ, {"SERPAPI_KEY": "fixture-secret"}), \
             patch.object(app.httpx, "AsyncClient", side_effect=AssertionError("Must ask for a date first")):
            result = self.client.get("/api/flights", params={"destination": "New York"}).json()
        self.assertEqual(result["status"], "needs_date")
        self.assertIn("day", result["say"])

    def test_voice_prompt_and_action_fallbacks_do_not_assume_a_screen(self):
        self.assertIn("voice-first", app.SYSTEM_PROMPT)
        self.assertNotIn("button will appear on their screen", app.SYSTEM_PROMPT)
        flight = next(f for f in app.FUNCTIONS if f["name"] == "search_flights")
        self.assertNotIn("screen", flight["description"])
        results = [self.client.post("/api/ride", json={"destination": "airport"}).json()]
        for result in results:
            self.assertNotRegex(result["say"].lower(), r"\b(screen|tap|click)\b")
            self.assertIn("helper", result["say"])

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
        audited = {"find_object", "get_schedule", "get_reminders", "set_reminder", "get_weather", "show_photo", "search_flights", "request_ride", "fetch_object"}
        self.assertTrue(audited.issubset(names))
        self.assertEqual(len(names), len(app.FUNCTIONS))
        self.assertIn("exactly ONE step", app.SYSTEM_PROMPT)
        for f in app.FUNCTIONS:
            if f["name"] in {"set_reminder", "request_ride", "fetch_object"}:
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

    async def test_startup_opens_consent_once_then_stays_connected(self):
        opened, messages = [], []
        self.assertTrue(gc.start_setup(opened.append, messages.append))
        self.assertEqual(opened, ["http://127.0.0.1:8000" + gc.CONNECT_PATH])
        self.assertTrue(any("browser" in message.lower() for message in messages))
        self.service.save(self.credentials())
        opened.clear()
        self.assertFalse(gc.start_setup(opened.append, messages.append))
        self.assertEqual(opened, [])
        self.assertTrue(any("connected" in message.lower() for message in messages))

    async def test_startup_without_credentials_explains_setup_and_opens_nothing(self):
        opened, messages = [], []
        with patch.dict(os.environ, {"GOOGLE_CALENDAR_CLIENT_ID": "", "GOOGLE_CALENDAR_CLIENT_SECRET": ""}):
            self.assertFalse(gc.start_setup(opened.append, messages.append))
        self.assertEqual(opened, [])
        self.assertTrue(any(gc.REDIRECT_URI in message for message in messages))

    async def test_startup_never_blocks_pam_when_the_browser_fails(self):
        def refuse(url):
            raise OSError("no browser")
        messages = []
        self.assertFalse(gc.start_setup(refuse, messages.append))
        self.assertTrue(any(gc.CONNECT_PATH in message for message in messages))

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
        matched, observed, approved = False, [], False
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
                    elif expected in {"request_ride", "fetch_object"} and not approved and message.get("type") == "ConversationText" and message.get("role") == "assistant" and "?" in message.get("content", ""):
                        approved = True
                        await ws.send(json.dumps({"type": "InjectUserMessage", "content": "Yes, I approve the action as you just described it."}))
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
