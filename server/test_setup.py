"""Tests for setup.py: the laptop-only first-run page that collects the Google Calendar
OAuth client and the flight-search key, writes them to server/.env and applies them live.

    python server/test_setup.py

The ones that matter most: a secret is never echoed back, logged, or readable off the
laptop; a bad key is refused before anything is written; and an existing .env keeps every
line it already had.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent))
import google_calendar as gc  # noqa: E402
import setup as st  # noqa: E402

CLIENT_ID = "1234567890-fixture.apps.googleusercontent.com"
CLIENT_SECRET = "FIXTURE-google-client-secret"
FLIGHT_KEY = "fixture0flight0key" + "0" * 46
EXISTING = "# Pam config\nDEEPGRAM_API_KEY=keep-me\n\n# comment\nHOME_LAT=42.3601\n"


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.env = self.tmp / ".env"
        self.env.write_text(EXISTING, encoding="utf-8")
        self.service = gc.CalendarService(self.tmp / "calendar.dat")
        self.logged = []
        patches = [mock.patch.object(st, "ENV_PATH", self.env),
                   mock.patch.object(st, "_log", self.logged.append),
                   mock.patch.object(gc, "calendar_service", self.service),
                   mock.patch.dict(os.environ, {"GOOGLE_CALENDAR_CLIENT_ID": "", "GOOGLE_CALENDAR_CLIENT_SECRET": "",
                                                "SERPAPI_KEY": ""}, clear=False)]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        app = FastAPI()
        app.include_router(st.router)
        self.client = TestClient(app, base_url="http://127.0.0.1:8000", client=("127.0.0.1", 50505))
        self.addCleanup(self.client.close)

    def env_values(self):
        values = {}
        for line in self.env.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition("=")
            if key.strip() and not key.startswith("#"):
                values[key.strip()] = value.strip()
        return values

    def save_google(self, client_id=CLIENT_ID, secret=CLIENT_SECRET):
        return self.client.post("/api/setup/google", json={"client_id": client_id, "client_secret": secret})

    def save_flights(self, key=FLIGHT_KEY, status=200, payload=None):
        seen = []
        def respond(request):
            seen.append(request)
            return httpx.Response(status, json=payload if payload is not None else {"plan_name": "Free"})
        original = httpx.AsyncClient
        with mock.patch.object(st.httpx, "AsyncClient", side_effect=lambda **kw: original(transport=httpx.MockTransport(respond), **kw)):
            return self.client.post("/api/setup/flights", json={"api_key": key}), seen


class Storing(Base):
    def test_google_client_is_saved_applied_and_never_echoed(self):
        response = self.save_google()
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["connect_path"], gc.CONNECT_PATH)
        self.assertNotIn(CLIENT_SECRET, json.dumps(body))
        self.assertEqual(self.env_values()["GOOGLE_CALENDAR_CLIENT_ID"], CLIENT_ID)
        self.assertEqual(self.env_values()["GOOGLE_CALENDAR_CLIENT_SECRET"], CLIENT_SECRET)
        self.assertEqual(os.environ["GOOGLE_CALENDAR_CLIENT_SECRET"], CLIENT_SECRET)   # live, no restart
        self.assertTrue(self.service.status()["configured"])
        self.assertNotIn(CLIENT_SECRET, " ".join(self.logged))

    def test_the_rest_of_the_env_file_survives(self):
        self.save_google()
        self.save_flights()
        text = self.env.read_text(encoding="utf-8")
        self.assertIn("# Pam config", text)
        self.assertIn("# comment", text)
        self.assertEqual(self.env_values()["DEEPGRAM_API_KEY"], "keep-me")
        self.assertEqual(self.env_values()["HOME_LAT"], "42.3601")
        self.assertTrue(text.endswith("\n"))

    def test_saving_twice_replaces_rather_than_duplicates(self):
        self.save_google()
        self.save_google(client_id="999-second.apps.googleusercontent.com", secret="second-secret")
        lines = [ln for ln in self.env.read_text(encoding="utf-8").splitlines() if ln.startswith("GOOGLE_CALENDAR_CLIENT_ID=")]
        self.assertEqual(lines, ["GOOGLE_CALENDAR_CLIENT_ID=999-second.apps.googleusercontent.com"])
        self.assertEqual(os.environ["GOOGLE_CALENDAR_CLIENT_SECRET"], "second-secret")

    def test_a_missing_env_file_is_created_privately(self):
        self.env.unlink()
        self.save_google()
        self.assertEqual(self.env_values()["GOOGLE_CALENDAR_CLIENT_ID"], CLIENT_ID)
        if os.name == "nt":
            script = ("$acl = Get-Acl -LiteralPath $env:PAM_TEST_ENV; $sid=[System.Security.Principal.SecurityIdentifier];"
                      "@{owner=$acl.GetOwner($sid).Value; user=[System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value;"
                      "protected=$acl.AreAccessRulesProtected; count=@($acl.GetAccessRules($true,$true,$sid)).Count} | ConvertTo-Json -Compress")
            result = json.loads(subprocess.check_output(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
                                env={**os.environ, "PAM_TEST_ENV": str(self.env)}, text=True))
            self.assertEqual(result["owner"], result["user"])
            self.assertTrue(result["protected"])
            self.assertEqual(result["count"], 1)
        else:
            self.assertEqual(os.stat(self.env).st_mode & 0o777, 0o600)

    def test_flight_key_is_checked_against_the_provider_before_saving(self):
        response, seen = self.save_flights()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.env_values()["SERPAPI_KEY"], FLIGHT_KEY)
        self.assertEqual(seen[0].url.host, "serpapi.com")
        self.assertEqual(seen[0].url.path, "/account")
        self.assertNotIn(FLIGHT_KEY, json.dumps(response.json()))

    def test_a_rejected_flight_key_is_not_saved(self):
        response, _ = self.save_flights(status=401, payload={"error": "Invalid API key"})
        self.assertEqual(response.status_code, 400)
        self.assertNotIn("SERPAPI_KEY", self.env_values())
        self.assertEqual(os.environ.get("SERPAPI_KEY"), "")
        self.assertNotIn(FLIGHT_KEY, json.dumps(response.json()))

    def test_provider_outage_does_not_silently_discard_the_key(self):
        original = httpx.AsyncClient
        def fail(request):
            raise httpx.ConnectError("offline")
        with mock.patch.object(st.httpx, "AsyncClient", side_effect=lambda **kw: original(transport=httpx.MockTransport(fail), **kw)):
            response = self.client.post("/api/setup/flights", json={"api_key": FLIGHT_KEY})
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("SERPAPI_KEY", self.env_values())

    def test_obviously_wrong_values_are_refused_without_touching_the_file(self):
        before = self.env.read_text(encoding="utf-8")
        for payload in ({"client_id": "", "client_secret": CLIENT_SECRET},
                        {"client_id": CLIENT_ID, "client_secret": ""},
                        {"client_id": "not-a-google-client", "client_secret": CLIENT_SECRET},
                        {"client_id": CLIENT_ID, "client_secret": "has spaces"},
                        {"client_id": CLIENT_ID + "\nINJECTED=1", "client_secret": CLIENT_SECRET}):
            self.assertEqual(self.client.post("/api/setup/google", json=payload).status_code, 400, payload)
        for payload in ({}, {"api_key": ""}, {"api_key": "short"}, {"api_key": "bad key with spaces"}):
            self.assertEqual(self.client.post("/api/setup/flights", json=payload).status_code, 400, payload)
        self.assertEqual(self.env.read_text(encoding="utf-8"), before)


class Access(Base):
    def test_status_reports_what_is_configured_not_the_values(self):
        self.save_google()
        body = self.client.get("/api/setup/status").json()
        self.assertEqual(body["calendar"], {"configured": True, "connected": False})
        self.assertFalse(body["flights"]["configured"])
        self.assertNotIn(CLIENT_SECRET, json.dumps(body))
        self.assertNotIn(CLIENT_ID, json.dumps(body))

    def test_setup_is_laptop_only(self):
        with TestClient(st.app_for_tests(), base_url="http://10.0.0.7:8000", client=("10.0.0.7", 51000)) as remote:
            self.assertEqual(remote.get("/setup").status_code, 403)
            self.assertEqual(remote.get("/api/setup/status").status_code, 403)
            self.assertEqual(remote.post("/api/setup/google", json={"client_id": CLIENT_ID, "client_secret": CLIENT_SECRET}).status_code, 403)
            self.assertEqual(remote.post("/api/setup/flights", json={"api_key": FLIGHT_KEY}).status_code, 403)
        self.assertEqual(self.env.read_text(encoding="utf-8"), EXISTING)

    def test_the_page_is_served_and_never_caches(self):
        response = self.client.get("/setup")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertIn("Google Calendar", response.text)
        self.assertNotIn(CLIENT_SECRET, response.text)


class Startup(Base):
    def test_startup_opens_setup_when_anything_is_missing(self):
        opened, said = [], []
        self.assertTrue(st.start(opened.append, said.append))
        self.assertEqual(opened, ["http://127.0.0.1:8000/setup"])

    def test_startup_goes_straight_to_google_consent_once_credentials_exist(self):
        self.save_google()
        self.save_flights()
        opened, said = [], []
        self.assertTrue(st.start(opened.append, said.append))
        self.assertEqual(opened, ["http://127.0.0.1:8000" + gc.CONNECT_PATH])

    def test_startup_is_quiet_once_everything_is_connected(self):
        self.save_google()
        self.save_flights()
        self.service.credentials = object()
        opened, said = [], []
        self.assertFalse(st.start(opened.append, said.append))
        self.assertEqual(opened, [])
        self.assertTrue(any("connected" in message.lower() for message in said))

    def test_startup_never_raises_when_there_is_no_browser(self):
        def refuse(url):
            raise OSError("no browser")
        said = []
        self.assertFalse(st.start(refuse, said.append))
        self.assertTrue(any("/setup" in message for message in said))


if __name__ == "__main__":
    unittest.main(verbosity=2)
