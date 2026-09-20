"""Tests for caregiver.py: the shared-PIN gate. No network, browser or real .env needed.

    python server/test_caregiver.py

The ones that matter most: it must fail CLOSED (no PIN configured means no access), a wrong
or missing PIN must never reveal data, and the PIN itself must never come back out.
"""
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import caregiver as c  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

PIN = "correct-horse-7"
PROTECTED = "/api/caregiver/status"


def make_client(base_url="http://testserver"):
    app = FastAPI()
    app.include_router(c.router)
    return TestClient(app, base_url=base_url)


class Base(unittest.TestCase):
    def setUp(self):
        self.now = 1_000_000.0
        self.patches = [mock.patch.object(c, "_now", lambda: self.now),
                        mock.patch.object(c, "_log", lambda msg: None),
                        mock.patch.dict(os.environ, {"CAREGIVER_PIN": PIN}, clear=False)]
        for p in self.patches:
            p.start()
        c._sessions.clear()
        c._failures.clear()
        self.client = make_client()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        c._sessions.clear()
        c._failures.clear()

    def login(self, pin=PIN, client=None):
        return (client or self.client).post("/api/caregiver/login", json={"pin": pin})


class FailsClosed(Base):
    def test_no_pin_configured_means_off(self):
        os.environ.pop("CAREGIVER_PIN")
        self.assertFalse(c.enabled())
        self.assertEqual(self.client.get("/api/caregiver/me").json(), {"enabled": False, "authenticated": False})
        self.assertEqual(self.login().status_code, 404)
        self.assertEqual(self.client.get(PROTECTED).status_code, 404)

    def test_an_empty_pin_is_off_not_open(self):
        os.environ["CAREGIVER_PIN"] = ""
        self.assertFalse(c.enabled())
        self.assertEqual(self.login("").status_code, 404)          # "" must not match ""
        self.assertEqual(self.client.get(PROTECTED).status_code, 404)

    def test_a_pin_that_is_too_short_is_off(self):
        os.environ["CAREGIVER_PIN"] = "12345"                       # one under the minimum
        self.assertFalse(c.enabled())
        self.assertEqual(self.login("12345").status_code, 404)
        os.environ["CAREGIVER_PIN"] = "123456"
        self.assertTrue(c.enabled())

    def test_the_page_itself_is_served_even_when_off_and_holds_no_data(self):
        os.environ.pop("CAREGIVER_PIN")
        r = self.client.get("/caregiver")
        self.assertEqual(r.status_code, 200)
        self.assertIn("isn't set up yet", r.text)


class LoggingIn(Base):
    def test_right_pin_signs_in_and_opens_protected_routes(self):
        self.assertEqual(self.client.get(PROTECTED).status_code, 401)   # before
        self.assertEqual(self.login().status_code, 200)
        self.assertEqual(self.client.get(PROTECTED).status_code, 200)   # after
        self.assertEqual(self.client.get("/api/caregiver/me").json(), {"enabled": True, "authenticated": True})

    def test_wrong_pin_is_refused_and_opens_nothing(self):
        r = self.login("wrong-pin-99")
        self.assertEqual(r.status_code, 401)
        self.assertNotIn("set-cookie", r.headers)
        self.assertEqual(self.client.get(PROTECTED).status_code, 401)

    def test_missing_or_odd_pin_values_are_refused(self):
        for body in ({}, {"pin": None}, {"pin": ""}, {"pin": 123456}, {"pin": [PIN]}):
            r = self.client.post("/api/caregiver/login", json=body)
            self.assertEqual(r.status_code, 401, body)

    def test_a_forged_or_guessed_cookie_does_not_work(self):
        self.client.cookies.set(c.COOKIE, "not-a-real-token")
        self.assertEqual(self.client.get(PROTECTED).status_code, 401)

    def test_pin_is_compared_in_constant_time(self):
        with mock.patch.object(c.hmac, "compare_digest", wraps=c.hmac.compare_digest) as spy:
            self.login("anything-at-all")
            self.assertTrue(spy.called)

    def test_the_pin_never_comes_back_out(self):
        for r in (self.login(), self.login("wrong-pin-99"), self.client.get("/api/caregiver/me"),
                  self.client.get(PROTECTED), self.client.get("/caregiver")):
            self.assertNotIn(PIN, r.text)
            self.assertNotIn(PIN, str(r.headers))

    def test_a_submitted_wrong_pin_is_not_echoed_back(self):
        for guess in ("wrong-pin-99", "<script>alert(1)</script>"):
            r = self.login(guess)
            self.assertNotIn(guess, r.text)
            self.assertNotIn(guess, str(r.headers))

    def test_a_pin_in_the_url_is_not_accepted(self):
        r = self.client.post(f"/api/caregiver/login?pin={PIN}", json={})
        self.assertEqual(r.status_code, 401)


class Cookie(Base):
    def test_cookie_is_httponly_samesite_strict(self):
        header = self.login().headers["set-cookie"].lower()
        self.assertIn("httponly", header)
        self.assertIn("samesite=strict", header)
        self.assertIn("path=/", header)

    def test_cookie_is_secure_only_over_https(self):
        self.assertNotIn("secure", self.login().headers["set-cookie"].lower().replace("samesite", ""))
        https = make_client("https://testserver")
        self.assertIn("secure", self.login(client=https).headers["set-cookie"].lower())

    def test_token_is_long_random_and_different_each_time(self):
        tokens = []
        for _ in range(3):
            cl = make_client()
            self.login(client=cl)
            tokens.append(cl.cookies.get(c.COOKIE))
        self.assertEqual(len(set(tokens)), 3)
        self.assertTrue(all(len(t) >= 32 for t in tokens))


class Sessions(Base):
    def test_session_expires(self):
        self.login()
        self.now += c.SESSION_TTL_S - 1
        self.assertEqual(self.client.get(PROTECTED).status_code, 200)
        self.now += 2
        self.assertEqual(self.client.get(PROTECTED).status_code, 401)
        self.assertEqual(c._sessions, {})                           # and it was swept, not just ignored

    def test_logout_ends_the_session_on_the_server(self):
        self.login()
        token = self.client.cookies.get(c.COOKIE)
        self.assertEqual(self.client.post("/api/caregiver/logout").status_code, 200)
        self.assertNotIn(token, c._sessions)
        self.client.cookies.set(c.COOKIE, token)                    # replaying the old token must fail
        self.assertEqual(self.client.get(PROTECTED).status_code, 401)

    def test_sessions_are_tokens_not_the_pin(self):
        # Sessions are tokens, not the PIN. Documented behaviour: a restart is what clears them.
        self.login()
        os.environ["CAREGIVER_PIN"] = "a-different-pin-1"
        self.assertEqual(self.client.get(PROTECTED).status_code, 200)

    def test_turning_the_feature_off_closes_open_sessions(self):
        self.login()
        os.environ.pop("CAREGIVER_PIN")
        self.assertEqual(self.client.get(PROTECTED).status_code, 404)


class Lockout(Base):
    def test_five_wrong_pins_lock_out_even_the_right_one(self):
        for _ in range(c.MAX_FAILURES):
            self.assertEqual(self.login("wrong-pin-99").status_code, 401)
        r = self.login()                                            # the RIGHT pin, now refused
        self.assertEqual(r.status_code, 429)
        self.assertGreater(r.json()["retry_after_s"], 0)
        self.assertIn("retry-after", r.headers)
        self.assertEqual(self.client.get(PROTECTED).status_code, 401)

    def test_four_wrong_pins_do_not_lock(self):
        for _ in range(c.MAX_FAILURES - 1):
            self.login("wrong-pin-99")
        self.assertEqual(self.login().status_code, 200)

    def test_lockout_ends_after_the_window(self):
        for _ in range(c.MAX_FAILURES):
            self.login("wrong-pin-99")
        self.now += c.LOCKOUT_WINDOW_S + 1
        self.assertEqual(self.login().status_code, 200)

    def test_a_success_clears_the_failure_count(self):
        for _ in range(c.MAX_FAILURES - 1):
            self.login("wrong-pin-99")
        self.login()
        for _ in range(c.MAX_FAILURES - 1):                         # four more must still not lock
            self.login("wrong-pin-99")
        self.assertEqual(self.login().status_code, 200)

    def test_lockout_holds_while_a_guesser_keeps_trying(self):
        for _ in range(c.MAX_FAILURES):
            self.login("wrong-pin-99")
        for _ in range(20):
            self.now += 5
            self.assertEqual(self.login("still-wrong-1").status_code, 429)


class Status(Base):
    def test_status_reports_switches_not_data(self):
        self.login()
        body = self.client.get(PROTECTED).json()
        self.assertEqual(set(body), {"medication_check", "demo_timings", "schedule_reminders", "schedule",
                                     "schedule_version", "scheduler_running", "elasticsearch"})
        for key in ("medication_check", "demo_timings", "schedule_reminders", "scheduler_running", "elasticsearch"):
            self.assertIsInstance(body[key], bool, key)
        self.assertIn(body["schedule"], {"none", "active", "damaged", "error", "off"})


if __name__ == "__main__":
    unittest.main(verbosity=1)
