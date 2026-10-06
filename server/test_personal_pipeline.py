"""Offline personal-memory integration and security regressions.

    perception/.venv/Scripts/python.exe -B server/test_personal_pipeline.py -v

Synthetic rectangles, temporary databases/JPEGs, injected verifiers, and ASGI only.
No camera, model, real server, real credentials, TLS material, or provider calls.
app.py is executed under file-read guards with synthetic configuration. The phone
file handler receives BytesIO requests, not a listening socket. Confidence scores
are fabricated gate inputs, not estimates of real-world model accuracy.
"""
from __future__ import annotations

import asyncio
import base64
import copy
import functools
import importlib
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

import cv2
import httpx
import numpy as np
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "server"))
from perception.capture import decode_frame, encode_frame
from perception.object_memory import ObjectStore, Verification
from perception import personal_memory as pm


TRACK = "11111111-1111-4111-8111-111111111111"
CAMERA = "synthetic-camera-session-001"
PIN = "synthetic-pin-123456"
LONG_KEY = "synthetic-long-lived-secret-DO-NOT-RETURN"
LEGACY_MARKER = "SYNTHETIC-LEGACY-LOCATION-MUST-NOT-LEAK"
CONTACTS = {"user": "Synthetic user", "contacts": [], "places": {}, "home_airport": "BOS",
            "caregiver": {"name": "Synthetic helper"}}


def verification(action="placed", identity="new", matched=None, actor="wearer", **changes):
    fields = {
        "actor": actor, "actor_confidence": .99,
        "actor_evidence": ["Synthetic wearer contact and coherent motion"],
        "action": action, "action_confidence": .99, "contact_observed": True,
        "released": action == "placed", "resting_after": action == "placed",
        "object": "pill bottle", "appearance": "Synthetic red diagonal marking",
        "identity": identity, "matched_object_id": matched, "identity_confidence": .99,
        "identity_basis": "continuous_track" if matched else "distinctive_features",
        "identity_evidence": "Synthetic uninterrupted segment" if matched else "Synthetic unique diagonal marking",
        "scene": {"surface": "synthetic table", "description": "On the synthetic table beside a test cube.",
                  "landmarks": ["test cube"]},
    }
    fields.update(changes)
    return Verification.model_validate(fields)


class FakeVerifier:
    def __init__(self, actor="wearer"):
        self.actor = actor
        self.prepared, self.verified = [], []
        self.prepare_error = self.verify_error = None

    def prepare(self, candidate, gallery):
        self.prepared.append(copy.deepcopy((candidate, gallery)))
        if self.prepare_error is not None:
            raise self.prepare_error
        return {"candidate": copy.deepcopy(candidate), "gallery": copy.deepcopy(gallery)}

    def verify(self, content):
        self.verified.append(copy.deepcopy(content))
        if self.verify_error is not None:
            raise self.verify_error
        candidate, gallery = content["candidate"], content["gallery"]
        continuous = [obj for obj in gallery if obj["continuous_track"]]
        matched = continuous[0]["object_id"] if len(continuous) == 1 else None
        identity = "match" if matched else ("new" if candidate["event_type"] == "placed" else "uncertain")
        return (verification(candidate["event_type"], identity, matched, self.actor), .004,
                {"model": "synthetic-verifier", "input_tokens": 1000, "output_tokens": 200, "latency_s": .01})


class IsolatedCase(unittest.TestCase):
    asgi = False

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="compass-personal-integration-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.now = time.time()
        self.enterContext(patch("httpx.HTTPTransport.handle_request", side_effect=AssertionError("External HTTP forbidden")))
        self.enterContext(patch("httpx.AsyncHTTPTransport.handle_async_request",
                                new=AsyncMock(side_effect=AssertionError("External HTTP forbidden"))))
        self.enterContext(patch("urllib.request.urlopen", side_effect=AssertionError("External URL forbidden")))
        self.enterContext(patch("socket.create_connection", side_effect=AssertionError("Network forbidden")))
        self.enterContext(patch("subprocess.run", side_effect=AssertionError("Subprocess forbidden")))
        self.enterContext(patch("subprocess.Popen", side_effect=AssertionError("Subprocess forbidden")))
        if not self.asgi:
            self.block_socket_connects()

    def block_socket_connects(self):
        # An ASGI portal is created first: Windows asyncio may construct its own
        # socketpair internally. After that even loopback outbound connects fail.
        self.enterContext(patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden")))
        self.enterContext(patch.object(socket.socket, "connect_ex", side_effect=AssertionError("Network forbidden")))

    def jpeg(self, path, shape=(120, 160, 3), value=80):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        image = np.zeros(shape, dtype=np.uint8)
        image[10:30, 15:35] = value
        self.assertTrue(cv2.imwrite(str(path), image))
        return path

    def candidate(self, artifact_root, *, at=None, action="placed", track=TRACK, frames=None, **changes):
        at = self.now if at is None else at
        event_id = str(uuid.uuid4())
        directory = Path(artifact_root) / event_id
        if frames is None:
            frames = [str(self.jpeg(directory / f"frame_{n}.jpg", value=60 + n * 50)) for n in range(3)]
        crop = str(self.jpeg(directory / "crop.jpg", shape=(40, 40, 3)))
        out = {
            "event_id": event_id, "session_id": CAMERA, "track_id": track,
            "observed_at": float(at), "time_source": "capture", "sequence": 1,
            "event_type": action, "object": "pill bottle",
            "evidence": {"contact_candidate": True, "motion_valid": True, "released": action == "placed",
                         "track_continuous": True, "track_started_at": float(self.now - 10)},
            "frames": frames, "frame_times": [float(at - 1), float(at - .5), float(at)], "crop": crop,
            "co_visible_track_ids": [],
            "location": {"lat": 1.0, "lon": 2.0, "accuracy_m": 8.0, "observed_at": float(at - 5)},
        }
        out.update(changes)
        return out


class PipelineFixture(IsolatedCase):
    def setUp(self):
        super().setUp()
        self.notices = []
        self.fake = FakeVerifier()
        self.pipe = pm.PersonalMemory(self.root / "objects.sqlite3", fps=5, verifier=self.fake,
                                      notify=self.notices.append, start_worker=False)
        self.pipe.store.clock = lambda: self.now
        self.addCleanup(self.pipe.close)
        self.start = self.now - 30
        self.frame_number = 0
        self.x = 20

    def feed(self, x=None, contact=False, *, session=CAMERA, source="capture", static=True,
             homography=None, raw_id=7, present=True, gap=0.0):
        self.start += gap
        at = self.start + self.frame_number * .2
        self.frame_number += 1
        self.x = self.x if x is None else x
        processed = np.zeros((120, 160, 3), dtype=np.uint8)
        box = np.array([self.x, 45, self.x + 20, 65], dtype=np.float32)
        if present:
            cv2.rectangle(processed, tuple(box[:2].astype(int)), tuple(box[2:].astype(int)), (180, 180, 180), -1)
        original = cv2.resize(processed, (320, 240))
        arms = [np.array([self.x - 2, 43, self.x + 22, 120], dtype=np.float32)] if contact else []
        metadata = {"session_id": session, "frame_id": self.frame_number, "captured_at": at,
                    "time_source": source,
                    "location": {"lat": 1.0, "lon": 2.0, "accuracy_m": 8.0, "observed_at": self.start - 1}}
        self.pipe.process_frame(original, processed, [box] if present else [],
                                ["pill bottle"] if present else [], [raw_id] if present else [],
                                arms, homography, static, metadata, ["pill bottle"])
        return at

    def held_motion(self, start_x=20, count=6, **kwargs):
        for index in range(count):
            self.feed(start_x + index * 10, contact=True, **kwargs)

    def release(self, count=6, **kwargs):
        for _ in range(count):
            self.feed(contact=False, **kwargs)

    def first_placement(self):
        self.held_motion()
        self.release()
        self.assertEqual(self.pipe.events, 1)
        return next(n for n in self.notices if n.get("status") == "queued")


class SyntheticFrameTests(PipelineFixture):
    def test_untouched_shelf_and_background_motion_create_no_events_or_artifacts(self):
        for _ in range(8):
            self.feed()
        for x in range(20, 81, 10):
            self.feed(x)
        self.release(30)
        self.assertEqual(self.pipe.events, 0)
        self.assertEqual(self.pipe.store.usage()["events"], {})
        self.assertEqual(self.pipe.store.list_objects(include_pending=True), [])
        self.assertEqual(list(self.pipe.artifacts.iterdir()), [])
        self.assertFalse((self.root / "last_seen").exists())
        self.assertFalse((self.root / "memory.jsonl").exists())
        self.assertFalse(self.pipe.run_once())
        self.assertEqual(self.fake.verified, [])

    def test_rest_seen_then_hand_motion_and_release_emit_pickup_and_placement(self):
        self.release(6)
        self.held_motion(start_x=30, count=5)
        self.release()
        events = [n for n in self.notices if n.get("status") == "queued"]
        self.assertEqual([e["event_type"] for e in events], ["picked_up", "placed"])
        self.assertEqual(events[0]["track_id"], events[1]["track_id"])
        self.assertNotEqual(events[0]["track_id"], "7")
        self.assertLess(events[0]["observed_at"], events[1]["observed_at"])
        self.assertLess(events[0]["sequence"], events[1]["sequence"])
        for event in events:
            self.assertEqual(event["time_source"], "capture")
            self.assertTrue(event["evidence"]["track_continuous"])
            self.assertEqual(event["frame_times"], sorted(set(event["frame_times"])))
            self.assertGreaterEqual(len(event["frames"]), 2)
            self.assertLessEqual(len(event["frames"]), 5)
            self.assertEqual(len(event["frames"]), len(event["frame_times"]))
            for name in event["frames"] + [event["crop"]]:
                self.assertTrue(Path(name).resolve().is_relative_to(self.pipe.artifacts))
                self.assertTrue(Path(name).is_file())
        self.assertFalse((self.root / "last_seen").exists())

    def test_first_seen_in_hand_places_without_inventing_a_pickup_and_keeps_full_resolution(self):
        event = self.first_placement()
        self.assertEqual(event["event_type"], "placed")
        self.assertEqual(cv2.imread(event["frames"][-1]).shape[:2], (240, 320))
        self.assertEqual(cv2.imread(event["crop"]).shape[:2], (40, 40))
        self.release(20)
        self.assertEqual(self.pipe.events, 1, "Rest is not a recurring snapshot event")

    def test_actual_candidate_to_verifier_to_confirmation_pickup_and_replace(self):
        event = self.first_placement()
        self.assertTrue(self.pipe.run_once())
        self.assertEqual(len(self.fake.verified), 1)
        prepared, gallery = self.fake.prepared[0]
        self.assertTrue(prepared["gallery_complete"])
        self.assertIsNotNone(prepared["call_id"])
        self.assertEqual(gallery, [])
        result = self.pipe.store.event(event["event_id"])
        self.assertEqual(result["reason"], "confirm_tracking")
        self.assertEqual(self.pipe.store.search("pill bottle"), [])
        object_id = result["object_id"]
        self.pipe.store.confirm_object(object_id, name="Test bottle")
        self.held_motion(start_x=80, count=4)
        self.assertEqual(self.pipe.store.get_object(object_id)["state"], "uncertain")
        self.assertIsNone(self.pipe.store.get_object(object_id)["location"])
        self.assertTrue(self.pipe.run_once())
        self.assertEqual(self.pipe.store.get_object(object_id)["state"], "carried")
        self.release()
        self.assertTrue(self.pipe.run_once())
        obj = self.pipe.store.get_object(object_id)
        self.assertEqual(obj["state"], "placed")
        self.assertEqual(len(self.pipe.store.list_objects()), 1)
        self.assertEqual([h["action"] for h in self.pipe.store.history(object_id)], ["placed", "picked_up", "placed"])
        self.assertEqual(self.pipe.calls, 3)
        self.assertEqual((self.pipe.input_tokens, self.pipe.output_tokens), (3000, 600))
        self.assertAlmostEqual(self.pipe.store.usage()["cost_or_reserved_usd"], .012)

    def test_bystander_verification_rejects_even_a_strong_detector_proposal(self):
        self.fake.actor = "bystander"
        event = self.first_placement()
        self.assertTrue(self.pipe.run_once())
        self.assertEqual(self.pipe.store.event(event["event_id"])["status"], "rejected")
        self.assertEqual(self.pipe.store.list_objects(include_pending=True), [])

    def test_legacy_received_time_frames_are_not_saved_or_verified(self):
        for source in ("received", None):
            self.held_motion(source=source)
            self.release(source=source)
        self.assertEqual(self.pipe.events, 0)
        self.assertEqual(list(self.pipe.artifacts.iterdir()), [])
        self.assertEqual(self.pipe.store.usage()["api_calls"], 0)

    def test_session_change_discards_an_incomplete_hand_episode(self):
        self.held_motion()
        first_segment = self.pipe.tracks[7].segment_id
        self.release(session="synthetic-camera-session-002")
        self.assertNotEqual(self.pipe.tracks[7].segment_id, first_segment)
        self.assertEqual(self.pipe.events, 0)
        self.assertEqual(list(self.pipe.artifacts.iterdir()), [])

    def test_missing_ego_motion_fit_is_not_evidence_of_release_and_rest(self):
        self.held_motion()
        self.release(static=False, homography=None)
        self.assertEqual(self.pipe.events, 0)
        self.assertEqual(list(self.pipe.artifacts.iterdir()), [])

    def test_camera_motion_compensation_does_not_arm_a_stationary_held_object(self):
        translated = np.array([[1, 0, 10], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        self.held_motion(static=False, homography=translated)
        self.release(static=False, homography=np.eye(3, dtype=np.float32))
        self.assertEqual(self.pipe.events, 0)

    def test_gap_with_reused_raw_tracker_id_does_not_complete_old_placement(self):
        self.held_motion()
        old_segment = self.pipe.tracks[7].segment_id
        self.feed(gap=3.0)
        self.release()
        self.assertNotEqual(self.pipe.tracks[7].segment_id, old_segment)
        self.assertEqual(self.pipe.events, 0)


class WorkerBoundaryTests(PipelineFixture):
    def test_cloud_disabled_does_not_construct_a_client_or_spend_budget(self):
        with patch.object(pm, "EventVerifier", side_effect=AssertionError("Cloud client must remain unconstructed")):
            offline = pm.PersonalMemory(self.root / "offline" / "objects.sqlite3", allow_cloud=False, start_worker=False)
        self.addCleanup(offline.close)
        offline.store.enqueue(self.candidate(offline.artifacts))
        self.assertFalse(offline.run_once())
        self.assertEqual(offline.store.usage()["api_calls"], 0)
        self.assertIsNone(offline.thread)

    def test_budget_wait_does_not_prepare_or_call_the_provider(self):
        event = self.first_placement()
        self.pipe.daily_calls = 0
        self.assertFalse(self.pipe.run_once())
        self.assertEqual(self.fake.prepared, [])
        self.assertEqual(self.fake.verified, [])
        self.assertEqual(self.pipe.store.event(event["event_id"])["status"], "budget_wait")
        self.assertEqual(self.pipe.store.usage()["api_calls"], 0)

    def test_local_prepare_failure_has_no_provider_charge_and_remains_reviewable(self):
        event = self.first_placement()
        self.fake.prepare_error = ValueError("synthetic bad artifact")
        self.assertTrue(self.pipe.run_once())
        self.assertEqual(self.fake.verified, [])
        self.assertEqual(self.pipe.store.event(event["event_id"])["status"], "failed")
        self.assertAlmostEqual(self.pipe.store.usage()["cost_or_reserved_usd"], 0)
        self.assertIn(event["event_id"], [e["event_id"] for e in self.pipe.store.reviews()])

    def test_provider_failure_preserves_reserved_cost_and_never_saves_a_location(self):
        event = self.first_placement()
        self.fake.verify_error = RuntimeError("synthetic provider outage")
        self.assertTrue(self.pipe.run_once())
        self.assertEqual(len(self.fake.verified), 1)
        self.assertEqual(self.pipe.store.event(event["event_id"])["status"], "failed")
        self.assertAlmostEqual(self.pipe.store.usage()["cost_or_reserved_usd"], .10)
        self.assertEqual(self.pipe.store.list_objects(include_pending=True), [])

    def test_three_failures_pause_worker_without_losing_queued_candidates(self):
        self.fake.verify_error = RuntimeError("synthetic unavailable service")
        for n in range(4):
            self.pipe.store.enqueue(self.candidate(self.pipe.artifacts, at=self.now + n, track=str(uuid.uuid4())))
        for _ in range(3):
            self.assertTrue(self.pipe.run_once())
        self.assertTrue(self.pipe.paused)
        self.assertFalse(self.pipe.run_once())
        self.assertEqual(len(self.fake.verified), 3)
        usage = self.pipe.store.usage()
        self.assertEqual(usage["api_calls"], 3)
        self.assertEqual(usage["events"].get("queued"), 1)
        self.assertEqual(sum(usage["events"].values()), 4)


class EventVerifierTests(IsolatedCase):
    def setUp(self):
        super().setUp()
        self.artifacts = self.root / "object_evidence"
        self.event = self.candidate(self.artifacts)
        self.client = SimpleNamespace(messages=SimpleNamespace(parse=MagicMock()))
        self.verifier = pm.EventVerifier(self.artifacts, client=self.client)

    def test_prepare_uses_only_temporal_frames_and_approved_target_and_gallery_crops(self):
        reference = self.jpeg(self.artifacts / "reference.jpg", shape=(600, 800, 3))
        content = self.verifier.prepare({**self.event, "gallery_complete": True}, [{
            "object_id": TRACK, "category": "pill bottle", "continuous_track": True, "crop": str(reference)}])
        images = [block for block in content if block["type"] == "image"]
        self.assertEqual(len(images), 5)
        text = " ".join(block["text"] for block in content if block["type"] == "text")
        self.assertIn(TRACK, text)
        self.assertIn("captured_at=", text)
        self.assertNotIn(self.root.name, text)
        decoded = cv2.imdecode(np.frombuffer(base64.b64decode(images[-1]["source"]["data"]), np.uint8), cv2.IMREAD_COLOR)
        self.assertLessEqual(max(decoded.shape[:2]), 384)
        self.client.messages.parse.assert_not_called()

    def test_prepare_rejects_frame_target_crop_and_gallery_paths_outside_artifact_root(self):
        outside = self.jpeg(self.root / "outside.jpg")
        lookalike = self.jpeg(self.root / "object_evidence-private" / "private.jpg")
        for forbidden in (str(outside), str(self.artifacts / ".." / "outside.jpg"), str(lookalike)):
            cases = [({**self.event, "frames": [self.event["frames"][0], forbidden]}, []),
                     ({**self.event, "crop": forbidden}, []),
                     (self.event, [{"object_id": TRACK, "category": "pill bottle", "continuous_track": False, "crop": forbidden}])]
            for candidate, gallery in cases:
                with self.subTest(path=forbidden, gallery=bool(gallery)), self.assertRaises(ValueError):
                    self.verifier.prepare(candidate, gallery)
        self.client.messages.parse.assert_not_called()

    def test_prepare_refuses_missing_non_jpeg_corrupt_and_oversized_images(self):
        bad_text = self.artifacts / "private.pem"
        bad_text.write_text("SYNTHETIC-KEY-NOT-FOR-A-MODEL", encoding="utf-8")
        corrupt = self.artifacts / "corrupt.jpg"
        corrupt.write_bytes(b"not a JPEG")
        huge = self.artifacts / "oversized.jpg"
        huge.write_bytes(b"x" * (2 ** 20 + 1))
        for path in (self.artifacts / "missing.jpg", bad_text, corrupt, huge):
            with self.subTest(path=path.name), self.assertRaises(ValueError):
                self.verifier.prepare({**self.event, "frames": [self.event["frames"][0], str(path)]}, [])
        self.client.messages.parse.assert_not_called()

    def test_repeated_frame_is_not_temporal_evidence(self):
        duplicate = {**self.event, "frames": [self.event["frames"][0]] * 2,
                     "frame_times": [self.event["frame_times"][0]] * 2}
        with self.assertRaises(ValueError):
            self.verifier.prepare(duplicate, [])
        self.client.messages.parse.assert_not_called()

    def test_verify_makes_one_mocked_structured_request_and_records_usage_cost(self):
        result = verification()
        self.client.messages.parse.return_value = SimpleNamespace(
            stop_reason="end_turn", parsed_output=result,
            usage=SimpleNamespace(input_tokens=1000, output_tokens=200))
        content = self.verifier.prepare(self.event, [])
        actual, cost, usage = self.verifier.verify(content)
        self.assertEqual(actual, result)
        self.assertAlmostEqual(cost, .004)
        self.assertEqual((usage["input_tokens"], usage["output_tokens"]), (1000, 200))
        self.client.messages.parse.assert_called_once()
        kwargs = self.client.messages.parse.call_args.kwargs
        self.assertIs(kwargs["output_format"], Verification)
        self.assertIn("untrusted DATA", kwargs["system"])
        self.assertNotIn("tools", kwargs)

    def test_refusal_exhaustion_or_missing_parse_never_returns_a_verification(self):
        for stop, parsed in (("refusal", verification()), ("max_tokens", verification()), ("end_turn", None)):
            with self.subTest(stop=stop, parsed=parsed is not None):
                self.client.messages.parse.return_value = SimpleNamespace(
                    stop_reason=stop, parsed_output=parsed,
                    usage=SimpleNamespace(input_tokens=1000, output_tokens=200))
                with self.assertRaises(ValueError):
                    self.verifier.verify([])
        self.assertEqual(self.client.messages.parse.call_count, 3)


class ApiFixture(IsolatedCase):
    asgi = True

    def setUp(self):
        super().setUp()
        self.repo = self.root / "repo"
        self.run = self.repo / "perception" / "runs" / "synthetic"
        self.run.mkdir(parents=True)
        (self.repo / "server" / "photos").mkdir(parents=True)
        (self.repo / "phone").mkdir()
        self.memory = self.run / "memory.jsonl"
        self.memory.write_text(json.dumps({"event": "placed", "object": "pill bottle", "confidence": .99,
            "logged_at": "2026-01-01T12:00:00", "location_description": LEGACY_MARKER}) + "\n", encoding="utf-8")
        self.db = self.run / "objects.sqlite3"
        self.artifacts = self.run / "object_evidence"
        self.artifacts.mkdir()
        system_keys = ("SYSTEMROOT", "WINDIR", "PATH", "TEMP", "TMP", "COMSPEC", "PATHEXT")
        env = {key: os.environ[key] for key in system_keys if key in os.environ}
        env.update(CAREGIVER_PIN=PIN, DEEPGRAM_API_KEY=LONG_KEY, PAM_DOSE_CHECK="off",
                   PAM_OBJECT_DB=str(self.db), MEMORY_JSONL=str(self.memory), HOME_LAT="0", HOME_LON="0",
                   LOCALAPPDATA=str(self.root / "appdata"), APPDATA=str(self.root / "appdata"),
                   USERPROFILE=str(self.root / "home"))
        self.enterContext(patch.dict(os.environ, env, clear=True))
        self.install_read_guards()
        self.store = ObjectStore(self.db, clock=lambda: self.now)
        module_name = "_synthetic_personal_app_" + uuid.uuid4().hex
        spec = importlib.util.spec_from_file_location(module_name, ROOT / "server" / "app.py")
        self.app = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = self.app
        self.addCleanup(sys.modules.pop, module_name, None)
        spec.loader.exec_module(self.app)
        self.caregiver = self.app.caregiver
        self.enterContext(patch.multiple(self.app, ROOT=self.repo, HERE=self.repo / "server", PHONE=self.repo / "phone",
            MEMORY_JSONL=self.memory, REMINDERS=self.run / "reminders.jsonl", CONTACTS=copy.deepcopy(CONTACTS),
            CERT=self.root / "synthetic-cert.pem", KEY=self.root / "synthetic-key.pem", _subscribers=set(),
            _last_fix={}, _last_frame=None, _last_frame_ts=0.0, _active_personal_camera=None))
        self.enterContext(patch.multiple(self.caregiver, _sessions={}, _failures=[], _now=lambda: self.now,
                                        _log=lambda message: None))
        self.enterContext(patch.object(self.app, "log", lambda *args: None))
        self.enterContext(patch.object(self.app.google_calendar, "calendar_service",
                                       self.app.google_calendar.CalendarService(self.root / "synthetic-calendar.dat")))
        self.enterContext(patch.object(self.app.setup, "ENV_PATH", self.root / "synthetic-setup.env"))
        self.enterContext(patch.multiple(self.app.doses, LOG=self.run / "doses.jsonl",
                                        CONTACTS=self.root / "synthetic-contacts.json"))
        self.relay_open = self.enterContext(patch.object(self.app, "open_camera_relay",
            new=AsyncMock(side_effect=AssertionError("Unexpected camera relay"))))
        es = importlib.import_module("es")
        self.legacy_search = self.enterContext(patch.object(es, "search_all",
            side_effect=AssertionError("Personal queries must never fall back to Elasticsearch or JSONL")))
        self.client = self.enterContext(TestClient(self.app.app, base_url="https://testserver", raise_server_exceptions=False))
        self.block_socket_connects()

    def install_read_guards(self):
        exists, read_text, read_bytes, path_open = Path.exists, Path.read_text, Path.read_bytes, Path.open
        real_contacts = ROOT / "server" / "contacts.json"

        def protected(path):
            path = Path(path).resolve()
            if path.is_relative_to(self.root):
                return False
            if path.name in {".env", "contacts.json", "cert.pem", "key.pem", "google-calendar.dat"}:
                return True
            return path.is_relative_to(ROOT) and (path.suffix == ".jsonl" or
                any(part in {"runs", "photos", "faces", "gallery"} for part in path.parts))

        def safe_exists(path):
            return False if protected(path) else exists(path)

        def safe_read_text(path, *args, **kwargs):
            if Path(path).resolve() == real_contacts:
                return json.dumps(CONTACTS)
            if protected(path):
                raise AssertionError("Attempt to read real private data")
            return read_text(path, *args, **kwargs)

        def safe_read_bytes(path, *args, **kwargs):
            if protected(path):
                raise AssertionError("Attempt to read real private bytes")
            return read_bytes(path, *args, **kwargs)

        def safe_open(path, *args, **kwargs):
            if protected(path):
                raise AssertionError("Attempt to open real private data")
            return path_open(path, *args, **kwargs)

        for name, replacement in (("exists", safe_exists), ("read_text", safe_read_text),
                                  ("read_bytes", safe_read_bytes), ("open", safe_open)):
            self.enterContext(patch.object(Path, name, replacement))

    def login(self):
        response = self.client.post("/api/caregiver/login", json={"pin": PIN})
        self.assertEqual(response.status_code, 200, response.text)
        return response

    def review(self, *, result=None, store=None, **candidate_changes):
        store = self.store if store is None else store
        candidate_changes.setdefault("track", str(uuid.uuid4()))
        candidate = self.candidate(self.artifacts, **candidate_changes)
        store.enqueue(candidate)
        store.gallery(candidate["event_id"])
        event = store.apply_result(candidate["event_id"], result or verification())
        return candidate, event

    def tracked(self):
        candidate, event = self.review()
        self.store.confirm_object(event["object_id"], name="Test bottle")
        return candidate, event["object_id"]


class ObjectApiSecurityTests(ApiFixture):
    def test_imported_app_uses_synthetic_configuration_only(self):
        self.assertEqual(self.app.CONTACTS, CONTACTS)
        self.assertEqual(os.environ["DEEPGRAM_API_KEY"], LONG_KEY)
        self.assertEqual(self.app.MEMORY_JSONL, self.memory)
        self.assertFalse(Path(ROOT / "server" / ".env").exists())

    def test_personal_routes_deny_unauthenticated_reads_and_writes(self):
        reads = ["/api/find?q=pill+bottle", "/api/objects", "/api/objects/reviews",
                 "/api/objects/missing/history", "/api/objects/events/missing/image", "/api/location",
                 "/api/push", "/api/dg-token", "/api/agent-config", "/frames/anything.jpg", "/photos/anything"]
        for path in reads:
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 401)
                self.assertEqual(response.headers.get("cache-control"), "no-store")
                self.assertNotIn(LEGACY_MARKER, response.text)
        for path, body in (("/api/location", {"status": "denied"}),
                           ("/api/objects/reviews/missing", {"action": "track"}),
                           ("/api/es/index", {"object": "synthetic"})):
            with self.subTest(path=path):
                self.assertEqual(self.client.post(path, json=body).status_code, 401)
        self.legacy_search.assert_not_called()
        self.relay_open.assert_not_called()

    def test_missing_pin_and_expired_sessions_fail_closed(self):
        self.login()
        self.now += self.caregiver.SESSION_TTL_S + 1
        self.assertEqual(self.client.get("/api/objects/reviews").status_code, 401)
        with patch.dict(os.environ, {"CAREGIVER_PIN": ""}):
            self.assertEqual(self.client.get("/api/find?q=medicine").status_code, 404)
        self.legacy_search.assert_not_called()

    def test_pending_review_confirmation_drives_find_history_and_evidence(self):
        candidate, event = self.review()
        self.login()
        self.assertEqual(self.client.get("/api/find?q=medicine").json()["objects"], [])
        reviews = self.client.get("/api/objects/reviews")
        self.assertEqual(reviews.status_code, 200)
        self.assertEqual(reviews.headers["cache-control"], "no-store")
        self.assertEqual([r["event_id"] for r in reviews.json()["reviews"]], [candidate["event_id"]])
        self.assertNotIn(self.root.name, reviews.text)
        tracked = self.client.post(f"/api/objects/reviews/{candidate['event_id']}",
            json={"action": "track", "name": "Travel bottle"}, headers={"origin": "https://testserver"})
        self.assertEqual(tracked.status_code, 200, tracked.text)
        result = self.client.get("/api/find?q=medicine").json()
        self.assertEqual(result["source"], "object_memory")
        self.assertEqual(result["objects"][0]["object_id"], event["object_id"])
        self.assertIn("last saw", result["say"])
        self.assertNotIn(LEGACY_MARKER, json.dumps(result))
        image = self.client.get(result["card"]["image"])
        self.assertEqual(image.status_code, 200)
        self.assertEqual(image.headers["cache-control"], "no-store")
        self.assertTrue(image.content.startswith(b"\xff\xd8"))
        history = self.client.get(f"/api/objects/{event['object_id']}/history")
        self.assertEqual(history.status_code, 200)
        self.assertEqual(len(history.json()["observations"]), 1)
        self.assertNotIn(self.root.name, history.text)
        self.legacy_search.assert_not_called()

    def test_queued_pickup_find_response_exposes_no_stale_scene_image_or_map(self):
        first, object_id = self.tracked()
        self.store.enqueue(self.candidate(self.artifacts, at=self.now + 1, action="picked_up",
                                         track=first["track_id"], sequence=2))
        self.login()
        result = self.client.get("/api/find?q=medicine").json()
        self.assertEqual(result["objects"][0]["state"], "uncertain")
        self.assertIsNone(result["objects"][0]["scene"])
        self.assertIsNone(result["objects"][0]["location"])
        self.assertNotIn("image", result["card"])
        self.assertNotIn("action", result["card"])
        self.assertNotIn("synthetic table", result["say"])
        self.assertEqual(result["objects"][0]["object_id"], object_id)

    def test_review_link_and_ignore_are_explicit_and_do_not_duplicate_instances(self):
        _, object_id = self.tracked()
        candidate, _ = self.review(at=self.now + 1, result=verification(identity="uncertain"))
        self.login()
        linked = self.client.post(f"/api/objects/reviews/{candidate['event_id']}",
                                  json={"action": "link", "object_id": object_id})
        self.assertEqual(linked.status_code, 200, linked.text)
        self.assertEqual(linked.json()["object"]["object_id"], object_id)
        self.assertEqual(len(self.store.list_objects()), 1)
        self.assertEqual(len(self.store.history(object_id)), 2)
        ignored, _ = self.review(at=self.now + 2, result=verification(identity="uncertain"))
        self.assertEqual(self.client.post(f"/api/objects/reviews/{ignored['event_id']}",
                                         json={"action": "ignore"}).status_code, 200)
        self.assertEqual(self.client.get(f"/api/objects/events/{ignored['event_id']}/image").status_code, 404)

    def test_review_refuses_cross_origin_invalid_decisions_and_unverified_actor(self):
        candidate, _ = self.review()
        self.login()
        url = f"/api/objects/reviews/{candidate['event_id']}"
        self.assertEqual(self.client.post(url, json={"action": "track"},
                                         headers={"origin": "https://evil.invalid"}).status_code, 403)
        for body, status in (({"action": "link"}, 409), ({"action": "track", "object_id": TRACK}, 409),
                             ({"action": "delete"}, 422), ({"action": "track", "frames": []}, 422)):
            with self.subTest(body=body):
                self.assertEqual(self.client.post(url, json=body).status_code, status)
        unknown, _ = self.review(result=verification(actor="unknown"))
        self.assertEqual(self.client.post(f"/api/objects/reviews/{unknown['event_id']}",
                                         json={"action": "track"}).status_code, 409)
        self.assertEqual(self.store.list_objects(), [])

    def test_image_route_restricts_root_indices_and_profile(self):
        outside = self.jpeg(self.run / "outside-evidence.jpg")
        candidate, _ = self.review(frames=[str(outside)])
        other = ObjectStore(self.db, profile_id="other-profile", clock=lambda: self.now)
        foreign, foreign_event = self.review(store=other)
        self.login()
        self.assertEqual(self.client.get(f"/api/objects/events/{candidate['event_id']}/image").status_code, 404)
        self.assertEqual(self.client.get(f"/api/objects/events/{foreign['event_id']}/image").status_code, 404)
        self.assertEqual(self.client.get(f"/api/objects/{foreign_event['object_id']}/history").status_code, 404)
        for index in (-1, 5):
            self.assertEqual(self.client.get(f"/api/objects/events/{candidate['event_id']}/image?index={index}").status_code, 422)
        self.assertNotIn(foreign["event_id"], self.client.get("/api/objects/reviews").text)

    def test_missing_or_corrupt_explicit_database_never_falls_back_to_legacy_memory(self):
        import sqlite3

        connections = []
        original_connect = sqlite3.connect

        class ObservedConnection(sqlite3.Connection):
            closed = False

            def close(self):
                super().close()
                self.closed = True

        def tracked_connect(*args, **kwargs):
            # Only these temporary test connections permit cross-thread cleanup.
            # The real SQL/error path still runs; assert close() was called before
            # fixture cleanup, then release any leak to avoid Windows file locks.
            kwargs.update(factory=ObservedConnection, check_same_thread=False)
            connection = original_connect(*args, **kwargs)
            connections.append(connection)
            self.addCleanup(connection.close)
            return connection

        self.login()
        missing = self.root / "missing" / "objects.sqlite3"
        corrupt = self.root / "corrupt.sqlite3"
        corrupt.write_bytes(b"SYNTHETIC-NOT-A-SQLITE-DATABASE")
        with patch.object(sqlite3, "connect", tracked_connect):
            for path in (missing, corrupt):
                with self.subTest(path=path.name), patch.dict(os.environ, {"PAM_OBJECT_DB": str(path)}):
                    result = self.client.get("/api/find?q=medicine")
                    self.assertGreaterEqual(result.status_code, 500)
                    self.assertNotIn(LEGACY_MARKER, result.text)
        self.legacy_search.assert_not_called()
        self.assertTrue(all(connection.closed for connection in connections),
                        "Database initialization failures must close their connections, including failed PRAGMAs")

    def test_sibling_database_activates_personal_mode_without_environment_override(self):
        self.tracked()
        self.login()
        with patch.dict(os.environ, {"PAM_OBJECT_DB": ""}):
            result = self.client.get("/api/find?q=medicine")
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["source"], "object_memory")
        self.legacy_search.assert_not_called()

    def test_explicit_legacy_mode_still_uses_legacy_search(self):
        legacy_path = self.root / "legacy-only" / "memory.jsonl"
        legacy_path.parent.mkdir()
        legacy_path.write_text("", encoding="utf-8")
        es = importlib.import_module("es")
        with patch.dict(os.environ, {"PAM_OBJECT_DB": ""}), patch.object(self.app, "MEMORY_JSONL", legacy_path), \
                patch.object(es, "search_all", return_value=([], "synthetic-legacy")) as search:
            response = self.client.get("/api/find?q=keys")
        self.assertEqual(response.status_code, 200)
        search.assert_called_once_with("keys", legacy_path, limit=3)

    def test_location_denial_stopping_and_stale_fixes_clear_prior_location(self):
        self.login()
        places = importlib.import_module("places")
        with patch.object(places, "resolve", return_value={"place": "test place", "source": "known", "lat": 1, "lon": 2}) as resolve:
            good = {"lat": 1.0, "lon": 2.0, "accuracy_m": 8.0, "observed_at": self.now - 1}
            self.assertEqual(self.client.post("/api/location", json=good).status_code, 200)
            self.assertFalse(resolve.call_args.kwargs["allow_google"])
            self.assertEqual(self.app._last_fix["at"], good["observed_at"])
            for status in ("denied", "stopped", "unavailable"):
                self.app._last_fix.update(place="test place", source="known", at=self.now)
                self.assertEqual(self.client.post("/api/location", json={"status": status}).status_code, 200)
                self.assertEqual(self.app._last_fix, {})
            self.app._last_fix.update(place="test place", source="known", at=self.now)
            self.client.post("/api/location", json={**good, "observed_at": self.now - 1000})
            self.assertEqual(self.app._last_fix, {})
        self.assertEqual(resolve.call_count, 1)


class FakeRelay:
    def __init__(self):
        self.sent = []
        self.closed = False

    async def send(self, data):
        self.sent.append(data)

    async def wait_closed(self):
        await asyncio.Future()

    async def close(self):
        self.closed = True


class CameraRelaySecurityTests(ApiFixture):
    def relay(self):
        relay = FakeRelay()
        self.relay_open.side_effect = None
        self.relay_open.return_value = relay
        return relay

    def packet(self, frame_id=1, session=CAMERA):
        jpeg = cv2.imencode(".jpg", np.zeros((24, 32, 3), np.uint8))[1].tobytes()
        packet = encode_frame(jpeg, {"session_id": session, "frame_id": frame_id,
            "captured_at": time.time(), "location": {"lat": 1.0, "lon": 2.0, "accuracy_m": 8.0,
                                                       "observed_at": time.time() - 1}})
        return packet, jpeg

    def test_unauthenticated_camera_never_opens_relay(self):
        with self.client.websocket_connect("wss://testserver/api/camera") as ws:
            message = ws.receive_json()
            self.assertEqual(message["type"], "camera_error")
            self.assertFalse(message["retry"])
        self.relay_open.assert_not_called()

    def test_authenticated_cmp2_envelope_is_relayed_intact_not_as_raw_jpeg(self):
        self.login()
        relay = self.relay()
        packet, jpeg = self.packet()
        with self.client.websocket_connect("wss://testserver/api/camera", headers={"origin": "https://testserver"}) as ws:
            self.assertEqual(ws.receive_json()["type"], "camera_ready")
            ws.send_bytes(packet)
            self.assertEqual(ws.receive_json()["type"], "frame_received")
        self.assertEqual(relay.sent, [packet])
        self.assertNotEqual(relay.sent[0], jpeg)
        self.assertEqual(self.app._last_frame, jpeg)
        self.assertEqual(decode_frame(packet)[1]["time_source"], "capture")
        self.assertTrue(relay.closed)

    def test_personal_camera_rejects_raw_frames_before_forwarding(self):
        self.login()
        relay = self.relay()
        _, jpeg = self.packet()
        with self.client.websocket_connect("wss://testserver/api/camera") as ws:
            self.assertEqual(ws.receive_json()["type"], "camera_ready")
            ws.send_bytes(jpeg)
            error = ws.receive_json()
            self.assertEqual(error["type"], "camera_error")
            self.assertFalse(error["retry"])
        self.assertEqual(relay.sent, [])
        self.assertIsNone(self.app._last_frame)

    def test_legacy_camera_keeps_raw_jpeg_compatibility_without_personal_database(self):
        relay = self.relay()
        _, jpeg = self.packet()
        with patch.dict(os.environ, {"PAM_OBJECT_DB": ""}), \
                patch.object(self.app, "MEMORY_JSONL", self.root / "legacy-camera" / "memory.jsonl"):
            with self.client.websocket_connect("wss://testserver/api/camera") as ws:
                self.assertEqual(ws.receive_json()["type"], "camera_ready")
                ws.send_bytes(jpeg)
                self.assertEqual(ws.receive_json()["type"], "frame_received")
        self.assertEqual(relay.sent, [jpeg])
        self.assertEqual(self.app._last_frame, jpeg)

    def test_duplicate_frame_sequence_is_not_relayed_twice(self):
        self.login()
        relay = self.relay()
        first, _ = self.packet(1)
        second, _ = self.packet(2)
        with self.client.websocket_connect("wss://testserver/api/camera") as ws:
            ws.receive_json()
            ws.send_bytes(first)
            ws.receive_json()
            ws.send_bytes(first)
            ws.send_bytes(second)
            ws.close()
            self.assertEqual(ws.receive()["type"], "websocket.close")
        self.assertEqual(relay.sent, [first, second])

    def test_revoked_session_stops_camera_before_the_next_forward(self):
        self.login()
        relay = self.relay()
        packet, _ = self.packet()
        with self.client.websocket_connect("wss://testserver/api/camera") as ws:
            ws.receive_json()
            self.assertEqual(self.client.post("/api/caregiver/logout").status_code, 200)
            ws.send_bytes(packet)
            self.assertEqual(ws.receive()["type"], "websocket.close")
        self.assertEqual(relay.sent, [])

    def test_cross_origin_camera_is_rejected_before_opening_relay(self):
        self.login()
        with self.client.websocket_connect("wss://testserver/api/camera", headers={"origin": "https://evil.invalid"}) as ws:
            error = ws.receive_json()
            self.assertEqual(error["type"], "camera_error")
            self.assertFalse(error["retry"])
        self.relay_open.assert_not_called()


class TokenSecurityTests(ApiFixture):
    def grant(self, response=None, error=None):
        post = AsyncMock(return_value=response, side_effect=error)
        provider = MagicMock()
        provider.__aenter__ = AsyncMock(return_value=SimpleNamespace(post=post))
        provider.__aexit__ = AsyncMock(return_value=False)
        self.enterContext(patch.object(self.app.httpx, "AsyncClient", return_value=provider))
        return post

    def assertNoSecret(self, response):
        self.assertNotIn(LONG_KEY, response.text)
        self.assertNotIn(LONG_KEY, str(response.headers))
        self.assertNotIn("UPSTREAM-PRIVATE-DETAIL", response.text)

    def test_grant_403_never_falls_back_to_long_lived_api_key(self):
        self.login()
        post = self.grant(httpx.Response(403, text=LONG_KEY + " UPSTREAM-PRIVATE-DETAIL"))
        response = self.client.get("/api/dg-token")
        self.assertGreaterEqual(response.status_code, 400)
        self.assertNoSecret(response)
        self.assertEqual(response.headers.get("cache-control"), "no-store")
        post.assert_awaited_once()
        self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], "Token " + LONG_KEY)

    def test_upstream_errors_and_exceptions_do_not_echo_credentials_or_provider_body(self):
        self.login()
        post = self.grant()
        for status in (401, 429, 500):
            with self.subTest(status=status):
                post.return_value = httpx.Response(status, text=LONG_KEY + " UPSTREAM-PRIVATE-DETAIL")
                response = self.client.get("/api/dg-token")
                self.assertGreaterEqual(response.status_code, 400)
                self.assertNoSecret(response)
        post.side_effect = httpx.ConnectError(LONG_KEY + " UPSTREAM-PRIVATE-DETAIL")
        response = self.client.get("/api/dg-token")
        self.assertGreaterEqual(response.status_code, 400)
        self.assertNoSecret(response)

    def test_success_returns_only_the_short_lived_grant_and_requests_bounded_ttl(self):
        self.login()
        post = self.grant(httpx.Response(200, json={"access_token": "synthetic-short-lived-grant", "other": LONG_KEY}))
        response = self.client.get("/api/dg-token")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.text, "synthetic-short-lived-grant")
        self.assertNoSecret(response)
        self.assertEqual(response.headers.get("cache-control"), "no-store")
        self.assertEqual(post.call_args.kwargs["json"], {"ttl_seconds": 60})

    def test_legacy_mode_does_not_restore_the_long_lived_key_fallback(self):
        self.grant(httpx.Response(403, text=LONG_KEY))
        with patch.dict(os.environ, {"PAM_OBJECT_DB": ""}), \
                patch.object(self.app, "MEMORY_JSONL", self.root / "legacy-only" / "memory.jsonl"):
            response = self.client.get("/api/dg-token")
        self.assertGreaterEqual(response.status_code, 400)
        self.assertNoSecret(response)


class MemoryConnection:
    def __init__(self, request):
        self.input = io.BytesIO(request)
        self.output = io.BytesIO()

    def makefile(self, mode, *args, **kwargs):
        return self.input

    def sendall(self, data):
        self.output.write(data)

    def setsockopt(self, *args):
        pass


class PublicFileSecurityTests(IsolatedCase):
    def setUp(self):
        super().setUp()
        self.serve = importlib.import_module("phone.serve")
        self.public = self.root / "public"
        self.public.mkdir()
        (self.public / "assets").mkdir()
        for name, text in (("index.html", "SYNTHETIC-CAMERA-PAGE"), ("agent.html", "SYNTHETIC-AGENT-PAGE"),
                           ("assets/test.css", "SYNTHETIC-PUBLIC-CSS"), ("key.pem", LONG_KEY),
                           ("cert.pem", "SYNTHETIC-CERT-PRIVATE"), (".env", LONG_KEY),
                           ("private.json", LONG_KEY), ("assets/.env", LONG_KEY), ("assets/key.pem", LONG_KEY)):
            (self.public / name).write_text(text, encoding="utf-8")

    def request(self, path, directory=None):
        self.assertTrue(hasattr(self.serve, "PublicFiles"), "phone.serve must install the PublicFiles allowlist handler")
        connection = MemoryConnection(f"GET {path} HTTP/1.0\r\nHost: testserver\r\n\r\n".encode("ascii"))
        server = SimpleNamespace(server_name="testserver", server_port=8443)
        with patch.object(self.serve.PublicFiles, "log_message", lambda *args: None):
            self.serve.PublicFiles(connection, ("127.0.0.1", 12345), server,
                                   directory=str(directory or self.public))
        raw = connection.output.getvalue()
        header, _, body = raw.partition(b"\r\n\r\n")
        return int(header.split(b" ", 2)[1]), body

    def test_only_camera_agent_and_public_assets_are_served(self):
        for path, marker in (("/", b"SYNTHETIC-CAMERA-PAGE"), ("/index.html", b"SYNTHETIC-CAMERA-PAGE"),
                             ("/agent.html", b"SYNTHETIC-AGENT-PAGE"), ("/assets/test.css", b"SYNTHETIC-PUBLIC-CSS")):
            with self.subTest(path=path):
                status, body = self.request(path)
                self.assertEqual(status, 200)
                self.assertIn(marker, body)

    def test_private_material_is_denied_even_when_it_exists(self):
        for path in ("/key.pem", "/cert.pem", "/.env", "/private.json", "/assets/.env", "/assets/key.pem"):
            with self.subTest(path=path):
                status, body = self.request(path)
                self.assertIn(status, (400, 403, 404))
                self.assertNotIn(LONG_KEY.encode(), body)
                self.assertNotIn(b"SYNTHETIC-CERT-PRIVATE", body)

    def test_directory_listing_is_never_exposed(self):
        empty_index = self.root / "without-index"
        empty_index.mkdir()
        (empty_index / "secret-name.txt").write_text(LONG_KEY, encoding="utf-8")
        for path, directory in (("/assets/", self.public), ("/", empty_index)):
            with self.subTest(path=path, directory=directory.name):
                status, body = self.request(path, directory)
                self.assertIn(status, (400, 403, 404))
                self.assertNotIn(b"Directory listing", body)
                self.assertNotIn(b"secret-name.txt", body)

    def test_encoded_and_plain_traversal_never_expose_private_files(self):
        paths = ("/../key.pem", "/%2e%2e/key.pem", "/assets/../key.pem", "/assets/%2e%2e/key.pem",
                 "/assets/%2e%2e%5ckey.pem", "/assets/%252e%252e/key.pem", "/assets-extra/key.pem")
        for path in paths:
            with self.subTest(path=path):
                status, body = self.request(path)
                self.assertIn(status, (400, 403, 404))
                self.assertNotIn(LONG_KEY.encode(), body)

    def test_main_wires_the_allowlist_handler_without_opening_a_server_or_certificate(self):
        self.assertTrue(hasattr(self.serve, "PublicFiles"), "PublicFiles handler is required")
        httpd = MagicMock()
        httpd.serve_forever.side_effect = KeyboardInterrupt
        context = MagicMock()
        with patch.object(sys, "argv", ["serve.py"]), patch.object(self.serve, "HERE", self.public), \
                patch.object(self.serve, "CERT", self.public / "cert.pem"), patch.object(self.serve, "KEY", self.public / "key.pem"), \
                patch.object(self.serve, "lan_ip", return_value="127.0.0.1"), patch.object(self.serve, "ensure_cert"), \
                patch.object(self.serve.ssl, "SSLContext", return_value=context), \
                patch.object(self.serve.http.server, "ThreadingHTTPServer", return_value=httpd) as factory, \
                patch("builtins.print"):
            self.serve.main()
        handler = factory.call_args.args[1]
        self.assertIsInstance(handler, functools.partial)
        self.assertIs(handler.func, self.serve.PublicFiles)
        self.assertEqual(handler.keywords["directory"], str(self.public))


if __name__ == "__main__":
    unittest.main(verbosity=2)
