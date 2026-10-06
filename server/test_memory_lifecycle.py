"""Synthetic object-memory safety regressions; no server, model, API, or personal data.

Run from the repository root with the existing perception virtual environment:
    perception/.venv/Scripts/python.exe -B server/test_memory_lifecycle.py -v

Every test owns a TemporaryDirectory and an injected clock. Frame/crop names refer
only to that directory; no images are opened. Network connections are forbidden.
Verification scores below are fabricated decision-gate inputs, NOT calibrated
probabilities or measurements of a vision model's accuracy. Known regressions are
ordinary assertions, not skipped tests or expectedFailure decorators.
"""
from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import itertools
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from perception.capture import FIX_MAX_ACCURACY_M, FIX_MAX_AGE_S, location_at
from perception.object_memory import ObjectStore, Verification


T0 = datetime(2026, 1, 15, 12, tzinfo=timezone.utc).timestamp()
SESSION_A = "camera-session-a-0001"
SESSION_B = "camera-session-b-0002"
TRACK_A = "11111111-1111-4111-8111-111111111111"
TRACK_B = "22222222-2222-4222-8222-222222222222"
TRACK_C = "33333333-3333-4333-8333-333333333333"
UNSEEN_ID = "ffffffff-ffff-4fff-8fff-ffffffffffff"
_UNSET = object()


class MemoryFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="compass-memory-tests-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "objects.sqlite3"
        self.now = T0 + 60
        self.serial = itertools.count(1)
        for target in ("socket.create_connection", "socket.socket.connect", "socket.socket.connect_ex"):
            self.enterContext(patch(target, side_effect=AssertionError("Network is forbidden in memory tests")))
        self.store = self.reopen()

    def reopen(self, profile_id="local"):
        return ObjectStore(self.path, profile_id=profile_id, clock=lambda: self.now)

    def candidate(self, *, at=T0, action="placed", track=TRACK_A, session=SESSION_A,
                  label="pill bottle", evidence=None, location=_UNSET, **overrides):
        n = next(self.serial)
        interaction = {
            "contact_candidate": True, "motion_valid": True, "released": action == "placed",
            "track_continuous": True, "track_started_at": T0 - 10,
        }
        if evidence is not None:
            interaction.update(evidence)
        if location is _UNSET:
            location = {"lat": 1.0, "lon": 2.0, "accuracy_m": 8.0, "observed_at": at - 1}
        out = {
            "event_id": f"event-{n:04d}", "session_id": session, "track_id": track,
            "observed_at": at, "sequence": n, "event_type": action, "object": label,
            "time_source": "capture", "evidence": interaction, "co_visible_track_ids": [],
            "frames": [str(self.root / f"{n}-{role}.jpg") for role in ("before", "during", "after")],
            "crop": str(self.root / f"{n}-crop.jpg"), "location": copy.deepcopy(location),
        }
        out.update(overrides)
        return out

    def verification(self, *, action="placed", identity="new", object_id=None, **overrides):
        fields = {
            "actor": "wearer", "actor_confidence": .99,
            "actor_evidence": ["Synthetic near-field wearer contact and object motion"],
            "action": action, "action_confidence": .99, "contact_observed": True,
            "released": action == "placed", "resting_after": action == "placed",
            "object": "pill bottle", "appearance": "Synthetic amber container with a red diagonal marking",
            "identity": identity, "matched_object_id": object_id, "identity_confidence": .99,
            "identity_basis": "continuous_track" if identity == "match" else "distinctive_features",
            "identity_evidence": "Synthetic uninterrupted segment" if identity == "match" else "Distinctive red diagonal marking",
            "scene": {"room": "test room A", "surface": "test table A", "landmarks": ["test cube"],
                      "description": "On test table A beside the test cube."},
        }
        fields.update(overrides)
        return Verification.model_validate(fields)

    def verify(self, candidate, result=None, *, store=None, cost_usd=0.0):
        store = self.store if store is None else store
        store.enqueue(candidate)
        store.gallery(candidate["event_id"])
        result = result if result is not None else self.verification(
            action=candidate["event_type"], object=candidate["object"])
        return store.apply_result(candidate["event_id"], result, cost_usd=cost_usd)

    def tracked(self, *, store=None, name=None, **candidate_kwargs):
        """Fixture identity is explicitly confirmed by a human, never silently inferred."""
        store = self.store if store is None else store
        candidate = self.candidate(**candidate_kwargs)
        result = self.verify(candidate, store=store)
        self.assertEqual(result["status"], "review", result)
        if result["object_id"] is None:
            obj = store.resolve_review(candidate["event_id"], name=name)
        else:
            obj = store.confirm_object(result["object_id"], name=name)
        self.assertEqual(obj["status"], "tracked")
        return obj["object_id"], candidate

    def assertNoMap(self, location):
        for key in ("maps_url", "lat", "lon"):
            self.assertNotIn(key, location or {}, location)

    def assertNoCurrentLocation(self, obj):
        self.assertNotEqual(obj["state"], "placed", obj)
        self.assertIsNone(obj["location"], obj)

    def assertDeniedCandidate(self, candidate):
        """Validation errors or explicit rejections both fail closed; reviews/queues do not."""
        before = self.store.usage()
        try:
            result = self.store.enqueue(candidate)
        except ValueError:
            self.assertEqual(self.store.usage(), before)
            return
        self.assertEqual(result["status"], "rejected",
                         f"Malformed candidate {candidate.get('event_id')!r} must not enter verification")
        self.assertIsNone(result["object_id"])
        self.assertEqual(self.store.list_objects(include_pending=True), [])


class LifecycleTests(MemoryFixture):
    def test_store_shelf_sighting_is_rejected_without_frames_crop_or_geo(self):
        for at in (T0, T0 + 30, T0 + 3600):
            with self.subTest(at=at):
                candidate = self.candidate(at=at, action="sighted")
                result = self.store.enqueue(candidate)
                self.assertEqual(result["status"], "rejected")
                self.assertEqual(result["reason"], "background_sighting")
                for field in ("frames", "crop", "location"):
                    self.assertNotIn(field, result)
                self.store.gallery(candidate["event_id"])
                forced = self.store.apply_result(candidate["event_id"], self.verification())
                self.assertEqual(forced["status"], "rejected")
        self.assertEqual(self.store.list_objects(include_pending=True), [])
        self.assertEqual(self.store.search("pill bottle"), [])
        self.assertIsNone(self.store.claim_next())
        self.assertEqual(self.store.usage()["api_calls"], 0)

    def test_bystander_movement_never_creates_a_personal_object(self):
        result = self.verify(self.candidate(), self.verification(actor="bystander"))
        self.assertEqual(result["status"], "rejected")
        self.assertIsNone(result["object_id"])
        self.assertEqual(self.store.list_objects(include_pending=True), [])
        self.assertEqual(self.store.search("pill bottle"), [])

    def test_first_wearer_placement_requires_explicit_tracking_confirmation(self):
        candidate = self.candidate()
        result = self.verify(candidate)
        self.assertEqual((result["status"], result["reason"]), ("review", "confirm_tracking"))
        object_id = result["object_id"]
        self.assertIsNotNone(object_id)
        self.assertEqual(self.store.get_object(object_id)["status"], "pending")
        self.assertEqual(self.store.list_objects(), [])
        self.assertEqual(self.store.search("pill bottle"), [])
        self.assertEqual([r["event_id"] for r in self.store.reviews()], [candidate["event_id"]])
        obj = self.store.confirm_object(object_id, name="  Travel bottle  ")
        self.assertEqual((obj["object_id"], obj["name"], obj["status"], obj["state"]),
                         (object_id, "Travel bottle", "tracked", "placed"))
        self.assertEqual(obj["location"]["status"], "available")
        self.assertEqual([o["object_id"] for o in self.store.search("my travel bottle")], [object_id])
        self.assertEqual(self.store.reviews(), [])

    def test_queued_pickup_immediately_blocks_the_old_location_then_becomes_carried(self):
        object_id, placed = self.tracked()
        pickup = self.candidate(at=T0 + 10, action="picked_up")
        self.store.enqueue(pickup)
        obj = self.store.get_object(object_id)
        self.assertEqual(obj["state"], "uncertain")
        self.assertTrue(obj["needs_review"])
        self.assertNoCurrentLocation(obj)
        self.assertNoCurrentLocation(self.store.search("pill bottle")[0])
        self.store.gallery(pickup["event_id"])
        result = self.store.apply_result(pickup["event_id"], self.verification(
            action="picked_up", identity="match", object_id=object_id))
        self.assertEqual((result["status"], result["object_id"]), ("verified", object_id))
        obj = self.store.get_object(object_id)
        self.assertEqual(obj["state"], "carried")
        self.assertFalse(obj["needs_review"])
        self.assertNoCurrentLocation(obj)
        self.assertEqual([h["event_id"] for h in self.store.history(object_id)],
                         [pickup["event_id"], placed["event_id"]])

    def test_replacement_updates_the_same_id_and_preserves_historical_location(self):
        object_id, first = self.tracked()
        pickup = self.candidate(at=T0 + 5, action="picked_up")
        self.verify(pickup, self.verification(action="picked_up", identity="match", object_id=object_id))
        replacement = self.candidate(at=T0 + 10, location={
            "lat": 3.0, "lon": 4.0, "accuracy_m": 7.0, "observed_at": T0 + 9})
        result = self.verify(replacement, self.verification(identity="match", object_id=object_id,
            scene={"surface": "test table B", "description": "On test table B."}))
        self.assertEqual((result["status"], result["object_id"]), ("verified", object_id))
        obj = self.store.get_object(object_id)
        self.assertEqual(obj["state"], "placed")
        self.assertEqual(obj["latest"]["scene"]["surface"], "test table B")
        self.assertEqual((obj["location"]["lat"], obj["location"]["lon"]), (3.0, 4.0))
        self.assertEqual(len(self.store.list_objects()), 1)
        history = self.store.history(object_id)
        self.assertEqual([h["event_id"] for h in history],
                         [replacement["event_id"], pickup["event_id"], first["event_id"]])
        self.assertEqual(history[-1]["location"]["lat"], 1.0)

    def test_pickup_then_out_of_view_never_resurrects_a_resting_place(self):
        object_id, _ = self.tracked()
        pickup = self.candidate(at=T0 + 5, action="picked_up")
        self.verify(pickup, self.verification(action="picked_up", identity="match", object_id=object_id))
        self.store.enqueue(self.candidate(at=T0 + 600, action="sighted", track=TRACK_B, location=None))
        self.now += 86400
        restored = self.reopen()
        self.assertNoCurrentLocation(restored.get_object(object_id))
        self.assertNoCurrentLocation(restored.search("pill bottle")[0])
        self.assertEqual(len(restored.history(object_id)), 2)

    def test_bystander_moving_an_owned_object_invalidates_but_does_not_own_it(self):
        object_id, _ = self.tracked()
        result = self.verify(self.candidate(at=T0 + 5, action="picked_up"),
                             self.verification(actor="bystander", action="picked_up"))
        self.assertEqual(result["status"], "rejected")
        obj = self.store.get_object(object_id)
        self.assertEqual(obj["state"], "unknown")
        self.assertNoCurrentLocation(obj)
        self.assertEqual(len(self.store.list_objects(include_pending=True)), 1)
        self.assertEqual(self.store.history(object_id)[0]["actor"], "bystander")

    def test_unknown_actor_cannot_be_promoted_by_identity_confirmation(self):
        candidate = self.candidate()
        result = self.verify(candidate, self.verification(actor="unknown"))
        self.assertEqual(result["status"], "review")
        self.assertIsNone(result["object_id"])
        with self.assertRaises(ValueError):
            self.store.resolve_review(candidate["event_id"], name="My bottle")
        self.assertEqual(self.store.list_objects(include_pending=True), [])

    def test_action_conflict_cannot_be_promoted_by_review(self):
        candidate = self.candidate(action="placed")
        result = self.verify(candidate, self.verification(action="picked_up"))
        self.assertEqual(result["status"], "review")
        self.assertIsNone(result["object_id"])
        with self.assertRaises(ValueError):
            self.store.resolve_review(candidate["event_id"])

    def test_action_gates_require_evidence_not_just_high_self_reported_scores(self):
        cases = [{"actor_confidence": .849}, {"action_confidence": .849}, {"actor_evidence": []},
                 {"contact_observed": False}, {"released": False}, {"resting_after": False}]
        for changes in cases:
            with self.subTest(changes=changes):
                candidate = self.candidate()
                result = self.verify(candidate, self.verification(**changes))
                self.assertEqual(result["status"], "review")
                self.assertIsNone(result["object_id"])
                with self.assertRaises(ValueError):
                    self.store.resolve_review(candidate["event_id"])
        self.assertEqual(self.store.list_objects(include_pending=True), [])

    def test_unseen_pickup_cannot_create_a_new_resting_object(self):
        candidate = self.candidate(action="picked_up")
        result = self.verify(candidate, self.verification(action="picked_up", identity="new"))
        self.assertEqual(result["status"], "review")
        self.assertIsNone(result["object_id"])
        with self.assertRaises(ValueError):
            self.store.resolve_review(candidate["event_id"])
        self.assertEqual(self.store.list_objects(include_pending=True), [])

    def test_user_can_resolve_ambiguous_identity_without_creating_a_second_instance(self):
        object_id, _ = self.tracked()
        candidate = self.candidate(at=T0 + 5, track=TRACK_B)
        result = self.verify(candidate, self.verification(identity="uncertain"))
        self.assertEqual(result["status"], "review")
        self.assertIsNone(result["object_id"])
        obj = self.store.resolve_review(candidate["event_id"], object_id=object_id, name="Travel bottle")
        self.assertEqual(obj["object_id"], object_id)
        self.assertEqual(len(self.store.list_objects()), 1)
        self.assertEqual(self.store.history(object_id)[0]["identity_basis"], "user_confirmation")

    def test_pending_object_cannot_be_silently_merged_into_another_confirmed_object(self):
        first_id, _ = self.tracked()
        candidate = self.candidate(at=T0 + 5, track=TRACK_B, label="keys")
        result = self.verify(candidate)
        self.assertEqual(result["reason"], "confirm_tracking")
        self.assertNotEqual(result["object_id"], first_id)
        with self.assertRaises(ValueError):
            self.store.resolve_review(candidate["event_id"], object_id=first_id)
        self.assertEqual(len(self.store.history(first_id)), 1)


class IdentitySafetyTests(MemoryFixture):
    def test_category_similarity_or_missing_identity_evidence_is_not_a_match(self):
        object_id, _ = self.tracked()
        cases = [{"identity_basis": "category_only"}, {"identity_basis": "uncertain"},
                 {"identity_evidence": ""}, {"identity_confidence": .899}]
        for changes in cases:
            with self.subTest(changes=changes):
                candidate = self.candidate(at=T0 + 5, track=TRACK_B)
                result = self.verify(candidate, self.verification(identity="match", object_id=object_id,
                    **{"identity_basis": "distinctive_features", **changes}))
                self.assertEqual(result["status"], "review")
                self.assertIsNone(result["object_id"])
        self.assertEqual(len(self.store.history(object_id)), 1)

    def test_model_cannot_match_an_id_that_was_not_in_the_gallery(self):
        object_id, _ = self.tracked()
        result = self.verify(self.candidate(at=T0 + 5, track=TRACK_B), self.verification(
            identity="match", object_id=UNSEEN_ID, identity_basis="distinctive_features"))
        self.assertEqual(result["status"], "review")
        self.assertIsNone(result["object_id"])
        self.assertEqual(len(self.store.history(object_id)), 1)

    def test_explicit_uninterrupted_track_can_continue_beyond_thirty_seconds(self):
        object_id, _ = self.tracked()
        candidate = self.candidate(at=T0 + 60, evidence={"track_started_at": T0 - 10})
        self.store.enqueue(candidate)
        gallery = self.store.gallery(candidate["event_id"])
        entry = next(o for o in gallery if o["object_id"] == object_id)
        self.assertTrue(entry["continuous_track"], "A proven continuous segment is not a 30-second heuristic")
        result = self.store.apply_result(candidate["event_id"], self.verification(
            identity="match", object_id=object_id))
        self.assertEqual(result["status"], "verified")

    def test_missing_or_false_continuity_evidence_cannot_authorize_a_track_match(self):
        object_id, _ = self.tracked()
        for continuity in (_UNSET, False):
            with self.subTest(continuity="missing" if continuity is _UNSET else continuity):
                candidate = self.candidate(at=T0 + 5)
                if continuity is _UNSET:
                    candidate["evidence"].pop("track_continuous")
                else:
                    candidate["evidence"]["track_continuous"] = continuity
                self.store.enqueue(candidate)
                gallery = self.store.gallery(candidate["event_id"])
                self.assertFalse(next(o for o in gallery if o["object_id"] == object_id)["continuous_track"])
                result = self.store.apply_result(candidate["event_id"], self.verification(
                    identity="match", object_id=object_id))
                self.assertEqual(result["status"], "review")
                self.assertIsNone(result["object_id"])
        self.assertEqual(len(self.store.history(object_id)), 1)

    def test_restarted_track_segment_is_not_continuous_with_an_older_observation(self):
        object_id, _ = self.tracked()
        candidate = self.candidate(at=T0 + 10, evidence={"track_started_at": T0 + 5})
        self.store.enqueue(candidate)
        gallery = self.store.gallery(candidate["event_id"])
        self.assertFalse(next(o for o in gallery if o["object_id"] == object_id)["continuous_track"])
        result = self.store.apply_result(candidate["event_id"], self.verification(
            identity="match", object_id=object_id))
        self.assertEqual(result["status"], "review")
        self.assertIsNone(result["object_id"])

    def test_simultaneously_visible_tracks_cannot_be_merged_despite_a_model_match(self):
        object_id, _ = self.tracked()
        candidate = self.candidate(at=T0 + 5, track=TRACK_B, co_visible_track_ids=[TRACK_A])
        result = self.verify(candidate, self.verification(identity="match", object_id=object_id,
                                                         identity_basis="distinctive_features"))
        self.assertEqual(result["status"], "review", "Two simultaneously visible tracks cannot be one item")
        self.assertIsNone(result["object_id"])
        self.assertEqual(len(self.store.history(object_id)), 1)

    def test_cross_category_model_match_requires_review_instead_of_mutating_identity(self):
        object_id, _ = self.tracked()
        candidate = self.candidate(at=T0 + 5, track=TRACK_B, label="keys")
        result = self.verify(candidate, self.verification(identity="match", object_id=object_id,
            object="keys", identity_basis="distinctive_features"))
        self.assertEqual(result["status"], "review")
        self.assertIsNone(result["object_id"])
        self.assertEqual(self.store.get_object(object_id)["category"], "pill bottle")
        self.assertEqual(len(self.store.history(object_id)), 1)

    def test_incomplete_gallery_cannot_approve_a_match_among_identical_categories(self):
        first_id, _ = self.tracked()
        second_id, _ = self.tracked(at=T0 + 1, track=TRACK_B)
        candidate = self.candidate(at=T0 + 5, track=TRACK_C)
        self.store.enqueue(candidate)
        gallery = self.store.gallery(candidate["event_id"], limit=1)
        self.assertEqual(len(gallery), 1)
        self.assertFalse(self.store.event(candidate["event_id"])["gallery_complete"])
        result = self.store.apply_result(candidate["event_id"], self.verification(
            identity="match", object_id=gallery[0]["object_id"], identity_basis="distinctive_features"))
        self.assertEqual(result["status"], "review", "An omitted same-category alternative was never compared")
        self.assertIsNone(result["object_id"])
        self.assertEqual(len(self.store.history(first_id)), 1)
        self.assertEqual(len(self.store.history(second_id)), 1)

    def test_incomplete_gallery_cannot_allocate_a_new_identity_automatically(self):
        self.tracked()
        self.tracked(at=T0 + 1, track=TRACK_B)
        candidate = self.candidate(at=T0 + 5, track=TRACK_C)
        self.store.enqueue(candidate)
        self.store.gallery(candidate["event_id"], limit=1)
        result = self.store.apply_result(candidate["event_id"], self.verification(identity="new"))
        self.assertEqual(result["status"], "review")
        self.assertIsNone(result["object_id"])
        self.assertEqual(len(self.store.list_objects(include_pending=True)), 2)

    def test_gallery_must_be_prepared_before_a_result_can_allocate_an_identity(self):
        candidate = self.candidate()
        self.store.enqueue(candidate)
        result = self.store.apply_result(candidate["event_id"], self.verification())
        self.assertEqual(result["status"], "review")
        self.assertIsNone(result["object_id"], "Missing gallery preparation is not an empty complete gallery")
        self.assertEqual(self.store.list_objects(include_pending=True), [])

    def test_identical_appearance_can_be_two_instances_after_explicit_user_confirmation(self):
        first_id, _ = self.tracked()
        candidate = self.candidate(at=T0 + 5, track=TRACK_B, co_visible_track_ids=[TRACK_A])
        result = self.verify(candidate, self.verification(identity="uncertain",
                                                         identity_basis="category_only"))
        self.assertEqual(result["status"], "review")
        self.assertIsNone(result["object_id"])
        second = self.store.resolve_review(candidate["event_id"], name="Second bottle")
        self.assertNotEqual(second["object_id"], first_id)
        self.assertEqual(len(self.store.list_objects()), 2)
        self.assertEqual(len(self.store.history(first_id)), 1)


class OrderingAndIsolationTests(MemoryFixture):
    def test_completion_order_does_not_override_observed_order_even_after_restart(self):
        object_id, _ = self.tracked()
        older = self.candidate(at=T0 + 5, action="picked_up")
        newer = self.candidate(at=T0 + 10)
        for candidate in (older, newer):
            self.store.enqueue(candidate)
            self.store.gallery(candidate["event_id"])
        self.store.apply_result(newer["event_id"], self.verification(identity="match", object_id=object_id))
        self.now += 600
        self.store.apply_result(older["event_id"], self.verification(
            action="picked_up", identity="match", object_id=object_id))
        restored = self.reopen()
        obj = restored.get_object(object_id)
        self.assertEqual((obj["state"], obj["latest"]["event_id"]), ("placed", newer["event_id"]))
        self.assertEqual([h["event_id"] for h in restored.history(object_id)][:2],
                         [newer["event_id"], older["event_id"]])

    def test_sequence_orders_results_with_identical_capture_times(self):
        object_id, _ = self.tracked()
        older = self.candidate(at=T0 + 5, action="picked_up", sequence=20)
        newer = self.candidate(at=T0 + 5, sequence=21)
        self.verify(newer, self.verification(identity="match", object_id=object_id))
        self.verify(older, self.verification(action="picked_up", identity="match", object_id=object_id))
        obj = self.store.get_object(object_id)
        self.assertEqual((obj["state"], obj["latest"]["event_id"]), ("placed", newer["event_id"]))
        self.assertEqual([h["sequence"] for h in self.store.history(object_id)][:2], [21, 20])

    def test_queued_higher_sequence_pickup_blocks_location_at_the_same_capture_time(self):
        object_id, first = self.tracked()
        self.store.enqueue(self.candidate(at=first["observed_at"], action="picked_up",
                                          sequence=first["sequence"] + 1))
        obj = self.store.get_object(object_id)
        self.assertEqual(obj["state"], "uncertain")
        self.assertNoCurrentLocation(obj)

    def test_older_queued_observation_does_not_hide_a_newer_verified_placement(self):
        object_id, _ = self.tracked()
        newer = self.candidate(at=T0 + 10)
        self.verify(newer, self.verification(identity="match", object_id=object_id))
        self.store.enqueue(self.candidate(at=T0 + 5, action="picked_up"))
        obj = self.store.get_object(object_id)
        self.assertEqual((obj["state"], obj["latest"]["event_id"]), ("placed", newer["event_id"]))
        self.assertFalse(obj["needs_review"])

    def test_duplicate_payload_is_idempotent_before_after_gallery_and_after_restart(self):
        candidate = self.candidate()
        original = copy.deepcopy(candidate)
        first = self.store.enqueue(candidate)
        self.assertEqual(self.store.enqueue(copy.deepcopy(original)), first)
        self.store.gallery(candidate["event_id"])
        self.assertEqual(self.store.enqueue(original)["status"], "queued")
        result = self.store.apply_result(candidate["event_id"], self.verification())
        object_id = result["object_id"]
        self.store.confirm_object(object_id)
        restored = self.reopen()
        self.assertEqual(restored.enqueue(original)["object_id"], object_id)
        restored.gallery(candidate["event_id"])
        restored.apply_result(candidate["event_id"], self.verification())
        self.assertEqual(len(restored.list_objects()), 1)
        self.assertEqual(len(restored.history(object_id)), 1)
        self.assertEqual(sum(restored.usage()["events"].values()), 1)
        self.assertEqual(candidate, original, "Enqueue must not mutate the caller's evidence")

    def test_reused_event_id_with_changed_evidence_is_rejected(self):
        candidate = self.candidate()
        self.store.enqueue(candidate)
        mutations = [{"observed_at": T0 + 1}, {"sequence": 99}, {"object": "keys"},
                     {"frames": [str(self.root / "different.jpg")]}, {"session_id": SESSION_B},
                     {"evidence": {**candidate["evidence"], "released": False}}]
        for changes in mutations:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.store.enqueue({**copy.deepcopy(candidate), **changes})
        self.assertEqual(sum(self.store.usage()["events"].values()), 1)

    def test_same_track_id_in_another_session_is_not_continuous_identity(self):
        object_id, _ = self.tracked()
        candidate = self.candidate(at=T0 + 5, session=SESSION_B)
        self.store.enqueue(candidate)
        gallery = self.store.gallery(candidate["event_id"])
        self.assertFalse(next(o for o in gallery if o["object_id"] == object_id)["continuous_track"])
        result = self.store.apply_result(candidate["event_id"], self.verification(
            identity="match", object_id=object_id))
        self.assertEqual(result["status"], "review")
        self.assertIsNone(result["object_id"])
        self.assertEqual(len(self.store.history(object_id)), 1)

    def test_profiles_isolate_gallery_search_objects_history_and_reviews(self):
        local_id, local_event = self.tracked()
        other = self.reopen("other-user")
        other_candidate = self.candidate(at=T0 + 5)
        other.enqueue(other_candidate)
        self.assertEqual(other.gallery(other_candidate["event_id"]), [])
        result = other.apply_result(other_candidate["event_id"], self.verification())
        other_id = result["object_id"]
        self.assertEqual([r["event_id"] for r in other.reviews()], [other_candidate["event_id"]])
        self.assertEqual(self.store.reviews(), [])
        other.confirm_object(other_id)
        self.assertEqual([o["object_id"] for o in self.store.search("pill bottle")], [local_id])
        self.assertEqual([o["object_id"] for o in other.search("pill bottle")], [other_id])
        for store, foreign_id in ((self.store, other_id), (other, local_id)):
            for operation in (store.get_object, store.history, store.confirm_object):
                with self.subTest(profile=store.profile_id, operation=operation.__name__), self.assertRaises(KeyError):
                    operation(foreign_id)
        with self.assertRaises(KeyError):
            other.event(local_event["event_id"])
        self.assertEqual(len(self.store.history(local_id)), 1)

    def test_another_profile_cannot_reuse_an_existing_global_event_id(self):
        candidate = self.candidate()
        self.store.enqueue(candidate)
        other = self.reopen("other-user")
        with self.assertRaises(ValueError):
            other.enqueue(copy.deepcopy(candidate))
        self.assertEqual(other.usage()["events"], {})
        self.assertEqual(other.list_objects(include_pending=True), [])


class ValidationTests(MemoryFixture):
    def test_candidate_ids_must_be_strings_with_safe_syntax(self):
        for field in ("event_id", "session_id", "track_id"):
            for value in ("", "../another", "with spaces", "x" * 161, None, 7, True):
                with self.subTest(field=field, value=value):
                    self.assertDeniedCandidate(self.candidate(**{field: value}))

    def test_raw_tracker_number_is_not_a_unique_track_segment(self):
        self.assertDeniedCandidate(self.candidate(track="7"))

    def test_capture_timestamp_must_be_positive_finite_numeric_and_not_boolean(self):
        for value in (0, -1, float("nan"), float("inf"), True, "1000", None):
            with self.subTest(value=value):
                self.assertDeniedCandidate(self.candidate(observed_at=value))
        candidate = self.candidate()
        candidate.pop("observed_at")
        self.assertDeniedCandidate(candidate)

    def test_sequence_is_a_nonnegative_integer_not_a_coercible_value(self):
        for value in (-1, 1.5, True, "2", None):
            with self.subTest(value=value):
                self.assertDeniedCandidate(self.candidate(sequence=value))
        self.assertEqual(self.store.enqueue(self.candidate(sequence=0))["sequence"], 0)

    def test_missing_or_received_time_source_cannot_claim_a_capture_event(self):
        for source in (_UNSET, "received", "video", "", None):
            with self.subTest(source="missing" if source is _UNSET else source):
                candidate = self.candidate()
                if source is _UNSET:
                    candidate.pop("time_source")
                else:
                    candidate["time_source"] = source
                self.assertDeniedCandidate(candidate)

    def test_truthy_strings_or_numbers_cannot_replace_boolean_interaction_evidence(self):
        for field in ("contact_candidate", "motion_valid", "released"):
            for value in (False, "false", "true", 1):
                with self.subTest(field=field, value=value):
                    self.assertDeniedCandidate(self.candidate(evidence={field: value}))

    def test_frames_are_an_optional_list_of_at_most_five_paths(self):
        for frames in ("x.jpg", 7, [None], [{"path": "frame.jpg"}],
                       [str(self.root / f"{i}.jpg") for i in range(6)]):
            with self.subTest(frames=frames):
                self.assertDeniedCandidate(self.candidate(frames=frames))
        for frames in ([], [str(self.root / f"{i}.jpg") for i in range(5)]):
            with self.subTest(valid_frames=len(frames)):
                self.assertEqual(self.store.enqueue(self.candidate(frames=frames))["status"], "queued")
        optional = self.candidate()
        optional.pop("frames")
        optional.pop("crop")
        self.assertEqual(self.store.enqueue(optional)["status"], "queued")

    def test_invalid_profile_and_object_names_do_not_write_partial_state(self):
        for profile in ("", "../outside", "with spaces", "x" * 81):
            with self.subTest(profile=profile), self.assertRaises(ValueError):
                self.reopen(profile)
        object_id, _ = self.tracked()
        for name in ("", "   ", "x" * 101, 12):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.store.confirm_object(object_id, name=name)
        self.assertEqual(self.store.get_object(object_id)["name"], "pill bottle")
        candidate = self.candidate(at=T0 + 5, track=TRACK_B)
        self.verify(candidate, self.verification(identity="uncertain"))
        with self.assertRaises(ValueError):
            self.store.resolve_review(candidate["event_id"], name=" ")
        self.assertEqual(len(self.store.list_objects(include_pending=True)), 1)
        self.assertEqual(self.store.event(candidate["event_id"])["status"], "review")

    def test_invalid_verification_and_model_invented_geo_are_not_persisted(self):
        candidate = self.candidate()
        self.store.enqueue(candidate)
        self.store.gallery(candidate["event_id"])
        valid = self.verification().model_dump()
        cases = [{field: value} for field in ("actor_confidence", "action_confidence", "identity_confidence")
                 for value in (-.01, 1.01, float("nan"), float("inf"))]
        cases += [{"actor": "owner"}, {"action": "vanished"},
                  {"location": {"lat": 3, "lon": 4}}, {"maps_url": "https://example.invalid/invented"}]
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.store.apply_result(candidate["event_id"], {**valid, **changes})
        self.assertEqual(self.store.event(candidate["event_id"])["status"], "queued")
        self.assertEqual(self.store.list_objects(include_pending=True), [])


class LocationProvenanceTests(MemoryFixture):
    def test_location_uses_capture_time_not_delayed_completion_or_mutated_input(self):
        candidate = self.candidate(location={
            "lat": 0.0, "lon": 0.0, "accuracy_m": 9.0, "observed_at": T0 - 5})
        self.store.enqueue(candidate)
        candidate["location"].update(lat=30, lon=40, observed_at=T0 + 3599)
        self.now = T0 + 3600
        self.store.gallery(candidate["event_id"])
        result = self.store.apply_result(candidate["event_id"], self.verification())
        obj = self.store.confirm_object(result["object_id"])
        self.assertEqual(obj["latest"]["observed_at"], T0)
        loc = obj["location"]
        self.assertEqual((loc["lat"], loc["lon"], loc["observed_at"], loc["age_s"]), (0.0, 0.0, T0 - 5, 5.0))
        self.assertIn("query=0.000000,0.000000", loc["maps_url"])
        restored = self.reopen().get_object(obj["object_id"])
        self.assertEqual(restored["location"], loc)

    def test_denied_missing_or_invented_location_never_invents_a_map_or_home(self):
        cases = [None, {}, {"status": "denied"}, {"lat": 1.0},
                 {"place": "home", "maps_url": "https://example.invalid/guess"}]
        for fix in cases:
            with self.subTest(fix=fix):
                candidate = self.candidate(location=fix)
                event = self.store.enqueue(candidate)
                self.assertNotEqual(event["location"]["status"], "available")
                self.assertNoMap(event["location"])
                self.assertNotIn("place", event["location"])
        object_id, _ = self.tracked(location=None)
        obj = self.store.get_object(object_id)
        self.assertEqual(obj["state"], "placed", "A visual resting place does not require GPS")
        self.assertNoMap(obj["location"])

    def test_freshness_and_accuracy_boundaries_are_relative_to_observation(self):
        base = {"lat": 1.0, "lon": 2.0, "accuracy_m": FIX_MAX_ACCURACY_M,
                "observed_at": T0 - FIX_MAX_AGE_S}
        cases = [(base, "available"), ({**base, "observed_at": T0 - FIX_MAX_AGE_S - .01}, "stale"),
                 ({**base, "observed_at": T0 + .01}, "future"),
                 ({**base, "accuracy_m": FIX_MAX_ACCURACY_M + .01}, "inaccurate")]
        for fix, expected in cases:
            with self.subTest(expected=expected):
                event = self.store.enqueue(self.candidate(location=fix))
                self.assertEqual(event["location"]["status"], expected)
                if expected != "available":
                    self.assertNoMap(event["location"])

    def test_nonfinite_out_of_range_or_unmeasured_fixes_have_no_map(self):
        base = {"lat": 1.0, "lon": 2.0, "accuracy_m": 5.0, "observed_at": T0 - 1}
        changes = [{"lat": value} for value in (float("nan"), float("inf"), 91, -91, True, "1")]
        changes += [{"lon": 181}, {"lon": -181}, {"accuracy_m": -1}, {"accuracy_m": None},
                    {"accuracy_m": float("nan")}, {"observed_at": 0}, {"observed_at": True}]
        for change in changes:
            with self.subTest(change=change):
                event = self.store.enqueue(self.candidate(location={**base, **change}))
                self.assertEqual(event["location"]["status"], "invalid")
                self.assertNoMap(event["location"])

    def test_normalized_capture_fix_retains_unavailable_reason_through_store(self):
        base = {"lat": 1.0, "lon": 2.0, "accuracy_m": 5.0, "observed_at": T0 - 1}
        fixes = [None, {**base, "observed_at": T0 - FIX_MAX_AGE_S - 1},
                 {**base, "observed_at": T0 + 1}, {**base, "accuracy_m": FIX_MAX_ACCURACY_M + 1}]
        for fix in fixes:
            normalized = location_at(fix, T0)
            with self.subTest(status=normalized["status"]):
                event = self.store.enqueue(self.candidate(location=normalized))
                self.assertEqual(event["location"]["status"], normalized["status"])
                self.assertNoMap(event["location"])

    def test_nearby_map_guess_is_not_upgraded_to_a_known_place(self):
        fix = {"lat": 1.0, "lon": 2.0, "accuracy_m": 5.0, "observed_at": T0 - 1,
               "place": "Invented pharmacy", "place_source": "google",
               "maps_url": "https://example.invalid/incorrect"}
        event = self.store.enqueue(self.candidate(location=fix))
        self.assertEqual(event["location"]["status"], "available")
        self.assertNotIn("place", event["location"])
        self.assertIn("query=1.000000,2.000000", event["location"]["maps_url"])
        self.assertNotIn("example.invalid", event["location"]["maps_url"])


class QueueAndBudgetTests(MemoryFixture):
    def test_daily_budget_wait_survives_restart_and_is_claimable_next_day(self):
        first, second = self.candidate(), self.candidate(at=T0 + 1, track=TRACK_B)
        self.store.enqueue(first)
        self.store.enqueue(second)
        self.assertEqual(self.store.claim_next(daily_calls=1)["event_id"], first["event_id"])
        self.store.fail(first["event_id"], "synthetic timeout")
        self.assertIsNone(self.store.claim_next(daily_calls=1))
        self.assertEqual(self.store.event(second["event_id"])["status"], "budget_wait")
        restored = self.reopen()
        self.assertIsNone(restored.claim_next(daily_calls=1))
        self.assertEqual(restored.usage()["api_calls"], 1)
        self.now += 86400
        claimed = restored.claim_next(daily_calls=1)
        self.assertIsNotNone(claimed)
        self.assertEqual(claimed["event_id"], second["event_id"])
        self.assertEqual(restored.usage()["api_calls"], 2)

    def test_monthly_budget_persists_across_restart_and_daily_reset(self):
        first, second = self.candidate(), self.candidate(at=T0 + 1, track=TRACK_B)
        self.store.enqueue(first)
        self.store.enqueue(second)
        self.store.claim_next(monthly_usd=.15, reserve_usd=.10)
        self.store.gallery(first["event_id"])
        self.store.apply_result(first["event_id"], self.verification(actor="bystander"), cost_usd=.10)
        self.now += 86400
        restored = self.reopen()
        self.assertIsNone(restored.claim_next(monthly_usd=.15, reserve_usd=.10))
        self.assertEqual(restored.usage()["api_calls"], 1)
        self.now = datetime(2026, 2, 1, 12, tzinfo=timezone.utc).timestamp()
        self.assertEqual(restored.claim_next(monthly_usd=.15, reserve_usd=.10)["event_id"], second["event_id"])
        self.assertEqual(restored.usage()["api_calls"], 2)

    def test_reservation_blocks_overspend_and_actual_cost_releases_only_unused_budget(self):
        first, second = self.candidate(), self.candidate(at=T0 + 1, track=TRACK_B)
        self.store.enqueue(first)
        self.store.enqueue(second)
        self.store.claim_next(monthly_usd=.15, reserve_usd=.10)
        self.assertAlmostEqual(self.store.usage()["cost_or_reserved_usd"], .10)
        self.assertIsNone(self.store.claim_next(monthly_usd=.15, reserve_usd=.10))
        self.store.gallery(first["event_id"])
        self.store.apply_result(first["event_id"], self.verification(actor="bystander"), cost_usd=.01)
        claimed = self.store.claim_next(monthly_usd=.15, reserve_usd=.10)
        self.assertEqual(claimed["event_id"], second["event_id"])
        self.assertAlmostEqual(self.store.usage()["cost_or_reserved_usd"], .11)

    def test_failed_pickup_remains_reviewable_and_never_restores_old_location(self):
        object_id, _ = self.tracked()
        pickup = self.candidate(at=T0 + 5, action="picked_up")
        self.store.enqueue(pickup)
        self.assertEqual(self.store.claim_next()["event_id"], pickup["event_id"])
        self.store.fail(pickup["event_id"], "synthetic provider unavailable")
        restored = self.reopen()
        self.assertNoCurrentLocation(restored.get_object(object_id))
        self.assertIn(pickup["event_id"], [r["event_id"] for r in restored.reviews()])
        self.assertEqual(restored.usage()["api_calls"], 1)
        self.assertAlmostEqual(restored.usage()["cost_or_reserved_usd"], .10)

    def test_active_processing_is_not_double_claimed_when_another_store_opens(self):
        candidate = self.candidate()
        self.store.enqueue(candidate)
        self.store.claim_next()
        restored = self.reopen()
        self.assertIsNone(restored.claim_next())
        self.assertEqual(restored.event(candidate["event_id"])["status"], "processing")
        self.assertEqual(restored.usage()["api_calls"], 1)

    def test_expired_processing_after_crash_is_reclaimable_without_losing_reserved_cost(self):
        candidate = self.candidate()
        self.store.enqueue(candidate)
        self.store.claim_next(monthly_usd=1.0, reserve_usd=.10)
        self.now += 86400  # Far beyond a normal provider call/lease; no real sleep or process crash.
        restored = self.reopen()
        recovered = restored.claim_next(monthly_usd=1.0, reserve_usd=.10)
        self.assertIsNotNone(recovered, "Crashed work must not remain permanently processing")
        self.assertEqual(recovered["event_id"], candidate["event_id"])
        self.assertEqual(restored.usage()["api_calls"], 2)
        self.assertAlmostEqual(restored.usage()["cost_or_reserved_usd"], .20)
        restored.gallery(candidate["event_id"])
        restored.apply_result(candidate["event_id"], self.verification(actor="bystander"), cost_usd=.02)
        self.assertAlmostEqual(restored.usage()["cost_or_reserved_usd"], .12,
                               msg="Settling this attempt must not refund the crashed attempt's unknown cost")

    def test_capacity_overflow_remains_durable_and_is_claimed_after_capacity_frees(self):
        first, overflow = self.candidate(), self.candidate(at=T0 + 1, track=TRACK_B)
        self.store.enqueue(first, max_pending=1)
        self.store.enqueue(overflow, max_pending=1)
        self.assertEqual(sum(self.store.usage()["events"].values()), 2)
        self.assertEqual(self.store.claim_next()["event_id"], first["event_id"])
        self.store.gallery(first["event_id"])
        self.store.apply_result(first["event_id"], self.verification(actor="bystander"), cost_usd=.01)
        restored = self.reopen()
        claimed = restored.claim_next()
        self.assertIsNotNone(claimed, "Capacity is backpressure, not an unresolvable identity review")
        self.assertEqual(claimed["event_id"], overflow["event_id"])

    def test_capacity_delayed_pickup_also_blocks_the_previous_location(self):
        object_id, _ = self.tracked()
        self.store.enqueue(self.candidate(at=T0 + 1, track=TRACK_B), max_pending=1)
        pickup = self.candidate(at=T0 + 5, action="picked_up")
        self.store.enqueue(pickup, max_pending=1)
        obj = self.store.get_object(object_id)
        self.assertEqual(obj["state"], "uncertain")
        self.assertNoCurrentLocation(obj)

    def test_concurrent_claims_cannot_exceed_the_same_profile_daily_budget(self):
        self.store.enqueue(self.candidate())
        self.store.enqueue(self.candidate(at=T0 + 1, track=TRACK_B))
        stores = [self.reopen(), self.reopen()]
        barrier = threading.Barrier(2)

        def claim(store):
            barrier.wait(timeout=10)
            return store.claim_next(daily_calls=1, monthly_usd=1, reserve_usd=.10)

        with ThreadPoolExecutor(max_workers=2) as executor:
            claims = list(executor.map(claim, stores))
        self.assertEqual(sum(c is not None for c in claims), 1)
        self.assertEqual(self.store.usage()["api_calls"], 1)
        self.assertAlmostEqual(self.store.usage()["cost_or_reserved_usd"], .10)

    def test_budgets_are_partitioned_by_profile_even_in_the_same_database(self):
        other = self.reopen("other-user")
        self.store.enqueue(self.candidate())
        other_candidate = self.candidate(track=TRACK_B)
        other.enqueue(other_candidate)
        self.store.claim_next(daily_calls=1, monthly_usd=.1)
        self.assertEqual(other.usage()["api_calls"], 0)
        self.assertEqual(other.claim_next(daily_calls=1, monthly_usd=.1)["event_id"], other_candidate["event_id"])
        self.assertEqual(self.store.usage()["api_calls"], 1)
        self.assertEqual(other.usage()["api_calls"], 1)

    def test_result_replay_does_not_double_charge_or_change_settled_cost(self):
        candidate = self.candidate()
        self.store.enqueue(candidate)
        self.store.claim_next()
        self.store.gallery(candidate["event_id"])
        result = self.store.apply_result(candidate["event_id"], self.verification(), cost_usd=.02)
        object_id = result["object_id"]
        self.store.confirm_object(object_id)
        restored = self.reopen()
        restored.enqueue(candidate)
        restored.apply_result(candidate["event_id"], self.verification(), cost_usd=.09)
        self.assertEqual(restored.usage()["api_calls"], 1)
        self.assertAlmostEqual(restored.usage()["cost_or_reserved_usd"], .02)
        self.assertEqual(len(restored.history(object_id)), 1)

    def test_invalid_budget_parameters_fail_before_claiming_or_reserving(self):
        self.store.enqueue(self.candidate())
        cases = [{"daily_calls": v} for v in (-1, float("nan"), float("inf"), True, 1.5)]
        cases += [{"monthly_usd": v} for v in (-1, float("nan"), float("inf"), True)]
        cases += [{"reserve_usd": v} for v in (0, -1, float("nan"), float("inf"), True)]
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.store.claim_next(**changes)
        self.assertEqual(self.store.usage()["api_calls"], 0)

    def test_invalid_reported_cost_does_not_corrupt_reserved_usage(self):
        candidate = self.candidate()
        self.store.enqueue(candidate)
        self.store.claim_next()
        self.store.gallery(candidate["event_id"])
        for cost in (-1, float("nan"), float("inf"), True, "0.01"):
            with self.subTest(cost=cost), self.assertRaises(ValueError):
                self.store.apply_result(candidate["event_id"], self.verification(), cost_usd=cost)
        self.assertEqual(self.store.event(candidate["event_id"])["status"], "processing")
        self.assertAlmostEqual(self.store.usage()["cost_or_reserved_usd"], .10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
