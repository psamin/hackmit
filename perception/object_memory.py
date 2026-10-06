from __future__ import annotations

import hashlib
import json
import math
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import re
import sqlite3
import sys
import time
import uuid

from pydantic import BaseModel, ConfigDict, Field
from typing import Literal

try:
    from .capture import finite_number, location_at
except ImportError:
    from capture import finite_number, location_at


class InteractionEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    contact_candidate: bool = False
    motion_valid: bool = False
    released: bool = False
    track_continuous: bool = False
    track_started_at: float | None = None
    contact_started_at: float | None = None
    contact_ended_at: float | None = None
    release_at: float | None = None
    contact_source: str = Field(default="person_box_overlap", max_length=80)
    contact_duration_s: float = Field(default=0, ge=0)


class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    schema_version: Literal[2] = 2
    event_id: str = Field(min_length=1, max_length=160, pattern=r"^[A-Za-z0-9_:-]+$")
    session_id: str = Field(min_length=1, max_length=160, pattern=r"^[A-Za-z0-9_:-]+$")
    track_id: str = Field(min_length=8, max_length=160, pattern=r"^[A-Za-z0-9_:-]+$")
    observed_at: float = Field(gt=0)
    time_source: Literal["capture"]
    sequence: int = Field(ge=0)
    event_type: str = Field(max_length=40)
    object: str = Field(default="object", min_length=1, max_length=100)
    evidence: InteractionEvidence
    co_visible_track_ids: list[str] = Field(default_factory=list, max_length=100)
    frames: list[str] = Field(default_factory=list, max_length=5)
    frame_times: list[float] = Field(default_factory=list, max_length=5)
    crop: str | None = None
    box: list[float] = Field(default_factory=list, max_length=4)
    location: dict = Field(default_factory=lambda: {"status": "missing"})


class Scene(BaseModel):
    model_config = ConfigDict(extra="forbid")
    room: str = Field(default="", max_length=120)
    surface: str = Field(default="", max_length=160)
    landmarks: list[str] = Field(default_factory=list, max_length=8)
    description: str = Field(default="", max_length=600)


class Verification(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    actor: Literal["wearer", "bystander", "unknown"]
    actor_confidence: float = Field(ge=0, le=1)
    actor_evidence: list[str] = Field(default_factory=list, max_length=8)
    action: Literal["placed", "picked_up", "no_change", "unclear"]
    action_confidence: float = Field(ge=0, le=1)
    contact_observed: bool
    released: bool = False
    resting_after: bool = False
    object: str = Field(min_length=1, max_length=100)
    appearance: str = Field(default="", max_length=600)
    identity: Literal["new", "match", "uncertain"]
    matched_object_id: str | None = Field(default=None, max_length=80)
    identity_confidence: float = Field(default=0, ge=0, le=1)
    identity_basis: Literal["continuous_track", "distinctive_features", "category_only", "uncertain"] = "uncertain"
    identity_evidence: str = Field(default="", max_length=600)
    scene: Scene = Field(default_factory=Scene)
    reason: str = Field(default="", max_length=600)


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _words(text):
    return set(re.findall(r"[\w]+", text.lower())) - {"my", "the", "a", "an", "is", "where", "did", "i", "leave"}


def _private_file(path):
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)
    from server.schedule import private_append_fd
    fd = private_append_fd(path)
    os.close(fd)


class ObjectStore:
    def __init__(self, path, profile_id="local", clock=time.time, create=True):
        self.path, self.profile_id, self.clock = Path(path).resolve(), str(profile_id), clock
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", self.profile_id):
            raise ValueError("Invalid profile ID")
        if not create and not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS objects (
                    object_id TEXT PRIMARY KEY, profile_id TEXT NOT NULL, name TEXT NOT NULL,
                    category TEXT NOT NULL, status TEXT NOT NULL, created_at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS candidates (
                    event_id TEXT PRIMARY KEY, profile_id TEXT NOT NULL, session_id TEXT NOT NULL,
                    track_id TEXT NOT NULL, observed_at REAL NOT NULL, payload TEXT NOT NULL,
                    digest TEXT NOT NULL, status TEXT NOT NULL, reason TEXT NOT NULL DEFAULT '',
                    verification TEXT, object_id TEXT, created_at REAL NOT NULL,
                    sequence INTEGER NOT NULL, active_call_id INTEGER);
                CREATE TABLE IF NOT EXISTS observations (
                    event_id TEXT PRIMARY KEY, profile_id TEXT NOT NULL, object_id TEXT NOT NULL,
                    observed_at REAL NOT NULL, sequence INTEGER NOT NULL, action TEXT NOT NULL,
                    record TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS observations_object_time
                    ON observations(profile_id, object_id, observed_at, sequence);
                CREATE TABLE IF NOT EXISTS bindings (
                    profile_id TEXT NOT NULL, session_id TEXT NOT NULL, track_id TEXT NOT NULL,
                    object_id TEXT NOT NULL, observed_at REAL NOT NULL, first_observed_at REAL NOT NULL,
                    PRIMARY KEY(profile_id, session_id, track_id));
                CREATE TABLE IF NOT EXISTS api_calls (
                    call_id INTEGER PRIMARY KEY, profile_id TEXT NOT NULL, event_id TEXT NOT NULL,
                    attempted_at REAL NOT NULL, reserved_usd REAL NOT NULL, cost_usd REAL);
                CREATE TABLE IF NOT EXISTS decisions (
                    decision_id INTEGER PRIMARY KEY, profile_id TEXT NOT NULL, at REAL NOT NULL,
                    action TEXT NOT NULL, event_id TEXT, object_id TEXT, detail TEXT NOT NULL);
                PRAGMA user_version=1;
            """)

    @contextmanager
    def _connection(self):
        _private_file(self.path)
        _private_file(Path(str(self.path) + "-journal"))
        db = sqlite3.connect(str(self.path), timeout=10)
        try:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA journal_mode=PERSIST")
            db.execute("PRAGMA synchronous=FULL")
            db.execute("PRAGMA temp_store=MEMORY")
            db.execute("PRAGMA foreign_keys=ON")
            if db.execute("PRAGMA user_version").fetchone()[0] not in (0, 1):
                raise ValueError("Unsupported object database version; do not overwrite it")
            with db:
                yield db
        finally:
            db.close()

    def _candidate(self, db, event_id):
        row = db.execute("SELECT * FROM candidates WHERE event_id=? AND profile_id=?",
                         (event_id, self.profile_id)).fetchone()
        if row is None:
            raise KeyError("Unknown event")
        return row

    def _binding(self, db, candidate):
        evidence = candidate.get("evidence", {})
        start = evidence.get("track_started_at")
        if evidence.get("track_continuous") is not True or start is None:
            return None
        row = db.execute("SELECT b.object_id,b.first_observed_at FROM bindings b JOIN objects o ON b.object_id=o.object_id WHERE b.profile_id=? AND b.session_id=? AND b.track_id=? AND o.status!='ignored'",
                         (self.profile_id, candidate["session_id"], candidate["track_id"])).fetchone()
        return row["object_id"] if row and start <= row["first_observed_at"] <= candidate["observed_at"] else None

    def _identity_conflict(self, db, candidate, object_id, category):
        obj = db.execute("SELECT category FROM objects WHERE object_id=? AND profile_id=? AND status!='ignored'", (object_id, self.profile_id)).fetchone()
        if obj is None or self._category(obj["category"]) != self._category(category):
            return True
        bindings = db.execute("SELECT track_id,object_id FROM bindings WHERE profile_id=? AND session_id=?", (self.profile_id, candidate["session_id"])).fetchall()
        visible = set(candidate.get("co_visible_track_ids", []))
        return any((r["track_id"] == candidate["track_id"] and r["object_id"] != object_id)
                   or (r["track_id"] != candidate["track_id"] and r["track_id"] in visible and r["object_id"] == object_id)
                   for r in bindings)

    @staticmethod
    def _category(label):
        label = " ".join(sorted(_words(label)))
        aliases = {"medicine": "bottle pill", "medication": "bottle pill", "meds": "bottle pill", "pills": "bottle pill",
                   "bottle prescription": "bottle pill", "spectacles": "glasses", "eyeglasses": "glasses", "key": "keys", "keychain": "keys",
                   "cellphone": "phone", "mobile phone": "phone", "smartphone": "phone"}
        return aliases.get(label, label)

    def _audit(self, db, action, event_id=None, object_id=None, detail=None):
        db.execute("INSERT INTO decisions(profile_id,at,action,event_id,object_id,detail) VALUES(?,?,?,?,?,?)",
                   (self.profile_id, self.clock(), action, event_id, object_id, _json(detail or {})))

    def _media_path(self, path):
        if not isinstance(path, str) or not path or len(path) > 1024:
            raise ValueError("Invalid media path")
        resolved = Path(path).resolve()
        if not resolved.is_relative_to(self.path.parent) or resolved.suffix.lower() not in {".jpg", ".jpeg"}:
            raise ValueError("Media must be JPEGs within the object database's artifact directory")
        return str(resolved)

    def enqueue(self, candidate, max_pending=128):
        if type(max_pending) is not int or max_pending < 1:
            raise ValueError("Invalid pending limit")
        if not isinstance(candidate, dict):
            raise ValueError("Invalid candidate")
        payload = dict(candidate)
        observed_at = finite_number(payload.get("observed_at"))
        payload["location"] = location_at(payload.get("location"), observed_at)
        payload = Candidate.model_validate(payload).model_dump()
        for path in payload["frames"]:
            self._media_path(path)
        if payload["crop"] is not None:
            self._media_path(payload["crop"])
        if len(_json(payload)) > 64000:
            raise ValueError("Candidate too large")
        digest = hashlib.sha256(_json(payload).encode()).hexdigest()
        status, reason = "queued", ""
        evidence = payload["evidence"]
        if payload["event_type"] not in ("placed", "picked_up"):
            status, reason = "rejected", "background_sighting"
        elif not evidence["contact_candidate"] or not evidence["motion_valid"]:
            status, reason = "rejected", "insufficient_interaction_evidence"
        elif payload["event_type"] == "placed" and not evidence["released"]:
            status, reason = "rejected", "release_not_observed"
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            previous = db.execute("SELECT digest, profile_id FROM candidates WHERE event_id=?", (payload["event_id"],)).fetchone()
            if previous:
                if previous["profile_id"] != self.profile_id or previous["digest"] != digest:
                    raise ValueError("Event ID was reused with different evidence")
                return self.event(payload["event_id"], db=db)
            count = db.execute("SELECT COUNT(*) FROM candidates WHERE profile_id=? AND status IN ('queued','processing','budget_wait','capacity_wait')",
                               (self.profile_id,)).fetchone()[0]
            if status == "queued" and count >= max_pending:
                status, reason = "capacity_wait", "queue_capacity"
            if status == "capacity_wait" and count >= max(512, max_pending * 4):
                status, reason = "dropped", "storage_backpressure"
            if status in ("rejected", "dropped"):
                payload.pop("frames", None)
                payload.pop("crop", None)
                payload.pop("location", None)
            db.execute("INSERT INTO candidates(event_id,profile_id,session_id,track_id,observed_at,payload,digest,status,reason,created_at,sequence) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                       (payload["event_id"], self.profile_id, payload["session_id"], payload["track_id"], observed_at,
                        _json(payload), digest, status, reason, self.clock(), payload["sequence"]))
            return self.event(payload["event_id"], db=db)

    def event(self, event_id, db=None):
        if db is None:
            with self._connection() as connection:
                return self.event(event_id, db=connection)
        row = self._candidate(db, event_id)
        return {**json.loads(row["payload"]), "status": row["status"], "reason": row["reason"],
                "object_id": row["object_id"], "call_id": row["active_call_id"],
                "verification": json.loads(row["verification"]) if row["verification"] else None}

    def gallery(self, event_id, limit=6):
        with self._connection() as db:
            row = self._candidate(db, event_id)
            candidate = json.loads(row["payload"])
            bound = self._binding(db, candidate)
            objects = db.execute("SELECT * FROM objects WHERE profile_id=? AND (status='tracked' OR (status='pending' AND object_id=?))", (self.profile_id, bound)).fetchall()
            words = _words(candidate["object"])
            objects = sorted(objects, key=lambda o: (o["object_id"] == bound, len(words & _words(o["category"] + " " + o["name"]))), reverse=True)
            selected = []
            for obj in objects[:limit]:
                observation = db.execute("SELECT record FROM observations WHERE profile_id=? AND object_id=? ORDER BY observed_at DESC, sequence DESC LIMIT 1",
                                         (self.profile_id, obj["object_id"])).fetchone()
                if observation:
                    record = json.loads(observation[0])
                    selected.append({"object_id": obj["object_id"], "name": obj["name"], "category": obj["category"],
                                     "appearance": record.get("appearance", ""), "crop": record.get("crop"),
                                     "continuous_track": obj["object_id"] == bound})
            candidate.update(gallery_ids=[o["object_id"] for o in selected], gallery_complete=len(objects) <= limit,
                             bound_object_id=bound)
            db.execute("UPDATE candidates SET payload=? WHERE event_id=? AND profile_id=?",
                       (_json(candidate), event_id, self.profile_id))
            return selected

    def claim_next(self, daily_calls=200, monthly_usd=5.0, reserve_usd=0.10):
        if type(daily_calls) is not int or daily_calls < 0 or finite_number(monthly_usd) < 0 or finite_number(reserve_usd) <= 0:
            raise ValueError("Invalid VLM budget")
        now = self.clock()
        day = datetime.fromtimestamp(now, timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        month = day.replace(day=1).timestamp()
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("UPDATE candidates SET status='queued',reason='expired_lease' WHERE profile_id=? AND status='processing' AND active_call_id IN (SELECT call_id FROM api_calls WHERE attempted_at<?)", (self.profile_id, now - 300))
            db.execute("UPDATE candidates SET status='failed',reason='attempt_limit' WHERE profile_id=? AND status='queued' AND (SELECT COUNT(*) FROM api_calls a WHERE a.event_id=candidates.event_id)>=2", (self.profile_id,))
            row = db.execute("SELECT event_id FROM candidates WHERE profile_id=? AND status IN ('queued','budget_wait','capacity_wait') ORDER BY observed_at,sequence,event_id LIMIT 1",
                             (self.profile_id,)).fetchone()
            if row is None:
                return None
            calls = db.execute("SELECT COUNT(*) FROM api_calls WHERE profile_id=? AND attempted_at>=?", (self.profile_id, day.timestamp())).fetchone()[0]
            spent = db.execute("SELECT COALESCE(SUM(COALESCE(cost_usd,reserved_usd)),0) FROM api_calls WHERE profile_id=? AND attempted_at>=?",
                               (self.profile_id, month)).fetchone()[0]
            if calls >= daily_calls or spent + reserve_usd > monthly_usd + 1e-9:
                db.execute("UPDATE candidates SET status='budget_wait',reason='api_budget' WHERE event_id=? AND profile_id=?", (row[0], self.profile_id))
                return None
            call_id = db.execute("INSERT INTO api_calls(profile_id,event_id,attempted_at,reserved_usd) VALUES(?,?,?,?)", (self.profile_id, row[0], now, reserve_usd)).lastrowid
            db.execute("UPDATE candidates SET status='processing',reason='',active_call_id=? WHERE event_id=? AND profile_id=?", (call_id, row[0], self.profile_id))
            return self.event(row[0], db=db)

    @staticmethod
    def _action_verified(result):
        return (result.actor == "wearer" and result.actor_confidence >= .85 and bool(result.actor_evidence)
                and result.action_confidence >= .85 and result.contact_observed
                and (result.action == "picked_up" or (result.action == "placed" and result.released and result.resting_after)))

    def apply_result(self, event_id, result, cost_usd=0.0, call_id=None, usage=None):
        result = result if isinstance(result, Verification) else Verification.model_validate(result)
        if finite_number(cost_usd) < 0:
            raise ValueError("Invalid API cost")
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = self._candidate(db, event_id)
            if row["status"] not in ("queued", "processing", "budget_wait"):
                return self.event(event_id, db=db)
            if call_id is not None and (row["status"] != "processing" or call_id != row["active_call_id"]):
                raise ValueError("Expired or invalid verification claim")
            if cost_usd and row["active_call_id"] is None:
                raise ValueError("Cloud results require a reserved API call")
            candidate = json.loads(row["payload"])
            db.execute("UPDATE api_calls SET cost_usd=? WHERE call_id=? AND profile_id=? AND cost_usd IS NULL", (cost_usd, row["active_call_id"], self.profile_id))
            db.execute("UPDATE candidates SET verification=? WHERE event_id=? AND profile_id=?", (_json(result.model_dump()), event_id, self.profile_id))
            status, reason, object_id = "review", "uncertain_action_or_actor", None
            if result.actor == "bystander" or result.action == "no_change":
                status, reason = "rejected", "not_wearer_placement_or_pickup"
                bound = self._binding(db, candidate)
                if bound and result.actor == "bystander" and result.actor_confidence >= .85 and result.action in ("placed", "picked_up") and result.action_confidence >= .85:
                    self._observe(db, candidate, result, bound, action="unknown")
            elif self._action_verified(result):
                if result.action != candidate["event_type"]:
                    reason = "action_conflicts_with_candidate"
                elif result.identity == "match":
                    object_id = result.matched_object_id
                    basis_ok = result.identity_basis == "distinctive_features" or (result.identity_basis == "continuous_track" and object_id == self._binding(db, candidate))
                    allowed = candidate.get("gallery_ids", [])
                    complete = candidate.get("gallery_complete") is True or (object_id is not None and object_id == self._binding(db, candidate))
                    if object_id in allowed and complete and result.identity_confidence >= .90 and basis_ok and result.identity_evidence and not self._identity_conflict(db, candidate, object_id, result.object):
                        self._observe(db, candidate, result, object_id)
                        status, reason = "verified", "matched_instance"
                    else:
                        object_id, reason = None, "ambiguous_identity"
                elif result.identity == "new" and result.action == "placed" and candidate.get("gallery_complete") is True:
                    if self._binding(db, candidate) or (candidate.get("gallery_ids") and (result.identity_confidence < .90 or result.identity_basis != "distinctive_features" or not result.identity_evidence)):
                        reason = "ambiguous_identity"
                    else:
                        object_id = self._create_object(db, result.object, "pending", candidate["observed_at"])
                        self._observe(db, candidate, result, object_id)
                        status, reason = "review", "confirm_tracking"
                else:
                    reason = "ambiguous_identity"
            db.execute("UPDATE candidates SET status=?,reason=?,object_id=? WHERE event_id=? AND profile_id=?",
                       (status, reason, object_id, event_id, self.profile_id))
            self._audit(db, "verification", event_id, object_id, {"status": status, "reason": reason, "api_usage": usage or {}, "cost_usd": cost_usd, "call_id": row["active_call_id"]})
            return self.event(event_id, db=db)

    def _create_object(self, db, name, status, observed_at):
        object_id = str(uuid.uuid4())
        db.execute("INSERT INTO objects VALUES(?,?,?,?,?,?)", (object_id, self.profile_id, name, name, status, observed_at))
        return object_id

    def _observe(self, db, candidate, result, object_id, action=None, user_confirmed=False):
        obj = db.execute("SELECT object_id FROM objects WHERE object_id=? AND profile_id=? AND status!='ignored'", (object_id, self.profile_id)).fetchone()
        if obj is None:
            raise ValueError("Unknown object identity")
        action = action or result.action
        record = {"schema_version": 2, "event_id": candidate["event_id"], "object_id": object_id,
                  "session_id": candidate["session_id"], "track_id": candidate["track_id"],
                  "observed_at": candidate["observed_at"], "sequence": candidate["sequence"], "action": action,
                  "actor": result.actor, "actor_confidence": result.actor_confidence,
                  "action_confidence": result.action_confidence, "identity_confidence": result.identity_confidence,
                  "identity_basis": "user_confirmation" if user_confirmed else result.identity_basis,
                  "object": result.object, "appearance": result.appearance, "scene": result.scene.model_dump(),
                  "location": candidate.get("location", {"status": "missing"}),
                  "frames": candidate.get("frames", []), "crop": candidate.get("crop")}
        previous = db.execute("SELECT object_id,record FROM observations WHERE event_id=?", (candidate["event_id"],)).fetchone()
        if previous:
            if previous["object_id"] != object_id or previous["record"] != _json(record):
                raise ValueError("An observation cannot be silently rewritten")
            return
        binding = db.execute("SELECT object_id FROM bindings WHERE profile_id=? AND session_id=? AND track_id=?", (self.profile_id, candidate["session_id"], candidate["track_id"])).fetchone()
        if binding and binding[0] != object_id:
            raise ValueError("Track identity conflict; use a new track segment")
        db.execute("INSERT INTO observations VALUES(?,?,?,?,?,?,?)", (candidate["event_id"], self.profile_id, object_id,
                   candidate["observed_at"], candidate["sequence"], action, _json(record)))
        db.execute("INSERT INTO bindings VALUES(?,?,?,?,?,?) ON CONFLICT(profile_id,session_id,track_id) DO UPDATE SET observed_at=MAX(bindings.observed_at,excluded.observed_at),first_observed_at=MIN(bindings.first_observed_at,excluded.first_observed_at)",
                   (self.profile_id, candidate["session_id"], candidate["track_id"], object_id, candidate["observed_at"], candidate["observed_at"]))

    def confirm_object(self, object_id, name=None):
        if name is not None and (not isinstance(name, str) or not name.strip() or len(name.strip()) > 100):
            raise ValueError("Name must contain 1-100 characters")
        with self._connection() as db:
            row = db.execute("SELECT * FROM objects WHERE object_id=? AND profile_id=? AND status!='ignored'", (object_id, self.profile_id)).fetchone()
            if row is None:
                raise KeyError("Unknown object")
            db.execute("UPDATE objects SET status='tracked',name=? WHERE object_id=?", ((name or row["name"]).strip(), object_id))
            db.execute("UPDATE candidates SET status='verified',reason='tracking_confirmed' WHERE profile_id=? AND object_id=? AND reason='confirm_tracking'", (self.profile_id, object_id))
            self._audit(db, "confirm_tracking", object_id=object_id, detail={"name": name or row["name"]})
        return self.get_object(object_id)

    def resolve_review(self, event_id, object_id=None, name=None):
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = self._candidate(db, event_id)
            if row["status"] != "review" or not row["verification"]:
                raise ValueError("Only verified interaction reviews can be resolved")
            result = Verification.model_validate_json(row["verification"])
            if not self._action_verified(result) or result.action != json.loads(row["payload"])["event_type"]:
                raise ValueError("Wearer/action evidence is insufficient")
            candidate = json.loads(row["payload"])
            if row["object_id"]:
                if object_id and object_id != row["object_id"]:
                    raise ValueError("An existing instance cannot be silently merged")
                object_id = row["object_id"]
                db.execute("UPDATE objects SET status='tracked' WHERE object_id=? AND profile_id=?", (object_id, self.profile_id))
            else:
                if object_id is None:
                    if result.action != "placed":
                        raise ValueError("An unseen pickup cannot create a resting location")
                    object_id = self._create_object(db, result.object, "tracked", candidate["observed_at"])
                if self._identity_conflict(db, candidate, object_id, result.object):
                    raise ValueError("Conflicting track, category, or simultaneously visible instance")
                self._observe(db, candidate, result, object_id, user_confirmed=True)
            if name is not None:
                if not isinstance(name, str) or not name.strip() or len(name.strip()) > 100:
                    raise ValueError("Name must contain 1-100 characters")
                db.execute("UPDATE objects SET name=? WHERE object_id=? AND profile_id=?", (name.strip(), object_id, self.profile_id))
            db.execute("UPDATE candidates SET status='verified',reason='user_resolved_identity',object_id=? WHERE event_id=? AND profile_id=?", (object_id, event_id, self.profile_id))
            self._audit(db, "resolve_identity", event_id, object_id, {"name": name})
        return self.get_object(object_id)

    def ignore(self, event_id):
        with self._connection() as db:
            row = self._candidate(db, event_id)
            if row["status"] != "review":
                raise ValueError("Only a review can be ignored")
            if row["object_id"]:
                db.execute("UPDATE objects SET status='ignored' WHERE object_id=? AND profile_id=? AND status='pending'", (row["object_id"], self.profile_id))
            db.execute("UPDATE candidates SET status='rejected',reason='user_ignored' WHERE event_id=? AND profile_id=?", (event_id, self.profile_id))
            self._audit(db, "ignore", event_id, row["object_id"])

    def fail(self, event_id, reason="verification_failed", call_id=None, sent_to_provider=True):
        with self._connection() as db:
            row = self._candidate(db, event_id)
            if call_id is not None and call_id != row["active_call_id"]:
                raise ValueError("Expired verification claim")
            if not sent_to_provider:
                db.execute("UPDATE api_calls SET cost_usd=0 WHERE call_id=? AND profile_id=?", (row["active_call_id"], self.profile_id))
            db.execute("UPDATE candidates SET status='failed',reason=? WHERE event_id=? AND profile_id=? AND status='processing'", (reason[:120], event_id, self.profile_id))
            self._audit(db, "verification_failed", event_id, detail={"reason": reason[:120]})

    def reviews(self, limit=50):
        with self._connection() as db:
            rows = db.execute("SELECT event_id FROM candidates WHERE profile_id=? AND status IN ('review','failed','budget_wait','processing') ORDER BY observed_at DESC LIMIT ?", (self.profile_id, limit)).fetchall()
            return [self.event(r[0], db=db) for r in rows]

    def get_object(self, object_id, db=None):
        if db is None:
            with self._connection() as connection:
                return self.get_object(object_id, db=connection)
        obj = db.execute("SELECT * FROM objects WHERE object_id=? AND profile_id=?", (object_id, self.profile_id)).fetchone()
        if obj is None:
            raise KeyError("Unknown object")
        row = db.execute("SELECT record FROM observations WHERE profile_id=? AND object_id=? ORDER BY observed_at DESC,sequence DESC,event_id DESC LIMIT 1", (self.profile_id, object_id)).fetchone()
        latest = json.loads(row[0]) if row else None
        state = {"placed": "placed", "picked_up": "carried", "unknown": "unknown"}.get(latest["action"], "unknown") if latest else "unknown"
        unresolved = db.execute("SELECT 1 FROM candidates c JOIN bindings b ON c.profile_id=b.profile_id AND c.session_id=b.session_id AND c.track_id=b.track_id WHERE c.profile_id=? AND b.object_id=? AND (c.observed_at>? OR (c.observed_at=? AND c.sequence>?)) AND c.status IN ('queued','processing','budget_wait','capacity_wait','review','failed','dropped') LIMIT 1",
                                (self.profile_id, object_id, latest["observed_at"] if latest else 0,
                                 latest["observed_at"] if latest else 0, latest["sequence"] if latest else -1)).fetchone()
        if unresolved:
            state = "uncertain"
        age = max(0.0, self.clock() - latest["observed_at"]) if latest else None
        return {**dict(obj), "state": state, "latest": latest,
                "location": latest["location"] if latest and state == "placed" else None,
                "needs_review": bool(unresolved), "observed_age_s": age,
                "stale": age is None or age > 3600, "answer_kind": "last_verified" if state == "placed" else state}

    def list_objects(self, include_pending=False):
        with self._connection() as db:
            rows = db.execute("SELECT object_id FROM objects WHERE profile_id=? AND status IN ('tracked',?) ORDER BY created_at", (self.profile_id, "pending" if include_pending else "tracked")).fetchall()
            return [self.get_object(row[0], db=db) for row in rows]

    def search(self, query):
        words = _words(query)
        words.update(_words(self._category(query)))
        return [obj for obj in self.list_objects() if words & _words(obj["name"] + " " + obj["category"])] if words else []

    def history(self, object_id, limit=100):
        with self._connection() as db:
            self.get_object(object_id, db=db)
            rows = db.execute("SELECT record FROM observations WHERE profile_id=? AND object_id=? ORDER BY observed_at DESC,sequence DESC,event_id DESC LIMIT ?", (self.profile_id, object_id, limit)).fetchall()
            return [json.loads(row[0]) for row in rows]

    def usage(self):
        with self._connection() as db:
            rows = db.execute("SELECT status,COUNT(*) AS n FROM candidates WHERE profile_id=? GROUP BY status", (self.profile_id,)).fetchall()
            calls = db.execute("SELECT COUNT(*),COALESCE(SUM(COALESCE(cost_usd,reserved_usd)),0) FROM api_calls WHERE profile_id=?", (self.profile_id,)).fetchone()
            return {"events": {r["status"]: r["n"] for r in rows}, "api_calls": calls[0], "cost_or_reserved_usd": round(calls[1], 6)}
