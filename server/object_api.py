from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import sys
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

import caregiver

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from perception.object_memory import ObjectStore


def database_path(memory_jsonl):
    return Path(os.environ.get("PAM_OBJECT_DB") or Path(memory_jsonl).parent / "objects.sqlite3").resolve()


def active(memory_jsonl):
    return bool(os.environ.get("PAM_OBJECT_DB")) or database_path(memory_jsonl).is_file()


def store_for(memory_jsonl):
    try:
        return ObjectStore(database_path(memory_jsonl), profile_id="local", create=False)
    except FileNotFoundError as exc:
        raise HTTPException(503, "The personal-memory pipeline has not initialized its database.") from exc


def observation_dto(record):
    return {key: record.get(key) for key in ("event_id", "object_id", "observed_at", "action", "actor",
                                            "scene", "location", "identity_basis")}


def object_dto(obj):
    latest = obj.get("latest")
    return {"object_id": obj["object_id"], "name": obj["name"], "category": obj["category"],
            "status": obj["status"], "state": obj["state"], "answer_kind": obj["answer_kind"],
            "needs_review": obj["needs_review"], "stale": obj["stale"], "observed_age_s": obj["observed_age_s"],
            "observed_at": latest["observed_at"] if latest else None,
            "scene": latest["scene"] if latest and obj["state"] == "placed" else None,
            "location": obj["location"]}


def _when(age):
    if age is None:
        return "at an unrecorded time"
    if age < 60:
        return "just now"
    if age < 3600:
        value = max(1, int(age // 60))
        return f"{value} minute{'s' if value != 1 else ''} ago"
    if age < 86400:
        value = int(age // 3600)
        return f"{value} hour{'s' if value != 1 else ''} ago"
    value = int(age // 86400)
    return f"{value} day{'s' if value != 1 else ''} ago"


def find_objects(store, query):
    objects = store.search(query)
    if not objects:
        return {"say": f"I don't have a verified location for your {query}. A new item may still need tracking confirmation.",
                "source": "object_memory", "objects": []}
    summaries = []
    for obj in objects[:3]:
        when = _when(obj["observed_age_s"])
        if obj["state"] == "placed":
            description = obj["latest"]["scene"].get("description") or obj["latest"]["scene"].get("surface") or "at a place I could not describe"
            description = description.rstrip(". ")
            if description and description[0].isupper() and len(description) > 1 and description[1].islower():
                description = description[0].lower() + description[1:]
            summaries.append(f"I last saw your {obj['name']} {description}, {when}.")
        elif obj["state"] == "carried":
            summaries.append(f"I saw your {obj['name']} being picked up {when}. I haven't verified a new resting place.")
        elif obj["state"] == "uncertain":
            summaries.append(f"There is a newer unverified interaction involving your {obj['name']}, so I can't confirm its resting place.")
        else:
            summaries.append(f"Your {obj['name']} may have been moved by someone else. I can't confirm its resting place.")
    if any(obj["stale"] and obj["state"] == "placed" for obj in objects[:3]):
        summaries.append("Those are last observed locations; the items may have moved since then.")
    prefix = f"{len(objects)} tracked items match. " if len(objects) > 1 else ""
    out = {"say": prefix + " ".join(summaries), "source": "object_memory",
           "objects": [object_dto(obj) for obj in objects],
           "card": {"title": objects[0]["name"] if len(objects) == 1 else "Matching tracked items",
                    "body": "\n".join(summaries)}}
    if len(objects) == 1 and objects[0]["state"] == "placed":
        latest = objects[0]["latest"]
        if latest.get("frames"):
            out["card"]["image"] = f"/api/objects/events/{latest['event_id']}/image?index={len(latest['frames']) - 1}"
        location = objects[0]["location"] or {}
        if location.get("status") == "available":
            out["card"]["action"] = {"label": f"Approximate map location (+/-{location['accuracy_m']:.0f} m)",
                                      "href": location["maps_url"]}
    return out


def notices(store):
    result = []
    for obj in store.list_objects():
        latest = obj.get("latest")
        if latest:
            result.append({"id": latest["event_id"] + ":" + obj["state"], "object": obj["name"],
                "object_id": obj["object_id"], "state": obj["state"],
                "logged_at": datetime.fromtimestamp(latest["observed_at"], timezone.utc).isoformat(),
                "title": "Location observed" if obj["state"] == "placed" else "Object state updated",
                "detail": latest["scene"].get("description", "") if obj["state"] == "placed" else "A resting location is not currently verified.",
                "image": None})
    return sorted(result, key=lambda row: row["logged_at"])[-20:]


class ReviewDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    action: Literal["track", "link", "ignore"]
    object_id: str | None = Field(default=None, max_length=80)
    name: str | None = Field(default=None, min_length=1, max_length=100)


def install(app, memory_path):
    router = APIRouter(prefix="/api/objects", route_class=caregiver.NoStoreRoute,
                       dependencies=[Depends(caregiver.require_caregiver)])

    @router.get("")
    async def objects():
        store = store_for(memory_path())
        return {"objects": [object_dto(obj) for obj in store.list_objects()], "usage": store.usage()}

    @router.get("/reviews")
    async def reviews():
        store = store_for(memory_path())
        result = []
        for event in store.reviews():
            result.append({"event_id": event["event_id"], "object": event.get("object"),
                "object_id": event["object_id"], "status": event["status"], "reason": event["reason"],
                "observed_at": event["observed_at"], "verification": event["verification"],
                "possible_object_ids": event.get("gallery_ids", []),
                "images": [f"/api/objects/events/{event['event_id']}/image?index={i}" for i in range(len(event.get("frames", [])))]})
        return {"reviews": result}

    @router.post("/reviews/{event_id}")
    async def decide(event_id: str, decision: ReviewDecision, request: Request):
        origin = request.headers.get("origin")
        if origin and origin.rstrip("/") != str(request.base_url).rstrip("/"):
            raise HTTPException(403, "Use the paired Pam page")
        store = store_for(memory_path())
        try:
            if decision.action == "ignore":
                store.ignore(event_id)
                return {"ok": True}
            if decision.action == "link" and not decision.object_id:
                raise ValueError("Choose an existing object ID for a link")
            if decision.action == "track" and decision.object_id:
                raise ValueError("Use link when selecting an existing instance")
            obj = store.resolve_review(event_id, object_id=decision.object_id, name=decision.name)
            return {"ok": True, "object": object_dto(obj)}
        except KeyError as exc:
            raise HTTPException(404, "Unknown object or event") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @router.get("/events/{event_id}/image")
    async def image(event_id: str, index: int = Query(default=0, ge=0, le=4)):
        store = store_for(memory_path())
        try:
            event = store.event(event_id)
            if event["status"] in {"rejected", "dropped"}:
                raise KeyError("Rejected media")
            path = Path(event.get("frames", [])[index]).resolve()
        except (KeyError, IndexError, TypeError) as exc:
            raise HTTPException(404, "No evidence image") from exc
        if not path.is_relative_to(store.path.parent / "object_evidence") or path.suffix.lower() not in {".jpg", ".jpeg"} or not path.is_file():
            raise HTTPException(404, "No evidence image")
        return FileResponse(path, headers={"Cache-Control": "no-store"})

    @router.get("/{object_id}/history")
    async def history(object_id: str):
        try:
            return {"observations": [observation_dto(row) for row in store_for(memory_path()).history(object_id)]}
        except KeyError as exc:
            raise HTTPException(404, "Unknown object") from exc

    app.include_router(router)
