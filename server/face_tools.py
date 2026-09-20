"""The two face tools Pam can call: remember this person, and who is this.

    save_face("Jacob")  -> embeds the largest face in the current frame, stores it
    who_is_this()       -> embeds it again and matches against everyone enrolled

Frames come from whichever camera is live: the phone streams through the server's
/api/camera relay, so those are tapped in flight; a `--source 0` laptop run never
touches the server, so the pipeline's once-a-second snapshot is the fallback.

The recognition itself lives in perception/faces.py -- quality gates, the gallery, and
the threshold that lets it answer "I don't know". Read the docstring there before
changing any number in it.

Two decisions worth stating, because neither is obvious:

YOLOE is NOT used to find the face first. It is already loaded and already looking at
these frames, so gating on it sounds free -- but InsightFace does detection, five-point
landmarking and the embedding in a single pass, and the landmarks are not optional:
without the affine warp to a canonical crop, embeddings of the same person at different
angles drift apart. A YOLOE box would have to be re-detected by InsightFace anyway.

Nothing is stored until a human says a name. A face that walks through frame is
embedded in memory, compared, and dropped. That is the privacy position, and it is a
property of this code rather than a promise in a document: `save_face` is the only
function here that writes.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "perception"))

# A relay frame older than this is not "what the camera sees now" -- fall back to disk.
FRESH_S = 5.0


def _decode(jpeg: bytes):
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    return img if img is not None and img.size else None


def current_frame(relay_frame, relay_ts, snapshot_paths):
    """The newest view of the camera: live relay bytes if fresh, else the pipeline's
    on-disk snapshot. Returns (image, source) or (None, why-not)."""
    if relay_frame is not None and time.time() - relay_ts <= FRESH_S:
        img = _decode(relay_frame)
        if img is not None:
            return img, "phone"
    newest, newest_age = None, None
    for p in snapshot_paths:
        if p.exists():
            age = time.time() - p.stat().st_mtime
            if newest_age is None or age < newest_age:
                newest, newest_age = p, age
    if newest is not None and newest_age is not None and newest_age <= 30:
        img = cv2.imread(str(newest))
        if img is not None:
            return img, f"laptop snapshot ({newest_age:.0f}s old)"
    return None, "no camera frame available - is the pipeline running and the camera on?"


def save_face(name: str, frame):
    """Enrol the largest face. Returns a dict the agent can speak."""
    import faces

    if not name or not name.strip():
        return {"ok": False, "say": "I need a name to save a face."}
    name = name.strip()
    gallery = faces.load()
    ok, msg = faces.enroll(name, frame, gallery)
    if not ok:
        # The quality bar is deliberately higher for enrolment than recognition: one bad
        # embedding quietly degrades every future match against this person.
        return {"ok": False, "say": f"I couldn't get a clear enough look - {msg}. "
                                    f"Can they face the camera a little closer?"}
    n = len(gallery[name])
    # Local file first, then mirror: the same order vlm.py uses for memories, for the
    # same reason. A cluster that is down must not be able to lose a person.
    import es_faces
    from datetime import datetime
    mirrored = es_faces.upsert(name, gallery[name][-1], datetime.now().isoformat(timespec="seconds"))
    return {"ok": True, "name": name, "embeddings": n, "mirrored": mirrored,
            "say": f"Got it, I'll remember {name}." if n == 1
                   else f"Thanks, that's another look at {name} - I'll recognise them better now."}


def _match(vec, gallery, es_faces):
    """(name|None, best cosine, runner-up cosine).

    Elasticsearch first, local gallery when it is unreachable. Both paths end in the
    SAME threshold test, on true cosine, so the answer cannot depend on which store
    replied -- and either can answer "nobody", which is a real answer, not a failure.
    """
    import faces

    hits = es_faces.search(vec, k=5)
    if hits is None:                       # ES down or unconfigured
        return faces.identify(vec, gallery)
    if not hits:
        return None, 0.0, 0.0
    # Several embeddings per person: take each person's best, exactly as the local
    # path does with max() rather than a mean.
    per_person = {}
    for name, cos in hits:
        per_person[name] = max(per_person.get(name, -1.0), cos)
    ranked = sorted(((c, n) for n, c in per_person.items()), reverse=True)
    best, name = ranked[0]
    second = ranked[1][0] if len(ranked) > 1 else 0.0
    if best < faces.MATCH_THR or (best - second) < faces.MARGIN:
        return None, best, second
    return name, best, second


def who_is_this(frame):
    """Identify every usable face in view."""
    import faces

    import es_faces

    gallery = faces.load()
    es_people = es_faces.people()
    if not gallery and not es_people:
        return {"ok": False, "say": "I don't know anyone yet. Tell me a name and I'll remember them."}
    found = faces.faces_in(frame)
    if not found:
        return {"ok": False, "say": "I can't see anyone's face at the moment."}

    people, unknown, skipped, source = [], 0, 0, "local"
    for f in found:
        if f["reject"]:
            skipped += 1
            continue
        who, best, second = _match(f["vec"], gallery, es_faces)
        source = "elasticsearch" if es_people is not None else "local"
        if who:
            people.append({"name": who, "similarity": round(best, 3)})
        else:
            unknown += 1
    if not people and not unknown:
        return {"ok": False, "say": "I can see someone, but not clearly enough to tell who."}
    if not people:
        # Saying "I don't know" is the correct answer, not a failure. The alternative --
        # naming the nearest match -- tells someone who cannot check that a stranger is
        # their son.
        return {"ok": True, "people": [], "unknown": unknown,
                "say": "I can see someone, but I don't recognise them."}
    names = [p["name"] for p in people]
    said = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    extra = f" There's also someone I don't recognise." if unknown else ""
    return {"ok": True, "people": people, "unknown": unknown, "source": source,
            "say": f"That's {said}.{extra}"}


def known_people():
    import es_faces
    import faces

    local = {name: len(v) for name, v in faces.load().items()}
    return {"local": local, "elasticsearch": es_faces.people()}
