"""Face embeddings in Elasticsearch: durable, shareable, and still able to say no.

The local gallery (perception/faces.json) stays the source of truth -- it is fast, it
works with no network, and recognition must not stop when the cluster does. This mirrors
it, exactly as vlm.py mirrors memory.jsonl, so losing the laptop no longer loses everyone
who was ever introduced.

--------------------------------------------------------------------------------
THE SCORE CONVERSION, WHICH IS EASY TO GET SILENTLY WRONG
--------------------------------------------------------------------------------
Elasticsearch does NOT return cosine similarity from a kNN query. For a dense_vector
with similarity "cosine" it returns a rescaled score:

    _score = (1 + cosine) / 2          so cosine = 2 * _score - 1

Applying MATCH_THR to the raw _score would move the threshold from 0.38 to an effective
0.69 without anyone noticing -- every real match would start being rejected. So scores
are converted back to true cosine here, and the SAME faces.identify() logic decides,
against the same numbers it was calibrated on:

    same person        0.986
    worst impostor     0.180
    MATCH_THR          0.38     <- sits in the gap
    MARGIN             0.06     <- and must beat the runner-up

Anything that clears neither is not a person we know, and the honest answer is
"I don't recognise them". Naming the nearest match to someone who cannot check is the
worst failure this system has.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "perception"))

INDEX = "faces"
DIMS = 512

_ready = False


def client():
    """Reuses the memories client, so one ELASTICSEARCH_URL configures both."""
    import es

    return es.client()


def ensure_index(c) -> bool:
    global _ready
    if _ready:
        return True
    if not c.indices.exists(index=INDEX):
        c.indices.create(index=INDEX, mappings={"properties": {
            "name": {"type": "keyword"},
            # index=True builds the kNN structure; cosine matches how the vectors are
            # already normalised, so no extra scaling is needed on write.
            "embedding": {"type": "dense_vector", "dims": DIMS,
                          "index": True, "similarity": "cosine"},
            "added_at": {"type": "date"}}})
    _ready = True
    return True


def upsert(name: str, vec, added_at: str) -> bool:
    """Mirror one embedding. Returns False when ES is not configured or is down --
    never raises, because the local gallery has already taken the write."""
    c = client()
    if not c:
        return False
    try:
        ensure_index(c)
        c.index(index=INDEX, document={"name": name, "embedding": [float(x) for x in vec],
                                       "added_at": added_at}, refresh=True)
        return True
    except Exception:
        return False


def search(vec, k: int = 5):
    """[(name, cosine)] best first, or None if ES is unavailable.

    None and [] mean different things: None is "ask the local gallery instead",
    [] is "ES answered, and nobody is stored".
    """
    c = client()
    if not c:
        return None
    try:
        ensure_index(c)
        r = c.search(index=INDEX, size=k, source=["name"],
                     knn={"field": "embedding", "query_vector": [float(x) for x in vec],
                          "k": k, "num_candidates": max(50, k * 10)})
        return [(h["_source"]["name"], 2.0 * h["_score"] - 1.0)   # back to true cosine
                for h in r["hits"]["hits"]]
    except Exception:
        return None


def people():
    c = client()
    if not c:
        return None
    try:
        ensure_index(c)
        r = c.search(index=INDEX, size=0, aggs={"n": {"terms": {"field": "name", "size": 500}}})
        return {b["key"]: b["doc_count"] for b in r["aggregations"]["n"]["buckets"]}
    except Exception:
        return None


def forget(name: str) -> int:
    c = client()
    if not c:
        return 0
    try:
        ensure_index(c)
        return c.delete_by_query(index=INDEX, query={"term": {"name": name}},
                                 refresh=True).get("deleted", 0)
    except Exception:
        return 0


def sync_from_local() -> dict:
    """Push the whole local gallery up. For first run, and for recovery."""
    import faces

    c = client()
    if not c:
        return {"synced": 0, "error": "ELASTICSEARCH_URL not set"}
    try:
        ensure_index(c)
        c.delete_by_query(index=INDEX, query={"match_all": {}}, refresh=True)
    except Exception as exc:
        return {"synced": 0, "error": str(exc)}
    n = 0
    for name, vecs in faces.load().items():
        for v in vecs:
            n += upsert(name, v, "1970-01-01T00:00:00")
    return {"synced": n, "people": len(faces.load())}


def restore_to_local() -> dict:
    """Rebuild perception/faces.json from ES. For a fresh machine."""
    import faces

    c = client()
    if not c:
        return {"restored": 0, "error": "ELASTICSEARCH_URL not set"}
    try:
        ensure_index(c)
        r = c.search(index=INDEX, size=1000, source=["name", "embedding"],
                     query={"match_all": {}})
    except Exception as exc:
        return {"restored": 0, "error": str(exc)}
    gallery = {}
    for h in r["hits"]["hits"]:
        gallery.setdefault(h["_source"]["name"], []).append(
            np.asarray(h["_source"]["embedding"], dtype=np.float32))
    if gallery:
        faces.save(gallery)
    return {"restored": sum(len(v) for v in gallery.values()), "people": list(gallery)}
