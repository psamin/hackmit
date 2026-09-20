"""Elasticsearch layer for memories. Optional: if ELASTICSEARCH_URL isn't set or
ES is down, find_object falls back to scanning memory.jsonl — a demo that keeps
working is worth more than a pure architecture.

memory.jsonl stays the source of truth; ES is the query index. vlm.py dual-writes,
or POST /api/es/sync to backfill an existing log.
"""
import json, os
from pathlib import Path

INDEX = "memories"
_es = None


def client():
    global _es
    url = os.environ.get("ELASTICSEARCH_URL")
    if not url:
        return None
    if _es is None:
        from elasticsearch import Elasticsearch
        _es = Elasticsearch(url, request_timeout=5)
        if not _es.indices.exists(index=INDEX):
            _es.indices.create(index=INDEX, mappings={"properties": {
                "object": {"type": "text", "fields": {"kw": {"type": "keyword"}}},
                "event": {"type": "keyword"}, "surface": {"type": "text"},
                "landmarks": {"type": "text"}, "location_description": {"type": "text"},
                "confidence": {"type": "float"}, "logged_at": {"type": "date"},
                "video_t": {"type": "float"}, "event_id": {"type": "integer"},
                "after_frame": {"type": "keyword", "index": False}}})
    return _es


def index_memory(mem):
    """One memory.jsonl line -> ES. Called by vlm.py after each write, or by the
    /api/es/sync backfill. Failures are swallowed: ES is a cache, not the store."""
    es = client()
    if not es:
        return False
    try:
        doc = {k: mem.get(k) for k in ("object", "event", "surface", "landmarks",
               "location_description", "confidence", "logged_at", "video_t", "event_id")}
        doc["after_frame"] = (mem.get("frames") or [None])[-1]
        es.index(index=INDEX, document=doc)
        return True
    except Exception:
        return False


def search(item, memory_jsonl: Path):
    """Newest 'placed' memory matching `item`. Returns (memory_dict, source)."""
    es = client()
    if es:
        try:
            r = es.search(index=INDEX, size=3, sort=[{"logged_at": "desc"}], query={
                "bool": {"must": {"multi_match": {"query": " ".join(expand(item)) or item, "fields": [
                    "object^3", "location_description", "landmarks", "surface"], "fuzziness": "AUTO"}},
                    "filter": {"term": {"event": "placed"}}}})
            hits = r["hits"]["hits"]
            if hits:
                return hits[0]["_source"], "elasticsearch"
        except Exception:
            pass  # fall through to the file
    return search_jsonl(item, memory_jsonl), "jsonl"


# How users actually ask vs. how detectors label things. Both sides expand the
# query, so "medicine" still finds a memory logged under "pill bottle".
SYNONYMS = {
    "medicine": "pill bottle", "meds": "pill bottle", "medication": "pill bottle",
    "pills": "pill bottle", "medications": "pill bottle", "prescription": "pill bottle",
    "spectacles": "glasses", "eyeglasses": "glasses",
    "phone": "phone", "cell": "phone", "cellphone": "phone", "mobile": "phone",
    "wallet": "wallet", "purse": "purse", "bag": "purse",
}


def expand(item):
    terms = set()
    for t in item.lower().split():
        terms.add(t)
        terms.update(SYNONYMS.get(t, "").split())
    terms -= {"my", "the", "a", "is", "are", "did", "i", "leave", "put"}
    return terms


def search_jsonl(item, memory_jsonl: Path):
    """Substring + token-overlap match over memory.jsonl, newest placed first."""
    if not memory_jsonl.exists():
        return None
    terms = expand(item)
    best = None
    for line in memory_jsonl.read_text().splitlines():
        try:
            m = json.loads(line)
        except json.JSONDecodeError:
            continue
        if m.get("event") != "placed":
            continue
        hay = " ".join(str(m.get(k, "")) for k in ("object", "location_description", "surface")
                       ).lower() + " " + " ".join(m.get("landmarks") or []).lower()
        score = sum(t in hay for t in terms) or (1 if terms & set(hay.split()) else 0)
        if score and (best is None or m.get("logged_at", "") >= best.get("logged_at", "")):
            best = m
    return best
