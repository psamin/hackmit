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
        api_key = os.environ.get("ELASTICSEARCH_API_KEY")
        _es = Elasticsearch(url, api_key=api_key, request_timeout=5) if api_key \
              else Elasticsearch(url, request_timeout=5)
        if not _es.indices.exists(index=INDEX):
            _es.indices.create(index=INDEX, mappings={"properties": {
                "object": {"type": "text", "fields": {"kw": {"type": "keyword"}}},
                "event": {"type": "keyword"}, "surface": {"type": "text"},
                "landmarks": {"type": "text"}, "location_description": {"type": "text"},
                "confidence": {"type": "float"}, "logged_at": {"type": "date"},
                "video_t": {"type": "float"}, "event_id": {"type": "integer"},
                "place": {"type": "keyword"}, "place_source": {"type": "keyword"},
                "lat": {"type": "float"}, "lon": {"type": "float"},
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
               "location_description", "confidence", "logged_at", "video_t", "event_id",
               "place", "place_source", "lat", "lon")}
        doc["after_frame"] = (mem.get("frames") or [None])[-1]
        es.index(index=INDEX, document=doc)
        return True
    except Exception:
        return False


def search(item, memory_jsonl: Path):
    """Newest 'placed' memory matching `item`. Returns (memory_dict, source)."""
    hits, source = search_all(item, memory_jsonl, limit=1)
    return (hits[0] if hits else None), source


# Words that carry no information about WHICH place is meant.
_STOP = {"the", "a", "an", "on", "in", "of", "to", "at", "next", "near", "and", "with",
         "is", "it", "its", "left", "right", "front", "behind", "side", "roughly",
         "middle", "centre", "center", "there", "by", "beside", "under", "over"}


def _place_words(m):
    text = f"{m.get('location_description') or ''} {m.get('surface') or ''}".lower()
    return {w.strip(".,;:\"'") for w in text.split()} - _STOP - {""}


def _same_place(a, b, thr=0.45):
    """Jaccard over content words. The descriptions are free text from a VLM looking at
    different frames of the same scene, so they never repeat verbatim -- "a gray backpack
    ... recycling bins" and "a backpack on the ground ... recycling boxes" are one chair.
    Exact-string dedupe let those through as two places, which is worse than useless: it
    tells someone their medication is in two rooms when it is in one.
    """
    wa, wb = _place_words(a), _place_words(b)
    if not wa or not wb:
        return False
    return len(wa & wb) / len(wa | wb) >= thr


def _distinct(mems, limit):
    """Newest memory per distinct place. Collapses repeated sightings of a thing that
    has not moved -- including the same spot described in different words - and keeps
    genuinely different locations.

    Note what this cannot do: nothing here knows whether two locations mean two pill
    bottles or one that was moved. There is no instance identity - track IDs do not
    survive a run, and the VLM only ever says "pill bottle". So the caller must present
    these as places the item has been seen, with times, and let the person judge. Do not
    phrase them as separate objects.
    """
    out = []
    for m in sorted(mems, key=lambda m: m.get("logged_at") or "", reverse=True):
        if not _place_words(m):
            continue
        if any(_same_place(m, kept) for kept in out):
            continue          # same spot, older wording: the newest one already stands
        out.append(m)
        if len(out) >= limit:
            break
    return out


def search_all(item, memory_jsonl: Path, limit=3):
    """Every distinct place `item` has been seen, newest first. Returns (list, source)."""
    es = client()
    if es:
        try:
            # Over-fetch so there is something left to dedupe: one object sitting in one
            # place for an hour produces many near-identical memories.
            # "pills" expands to include the bare word "bottle", and sorting purely by recency then answers
            # "where are my pills" with whichever bottle was last put down - a water bottle. When the question
            # names something we have a canonical label for, ask for that label first and only widen if the
            # index genuinely holds none, so a near-miss can never outrank the thing that was asked for.
            hits = []
            for phrase in canonical(item):
                strict = es.search(index=INDEX, size=25, sort=[{"logged_at": "desc"}], query={
                    "bool": {"must": {"match_phrase": {"object": phrase}},
                             "filter": {"term": {"event": "placed"}}}})
                hits += [h["_source"] for h in strict["hits"]["hits"]]
            if not hits:
                r = es.search(index=INDEX, size=25, sort=[{"logged_at": "desc"}], query={
                    "bool": {"must": {"multi_match": {"query": " ".join(expand(item)) or item, "fields": [
                        "object^3", "location_description", "landmarks", "surface"], "fuzziness": "AUTO"}},
                        "filter": {"term": {"event": "placed"}}}})
                hits = [h["_source"] for h in r["hits"]["hits"]]
            if hits:
                return _distinct(hits, limit), "elasticsearch"
        except Exception:
            pass  # fall through to the file
    return _distinct(search_jsonl_all(item, memory_jsonl), limit), "jsonl"


# How users actually ask vs. how detectors label things. Both sides expand the
# query, so "medicine" still finds a memory logged under "pill bottle".
SYNONYMS = {
    "medicine": "pill bottle", "meds": "pill bottle", "medication": "pill bottle",
    "pills": "pill bottle", "medications": "pill bottle", "prescription": "pill bottle",
    "spectacles": "glasses", "eyeglasses": "glasses",
    "phone": "phone", "cell": "phone", "cellphone": "phone", "mobile": "phone",
    "wallet": "wallet", "purse": "purse", "bag": "purse",
}


def canonical(item):
    """The multi-word names `item` maps to, e.g. "pills" -> ["pill bottle"]. Used to rank the phrase above
    its individual words, so "bottle" alone cannot pull a water bottle to the top."""
    out = []
    for t in item.lower().split():
        mapped = SYNONYMS.get(t, "")
        if " " in mapped:
            out.append(mapped)
    return out or ([item.strip()] if " " in item.strip() else [])


def expand(item):
    terms = set()
    for t in item.lower().split():
        terms.add(t)
        terms.update(SYNONYMS.get(t, "").split())
    terms -= {"my", "the", "a", "is", "are", "did", "i", "leave", "put"}
    return terms


def search_jsonl(item, memory_jsonl: Path):
    """Newest matching 'placed' memory in memory.jsonl, or None."""
    hits = search_jsonl_all(item, memory_jsonl)
    return max(hits, key=lambda m: m.get("logged_at", "")) if hits else None


def search_jsonl_all(item, memory_jsonl: Path):
    """Every matching 'placed' memory. Substring + token overlap; the file is small."""
    if not memory_jsonl.exists():
        return []
    terms = expand(item)
    out = []
    for line in memory_jsonl.read_text(encoding="utf-8").splitlines():
        try:
            m = json.loads(line)
        except json.JSONDecodeError:
            continue
        if m.get("event") != "placed":
            continue
        hay = " ".join(str(m.get(k, "")) for k in ("object", "location_description", "surface")
                       ).lower() + " " + " ".join(m.get("landmarks") or []).lower()
        if sum(t in hay for t in terms) or (terms & set(hay.split())):
            out.append(m)
    return out
