"""Coordinates -> a place a person would actually say: "at home", "at the pharmacy".

    from places import resolve
    resolve(42.3601, -71.0942)   -> {"place": "home", "source": "known", ...}

--------------------------------------------------------------------------------
WHAT THIS CAN AND CANNOT TELL YOU
--------------------------------------------------------------------------------
A phone indoors is accurate to roughly 10-50 m. That is BUILDING granularity, not
room granularity. "at the library" is answerable; "in the kitchen, not the bedroom"
is not, and no amount of API spend changes that -- the error bars are larger than a
house.

So this is deliberately not the answer to "where is my medication". The VLM already
answers that far better, from pixels: "on the kitchen counter, left of the kettle".
This adds the other half - WHICH BUILDING that counter was in - so a memory from the
doctor's office is not confused with one from home. Visual description for the room,
coordinates for the place. Do not let them compete.

--------------------------------------------------------------------------------
KNOWN PLACES BEAT THE PLACES API
--------------------------------------------------------------------------------
contacts.json already carries a curated table: home, doctor, pharmacy, airport. Those
are checked first, and they win on every axis that matters - free, instant, offline,
and named the way the user speaks ("the doctor's", not "Massachusetts General Hospital
Building 3"). Google is the fallback for coordinates that match nothing known.

Google also has the failure mode we keep meeting: a nearby-search always returns the
nearest something. At 30 m of error in a dense area that can confidently name the cafe
next door. So its answers are marked `source: "google"`, kept out of the confident
phrasing, and only used when nothing known matches.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONTACTS = HERE / "contacts.json"

# A known place claims any fix within this many metres. Generous on purpose: the fix
# itself is worth +/-50 m, so a tight radius would miss the building you are standing in.
KNOWN_RADIUS_M = 120
# Fixes worse than this are not worth resolving at all - they would name a neighbourhood.
MAX_ACCURACY_M = 200
# Cache key precision. 4 decimal places is ~11 m, finer than the fix, so this collapses
# a stationary phone's stream of fixes onto one lookup instead of one per memory.
CACHE_PRECISION = 4

_cache: dict = {}


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def known_places() -> dict:
    try:
        return json.loads(CONTACTS.read_text(encoding="utf-8")).get("places", {}) or {}
    except (OSError, json.JSONDecodeError):
        return {}


def nearest_known(lat, lon):
    """(name, metres) of the closest curated place, or (None, None)."""
    best, best_d = None, None
    for name, p in known_places().items():
        if p.get("lat") is None or p.get("lon") is None:
            continue
        d = haversine_m(lat, lon, float(p["lat"]), float(p["lon"]))
        if best_d is None or d < best_d:
            best, best_d = name, d
    return best, best_d


def _short_address(formatted: str) -> str:
    """"77 Massachusetts Ave, Cambridge, MA 02139, USA" -> "77 Massachusetts Ave,
    Cambridge". The full string is for a map label, not for something read aloud to
    someone who only wants to know which building they were in."""
    parts = [p.strip() for p in formatted.split(",")]
    return ", ".join(parts[:2]) if len(parts) > 2 else formatted


def _google(lat, lon):
    """Nearest named place, else a street address. None without a key or without the
    APIs enabled.

    Places API (New) is tried first and gives what we actually want - a building name
    like "Hayden Library". Geocoding is the fallback and only ever gives an address,
    which is still worth having: "near 77 Massachusetts Ave" beats saying nothing.
    The legacy Places endpoint is not attempted at all; Google no longer lets new
    projects enable it, so it is a guaranteed round trip to a 403.
    """
    key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not key:
        return None
    import httpx

    with httpx.Client(timeout=4.0) as c:
        try:
            r = c.post("https://places.googleapis.com/v1/places:searchNearby",
                       headers={"Content-Type": "application/json", "X-Goog-Api-Key": key,
                                "X-Goog-FieldMask": "places.displayName,places.primaryType"},
                       json={"locationRestriction": {"circle": {
                                 "center": {"latitude": lat, "longitude": lon}, "radius": 100.0}},
                             "maxResultCount": 5})
            for hit in (r.json().get("places") or []):
                name = (hit.get("displayName") or {}).get("text")
                if name:
                    return {"place": name, "source": "google"}
        except Exception:
            pass          # not enabled, or offline: the address below is still useful

        try:
            r = c.get("https://maps.googleapis.com/maps/api/geocode/json",
                      params={"latlng": f"{lat},{lon}", "key": key})
            res = r.json().get("results") or []
            if res:
                return {"place": _short_address(res[0]["formatted_address"]),
                        "source": "google_address"}
        except Exception:
            pass
    return None


def resolve(lat, lon, accuracy_m=None) -> dict:
    """Coordinates -> {"place", "source", "lat", "lon", "accuracy_m", "distance_m"}.

    `source` is "known" (curated, trustworthy), "google"/"google_address" (nearest
    match, treat as a hint) or "unknown". Callers should phrase the last two loosely.
    """
    out = {"lat": round(float(lat), 6), "lon": round(float(lon), 6),
           "accuracy_m": accuracy_m, "place": None, "source": "unknown"}
    if accuracy_m is not None and accuracy_m > MAX_ACCURACY_M:
        out["source"] = "too_inaccurate"
        return out

    name, dist = nearest_known(lat, lon)
    if name and dist is not None and dist <= KNOWN_RADIUS_M:
        out.update(place=name, source="known", distance_m=round(dist))
        return out

    key = (round(lat, CACHE_PRECISION), round(lon, CACHE_PRECISION))
    if key not in _cache:
        _cache[key] = _google(lat, lon)
    hit = _cache[key]
    if hit:
        out.update(hit)
    return out


# How a person says each place. A key in contacts.json places may override this with
# its own "say" field; these are just defaults that read naturally out loud.
SPOKEN = {"home": "at home", "doctor": "at the doctor's", "work": "at work"}


def phrase(loc: dict) -> str:
    """How to say it mid-sentence, honest about confidence. '' when there is nothing
    worth saying -- an empty string disappears cleanly into an f-string."""
    if not loc or not loc.get("place"):
        return ""
    name = loc["place"]
    if loc["source"] != "known":
        return f"near {name}"          # google: a hint, not a claim
    override = (known_places().get(name) or {}).get("say")
    return override or SPOKEN.get(name) or f"at the {name}"
