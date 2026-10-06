from __future__ import annotations

import json
import math
import re
import struct
import time

MAGIC = b"CMP2"
MAX_PACKET_BYTES = 2 ** 20
MAX_METADATA_BYTES = 8192
FIX_MAX_AGE_S = 120.0
FIX_MAX_ACCURACY_M = 100.0


def finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Expected a finite number")
    if not math.isfinite(value):
        raise ValueError("Expected a finite number")
    return float(value)


def location_at(fix, captured_at, max_age_s=FIX_MAX_AGE_S, max_accuracy_m=FIX_MAX_ACCURACY_M):
    if not fix:
        return {"status": "missing"}
    if isinstance(fix, dict) and "lat" not in fix and fix.get("status") in {"missing", "denied", "stale", "future", "inaccurate", "invalid"}:
        return {"status": fix["status"]}
    try:
        lat, lon = finite_number(fix["lat"]), finite_number(fix["lon"])
        accuracy, observed_at = finite_number(fix["accuracy_m"]), finite_number(fix["observed_at"])
        age = finite_number(captured_at) - observed_at
        if not (-90 <= lat <= 90 and -180 <= lon <= 180 and accuracy >= 0 and observed_at > 0):
            raise ValueError("Invalid location fix")
    except (KeyError, TypeError, ValueError):
        return {"status": "invalid"}
    if age < 0:
        return {"status": "future"}
    if age > max_age_s:
        return {"status": "stale"}
    if accuracy > max_accuracy_m:
        return {"status": "inaccurate"}
    result = {"status": "available", "lat": lat, "lon": lon, "accuracy_m": accuracy,
              "observed_at": observed_at, "age_s": round(age, 3),
              "maps_url": f"https://www.google.com/maps/search/?api=1&query={lat:.6f},{lon:.6f}"}
    return result


def encode_frame(jpeg, metadata):
    header = json.dumps(metadata, allow_nan=False, separators=(",", ":")).encode("utf-8")
    if len(header) > MAX_METADATA_BYTES or 8 + len(header) + len(jpeg) > MAX_PACKET_BYTES:
        raise ValueError("Frame packet too large")
    return MAGIC + struct.pack(">I", len(header)) + header + jpeg


def decode_frame(data, received_at=None):
    now = time.time() if received_at is None else finite_number(received_at)
    if not isinstance(data, bytes) or not data or len(data) > MAX_PACKET_BYTES:
        raise ValueError("Invalid frame packet")
    if not data.startswith(MAGIC):
        return data, {"captured_at": now, "received_at": now, "time_source": "received",
                      "location": {"status": "missing"}}
    if len(data) < 8:
        raise ValueError("Incomplete frame header")
    size = struct.unpack(">I", data[4:8])[0]
    if not 0 < size <= MAX_METADATA_BYTES or len(data) <= 8 + size:
        raise ValueError("Invalid frame header length")
    try:
        meta = json.loads(data[8:8 + size])
        captured_at = finite_number(meta["captured_at"])
        session_id, frame_id = meta["session_id"], meta["frame_id"]
        if not isinstance(session_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{8,80}", session_id):
            raise ValueError("Invalid camera session")
        if type(frame_id) is not int or frame_id < 0:
            raise ValueError("Invalid frame sequence")
        if not -5 <= now - captured_at <= 30:
            raise ValueError("Camera clock or frame is stale; synchronize device clocks")
    except (KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid capture metadata") from exc
    return data[8 + size:], {"session_id": session_id, "frame_id": frame_id,
                             "captured_at": captured_at, "received_at": now,
                             "time_source": "capture", "location": location_at(meta.get("location"), captured_at)}
