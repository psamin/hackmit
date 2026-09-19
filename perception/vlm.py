"""Event frames -> Claude -> one structured memory appended to memory.jsonl, plus `ask()` to query memories.

    python vlm.py runs/p02/events/003_placed_0120.3      # describe one saved event
    python vlm.py --ask "where is the sauce bottle?" runs/p02/memory.jsonl
"""
import base64, codecs, json, os, sys, threading, time
from datetime import datetime
from pathlib import Path
from typing import Literal

import anthropic
from pydantic import BaseModel

BOMS = ((codecs.BOM_UTF16_LE, "utf-16"), (codecs.BOM_UTF16_BE, "utf-16"), (codecs.BOM_UTF8, "utf-8-sig"))


def _read_env(path):
    """Decode .env whatever shell or editor wrote it.

    Path.read_text() would use the cp1252 locale codec, and Windows shells do not write
    cp1252: PowerShell 5.1's `>` and Out-File default to UTF-16 LE, newer PowerShell to
    UTF-8 with a BOM. Either way the first key name arrives as mojibake, so
    ANTHROPIC_API_KEY is never set and the only symptom is an auth failure on the first
    event, with nothing pointing back at this file. Sniff the BOM instead of guessing.
    """
    raw = path.read_bytes()
    for bom, enc in BOMS:
        if raw.startswith(bom):
            return raw.decode(enc)
    return raw.decode("utf-8", errors="replace")


_env = Path(__file__).with_name(".env")  # ANTHROPIC_API_KEY=... (gitignored)
if _env.exists():
    for line in _read_env(_env).splitlines():
        k, _, v = line.partition("=")
        k = k.strip()
        if k and not k.startswith("#"):
            os.environ.setdefault(k, v.strip())

# Sonnet, deliberately. This is label-reading and scene description from a few small
# stills, not reasoning, and it is 2.5x cheaper per token than Opus in and out.
# Override without editing: COMPASS_VLM_MODEL=claude-opus-5 python memory_pipeline.py ...
MODEL = os.environ.get("COMPASS_VLM_MODEL", "claude-sonnet-5")

# Output ceilings. Worth being precise about what these do and do not protect against:
# nothing here is an agent loop. Each event is exactly one request and one response, no
# tools, no retries, no continuation -- so there is no runaway-loop failure mode to guard
# against, and the only way to spend more than expected is a long single answer. These
# cap that. A Memory is ~150 tokens of JSON and a spoken answer is one or two sentences;
# the remaining headroom is for adaptive thinking, which counts against max_tokens.
MAX_TOKENS_MEMORY = 2048
MAX_TOKENS_ANSWER = 1024
EFFORT = "low"  # both jobs are description, not reasoning; also bounds thinking tokens

# Server-side refusal fallbacks exist because the Opus and Fable safety classifiers
# sometimes decline benign requests. Sonnet does not carry them, and the beta is only
# documented for those models, so only send it when we are actually on one.
FALLBACK = ({"extra_headers": {"anthropic-beta": "server-side-fallback-2026-07-01"},
             "extra_body": {"fallbacks": "default"}}
            if MODEL.startswith(("claude-opus", "claude-fable")) else {})
# USD per million tokens, for the end-of-run summary only. Keep in step with MODEL.
PRICES = {"claude-sonnet-5": (2.0, 10.0), "claude-opus-5": (5.0, 25.0), "claude-haiku-4-5": (1.0, 5.0)}


def cost_usd(input_tokens, output_tokens, model=MODEL):
    per_in, per_out = PRICES.get(model, (0.0, 0.0))
    return input_tokens / 1e6 * per_in + output_tokens / 1e6 * per_out


_write_lock = threading.Lock()
client = anthropic.Anthropic()

SYSTEM = """You are the memory module of an assistive device for someone with memory and mobility limitations.
You get three frames from a head-worn camera: BEFORE, DURING and AFTER a possible event involving a tracked object.
In AFTER, a yellow box marks where the tracker last saw the object. If there is no box, the trigger was the
wearer's hand activity: find the object yourself. A single AFTER frame is a snapshot of the object at rest.
Say what happened to that object and where it ended up, in words that would help the person find it later:
the surface it is on and the nearby landmarks (\"on the counter, left of the sink, next to the kettle\").
If the frames do not show the object being set down, say so in `event` rather than guessing.

YOU decide what the object is, not the detector. The detector is open-vocabulary: it scores a fixed list of text
prompts against every box and applies the single best-scoring one. It has no way to answer "none of these", so it
always returns one of the candidates, and it confuses visually similar ones - a pill bottle against a water bottle
especially. Its label is a hint, not a fact. You can read printed labels, caps, sizes and materials that it cannot.
Choose from the candidate list you are given and put that in `object`; use "other" if it is genuinely none of them.
Do not repeat the detector's guess just because it was given to you.

This matters because the person may ask "where is my medication?" and act on the answer. Calling a water bottle a
pill bottle is worse than admitting uncertainty - lower `confidence` when the identity is not clear from the
frames."""


class Memory(BaseModel):
    event: Literal["placed", "picked_up", "still_in_hand", "no_change", "unclear"]
    object: str
    surface: str
    landmarks: list[str]
    location_description: str
    confidence: float


def _image(path):
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                        "data": base64.standard_b64encode(Path(path).read_bytes()).decode()}}


def describe_event(ev, memory_path):
    content = []
    for label, path in zip(("BEFORE", "DURING", "AFTER")[-len(ev["frames"]):], ev["frames"]):
        content += [{"type": "text", "text": label}, _image(path)]
    # The candidate list is exactly what the detector was allowed to say, so it is also the
    # set the VLM should choose from. Falls back to the detector's own label for events
    # written by an older pipeline that did not record the targets.
    candidates = ev.get("targets") or [c.strip() for c in str(ev["object"]).split(",")]
    content.append({"type": "text", "text":
                    f"Candidate objects: {', '.join(candidates)}. "
                    f"Detector's best guess, which may be wrong: {ev['object']}. "
                    f"Trigger: {ev['type']}."})
    t0 = time.perf_counter()
    resp = client.messages.parse(model=MODEL, max_tokens=MAX_TOKENS_MEMORY, system=SYSTEM,
                                 output_config={"effort": EFFORT},
                                 messages=[{"role": "user", "content": content}], output_format=Memory, **FALLBACK)
    latency = time.perf_counter() - t0
    if resp.stop_reason == "refusal":
        print(f"event {ev['id']}: refused", flush=True)
        return None
    if resp.stop_reason == "max_tokens":
        # The schema is small, so this means thinking ate the budget. Raise
        # MAX_TOKENS_MEMORY rather than lowering effort, which would cost accuracy.
        print(f"event {ev['id']}: hit max_tokens ({MAX_TOKENS_MEMORY}); no memory written", flush=True)
        return None
    mem = {"logged_at": datetime.now().isoformat(timespec="seconds"), "video_t": ev["t"], "event_id": ev["id"],
           "trigger": ev["type"], **resp.parsed_output.model_dump(),
           # Kept next to the VLM's own answer so disagreements are greppable: they are the
           # evidence for whether the prompt set discriminates on real objects.
           "detector_label": ev["object"], "frames": ev["frames"],
           "vlm_latency_s": round(latency, 2), "input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
    with _write_lock, open(memory_path, "a") as f:
        f.write(json.dumps(mem) + "\n")
    print(f"event {ev['id']}: {mem['event']} {mem['object']} -> {mem['location_description']} ({latency:.1f}s)", flush=True)
    return mem


def ask(question, memory_path):
    """Few memories at demo scale, so all of them go in the prompt: no vector DB needed yet.
    A last-seen snapshot newer than every memory is described first (one VLM call, only when asked)."""
    memory_path = Path(memory_path)
    memories = [json.loads(l) for l in open(memory_path)] if memory_path.exists() else []
    newest = max((m["video_t"] for m in memories), default=float("-inf"))
    for snap in sorted((memory_path.parent / "last_seen").glob("*.json")):
        s = json.loads(snap.read_text())
        if s["t"] > newest:
            m = describe_event({"id": "last_seen", "t": s["t"], "type": "last_seen", "object": s["object"],
                                "frames": [s["frame"]]}, memory_path)
            memories += [m] if m else []
    memories = [m for m in memories if m["event"] not in ("still_in_hand", "no_change")]
    lines = [f"- {m['logged_at']} (video {m['video_t']}s): {m['event']} {m['object']}: {m['location_description']} (confidence {m['confidence']})"
             for m in memories]
    t0 = time.perf_counter()
    resp = client.messages.create(
        model=MODEL, max_tokens=MAX_TOKENS_ANSWER, output_config={"effort": EFFORT},
        system="You answer questions about where the user left things, from the memory log below. Answer in one or two "
               "short spoken sentences. Use the most recent 'placed' memory for the object. If the log doesn't say, say so.\n\n"
               "Memory log (oldest first):\n" + "\n".join(lines),
        messages=[{"role": "user", "content": question}], **FALLBACK)
    # describe_event already guards this; ask() is the path the user actually hears, so a
    # refusal that survives the fallback chain must not come back as an empty spoken answer.
    if resp.stop_reason == "refusal":
        return "Sorry, I can't answer that one.", time.perf_counter() - t0
    text = "".join(b.text for b in resp.content if b.type == "text")
    return text, time.perf_counter() - t0


if __name__ == "__main__":
    if sys.argv[1] == "--ask":
        answer, s = ask(sys.argv[2], sys.argv[3])
        print(f"{answer}\n({s:.1f}s)")
    else:
        d = Path(sys.argv[1])
        ev = {"id": d.name, "t": 0, "type": d.name.split("_")[1], "object": "bottle",
              "frames": [str(d / f"{n}.jpg") for n in ("before", "during", "after")]}
        describe_event(ev, d.parent.parent / "memory.jsonl")
