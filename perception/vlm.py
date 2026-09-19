"""Event frames -> Claude -> one structured memory appended to memory.jsonl, plus `ask()` to query memories.

    python vlm.py runs/p02/events/003_placed_0120.3      # describe one saved event
    python vlm.py --ask "where is the sauce bottle?" runs/p02/memory.jsonl
"""
import base64, json, os, sys, threading, time
from datetime import datetime
from pathlib import Path
from typing import Literal

import anthropic
from pydantic import BaseModel

_env = Path(__file__).with_name(".env")  # ANTHROPIC_API_KEY=... (gitignored)
if _env.exists():
    for line in _env.read_text().splitlines():
        k, _, v = line.partition("=")
        if k.strip() and not k.startswith("#"):
            os.environ.setdefault(k.strip(), v.strip())

MODEL = "claude-opus-5"
FALLBACK = {"extra_headers": {"anthropic-beta": "server-side-fallback-2026-07-01"}, "extra_body": {"fallbacks": "default"}}
_write_lock = threading.Lock()
client = anthropic.Anthropic()

SYSTEM = """You are the memory module of an assistive device for someone with memory and mobility limitations.
You get three frames from a head-worn camera: BEFORE, DURING and AFTER a possible event involving a tracked object.
In AFTER, a yellow box marks where the tracker last saw the object.
Say what happened to that object and where it ended up, in words that would help the person find it later:
the surface it is on and the nearby landmarks (\"on the counter, left of the sink, next to the kettle\").
If the frames do not show the object being set down, say so in `event` rather than guessing."""


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
    for label, path in zip(("BEFORE", "DURING", "AFTER"), ev["frames"]):
        content += [{"type": "text", "text": label}, _image(path)]
    content.append({"type": "text", "text": f"Tracked object class: {ev['object']}. Trigger: {ev['type']}."})
    t0 = time.perf_counter()
    resp = client.messages.parse(model=MODEL, max_tokens=16000, system=SYSTEM,
                                 messages=[{"role": "user", "content": content}], output_format=Memory, **FALLBACK)
    latency = time.perf_counter() - t0
    if resp.stop_reason == "refusal":
        print(f"event {ev['id']}: refused", flush=True)
        return None
    mem = {"logged_at": datetime.now().isoformat(timespec="seconds"), "video_t": ev["t"], "event_id": ev["id"],
           "trigger": ev["type"], **resp.parsed_output.model_dump(), "frames": ev["frames"],
           "vlm_latency_s": round(latency, 2), "input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
    with _write_lock, open(memory_path, "a") as f:
        f.write(json.dumps(mem) + "\n")
    print(f"event {ev['id']}: {mem['event']} {mem['object']} -> {mem['location_description']} ({latency:.1f}s)", flush=True)
    return mem


def ask(question, memory_path):
    """Few memories at demo scale, so all of them go in the prompt: no vector DB needed yet."""
    memories = [json.loads(l) for l in open(memory_path)]
    lines = [f"- {m['logged_at']} (video {m['video_t']}s): {m['event']} {m['object']}: {m['location_description']} (confidence {m['confidence']})"
             for m in memories]
    t0 = time.perf_counter()
    resp = client.messages.create(
        model=MODEL, max_tokens=16000, output_config={"effort": "low"},
        system="You answer questions about where the user left things, from the memory log below. Answer in one or two "
               "short spoken sentences. Use the most recent 'placed' memory for the object. If the log doesn't say, say so.\n\n"
               "Memory log (oldest first):\n" + "\n".join(lines),
        messages=[{"role": "user", "content": question}], **FALLBACK)
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
