"""The detection -> VLM contract: read the events memory_pipeline emits, without importing it.

This is the seam between the two halves of the system. The detection side decides
*that* something happened and to which object; the VLM side decides *what* happened
and where it ended up. They meet here and nowhere else.

The pipeline appends one JSON object per line to <out>/events.jsonl and flushes
immediately, so a separate process can follow the file live:

    from events import tail_events
    for event in tail_events("runs/phone/events.jsonl"):
        print(event.type, event.object, event.frames)   # -> your VLM call

Or from the shell, to watch the contract without writing any code:

    python events.py runs/phone/events.jsonl            # follow live
    python events.py runs/phone/events.jsonl --replay   # existing events, then exit

Running the VLM as its own process rather than a thread inside the pipeline means
it can be restarted, rerun over old events, or rewritten entirely without touching
detection, and a crash in one does not take down the other.

--------------------------------------------------------------------------------
EVENT SCHEMA — one JSON object per line of events.jsonl
--------------------------------------------------------------------------------
  id         int    1-based counter, unique within a run
  type       str    what fired the event. One of:
                      "placed"      an object the wearer moved has come to rest.
                                    This is the main one; `box` marks the object.
                      "sighted"     a target class seen at rest after being out of
                                    view for a while, or seen for the first time.
                                    The put-down itself was missed, so treat the
                                    location as current but the action as
                                    unobserved. Nothing was seen moving, so these
                                    carry a SINGLE frame and t_before/t_during are
                                    null -- do not describe an action from them.
                      "arm_episode" the wearer's arm moved and then stopped while a
                                    target class was visible, but the tracker could
                                    not say which object moved. `box` is null —
                                    the VLM has to find the object itself.
  object     str    the target class that fired, as prompted (e.g. "pill bottle").
                    For "arm_episode" this is the comma-joined list of all targets.
  track      int    tracker ID for the object; -1 for "arm_episode".
  t          float  seconds into the run (or wall-clock epoch on a live source)
                    at which the event fired.
  t_before   float  timestamp of the BEFORE frame; null on a single-frame event.
  t_during   float  timestamp of the DURING frame; null on a single-frame event.
  targets    list   Every prompt the detector could have chosen from. The VLM
                    receives this as its candidate set and decides which one the
                    object actually is; `object` above is only the detector's
                    single best-scoring guess, and it cannot say "none of these".
  box        list   [x1, y1, x2, y2] in the 640px-wide processed frame, or null.
                    The AFTER frame already has this drawn on it in yellow.
  frames     list   Paths to the saved JPEGs, in order: BEFORE, DURING, AFTER.
                    Three when motion was observed, one (AFTER only) for a
                    "sighted" snapshot. Treat the list as ordered-and-possibly-
                    shorter rather than assuming a length -- the `before`,
                    `during` and `after` properties below already do.

Paths in `frames` are written relative to wherever the pipeline was run from, so
resolve them against that directory if the VLM process starts somewhere else.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

# Event types the pipeline can emit. Anything not in here is from a newer pipeline
# than this file knows about — handle it, do not crash on it.
PLACED = "placed"
SIGHTED = "sighted"
ARM_EPISODE = "arm_episode"
KNOWN_TYPES = (PLACED, SIGHTED, ARM_EPISODE)


@dataclass(frozen=True)
class DetectionEvent:
    """One line of events.jsonl, parsed. See the module docstring for field meanings."""

    id: int
    type: str
    object: str
    t: float
    frames: list[str] = field(default_factory=list)
    track: int = -1
    t_before: float | None = None
    t_during: float | None = None
    box: list[float] | None = None
    raw: dict = field(default_factory=dict, repr=False)  # everything as written, for forward compatibility

    @classmethod
    def from_json(cls, line: str | dict) -> "DetectionEvent":
        d = json.loads(line) if isinstance(line, str) else line
        return cls(
            id=d.get("id", -1),
            type=d.get("type", "unknown"),
            object=d.get("object", ""),
            t=float(d.get("t", 0.0)),
            frames=list(d.get("frames", [])),
            track=d.get("track", -1),
            t_before=d.get("t_before"),
            t_during=d.get("t_during"),
            box=d.get("box"),
            raw=d,
        )

    # The three frames, by role. Each is None when the pipeline saved fewer than
    # three, so callers can build a VLM request without index errors.
    @property
    def before(self) -> str | None:
        return self.frames[0] if len(self.frames) >= 3 else None

    @property
    def during(self) -> str | None:
        return self.frames[1] if len(self.frames) >= 3 else None

    @property
    def after(self) -> str | None:
        """The frame showing where the object ended up — the one that matters most."""
        return self.frames[-1] if self.frames else None

    @property
    def has_box(self) -> bool:
        """False for arm_episode: the tracker could not say which object moved."""
        return self.box is not None

    def resolve(self, root: str | Path) -> "DetectionEvent":
        """Return a copy with frame paths resolved against `root`.

        Use this when the VLM process runs from a different directory than the
        pipeline did, which is the normal case when they are separate services.
        """
        root = Path(root)
        return DetectionEvent(
            id=self.id, type=self.type, object=self.object, t=self.t,
            frames=[str(root / f) for f in self.frames], track=self.track,
            t_before=self.t_before, t_during=self.t_during, box=self.box, raw=self.raw,
        )


def read_events(path: str | Path) -> list[DetectionEvent]:
    """Every event written so far. Skips malformed lines rather than raising.

    A partially written last line is possible if the pipeline is mid-write, so a
    bad line is not an error condition — it will be complete on the next read.
    """
    path = Path(path)
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(DetectionEvent.from_json(line))
        except (json.JSONDecodeError, ValueError):
            continue
    return out


def tail_events(path: str | Path, from_start: bool = False, poll_s: float = 0.25) -> Iterator[DetectionEvent]:
    """Yield events as the pipeline writes them. Blocks between events.

    Handles the two things that actually happen in practice:
      - the file does not exist yet, because the VLM service started first
      - the file got shorter, because the pipeline was restarted and reopened it
        with "w"; that is a new run, so start reading it from the top again

    from_start=True replays everything already in the file before following.
    """
    path = Path(path)
    pos = 0
    pending = ""

    # Skip what is already there unless asked to replay it.
    if not from_start and path.exists():
        pos = path.stat().st_size

    while True:
        if not path.exists():
            time.sleep(poll_s)
            continue

        size = path.stat().st_size
        if size < pos:  # truncated: the pipeline restarted
            pos, pending = 0, ""

        if size == pos:
            time.sleep(poll_s)
            continue

        with path.open("r") as f:
            f.seek(pos)
            chunk = f.read()
            pos = f.tell()

        pending += chunk
        # Keep any trailing partial line for the next pass.
        *lines, pending = pending.split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                yield DetectionEvent.from_json(line)
            except (json.JSONDecodeError, ValueError):
                continue


def main() -> None:
    ap = argparse.ArgumentParser(description="Print detection events as the pipeline emits them.")
    ap.add_argument("events", help="path to events.jsonl")
    ap.add_argument("--replay", action="store_true", help="print existing events and exit instead of following")
    args = ap.parse_args()

    def show(e: DetectionEvent) -> None:
        where = "no box (VLM must locate the object)" if not e.has_box else f"box {e.box}"
        print(f"#{e.id:<4} {e.type:<12} {e.object:<20} t={e.t:<9.2f} {where}")
        for role, p in (("BEFORE", e.before), ("DURING", e.during), ("AFTER", e.after)):
            if p:
                print(f"        {role:<7} {p}")

    if args.replay:
        events = read_events(args.events)
        print(f"{len(events)} event(s) in {args.events}")
        for e in events:
            show(e)
        return

    print(f"following {args.events} (Ctrl-C to stop)")
    try:
        for e in tail_events(args.events):
            show(e)
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
