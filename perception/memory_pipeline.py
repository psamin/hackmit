"""Camera -> YOLOE + BoT-SORT -> put-down trigger -> before/during/after frames -> VLM -> memory.jsonl

    python memory_pipeline.py --source data/epic/P02_102.MP4 --out runs/p02 --no-vlm
    python memory_pipeline.py --source 0 --out runs/live          # webcam / Continuity Camera
    python memory_pipeline.py --source ws://0.0.0.0:8765 --out runs/glasses   # glasses relay (see glasses_rx.py)

Trigger, per tracked object:
  MOVING  = the object moves after head motion is removed
  ARMED   = MOVING for >= ACTIVE_MIN_S AND net displacement >= ARM_DISP (jitter on a still object cancels out)
  AT REST = not MOVING and not covered by the wearer's arm
            (arm = a 'person' box touching the bottom edge; YOLOE's 'hand' prompt missed most held objects)
  PLACED  = ARMED, then AT REST for >= REST_MIN_S
            (a track born right after a same-class ARMED track was lost inherits it: ID switch in hand)
            or: ARMED, then AT REST for >= LOST_REST_MIN_S, then out of view (you look away after putting it down)
  SIGHTED = a class seen at rest after being out of view >= UNSEEN_S (catches put-downs the camera missed),
            including the first time it is ever seen, which seeds the log with what is already on the table.
            Nothing was observed moving, so the event carries ONE frame, not before/during/after.

Per wearer's arm (the VLM, not the tracker, decides which object moved):
  ARM_EPISODE = the arm moved (head motion removed) for >= ARM_EP_MIN_S, then went still or left view for
                ARM_QUIET_S, with a target class seen during the episode

LAST SEEN: the newest frame of each target class at rest goes to <out>/last_seen/<class>.jpg (at most once a second);
vlm.ask() describes it when it is newer than the last memory.
"""
import argparse, collections, json, textwrap, threading, time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLOE

ARM = "person"
PROC_W = 640          # frames are resized to this width before anything else
# The gate is specified in SECONDS and converted to frames from --fps at startup.
# It used to be written in frames, which meant --fps silently rescaled every timing in
# it: at 10 fps "6 frames of rest" is 0.6 s, at 3 fps the same 6 frames would have been
# 2 s, so an object had to sit still three times longer before a put-down fired. MOVE_THR
# had the same problem in reverse -- it is a per-frame displacement standing in for a
# speed, so at a third of the frame rate the same physical motion produces three times
# the step and everything reads as moving. These values reproduce 10 fps exactly.
ACTIVE_MIN_S = 0.3      # motion sustained this long can arm the trigger
REST_MIN_S = 0.6        # stillness this long after an armed episode is a put-down
LOST_REST_MIN_S = 0.2   # stillness this long, then out of view, also counts
MOVE_THR_PER_S = 0.15   # ego-compensated centre speed, as a fraction of the diagonal per second
ARM_DISP = 0.08       # net ego-compensated displacement over an episode, as a fraction of the diagonal
CONTACT_THR = 0.5     # share of the object box covered by an arm box
# The user's own arm is CLOSE, so its box is big. A bystander across the room is small
# even when they happen to stand at the bottom edge of the frame. Without this, anyone
# walking past could be read as the user reaching in, and an object they pass in front of
# would count as "held" -- suppressing its put-down. Share of total frame area.
ARM_MIN_AREA = 0.06
ACTIVE_WINDOW_S = 6.0 # an ACTIVE episode older than this no longer arms the trigger
UNSEEN_S = 20.0
BUFFER_S = 12.0
ARM_EP_MIN_S = 1.0    # arm movement needed for an arm episode
ARM_QUIET_S = 0.6     # no arm movement for this long ends the episode
SNAPSHOT_EVERY_S = 1.0
# A copy of the newest frame, whatever is or is not in it, for tools that need to ask
# "what is in front of the camera right now" -- face identification in particular.
# The relay path lets server/app.py tap frames in flight, but a `--source 0` run never
# goes through the server, so without this the face tools would only work off a phone.
FRAME_SNAPSHOT_EVERY_S = 1.0
# A track this far past its last sighting can no longer affect anything: the longest
# lookback in the loop is ACTIVE_WINDOW_S, and the two rules that reach into lost
# tracks need them within 2.0s (ID-switch donor) and 0.5s (lost-then-placed).
TRACK_TTL_S = ACTIVE_WINDOW_S + 2.0
PRUNE_EVERY_S = 2.0   # how often to sweep dead tracks out of the table
KEEP_S = 2.0          # how long a confident sighting lets weaker ones through (see --conf-keep)


def ego_homography(prev_g, g, boxes):
    """Camera motion between frames from background features (object and hand boxes masked out)."""
    mask = np.full(g.shape, 255, np.uint8)
    for x1, y1, x2, y2 in boxes:
        mask[int(y1):int(y2), int(x1):int(x2)] = 0
    p0 = cv2.goodFeaturesToTrack(prev_g, 300, 0.01, 8, mask=mask)
    if p0 is None or len(p0) < 12:
        return None
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_g, g, p0, None)
    ok = st.ravel() == 1
    if ok.sum() < 12:
        return None
    H, _ = cv2.findHomography(p0[ok], p1[ok], cv2.RANSAC, 3.0)
    return H


def pick_device(requested=None):
    """Resolve --device, defaulting to the fastest backend this machine actually has.

    The pipeline was written on Apple Silicon, where "mps" is right. On a
    Windows or Linux box mps does not exist, and asking for it raises inside
    ultralytics instead of falling back — so a teammate on another OS could not
    run the pipeline at all without editing the source. Auto-detection keeps the
    Mac behaviour identical and makes everyone else work.

    An explicit --device is always honoured, including to force cpu when a GPU
    backend is misbehaving.
    """
    if requested:
        return requested
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def log(tag, msg):
    """Same line format as vlm.py and server/app.py, so the three terminals read alike."""
    print(f"{time.strftime('%H:%M:%S')} [{tag:<6}] {msg}", flush=True)


def overlap(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return ix * iy / small if small > 0 else 0.0


class Track:
    def __init__(self, cls):
        self.cls = cls
        self.active_frames = 0
        self.active_start = self.active_end = None
        self.rest = 0
        self.armed = False
        self.disp = np.zeros(2)
        self.last_t, self.last_box = None, None


def annotate(frame, box, label):
    f = frame.copy()
    if box is not None:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(f, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cv2.putText(f, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return f


def frame_at(buffer, t):
    return min(buffer, key=lambda item: abs(item[0] - t))


def draw_overlay(frame, boxes, names, confs, ids, targets, hud, fps, subtitle, arm_boxes=()):
    """The --show window: what the detector sees on top, what the VLM said underneath.

    Deliberately reads off the same `r.boxes` the trigger uses, so the window cannot
    disagree with what the pipeline actually acted on.
    """
    f = frame.copy()
    h, w = f.shape[:2]
    # The person class stays in the vocabulary whatever happens -- it absorbs people, who
    # would otherwise be labelled as one of the real targets. But only a box that passed
    # the arm test influences anything, so only those are worth drawing; bystanders in the
    # background are noise on screen and nothing at all to the trigger.
    arm_keys = {tuple(round(float(v), 1) for v in a) for a in arm_boxes}
    for b, n, c, i in zip(boxes, names, confs, ids):
        target = n in targets
        if not target and tuple(round(float(v), 1) for v in b) not in arm_keys:
            continue
        x1, y1, x2, y2 = (int(v) for v in b)
        colour = (0, 255, 255) if target else (200, 130, 0)   # targets yellow, the arm/person blue
        cv2.rectangle(f, (x1, y1), (x2, y2), colour, 2 if target else 1)
        label = f"{n} {c:.2f}" + (f" #{i}" if target and i >= 0 else "")
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(f, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), colour, -1)
        cv2.putText(f, label, (x1 + 3, max(10, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    cv2.rectangle(f, (0, 0), (w, 22), (0, 0, 0), -1)
    cv2.putText(f, f"{fps:4.1f} fps   {subtitle}   events {hud['events']}   "
                   f"vlm {hud['calls']} calls {hud['in_tok']}>{hud['out_tok']} tok   q=quit",
                (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # VLM state: in flight, then the answer it came back with.
    lines, colour = [], (60, 200, 60)
    if hud["pending"]:
        lines, colour = [f"VLM THINKING... {hud['last_event']}"], (0, 165, 255)
    elif hud["memory"]:
        m = hud["memory"]
        lines = wrap_lines(f"{m.get('event', '?')}  {m.get('object', '?')}  "
                           f"(confidence {m.get('confidence', 0)})", w)
        lines += wrap_lines(m.get("location_description", ""), w)
        if m.get("event") == "error":
            colour = (60, 60, 230)
    if lines:
        top = h - 20 * len(lines) - 10
        cv2.rectangle(f, (0, top), (w, h), (0, 0, 0), -1)
        cv2.rectangle(f, (0, top), (w, top + 3), colour, -1)
        for k, line in enumerate(lines):
            cv2.putText(f, line, (8, top + 24 + 20 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.52, colour, 1)
    return f


def wrap_lines(text, width_px):
    return textwrap.wrap(text, max(20, int(width_px / 9))) if text else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="video path or camera index")
    # Never put a generic prompt ("bottle") in here alongside a specific one
    # ("pill bottle"). YOLOE labels each box with the single best-scoring prompt, so
    # the generic one wins every time and the specific one never appears. Measured on
    # a photo of real prescription bottles: with ["pill bottle","water bottle"] it
    # returned pill bottle x7; adding "bottle" turned all 9 into "bottle".
    ap.add_argument("--targets", default="pill bottle,water bottle,keys,phone,glasses",
                    help="comma-separated object prompts; keep them mutually specific, no generic catch-alls")
    ap.add_argument("--weights", default="weights/yoloe-26s-seg.pt")
    ap.add_argument("--fps", type=float, default=10.0, help="processing rate")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    # 0.10 sat inside the noise floor. Measured over 128 photos containing none of
    # these objects, the worst false "pill bottle" scored 0.32, while real pill
    # bottles reach 0.46 -- so 0.10 was admitting every false fire. 0.30 clears most
    # of them; 0.35 cleared all in testing, at some cost to small/distant objects.
    # The multi-frame gate (ACTIVE_MIN, REST_MIN) already discards one-frame flukes,
    # so this does not have to be set as high as a single-frame classifier would need.
    ap.add_argument("--conf", type=float, default=0.30,
                    help="detection threshold; 0.35 removed all measured false positives, 0.10 is too low")
    # Inference resolution. Measured on CPU: 640 = 3.5fps, 480 = 5.6, 416 = 9.2, 320 = 14.2.
    # 416 is the CPU sweet spot, but smaller hurts small objects -- keys first -- so the default
    # stays at ultralytics' 640, which is what every run so far actually used. Drop to 416 on a
    # CPU-only machine; on mps/cuda 640 is already fast enough.
    ap.add_argument("--imgsz", type=int, default=640, help="inference size; 416 roughly triples CPU fps, at some cost to small objects")
    # Hysteresis. A handheld camera walks the same object back and forth across a single
    # threshold: measured on one run, the same pill bottle scored 0.56, 0.41, 0.24, 0.16
    # on consecutive appearances, so a fixed --conf makes it blink in and out and the
    # tracker keeps losing and re-acquiring it. Two thresholds fix that the way a Schmitt
    # trigger does: --conf to START believing a class is there, --conf-keep (lower) to GO
    # ON believing it for KEEP_S afterwards. Default equals --conf, i.e. off.
    ap.add_argument("--conf-keep", type=float, default=None,
                    help="lower threshold that sustains an already-seen class (handheld cameras); default: same as --conf")
    ap.add_argument("--device", default=None, help="mps/cuda/cpu; default: best available on this machine")
    ap.add_argument("--cert", help="TLS certificate for a wss:// source (see phone/serve.py)")
    ap.add_argument("--key", help="TLS private key for a wss:// source")
    ap.add_argument("--out", default="runs/latest")
    ap.add_argument("--no-vlm", action="store_true")
    # Not merely an ablation. The arm rules assume a HEAD-WORN camera, where a person box
    # running off the bottom edge is the wearer's own arm reaching in. On a webcam or a
    # propped phone facing the user, that same test matches their whole seated body, so
    # any object overlapping them counts as "held", never reaches rest, and never fires a
    # put-down at all. Use --no-arm whenever the camera looks AT a person rather than out
    # from one; it also drops the person boxes from the --show window.
    ap.add_argument("--no-arm", action="store_true",
                    help="camera faces the user (webcam/propped phone) or ablation: trigger on motion only")
    ap.add_argument("--show", action="store_true",
                    help="live demo window: detections, then the VLM's memory as it comes back")
    ap.add_argument("--static-camera", action="store_true",
                    help="camera does not move (phone propped on a table/dock): skip ego-motion removal")
    args = ap.parse_args()

    args.device = pick_device(args.device)
    # Seconds -> frames. The floor of 2 keeps the multi-frame filter meaningful at a low
    # --fps: one frame is a fluke, two is a signal, and at 3 fps 0.3 s rounds to 1.
    ACTIVE_MIN = max(2, round(ACTIVE_MIN_S * args.fps))
    REST_MIN = max(2, round(REST_MIN_S * args.fps))
    LOST_REST_MIN = max(1, round(LOST_REST_MIN_S * args.fps))
    MOVE_THR = MOVE_THR_PER_S / args.fps
    log("START", f"device={args.device} fps={args.fps} imgsz={args.imgsz} "
                 f"conf={args.conf}{'' if args.conf_keep is None else f'/keep {args.conf_keep}'} "
                 f"static_camera={args.static_camera} arm={'off' if args.no_arm else 'on'}")
    log("START", f"gate: arm after {ACTIVE_MIN} frames of motion, fire after {REST_MIN} frames "
                 f"at rest ({REST_MIN / args.fps:.1f}s), move threshold {MOVE_THR:.4f}/frame")
    targets = [t.strip() for t in args.targets.split(",")]
    log("START", f"targets: {', '.join(targets)}   (+ '{ARM}' as the arm/decoy class)")
    out = Path(args.out); (out / "events").mkdir(parents=True, exist_ok=True); (out / "last_seen").mkdir(exist_ok=True)
    model = YOLOE(args.weights); model.set_classes(targets + [ARM])

    live = args.source.isdigit() or args.source.startswith("ws")
    if args.source.startswith("ws"):
        from glasses_rx import GlassesStream
        cap = GlassesStream(args.source, args.fps, args.cert, args.key)
    else:
        cap = cv2.VideoCapture(int(args.source) if live else args.source)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(src_fps / args.fps))
    if not live and args.start:
        cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000)

    vlm, vlm_cost = None, None
    if not args.no_vlm:
        from vlm import MODEL, cost_usd, describe_event
        vlm, vlm_cost = describe_event, cost_usd
        log("START", f"vlm: {MODEL} (an event costs roughly $0.005)")

    buffer = collections.deque(maxlen=int(BUFFER_S * args.fps))
    tracks, last_seen, reappeared, snap_t = {}, {}, {}, {}
    prev_g, prev_boxes, prev_centres, prev_arm_c, arm_ep = None, [], {}, None, None
    events_f = open(out / "events.jsonl", "w")
    vlm_threads = []
    # What the --show window reports. Written from the VLM threads, read by the draw call,
    # so it is guarded: dict item assignment is atomic in CPython but a counter is not.
    hud = {"events": 0, "pending": 0, "memory": None, "last_event": "",
           "calls": 0, "in_tok": 0, "out_tok": 0}
    # What was visible last frame. Logging every frame would be a flood at 5fps and
    # unreadable; what you actually want to see is the moment something appears or goes.
    seen_before = set()
    conf_keep = args.conf if args.conf_keep is None else min(args.conf_keep, args.conf)
    last_strong = {}   # class -> when it was last seen above the full --conf
    hud_lock = threading.Lock()
    timing = collections.defaultdict(float)
    n_frames, n_events, t_wall, last_prune = 0, 0, time.perf_counter(), -1e18
    idx = int(args.start * src_fps) if not live else 0

    def emit(fire, n, i, tr, t, b, frame):
        tr.armed, tr.active_frames, reappeared[n] = False, 0, False
        # A track that never had an ACTIVE episode -- a "sighted" object, at rest and never
        # observed moving -- has no before or during to sample. Passing t_before=None marks
        # it a snapshot: one frame. Sampling three anyway gave the VLM the same moment three
        # times labelled BEFORE/DURING/AFTER, which reads as a placement it then described.
        if tr.active_start is None:
            save_event(fire, n, i, t, None, None, b, frame)
        else:
            save_event(fire, n, i, t, tr.active_start - 1.0, tr.active_end, b, frame)

    def save_event(fire, n, i, t, t_before, t_during, b, frame):
        nonlocal n_events
        n_events += 1
        snapshot = t_before is None  # no observed motion: AFTER alone, see emit()
        ev = {"id": n_events, "type": fire, "object": n, "track": i, "t": round(t, 2),
              "t_before": None if snapshot else round(t_before, 2),
              "t_during": None if snapshot else round(t_during, 2),
              "box": None if b is None else [round(float(v), 1) for v in b],
              # Every prompt the detector could have chosen from. The VLM gets this as its
              # candidate set: the detector has no "none of these" option and confuses
              # visually similar prompts, so its single label is a hint, not an answer.
              "targets": list(targets)}
        d = out / "events" / f"{n_events:03d}_{fire}_{t:07.1f}"
        d.mkdir(exist_ok=True)
        ev["frames"] = []
        shots = [("after", (t, frame))] if snapshot else [
            ("before", frame_at(buffer, t_before)), ("during", frame_at(buffer, t_during)), ("after", (t, frame))]
        for name, (_, fr) in shots:
            path = d / f"{name}.jpg"
            cv2.imwrite(str(path), annotate(fr, b if name == "after" else None, n))
            ev["frames"].append(str(path))
        events_f.write(json.dumps(ev) + "\n"); events_f.flush()
        log("EVENT", f"#{n_events} {fire} {n} (track {i}) at t={t:.1f}s, "
                     f"{len(ev['frames'])} frame(s) -> events.jsonl")
        with hud_lock:
            hud["events"] = n_events
            hud["last_event"] = f"{fire} {n}"
        if vlm:
            with hud_lock:
                hud["pending"] += 1
            # Daemon so Ctrl-C is never blocked by a hung request, but kept in a list and
            # joined at the end: a VLM call takes seconds, events fire right up to the last
            # frame, and without the join the interpreter exits first and those memories are
            # silently lost -- exactly the ones a short demo clip produces.
            log("VLM", f"queued event {n_events} ({hud['pending']} in flight)")
            th = threading.Thread(target=run_vlm, args=(ev,), daemon=True)
            th.start()
            vlm_threads[:] = [x for x in vlm_threads if x.is_alive()] + [th]

    def run_vlm(ev):
        """Call the VLM and report the answer to the window. A failure here -- no API key,
        no network, a refusal -- is shown rather than left as a thread traceback nobody reads."""
        try:
            mem = vlm(ev, out / "memory.jsonl")
        except Exception as exc:
            mem = {"event": "error", "object": ev["object"], "confidence": 0.0,
                   "location_description": f"{type(exc).__name__}: {exc}"}
            log("VLM", f"event {ev['id']} FAILED: {type(exc).__name__}: {exc}")
        with hud_lock:
            hud["pending"] -= 1
            if mem:
                hud["memory"] = mem
                hud["calls"] += 1
                hud["in_tok"] += mem.get("input_tokens", 0)
                hud["out_tok"] += mem.get("output_tokens", 0)

    while True:
        try:
            ok = cap.grab()
        except KeyboardInterrupt:  # live sources only end on Ctrl-C
            break
        if not ok:
            break
        idx += 1
        if idx % step:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break
        t = time.time() if live else idx / src_fps
        if args.end is not None and t > args.end:
            break
        frame = cv2.resize(frame, (PROC_W, int(frame.shape[0] * PROC_W / frame.shape[1])))
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diag = float(np.hypot(*g.shape))

        t0 = time.perf_counter()
        # The model runs at the LOWER threshold so the tracker keeps seeing a weak object
        # and holds its ID; the hysteresis below decides what the trigger is allowed to act on.
        r = model.track(frame, persist=True, tracker="botsort.yaml", conf=conf_keep, imgsz=args.imgsz,
                        device=args.device, verbose=False)[0]
        timing["detect_track"] += time.perf_counter() - t0

        boxes = r.boxes.xyxy.cpu().numpy() if len(r.boxes) else np.zeros((0, 4))
        names = [r.names[int(c)] for c in r.boxes.cls] if len(r.boxes) else []
        ids = r.boxes.id.int().tolist() if r.boxes.id is not None else [-1] * len(names)
        confs = r.boxes.conf.cpu().numpy() if len(r.boxes) else np.zeros(0)

        if conf_keep < args.conf:
            for n, c in zip(names, confs):
                if c >= args.conf:
                    last_strong[n] = t
            keep = [i for i, (n, c) in enumerate(zip(names, confs))
                    if c >= args.conf or t - last_strong.get(n, -1e18) <= KEEP_S]
            boxes = boxes[keep] if len(keep) else np.zeros((0, 4))
            names = [names[i] for i in keep]
            ids = [ids[i] for i in keep]
            confs = confs[keep] if len(keep) else np.zeros(0)
        frame_area = frame.shape[0] * frame.shape[1]
        arms = [] if args.no_arm else [
            b for b, n in zip(boxes, names)
            if n == ARM and b[3] >= 0.9 * frame.shape[0]                       # reaches the near edge
            and (b[2] - b[0]) * (b[3] - b[1]) >= ARM_MIN_AREA * frame_area]    # and is close enough to be ours

        # A propped phone has no camera motion to remove, so the homography is both
        # wasted work (feature detection + optical flow + RANSAC on every frame) and a
        # source of error: on a mostly-empty table there are few background features to
        # fit, and a bad fit shows up as phantom object motion.
        t0 = time.perf_counter()
        H = None if args.static_camera else (ego_homography(prev_g, g, prev_boxes) if prev_g is not None else None)
        timing["ego_motion"] += time.perf_counter() - t0

        centres = {}
        buffer.append((t, frame))
        present = {n for n in names if n in targets}
        if present != seen_before:
            best = {}
            for nm, cf in zip(names, (r.boxes.conf.cpu().numpy() if len(r.boxes) else [])):
                if nm in present:
                    best[nm] = max(best.get(nm, 0.0), float(cf))
            gone = seen_before - present
            if best:
                log("DETECT", "seeing " + ", ".join(f"{k} {v:.2f}" for k, v in sorted(best.items()))
                              + (f"   (lost: {', '.join(sorted(gone))})" if gone else ""))
            elif gone:
                log("DETECT", f"nothing in view (lost: {', '.join(sorted(gone))})")
            seen_before = present
        for n in present:
            if n not in last_seen or t - last_seen[n] > UNSEEN_S:
                reappeared[n] = True
        for b, n, i in zip(boxes, names, ids):
            if n not in targets or i < 0:
                continue
            c = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
            centres[i] = c
            tr = tracks.get(i)
            if tr is None:
                tr = tracks[i] = Track(n)
                lost = [o for j, o in tracks.items() if j not in ids and o.cls == n and o.armed and t - o.active_end <= 2.0]
                if lost:  # ID switch while in hand: carry the ACTIVE episode over
                    donor = max(lost, key=lambda o: o.active_end)
                    tr.armed, tr.active_start, tr.active_end, tr.active_frames, tr.disp = True, donor.active_start, donor.active_end, donor.active_frames, donor.disp
                    donor.armed = False
            # How far the object moved since the last frame, with camera motion removed.
            # Static camera: the raw centre delta already is that. Moving camera: warp the
            # previous centre through the homography first. If the homography failed we
            # leave this at zero, which reads as "not moving" — deliberately conservative,
            # since a bad fit would otherwise fire put-downs at random.
            step_vec = np.zeros(2)
            if i in prev_centres:
                if args.static_camera:
                    step_vec = c - prev_centres[i]
                elif H is not None:
                    step_vec = c - cv2.perspectiveTransform(prev_centres[i].reshape(1, 1, 2), H).ravel()
            moving = np.linalg.norm(step_vec) / diag > MOVE_THR
            covered = any(overlap(b, a) > CONTACT_THR for a in arms)

            tr.last_t, tr.last_box = t, b
            fire = None
            if moving:
                if tr.active_frames == 0 or t - tr.active_end > ACTIVE_WINDOW_S:
                    tr.active_start, tr.active_frames, tr.disp = t, 0, np.zeros(2)
                tr.active_frames += 1
                tr.active_end = t
                tr.disp = tr.disp + step_vec
                tr.armed = tr.armed or (tr.active_frames >= ACTIVE_MIN and np.linalg.norm(tr.disp) / diag >= ARM_DISP)
                tr.rest = 0
            elif covered:  # held still: not at rest, and an armed episode stays alive
                if tr.armed:
                    tr.active_end = t
                tr.rest = 0
            else:
                tr.rest += 1
                if tr.rest >= REST_MIN and t - snap_t.get(n, -1e18) >= SNAPSHOT_EVERY_S:
                    snap_t[n] = t
                    cv2.imwrite(str(out / "last_seen" / f"{n}.jpg"), annotate(frame, b, n))
                    json.dump({"object": n, "t": round(t, 2), "box": [round(float(v), 1) for v in b],
                               "frame": str(out / "last_seen" / f"{n}.jpg")}, open(out / "last_seen" / f"{n}.json", "w"))
                if tr.armed and tr.rest >= REST_MIN and t - tr.active_end <= ACTIVE_WINDOW_S:
                    fire = "placed"
                elif reappeared.get(n) and tr.rest >= REST_MIN:
                    fire = "sighted"
            if not fire:
                continue
            emit(fire, n, i, tr, t, b, frame)
        for j, tr in tracks.items():
            if j not in ids and tr.armed and tr.rest >= LOST_REST_MIN and t - tr.last_t <= 0.5 and t - tr.active_end <= ACTIVE_WINDOW_S:
                emit("placed", tr.cls, j, tr, tr.last_t, tr.last_box, frame_at(buffer, tr.last_t)[1])

        arm_c = None
        if arms:
            a = max(arms, key=lambda a: (a[2] - a[0]) * (a[3] - a[1]))
            arm_c = np.array([(a[0] + a[2]) / 2, (a[1] + a[3]) / 2])
        if arm_c is not None and prev_arm_c is not None and (H is not None or args.static_camera):
            arm_step = (arm_c - prev_arm_c) if args.static_camera else \
                       (arm_c - cv2.perspectiveTransform(prev_arm_c.reshape(1, 1, 2), H).ravel())
            if np.linalg.norm(arm_step) / diag > MOVE_THR:
                arm_ep = arm_ep or {"start": t, "saw": False}
                arm_ep["last_move"] = t
        if arm_ep:
            arm_ep["saw"] = arm_ep["saw"] or bool(present)
            if t - arm_ep["last_move"] >= ARM_QUIET_S:
                if arm_ep["last_move"] - arm_ep["start"] >= ARM_EP_MIN_S and arm_ep["saw"]:
                    save_event("arm_episode", ",".join(targets), -1, t, arm_ep["start"] - 1.0, arm_ep["last_move"], None, frame)
                arm_ep = None
        for n in present:
            last_seen[n] = t

        # Drop tracks that can no longer fire anything. Without this the table keeps
        # every track ID the tracker ever issued: on a run of any length that is an
        # unbounded dict, and the ID-switch donor lookup below rescans all of it every
        # time a new track appears, so the cost grows with uptime. That is fine for a
        # 60-second clip and not fine for a device meant to watch a room all day.
        if t - last_prune >= PRUNE_EVERY_S:
            last_prune = t
            for j in [j for j, tr in tracks.items() if tr.last_t is not None and t - tr.last_t > TRACK_TTL_S]:
                del tracks[j]

        if t - snap_t.get("_frame", -1e18) >= FRAME_SNAPSHOT_EVERY_S:
            snap_t["_frame"] = t
            cv2.imwrite(str(out / "last_seen" / "_frame.jpg"), frame)

        prev_g, prev_boxes, prev_centres, prev_arm_c = g, list(boxes), centres, arm_c
        n_frames += 1

        if args.show:
            now = time.perf_counter()
            shown_fps = n_frames / max(now - t_wall, 1e-6)
            with hud_lock:
                snapshot = dict(hud)
            cv2.imshow("Compass - memory pipeline",
                       draw_overlay(frame, boxes, names, confs, ids, targets, snapshot, shown_fps,
                                    f"{args.device} imgsz={args.imgsz} conf={args.conf}",
                                    arm_boxes=arms))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    pending = [th for th in vlm_threads if th.is_alive()]
    if pending:
        print(f"waiting for {len(pending)} VLM call(s) to finish (Ctrl-C to abandon them)...", flush=True)
        try:
            for th in pending:
                th.join()
        except KeyboardInterrupt:
            print("abandoned; those memories were not written", flush=True)
    events_f.close()
    if hasattr(cap, "release"):  # GlassesStream has no release(); cv2.VideoCapture does
        cap.release()
    if args.show:
        cv2.destroyAllWindows()

    wall = time.perf_counter() - t_wall
    stats = {"frames": n_frames, "events": n_events, "fps": args.fps, "imgsz": args.imgsz,
             "wall_s": round(wall, 1), "proc_fps": round(n_frames / wall, 1),
             **{f"{k}_ms": round(1000 * v / max(n_frames, 1), 1) for k, v in timing.items()},
             "vlm_calls": hud["calls"], "vlm_input_tokens": hud["in_tok"], "vlm_output_tokens": hud["out_tok"]}
    if vlm_cost:
        stats["vlm_cost_usd"] = round(vlm_cost(hud["in_tok"], hud["out_tok"]), 4)
    log("DONE", f"{n_frames} frames, {n_events} events, {hud['calls']} VLM call(s), "
                f"{hud['in_tok']}/{hud['out_tok']} tokens"
                + (f", ${stats['vlm_cost_usd']:.4f}" if "vlm_cost_usd" in stats else ""))
    print(json.dumps(stats))
    json.dump(stats, open(out / "stats.json", "w"))


if __name__ == "__main__":
    main()
