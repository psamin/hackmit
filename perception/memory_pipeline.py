"""Camera -> YOLOE + BoT-SORT -> put-down trigger -> before/during/after frames -> VLM -> memory.jsonl

    python memory_pipeline.py --source data/epic/P02_102.MP4 --out runs/p02 --no-vlm
    python memory_pipeline.py --source 0 --out runs/live          # webcam / Continuity Camera
    python memory_pipeline.py --source ws://0.0.0.0:8765 --out runs/glasses   # glasses relay (see glasses_rx.py)

Trigger, per tracked object:
  MOVING  = the object moves after head motion is removed
  ARMED   = MOVING for >= ACTIVE_MIN frames AND net displacement >= ARM_DISP (jitter on a still object cancels out)
  AT REST = not MOVING and not covered by the wearer's arm
            (arm = a 'person' box touching the bottom edge; YOLOE's 'hand' prompt missed most held objects)
  PLACED  = ARMED, then AT REST for >= REST_MIN frames
            (a track born right after a same-class ARMED track was lost inherits it: ID switch in hand)
            or: ARMED, then AT REST for >= LOST_REST_MIN frames, then out of view (you look away after putting it down)
  SIGHTED = a class seen at rest after being out of view >= UNSEEN_S (catches put-downs the camera missed)

Per wearer's arm (the VLM, not the tracker, decides which object moved):
  ARM_EPISODE = the arm moved (head motion removed) for >= ARM_EP_MIN_S, then went still or left view for
                ARM_QUIET_S, with a target class seen during the episode

LAST SEEN: the newest frame of each target class at rest goes to <out>/last_seen/<class>.jpg (at most once a second);
vlm.ask() describes it when it is newer than the last memory.
"""
import argparse, collections, json, threading, time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLOE

ARM = "person"
PROC_W = 640          # frames are resized to this width before anything else
ACTIVE_MIN = 3        # frames
REST_MIN = 6          # frames of rest before a put-down fires
LOST_REST_MIN = 2     # frames of rest before an object that then leaves the view counts as put down
MOVE_THR = 0.015      # ego-compensated centre motion per frame, as a fraction of the frame diagonal
ARM_DISP = 0.08       # net ego-compensated displacement over an episode, as a fraction of the diagonal
CONTACT_THR = 0.5     # share of the object box covered by an arm box
ACTIVE_WINDOW_S = 6.0 # an ACTIVE episode older than this no longer arms the trigger
UNSEEN_S = 20.0
BUFFER_S = 12.0
ARM_EP_MIN_S = 1.0    # arm movement needed for an arm episode
ARM_QUIET_S = 0.6     # no arm movement for this long ends the episode
SNAPSHOT_EVERY_S = 1.0


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="video path or camera index")
    ap.add_argument("--targets", default="bottle", help="comma-separated object prompts")
    ap.add_argument("--weights", default="weights/yoloe-26s-seg.pt")
    ap.add_argument("--fps", type=float, default=10.0, help="processing rate")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--cert", help="TLS certificate for a wss:// source (see phone/serve.py)")
    ap.add_argument("--key", help="TLS private key for a wss:// source")
    ap.add_argument("--out", default="runs/latest")
    ap.add_argument("--no-vlm", action="store_true")
    ap.add_argument("--no-arm", action="store_true", help="ablation: trigger on motion only")
    args = ap.parse_args()

    targets = [t.strip() for t in args.targets.split(",")]
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

    vlm = None
    if not args.no_vlm:
        from vlm import describe_event
        vlm = describe_event

    buffer = collections.deque(maxlen=int(BUFFER_S * args.fps))
    tracks, last_seen, reappeared, snap_t = {}, {}, {}, {}
    prev_g, prev_boxes, prev_centres, prev_arm_c, arm_ep = None, [], {}, None, None
    events_f = open(out / "events.jsonl", "w")
    timing = collections.defaultdict(float)
    n_frames, n_events, t_wall = 0, 0, time.perf_counter()
    idx = int(args.start * src_fps) if not live else 0

    def emit(fire, n, i, tr, t, b, frame):
        tr.armed, tr.active_frames, reappeared[n] = False, 0, False
        save_event(fire, n, i, t, (tr.active_start or t) - 1.0, tr.active_end or t, b, frame)

    def save_event(fire, n, i, t, t_before, t_during, b, frame):
        nonlocal n_events
        n_events += 1
        ev = {"id": n_events, "type": fire, "object": n, "track": i, "t": round(t, 2),
              "t_before": round(t_before, 2), "t_during": round(t_during, 2),
              "box": None if b is None else [round(float(v), 1) for v in b]}
        d = out / "events" / f"{n_events:03d}_{fire}_{t:07.1f}"
        d.mkdir(exist_ok=True)
        ev["frames"] = []
        for name, (_, fr) in (("before", frame_at(buffer, t_before)), ("during", frame_at(buffer, t_during)), ("after", (t, frame))):
            path = d / f"{name}.jpg"
            cv2.imwrite(str(path), annotate(fr, b if name == "after" else None, n))
            ev["frames"].append(str(path))
        events_f.write(json.dumps(ev) + "\n"); events_f.flush()
        print(f"[{t:7.1f}s] EVENT {fire} {n} #{i}", flush=True)
        if vlm:
            threading.Thread(target=vlm, args=(ev, out / "memory.jsonl"), daemon=True).start()

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
        r = model.track(frame, persist=True, tracker="botsort.yaml", conf=args.conf, device=args.device, verbose=False)[0]
        timing["detect_track"] += time.perf_counter() - t0

        boxes = r.boxes.xyxy.cpu().numpy() if len(r.boxes) else np.zeros((0, 4))
        names = [r.names[int(c)] for c in r.boxes.cls] if len(r.boxes) else []
        ids = r.boxes.id.int().tolist() if r.boxes.id is not None else [-1] * len(names)
        arms = [] if args.no_arm else [b for b, n in zip(boxes, names) if n == ARM and b[3] >= 0.9 * frame.shape[0]]

        t0 = time.perf_counter()
        H = ego_homography(prev_g, g, prev_boxes) if prev_g is not None else None
        timing["ego_motion"] += time.perf_counter() - t0

        centres = {}
        buffer.append((t, frame))
        present = {n for n in names if n in targets}
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
            step_vec = np.zeros(2)
            if H is not None and i in prev_centres:
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
        if arm_c is not None and prev_arm_c is not None and H is not None:
            arm_step = arm_c - cv2.perspectiveTransform(prev_arm_c.reshape(1, 1, 2), H).ravel()
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

        prev_g, prev_boxes, prev_centres, prev_arm_c = g, list(boxes), centres, arm_c
        n_frames += 1

    wall = time.perf_counter() - t_wall
    stats = {"frames": n_frames, "events": n_events, "wall_s": round(wall, 1), "proc_fps": round(n_frames / wall, 1),
             **{f"{k}_ms": round(1000 * v / max(n_frames, 1), 1) for k, v in timing.items()}}
    print(json.dumps(stats))
    json.dump(stats, open(out / "stats.json", "w"))


if __name__ == "__main__":
    main()
