"""Sweep --conf against real frames: does the pill bottle survive, do the distractors die?

    python sweep_conf.py

Two capture phases, then one scoring pass:
  POSITIVE  frames that DO contain the pill bottle  -> every miss is a false negative
  NEGATIVE  frames that do NOT                      -> every target hit is a false positive

Scoring is COMBINED mode, which is what the pipeline actually logs: all prompts go to the
model at once and each box wears the single best-scoring one. Each frame is predicted once
at the lowest threshold and filtered upward, so every threshold sees identical detections.
"""
import time
import warnings

warnings.filterwarnings("ignore")
import cv2
from ultralytics import YOLOE

TARGETS = ["pill bottle", "water bottle", "keys", "phone"]
ARM = "person"
PROMPTS = TARGETS + [ARM]
THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
WEIGHTS = ["weights/yoloe-26s-seg.pt", "yoloe-11s-seg.pt"]
N_FRAMES = 6


def capture(cap, banner, seconds=8):
    print("\n" + "=" * 70)
    print(banner)
    print("=" * 70, flush=True)
    for n in range(seconds, 0, -1):
        print(f"   capturing in {n}...", flush=True)
        t = time.perf_counter()
        while time.perf_counter() - t < 1.0:
            cap.read()          # keep the pipeline fresh so we get a live frame
    out = []
    for _ in range(N_FRAMES):
        ok, f = cap.read()
        if ok:
            out.append(f)
        time.sleep(0.25)
    print(f"   got {len(out)} frames", flush=True)
    return out


def detections(model, frames):
    """[[(label, conf), ...], ...] - one list per frame, at the lowest threshold."""
    per_frame = []
    for f in frames:
        r = model.predict(f, imgsz=640, conf=min(THRESHOLDS), device="cpu", verbose=False)[0]
        per_frame.append([(r.names[int(b.cls)], float(b.conf)) for b in r.boxes])
    return per_frame


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise SystemExit("camera busy - stop the pipeline first")
pos = capture(cap, "PHASE 1 of 2 -- HOLD THE PILL BOTTLE UP, facing the camera.\n"
                   "Keep it in view until this says 'got N frames'.")
neg = capture(cap, "PHASE 2 of 2 -- PUT THE PILL BOTTLE AWAY, out of frame.\n"
                   "Hold up a WATER BOTTLE instead if you have one, and stay in shot yourself.")
cap.release()
for i, f in enumerate(pos):
    cv2.imwrite(f"sweep_pos_{i}.jpg", f)
for i, f in enumerate(neg):
    cv2.imwrite(f"sweep_neg_{i}.jpg", f)

for w in WEIGHTS:
    m = YOLOE(w)
    m.set_classes(PROMPTS)
    print(f"\n\n{'#' * 70}\n# {w}   imgsz=640   COMBINED mode (what the pipeline logs)\n{'#' * 70}")
    dpos, dneg = detections(m, pos), detections(m, neg)

    best_pill = max([c for fr in dpos for lab, c in fr if lab == "pill bottle"], default=0.0)
    best_other = max([c for fr in dpos for lab, c in fr if lab in TARGETS and lab != "pill bottle"], default=0.0)
    print(f"\nWith the pill bottle in frame: best 'pill bottle' score = {best_pill:.3f}")
    print(f"                               best OTHER target score  = {best_other:.3f}"
          f"  {'(a wrong label is outscoring it)' if best_other > best_pill else ''}")
    labs = sorted({lab for fr in dpos for lab, c in fr})
    print(f"                               labels seen              = {labs or 'none'}")

    print(f"\n{'conf':>6} | {'RECALL pill bottle':>19} | {'any target (pos)':>17} | "
          f"{'FALSE POS (neg)':>16} | {'person (ignored)':>16}")
    print("-" * 90)
    for t in THRESHOLDS:
        hit = sum(any(lab == "pill bottle" and c >= t for lab, c in fr) for fr in dpos)
        anyt = sum(any(lab in TARGETS and c >= t for lab, c in fr) for fr in dpos)
        fp = sum(any(lab in TARGETS and c >= t for lab, c in fr) for fr in dneg)
        ppl = sum(any(lab == ARM and c >= t for lab, c in fr) for fr in dneg)
        print(f"{t:>6.2f} | {hit:>8}/{len(dpos)} frames | {anyt:>10}/{len(dpos)} | "
              f"{fp:>9}/{len(dneg)} | {ppl:>9}/{len(dneg)}")
    print("-" * 90)
    print("RECALL should stay at max as long as possible; FALSE POS should reach 0.")
    print("Pick the highest conf where RECALL is still full. person is never a target - it")
    print("cannot fire an event, it only absorbs people so they are not mislabelled.")
