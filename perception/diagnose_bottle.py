"""Why isn't the bottle detected? Counts you in, grabs several frames, and scores each
prompt alone with no confidence floor hiding the answer.

    python diagnose_bottle.py                 # 8s countdown, then capture
    python diagnose_bottle.py shot.jpg        # score a saved photo instead

Hold the bottle where the CAMERA can see it, not where you can. A laptop webcam is
angled up at your face; something on the desk is out of frame entirely.
"""
import sys
import time
import warnings

warnings.filterwarnings("ignore")
import cv2
from ultralytics import YOLOE

PROMPTS = ["pill bottle", "medicine bottle", "prescription bottle", "pill container",
           "water bottle", "bottle", "cup", "person"]
WEIGHTS = ["weights/yoloe-26s-seg.pt", "yoloe-11s-seg.pt"]

frames = []
if len(sys.argv) > 1:
    frames = [cv2.imread(sys.argv[1])]
    print(f"scoring {sys.argv[1]}")
else:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit("camera busy - is the pipeline still running?")
    print("Hold the pill bottle UP, in the camera's view (next to your face works).")
    for n in range(8, 0, -1):
        print(f"  capturing in {n}...", flush=True)
        t = time.perf_counter()
        while time.perf_counter() - t < 1.0:
            cap.read()
    for i in range(5):
        ok, f = cap.read()
        if ok:
            frames.append(f)
        time.sleep(0.3)
    cap.release()
    for i, f in enumerate(frames):
        cv2.imwrite(f"bottle_shot_{i}.jpg", f)
    print(f"captured {len(frames)} frames -> bottle_shot_0..{len(frames)-1}.jpg")

for w in WEIGHTS:
    m = YOLOE(w)
    print(f"\n=== {w}  imgsz=640, best across {len(frames)} frame(s), conf floor 0.01 ===")
    m.set_classes(PROMPTS)
    best_combined = []
    for f in frames:
        r = m.predict(f, imgsz=640, conf=0.01, device="cpu", verbose=False)[0]
        best_combined += [(r.names[int(b.cls)], round(float(b.conf), 3)) for b in r.boxes]
    best_combined.sort(key=lambda x: -x[1])
    print(f"  COMBINED (what the pipeline logs): {best_combined[:6] or 'NOTHING AT ALL'}")
    print(f"  {'prompt':<22}{'best':>7}")
    for p in PROMPTS:
        m.set_classes([p])
        top = 0.0
        for f in frames:
            rr = m.predict(f, imgsz=640, conf=0.01, device="cpu", verbose=False)[0]
            top = max([top] + [float(b.conf) for b in rr.boxes])
        print(f"  {p:<22}{(f'{top:.3f}' if top else '-'):>7}")
