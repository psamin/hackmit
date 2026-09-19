"""Blocker 1: can an off-the-shelf detector see a pill bottle, and how fast on this laptop?
Hit = any box of a target class above the threshold (single-bottle close-ups, so no IoU check)."""
import glob, sys, time
from ultralytics import YOLO, YOLOE, YOLOWorld

imgs = sorted(glob.glob("data/pill_web/*.jpg"))
device = sys.argv[1] if len(sys.argv) > 1 else "mps"
THRS = (0.05, 0.10, 0.25)

def run(name, model, target):
    model.predict(imgs[0], device=device, verbose=False)  # warmup
    confs, t = [], 0.0
    for f in imgs:
        t0 = time.perf_counter()
        r = model.predict(f, device=device, imgsz=640, conf=0.01, verbose=False)[0]
        t += time.perf_counter() - t0
        c = [float(x) for x, k in zip(r.boxes.conf, r.boxes.cls) if r.names[int(k)] in target]
        confs.append(max(c, default=0.0))
    hits = "  ".join(f"@{th:.2f}: {sum(c >= th for c in confs):2d}/{len(imgs)}" for th in THRS)
    print(f"{name:38s} {hits}   {1000*t/len(imgs):4.0f} ms/img", flush=True)

def yoloe(w, classes):
    m = YOLOE(w); m.set_classes(classes); return m

run("yolo11n COCO bottle", YOLO("weights/yolo11n.pt"), {"bottle"})
run("yolo26n COCO bottle", YOLO("weights/yolo26n.pt"), {"bottle"})
run("yolo26s COCO bottle", YOLO("weights/yolo26s.pt"), {"bottle"})
run("yolo26x COCO bottle", YOLO("weights/yolo26x.pt"), {"bottle"})
for w in ("yoloe-26s-seg", "yoloe-26l-seg"):
    for p in ("pill bottle", "medicine bottle", "bottle", "plastic bottle"):
        run(f"{w} '{p}'", yoloe(f"weights/{w}.pt", [p, "hand"]), {p})
    run(f"{w} any of 4 prompts", yoloe(f"weights/{w}.pt", ["pill bottle", "medicine bottle", "bottle", "plastic bottle", "hand"]),
        {"pill bottle", "medicine bottle", "bottle", "plastic bottle"})
m = YOLOWorld("weights/yolov8x-worldv2.pt"); m.set_classes(["pill bottle", "bottle", "hand"])
run("yolov8x-worldv2 'pill bottle'|'bottle'", m, {"pill bottle", "bottle"})
