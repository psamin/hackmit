"""Does YOLOE actually tell your pill bottle apart from a water bottle? Point a camera at both and find out.

    python test_prompts.py                          # webcam, live window
    python test_prompts.py --image shot.jpg         # one photo instead
    python test_prompts.py --isolate                # the important one, see below

Why this exists: YOLOE is open-vocabulary, so "pill bottle" is a text prompt, not
a trained class. Testing on 5 real photos of ORDINARY bottles (no pill bottles in
frame at all), the prompt "medicine bottle" fired on 4 of them at 0.14-0.34
confidence. It matches "bottle-shaped thing" and then wears whatever label you
typed. Before building a demo on top of that, measure it on your own objects.

Two modes, and the difference matters:

COMBINED (default) is how the pipeline really runs: every prompt goes to the model
at once and each detection is labelled with the single best-scoring prompt. This is
winner-take-all -- in testing, putting "bottle" and "pill bottle" in together meant
"bottle" won every detection and "pill bottle" never appeared once. So combined mode
tells you what the pipeline will actually log.

ISOLATE runs each prompt on its own against the same frame and reports every score.
That is what reveals whether a prompt discriminates or just fires at anything
bottle-shaped. Read it like this, pointing the camera at BOTH bottles:

  good   "pill bottle" scores high on the pill bottle, low/zero on the water bottle
  bad    "pill bottle" scores about the same on both -> no discrimination, the label
         is meaningless and the pipeline will confuse the two

Keys are the other thing to check here: they are small, and small objects are the
first casualty when you shrink --imgsz for speed on a CPU-only laptop.

Setup, if this is a fresh machine:
    pip install ultralytics opencv-python
"""
import argparse
import time

import cv2
from ultralytics import YOLOE

# Prompts worth comparing. The pairs matter: a specific prompt next to the generic
# one it might collapse into, so you can see which way a detection goes.
DEFAULT_PROMPTS = [
    "pill bottle",
    "medicine bottle",
    "prescription bottle",
    "water bottle",
    "bottle",
    "keys",
    "cell phone",
]

# One distinct colour per prompt so the live window is readable at a glance.
COLOURS = [
    (0, 255, 255), (0, 200, 0), (255, 120, 0), (200, 0, 255),
    (0, 128, 255), (255, 255, 0), (128, 128, 255), (0, 0, 255),
]


def draw(frame, result, prompts):
    """Box + label + confidence for every detection, colour-coded by prompt."""
    for b in result.boxes:
        name = result.names[int(b.cls)]
        conf = float(b.conf)
        x1, y1, x2, y2 = (int(v) for v in b.xyxy[0])
        colour = COLOURS[prompts.index(name) % len(COLOURS)] if name in prompts else (255, 255, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        label = f"{name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), colour, -1)
        cv2.putText(frame, label, (x1 + 3, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return frame


def isolate_report(model, frame, prompts, conf, imgsz, device):
    """Run every prompt alone against one frame and print all the scores side by side.

    Each row is one prompt's view of the same picture. Compare rows: a prompt that
    only discriminates in your head will score on everything bottle-shaped.
    """
    print("\n" + "=" * 68)
    print("ISOLATED — each prompt run alone on this frame")
    print("=" * 68)
    print(f"{'prompt':<24} {'hits':>5}  {'confidences (highest first)'}")
    print("-" * 68)
    for p in prompts:
        model.set_classes([p])
        r = model.predict(frame, conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
        confs = sorted((float(b.conf) for b in r.boxes), reverse=True)
        shown = ", ".join(f"{c:.2f}" for c in confs[:6]) if confs else "—"
        print(f"{p:<24} {len(confs):>5}  {shown}")
    print("-" * 68)
    print("Point the camera at BOTH bottles, then read down the column:")
    print("  a prompt scoring high on one object only  -> it discriminates, usable")
    print("  a prompt scoring on everything round      -> no discrimination, fine-tune instead")
    print("=" * 68 + "\n")
    model.set_classes(prompts)  # restore combined mode


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", help="test one photo instead of the webcam")
    ap.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    ap.add_argument("--weights", default="yoloe-11s-seg.pt", help="downloads automatically if absent")
    ap.add_argument("--prompts", default=",".join(DEFAULT_PROMPTS), help="comma-separated")
    ap.add_argument("--conf", type=float, default=0.10, help="same default the pipeline uses")
    ap.add_argument("--imgsz", type=int, default=416, help="416 gives ~9fps on CPU; 640 is more accurate but ~3.5fps")
    ap.add_argument("--device", default=None, help="mps/cuda/cpu; default: best available")
    ap.add_argument("--isolate", action="store_true", help="also print the per-prompt isolated report")
    args = ap.parse_args()

    if args.device is None:
        import torch
        args.device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
    print(f"device={args.device}  imgsz={args.imgsz}  conf={args.conf}")
    print(f"prompts: {prompts}")

    model = YOLOE(args.weights)
    model.set_classes(prompts)

    # ---- single photo ------------------------------------------------------
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise SystemExit(f"could not read {args.image}")
        r = model.predict(frame, conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False)[0]
        print("\nCOMBINED — what the pipeline would actually log:")
        if not len(r.boxes):
            print("  (nothing detected)")
        for b in sorted(r.boxes, key=lambda b: -float(b.conf)):
            print(f"  {r.names[int(b.cls)]:<24} {float(b.conf):.2f}")
        if args.isolate:
            isolate_report(model, frame, prompts, args.conf, args.imgsz, args.device)
        out = args.image.rsplit(".", 1)[0] + "_detected.jpg"
        cv2.imwrite(out, draw(frame.copy(), r, prompts))
        print(f"annotated image written to {out}")
        return

    # ---- live webcam -------------------------------------------------------
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {args.camera}")
    print("\nlive. keys:  i = isolated report on the current frame   q = quit\n")

    fps, last = 0.0, time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = model.predict(frame, conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False)[0]

        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
        last = now

        shown = draw(frame.copy(), r, prompts)
        cv2.putText(shown, f"{fps:4.1f} fps   i=isolate  q=quit", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("YOLOE prompt test", shown)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("i"):
            isolate_report(model, frame, prompts, args.conf, args.imgsz, args.device)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
