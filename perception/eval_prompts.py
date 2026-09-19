"""Score a YOLOE prompt set over a folder of images: false-positive rate, recall, and speed.

    # how often do our prompts fire on scenes that contain none of our objects?
    python eval_prompts.py --images data/coco128/images/train2017 --absent

    # recall against YOLO-format ground truth, when you have labels
    python eval_prompts.py --images data/mine/images --labels data/mine/labels --map "pill bottle=0,keys=1"

    # speed only
    python eval_prompts.py --images data/coco128/images/train2017 --speed --limit 20

Why --absent matters more than recall here. The demo failure that actually
embarrasses you is not missing a put-down -- it is logging "you placed your
medication on the table" when the user set down a mug. That is a false positive,
and you can measure it without labelling a single image: point this at any folder
of photos that contains none of your target objects, and every single detection
is by definition wrong. COCO128 works fine for this: no pill bottles, no keys,
no glasses anywhere in it.

Read the output as a per-prompt false-positive rate at each confidence
threshold, then set --conf in the pipeline above the level where the noise dies.
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_images(folder: str, limit: int | None) -> list[Path]:
    paths = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMG_EXT)
    return paths[:limit] if limit else paths


def absent_sweep(model, images, prompts, imgsz, device, thresholds):
    """Every detection here is a false positive, because none of these objects are present.

    Reports, per prompt and per confidence threshold, what fraction of images got
    at least one (wrong) detection.
    """
    # Predict once at the lowest threshold, then filter upward: one pass, many thresholds.
    lowest = min(thresholds)
    fired = defaultdict(lambda: defaultdict(int))  # prompt -> threshold -> images with a hit
    best_conf = defaultdict(float)

    model.set_classes(prompts)
    for i, p in enumerate(images, 1):
        r = model.predict(str(p), conf=lowest, imgsz=imgsz, device=device, verbose=False)[0]
        per_prompt_max = defaultdict(float)
        for b in r.boxes:
            name = r.names[int(b.cls)]
            per_prompt_max[name] = max(per_prompt_max[name], float(b.conf))
        for name, c in per_prompt_max.items():
            best_conf[name] = max(best_conf[name], c)
            for t in thresholds:
                if c >= t:
                    fired[name][t] += 1
        if i % 25 == 0:
            print(f"  ...{i}/{len(images)}", flush=True)

    n = len(images)
    print(f"\nFALSE POSITIVES over {n} images containing none of these objects")
    print("(lower is better; 0% means the prompt never fired on the wrong thing)\n")
    header = "prompt".ljust(22) + "".join(f"@{t:<7.2f}" for t in thresholds) + "  worst conf"
    print(header)
    print("-" * len(header))
    for name in prompts:
        row = name.ljust(22)
        for t in thresholds:
            pct = 100.0 * fired[name][t] / n
            row += f"{pct:5.1f}%  "
        row += f"   {best_conf[name]:.2f}"
        print(row)
    print("-" * len(header))
    print("Pick the pipeline's --conf above the point where your target prompts stop firing here.")


def recall_check(model, images, labels_dir, prompts, mapping, imgsz, device, conf):
    """Recall against YOLO-format labels: of the images that really contain X, how many did we find X in?"""
    labels_dir = Path(labels_dir)
    truth_imgs = defaultdict(set)
    for p in images:
        lf = labels_dir / f"{p.stem}.txt"
        if not lf.exists():
            continue
        for line in lf.read_text().splitlines():
            if line.strip():
                truth_imgs[int(line.split()[0])].add(p.stem)

    model.set_classes(prompts)
    found = defaultdict(set)
    for p in images:
        r = model.predict(str(p), conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
        for b in r.boxes:
            found[r.names[int(b.cls)]].add(p.stem)

    print(f"\nRECALL at conf={conf} (of images that truly contain the object, how many did we detect it in)\n")
    print(f"{'prompt':<22}{'truth':>7}{'found':>7}{'recall':>9}")
    print("-" * 45)
    for prompt, cls_id in mapping.items():
        truth = truth_imgs.get(cls_id, set())
        hit = truth & found.get(prompt, set())
        rec = 100.0 * len(hit) / len(truth) if truth else float("nan")
        print(f"{prompt:<22}{len(truth):>7}{len(hit):>7}{rec:>8.1f}%")
    print("-" * 45)


def speed_check(model, images, prompts, imgsz_list, device):
    model.set_classes(prompts)
    print(f"\nSPEED on {len(images)} images, device={device}, {len(prompts)} prompts\n")
    print(f"{'imgsz':<9}{'ms/frame':>11}{'fps':>8}")
    print("-" * 28)
    for sz in imgsz_list:
        model.predict(str(images[0]), imgsz=sz, device=device, verbose=False)  # warm up
        t = time.perf_counter()
        for p in images:
            model.predict(str(p), imgsz=sz, device=device, verbose=False)
        ms = (time.perf_counter() - t) / len(images) * 1000
        print(f"{sz:<9}{ms:>10.1f}{1000/ms:>8.1f}")
    print("-" * 28)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", help="YOLO-format label dir, enables the recall check")
    ap.add_argument("--map", default="", help='prompt=class_id pairs, e.g. "cell phone=67,bottle=39"')
    ap.add_argument("--prompts", default="pill bottle,water bottle,keys,phone,glasses")
    ap.add_argument("--weights", default="yoloe-11s-seg.pt")
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--absent", action="store_true", help="treat every detection as a false positive")
    ap.add_argument("--speed", action="store_true", help="benchmark across input sizes")
    args = ap.parse_args()

    if args.device is None:
        import torch
        args.device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    from ultralytics import YOLOE

    prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
    images = load_images(args.images, args.limit)
    if not images:
        raise SystemExit(f"no images found in {args.images}")

    print(f"model={args.weights}  device={args.device}  imgsz={args.imgsz}  images={len(images)}")
    print(f"prompts: {prompts}")
    model = YOLOE(args.weights)

    if args.speed:
        speed_check(model, images[: min(len(images), 20)], prompts, [640, 480, 416, 320], args.device)
    if args.absent:
        absent_sweep(model, images, prompts, args.imgsz, args.device, [0.10, 0.20, 0.30, 0.40, 0.50])
    if args.labels and args.map:
        mapping = {}
        for pair in args.map.split(","):
            k, _, v = pair.partition("=")
            mapping[k.strip()] = int(v)
        recall_check(model, images, args.labels, prompts, mapping, args.imgsz, args.device, args.conf)


if __name__ == "__main__":
    main()
