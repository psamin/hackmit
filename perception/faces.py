"""Face identity for the memory pipeline: embed a face, match it against people the user
has explicitly named, and refuse to guess when it is not sure.

    python faces.py enroll Jacob          # webcam: capture, quality-check, save
    python faces.py who                   # webcam: who is in front of the camera?
    python faces.py list                  # who is enrolled
    python faces.py forget Jacob          # delete a person
    python faces.py calibrate             # measure same-person vs different-person scores

--------------------------------------------------------------------------------
WHY IT IS SHAPED LIKE THIS
--------------------------------------------------------------------------------
Nothing is stored until a human says a name. There is no passive enrolment: a face
that walks through frame is embedded in memory, compared, and dropped. That is the
whole privacy position, and it is a property of this module, not a policy document --
`enroll()` is the only function that writes.

The embedding is NOT YOLOE's. YOLOE's space is semantic: it encodes "a face", so two
different people land in nearly the same place. Identity needs a model trained with
metric learning on identities, which is what ArcFace (via InsightFace `buffalo_l`) is.
YOLOE's job upstream is only to notice a face is present cheaply; this module decides
who. Same split as detector-proposes / VLM-decides for objects.

Matching is exhaustive. With a household's worth of people a dot product against an
Nx512 array is microseconds, and an ANN index would be pure ceremony -- the same call
CLAUDE.md already makes about memories and the prompt.

--------------------------------------------------------------------------------
THE TRAP, AND THE THRESHOLD
--------------------------------------------------------------------------------
Nearest-neighbour ALWAYS returns something. Without a floor, the first stranger who
walks past becomes whoever they resemble most, stated confidently to someone who cannot
independently check. That is worse than the pill-bottle mislabel, because there is no
second opinion behind it: unlike objects, identity cannot be handed to the VLM to
adjudicate.

So `identify()` returns None below MATCH_THR, and the margin to the runner-up must also
clear MARGIN -- a face that is nearly equidistant from two enrolled people is not a
confident match either. Run `calibrate` on your own people and move MATCH_THR into the
gap between the two distributions; do not inherit a number from a paper.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

GALLERY = Path(__file__).with_name("faces.json")   # gitignored: biometric data
MODEL = "buffalo_l"                                # SCRFD detect+landmark, ArcFace 512-d

# Quality gates. A bad embedding does not fail loudly, it lands at mid-distance from
# everything -- exactly where false matches live -- so junk is rejected before it is
# ever compared, and much harder at enrolment than at recognition.
MIN_FACE_PX = 80          # below this the embedding is unreliable at any threshold
MIN_DET_SCORE = 0.60
MIN_BLUR = 40.0           # variance of Laplacian on the crop
# ArcFace's own input is 112x112, so a detection at or above ~112px warps into the
# canonical crop with no upscaling -- that is the natural floor, not a number picked by
# feel. 120 was above native for no reason and rejected clean 112px faces from a laptop
# webcam at normal seated distance. Below 100 the crop is being invented, and a poor
# enrolment silently degrades every later match against that person.
ENROL_MIN_FACE_PX = 100
ENROL_MIN_BLUR = 80.0

MATCH_THR = 0.38          # cosine similarity; calibrate before trusting it
MARGIN = 0.06             # must beat the runner-up by this much

_app = None


def app():
    """InsightFace, loaded once. CPU is the target: this runs a few times a minute,
    not per frame, so there is nothing here worth a GPU (or worth shipping faces to one)."""
    global _app
    if _app is None:
        from insightface.app import FaceAnalysis
        _app = FaceAnalysis(name=MODEL, providers=["CPUExecutionProvider"])
        _app.prepare(ctx_id=-1, det_size=(640, 640))
    return _app


def blur_of(crop) -> float:
    return float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def faces_in(frame, enrolling: bool = False):
    """Every usable face in the frame, best first. Returns dicts with the unit-norm
    embedding, the box, and why anything was rejected (so failures are explainable)."""
    out = []
    for f in app().get(frame):
        x1, y1, x2, y2 = (int(v) for v in f.bbox)
        w, h = x2 - x1, y2 - y1
        crop = frame[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            continue
        blur = blur_of(crop)
        min_px = ENROL_MIN_FACE_PX if enrolling else MIN_FACE_PX
        min_blur = ENROL_MIN_BLUR if enrolling else MIN_BLUR
        reason = None
        if min(w, h) < min_px:
            reason = f"too small ({min(w, h)}px < {min_px})"
        elif float(f.det_score) < MIN_DET_SCORE:
            reason = f"weak detection ({float(f.det_score):.2f})"
        elif blur < min_blur:
            reason = f"too blurry ({blur:.0f} < {min_blur:.0f})"
        v = np.asarray(f.normed_embedding, dtype=np.float32)
        out.append({"vec": v, "box": (x1, y1, x2, y2), "px": min(w, h),
                    "det": float(f.det_score), "blur": blur, "reject": reason})
    out.sort(key=lambda d: -d["px"])           # biggest face = closest = the one meant
    return out


def load() -> dict:
    if not GALLERY.exists():
        return {}
    raw = json.loads(GALLERY.read_text(encoding="utf-8"))
    return {name: [np.asarray(v, np.float32) for v in vecs] for name, vecs in raw.items()}


def save(gallery: dict) -> None:
    GALLERY.write_text(json.dumps({n: [v.tolist() for v in vs] for n, vs in gallery.items()}),
                       encoding="utf-8")


def identify(vec, gallery: dict):
    """(name, score, runner_up_score) -- name is None when it is not confident.

    Max over each person's embeddings, not the mean: several enrolments of one person
    cover different poses and lighting, and averaging them blurs exactly the variation
    they were added to capture.
    """
    if not gallery:
        return None, 0.0, 0.0
    scored = sorted(((max(float(vec @ v) for v in vs), n) for n, vs in gallery.items()), reverse=True)
    best, name = scored[0]
    second = scored[1][0] if len(scored) > 1 else 0.0
    if best < MATCH_THR or (best - second) < MARGIN:
        return None, best, second
    return name, best, second


def enroll(name: str, frame, gallery: dict):
    """Add one embedding under `name`. Returns (ok, message)."""
    found = faces_in(frame, enrolling=True)
    if not found:
        return False, "no face found"
    f = found[0]
    if f["reject"]:
        return False, f"face not good enough to enrol: {f['reject']}"
    gallery.setdefault(name, []).append(f["vec"])
    save(gallery)
    return True, (f"enrolled {name} ({len(gallery[name])} embedding(s), "
                  f"{f['px']}px, blur {f['blur']:.0f})")


# ---------------------------------------------------------------- CLI ----
def grab(seconds=4):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit("camera busy - stop the pipeline first")
    for n in range(seconds, 0, -1):
        print(f"  {n}...", flush=True)
        t = time.perf_counter()
        while time.perf_counter() - t < 1.0:
            cap.read()
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("could not read a frame")
    return frame


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["enroll", "who", "list", "forget", "calibrate"])
    ap.add_argument("name", nargs="?")
    ap.add_argument("--image", help="use a photo instead of the webcam")
    args = ap.parse_args()
    gallery = load()

    if args.action == "list":
        if not gallery:
            print("nobody enrolled yet")
        for n, vs in gallery.items():
            print(f"  {n}: {len(vs)} embedding(s)")
        return

    if args.action == "forget":
        if gallery.pop(args.name, None) is None:
            print(f"{args.name} was not enrolled")
        else:
            save(gallery)
            print(f"forgot {args.name}")
        return

    frame = cv2.imread(args.image) if args.image else grab()

    if args.action == "enroll":
        if not args.name:
            raise SystemExit("enroll needs a name")
        t = time.perf_counter()
        ok, msg = enroll(args.name, frame, gallery)
        print(f"{msg}  ({time.perf_counter() - t:.2f}s)")
        return

    if args.action == "who":
        t = time.perf_counter()
        found = faces_in(frame)
        print(f"{len(found)} face(s) in frame  ({time.perf_counter() - t:.2f}s)")
        for f in found:
            if f["reject"]:
                print(f"  [skipped] {f['px']}px  {f['reject']}")
                continue
            name, best, second = identify(f["vec"], gallery)
            verdict = f"{name} (similarity {best:.3f}, runner-up {second:.3f})" if name else \
                      f"NOT RECOGNISED (best {best:.3f} < {MATCH_THR} or margin < {MARGIN})"
            print(f"  {f['px']}px det {f['det']:.2f} blur {f['blur']:.0f} -> {verdict}")
        return

    if args.action == "calibrate":
        names = list(gallery)
        if len(names) < 1:
            raise SystemExit("enrol someone first")
        print("same-person similarities (want HIGH):")
        for n, vs in gallery.items():
            for i in range(len(vs)):
                for j in range(i + 1, len(vs)):
                    print(f"  {n} vs {n}: {float(vs[i] @ vs[j]):.3f}")
        print("different-person similarities (want LOW):")
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                print(f"  {a} vs {b}: {max(float(x @ y) for x in gallery[a] for y in gallery[b]):.3f}")
        print(f"\nput MATCH_THR (now {MATCH_THR}) in the gap between those two groups.")


if __name__ == "__main__":
    main()
