"""Watch the camera and print what the hand detector sees, so the release threshold comes from data.

    python vla/hand_probe.py                 # the arm's wrist camera
    python vla/hand_probe.py --camera-index 1
    python vla/hand_probe.py --url http://127.0.0.1:8000/api/hand-visible   # the phone, through the server

Hold your hand where you would to catch the bottle. Each line is one frame: whether a hand was found and how
much of the frame it covers. vla/hand_release.py releases when that fraction stays above HAND_AREA_MIN for
CONSECUTIVE frames, so the number to set is a little below whatever your hand actually reads here.
"""
import argparse
import time

import cv2

from vla.hand_release import HAND_AREA_MIN, hand_area


def probe_remote(url: str, seconds: float) -> None:
    """Read areas off the phone, through the server, which is the camera the demo actually releases on."""
    import json
    import urllib.request

    sep = "&" if "?" in url else "?"
    url = f"{url}{sep}min_area=0"  # raw measurements: this is what sets the threshold, not what passes one
    print(f"threshold is {HAND_AREA_MIN:.3f}. Hold your hand where you would catch the bottle.")
    polls, seen, biggest = 0, 0, 0.0
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                answer = json.loads(response.read())
        except Exception as exc:
            print(f"  poll failed: {exc}", flush=True)
            time.sleep(0.5)
            continue
        polls += 1
        if answer.get("reason"):
            print(f"  {answer['reason']}", flush=True)
            time.sleep(0.5)
            continue
        area = float(answer.get("area") or 0.0)
        biggest = max(biggest, area)
        if answer.get("seen"):
            seen += 1
            bar = "#" * int(area * 200)
            print(f"  hand: {area:.4f} {'PASSES' if area >= HAND_AREA_MIN else 'below  '} {bar}", flush=True)
        elif polls % 10 == 0:
            print("  no hand in frame", flush=True)
        time.sleep(0.2)
    print(f"\n{polls} polls, a hand in {seen} of them ({100 * seen / max(polls, 1):.0f}%)")
    print(f"largest hand seen: {biggest:.4f}   threshold: {HAND_AREA_MIN:.3f}")
    if biggest and biggest < HAND_AREA_MIN:
        print(f"-> too high for this camera: set HAND_AREA_MIN = {biggest * 0.6:.4f} in vla/hand_release.py")
    elif biggest >= HAND_AREA_MIN:
        print("-> the threshold works for this setup; a hand held out to catch passes it")
    elif not seen:
        print("-> no hand was ever seen: is the phone page open and pointed at the catch?")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera-index", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--url", default=None,
                    help="poll the server's /api/hand-visible instead of opening a camera here, so the numbers "
                         "come from the phone the demo actually releases on")
    args = ap.parse_args()

    if args.url:
        return probe_remote(args.url, args.seconds)

    from mediapipe.python.solutions import hands as mp_hands

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera {args.camera_index}")
    detector = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                              min_detection_confidence=0.4, min_tracking_confidence=0.4)
    print(f"threshold is {HAND_AREA_MIN:.3f}. Hold your hand where you would catch the bottle.")
    seen, frames, biggest_ever = 0, 0, 0.0
    deadline = time.time() + args.seconds
    try:
        while time.time() < deadline:
            ok, frame = cap.read()
            if not ok:
                continue
            frames += 1
            found = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).multi_hand_landmarks or []
            area = max((hand_area(h) for h in found), default=0.0)
            biggest_ever = max(biggest_ever, area)
            if found:
                seen += 1
                if frames % 3 == 0:
                    bar = "#" * int(area * 200)
                    print(f"  hand: {area:.3f} {'PASSES' if area >= HAND_AREA_MIN else 'below  '} {bar}", flush=True)
            elif frames % 30 == 0:
                print("  no hand in frame", flush=True)
    finally:
        cap.release()
        detector.close()
    print(f"\n{frames} frames, a hand in {seen} of them ({100 * seen / max(frames, 1):.0f}%)")
    print(f"largest hand seen: {biggest_ever:.3f}   threshold: {HAND_AREA_MIN:.3f}")
    if biggest_ever and biggest_ever < HAND_AREA_MIN:
        print(f"-> the threshold is too high for this camera angle; try HAND_AREA_MIN = {biggest_ever * 0.6:.3f}")
    elif not seen:
        print("-> no hand was ever detected: the camera is probably not pointing where your hand goes")


if __name__ == "__main__":
    main()
