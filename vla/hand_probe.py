"""Watch the camera and print what the hand detector sees, so the release threshold comes from data.

    python vla/hand_probe.py                 # the arm's wrist camera
    python vla/hand_probe.py --camera-index 1

Hold your hand where you would to catch the bottle. Each line is one frame: whether a hand was found and how
much of the frame it covers. vla/hand_release.py releases when that fraction stays above HAND_AREA_MIN for
CONSECUTIVE frames, so the number to set is a little below whatever your hand actually reads here.
"""
import argparse
import time

import cv2

from vla.hand_release import HAND_AREA_MIN, hand_area


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera-index", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=20.0)
    args = ap.parse_args()

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
