"""Hold the bottle until a hand is under the gripper, nagging on a timer until one shows up.

    from vla.hand_release import wait_for_hand
    wait_for_hand(camera_index=0, timeout_s=30, nag_s=3.0, on_nag=lambda n: speak(PROMPTS[n]))

MediaPipe Hands, not a policy: it is pretrained, runs on the CPU in about 20 ms a frame, and needs no demos of its
own. Version matters - mediapipe 1.x routes the hand landmarker through Metal, which aborts on this machine
(`DrishtiMetalHelper ... Service is unavailable`), so pin 0.10.x, whose graph stays on the CPU.

The wrist camera looks out at the person from the hand-over pose rather than down at the gap under the gripper, so
"a hand is in view and close" is the honest signal, not "a hand is underneath". A hand filling a good part of the
frame means an open palm right in front of the gripper, which is what we want anyway.

`timeout_s` always wins: a detector that never fires must not leave the arm holding a bottle in front of an
audience, so release happens on the timeout regardless.
"""
from __future__ import annotations

import time
from typing import Callable

# A hand this much of the frame is close enough to be catching, rather than someone gesturing across the room.
HAND_AREA_MIN = 0.045
CONSECUTIVE = 3  # frames in a row, so one bad detection cannot open the gripper


def hand_area(landmarks) -> float:
    """Fraction of the frame the hand's bounding box covers, from its normalised landmarks."""
    xs = [p.x for p in landmarks.landmark]
    ys = [p.y for p in landmarks.landmark]
    return max(0.0, (max(xs) - min(xs))) * max(0.0, (max(ys) - min(ys)))


def wait_for_hand(camera_index: int = 0, timeout_s: float = 30.0, nag_s: float = 3.0,
                  on_nag: Callable[[int], None] | None = None,
                  area_min: float = HAND_AREA_MIN) -> bool:
    """Block until a hand is close in front of the camera, or `timeout_s` passes. True if a hand was seen.

    Calls `on_nag(n)` every `nag_s` seconds while waiting, n counting from 0, so the caller can vary what it says.
    """
    import cv2
    from mediapipe.python.solutions import hands as mp_hands

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("hand detector: cannot open the camera, falling back to the timer", flush=True)
        time.sleep(timeout_s)
        return False

    detector = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                              min_detection_confidence=0.5, min_tracking_confidence=0.5)
    deadline, next_nag, nagged, streak = time.time() + timeout_s, time.time(), 0, 0
    try:
        while time.time() < deadline:
            if time.time() >= next_nag:
                if on_nag:
                    on_nag(nagged)
                nagged += 1
                next_nag = time.time() + nag_s
            ok, frame = cap.read()
            if not ok:
                continue
            result = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            found = result.multi_hand_landmarks or []
            biggest = max((hand_area(h) for h in found), default=0.0)
            streak = streak + 1 if biggest >= area_min else 0
            if streak >= CONSECUTIVE:
                print(f"hand detected, covering {biggest * 100:.0f}% of the frame - releasing", flush=True)
                return True
        print(f"no hand after {timeout_s:.0f}s - releasing anyway", flush=True)
        return False
    finally:
        cap.release()
        detector.close()
