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
# The wrist camera looks outward at the hand-over pose, so a hand held out to catch is not large in frame - this
# is deliberately low. Run vla/hand_probe.py to see what your setup actually reads and set it from that.
HAND_AREA_MIN = 0.012
CONSECUTIVE = 2  # frames in a row, so one bad detection cannot open the gripper
POLL_S = 0.12    # how often the remote detector is asked; two polls is the fastest it can release


def hand_area(landmarks) -> float:
    """Fraction of the frame the hand's bounding box covers, from its normalised landmarks."""
    xs = [p.x for p in landmarks.landmark]
    ys = [p.y for p in landmarks.landmark]
    return max(0.0, (max(xs) - min(xs))) * max(0.0, (max(ys) - min(ys)))


def wait_for_hand_remote(poll_url: str, timeout_s: float = 30.0, nag_s: float = 7.0,
                         on_nag: Callable[[int], None] | None = None,
                         area_min: float = HAND_AREA_MIN) -> bool:
    """Ask another service whether a hand is in front of its camera, rather than opening one here.

    The phone is the camera pointed at the catch - the arm's looks outward from the hand-over pose. The server
    already keeps the newest phone frame for the face tools, so this just polls its answer.

    `area_min` goes with the request: the server detects hands but does not decide which ones count, and
    without it any hand anywhere in frame - the one holding the phone included - opens the gripper the moment
    the hand-over starts.
    """
    import json
    import urllib.request

    sep = "&" if "?" in poll_url else "?"
    poll_url = f"{poll_url}{sep}min_area={area_min}"
    deadline, next_nag, nagged, streak, polls, seen_any, biggest = (
        time.time() + timeout_s, time.time() + nag_s, 0, 0, 0, False, 0.0)
    while time.time() < deadline:
        if time.time() >= next_nag:
            if on_nag:
                on_nag(nagged)
            nagged += 1
            next_nag = time.time() + nag_s
        try:
            with urllib.request.urlopen(poll_url, timeout=3) as response:
                answer = json.loads(response.read())
            polls += 1
            seen_any = seen_any or answer.get("frame_age_s") is not None
            biggest = max(biggest, float(answer.get("area") or 0.0))
            streak = streak + 1 if answer.get("hand") else 0
            if streak >= CONSECUTIVE:
                print(f"hand seen by the phone camera (area {answer.get('area')}) - releasing", flush=True)
                return True
        except Exception as exc:
            print(f"hand poll failed ({exc})", flush=True)
        time.sleep(POLL_S)
    why = "no hand seen" if seen_any else "the phone camera sent nothing (is its page open?)"
    near = f", biggest hand {biggest:.4f} vs threshold {area_min}" if biggest else ""
    print(f"{why} in {timeout_s:.0f}s ({polls} polls{near}) - releasing anyway", flush=True)
    return False


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

    # Two hands and a lower confidence: someone reaching in from the side is a partial, low-confidence hand,
    # and missing it leaves the arm gripping while the person waits with their palm out.
    detector = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                              min_detection_confidence=0.4, min_tracking_confidence=0.4)
    # Wait a full interval before the first prompt: the caller has just announced the hand-over, and
    # nagging immediately after it lands as talking over yourself.
    deadline, next_nag, nagged, streak, frames = (time.time() + timeout_s, time.time() + nag_s, 0, 0, 0)
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
            frames += 1
            result = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            found = result.multi_hand_landmarks or []
            biggest = max((hand_area(h) for h in found), default=0.0)
            streak = streak + 1 if biggest >= area_min else 0
            if streak >= CONSECUTIVE:
                print(f"hand detected, covering {biggest * 100:.0f}% of the frame - releasing", flush=True)
                return True
        # Say how many frames arrived: "no hand" and "the camera gave us nothing" look identical otherwise,
        # and dimOS holds the same camera during a rollout.
        reason = "no hand seen" if frames else "the camera returned no frames"
        print(f"{reason} in {timeout_s:.0f}s ({frames} frames read) - releasing anyway", flush=True)
        return False
    finally:
        cap.release()
        detector.close()
