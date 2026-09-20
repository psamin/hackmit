"""Replay a recorded episode's camera frames into a mock rollout, so the policy sees a real pick instead of grey.

    PYTHONPATH=. python vla/run_policy.py --mock --viz --replay-episode 3 --server <url>

vla/sim_test.py's SyntheticCamera publishes a flat grey frame, which is fine for proving the plumbing but tells the
policy nothing: ACT is a visuomotor policy, so on grey input it has no idea where the bottle is and the arm wanders.
This module publishes the frames of one dataset episode at its recorded rate, which is what the policy was trained
on, so the mock arm performs the pick it learned and the Viser view shows it.

The frames are a recording, not a render of the mock arm, so this is open loop: the view does not react to where the
arm actually goes. It answers "does the policy drive a sane pick from real input", not "would it recover from a
disturbance" - only the real arm answers that.
"""
from __future__ import annotations

import glob
from threading import Event, Thread

import numpy as np

from dimos.core.module import Module, Out, rpc
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat

DATASET = "openyam_dataset"


def episode_frames(dataset: str, episode: int) -> tuple[np.ndarray, float]:
    """Every frame of one episode, as BGR, plus the dataset's frame rate."""
    import json

    import cv2
    import pandas as pd

    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{dataset}/data/**/*.parquet", recursive=True))])
    fps = float(json.load(open(f"{dataset}/meta/info.json"))["fps"])
    sub = df[df.episode_index == episode]
    if sub.empty:
        raise SystemExit(f"episode {episode} not in {dataset}; have {sorted(df.episode_index.unique())}")
    cap = cv2.VideoCapture(f"{dataset}/videos/observation.images.wrist/chunk-000/file-000.mp4")
    frames = []
    for idx in sub["index"].astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    if not frames:
        raise SystemExit(f"no frames decoded for episode {episode}")
    return np.stack(frames), fps


class ReplayCamera(Module):
    """Publishes one recorded episode's frames on a loop, at the rate they were recorded."""

    color_image: Out[Image]

    def __init__(self, *args, dataset: str = DATASET, episode: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self._dataset, self._episode = dataset, episode

    @rpc
    def start(self) -> None:
        super().start()
        self._frames, self._fps = episode_frames(self._dataset, self._episode)
        self._done = Event()
        Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        i = 0
        while not self._done.is_set():
            frame = self._frames[i % len(self._frames)]
            self.color_image.publish(Image.from_numpy(frame[:, :, ::-1].copy(), format=ImageFormat.RGB))
            i += 1
            self._done.wait(1 / self._fps)

    @rpc
    def stop(self) -> None:
        self._done.set()
        super().stop()
