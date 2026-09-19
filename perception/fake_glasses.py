"""Stands in for the iPhone glasses app: streams a video file to the laptop as JPEGs over WebSocket,
at the glasses' 720p and the relay's 10 fps.

    python fake_glasses.py data/epic/P02_102.MP4 --start 100 --end 160
    python fake_glasses.py data/clips/place_desk.mov --url ws://192.168.1.20:8765
"""
import argparse, time

import cv2
from websockets.sync.client import connect

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--url", default="ws://127.0.0.1:8765")
ap.add_argument("--fps", type=float, default=10.0)
ap.add_argument("--height", type=int, default=720)
ap.add_argument("--quality", type=int, default=70, help="JPEG quality")
ap.add_argument("--start", type=float, default=0.0)
ap.add_argument("--end", type=float, default=None)
args = ap.parse_args()

cap = cv2.VideoCapture(args.video)
src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
step = max(1, round(src_fps / args.fps))
cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000)
idx, sent, total_bytes = int(args.start * src_fps), 0, 0

with connect(args.url, max_size=None) as ws:
    t0 = time.perf_counter()
    while cap.grab():
        idx += 1
        if idx % step:
            continue
        if args.end is not None and idx / src_fps > args.end:
            break
        _, frame = cap.retrieve()
        frame = cv2.resize(frame, (round(frame.shape[1] * args.height / frame.shape[0]), args.height))
        jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, args.quality])[1].tobytes()
        ws.send(jpeg)
        sent, total_bytes = sent + 1, total_bytes + len(jpeg)
        time.sleep(max(0.0, t0 + sent / args.fps - time.perf_counter()))  # real time, like the glasses

wall = time.perf_counter() - t0
print(f"sent {sent} frames in {wall:.1f}s ({sent / wall:.1f} fps), avg {total_bytes / max(sent, 1) / 1024:.0f} KB/frame")
