"""Laptop end of the glasses relay: a WebSocket server that takes one JPEG per binary message.

    python memory_pipeline.py --source ws://0.0.0.0:8765 --out runs/glasses   # iPhone app connects to <laptop-ip>:8765
    python fake_glasses.py data/epic/P02_102.MP4                               # stands in for the iPhone app
"""
import asyncio, threading
from urllib.parse import urlparse

import cv2
import numpy as np
from websockets.asyncio.server import serve


class GlassesStream:
    """Covers the cv2.VideoCapture calls memory_pipeline makes. Always hands out the newest frame, so a slow
    pipeline drops frames instead of falling behind. Waits through phone disconnects; Ctrl-C ends the run."""

    def __init__(self, url, fps):
        u = urlparse(url)
        self.host, self.port, self.fps = u.hostname or "0.0.0.0", u.port or 8765, fps
        self.received = self.taken = 0
        self._jpeg = None
        self._cond = threading.Condition()
        threading.Thread(target=lambda: asyncio.run(self._serve()), daemon=True).start()

    async def _serve(self):
        async def handler(ws):
            print(f"glasses connected: {ws.remote_address}", flush=True)
            async for msg in ws:
                if isinstance(msg, bytes):
                    with self._cond:
                        self._jpeg, self.received = msg, self.received + 1
                        self._cond.notify_all()
            print("glasses disconnected; waiting for reconnect", flush=True)

        async with serve(handler, self.host, self.port, max_size=2**23):
            print(f"waiting for glasses on ws://{self.host}:{self.port}", flush=True)
            await asyncio.Future()

    def grab(self):
        with self._cond:
            self._cond.wait_for(lambda: self.received > self.taken)
            self.taken = self.received
            return True

    def retrieve(self):
        frame = cv2.imdecode(np.frombuffer(self._jpeg, np.uint8), cv2.IMREAD_COLOR)
        return frame is not None, frame

    def get(self, prop):
        return self.fps if prop == cv2.CAP_PROP_FPS else 0.0
