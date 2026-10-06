"""Laptop end of the camera relay: a WebSocket server that takes one JPEG per binary message.

    python memory_pipeline.py --source ws://0.0.0.0:8765 --out runs/glasses    # glasses / fake_glasses.py
    python memory_pipeline.py --source wss://0.0.0.0:8765 \\
        --cert ../phone/cert.pem --key ../phone/key.pem --out runs/phone       # browser on a phone
    python fake_glasses.py data/epic/P02_102.MP4                               # replays a video as a sender

Plain ws:// is fine for senders you control, like fake_glasses.py or a native
app. A browser sender needs wss://: the page itself must be served over HTTPS to
get camera access at all (see phone/README.md), and an HTTPS page is not allowed
to open an insecure ws:// socket. Pass the same cert/key pair phone/serve.py
generated, so the phone only has to trust one certificate.
"""
import asyncio, ssl, threading
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
from websockets.asyncio.server import serve

try:
    from .capture import decode_frame
except ImportError:
    from capture import decode_frame


class GlassesStream:
    """Covers the cv2.VideoCapture calls memory_pipeline makes. Always hands out the newest frame, so a slow
    pipeline drops frames instead of falling behind. Waits through phone disconnects; Ctrl-C ends the run."""

    def __init__(self, url, fps, cert=None, key=None):
        u = urlparse(url)
        self.host, self.port, self.fps = u.hostname or "0.0.0.0", u.port or 8765, fps
        self.secure = u.scheme == "wss"
        if self.secure and not cert:
            raise SystemExit("wss:// needs --cert and --key (run phone/serve.py once to generate them)")
        self._ssl = None
        if self.secure:
            for p in (cert, key):
                if not Path(p).exists():
                    raise SystemExit(f"certificate file not found: {p}")
            self._ssl = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            self._ssl.load_cert_chain(cert, key)
        self.received = self.taken = 0
        self._jpeg = self._selected_jpeg = None
        self._metadata, self.metadata = {}, {}
        self._active = None
        self._cond = threading.Condition()
        threading.Thread(target=lambda: asyncio.run(self._serve()), daemon=True).start()

    async def _serve(self):
        async def handler(ws):
            if self._active is not None:
                await ws.close(code=1008, reason="Only one camera may stream")
                return
            self._active = ws
            last_sequence, session_id = -1, None
            print(f"camera connected: {ws.remote_address}", flush=True)
            try:
                async for msg in ws:
                    if isinstance(msg, bytes):
                        try:
                            jpeg, metadata = decode_frame(msg)
                        except ValueError:
                            await ws.close(code=1008, reason="Invalid or stale camera frame")
                            break
                        if metadata.get("session_id"):
                            if session_id is not None and metadata["session_id"] != session_id:
                                await ws.close(code=1008, reason="Camera session changed")
                                break
                            session_id = metadata["session_id"]
                            if metadata["frame_id"] <= last_sequence:
                                continue
                            last_sequence = metadata["frame_id"]
                        with self._cond:
                            self._jpeg, self._metadata, self.received = jpeg, metadata, self.received + 1
                            self._cond.notify_all()
            finally:
                self._active = None
                print("camera disconnected; waiting for reconnect", flush=True)

        scheme = "wss" if self.secure else "ws"
        async with serve(handler, self.host, self.port, max_size=2**20, ssl=self._ssl):
            print(f"waiting for a camera on {scheme}://{self.host}:{self.port}", flush=True)
            await asyncio.Future()

    def grab(self):
        with self._cond:
            self._cond.wait_for(lambda: self.received > self.taken)
            self.taken = self.received
            self._selected_jpeg, self.metadata = self._jpeg, dict(self._metadata)
            return True

    def retrieve(self):
        frame = cv2.imdecode(np.frombuffer(self._selected_jpeg, np.uint8), cv2.IMREAD_COLOR)
        return frame is not None, frame

    def get(self, prop):
        return self.fps if prop == cv2.CAP_PROP_FPS else 0.0
