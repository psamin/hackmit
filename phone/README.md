# Phone camera client

Turns any iPhone or Android phone into the camera for the pipeline. The phone
captures and uploads; nothing else. All detection, tracking and VLM work stays
on the laptop.

The phone sends one JPEG per WebSocket binary message — the same wire format
`glasses_rx.py` already accepts — so the pipeline treats a phone exactly like
the glasses relay or `fake_glasses.py`.

## Why a web page and not an iOS app

| | Web page (this) | Native iOS app |
|---|---|---|
| Build tooling | none | Xcode, and Xcode only runs on macOS |
| Getting it on the phone | open a URL | USB cable + Xcode, re-signed every 7 days |
| Apple account | none | free account minimum, Developer Mode toggle |
| Works on Android | yes | no |
| On-device ML | none needed | would need a Core ML export |

For a demo and for an elderly user's own phone, the web page wins on every row.
A native app only becomes worth it if the phone must run detection locally.

## Run it

Three processes: the page server, the pipeline, and the phone.

```bash
# 1. serve the page (also generates the certificate both it and the relay use)
python phone/serve.py

# 2. start the pipeline listening for the phone, using that same certificate
cd perception
python memory_pipeline.py --source wss://0.0.0.0:8765 \
    --cert ../phone/cert.pem --key ../phone/key.pem \
    --targets "pill bottle,keys,phone" --out runs/phone

# 3. on the phone: open the https:// URL serve.py printed, tap Start
```

## First connection, once per laptop

Browsers only hand out the camera on HTTPS (or localhost). The phone reaches
the laptop by LAN IP, so the page must be HTTPS, and a page on HTTPS may not
open an insecure `ws://` socket — hence `wss://` for the relay too. Nobody
issues real certificates for private IPs, so `serve.py` makes a self-signed one
and the phone has to be told once to trust it:

1. Open `https://<laptop-ip>:8443/` on the phone → Safari warns → Show Details →
   visit this website.
2. Open `https://<laptop-ip>:8765` once as well and accept the same warning.
   The page will look broken or empty; that is expected. You are only there to
   accept the certificate so the WebSocket to that port is allowed.
3. Go back to the page and tap **Start**.

Both devices must be on the same Wi-Fi. Many university and conference networks
block device-to-device traffic — if the phone cannot reach the laptop, use a
personal hotspot from one of the phones and put both devices on it.

## What the page does about the things that usually break

- **Backpressure.** If the socket already has more than 256 KB queued, the next
  frame is dropped instead of queued. Without this, a weak link quietly builds a
  send backlog and the "live" stream drifts further and further behind real
  time. The dropped count is shown on screen.
- **Screen sleep.** A sleeping screen suspends capture. The page takes a Wake
  Lock while streaming.
- **Backgrounding.** iOS pauses the camera when the tab is not visible; the page
  says so rather than appearing to stream nothing.
- **Frame rate and size.** 10 fps at 720p, JPEG quality 0.7 — matching
  `fake_glasses.py`, so a phone run and a replayed-video run feed the pipeline
  comparable frames. Sending faster would only waste upload: the pipeline
  processes about 10 fps.

## Tuning

Constants at the top of the `<script>` block in `index.html`:

| Constant | Default | Raise it / lower it |
|---|---|---|
| `TARGET_FPS` | 10 | Match the pipeline's `--fps`. Higher costs upload for nothing. |
| `TARGET_HEIGHT` | 720 | The pipeline resizes to 640px wide anyway; lower this first on a slow link. |
| `JPEG_QUALITY` | 0.7 | Below ~0.5 small objects start to smear and detection suffers. |
| `MAX_BUFFERED_BYTES` | 256 KB | Lower = fresher frames, more drops. Higher = smoother, laggier. |
