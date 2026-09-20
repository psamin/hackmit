# HackMIT: where's my medicine?

Head camera sees the pill bottle put down → trigger → Claude describes where it landed → memory log → ask by voice.
A Damiao arm fetches it. Plan and test results: the team's "HackMIT Memory + Retrieval Build Plan" doc.

## Setup (macOS, Apple Silicon)

```bash
cd perception
uv venv --python 3.12 && uv pip install -r requirements.txt
echo "ANTHROPIC_API_KEY=..." > .env          # gitignored; needed for the VLM and voice answers
```

Model weights and videos are gitignored: copy `perception/weights/`, `perception/mobileclip2_b.ts` and
`perception/data/` from Praneeth's laptop (AirDrop). EPIC-KITCHENS clips come from data.bris.ac.uk.
The terminal app needs camera and microphone permission (System Settings → Privacy & Security).

## Run

| What | Command (from `perception/`) |
|---|---|
| Pipeline on a video | `.venv/bin/python memory_pipeline.py --source data/clips/place_desk.mov --out runs/desk` |
| Pipeline on a camera | `.venv/bin/python memory_pipeline.py --source 0 --out runs/live` |
| Pipeline on the glasses relay | `.venv/bin/python memory_pipeline.py --source ws://0.0.0.0:8765 --out runs/glasses` |
| Fake glasses (no phone needed) | `.venv/bin/python fake_glasses.py data/epic/P02_102.MP4 --start 95 --end 135` |
| Ask by voice | `.venv/bin/python voice.py runs/live/memory.jsonl` (`--text "..."` to type) |
| Score the trigger on EPIC | `.venv/bin/python blockers/eval_epic.py runs/p02 P02_102` |

Add `--no-vlm` to the pipeline to skip Claude calls. Ctrl-C ends a live run and prints stats.

The iPhone app sends one JPEG per WebSocket binary message to `ws://<laptop-ip>:8765`, 720p at 10 fps.

## Pam — voice agent (server/ + phone/agent.html)

```bash
cp server/.env.example server/.env        # DEEPGRAM_API_KEY is the only required key
perception/.venv/Scripts/python server/app.py
# Phone:  https://<laptop-ip>:8443/       — Pam's voice page
#         https://<laptop-ip>:8443/camera — the camera client
# Laptop: http://127.0.0.1:8000/?demo=1   — test panel: every function without voice
```

Deepgram handles listen/think/speak over one WebSocket; the page holds a token from
`/api/dg-token`, and each agentic action is a client-side function call back to
`server/app.py`. Texting and calling are not supported. Reminders fire through
`/api/push` (SSE) into the live session via `InjectUserMessage`. Memory lookup can
fall back to memory.jsonl. Calendar and flight services explicitly report when
they are not connected; flight results are spoken rather than handed off to a
search link. Ride booking still requires completion in Uber, and robot actions
require a configured policy and explicit user approval. See `server/RUNBOOK.md`
for Google Calendar sign-in, flight-provider setup, and safe verification.

## Arm (5× Damiao DM-J4340P-2EC, CAN at 1 Mbit/s)

Needs a gs_usb/candleLight USB-CAN adapter (USB ID `1d50:606f`) and 24 V power. The LED is solid red when powered
and disabled (normal), solid green when enabled, and blinking red on a fault.

```bash
pip install -r arm/requirements.txt        # or use a dimOS checkout's .venv
python arm/arm_probe.py                     # read-only: finds the adapter and motor IDs, never enables
python arm/teach.py record poses.json       # pose the limp arm by hand, name each pose
python arm/teach.py play poses.json         # replays at <= 0.3 rad/s; no brakes, so support the arm before release
```
