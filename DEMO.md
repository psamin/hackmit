# Compass — demo runbook

> "I forgot where I put my pills."
> → Pam: *"Your pill bottle is standing on the white paper covering the table. I can get it for you."*
> → the arm picks it up, turns round, says *"Here are your pills."*, drops them into your hand, and resets.

Everything below is what exists and runs today, on this laptop, with measurements where
they were taken. Where something is unverified it says so.

---

## 1. Start it

```bash
cd ~/Documents/GitHub/hackmit && ./demo.sh
```

Brings up four services. Safe to re-run — anything already up is left alone.

| Port | Service | What it is |
|---|---|---|
| 8000 | Pam (HTTP) | voice agent API, laptop testing |
| 8443 | Pam (HTTPS) | the phone's page — iOS needs HTTPS for the mic |
| 8011 | policy server | ACT checkpoint, serves action chunks |
| 8765 | camera relay | where the phone's frames land; **the perception pipeline is this server** |

Then the arm, **in its own terminal**, with a hand near the 24 V switch:

```bash
source ~/Documents/GitHub/dimos/.venv/bin/activate && cd ~/Documents/GitHub/hackmit
PYTHONPATH=. python vla/run_policy.py --real --camera-index 0 \
  --home vla/spots/left_01.json --handover vla/spots/handover.json \
  --present-s 0.5 --catch-s 1 \
  --voice-url http://127.0.0.1:8000/api/push \
  --announce "Here are your pills." \
  --return-after-s 3 --return-speed 0.15 --server http://127.0.0.1:8011
```

Type `go` (energizes the motors), then `p` (preflight). Leave it there.

Phone: `demo.sh` prints the URL — `https://<laptop-ip>:8443/`. Accept the certificate warning.
The IP changes with the network, so read it off `demo.sh` rather than remembering it.

Stop everything: `./demo.sh --stop`

The arm is deliberately **not** started by `demo.sh`. The `go` prompt exists so that a person
is looking at the workspace when the motors come alive.

---

## 2. Running the demo

1. Phone open at the URL `demo.sh` printed, propped so it sees the table.
2. Put the pill bottle down in view — perception records where it landed.
3. Say: *"I forgot where I put my pills."*
4. Pam names the place and offers to fetch.
5. Say yes. The arm drives to home, runs the policy, grips, turns 180°, says the line, drops.
6. It returns to home on its own. Repeat as often as you like.

### Controls while the arm runs

| Key | HTTP | Effect |
|---|---|---|
| `p` | `POST /preflight` | load policy, check camera — **moves nothing** |
| `s` | `POST /start` | drive to home, then run the policy |
| `h` | `POST /handover` | hand over now, don't wait for the jaws |
| `r` | `POST /home` | stop everything and reset to home |
| `x` | `POST /stop` | stop the policy |
| `q` | — | quit |

All on `127.0.0.1:8020`. `GET /status` is read-only.

---

## 3. What is actually built

```
phone browser ──WSS──> Pam (server/app.py) ──> relay :8765 ──> perception pipeline
  camera + mic            Deepgram STT/TTS         JPEG frames      YOLOE + BoT-SORT
                          Claude tool calls                        put-down gate
                                 │                                 Claude VLM
                                 │                                 memory.jsonl
                                 ▼                                      │
                          arm control :8020  <───────────────────────────┘
                          ACT policy :8011                        Elasticsearch
                          OpenYAM 7-DoF arm
```

### Phase 1 — memory (perception/)

A camera watches the room. When a tracked object **moves and then comes to rest**, a VLM writes a
plain-language memory of where it landed.

- **YOLOE** open-vocabulary detection, prompts not trained classes. Running `yoloe-11s-seg.pt`
  at imgsz 640, conf 0.15, 5 fps on `mps`.
- **The gate is motion AND rest, never presence alone.** A bottle sitting on a table for three
  hours is not an event. This is the difference between ~30 VLM calls in a demo and thousands.
- **The detector proposes, the VLM decides.** YOLOE has no "none of these" class, so every box
  wears one of your labels. Both answers are written: `object` (VLM) and `detector_label` (YOLOE),
  so disagreements are greppable.
- **Cost:** ~$0.005 per event, ~$0.12 for a 30-event demo.

**Observed live today:** on two of the recorded memories YOLOE said *water bottle* and the VLM
overruled it to *pill bottle* — "appears to be a Tylenol bottle". That is the split working on
real objects, and it also answers a question left open in CLAUDE.md: the pill-bottle /
water-bottle prompt split does **not** hold reliably on this bottle at conf 0.15, but the VLM
catches it.

### Phase 2 — retrieval (vla/, arm/)

An **OpenYAM 7-DoF arm** (6 joints + gripper, Damiao motors, CAN at 1 Mbps) driven by an
**ACT policy** trained on hand-guided demonstrations.

| | |
|---|---|
| Dataset | 31 episodes, 15,246 frames, 30 fps, wrist camera + joint states |
| Collection | bottle offset along one axis, left / centre / right plus intermediates |
| Model | ACT, ResNet-18 backbone, dim 512, chunk size 100 (3.33 s at 30 fps) |
| Trained on | RunPod RTX 4090 |
| Action error | 1.64° |
| Gripper timing | 31/31 episodes correct |
| Generalisation to unseen positions | 3.13° → **2.64°** with image augmentation |

**Everything after the grasp is scripted, not learned.** The policy was trained on the pick alone
and its episodes end when the gripper shuts, so it has nothing sensible to say afterwards. Once
the jaws settle on the bottle the rollout stops and a taught preset plays: turn 180°, hold,
announce, open, return home.

### Phase 3 — the voice agent (server/)

**Pam** — FastAPI, Deepgram `flux-general-en` listen + `aura-2-thalia-en` speak, Claude for
tool-calling, SSE push channel so the arm can make her speak.

Tools wired to the arm: `fetch_object`, `get_arm_status`, `set_gripper`. The arm posts a stage on
every transition (`homing → reaching → grasped → presenting → released → returning → idle`) so
Pam always knows where it is.

---

## 4. What works, and what doesn't

### Working

- Voice → memory lookup → spoken answer
- Voice → arm fetch → grasp → handover → reset, repeatable
- Perception recording new memories live (`memory: true`, 8 records)
- Elasticsearch (52 indexed memories) + local `memory.jsonl`
- Face records (5), reminders, dose recording, caregiver portal, `/setup` page
- Flights — live offers through SerpApi

### Not working / not configured

- **Google Calendar** — no `GOOGLE_CALENDAR_CLIENT_ID` / `_SECRET`. Configure at
  `http://127.0.0.1:8000/setup`.
- **Hand detection is out of the demo path.** It works (verified: a hand at catching distance
  reads area 0.32 against a 0.012 threshold; stale frames correctly rejected) but the handover
  now releases on a timer instead. Keep your hand underneath.
- **Text messaging and phone calls** — deliberately removed, not a bug.

### Honest limitations

- **The handover is a fixed preset, not learned.** It plays the same taught poses every time
  regardless of where the bottle was picked from.
- **The grasp works over a limited area.** The policy was trained on one axis of bottle
  positions; well outside that it will miss.
- **`--grasp-s 35` is a backstop, not the trigger.** The handover starts when the jaws settle.
  If a pick never closes, the script runs anyway after 35 s and will hand over nothing.
- **Motion smoothing is verified on the commanded trajectory, not on the motors.** Peak
  commanded acceleration fell 15.00 → 0.75 rad/s² (95%), same peak speed and duration. Whether
  the arm physically tracks it better is unmeasured.
- **Dose recording fires early.** Fetching pills marks the dose taken the moment the arm is
  *dispatched*, before it has moved — a documented product decision, but worth knowing.

---

## 5. Safety

- **The 24 V supply switch is the only real e-stop.** Keep a hand near it. A software watchdog
  exists (100 ms) but a wedged CAN bus is exactly the case software cannot be trusted to catch.
- Everyone stays clear of the arm's swing. It turns 180° with a bottle in its jaws.
- The `go` prompt is not a formality — do not script around it.
- **Pam can start the arm by voice on her own.** In a loud room a mis-hear is enough; it has
  happened twice. Once started, the handover runs on a timer with no further input.

---

## 6. When it goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| Pam: "I can't reach the arm" | arm not running, or 8020 taken | check `curl 127.0.0.1:8020/status` |
| Arm doesn't move after `p` | `p` is preflight only | press `s`, or ask Pam |
| Arm grips then just sits | *(fixed)* jaws closed past a "shut on air" threshold | already fixed; if it recurs, press `h` |
| Arm turns with nothing in its jaws | grasp window expired mid-approach | it waits for the jaws now; check the policy is reaching |
| Phone page dead / no audio | Pam restarted, WebSocket gone | reload the page |
| `memory: false` | no `memory.jsonl` yet | it appears on the first real put-down |
| No memories recorded | relay down, or nothing moved | check 8765; the gate needs motion **then** rest |
| Pam answers a stale place | Elasticsearch still holds older records | put the bottle down in view to record a newer one |
| Arm code changes not taking effect | Python loaded the old file | restart `run_policy.py` |

**Logs:** `/tmp/pam.log`, `/tmp/policy.log`, `/tmp/perception.log`.

Reset the arm from anywhere: `curl -X POST http://127.0.0.1:8020/home`

---

## 7. Layout

```
perception/   camera → detect → track → put-down gate → VLM → memory.jsonl
  memory_pipeline.py   the pipeline, and the WSS relay on 8765
  vlm.py               event frames → Claude → structured memory; ask() for queries
  events.py            the detection→VLM contract
server/
  app.py               Pam: Deepgram wiring, tools, arm bridge, HTTPS for the phone
  es.py                Elasticsearch search, canonical-phrase-first
  doses.py schedule.py caregiver_schedule.py   medication tracking
phone/
  agent.html           the phone's page: mic, camera, subtitles
vla/
  run_policy.py        the demo runner: rollout, handover, reset, control port
  policy_server.py     serves the ACT checkpoint
  remote_policy.py     dimOS module, critically-damped action blending
  scripted_demos.py    teach / record / replay, trajectory building, corner smoothing
  hand_release.py      MediaPipe hand detection
  spots/               32 taught poses
arm/                   low-level probe and CAN tools
act_final/             the trained checkpoint being served
```

---

## 8. Facts worth not re-deriving

- **dimOS module calls are cross-process RPC.** Polling one at 10 Hz saturates the channel and
  starves the policy's own calls until they time out. Poll at 3 Hz or less, never in a tight loop.
- **The trajectory executor interpolates position linearly and ignores the `velocities` field.**
  Setting via-point velocities does nothing. Smooth motion has to come from denser waypoints —
  `smooth_corners()` uses a quadratic Bézier through each taught pose.
- **j2 grows as the arm goes DOWN.** Hover ≈ 1.03, grasp ≈ 1.53, sagged rest ≈ 2.36.
- **Gripper: 1.0 open, 0.0 shut**, stalls around 0.37–0.52 holding the bottle — but a narrow
  bottle can close past 0.05, so never infer "holding nothing" from a low reading.
- **macOS:** dimOS workers must use `spawn`, not `forkserver` — a forked child cannot init Metal.
- **MediaPipe must be 0.10.x.** 1.x routes through Metal and aborts on this machine.
- **`clip` lives in `perception/.venv`, not the dimos venv.** The pipeline needs that interpreter.
- **`memory.jsonl` is appended; `events.jsonl` is truncated** on pipeline restart.
