# Compass — project context for Claude

Read this before changing anything in `perception/` or `phone/`. It carries the
reasoning and the measurements behind decisions that are not obvious from the
code, including several that look wrong until you see the numbers.

**Keep it current.** This file is the shared memory for everyone -- human or agent --
working in this repo. When you learn something that would have saved you an hour,
write it here in the same turn you learn it. Measurements beat opinions: paste the
number and say which machine produced it. If you find a claim here that is wrong,
correct it rather than working around it, and say what the new evidence was.

---

## What we are building

An assistive system for elderly people with memory impairment (and often
limited mobility). The product question is narrow and concrete:

> "Where did I leave my medication?" — and then, can something fetch it?

A camera watches the room. When a tracked object is put down, a vision-language
model writes a plain-language memory of where it landed. The user asks by voice
and gets the answer back. Later, a robot arm retrieves it.

Built for HackMIT. The target track reframed mid-project toward "make
non-believers believe in AI", with elderly Alzheimer's users as the audience.

### Phases

| Phase | Scope | Status |
|---|---|---|
| 1. Memory | camera → detect → put-down trigger → VLM → memory log → voice query | **built, on `main`** |
| 2. Retrieval | VLA-driven arm picks up the remembered object | built in parallel, `vla/` |
| 3. Assistant | calendar, reminders, transport, messaging | roadmap only — see caveat below |

**Phase 3 caveat.** Banking, real flight booking, and autonomous texting were
considered and deliberately scoped out. Not time — they are platform or trust
problems. iOS gives no API to silently send a message; giving an agent real bank
access for a cognitively impaired user is an elder-financial-abuse question that
a judge will ask about. Calendar/reminders and a transport request are the safe,
genuinely buildable ones. Everything else is a roadmap slide, presented as such.

### What we are actually building now

The phase-1 loop, on a **phone propped on a table** as the camera. Everything
heavy runs on a laptop. The phone runs nothing but a browser.

---

## Architecture as decided

```
phone browser ──WSS──> laptop                          ──> cloud
  camera capture         YOLOE detect (continuous)          VLM (per event)
  JPEG @10fps            BoT-SORT track                     memory store
  720p, ~20KB/frame      put-down gate (motion)             voice agent LLM
                         events.jsonl  ────────────────────┘
```

**The gate is presence AND motion, never presence alone.** A pill bottle sits on
a table for three hours — "in frame" is not an event. The object must move, then
come to rest. This is the difference between ~30 VLM calls in a demo and
thousands.

**The VLM gets 3–5 sampled stills in ONE call**, not a stream. Before / during /
after is enough to infer the action. Streaming frames means many calls or a huge
payload, for no accuracy gain.

**Where each model runs, and why:**

- **YOLOE on the laptop.** Not the phone (it would need a Core ML export and the
  phone should stay a dumb camera), not the glasses (their chip cannot run vision
  models — MultiSet AI, a shipped app on that same SDK, routes to cloud for
  exactly this reason), not a cloud API (a continuous loop means tens of
  thousands of calls and single-digit fps over conference wifi).
- **VLM in the cloud, via the Anthropic API.** Already wired in `perception/vlm.py`
  with structured output into a pydantic `Memory` schema. Self-hosting was
  rejected: no GPU on the team's Windows machine, and label-reading is the one
  task where quality matters most. At ~30 calls per demo the economics that
  justify self-hosting do not exist.
- **No vector DB yet.** At demo scale every memory fits in the prompt.
  `vlm.ask()` does this deliberately. Add Chroma or pgvector only when it stops
  fitting.

**`events.jsonl` is the contract between halves.** `perception/events.py`
documents the schema and provides `tail_events()` so the VLM can run as its own
process — restartable, rerunnable over old events, and a crash on one side does
not take down the other.

---

## Test results

All measured, all reproducible with the two scripts in `perception/`.

### Prompt discrimination — the finding that changed the defaults

YOLOE is open-vocabulary: prompts are text, not trained classes, and each box is
labelled with the **single best-scoring prompt**. That makes it winner-take-all.

On a photo of real prescription bottles:

| prompts | result |
|---|---|
| `pill bottle, water bottle` | **pill bottle ×7** (best 0.46) |
| `pill bottle, water bottle, bottle` | bottle ×9 (best 0.60) |

On a photo of ordinary (non-medicine) bottles, the same pair returned
**water bottle ×1** and zero pill bottle.

So the discrimination *works* — a generic `bottle` prompt was suppressing it.

**Those numbers are from `yoloe-11s-seg.pt`. The pipeline runs `yoloe-26s-seg.pt`**, which
downloads itself the first time you run with the default `--weights`, so it is easy to
believe you are reproducing a measurement you are not. On the Windows laptop's real
objects, 26s was observed labelling a **pill bottle as a water bottle** - the opposite of
the table above. Before trusting either prompt, run the same frame through both:

```bash
python test_prompts.py --weights yoloe-11s-seg.pt        --imgsz 640 --isolate
python test_prompts.py --weights weights/yoloe-26s-seg.pt --imgsz 640 --isolate
```

### Wheelchair mount: drop `--static-camera`, keep the arm logic

The deployment target is a camera **mounted on the user's wheelchair**, looking out, with
the user's arm reaching in to set things down. That is the head-worn case, not the
desk-webcam case:

- **Keep the arm logic on.** The arm placing the pill bottle is the event we exist to
  catch. `--no-arm` is for testing at a laptop, where the webcam sees your whole torso.
- **`--static-camera` is now actively dangerous and must be dropped.** It replaces the
  ego-motion homography with a raw centre delta, so while the chair rolls, every
  stationary object's centre moves across the frame. At 3 fps a modest roll clears both
  `MOVE_THR` and `ARM_DISP`, so every visible target arms - and the moment the chair
  stops, they all "come to rest" together and fire a burst of false `placed` events, one
  per object. (Derived from the trigger code, not yet measured on a real chair.)

Ego-motion is cheap enough to leave on: measured **13-39 ms/frame** on the Windows laptop
(39 ms at imgsz 640), against 7.6 ms on the Mac.

**But a moving camera argues against the low frame rate.** The homography is fitted with
pyramidal Lucas-Kanade optical flow, which has a limited search radius. At 3 fps the chair
has moved for 333 ms between frames, and if the fit fails the code leaves `step_vec` at
zero - deliberately conservative, so the failure is silent blindness rather than false
events. Higher fps keeps the inter-frame displacement inside what LK can track. **5 fps at
imgsz 480 is the recommended starting point for the chair**; 3 fps / 640 was the right
answer only for a camera that does not move.

Mitigating factor: put-downs happen while the chair is *stopped*, when the camera is
effectively static anyway. The risk is confined to what happens during and just after
motion.

### `--no-arm` is required when the camera faces the user

The arm rules assume a **head-worn** camera: a `person` box running off the bottom edge
is read as the wearer's own arm reaching in, and an object it covers by more than
`CONTACT_THR` is "held", so it is not at rest.

On a webcam or a propped phone pointed **at** someone, that same test matches their whole
seated body. Reproduced 2026-09-19 with a seated-person box (bottom edge at the frame
bottom) and a bottle held in front of them: the person box qualifies as "the arm", covers
**100%** of the bottle against a 0.5 threshold, so `covered` is true every frame, `tr.rest`
is reset every frame, `REST_MIN` is never reached and **`placed` never fires at all**. The
run looks alive - detections, tracks, fps - and silently produces nothing.

So: head-worn or over-the-shoulder camera, keep the arm logic. Camera looking at a person,
pass `--no-arm`. It is not an ablation in that setup, it is the correct mode.

**Keep `person` in the prompt list either way.** It is a decoy, exactly like `water
bottle`: it absorbs people so they do not get labelled as one of the real targets. With
`--no-arm` it stops affecting the trigger and is no longer drawn in the `--show` window,
but it must stay in the vocabulary.

### The trigger fires on EVERY detection, not just the "right" ones

Worth being explicit, because the code reads as though it filters and it does not.
`model.set_classes(targets + [ARM])` makes the target list YOLOE's **entire vocabulary**,
and the loop's `if n not in targets: continue` can therefore only ever exclude `person`.
A mug is labelled `pill bottle`, tracked, gated and fired exactly like a real one. There
is no "other" bucket to fall into and no filter standing between a wrong label and a
Claude call.

Two consequences:

1. **`--conf` is a recall and cost dial, not a correctness dial.** Raising it does not
   make surviving labels right; it only removes boxes. And the two errors are wildly
   asymmetric now: a false positive costs ~$0.005 and a memory the VLM marks `other` or
   low-confidence, while a false negative means no box, no track, no event, no VLM call
   at all - silent and total. Prefer the lower threshold.
2. **The spoken answer needs its own filter**, since junk events are now expected rather
   than exceptional. `vlm.ask()` drops memories the VLM called `other` or scored below
   `JUNK_CONFIDENCE` (0.2), and marks anything under `UNCERTAIN_CONFIDENCE` (0.4) so the
   answering model hedges instead of asserting. Nothing is deleted - `memory.jsonl` keeps
   every call, including `detector_label`, so the disagreements stay measurable.

   Verified 2026-09-19 on a log of one real memory plus three kinds of noise: the mug
   YOLOE called a pill bottle and a 0.1-confidence blur were both dropped, and "where is
   my medicine?" answered with the kitchen counter rather than the water bottle.

### The detector proposes, the VLM decides

This is the structural answer to the mislabelling above, and it is why the split not
holding is survivable.

YOLOE has **no background or "none of these" class**. It scores your prompt list against
each box and applies the single best-scoring one. Every detection therefore wears one of
your labels whether or not it is that thing, and visually similar prompts get swapped.
Raising `--conf` reduces how many boxes appear; it does not make the surviving labels
right.

So the detector's label is treated as a hint, not an answer. Each event carries a
`targets` list - every prompt the detector could have chosen from - and `vlm.py` hands
that to the model as its candidate set, telling it plainly that the detector cannot say
"none of these" and confuses a pill bottle with a water bottle. The VLM picks from the
candidates, or returns `other`. Both answers are written to `memory.jsonl`: `object` is
the VLM's, `detector_label` is YOLOE's, so disagreements are greppable and are the real
evidence on whether a prompt set discriminates.

Verified 2026-09-19: given a frame of an obvious pill bottle and told the detector called
it a **water bottle**, the VLM returned **pill bottle** at confidence 0.6.

This is the right division of labour anyway. YOLOE's job is "something moved and came to
rest, here"; reading a pharmacy label off a small amber cylinder is a job for a model that
can actually read. Do not try to fix identity purely with prompt engineering.
**Never put a generic prompt in `--targets` beside a specific one.** The old
default was `--targets bottle`, which would have logged the demo's water-bottle
distractor as the medication.

### False positives — over 128 real photos containing none of these objects

| prompt | @0.10 | @0.30 | @0.40 | worst conf |
|---|---|---|---|---|
| `keys` | 0.0% | 0.0% | 0.0% | — |
| `pill bottle` | 3.9% | 0.8% | 0.0% | 0.32 |
| `glasses` | 7.0% | 2.3% | 2.3% | 0.88 |

Real pill bottles reach **0.46**; the worst false one is **0.32**. The old
`--conf 0.10` sat inside the noise floor. Now 0.30 — not 0.35, because the
multi-frame gate (`ACTIVE_MIN`, `REST_MIN`) already discards one-frame flukes.

**`glasses` is not broken.** It looked bad at 7% until the images were inspected:
every hit was **correctly detected eyewear worn on a face**. In a wine-tasting
photo it found the sunglasses and ignored the wine glasses — it does not confuse
eyewear with drinkware. Worn glasses are filtered by the placement gate, not the
threshold. Do not "fix" this prompt.

### Speed — YOLOE on CPU (no GPU)

| imgsz | ms/frame | fps |
|---|---|---|
| 640 | 285 | 3.5 |
| 480 | 178 | 5.6 |
| 416 | 108 | 9.2 |
| 320 | 70 | 14.2 |

416 is the sweet spot on CPU. Smaller hurts small objects — **keys first**.
On Apple Silicon with `mps` it is far faster; run it and see.

### Speed — the same test on the Windows laptop (CPU, 2026-09-19)

YOLOE `yoloe-11s-seg.pt`, 6 prompts, `device=cpu`, torch 2.14.0+cpu, no GPU of any kind.

| imgsz | ms/frame | fps | vs. the Mac CPU numbers above |
|---|---|---|---|
| 640 | 237 | 4.2 | a shade faster |
| 480 | 160 | 6.3 | a shade faster |
| 416 | 110 | 9.1 | the same |
| 320 |  81 | 12.3 | slightly slower |

So **the Windows laptop is a viable detection machine**, and 416 lands at 9.1 fps
against the pipeline's 10 fps target. 640 gives 4.2 fps and will drop frames.

**`set_classes()` is not free and the first call is brutal.** It runs the MobileCLIP
text encoder over every prompt. First ever call: **173 s**, almost all of it a
one-time download of the text encoder. Warm: **7.0 s for 6 prompts, 1.5 s for 2**.
Budget that as startup cost, and do not add prompts you do not need. A full pipeline
run on this laptop measured `detect_track_ms` of 72-82 ms at imgsz 416.

### Bystanders: keep the `person` prompt, gate what counts as the arm

`person` must stay in `--targets`' vocabulary. It is a decoy exactly like `water bottle`:
it absorbs people so they are not labelled as a real target. Delete it and an arm starts
scoring as a pill bottle, because YOLOE cannot answer "none of these".

What was wrong was that *any* person box touching the near edge counted as the user's own
arm. In a crowded room that includes anyone walking past, and an object they pass in
front of gets marked "held" -- suppressing its put-down. `ARM_MIN_AREA` (0.06 of the
frame) now also requires the box to be large, i.e. close. Measured on a 640x480 frame:

| person box | at near edge | area | counts as arm |
|---|---|---|---|
| user's arm reaching in | yes | 19.5% | **yes** |
| user seated, torso filling frame | yes | 54.7% | **yes** |
| bystander stood at the bottom edge | yes | 3.4% | no |
| bystander mid-room | no | 5.5% | no |
| person walking past, far | yes | 3.8% | no |

The `--show` window now draws only person boxes that pass this test. A bystander is not
drawn and affects nothing: it cannot mark an object held and cannot suppress a put-down.
Note the seated torso still qualifies - that is the laptop-webcam case, and the fix for
it is camera aim or `--no-arm`, not this gate.

### Confidence sweep on a real scene (2026-09-19, Windows laptop)

Swept with `perception/sweep_conf.py`: capture frames, predict once at the lowest
threshold, filter upward so every threshold sees identical detections. COMBINED mode,
imgsz 640, prompts `pill bottle, water bottle, keys, phone` + `person`.

Scene: a busy hackathon hall - several people, laptops, chairs, tables, curtains.

| conf | false positives (6 frames, no target objects present) | `person` |
|---|---|---|
| 0.05 | **0/6** | 6/6 |
| 0.10 | **0/6** | 6/6 |
| 0.15 | **0/6** | 6/6 |
| 0.20 | **0/6** | 6/6 |
| 0.30 | **0/6** | 6/6 |
| 0.40 | **0/6** | 6/6 |

Same on `yoloe-26s-seg.pt` and `yoloe-11s-seg.pt`. **Zero false positives all the way down
to 0.05** on a cluttered scene full of people and furniture. The earlier "it detects
everything" impression was `person` firing 6/6 - correct behaviour, and it can never fire
an event, since `person` is not in `--targets`.

So on this evidence there is no false-positive reason to keep `--conf` high, and the
asymmetry (a missed bottle is silent and fatal, a false positive costs ~$0.005 and the VLM
relabels it) argues for going low. **Start at 0.15.**

**Recall is still unmeasured.** Two capture attempts produced no pill bottle in frame at
all - it was on the desk, below a laptop webcam angled up at the user's face. That is the
single most important number still missing, and it cannot be obtained without the physical
bottle in the camera's field of view. Nothing about the threshold can be concluded until
it is.

**Aim the camera at the surface, not the face.** A laptop webcam sees the user; a pill
bottle on the desk is out of frame entirely, and every "it isn't detecting it" symptom
follows from that rather than from any threshold. On the wheelchair the camera must point
down and forward at the lap/table - which also makes the `person` box an arm reaching in
rather than a torso, the geometry the contact rule needs.

### Frame rate is a detection-resolution trade, not a VLM cost

The VLM never saw 10 fps. It gets 1 or 3 stills per *event*, so lowering `--fps`
saves nothing on Claude -- it buys CPU, and CPU buys resolution:

| --fps | budget/frame | imgsz that fits | measured |
|---|---|---|---|
| 3 | 333 ms | **640** | 319 ms — keeps up |
| 4 | 250 ms | 640 | 267 ms — marginal |
| 5 | 200 ms | 480 | 172 ms — keeps up |
| 10 | 100 ms | 416 | 110 ms — and 416 is where keys start to vanish |

**3 fps at 640 is the configuration to reach for when small objects matter.** What you
give up is tracking robustness: BoT-SORT has to associate a hand-carried object across
333 ms gaps, and the ID-switch donor rule is a mitigation, not a fix. If tracks start
breaking mid-carry, go to 5 fps / 480 before giving up resolution.

**The gate is specified in seconds, not frames.** It used to be frames, which meant
`--fps` silently rescaled every threshold in it. Do not reintroduce a frame count:
put the duration in `*_S` and let the startup conversion do the rest.

### What a VLM call actually costs

**Measured** on `claude-sonnet-5` ($2/$10 per MTok), 640x480 frames, 2026-09-19:

| event | images | input tok | output tok | latency | cost |
|---|---|---|---|---|---|
| `placed` | 3 | 1,975 | 79 | 2.7 s | $0.0047 |
| `sighted` | 1 | 1,132 | 75 | 3.3 s | $0.0030 |

A ~30-event demo is about **$0.12**. Note the single-image call is 1,132 tokens,
not the ~600 a naive `w*h/750` estimate gives: the system prompt and the injected
`Memory` JSON schema are most of the difference, and they are paid on every call.
Output is tiny (~75 tokens) because the schema is small and effort is `low`. Do not guess at this: every call records
`input_tokens` / `output_tokens` in `memory.jsonl`, and a run totals them into
`stats.json` as `vlm_calls` / `vlm_input_tokens` / `vlm_output_tokens` /
`vlm_cost_usd`. **There is no agent loop anywhere in this system** -- one event is one
request and one response, no tools, no retries -- so `max_tokens` is bounding answer
length, not runaway iteration.

### Other measurements

- **Ego-motion homography**: 7.6 ms/frame, 0 with `--static-camera`. The real
  reason to use the flag on a propped phone is not speed — a sparse table gives
  few background features, and a bad homography fit reads as phantom object
  motion that can arm the trigger on a still object.
- **Track-table leak** (fixed): `tracks{}` was never pruned. 6h of simulated
  churn went from 21,600 entries to 10.
- **Phone uplink**: 720p JPEG at quality 0.7 ≈ 20 KB/frame → ~200 KB/s at 10fps.
  Trivial for any phone; the page drops frames above 256 KB buffered rather than
  building an unbounded backlog.

---

## Learnings that cost time — do not relitigate

- **Meta glasses were dropped.** The SDK does support continuous video
  (2/7/15/24/30 fps; LOW 360×640 / MED 504×896 / HIGH 720×1280, adaptive but
  never below 15fps) plus 12MP stills. What killed it: **no custom wake word and
  no raw always-on mic for third-party apps**, so "Hey Pammy" is impossible — a
  frame tap is the only trigger. Combined with hardware availability, the phone
  won.
- **Hailo / ASUS UGen300 does not work with YOLOE.** Ultralytics' Hailo export
  docs state plainly that YOLOE, YOLO-World, YOLOv10 and RT-DETR are not
  supported. Hailo's own forum confirms multi-input open-vocabulary models are
  not in their Model Zoo. Going Hailo means abandoning YOLOE for a fine-tuned
  standard YOLO plus the Dataflow Compiler and ≥1,024 calibration images. Latency
  was never the problem (Hailo-10H does YOLOv8n at ~400fps); model compatibility
  is.
- **The open-world version of this problem is unsolved research.** Ego4D's online
  visual-query benchmark — literally "where did I leave my keys" — tops out around
  **4% success** with real detection and tracking, though **82%** with oracle
  components. That is not our problem: we use one known object set, one room,
  explicit event triggering, and a VLM writing text rather than pixel-perfect
  re-localization. Do not panic when you read that number.
- **A `sighted` event carries ONE frame, not three.** It used to send three. A track
  that never moved has `active_start`/`active_end` of `None`, and the old
  `(tr.active_start or t) - 1.0` collapsed all three samples onto the same moment --
  then labelled them BEFORE / DURING / AFTER, which reads to the VLM as a placement it
  would go on to describe. Verified on a real run: `t_before` and `t_during` are now
  `null` and exactly one JPEG is written.
- **`sighted` also fires for anything already in frame at startup**, because
  `last_seen` is empty on the first pass. That is presence alone, which is the one
  thing the gate is supposed to never trigger on -- but it is useful (it seeds the log
  with what is already on the table) and now costs one frame per object rather than
  three. Know that starting the pipeline with five targets in view means five Claude
  calls in the first second.
- **No fine-tuning has happened.** YOLOE is zero-shot. There is training
  scaffolding from before YOLOE was chosen, but it only ever saw synthetic
  rectangles. Fine-tuning (`YOLOEPESegTrainer`) is the fallback if prompts prove
  insufficient on real objects.

---

## Running on Windows, without a GPU

The pipeline was written on a Mac and several things quietly assumed it. All of the
following is measured on the team's Windows 11 laptop, 2026-09-19.

```bash
python -m venv perception/.venv
perception/.venv/Scripts/python.exe -m pip install -r perception/requirements.txt
```

- **Use the venv, not the global interpreter.** The global environment on that laptop
  is broken: opencv 4.6.0.66 is compiled against numpy 1.x while numpy 2.2.4 is
  installed, so a bare `import cv2` raises `RuntimeError: module compiled against ABI
  version ...`. Nothing in this repo runs there.
- **`mlx-whisper` is Apple-Silicon only** and used to be an unconditional line in
  `requirements.txt`, which made `pip install -r` fail outright on Windows. It is now
  gated on `sys_platform == "darwin"`.
- **`voice.py` imports lazily** for the same reason. `--text` needs no microphone and
  no Whisper, so it works everywhere; `speak()` uses `say` on macOS, SAPI via
  PowerShell on Windows, `espeak` otherwise.
- **`pip install -r` resolves `anthropic` to the 1.x line** (1.7.0 as of writing), not
  the 0.x some machines still have. `vlm.py` is compatible with both: `messages.parse`,
  `output_format=`, `output_config=`, `extra_headers`/`extra_body` all exist in 1.7.0.
  The fallback beta is passed via `extra_headers`/`extra_body` rather than `betas=`
  precisely because `betas=` only exists on `client.beta.messages`.
- **Weights.** `yoloe-11s-seg.pt` downloads itself (26 MB) on first use, so a fresh
  machine can run `test_prompts.py` immediately. `weights/yoloe-26s-seg.pt` -- what the
  pipeline defaults to -- is shared out of band and is not on the Windows laptop yet.
- **Editing these files: write UTF-8 explicitly.** Several of them contain em dashes.
  On Windows, Python's `Path.read_text()`/`write_text()` default to cp1252, which
  round-trips existing UTF-8 bytes by luck but silently mangles any non-ASCII character
  you add. Pass `encoding="utf-8"`.
- **Firewall.** The phone path needs inbound 8443 (page) and 8765 (relay). Windows
  Defender will prompt on first bind, or silently drop if a previous "Cancel" is
  remembered. `phone/README.md` was written for macOS and does not mention this.

### The demo window

`memory_pipeline.py --show` opens a live view: every detection boxed (targets yellow,
the arm/person blue) with class, confidence and track ID, a header with fps/device/
imgsz/conf and the event count, an orange **VLM THINKING...** banner while a call is in
flight, and the returned memory -- event, object, confidence, location description --
in a panel underneath. VLM failures (no key, no network, a refusal) are shown in the
panel in red instead of vanishing into a thread traceback. `q` quits cleanly, so stats
are still written. It draws from the same `r.boxes` the trigger consumes, so the window
cannot disagree with what the pipeline acted on.

## Open questions

1. **Does the pill-bottle / water-bottle split hold on the team's real objects?**
   This is the single biggest unverified assumption — everything on `main`
   depends on it. Two photos say yes; the actual bottle under actual lighting has
   never been tested. Run `python perception/test_prompts.py`, put both bottles in
   frame, press `i`.
2. **Are keys detectable at `--imgsz 416`?** Small object, and 416 is where the
   CPU budget lands. Untested on real keys. The pipeline now takes `--imgsz`, so this
   can be answered end-to-end and not just in `test_prompts.py`.
3. **Trigger accuracy on real footage.** `blockers/eval_epic.py` exists to score
   this; no numbers recorded yet.
4. **End-to-end latency on the demo Mac.** All speed numbers above are CPU.

---

## Repo map

```
perception/
  memory_pipeline.py   camera → detect → track → put-down gate → events.jsonl
  events.py            the detection→VLM contract + tail_events()
  vlm.py               event frames → Claude → structured memory; ask() for queries
  voice.py             push-to-talk → Whisper → vlm.ask() → spoken answer
  glasses_rx.py        WebSocket relay, ws:// and wss://; any JPEG sender works
  fake_glasses.py      replays a video file as a sender (no phone needed)
  test_prompts.py      does a prompt discriminate? webcam or image, press i
  eval_prompts.py      false-positive / recall / speed over a folder
  blockers/            trigger scoring against EPIC-KITCHENS
phone/
  index.html           browser camera client → WSS (no app, works on Android too)
  serve.py             HTTPS server + self-signed cert (camera needs secure context)
arm/, vla/             phase 2 — arm probe, teach-and-replay, VLA policy stack
```

### Running it

```bash
# phone as camera
python phone/serve.py                       # then open the printed https:// URL
cd perception && python memory_pipeline.py \
    --source wss://0.0.0.0:8765 --cert ../phone/cert.pem --key ../phone/key.pem \
    --static-camera --out runs/phone

# no phone needed
python fake_glasses.py data/clips/place_desk.mov
python memory_pipeline.py --source 0 --static-camera --out runs/live   # webcam
```

`phone/cert.pem` is gitignored — each person generates their own via `serve.py`.
Model weights are gitignored too and shared out of band.

### Pam camera and accessible UI verification

- `phone/agent.html` now sends JPEGs to the same-origin `/api/camera` WebSocket.
  `server/app.py` forwards them to `wss://127.0.0.1:8765`, verifying the relay
  against `phone/cert.pem`. Only the standalone `/camera` page still connects to
  port 8765 directly. An optional `CAMERA_RELAY_URL` supports a different local
  relay address (loopback only).
- Restart `python server/app.py` after changing backend routes. The HTML is read
  from disk immediately, so a new page with an old Python process can produce
  403 WebSocket rejections. Use the direct localhost/HTTPS origin, not an IDE
  preview proxy, for microphone and camera testing.
- A visible camera preview proves capture, not delivery. Pam marks it connected
  after the server forwards a frame to the relay. It retries interrupted links;
  Stop camera releases tracks and cancels retries independently of voice.
- Saved-memory notifications tail the configured `MEMORY_JSONL` through SSE.
  Existing records populate quietly; complete new records notify. Partial lines
  wait for completion, repeated records are deduplicated, and uncertain events
  are not described as confirmed resting places. This works without Elasticsearch.
- Do not casually restart perception with an existing `--out` directory:
  `memory_pipeline.py` opens `events.jsonl` in write mode and overwrites it.
  Use a fresh run directory and explicitly coordinate `MEMORY_JSONL`, or obtain
  permission to replace the previous artifacts.
- Run `perception/.venv/Scripts/python.exe server/test_ui.py -v` for isolated
  backend/browser tests: real local TLS, camera failure/recovery, all 12 UI
  handlers with mocked responses, file-to-SSE updates, reduced motion,
  light/dark contrast, preferences, keyboard access, and 320-1440px layouts.
  Fixtures use temporary memory files, never the real memory database.
- `server/test_real_dg.py` checks a real Deepgram browser session, asserts PCM
  bytes in both directions, and verifies an object question causes a function
  response followed by a spoken location and visible five-line subtitles.
  On Windows, `--spoken` synthesizes a question with SAPI into a temporary WAV
  and feeds Chromium's fake microphone: actual capture, STT, memory lookup, and
  answer audio all run. Actual phone microphone/speaker behavior still needs a
  human test.
- The medicine/reminder shortcuts were removed at the user's request. The
  Features dialog lists all 12 configured functions plus step-by-step guidance;
  browsing or searching it never executes an action. A rendered result card
  alone does not establish that a voice agent received the result.
- Camera origin mismatches must not be resolved by allowing arbitrary origins.
  The endpoint returns an explanatory error and closes before opening the relay;
  the page suppresses automatic retries for this non-transient failure.
- Pam uses Talk / Memories / Camera tabs in the TOP navigation. There is no
  bottom navigation anymore. The main region scrolls independently. A compact
  camera strip stays outside the tab panels: above the call room on desktop,
  below it on phones, and beside it on short landscape screens.
  Expanding it uses a second video element with the SAME MediaStream, never
  another getUserMedia call. Closing the dialog leaves capture and relay intact.
  Camera off, live preview, and connection-to-memory are distinct states.
- Conversation history is a native dialog, opened from the Conversation link
  or from Features on short screens. Tests must open it before reading visible
  transcript text; text_content() can inspect its retained content while closed.
- The 3D voice pearl uses native WebGL with a CSS fallback, no CDN or framework.
  Input and output AudioContext analysers drive its shape from actual audio.
  Output playback sources determine speaking state through their final onended
  event; AgentAudioDone can arrive before queued audio finishes playing.
- Rendering is capped at 30 animation frames per second and a 420px drawing
  buffer dimension. Reduced motion, Pause visual motion, background documents,
  offscreen visuals, and non-Talk views stop the animation loop. Muting disables
  microphone tracks without stopping Pam's playback or camera.
- `server/test_ui.py` also checks audio-reactive WebGL, CSS fallback, tab keyboard
  controls, recent-memory search, mute/unmute, and a 320x568 touch viewport.
- The idle Talk screen fits 320x568 through 430x932 and 768x1024 without
  scrolling. Controls and the persistent camera stay below the header and
  within the viewport. Long
  results may scroll inside main; accessibility text enlargement must remain
  usable rather than being clipped to force a fit. Long subtitle tests assert
  the Talk control remains onscreen, not just that the viewport lacks a scrollbar.
- The voice sphere is a plain shaded circle (no ray marching): it breathes
  slowly and swells with the live audio level. It needs
  `OES_standard_derivatives` for anti-aliased edges; the CSS fallback renders
  when WebGL or that extension is unavailable.
- The presentation layer was fully replaced with `phone/assets/pam.css`.
  Do not add inline CSS overrides to resurrect the previous orb-card layout.
  Tokens: white `#fff`, paper `#f5f7fa`, ink `#1b2938`, slate `#536375`,
  line `#e3e9f0`, blue `#2367bb`, pressed blue `#19549d`.
- White is now EXPLICITLY locked even under the system dark preference, per
  the user's request. Higher contrast, larger text, and reduced motion remain.
- Reference-driven structure: top pill navigation; greeting/date; compact
  camera session strip; a white telehealth-style call room with a small Pam
  presence header, central dialogue area and bottom call controls; real recent
  memories in a desktop activity column. The activity column is omitted on
  mobile because the Memories tab provides the same records.
- The elder SVG appears in the idle welcome scene and Features dialog. It is
  hidden when subtitles arrive. No fake charts, patient metrics or sample
  memories are added to the live UI. Sidebar rows are generated from the same
  SSE memory records as the Memories view. Tabler icons retain their license.
- The ring and sphere have their OWN fixed-size visual zone with paint
  containment. Never put text and the sphere inside a size-contained flex
  stage with min-height:0: added captions/results can collapse that stage and
  paint the sphere over text. Reserve separate grid rows, or columns on short
  screens. The ring uses an inset within the sphere box, not a negative inset.
  The 12-size live-resize regression also needs speaking-state overlap tests.
- Mute is hidden until SettingsApplied and hidden again on stop(). The
  .actions grid is Features + Talk, then Features + Talk + Mute when live.
- Saved-location images are gated by showCard(card, source): only source
  find_object may render card.image. Other function calls clear stale results.
  Memory notifications and browsing history never auto-display saved images.
  Explicit show_photo remains a separate family-photo capability.
- Agent ConversationText messages feed #subtitle-text. #subtitle-window is
  clipped to five exact line-heights and scrolls to its newest lines after
  updates and resizes. A new user turn resets on the next assistant chunk;
  stopping voice leaves the last answer readable. These are transcript-driven
  captions, not word-timestamp karaoke. Full history remains in the dialog.
- Memories are an interactive collection: By item groups observations by a
  normalized object label; Timeline shows individual records newest-first.
  Both modes use the same text search. Counts refer to observations, NOT
  unique physical objects or distinct places. Recent-memory sidebar cards
  share the same grouped representation and object icons (Tabler).
- Memory cards open a native history dialog, not the Talk result panel.
  Previous/Next and timeline entries select a recorded observation. Opening
  captures a snapshot of that item's history, so incoming SSE records do not
  change the observation being read. On close, focus returns to the matching
  newly rendered card if the original trigger was replaced by an SSE update.
- Uncertain observations retain their unconfirmed-location label; missing or
  invalid timestamps say Time not recorded rather than showing January 1970.
  Neither browsing mode nor the history dialog requests saved frame images.
- The welcome illustration can occupy up to 400px (previously 240px). On
  mobile, text stays on the left and the illustration on the right at every
  viewport height; do not reintroduce the old tall-phone vertical stack.
  It stays hidden during subtitles.
- The resize checks include top-nav geometry, a white-only system-theme
  invariant, caption/control separation with larger text, and real sidebar
  activity. Prefer running the speech smoke test separately from GPU-heavy
  screenshot tests: one concurrent run transcribed only part of the synthetic
  question, while an isolated retry passed on the Windows laptop.
- The logo is `phone/assets/pam-logo.png` (white star-in-"p" wordmark on
  transparency, served at `/assets/`). It is rendered via CSS `mask-image` on
  `.wordmark` with `background-color: var(--blue)`, so it takes the theme
  colour and stays visible on white. Never place the raw PNG on a light
  background; `test_logo_renders_from_png_mask` guards this.
- `test_contrast_and_keyboard` waits 250ms after switching colour scheme so
  button `background-color` transitions settle before sampling. Without that,
  a mid-fade colour can produce a false contrast failure.
- Reading options retain larger text, higher contrast, and reduced motion.
  Panels are solid; the old pam-opaque preference is no longer used.

### Calendar and capability audit

- `server/google_calendar.py` implements Google's read-only OAuth flow using
  google-auth-oauthlib, state/cookie binding and PKCE. Setup is laptop-only at
  `http://127.0.0.1:8000/api/calendar/google/connect`; the registered callback must
  be `http://127.0.0.1:8000/api/calendar/google/callback`. Client ID/secret belong
  in the gitignored server/.env, never in chat or source. Google Maps credentials
  do not authorize Calendar. Features shows the current connection/setup state.
- Tokens persist outside the repository in the user's Pam application-data
  directory, encrypted with Windows DPAPI on Windows and mode 0600 elsewhere.
  Callback codes/state are redacted from uvicorn access logs. Never serve token
  files via the frame route; frames are restricted to image files in run folders.
- Google Calendar reads the primary calendar, expands recurring events with
  singleEvents=true, follows pagination, handles cancellations/all-day events,
  and computes today in the calendar's timezone. There is no silent demo.ics
  fallback. Live account verification requires the user's OAuth credentials and
  consent; mocked integration tests alone do not prove a real account is linked.
- `server/test_capabilities.py -v` tests real backend contracts with temporary
  reminder files and mocked delivery/robot/Google providers. `--live-routing`
  tests real Deepgram function selection without executing ANY requested tools;
  `--only guide_me` checks the prompt-only step-by-step response separately.
  Browser tests also isolate the camera relay and calendar token store. Never
  inject synthetic video into a live memory pipeline during voice smoke tests.
- Reminder tests cover tomorrow/past-clock parsing, validation, due delivery,
  and persistence of fired flags. The scheduler must save the modified list,
  not re-read the old file and accidentally discard fired=True changes.
- Texting and calling were removed at the user's request: no send_message,
  call_contact, or call_caregiver tools, no message/call/caregiver-card endpoints,
  and no Get help calling button. Contacts still supply family-photo captions.
  Do not reintroduce phone or SMS actions through a merge or fallback.
- Capability claims must distinguish API/card tests from real delivery. Uber is
  an external-app handoff, and fetch_object starts a configured robot policy
  without passing the item name. Do not live-test side effects without approval.
- Voice is the primary interface. Flight function results must contain the actual
  offer details in `say`, because only `say` reaches the voice model. Do not put
  essential information solely in a card or tell the user to look at a screen.
  Flights no longer fall back to a Google Flights link when credentials are absent.
- Flight searches use AMADEUS_ENV=test by default and label cached test offers as
  non-live examples. Production credentials plus AMADEUS_ENV=production are needed
  for real-time quotes. Never silently switch to production or buy tickets.
  One-way/one-adult offers include airports, local dates/times, airline, stops,
  total price and its actual currency; the voice agent presents one at a time.
- Uber handoff responses explain the limitation through speech and can
  retain optional helper cards. Do not equate prompt-level verbal approval with a
  server-enforced confirmation gate. Restart the voice session after prompt changes.
