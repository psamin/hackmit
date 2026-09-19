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
