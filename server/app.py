"""Pam — the voice-agent backend. One FastAPI app serving the phone UI and every
function the Deepgram agent can call.

    python server/app.py                 # HTTPS :8443 for the phone + HTTP :8000 for laptop testing

The phone talks to Deepgram directly (voice WebSocket); every agentic action is a
client-side function call that lands back here over plain HTTP on the same origin
the phone already trusts. Nothing here needs to be publicly reachable.

Env (read from server/.env then perception/.env):
    DEEPGRAM_API_KEY     console.deepgram.com -> API Keys (Member role) -> /api/dg-token
    ELASTICSEARCH_URL    optional; find_object falls back to memory.jsonl without it
    MEMORY_JSONL         path to a pipeline run's memory.jsonl (fallback + source of truth)
    CALENDAR_ICS_URL     published .ics feed for get_schedule
    TWILIO_SID/TOKEN/FROM/USER_PHONE   silent SMS + call bridging; cards work without it
    AMADEUS_KEY/SECRET   flight search; falls back to a Google Flights link
    HOME_LAT/HOME_LON    weather + ride pickup (default: MIT campus)
"""
import asyncio, hashlib, json, os, ssl, subprocess, sys, time
from collections import deque
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import (FileResponse, JSONResponse, PlainTextResponse,
                               StreamingResponse)
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "server"
PHONE = ROOT / "phone"
CERT, KEY = PHONE / "cert.pem", PHONE / "key.pem"

for env in (HERE / ".env", ROOT / "perception" / ".env"):
    if env.exists():
        for line in env.read_text().splitlines():
            k, _, v = line.partition("=")
            if k.strip() and not k.startswith("#"):
                os.environ.setdefault(k.strip(), v.strip())

HOME = {"lat": float(os.environ.get("HOME_LAT", "42.3601")),
        "lon": float(os.environ.get("HOME_LON", "-71.0942"))}
CONTACTS = json.loads((HERE / "contacts.json").read_text())
MEMORY_JSONL = Path(os.environ.get("MEMORY_JSONL", ROOT / "perception" / "runs" / "live" / "memory.jsonl"))
REMINDERS = HERE / "reminders.jsonl"

app = FastAPI(title="Pam")


# --------------------------------------------------------------------------
# Deepgram browser auth: mint a short-lived JWT so the API key stays server-side.
# --------------------------------------------------------------------------
@app.get("/api/dg-token")
async def dg_token():
    key = os.environ.get("DEEPGRAM_API_KEY")
    if not key:
        return JSONResponse({"error": "DEEPGRAM_API_KEY not set (server/.env)"}, 500)
    async with httpx.AsyncClient() as c:
        r = await c.post("https://api.deepgram.com/v1/auth/grant",
                         headers={"Authorization": f"Token {key}"},
                         json={"ttl_seconds": 60})
    if r.status_code != 200:
        # The key lacks Member role, so it can't mint JWTs — but it still
        # authenticates the agent socket directly as the subprotocol token.
        # Fine here: this endpoint only serves pages on our own LAN.
        if r.status_code == 403:
            return PlainTextResponse(key)
        return JSONResponse({"error": f"Deepgram grant failed: {r.status_code} {r.text}"}, 502)
    return PlainTextResponse(r.json()["access_token"])


# --------------------------------------------------------------------------
# Agent config: prompt + functions + keyterms live server-side so the page is dumb
# and keyterms can be built from contacts.json at request time.
# --------------------------------------------------------------------------
def agent_config():
    names = [c["name"] for c in CONTACTS["contacts"]]
    places = list(CONTACTS["places"])
    keyterms = names + places + ["pill bottle", "medication", "Pam"]
    return {
        "listen": {"provider": {"type": "deepgram", "model": "flux-general-en",
                                "keyterms": keyterms, "eot_threshold": 0.7,
                                "eot_timeout_ms": 7000}},
        "think": {"provider": {"type": "anthropic", "model": "claude-haiku-4-5"},
                  "prompt": SYSTEM_PROMPT, "functions": FUNCTIONS},
        "speak": {"provider": {"type": "deepgram", "model": "aura-2-thalia-en"}},
        "greeting": "Hi, it's Pam. What can I do for you?",
    }


SYSTEM_PROMPT = """You are Pam, a warm voice companion for an elderly person with memory difficulties.

How you speak:
- One or two short sentences at a time. Warm, unhurried, never condescending.
- One question at a time. If something is unclear, gently ask again.
- Names, times, places and locations come from function results only — never guess them.

Actions:
- Before sending a message, calling, ordering a ride, or fetching something, say what
  you're about to do and that a button will appear on their screen to confirm.
- guide_me: when they ask how to do something, give exactly ONE step, then ask if
  they're ready for the next. Never list all the steps at once.
- If they sound confused, scared, or ask for help, offer to call their caregiver
  with call_caregiver.
- If find_object finds nothing, say honestly that you didn't see it — never invent a place.
- find_object may come back with SEVERAL places. Read out every one, newest first, with
  when you saw it. Never mention only the most recent: the medication they want may be
  the one in the other room. You cannot tell whether that means two bottles or one that
  was moved, so say where you have seen it, not how many there are.
"""


def fn(name, description, params=None, defer=False):
    d = {"name": name, "description": description,
         "parameters": params or {"type": "object", "properties": {}}}
    if defer:
        d["defer_until_eot"] = True
    return d


_str = lambda desc: {"type": "string", "description": desc}

FUNCTIONS = [
    # read-only: dispatched the moment the model emits them
    fn("find_object", "Find where the user left a belonging",
       {"type": "object", "properties": {"item": _str("the item, e.g. 'pill bottle'")}, "required": ["item"]}),
    fn("get_schedule", "List today's calendar events"),
    fn("get_reminders", "List the user's active reminders"),
    fn("get_weather", "Current weather at the user's home"),
    fn("show_photo", "Show a photo of a person on the user's screen",
       {"type": "object", "properties": {"name": _str("person's first name")}, "required": ["name"]}),
    fn("search_flights", "Show flight options on the user's screen",
       {"type": "object", "properties": {"destination": _str("city or airport"), "date": _str("travel date, e.g. 'next Friday'")},
        "required": ["destination"]}),
    # side effects: only fire after the user's turn is confirmed
    fn("set_reminder", "Set a reminder that Pam will speak aloud at the given time",
       {"type": "object", "properties": {"text": _str("what to remind"), "in_minutes": {"type": "number", "description": "minutes from now"}, "at": _str("or a time like '14:30'")},
        "required": ["text"]}, defer=True),
    fn("send_message", "Send a text message to a contact; a confirm button appears on screen",
       {"type": "object", "properties": {"to": _str("contact's first name"), "message": _str("the message")}, "required": ["to", "message"]}, defer=True),
    fn("call_contact", "Call a contact; a confirm button appears on screen",
       {"type": "object", "properties": {"name": _str("contact's first name")}, "required": ["name"]}, defer=True),
    fn("call_caregiver", "Call the caregiver right away when the user needs help", defer=True),
    fn("request_ride", "Prepare a ride to a named place; a confirm button appears on screen",
       {"type": "object", "properties": {"destination": _str("place name, e.g. 'home', 'airport', 'doctor'")}, "required": ["destination"]}, defer=True),
    fn("fetch_object", "Send the robot arm to fetch an item it knows where to find",
       {"type": "object", "properties": {"item": _str("the item")}, "required": ["item"]}, defer=True),
]


@app.get("/api/agent-config")
async def get_agent_config():
    return agent_config()


# --------------------------------------------------------------------------
# Push channel: SSE to the phone page + a POST to inject messages into the
# conversation (reminders firing, test hooks, "the arm is done").
# --------------------------------------------------------------------------
_subscribers: set[asyncio.Queue] = set()


def memory_notice(mem):
    identity = {k: mem.get(k) for k in ("event_id", "logged_at", "object", "event", "location_description", "frames")}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    obj = str(mem.get("object") or "an object")
    certain = mem.get("event") == "placed" and obj != "other" and float(mem.get("confidence") or 0) >= .4
    return {"id": key, "object": obj, "logged_at": mem.get("logged_at"),
            "title": "Memory saved" if certain else "Observation saved",
            "detail": f"{obj.capitalize()}: {mem.get('location_description') or 'location recorded'}" if certain
                      else f"A new observation of {obj}. Its resting place is not confirmed."}


class MemoryTail:
    def __init__(self, path):
        self.path, self.offset, self.pending = path, 0, b""
        self.identity = None
        self.recent = deque(maxlen=5)
        self.seen = deque(maxlen=256)
        while True:
            previous = self.offset
            self.poll()
            if self.offset == previous:
                break

    def poll(self):
        notices = []
        try:
            with self.path.open("rb") as f:
                stat = os.fstat(f.fileno())
                identity = (stat.st_dev, stat.st_ino)
                if identity != self.identity or stat.st_size < self.offset:
                    self.offset, self.pending, self.identity = 0, b"", identity
                f.seek(self.offset)
                data = f.read(262144)
                self.offset = f.tell()
        except OSError:
            return notices
        lines = (self.pending + data).split(b"\n")
        self.pending = lines.pop()
        if len(self.pending) > 1048576:
            self.pending = b""
        for line in lines:
            try:
                notice = memory_notice(json.loads(line))
            except (ValueError, TypeError, AttributeError):
                continue
            if notice["id"] not in self.seen:
                self.seen.append(notice["id"])
                self.recent.append(notice)
                notices.append(notice)
        return notices


@app.get("/api/push")
async def push_stream(request: Request):
    q: asyncio.Queue = asyncio.Queue()
    _subscribers.add(q)

    async def stream():
        try:
            memories = await asyncio.to_thread(MemoryTail, MEMORY_JSONL)
            yield f"data: {json.dumps({'type': 'hello', 'memories': list(memories.recent)})}\n\n"
            while not await request.is_disconnected():
                try:
                    yield f"data: {json.dumps(await asyncio.wait_for(q.get(), 1))}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                for notice in await asyncio.to_thread(memories.poll):
                    yield f"data: {json.dumps({'type': 'memory_saved', **notice})}\n\n"
        finally:
            _subscribers.discard(q)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/push")
async def push_send(body: dict):
    """Push an arbitrary event to every connected phone page. For testing and for
    the reminder scheduler. body = {"type": ..., "text": ...}"""
    for q in list(_subscribers):
        q.put_nowait(body)
    return {"delivered": len(_subscribers)}


# --------------------------------------------------------------------------
# Static: the agent page, the camera page, family photos, and pipeline frames.
# --------------------------------------------------------------------------
@app.get("/")
async def agent_page():
    return FileResponse(PHONE / "agent.html")


@app.get("/camera")
async def camera_page():
    return FileResponse(PHONE / "index.html")


@app.get("/photos/{name}")
async def photo(name: str):
    """Family photos live in server/photos as <firstname>.jpg, case-insensitive."""
    for p in (HERE / "photos").glob("*"):
        if p.stem.lower() == name.lower():
            return FileResponse(p)
    return JSONResponse({"error": f"no photo of {name}"}, 404)


@app.get("/frames/{path:path}")
async def frame(path: str):
    """Serve a pipeline event frame (AFTER image) for find_object results.
    Frame paths are written relative to wherever the pipeline ran — try the repo
    root and the perception/ dir, and never leave either."""
    for base in (ROOT, MEMORY_JSONL.parents[2]):  # memory.jsonl lives at perception/runs/<run>/;
        p = (base / path).resolve()
        if str(p).startswith(str(base)) and p.is_file():
            return FileResponse(p)
    return JSONResponse({"error": "not found"}, 404)


# --------------------------------------------------------------------------
# find_object: the core feature. Newest 'placed' memory wins; the AFTER frame is
# returned so the page can show a photo of where the thing actually is.
# --------------------------------------------------------------------------
def _ago(logged_at: str) -> str:
    """'20 minutes ago' - how a person refers to a time, not an ISO stamp."""
    try:
        delta = datetime.now() - datetime.fromisoformat(logged_at)
    except (TypeError, ValueError):
        return ""
    mins = int(delta.total_seconds() // 60)
    if mins < 1:
        return "just now"
    if mins < 60:
        return f"{mins} minute{'s' if mins != 1 else ''} ago"
    hours = mins // 60
    if hours < 24:
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    days = hours // 24
    return f"{days} day{'s' if days != 1 else ''} ago"


@app.get("/api/find")
async def find(q: str):
    """Every distinct place the item has been seen, newest first.

    There can genuinely be several - a household with medication in the kitchen and by
    the bed - and reporting only the newest would send someone to the wrong room. What
    we cannot tell them is whether that means two bottles or one that moved: there is no
    instance identity behind these memories. So the wording is "I've seen it in N
    places", with a time against each, and the person decides.
    """
    from es import search_all
    mems, source = search_all(q, MEMORY_JSONL, limit=3)
    if not mems:
        return {"say": f"I'm sorry, I didn't see where your {q} went."}

    name = mems[0].get("object", q)
    # The VLM capitalises its descriptions; these get spoken mid-sentence, where
    # "Also On the seat..." sounds wrong. Leave acronyms (FRAGMENT) alone.
    def lower_first(t):
        return t[0].lower() + t[1:] if len(t) > 1 and t[1].islower() else t
    places = [(lower_first(m.get("location_description") or "somewhere nearby"),
               _ago(m.get("logged_at", ""))) for m in mems]
    if len(places) == 1:
        where, when = places[0]
        say = f"Your {name} is {where}." + (f" I saw it {when}." if when else "")
    else:
        first, rest = places[0], places[1:]
        say = (f"I've seen your {name} in {len(places)} places. "
               f"Most recently {first[0]}" + (f", {first[1]}" if first[1] else "") + ". "
               + " ".join(f"Also {w}" + (f", {t}" if t else "") + "." for w, t in rest))

    out = {"say": say,
           "card": {"title": name,
                    "body": "\n".join(f"{w}" + (f"  ({t})" if t else "") for w, t in places)},
           "places": [{"where": w, "when": t} for w, t in places],
           "source": source}
    frame = mems[0].get("after_frame") or (mems[0].get("frames") or [None])[-1]
    if frame:
        out["card"]["image"] = "/frames/" + str(frame).replace("\\", "/")
    return out


@app.post("/api/es/index")
async def es_index(mem: dict):
    """vlm.py dual-writes each memory here; ES indexes it. No-op without ES."""
    from es import index_memory
    return {"indexed": index_memory(mem)}


@app.post("/api/es/sync")
async def es_sync():
    """Backfill ES from memory.jsonl (testing + recovery)."""
    from es import index_memory, client
    if not client():
        return {"indexed": 0, "error": "ELASTICSEARCH_URL not set"}
    n = 0
    if MEMORY_JSONL.exists():
        for line in MEMORY_JSONL.read_text().splitlines():
            try:
                if index_memory(json.loads(line)):
                    n += 1
            except json.JSONDecodeError:
                continue
    return {"indexed": n}


# --------------------------------------------------------------------------
# Reminders: jsonl store + a scheduler that pushes due ones over SSE. The page
# injects them into the live agent session, so Pam speaks them unprompted.
# --------------------------------------------------------------------------
def _read_reminders():
    if not REMINDERS.exists():
        return []
    out = []
    for line in REMINDERS.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _due_ts(body):
    """in_minutes wins; else parse `at` ("14:30", "2:30 pm", "tomorrow 9am")."""
    if body.get("in_minutes") is not None:
        return time.time() + float(body["in_minutes"]) * 60
    if body.get("at"):
        from dateutil import parser
        dt = parser.parse(str(body["at"]), fuzzy=True)
        if dt.tzinfo is None and dt.timestamp() < time.time() and ":" in str(body["at"]):
            dt = parser.parse(str(body["at"]) + " tomorrow", fuzzy=True)
        return dt.timestamp()
    return None


@app.get("/api/reminders")
async def list_reminders():
    active = [r for r in _read_reminders() if not r.get("fired")]
    if not active:
        return {"say": "You don't have any reminders right now.", "reminders": []}
    lines = [f"{r['text']} at {time.strftime('%I:%M %p', time.localtime(r['due_ts'])).lstrip('0')}"
             for r in active]
    return {"say": "You have " + ("; ".join(lines) if len(lines) < 4 else f"{len(lines)} reminders") + ".",
            "reminders": active}


@app.post("/api/reminders")
async def set_reminder(body: dict):
    due = _due_ts(body)
    if not due:
        return {"say": "I didn't catch the time for that reminder."}
    r = {"id": int(time.time() * 1000), "text": body["text"], "due_ts": due,
         "created_at": time.time(), "fired": False}
    with open(REMINDERS, "a") as f:
        f.write(json.dumps(r) + "\n")
    when = time.strftime("%I:%M %p", time.localtime(due)).lstrip("0")
    return {"say": f"Okay, I'll remind you to {r['text']} at {when}."}


# --------------------------------------------------------------------------
# Weather: Open-Meteo, no key. Calendar: any published .ics feed — Google
# Calendar's "secret address in iCal format" works, or drop a demo.ics in server/.
# --------------------------------------------------------------------------
WMO = {0:"clear", 1:"mostly clear", 2:"partly cloudy", 3:"overcast", 45:"foggy", 48:"foggy",
       51:"light drizzle", 53:"drizzle", 55:"heavy drizzle", 61:"light rain", 63:"rain", 65:"heavy rain",
       66:"freezing rain", 67:"freezing rain", 71:"light snow", 73:"snow", 75:"heavy snow",
       77:"snow grains", 80:"light showers", 81:"showers", 82:"heavy showers", 95:"thunderstorms"}


@app.get("/api/weather")
async def weather():
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get("https://api.open-meteo.com/v1/forecast", timeout=8, params={
                "latitude": HOME["lat"], "longitude": HOME["lon"],
                "current": "temperature_2m,weather_code", "temperature_unit": "fahrenheit"})
        cur = r.json()["current"]
        cond = WMO.get(cur["weather_code"], "calm")
        return {"say": f"Right now it's {round(cur['temperature_2m'])} degrees and {cond} outside."}
    except Exception:
        return {"say": "I couldn't reach the weather service just now."}


@app.get("/api/calendar")
async def calendar():
    src = os.environ.get("CALENDAR_ICS_URL")
    demo = HERE / "demo.ics"
    try:
        if src:
            async with httpx.AsyncClient() as c:
                text = (await c.get(src, timeout=10)).text
        elif demo.exists():
            text = demo.read_text()
        else:
            return {"say": "Your calendar isn't connected yet.", "events": []}
        from icalendar import Calendar
        import datetime
        cal = Calendar.from_ical(text)
        today = datetime.date.today()
        events = []
        for ev in cal.walk("VEVENT"):
            dt = ev.get("DTSTART").dt
            d = dt.date() if hasattr(dt, "date") else dt
            if d != today:
                continue
            at = dt.strftime("%I:%M %p").lstrip("0") if hasattr(dt, "strftime") else "all day"
            events.append({"time": at, "title": str(ev.get("SUMMARY", "event")),
                           "_dt": str(dt)})
        events.sort(key=lambda e: e["_dt"])
        if not events:
            return {"say": "Nothing on the calendar today — a free day.", "events": []}
        return {"say": "Today: " + "; ".join(f"{e['title']} at {e['time']}" for e in events),
                "events": events}
    except Exception as e:
        return {"say": "I couldn't read your calendar just now.", "error": str(e)}


# --------------------------------------------------------------------------
# Contacts: call, text, caregiver. Twilio = silent send / inbound bridge; without
# it the page shows a confirm card over a tel:/sms: link — the tap is the consent.
# --------------------------------------------------------------------------
def contact(name):
    if name == "__caregiver__":
        return CONTACTS["caregiver"]
    for c in CONTACTS["contacts"]:
        if c["name"].lower().split(".")[-1].startswith(name.lower().split()[0]):
            return c
    return None


def twilio():
    sid, tok, frm = (os.environ.get(k) for k in ("TWILIO_SID", "TWILIO_TOKEN", "TWILIO_FROM"))
    if not all((sid, tok, frm)):
        return None
    from twilio.rest import Client
    return Client(sid, tok), frm


@app.post("/api/message")
async def message(body: dict):
    c = contact(body.get("to", ""))
    if not c:
        return {"say": f"I don't have a contact called {body.get('to')}."}
    msg = body.get("message", "")
    tw = twilio()
    if tw:
        try:
            client, frm = tw
            client.messages.create(to=c["phone"], from_=frm, body=f"Pam (for {CONTACTS.get('user','the user')}): {msg}")
            return {"say": f"Done — I texted {c['name']} for you."}
        except Exception as e:
            return {"say": f"The text didn't go through: {e}"}
    from urllib.parse import quote
    return {"say": f"Tap the button to send that to {c['name']}.",
            "card": {"title": f"Text {c['name']}", "body": f"\"{msg}\"",
                     "action": {"label": f"Send to {c['name']}", "href": f"sms:{c['phone']}&body={quote(msg)}"}}}


@app.get("/api/caregiver-card")
async def caregiver_card():
    c = CONTACTS["caregiver"]
    return {"say": f"Tap Call {c['name']} to open your phone's dialer.",
            "card": {"title": f"Contact {c['name']}", "body": "Your caregiver. The call starts only after you confirm on your phone.",
                     "action": {"label": f"Call {c['name']}", "href": f"tel:{c['phone']}"}}}


@app.post("/api/call")
async def call(body: dict):
    c = contact(body.get("name", ""))
    if not c:
        return {"say": f"I don't have a contact called {body.get('name')}."}
    tw, user_phone = twilio(), os.environ.get("USER_PHONE")
    if tw and user_phone:
        try:  # ring the user's own phone, then bridge to the contact — they just answer
            client, frm = tw
            client.calls.create(to=user_phone, from_=frm, twiml=(
                f"<Response><Say>Pam here, connecting you to {c['name']}.</Say>"
                f"<Dial>{c['phone']}</Dial></Response>"))
            return {"say": f"Your phone will ring in a moment — answer it and I'll connect you to {c['name']}."}
        except Exception as e:
            return {"say": f"I couldn't place the call: {e}"}
    return {"say": f"Tap the button to call {c['name']}.",
            "card": {"title": f"Call {c['name']}", "body": c.get("relation", ""),
                     "action": {"label": f"Call {c['name']}", "href": f"tel:{c['phone']}"}}}


# --------------------------------------------------------------------------
# Ride: saved places -> prefilled m.uber.com link. The user confirms inside Uber.
# --------------------------------------------------------------------------
@app.post("/api/ride")
async def ride(body: dict):
    dest = (body.get("destination") or "").lower()
    place = next((p for k, p in CONTACTS["places"].items() if k in dest or dest in k), None)
    if not place:
        known = ", ".join(CONTACTS["places"])
        return {"say": f"I'm not sure where that is. I know: {known}."}
    from urllib.parse import quote
    href = (f"https://m.uber.com/ul/?action=setPickup"
            f"&pickup[latitude]={HOME['lat']}&pickup[longitude]={HOME['lon']}&pickup[nickname]=Home"
            f"&dropoff[latitude]={place['lat']}&dropoff[longitude]={place['lon']}"
            f"&dropoff[nickname]={quote(dest.title())}&dropoff[formatted_address]={quote(place['address'])}")
    return {"say": f"I've set up a ride to {place['address']}. Tap the button, then confirm in Uber.",
            "card": {"title": f"Ride to {dest.title()}", "body": place["address"],
                     "action": {"label": "Open Uber — ride is filled in", "href": href}}}


@app.get("/api/photo-info")
async def photo_info(name: str):
    c = contact(name)
    caption = f"{c['name']}, your {c['relation']}" if c else name
    has = any(p.stem.lower() == name.lower() for p in (HERE / "photos").glob("*"))
    if not has:
        return {"say": f"I don't have a photo of {name} yet."}
    return {"say": f"Here's {caption}.", "photo": {"url": f"/photos/{name}", "caption": caption}}


# --------------------------------------------------------------------------
# Flights: Amadeus test API when keys exist, else straight to a prefilled
# Google Flights link — either way the page shows a card.
# --------------------------------------------------------------------------
IATA = {"new york": "JFK", "boston": "BOS", "san francisco": "SFO", "los angeles": "LAX",
        "chicago": "ORD", "miami": "MIA", "london": "LHR", "paris": "CDG", "seattle": "SEA",
        "washington": "DCA", "denver": "DEN", "austin": "AUS", "atlanta": "ATL"}


@app.get("/api/flights")
async def flights(destination: str, date: str = ""):
    code = IATA.get(destination.lower(), destination.upper() if len(destination) == 3 else None)
    from urllib.parse import quote
    home_airport = CONTACTS["home_airport"]
    gfl = f"https://www.google.com/travel/flights?q={quote(f'flights from {home_airport} to {destination} {date}')}"
    if not code:
        return {"say": f"I'm not sure which airport that is — here's a search to pick from.",
                "card": {"title": f"Flights to {destination}", "action": {"label": "See flights", "href": gfl}}}
    key, sec = os.environ.get("AMADEUS_KEY"), os.environ.get("AMADEUS_SECRET")
    if key and sec:
        try:
            from dateutil import parser
            d = parser.parse(date or "tomorrow", fuzzy=True).date().isoformat()
            async with httpx.AsyncClient() as c:
                tok = (await c.post("https://test.api.amadeus.com/v1/security/oauth2/token",
                                    data={"grant_type": "client_credentials",
                                          "client_id": key, "client_secret": sec})).json()["access_token"]
                r = await c.get("https://test.api.amadeus.com/v2/shopping/flight-offers",
                                headers={"Authorization": f"Bearer {tok}"},
                                params={"originLocationCode": home_airport,
                                        "destinationLocationCode": code, "departureDate": d,
                                        "adults": 1, "max": 3})
                offers = r.json().get("data", [])
            if offers:
                lines = [f"{o['itineraries'][0]['segments'][0]['carrierCode']} — ${o['price']['total']}"
                         for o in offers]
                return {"say": f"I found {len(offers)} flights to {destination} on {d}, from ${offers[0]['price']['total']}. They're on your screen.",
                        "card": {"title": f"Flights to {destination} — {d}", "body": "  ·  ".join(lines),
                                 "action": {"label": "See them on Google Flights", "href": gfl}}}
        except Exception:
            pass
    return {"say": f"Here's a flight search for {destination} — tap the button to see options and prices.",
            "card": {"title": f"Flights to {destination}", "action": {"label": "See flights", "href": gfl}}}


# --------------------------------------------------------------------------
# Arm: fetch goes to the VLA policy server (vla/arm_client.py), if it's up.
# --------------------------------------------------------------------------
@app.post("/api/fetch")
async def fetch_item(body: dict):
    try:
        sys.path.insert(0, str(ROOT))
        from vla.arm_client import fetch
        status = fetch(os.environ.get("ARM_URL", "http://127.0.0.1:8020"))
        return {"say": "I'm getting it for you — the arm is on its way." if status["active"]
                else f"I can't start the arm right now: {status.get('last_error', 'it says no')}"}
    except Exception:
        return {"say": "I can't reach the arm right now — but I remember where it is if that helps."}


# --------------------------------------------------------------------------
# Fake Deepgram for end-to-end page testing without an API key. agent.html takes
# ?dg=ws://127.0.0.1:8000/api/fake-dg — this speaks just enough of the wire
# protocol (Welcome -> SettingsApplied -> FunctionCallRequest) to prove the whole
# client path: auth, settings, function dispatch, responses, injected reminders.
# --------------------------------------------------------------------------
from fastapi import WebSocket, WebSocketDisconnect
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed, InvalidHandshake


async def open_camera_relay():
    url = os.environ.get("CAMERA_RELAY_URL", "wss://127.0.0.1:8765")
    parsed = urlsplit(url)
    if parsed.hostname not in ("127.0.0.1", "localhost", "::1") or parsed.scheme not in ("ws", "wss"):
        raise ValueError("Camera relay must be a local WebSocket")
    options = {"ssl": ssl.create_default_context(cafile=str(CERT))} if parsed.scheme == "wss" else {}
    return await connect(url, open_timeout=4, close_timeout=1, max_size=2**20,
                         compression=None, proxy=None, **options)


@app.websocket("/api/camera")
async def camera_stream(ws: WebSocket):
    origin = ws.headers.get("origin")
    if origin and urlsplit(origin).netloc != ws.headers.get("host"):
        await ws.close(code=1008)
        return
    await ws.accept()
    relay, tasks = None, []
    try:
        relay = await open_camera_relay()
        await ws.send_json({"type": "camera_ready"})

        async def forward():
            count = 0
            while True:
                frame = await ws.receive_bytes()
                if len(frame) > 2**20:
                    await ws.close(code=1009)
                    return
                await relay.send(frame)
                count += 1
                if count == 1 or count % 20 == 0:
                    await ws.send_json({"type": "frame_received", "count": count})

        tasks = [asyncio.create_task(forward()), asyncio.create_task(relay.wait_closed())]
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
    except ssl.SSLCertVerificationError:
        with suppress(RuntimeError, WebSocketDisconnect):
            await ws.send_json({"type": "camera_error", "message": "The camera relay certificate does not match Pam's certificate. Ask your helper to restart the pipeline with phone/cert.pem."})
    except (OSError, TimeoutError, ValueError, ConnectionClosed, InvalidHandshake):
        with suppress(RuntimeError, WebSocketDisconnect):
            await ws.send_json({"type": "camera_error", "message": "Camera is on, but the memory pipeline is unavailable. Ask your helper to start the pipeline on the laptop. Pam will retry automatically."})
    except WebSocketDisconnect:
        pass
    finally:
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if relay:
            await relay.close()
        with suppress(RuntimeError, WebSocketDisconnect):
            await ws.close()


@app.websocket("/api/fake-dg")
async def fake_dg(ws: WebSocket):
    await ws.accept(subprotocol="token")
    await ws.send_json({"type": "Welcome"})
    settings = await ws.receive_json()
    assert settings["type"] == "Settings", settings
    await ws.send_json({"type": "SettingsApplied"})
    # Immediately exercise one client-side function the way the real agent would.
    await ws.send_json({"type": "FunctionCallRequest", "functions": [
        {"id": "fc_test_1", "name": "get_weather", "arguments": "{}", "client_side": True}]})
    try:
        while True:
            msg = await ws.receive()
            if msg.get("text"):
                m = json.loads(msg["text"])
                if m["type"] == "FunctionCallResponse":
                    await ws.send_json({"type": "ConversationText", "role": "assistant",
                                        "content": f"Function result received: {m['content'][:120]}"})
                elif m["type"] == "InjectUserMessage":
                    await ws.send_json({"type": "ConversationText", "role": "assistant",
                                        "content": f"(Pam would respond to: {m['content'][:80]})"})
            # binary = mic audio; a real test proves frames flow by counting them
    except Exception:
        pass


_fired: set[int] = set()


async def reminder_loop():
    while True:
        await asyncio.sleep(10)
        now = time.time()
        changed = False
        for r in _read_reminders():
            if not r.get("fired") and r["id"] not in _fired and r["due_ts"] <= now:
                r["fired"] = _fired.add(r["id"]) or True
                changed = True
                for q in list(_subscribers):
                    q.put_nowait({"type": "say",
                                  "text": f"A reminder is due now: {r['text']}. Please tell the user warmly."})
        if changed:
            REMINDERS.write_text("".join(json.dumps(r) + "\n" for r in _read_reminders()))


@app.get("/api/health")
async def health():
    return {"ok": True, "contacts": len(CONTACTS["contacts"]),
            "memory": MEMORY_JSONL.exists(), "es": bool(os.environ.get("ELASTICSEARCH_URL")),
            "twilio": bool(os.environ.get("TWILIO_SID")), "deepgram": bool(os.environ.get("DEEPGRAM_API_KEY"))}


# --------------------------------------------------------------------------
# Entry: HTTPS :8443 for the phone (same cert as serve.py generates), plus plain
# HTTP :8000 on localhost for laptop-side testing — a secure context either way.
# --------------------------------------------------------------------------
def ensure_cert():
    if CERT.exists() and KEY.exists():
        return
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80)); ip = s.getsockname()[0]
    except OSError:
        ip = "127.0.0.1"
    finally:
        s.close()
    print(f"generating a self-signed certificate for {ip} ...")
    subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
                    "-keyout", str(KEY), "-out", str(CERT), "-days", "365",
                    "-subj", "/CN=compass-laptop",
                    "-addext", f"subjectAltName=IP:{ip},IP:127.0.0.1,DNS:localhost"], check=True)


def main():
    import uvicorn
    ensure_cert()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    https = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=8443,
                                          ssl_certfile=str(CERT), ssl_keyfile=str(KEY), loop="none"))
    http = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=8000, loop="none"))
    loop.create_task(reminder_loop())
    import socket
    ip = socket.gethostbyname(socket.gethostname())
    print(f"\n  Phone:  https://{ip}:8443/        (Pam — voice agent)")
    print(f"          https://{ip}:8443/camera  (camera stream)")
    print(f"  Laptop: http://127.0.0.1:8000/  (testing)\n")
    loop.run_until_complete(asyncio.gather(https.serve(), http.serve()))


if __name__ == "__main__":
    main()
