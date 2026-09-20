# Pam — runbook

## Start everything

```bash
# 1. the agent backend + phone pages (laptop, perception/.venv)
python server/app.py
#    phone:  https://<laptop-ip>:8443/       Pam voice page
#            https://<laptop-ip>:8443/camera camera stream
#    laptop: http://127.0.0.1:8000/          same page, local
#            http://127.0.0.1:8000/?demo=1   function test panel (no voice needed)

# 2. the perception pipeline (separate terminal)
cd perception
python memory_pipeline.py --source wss://0.0.0.0:8765 \
    --cert ../phone/cert.pem --key ../phone/key.pem --out runs/live
#    wheelchair mount: NO --static-camera, keep arm logic, prefer --fps 5 --imgsz 480
#    bench test:       --source 0 --static-camera --no-arm
```

On the phone: open the https URL once, accept the cert warning, also visit
`https://<laptop-ip>:8443/api/health` once and accept it there — then WSS and the
page are both trusted. Mic permission prompt appears once inside "Talk to Pam".

## Environment

`server/.env` (see `.env.example`): `DEEPGRAM_API_KEY` is the only required key —
console.deepgram.com, Member role. Optional: `ELASTICSEARCH_URL`, `SERPAPI_KEY`,
`GOOGLE_CALENDAR_*`, `CAREGIVER_PIN`, `ARM_URL`, `HOME_LAT/LON`. `perception/.env` needs
`ANTHROPIC_API_KEY` for the VLM.

## First run: http://127.0.0.1:8000/setup

Start `server/app.py` and it opens this page by itself when anything is still missing. It
collects the two credentials that must belong to your own accounts, writes them to the
gitignored `server/.env` (owner-only), applies them to the running process, and sends the
browser into Google's consent screen. No file editing and no restart.

**Google Calendar** (read-only; a Google Maps key does *not* authorize Calendar):
enable the Calendar API, add yourself as a test user on the consent screen, create a
**Web application** OAuth client with redirect URI
`http://127.0.0.1:8000/api/calendar/google/callback`, then paste the client ID and secret.
Approve Google's page once; the stored refresh token keeps it connected across restarts, so
later startups just print `Setup: Google Calendar connected`.

**Flight search**: paste a free SerpApi key. It is checked against the provider's `/account`
endpoint before it is saved, so a typo fails on the page instead of mid-conversation.

The setup routes, like the OAuth routes, answer only on the laptop's loopback address: a
phone on the LAN gets 403. A saved secret is never rendered, returned or logged, and
`/api/setup/status` answers with booleans only. Pam starts and runs normally while either
service is unconnected; it says so out loud rather than inventing an answer.

Reminders need none of this. `set_reminder` writes to Pam's own `server/reminders.jsonl`,
`get_reminders` reads it back, and the scheduler speaks them through SSE. No Google, Apple
or iPhone reminder account is involved at any point.

Google expands recurring events; the query uses the calendar's own timezone and
handles all-day events and pagination. Missing access and provider failures are
reported explicitly, never as an empty/free day. `demo.ics` is no longer a silent
fallback. The legacy `CALENDAR_ICS_URL` path supports simple non-recurring feeds;
use Google sign-in for recurring calendars.

Credentials are stored outside the repository at
`%LOCALAPPDATA%\Pam\google-calendar.dat` on Windows, encrypted with the current
Windows user's DPAPI key. On other systems the file is owner-readable only under
`~/.local/share/Pam/`. OAuth setup is restricted to the laptop's loopback URL.
Treat the Pam server as a single-user app on a trusted network, not a public service.

## Voice-first flight information

**Google has no public flights API.** QPX Express, its last developer-facing flight
feed, was retired in 2018 and never replaced; the only flight API Google publishes
now returns carbon-emission estimates, not fares. **Amadeus self-service was also
decommissioned on July 17, 2026** (its API hostnames no longer resolve), so the old
`AMADEUS_*` integration could never have returned data again and has been removed.

Pam now reads Google Flights results through **SerpApi's `google_flights` engine**:
sign up at serpapi.com, no card, free plan 250 searches/month, and paste the key into
`/setup` (or set `SERPAPI_KEY` by hand). `FLIGHT_CURRENCY` (default `USD`) sets the
quoted currency. Without a
key, Pam says aloud that flight data is unavailable rather than inventing fares or
handing off a search link. These are Google Flights' own indicative prices; the
bookable fare is whatever the airline or agent charges at checkout.

Searches are one-way, for one adult, from the saved `home_airport`. The spoken
function result includes the airline, airports, local departure/arrival dates and
times, stops, total price, and the returned currency. Pam presents one option at a
time. No flight-purchase endpoint is called. A missing travel date prompts a spoken
question instead of silently assuming tomorrow.

Uber handoff cards remain available to a helper, but their spoken
responses explain voice-only limitations without instructing the user to tap a
screen. Verbal approval is requested in the agent prompt; it is not a substitute
for server-side confirmation enforcement or an external app's required consent.
After backend/prompt changes, stop and restart the voice session to get the new
agent settings.

## Safe capability verification

```bash
perception/.venv/Scripts/python.exe server/test_capabilities.py -v
perception/.venv/Scripts/python.exe server/test_ui.py -v
perception/.venv/Scripts/python.exe server/test_capabilities.py --live-routing
```

The backend tests isolate reminders and mock delivery providers and the robot.
The live routing test uses real Deepgram but never executes requested functions.
Browser tests isolate their relay so synthetic camera frames cannot enter a real
perception run. Do not use the old live `?demo=1` action buttons as a harmless smoke
test: they can invoke configured providers and the robot.

Text messaging, contact calls, and caregiver calls have been removed, including
API routes, voice tools, phone controls, and provider configuration. Saved contacts
remain available for family-photo names and captions. The robot wrapper starts its
configured policy; it does not yet pass the requested item to that policy.

## Test ladder (each rung works without the next)

| Rung | Command | Proves |
|---|---|---|
| API | `curl localhost:8000/api/health` + `/api/find?q=meds` | backend + memory store |
| Page | `?demo=1` buttons, or `python server/test_e2e.py` | every function + card/photo UI |
| Voice | `python server/test_voice.py` | Deepgram key, Settings, function round-trip |
| Real | phone → Talk to Pam → "where is my medication" | the product |

## Contacts / places / photos

`server/contacts.json` — names, phone numbers, saved places (ride destinations),
caregiver. Photos go in `server/photos/<firstname>.jpg`. Keyterms for the STT are
built from this file, so new names are recognized correctly.

## Demo script beats

1. "Where is my medication?" → finds latest placed memory + photo of the spot
2. "Remind me to take my pills in one minute" → fires over SSE, Pam speaks unprompted
3. "What's on my calendar?" → linked Google Calendar, or an explicit not-connected message
4. "Get me a ride to the airport" → prepared Uber handoff; a helper completes the booking
5. "Find flights to New York" → spoken offers when configured, otherwise an honest spoken service-availability message
6. "Fetch my pill bottle" → the arm (if vla server is up)
