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
console.deepgram.com, Member role. Optional: `ELASTICSEARCH_URL`, `TWILIO_*`,
`AMADEUS_*`, `CALENDAR_ICS_URL`, `ARM_URL`, `HOME_LAT/LON`. `perception/.env` needs
`ANTHROPIC_API_KEY` for the VLM.

## Google Calendar sign-in

Calendar access is read-only. A Google Maps API key does not authorize Calendar.

1. Enable the Google Calendar API in your Google Cloud project.
2. Configure the Google consent screen and add your account as a test user if the app is in Testing mode.
3. Create a **Web application** OAuth client with this exact authorized redirect URI:
   `http://127.0.0.1:8000/api/calendar/google/callback`.
4. Set `GOOGLE_CALENDAR_CLIENT_ID` and `GOOGLE_CALENDAR_CLIENT_SECRET` in the
   gitignored `server/.env`. Never paste the secret or a private iCal URL into chat.
5. Install `server/requirements.txt` in the project venv and restart `server/app.py`.
6. On the laptop, open `http://127.0.0.1:8000/`, then Features → Check your calendar →
   Connect Google Calendar. Finish Google's consent flow in the new browser tab.
7. The phone can now ask for today's schedule from the linked primary calendar.

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

Calls/texts need real contacts; the checked-in numbers are demo placeholders.
Twilio sends/bridges immediately when configured, so it needs a separate real-user
confirmation step before enabling it for an elderly user. The robot wrapper starts
its configured policy; it does not yet pass the requested item to that policy.

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
4. "Text Sarah that I love her" → card → tap → Messages prefilled (or Twilio sends)
5. "I need help" → call_caregiver → tel: card (or Twilio rings the phone)
6. "Get me a ride to the airport" → Uber opens fully filled in
7. "Show me flights to New York" → cards + Google Flights link
8. "Fetch my pill bottle" → the arm (if vla server is up)
