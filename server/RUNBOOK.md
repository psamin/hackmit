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
3. "What's on my calendar?" → demo.ics or a real Google Calendar feed
4. "Text Sarah that I love her" → card → tap → Messages prefilled (or Twilio sends)
5. "I need help" → call_caregiver → tel: card (or Twilio rings the phone)
6. "Get me a ride to the airport" → Uber opens fully filled in
7. "Show me flights to New York" → cards + Google Flights link
8. "Fetch my pill bottle" → the arm (if vla server is up)
