"""Reproduce the user's browser session against REAL Deepgram: load the page with
no ?dg override, click Talk, capture status/console for 15s. Needs the server
running and DEEPGRAM_API_KEY set. Fake mic feeds a tone; that's enough to prove
connect + Settings + agent audio."""
import json, os, subprocess, sys, tempfile, time, wave
from pathlib import Path
from playwright.sync_api import sync_playwright

spoken_input = "--spoken" in sys.argv
with tempfile.TemporaryDirectory(prefix="pam-speech-test-") as folder, sync_playwright() as p:
    flags = ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"]
    if spoken_input:
        raw, padded = Path(folder) / "speech.wav", Path(folder) / "microphone.wav"
        command = """Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.Rate = -1
$s.SetOutputToWaveFile($env:PAM_TEST_WAV)
$s.Speak('Where did I leave my pill bottle?')
$s.Dispose()"""
        subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
                       env={**os.environ, "PAM_TEST_WAV": str(raw)}, check=True, capture_output=True)
        with wave.open(str(raw), "rb") as source:
            params, speech = source.getparams(), source.readframes(source.getnframes())
        with wave.open(str(padded), "wb") as target:
            target.setparams(params)
            second = b"\0" * params.framerate * params.nchannels * params.sampwidth
            target.writeframes(second * 12 + speech + second * 60)
        flags.append(f"--use-file-for-fake-audio-capture={padded}")
    browser = p.chromium.launch(args=flags)
    page = browser.new_context(permissions=["microphone", "camera"]).new_page()
    page.route_web_socket("**/api/camera", lambda socket: socket.close())
    page.on("console", lambda m: print(f"[console:{m.type}] {m.text[:160]}"))
    audio = {"received": 0, "sent": 0, "after_result": 0}
    responses, spoken, calls = [], [], []
    def received(data):
        if isinstance(data, bytes):
            audio["received"] += len(data)
            if responses:
                audio["after_result"] += len(data)
            return
        message = json.loads(data)
        kind = message.get("type")
        if kind in ("Error", "Warning"):
            print(kind, message.get("code"), message.get("description"))
        elif kind == "FunctionCallRequest":
            for call in message["functions"]:
                calls.append(call["name"])
                print("Function requested:", call["name"], "client_side:", call.get("client_side"))
        elif kind == "ConversationText" and message.get("role") == "assistant":
            spoken.append(message.get("content", ""))
            print("Agent said:", message.get("content"))
    def sent(data):
        if isinstance(data, bytes):
            audio["sent"] += len(data)
        else:
            message = json.loads(data)
            if message.get("type") == "FunctionCallResponse":
                responses.append(message)
                print("Function answered:", message.get("name"), "characters:", len(message.get("content", "")))
    def socket_opened(ws):
        if ws.url != "wss://agent.deepgram.com/v1/agent/converse":
            return
        ws.on("framereceived", received)
        ws.on("framesent", sent)
    page.on("websocket", socket_opened)
    page.goto("http://127.0.0.1:8000/")
    page.click("#talk")
    for i in range(15):
        time.sleep(1)
        st = page.locator("#status").inner_text()
        cn = page.locator("#conn").inner_text()
        print(f"t+{i+1:2d}s  conn={cn!r}  status={st!r}")
        if "listening" in st.lower() or "error" in st.lower() or "denied" in st.lower():
            pass
    if page.locator("#transcript-open").is_visible():
        page.locator("#transcript-open").click()
    else:
        page.locator("#features-open").click()
        page.locator("#features-transcript").click()
    log = page.locator("#log").inner_text()
    page.locator('[data-close="transcript-dialog"]').click()
    print("--- transcript ---"); print(log[:800] or "(empty)")
    assert cn == "connected", (cn, st)
    assert "Pam:" in log, "No agent transcript"
    assert audio["received"] > 0, "No incoming TTS audio"
    assert audio["sent"] > 0, "No outgoing microphone audio"
    assert page.evaluate("player.ctx.state") == "running", "Playback context is suspended"
    print(f"Audio verified: {audio['sent']} microphone bytes sent, {audio['received']} TTS bytes received")
    if not spoken_input:
        spoken.clear()
        page.evaluate("ws.send(JSON.stringify({type: 'InjectUserMessage', content: 'Where did I leave my pill bottle?'}))")
    deadline = time.monotonic() + 50
    while time.monotonic() < deadline:
        page.wait_for_timeout(500)
        if responses and audio["after_result"] > 0 and any("chair" in line.lower() or "table" in line.lower() for line in spoken):
            break
    assert "find_object" in calls, "No real memory function request"
    assert responses, "Browser never sent the function result"
    assert page.locator("#card").is_visible(), "No memory card rendered"
    assert any("chair" in line.lower() or "table" in line.lower() for line in spoken), f"No spoken location after function result: {spoken}"
    assert audio["after_result"] > 0, "No TTS audio after the memory result"
    assert page.locator("#subtitle-window").is_visible(), "Agent subtitles are not visible"
    assert any(word in page.locator("#subtitle-text").text_content().lower() for word in ("chair", "table")), "Location is missing from agent subtitles"
    assert page.locator("#subtitle-window").evaluate("el => el.clientHeight <= parseFloat(getComputedStyle(el).lineHeight) * 5 + 1"), "Subtitles exceeded five lines"
    print(f"Memory answer and subtitles verified: {audio['after_result']} TTS bytes after the browser function response")
    page.click("#talk")
    assert page.locator("#conn").inner_text() == "offline"
    browser.close()
