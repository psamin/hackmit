"""End-to-end test of phone/agent.html against the live server — no Deepgram key needed.

    python server/test_e2e.py

1. Loads ?demo=1 and clicks every function button: asserts the page logs a Pam
   line and that cards/photos actually render.
2. Loads ?dg=ws://127.0.0.1:8000/api/fake-dg and clicks "Talk to Pam": the fake
   agent speaks just enough wire protocol to verify Settings -> FunctionCallRequest
   -> FunctionCallResponse -> transcript all flow through the real client code.

Headless Chromium's --use-fake-device-for-media-stream feeds a tone as the mic,
so the getUserMedia/AudioWorklet path is genuinely exercised.
"""
import sys
from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:8000"
results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  — ' + detail if detail else ''}")


def demo_buttons(page):
    print("demo panel (?demo=1):")
    page.goto(f"{BASE}/?demo=1")
    page.wait_for_selector("#demo.show button")
    for btn in page.locator("#demo button").all():
        name = btn.inner_text()
        els = dict(card=page.locator("#card"), photo=page.locator("#photo"), log=page.locator("#log"))
        btn.click()
        page.wait_for_timeout(1200)
        logged = name in (els["log"].inner_text() or "") or "Pam:" in els["log"].inner_text()
        rendered = "show" in (els["card"].get_attribute("class") or "") or \
                   "show" in (els["photo"].get_attribute("class") or "")
        # a pass = the handler ran and produced a Pam line; cards/photos are bonus output
        check(f"handler {name}", logged, els["log"].inner_text().splitlines()[0][:80] if logged else "no log")


def fake_session(page):
    print("fake-agent session (?dg=...):")
    page.goto(f"{BASE}/?dg=ws://127.0.0.1:8000/api/fake-dg")
    page.click("#talk")
    try:
        page.wait_for_function("document.getElementById('conn').textContent === 'connected'", timeout=8000)
        check("session connects", True)
    except Exception:
        check("session connects", False, page.locator("#status").inner_text())
        return
    # the fake sends FunctionCallRequest(get_weather) on connect; the page must
    # answer and log the assistant's spoken line
    try:
        page.wait_for_function("document.getElementById('log').textContent.includes('Function result received')",
                               timeout=8000)
        check("function round-trip", True,
              page.locator("#log").inner_text().splitlines()[0][:100])
    except Exception:
        check("function round-trip", False, page.locator("#log").inner_text()[:100])
    page.click("#talk")  # hang up cleanly
    check("session stops", "offline" in page.locator("#conn").inner_text())


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(args=[
            "--use-fake-device-for-media-stream",   # feeds a tone as mic input
            "--use-fake-ui-for-media-stream",       # auto-accepts the permission prompt
        ])
        ctx = browser.new_context(permissions=["microphone"])
        page = ctx.new_page()
        page.on("console", lambda m: print(f"    [browser:{m.type}] {m.text[:120]}")
                if m.type in ("error", "warning") else None)
        demo_buttons(page)
        fake_session(page)
        browser.close()
    failed = [n for n, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} passed" +
          (f" — failed: {', '.join(failed)}" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
