import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent))
import app


class MemoryTests(unittest.TestCase):
    def test_existing_partial_duplicate_and_new_memories(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "memory.jsonl"
            first = {"event_id": 1, "logged_at": "2026-09-19T12:00:00", "object": "keys", "event": "placed", "confidence": .9, "location_description": "On the table"}
            path.write_text(json.dumps(first) + "\n", encoding="utf-8")
            tail = app.MemoryTail(path)
            self.assertEqual(len(tail.recent), 1)
            self.assertEqual(tail.poll(), [])
            second = {**first, "event_id": 2, "object": "glasses"}
            encoded = json.dumps(second)
            with path.open("a", encoding="utf-8") as f:
                f.write(encoded[:20])
            self.assertEqual(tail.poll(), [])
            with path.open("a", encoding="utf-8") as f:
                f.write(encoded[20:] + "\n" + json.dumps(first) + "\n")
            notices = tail.poll()
            self.assertEqual([n["object"] for n in notices], ["glasses"])
            self.assertEqual(tail.poll(), [])

    def test_absent_file_and_uncertain_observation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "memory.jsonl"
            tail = app.MemoryTail(path)
            self.assertEqual(tail.poll(), [])
            path.write_text(json.dumps({"object": "pill bottle", "event": "still_in_hand", "confidence": .9, "location_description": "On the chair"}) + "\n", encoding="utf-8")
            notice = tail.poll()[0]
            self.assertEqual(notice["title"], "Observation saved")
            self.assertNotIn("On the chair", notice["detail"])


class CameraTests(unittest.TestCase):
    def test_wrong_origin_explains_direct_address_without_opening_relay(self):
        with patch.object(app, "open_camera_relay") as relay:
            with TestClient(app.app) as client:
                with client.websocket_connect("/api/camera", headers={"origin": "http://preview.example:65466"}) as ws:
                    result = ws.receive_json()
                    self.assertEqual(result["type"], "camera_error")
                    self.assertIn("directly", result["message"])
                    self.assertFalse(result["retry"])
            relay.assert_not_called()

    def test_missing_relay_is_actionable(self):
        async def unavailable():
            raise OSError("not listening")
        with patch.object(app, "open_camera_relay", unavailable):
            with TestClient(app.app) as client:
                with client.websocket_connect("/api/camera") as ws:
                    result = ws.receive_json()
                    self.assertEqual(result["type"], "camera_error")
                    self.assertIn("pipeline", result["message"].lower())

    def test_frames_forward_only_after_relay_ready(self):
        frames = []

        class Relay:
            async def send(self, data):
                frames.append(data)
            async def wait_closed(self):
                await asyncio.Future()
            async def close(self):
                pass

        async def ready():
            return Relay()

        with patch.object(app, "open_camera_relay", ready):
            with TestClient(app.app) as client:
                with client.websocket_connect("/api/camera") as ws:
                    self.assertEqual(ws.receive_json()["type"], "camera_ready")
                    ws.send_bytes(b"\xff\xd8test-frame\xff\xd9")
                    self.assertEqual(ws.receive_json()["type"], "frame_received")
                    ws.close()
                    self.assertEqual(ws.receive()["type"], "websocket.close")
        self.assertEqual(frames, [b"\xff\xd8test-frame\xff\xd9"])


class CameraTLSTests(unittest.IsolatedAsyncioTestCase):
    async def test_verified_local_tls_relay(self):
        import ssl
        from websockets.asyncio.server import serve

        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(app.CERT, app.KEY)
        frames = []

        async def receive(ws):
            frames.append(await ws.recv())
            await ws.send("received")

        async with serve(receive, "127.0.0.1", 0, ssl=context) as server:
            url = f"wss://127.0.0.1:{server.sockets[0].getsockname()[1]}"
            with patch.dict("os.environ", {"CAMERA_RELAY_URL": url}):
                relay = await app.open_camera_relay()
                await relay.send(b"camera-frame")
                self.assertEqual(await relay.recv(), "received")
                await relay.close()
        self.assertEqual(frames, [b"camera-frame"])


class BrowserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import socket
        import threading
        import time
        import uvicorn
        from playwright.sync_api import sync_playwright

        cls.temp = tempfile.TemporaryDirectory(prefix="pam-ui-tests-")
        cls.memory = Path(cls.temp.name) / "memory.jsonl"
        cls.memory.write_text("", encoding="utf-8")
        cls.memory_patch = patch.object(app, "MEMORY_JSONL", cls.memory)
        cls.memory_patch.start()
        cls.relay_patch = patch.object(app, "open_camera_relay", side_effect=OSError("Isolated test relay"))
        cls.relay_patch.start()
        cls.calendar_patch = patch.object(app.google_calendar, "calendar_service", app.google_calendar.CalendarService(Path(cls.temp.name) / "calendar.dat"))
        cls.calendar_patch.start()
        cls.socket = socket.socket()
        cls.socket.bind(("127.0.0.1", 0))
        cls.base = f"http://127.0.0.1:{cls.socket.getsockname()[1]}"
        cls.server = uvicorn.Server(uvicorn.Config(app.app, log_level="error", timeout_graceful_shutdown=6))
        cls.thread = threading.Thread(target=cls.server.run, kwargs={"sockets": [cls.socket]}, daemon=True)
        cls.thread.start()
        for _ in range(100):
            if cls.server.started:
                break
            time.sleep(.05)
        cls.pw = sync_playwright().start()
        cls.browser = cls.pw.chromium.launch(args=["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"])
        cls.artifacts = Path(tempfile.mkdtemp(prefix="pam-ui-captures-"))

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.pw.stop()
        cls.server.should_exit = True
        cls.thread.join(5)
        cls.socket.close()
        cls.memory_patch.stop()
        cls.relay_patch.stop()
        cls.calendar_patch.stop()
        cls.temp.cleanup()
        print(f"Screenshots: {cls.artifacts}")

    def setUp(self):
        self.memory.write_text("", encoding="utf-8")
        self.context = self.browser.new_context(permissions=["microphone", "camera"], viewport={"width": 1440, "height": 1000}, color_scheme="light")
        self.page = self.context.new_page()
        self.errors = []
        self.page.on("pageerror", lambda error: self.errors.append(str(error)))
        self.page.route("**/api/weather", lambda route: route.fulfill(json={"say": "It is 65 degrees and clear."}))

    def tearDown(self):
        self.context.close()
        self.assertEqual(self.errors, [])

    def test_google_calendar_setup_states(self):
        page = self.page
        state = {"configured": False, "connected": False, "can_connect_here": True}
        page.route("**/api/calendar/status", lambda route: route.fulfill(json=state))
        page.goto(self.base)
        page.locator("#features-open").click()
        page.wait_for_function("document.getElementById('calendar-connection-status').textContent.includes('OAuth client')")
        self.assertTrue(page.locator("#calendar-connect").is_hidden())
        state["configured"] = True
        page.evaluate("refreshCalendarStatus()")
        self.assertTrue(page.locator("#calendar-connect").is_visible())
        self.assertEqual(page.locator("#calendar-connect").get_attribute("href"), "/api/calendar/google/connect")
        self.assertEqual(page.locator("#calendar-connect").get_attribute("target"), "_blank")
        state["connected"] = True
        page.evaluate("refreshCalendarStatus()")
        self.assertIn("Read-only", page.locator("#calendar-connection-status").inner_text())
        state["can_connect_here"] = False
        page.evaluate("refreshCalendarStatus()")
        self.assertTrue(page.locator("#calendar-connect").is_hidden())

    def test_new_composition_and_real_activity(self):
        page = self.page
        page.goto(self.base)
        self.assertEqual(page.locator(".primary-nav").count(), 1)
        self.assertEqual(page.locator(".bottom-nav").count(), 0)
        self.assertEqual(page.locator(".call-room").count(), 1)
        self.assertEqual(page.locator(".voice-panel").count(), 0)
        for scheme in ("light", "dark"):
            page.emulate_media(color_scheme=scheme)
            self.assertEqual(page.evaluate("getComputedStyle(document.body).backgroundColor"), "rgb(255, 255, 255)")
        page.evaluate("showMemory({id:'new-layout',object:'keys',detail:'Keys on the kitchen table',logged_at:'2026-09-19T12:00:00'},false)")
        self.assertEqual(page.locator("#recent-list button").count(), 1)
        self.assertIn("kitchen table", page.locator("#recent-list").inner_text())
        page.locator("#recent-list button").click()
        self.assertTrue(page.locator("#memory-dialog").is_visible())
        self.assertIn("kitchen table", page.locator("#memory-selected-detail").inner_text())
        page.keyboard.press("Escape")
        self.assertTrue(page.locator("#view-talk").is_visible())
        self.assertTrue(page.locator(":focus").evaluate("el => el.closest('#recent-list') !== null"))
        self.assertTrue(page.locator("#camera-peek").is_visible())

    def test_memory_collection_and_history_navigation(self):
        page = self.page
        page.goto(self.base)
        page.evaluate("""[
          {id:'keys-old',object:'keys',title:'Memory saved',detail:'Keys on the kitchen table',logged_at:'2026-09-19T10:00:00'},
          {id:'keys-new',object:'keys',title:'Memory saved',detail:'Keys beside the front door',logged_at:'2026-09-19T12:00:00'},
          {id:'glasses-one',object:'glasses',title:'Memory saved',detail:'Glasses on the desk',logged_at:'2026-09-19T11:00:00'}
        ].forEach(m => showMemory(m, false))""")
        page.locator("#tab-memories").click()
        self.assertEqual(page.locator('#memory-history .memory-card').count(), 2)
        page.screenshot(path=str(self.artifacts / "memory-collection-desktop.png"))
        keys = page.locator('#memory-history [data-memory-object="keys"]')
        self.assertIn("2 observations", keys.inner_text())
        keys.click()
        self.assertTrue(page.locator("#memory-dialog").is_visible())
        self.assertIn("front door", page.locator("#memory-selected-detail").inner_text())
        self.assertTrue(page.locator("#memory-prev").is_disabled())
        page.locator("#memory-next").click()
        self.assertIn("kitchen table", page.locator("#memory-selected-detail").inner_text())
        self.assertTrue(page.locator("#memory-next").is_disabled())
        page.evaluate("showMemory({id:'keys-later',object:'keys',title:'Memory saved',detail:'Keys in the study',logged_at:'2026-09-19T14:00:00'}, false)")
        self.assertIn("kitchen table", page.locator("#memory-selected-detail").inner_text())
        page.keyboard.press("Escape")
        self.assertTrue(page.locator(":focus").evaluate("el => el.closest('#memory-history') !== null"))
        page.locator('[data-memory-view="timeline"]').click()
        self.assertEqual(page.locator("#memory-history .memory-card").count(), 4)
        page.screenshot(path=str(self.artifacts / "memory-timeline-desktop.png"))
        page.locator("#memory-search").fill("kitchen")
        self.assertEqual(page.locator("#memory-history .memory-card").count(), 1)
        page.locator("#memory-history .memory-card").click()
        self.assertIn("kitchen table", page.locator("#memory-selected-detail").inner_text())
        page.screenshot(path=str(self.artifacts / "memory-history-dialog.png"))
        page.keyboard.press("Escape")
        page.locator("#memory-search").fill("does-not-exist")
        self.assertIn("No memories match", page.locator("#memory-history").inner_text())
        page.locator("#memory-history .chip").click()
        page.locator('[data-memory-view="items"]').click()
        page.set_viewport_size({"width": 390, "height": 844})
        page.screenshot(path=str(self.artifacts / "memory-collection-phone.png"))
        page.locator("#memory-history .memory-card").first.click()
        page.screenshot(path=str(self.artifacts / "memory-history-phone.png"))
        page.set_viewport_size({"width": 320, "height": 568})
        page.evaluate("document.documentElement.dataset.large = 'true'")
        self.assertTrue(page.locator("#memory-dialog").evaluate("el => el.scrollWidth <= el.clientWidth + 1"))
        self.assertEqual(page.locator(".memory-pager").evaluate("el => getComputedStyle(el).gridTemplateColumns.split(' ').length"), 2)
        page.keyboard.press("Escape")
        self.assertTrue(page.evaluate("document.documentElement.scrollWidth <= innerWidth"))

    def test_memory_observation_is_not_presented_as_a_location(self):
        page = self.page
        requested_frames = []
        page.on("request", lambda request: requested_frames.append(request.url) if "/frames/" in request.url else None)
        page.goto(self.base)
        page.evaluate("showMemory({id:'uncertain',object:'pill bottle',title:'Observation saved',detail:'Its resting place is not confirmed.',logged_at:null,image:'/frames/not-requested.jpg'}, false)")
        page.locator("#tab-memories").click()
        page.locator("#memory-history .memory-card").click()
        self.assertIn("not confirmed", page.locator("#memory-selected-status").inner_text())
        self.assertIn("not recorded", page.locator("#memory-selected-time").inner_text())
        self.assertEqual(requested_frames, [])

    def test_larger_welcome_illustration(self):
        page = self.page
        page.goto(self.base)
        for width, height, minimum in ((1440, 1000, 340), (430, 932, 175), (390, 844, 160), (375, 812, 150), (320, 568, 130)):
            page.set_viewport_size({"width": width, "height": height})
            image = page.locator(".welcome-scene img").bounding_box()
            copy = page.locator(".welcome-copy").bounding_box()
            actions = page.locator(".actions").bounding_box()
            self.assertGreaterEqual(image["width"], minimum)
            self.assertLessEqual(copy["x"] + copy["width"], image["x"] + 1, f"{width}x{height}: image is not beside text")
            self.assertAlmostEqual(copy["y"] + copy["height"] / 2, image["y"] + image["height"] / 2, delta=1)
            self.assertLessEqual(image["y"] + image["height"], actions["y"] + 1)
            self.assertTrue(page.evaluate("document.documentElement.scrollWidth <= innerWidth"))
            page.screenshot(path=str(self.artifacts / f"welcome-horizontal-{width}x{height}.png"))
        page.evaluate("appendCaption('Hello, how can I help?')")
        self.assertTrue(page.locator(".welcome-scene img").is_hidden())

    def test_large_text_long_captions_remain_separate_from_controls(self):
        page = self.page
        page.goto(self.base)
        page.evaluate("document.documentElement.dataset.large = 'true'")
        for width, height in ((320, 568), (390, 844), (844, 390)):
            page.set_viewport_size({"width": width, "height": height})
            page.evaluate("onControl({type:'UserStartedSpeaking'}); onControl({type:'ConversationText',role:'assistant',content:'Your pill bottle is on the kitchen table. '.repeat(30)})")
            page.wait_for_timeout(120)
            bounds = page.evaluate("""() => {
              const caption = document.getElementById('subtitle-window').getBoundingClientRect();
              const controls = document.querySelector('.actions').getBoundingClientRect();
              return { captionBottom: caption.bottom, controlsTop: controls.top, overflow: document.documentElement.scrollWidth - innerWidth };
            }""")
            self.assertLessEqual(bounds["captionBottom"], bounds["controlsTop"] + 1, f"{width}x{height} captions overlap controls")
            self.assertEqual(bounds["overflow"], 0)
            page.locator("#talk").scroll_into_view_if_needed()
            self.assertTrue(page.locator("#talk").evaluate("el => { const r = el.getBoundingClientRect(); return document.elementFromPoint(r.x + r.width / 2, r.y + r.height / 2)?.closest('button') === el; }"))

    def test_rolling_captions_and_no_orb_overlap(self):
        page = self.page
        page.goto(self.base)
        self.assertEqual(page.locator("#subtitle-window").count(), 1)
        for width, height in ((1440, 1000), (390, 844), (320, 568), (844, 390)):
            page.set_viewport_size({"width": width, "height": height})
            page.evaluate("onControl({type:'UserStartedSpeaking'}); onControl({type:'ConversationText',role:'assistant',content:'I remember your pill bottle on the blue chair. '.repeat(24) + 'Final visible sentence.'})")
            page.wait_for_timeout(150)
            metrics = page.evaluate("""() => {
              const box = document.getElementById('subtitle-window'), text = document.getElementById('subtitle-text');
              const sphere = document.getElementById('orb-ring').getBoundingClientRect(), panel = document.querySelector('.visual-zone').getBoundingClientRect();
              const caption = box.getBoundingClientRect();
              return {height: box.clientHeight, line: parseFloat(getComputedStyle(box).lineHeight), atEnd: box.scrollHeight - box.clientHeight - box.scrollTop,
                      separate: sphere.bottom <= caption.top + 1 || sphere.right <= caption.left + 1,
                      contained: sphere.top >= panel.top - 1 && sphere.bottom <= panel.bottom + 1,
                      controlsFit: document.getElementById('talk').getBoundingClientRect().bottom <= document.getElementById('main').getBoundingClientRect().bottom + 1};
            }""")
            self.assertLessEqual(metrics["height"], metrics["line"] * 5 + 1)
            self.assertLessEqual(abs(metrics["atEnd"]), 1)
            self.assertTrue(metrics["separate"] and metrics["contained"], f"{width}x{height}: {metrics}")
            self.assertTrue(metrics["controlsFit"], f"{width}x{height}: subtitles pushed controls offscreen")
            page.screenshot(path=str(self.artifacts / f"subtitles-{width}x{height}.png"))
        page.evaluate("onControl({type:'UserStartedSpeaking'}); onControl({type:'ConversationText',role:'assistant',content:'A new answer.'})")
        self.assertEqual(page.locator("#subtitle-text").text_content(), "A new answer.")
        page.evaluate("stop()")
        self.assertEqual(page.locator("#subtitle-text").text_content(), "A new answer.")

    def test_features_dialog_lists_all_capabilities(self):
        page = self.page
        page.goto(self.base)
        self.assertEqual(page.locator("[data-help]").count(), 0)
        page.locator("#features-open").click()
        self.assertTrue(page.locator("#features-dialog").is_visible())
        self.assertEqual(set(page.locator("[data-capability]").evaluate_all("els => els.map(el => el.dataset.capability)")), {f["name"] for f in app.FUNCTIONS} | {"guide_me"})
        page.screenshot(path=str(self.artifacts / "features-desktop.png"))
        page.set_viewport_size({"width": 390, "height": 844})
        page.screenshot(path=str(self.artifacts / "features-phone.png"))
        page.locator("#feature-search").fill("weather")
        self.assertEqual(page.locator("[data-capability]:visible").count(), 1)
        page.keyboard.press("Escape")
        self.assertTrue(page.locator("#features-dialog").is_hidden())
        self.assertEqual(page.locator(":focus").get_attribute("id"), "features-open")
        page.locator("#features-open").click()
        page.locator("#features-transcript").click()
        self.assertTrue(page.locator("#transcript-dialog").is_visible())
        page.keyboard.press("Escape")
        self.assertEqual(page.locator(":focus").get_attribute("id"), "features-open")

    def test_saved_images_only_come_from_find_object(self):
        page = self.page
        page.route("**/api/find?*", lambda route: route.fulfill(json={"say": "On the chair.", "card": {"title": "pill bottle", "body": "On the chair", "image": "/photos/sarah"}}))
        page.goto(self.base)
        page.evaluate("showMemory({id:'image-test',object:'pill bottle',detail:'On the chair',image:'/photos/sarah'}, false)")
        self.assertTrue(page.locator("#card img").is_hidden())
        page.evaluate("runFunction({id:'find',name:'find_object',arguments:JSON.stringify({item:'pill bottle'})})")
        self.assertTrue(page.locator("#card img").is_visible(), page.evaluate("({card:document.getElementById('card').outerHTML, viewport:document.querySelector('.workspace').getBoundingClientRect().toJSON()})"))
        page.evaluate("runFunction({id:'weather',name:'get_weather',arguments:'{}'})")
        self.assertTrue(page.locator("#card img").is_hidden())
        page.evaluate("showCard({title:'Not a lookup',image:'/photos/sarah'})")
        self.assertTrue(page.locator("#card img").is_hidden())

    def test_persistent_camera_expands_without_restarting_capture(self):
        class Relay:
            async def send(self, data):
                pass
            async def wait_closed(self):
                await asyncio.Future()
            async def close(self):
                pass
        async def ready():
            return Relay()
        with patch.object(app, "open_camera_relay", ready):
            page = self.page
            page.add_init_script("const original = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices); window.cameraCalls = 0; navigator.mediaDevices.getUserMedia = options => { if (options.video) window.cameraCalls++; return original(options); };")
            page.goto(self.base)
            page.locator("#camera-peek").click()
            page.locator("#camera-dialog-start").click()
            page.wait_for_function("document.getElementById('camera-expanded').videoWidth > 0")
            self.assertTrue(page.evaluate("document.getElementById('camera-expanded').srcObject === document.getElementById('camera-preview').srcObject"))
            page.locator('[data-close="camera-dialog"]').click()
            for view in ("memories", "camera", "talk"):
                page.locator(f"#tab-{view}").click()
                self.assertTrue(page.locator("#camera-preview").is_visible())
                self.assertTrue(page.evaluate("camStream.getVideoTracks()[0].readyState === 'live'"))
            page.locator("#camera-peek").click()
            self.assertEqual(page.evaluate("window.cameraCalls"), 1)
            page.wait_for_function("document.getElementById('camera-expanded').readyState >= 2 && !document.getElementById('camera-expanded').paused && document.getElementById('camera-expanded').currentTime > .1")
            page.screenshot(path=str(self.artifacts / "camera-expanded.png"))
            page.keyboard.press("Escape")
            self.assertEqual(page.locator(":focus").get_attribute("id"), "camera-peek")
            page.set_viewport_size({"width": 390, "height": 844})
            page.screenshot(path=str(self.artifacts / "camera-mini-phone.png"))
            page.evaluate("stopCamera()")
            self.assertTrue(page.locator("#mini-camera-empty").is_visible())

    def test_layout_preferences_and_caregiver(self):
        page = self.page
        page.goto(self.base)
        page.screenshot(path=str(self.artifacts / "desktop.png"), full_page=True)
        for width in (320, 375, 768, 1024, 1440):
            page.set_viewport_size({"width": width, "height": 900})
            self.assertTrue(page.evaluate("document.documentElement.scrollWidth <= innerWidth"), str(width))
            bounds = page.locator("#talk").bounding_box()
            self.assertGreaterEqual(bounds["height"], 56)
            self.assertLess(bounds["y"] + bounds["height"], 900)
        page.set_viewport_size({"width": 375, "height": 812})
        page.screenshot(path=str(self.artifacts / "mobile.png"), full_page=True)
        page.locator(".settings summary").click()
        page.locator("#large-text").check()
        page.locator("#high-contrast").check()
        page.locator("#reduce-effects").check()
        page.keyboard.press("Escape")
        self.assertTrue(page.evaluate("document.documentElement.scrollWidth <= innerWidth"))
        page.reload()
        self.assertEqual(page.evaluate("document.documentElement.dataset.large"), "true")
        self.assertEqual(page.evaluate("document.documentElement.dataset.contrast"), "true")
        page.locator("#caregiver").click()
        page.wait_for_selector("#card.show a.action")
        self.assertTrue(page.locator("#card a.action").first.get_attribute("href").startswith("tel:"))
        page.evaluate("document.documentElement.removeAttribute('data-contrast'); document.documentElement.removeAttribute('data-large')")
        page.emulate_media(color_scheme="dark")
        page.screenshot(path=str(self.artifacts / "dark.png"), full_page=True)

    def test_logo_renders_from_png_mask(self):
        page = self.page
        response = page.request.get(f"{self.base}/assets/pam-logo.png")
        self.assertEqual(response.status, 200)
        self.assertTrue(response.body().startswith(b"\x89PNG"))
        page.goto(self.base)
        box = page.locator(".wordmark").bounding_box()
        self.assertGreater(box["width"], 60)
        self.assertGreater(box["height"], 20)
        mask = page.locator(".wordmark").evaluate("el => getComputedStyle(el).webkitMaskImage || getComputedStyle(el).maskImage")
        self.assertIn("pam-logo.png", mask)
        self.assertEqual(page.locator(".wordmark").get_attribute("aria-label"), "Pam")
        colour = page.locator(".wordmark").evaluate("el => getComputedStyle(el).backgroundColor")
        self.assertNotIn("255, 255, 255", colour, "logo would be white on white")

    def test_white_theme_single_accent_and_live_ring(self):
        page = self.page
        page.goto(self.base)
        self.assertEqual(page.evaluate("getComputedStyle(document.body).backgroundColor"), "rgb(255, 255, 255)")
        self.assertEqual(page.evaluate("getComputedStyle(document.body).backgroundImage"), "none")
        # Blue is signal only: the primary action, the logo and the active tab share one colour.
        talk_bg = page.locator("#talk").evaluate("el => getComputedStyle(el).backgroundColor")
        self.assertEqual(talk_bg, page.locator(".wordmark").evaluate("el => getComputedStyle(el).backgroundColor"))
        self.assertIn(page.locator("#greeting").inner_text(), ("Good morning.", "Good afternoon.", "Good evening."))
        # Mute is not offered until Pam is live; the ring is invisible until then.
        self.assertTrue(page.locator("#mic-toggle").is_hidden())
        self.assertEqual(page.locator("#orb-ring").evaluate("el => getComputedStyle(el).opacity"), "0")
        page.evaluate("setVoiceState('listening')")
        page.wait_for_timeout(300)
        self.assertGreater(float(page.locator("#orb-ring").evaluate("el => getComputedStyle(el).opacity")), .5)
        for theme in ("light", "dark"):
            page.emulate_media(color_scheme=theme)
            for view in ("talk", "memories", "camera"):
                page.locator(f"#tab-{view}").click()
                self.assertEqual(page.locator(f"#tab-{view}").get_attribute("aria-selected"), "true")
            page.locator("#tab-talk").click()
            page.wait_for_timeout(250)
            page.screenshot(path=str(self.artifacts / f"theme-{theme}.png"), full_page=True)

    def test_contrast_and_keyboard(self):
        page = self.page
        page.goto(self.base)
        page.keyboard.press("Tab")
        self.assertEqual(page.locator(":focus").inner_text(), "Skip to Pam")
        for theme in ("light", "dark"):
            page.emulate_media(color_scheme=theme)
            page.wait_for_timeout(250)  # let colour transitions finish before sampling
            contrasts = page.evaluate("""() => {
              const rgb = color => color.match(/[\\d.]+/g).slice(0, 3).map(Number);
              const lum = color => rgb(color).map(v => { v /= 255; return v <= .04045 ? v / 12.92 : ((v + .055) / 1.055) ** 2.4; }).reduce((sum, v, i) => sum + v * [.2126, .7152, .0722][i], 0);
              return [['#talk','#talk'], ['#status','body'], ['#memory-detail','#memory-note'], ['.privacy','.panel'], ['#tab-talk','#tab-talk'], ['#tab-camera','.primary-nav']].map(([text, bg]) => {
                const fg = lum(getComputedStyle(document.querySelector(text)).color);
                const back = lum(getComputedStyle(document.querySelector(bg)).backgroundColor);
                return [text, (Math.max(fg, back) + .05) / (Math.min(fg, back) + .05)];
              });
            }""")
            for selector, ratio in contrasts:
                self.assertGreaterEqual(ratio, 4.5, f"{theme} {selector}: {ratio}")
        page.set_viewport_size({"width": 375, "height": 812})
        greeting = page.locator("#greeting").bounding_box()
        orb = page.locator(".orb-scene").bounding_box()
        self.assertGreaterEqual(orb["y"], greeting["y"] + greeting["height"])

    def test_mobile_tabs_search_and_keyboard(self):
        page = self.page
        page.set_viewport_size({"width": 375, "height": 812})
        page.goto(self.base)
        page.locator("#tab-memories").click()
        self.assertTrue(page.locator("#view-talk").is_hidden())
        self.assertTrue(page.locator("#view-memories").is_visible())
        page.screenshot(path=str(self.artifacts / "memories-phone.png"), full_page=True)
        page.evaluate("showMemory({id:'test-keys',object:'keys',title:'Memory saved',detail:'Keys on the kitchen table', image:'/photos/sarah'}, false); showMemory({id:'test-glasses',object:'glasses',title:'Memory saved',detail:'Glasses by the window'}, false)")
        self.assertEqual(page.locator("#memory-history img").count(), 0, "saved-location photos are reserved for explicit object lookups")
        page.locator("#memory-search").fill("keys")
        self.assertEqual(page.locator("#memory-history button").count(), 1)
        page.locator("#memory-history button").click()
        self.assertTrue(page.locator("#memory-dialog").is_visible())
        self.assertIn("kitchen table", page.locator("#memory-selected-detail").inner_text())
        self.assertEqual(page.locator("#memory-dialog img").count(), 0, "browsing a memory doesn't trigger a location photo")
        page.keyboard.press("Escape")
        self.assertTrue(page.locator("#view-memories").is_visible())
        page.locator("#tab-talk").focus()
        page.keyboard.press("ArrowRight")
        self.assertEqual(page.locator(":focus").get_attribute("id"), "tab-memories")
        page.keyboard.press("End")
        self.assertTrue(page.locator("#view-camera").is_visible())
        page.screenshot(path=str(self.artifacts / "camera-phone.png"), full_page=True)

    def test_talk_screen_fits_common_phones_without_scrolling(self):
        for width, height in ((320, 568), (360, 640), (375, 667), (390, 844), (430, 932), (768, 1024)):
            context = self.browser.new_context(viewport={"width": width, "height": height}, is_mobile=width < 700, has_touch=True, device_scale_factor=2, color_scheme="light")
            page = context.new_page()
            try:
                page.goto(self.base)
                page.wait_for_timeout(120)
                label = f"{width}x{height}"
                self.assertTrue(page.evaluate("document.documentElement.scrollWidth <= innerWidth"), label)
                self.assertLessEqual(page.evaluate("document.getElementById('main').scrollHeight - document.getElementById('main').clientHeight"), 1, f"{label} talk screen scrolls")
                controls = ("#talk", "#features-open", "#camera-peek")
                boxes = {sel: page.locator(sel).bounding_box() for sel in ("header", "#greeting", ".orb-scene", ".primary-nav") + controls}
                header = boxes["header"]
                self.assertIsNone(page.locator("#mic-toggle").bounding_box(), f"{label} mute shown while offline")
                for sel in ("#camera-peek", "#talk"):
                    box = boxes[sel]
                    self.assertGreaterEqual(box["y"], header["y"] + header["height"] - 1, f"{label} {sel} overlaps header")
                    self.assertLessEqual(box["y"] + box["height"], height, f"{label} {sel} below viewport")
                for sel in controls:
                    box = boxes[sel]
                    self.assertGreaterEqual(box["height"], 54, f"{label} {sel} height")
                    self.assertGreaterEqual(box["x"], 0, label)
                    self.assertLessEqual(box["x"] + box["width"], width + .5, label)
                    self.assertFalse(page.locator(sel).evaluate("el => el.scrollWidth > el.clientWidth + 1"), f"{label} {sel} label clipped")
                # Once live, Mute appears beside Talk without wrapping or overlapping.
                page.evaluate("document.getElementById('mic-toggle').hidden = false")
                talk, mic = page.locator("#talk").bounding_box(), page.locator("#mic-toggle").bounding_box()
                self.assertLessEqual(talk["x"] + talk["width"], mic["x"] + .5, f"{label} talk/mute overlap")
                self.assertAlmostEqual(talk["y"], mic["y"], delta=1, msg=f"{label} mute wrapped to a new line")
                self.assertGreaterEqual(mic["height"], 54, label)
                page.evaluate("document.getElementById('mic-toggle').hidden = true")
                page.screenshot(path=str(self.artifacts / f"fit-{label}.png"))
            finally:
                context.close()

    def test_live_resize_reflows_without_reload(self):
        # A real browser window is dragged, not reloaded. Every size must settle
        # cleanly from the previous one, including landscape phones.
        page = self.page
        page.goto(self.base)
        page.evaluate("showMemory({id:'long',object:'pill bottle',title:'Memory saved',detail:'Pill bottle: on the seat of a blue folding chair, near a gray backpack on the floor next to it, in front of a white sign reading FRAGMENT and recycling bins'}, false)")
        sizes = [(1440, 1000), (1024, 700), (768, 1024), (600, 500), (430, 932), (390, 844), (844, 390), (375, 667), (1200, 560), (320, 568), (667, 375), (1440, 1000)]
        for width, height in sizes:
            page.set_viewport_size({"width": width, "height": height})
            page.wait_for_timeout(200)
            label = f"{width}x{height}"
            r = page.evaluate("""() => { const m = document.getElementById('main'); const nav = document.querySelector('.topbar').getBoundingClientRect();
              const talk = document.getElementById('talk').getBoundingClientRect(); const rem = document.getElementById('camera-peek').getBoundingClientRect();
              const orb = document.querySelector('.orb-scene').getBoundingClientRect(); const sub = document.getElementById('memory-peek-detail');
              return { hscroll: document.documentElement.scrollWidth - innerWidth, overflow: m.scrollHeight - m.clientHeight, talkOk: talk.top >= nav.bottom - .5 && talk.bottom <= m.getBoundingClientRect().bottom + .5,
                       remOk: rem.top >= nav.bottom - .5 && rem.bottom <= innerHeight, orb: Math.round(orb.width), subLines: sub.offsetParent ? Math.round(sub.getBoundingClientRect().height / parseFloat(getComputedStyle(sub).lineHeight)) : 1 }; }""")
            self.assertEqual(r["hscroll"], 0, f"{label} horizontal scroll")
            self.assertLessEqual(r["overflow"], 1, f"{label} talk screen scrolls after resize")
            self.assertTrue(r["talkOk"] and r["remOk"], f"{label} controls under the tab bar")
            self.assertGreaterEqual(r["orb"], 36, f"{label} sphere collapsed")
            self.assertLessEqual(r["subLines"], 1, f"{label} memory subtitle wrapped to {r['subLines']} lines")
        # Back at desktop size the sphere must have grown back to full size.
        self.assertEqual(page.locator(".orb-scene").bounding_box()["width"], 56)

    def test_small_touch_screen_and_webgl_fallback(self):
        context = self.browser.new_context(viewport={"width": 375, "height": 667}, is_mobile=True, has_touch=True, device_scale_factor=2, color_scheme="light")
        page = context.new_page()
        page.add_init_script("""const originalContext = HTMLCanvasElement.prototype.getContext;
          HTMLCanvasElement.prototype.getContext = function(type, ...args) {
            return type === 'webgl' ? null : originalContext.call(this, type, ...args);
          };""")
        try:
            page.goto(self.base)
            self.assertEqual(page.locator("#voice-orb").get_attribute("data-renderer"), "fallback")
            self.assertTrue(page.locator("#orb-fallback").is_visible())
            talk, nav, main = [page.locator(selector).bounding_box() for selector in ("#talk", ".topbar", "main")]
            self.assertGreaterEqual(talk["y"], nav["y"] + nav["height"])
            self.assertLessEqual(talk["y"] + talk["height"], main["y"] + main["height"] + .5)
            for button in page.locator(".primary-nav button").all():
                self.assertGreaterEqual(button.bounding_box()["height"], 44)
            page.locator("#tab-camera").tap()
            self.assertTrue(page.locator("#camera-start").is_visible())
            page.locator("#tab-talk").tap()
            page.wait_for_timeout(250)
            page.screenshot(path=str(self.artifacts / "small-phone-fallback.png"), full_page=True)
            page.set_viewport_size({"width": 320, "height": 568})
            page.evaluate("selectView('talk')")
            talk, main = [page.locator(selector).bounding_box() for selector in ("#talk", "main")]
            self.assertLessEqual(talk["y"] + talk["height"], main["y"] + main["height"])
            self.assertTrue(page.evaluate("document.documentElement.scrollWidth <= innerWidth"))
        finally:
            context.close()

    def test_audio_reactive_visual_and_reduced_motion(self):
        page = self.page
        page.goto(self.base)
        page.wait_for_function("typeof voiceVisual !== 'undefined' && voiceVisual !== null")
        self.assertEqual(page.locator("#voice-orb").get_attribute("data-renderer"), "webgl")
        page.locator("#greeting").click()
        page.evaluate("""async () => {
          player = makePlayer(); await player.ctx.resume();
          const pcm = new Int16Array(24000 * 2);
          for (let i = 0; i < pcm.length; i++) pcm[i] = Math.sin(i * 2 * Math.PI * 220 / 24000) * 9000;
          player.play(pcm.buffer);
        }""")
        page.wait_for_function("voiceVisual.level > .05")
        self.assertEqual(page.locator("body").get_attribute("data-voice"), "speaking")
        page.screenshot(path=str(self.artifacts / "speaking.png"), full_page=True)
        page.evaluate("player.stop(); player.ctx.close(); player = null")
        page.emulate_media(reduced_motion="reduce")
        page.wait_for_timeout(150)
        self.assertFalse(page.evaluate("voiceVisual.animating"))
        page.emulate_media(reduced_motion="no-preference")
        page.locator("#tab-camera").click()
        page.wait_for_timeout(150)
        self.assertFalse(page.evaluate("voiceVisual.animating"))

    def test_all_function_handlers_render_new_results(self):
        page = self.page
        from urllib.parse import urlsplit

        def respond(route):
            path = urlsplit(route.request.url).path
            if path == "/api/push":
                route.continue_()
                return
            output = {"say": f"Result for {path}"}
            if path in ("/api/find", "/api/call", "/api/message", "/api/ride", "/api/flights"):
                output["card"] = {"title": path, "body": "Test response"}
            if path == "/api/photo-info":
                output["photo"] = {"url": "/photos/sarah", "caption": "Sarah, your granddaughter"}
            route.fulfill(json=output)

        page.route("**/api/**", respond)
        page.goto(f"{self.base}/?demo=1")
        self.assertEqual(page.locator("#demo button").count(), 12)
        for button in page.locator("#demo button").all():
            count = page.locator("#log > div").count()
            button.click()
            page.wait_for_function("count => document.querySelectorAll('#log > div').length === count + 1", arg=count)
        self.assertEqual(page.locator("#log > div").count(), 12)
        self.assertEqual(page.locator("#photo img").get_attribute("alt"), "Sarah, your granddaughter")

    def test_memory_file_to_sse_note_and_reduced_motion(self):
        page = self.page
        page.goto(self.base)
        page.wait_for_function("document.getElementById('feed-state').textContent.includes('connected')")
        page.wait_for_timeout(200)
        page.evaluate("window.shineCount = 0; document.getElementById('memory-shine').addEventListener('animationstart', () => window.shineCount++)")
        record = {"event_id": 1001, "logged_at": "2026-09-19T15:00:00", "object": "pill bottle", "event": "placed", "confidence": .9, "location_description": "On the blue chair, beside your bag"}
        with self.memory.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        page.wait_for_function("document.getElementById('memory-detail').textContent.includes('blue chair')")
        page.wait_for_function("window.shineCount === 1")
        page.screenshot(path=str(self.artifacts / "memory-saved.png"), full_page=True)
        page.wait_for_timeout(1700)
        page.reload()
        page.wait_for_function("document.getElementById('memory-detail').textContent.includes('blue chair')")
        self.assertNotIn("shine", page.locator("#memory-shine").get_attribute("class") or "")
        page.emulate_media(reduced_motion="reduce")
        with self.memory.open("a", encoding="utf-8") as f:
            f.write(json.dumps({**record, "event_id": 1002, "object": "keys"}) + "\n")
        page.wait_for_function("document.getElementById('memory-detail').textContent.startsWith('Keys:')")
        self.assertNotIn("shine", page.locator("#memory-shine").get_attribute("class") or "")

    def test_camera_reconnect_and_stop(self):
        attempts, frames = [], []

        class Relay:
            async def send(self, data):
                frames.append(data)
            async def wait_closed(self):
                await asyncio.Future()
            async def close(self):
                pass

        async def open_relay():
            attempts.append(1)
            if len(attempts) == 1:
                raise OSError("not running")
            return Relay()

        with patch.object(app, "open_camera_relay", open_relay):
            page = self.page
            page.goto(self.base)
            page.locator("#tab-camera").click()
            page.locator("#camera-start").click()
            page.wait_for_function("document.getElementById('cam').textContent.includes('pipeline is unavailable')")
            page.wait_for_function("document.getElementById('cam').textContent.includes('Images are reaching')", timeout=10000)
            self.assertGreater(len(frames), 0)
            self.assertTrue(frames[0].startswith(b"\xff\xd8"))
            page.locator("#camera-stop").click()
            page.wait_for_function("document.getElementById('cam').textContent.includes('off')")
            count = len(attempts)
            page.wait_for_timeout(2500)
            self.assertEqual(len(attempts), count)
            self.assertTrue(page.evaluate("camStream === null && camSocket === null"))

    def test_camera_denied_is_not_overwritten(self):
        page = self.page
        page.add_init_script("navigator.mediaDevices.getUserMedia = async () => { throw new DOMException('denied', 'NotAllowedError'); }")
        page.goto(self.base)
        page.locator("#tab-camera").click()
        page.locator("#camera-start").click()
        page.wait_for_function("document.getElementById('cam').textContent.includes('permission is blocked')")
        page.wait_for_timeout(1000)
        self.assertIn("permission is blocked", page.locator("#cam").inner_text())

    def test_features_do_not_execute_actions_during_voice(self):
        page = self.page
        lookups = []
        page.route("**/api/find?*", lambda route: (lookups.append(1), route.fulfill(json={"say": "local-only lookup"})))
        page.goto(f"{self.base}/?dg={self.base.replace('http:', 'ws:')}/api/fake-dg")
        page.locator("#talk").click()
        page.wait_for_function("document.getElementById('conn').textContent === 'connected'")
        page.locator("#features-open").click()
        page.locator('[data-capability="fetch_object"]').click()
        self.assertEqual(lookups, [])
        page.locator('[data-close="features-dialog"]').click()
        page.locator("#talk").click()

    def test_voice_and_function_round_trip(self):
        page = self.page
        page.goto(f"{self.base}/?dg={self.base.replace('http:', 'ws:')}/api/fake-dg")
        page.locator("#talk").click()
        page.wait_for_function("document.getElementById('conn').textContent === 'connected'")
        page.wait_for_function("document.getElementById('log').textContent.includes('Function result received')")
        self.assertEqual(page.locator("#talk").get_attribute("aria-pressed"), "true")
        page.wait_for_function("voiceVisual.level > .01")
        page.locator("#mic-toggle").click()
        self.assertTrue(page.evaluate("micMuted && mic.stream.getAudioTracks().every(track => !track.enabled)"))
        page.locator("#mic-toggle").click()
        self.assertTrue(page.evaluate("!micMuted && mic.stream.getAudioTracks().every(track => track.enabled)"))
        page.locator("#talk").click()
        self.assertEqual(page.locator("#conn").inner_text(), "offline")
        self.assertEqual(page.locator("#talk").get_attribute("aria-pressed"), "false")
        self.assertIn("Microphone is off", page.locator("#status").inner_text())


if __name__ == "__main__":
    unittest.main()
