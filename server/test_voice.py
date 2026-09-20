"""Real Deepgram smoke test — run once DEEPGRAM_API_KEY is in server/.env.

    python server/test_voice.py

Connects to the Agent API with the exact Settings the phone sends, verifies
SettingsApplied comes back, then injects "where is my medication?" as a user
message and prints what the agent does — ideally a FunctionCallRequest for
find_object, which this script answers exactly the way agent.html would.
"""
import asyncio, json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
for env in (Path(__file__).parent / ".env", Path(__file__).parent.parent / "perception" / ".env"):
    if env.exists():
        for line in env.read_text().splitlines():
            k, _, v = line.partition("=")
            if k.strip() and not k.startswith("#"):
                os.environ.setdefault(k.strip(), v.strip())

import httpx
from websockets.asyncio.client import connect

from app import agent_config  # the same Settings the page sends


async def main():
    key = os.environ.get("DEEPGRAM_API_KEY")
    if not key:
        sys.exit("DEEPGRAM_API_KEY not set — put it in server/.env")

    # 1. does the grant endpoint accept this key?
    async with httpx.AsyncClient() as c:
        r = await c.post("https://api.deepgram.com/v1/auth/grant",
                         headers={"Authorization": f"Token {key}"}, json={"ttl_seconds": 60})
        assert r.status_code == 200, f"grant failed: {r.status_code} {r.text}"
    print("PASS  /v1/auth/grant accepted the key (Member role ok)")

    cfg = agent_config()
    settings = {"type": "Settings",
                "audio": {"input": {"encoding": "linear16", "sample_rate": 16000},
                          "output": {"encoding": "linear16", "sample_rate": 24000}},
                "agent": cfg}

    async with connect("wss://agent.deepgram.com/v1/agent/converse",
                       subprotocols=["token", key]) as ws:
        await ws.send(json.dumps(settings))
        applied = False
        for _ in range(50):
            m = json.loads(await asyncio.wait_for(ws.recv(), 15))
            print("  <-", m.get("type"), str(m)[:110])
            if m["type"] == "SettingsApplied":
                applied = True
                break
            if m["type"] == "Error":
                sys.exit(f"FAIL  settings rejected: {m}")
        assert applied, "no SettingsApplied"
        print("PASS  Settings accepted (listen/think/speak/functions all valid)")

        # 2. inject a question; the think layer should emit find_object
        await ws.send(json.dumps({"type": "InjectUserMessage",
                                  "content": "Pam, where did I leave my medication?"}))
        for _ in range(60):
            m = json.loads(await asyncio.wait_for(ws.recv(), 20))
            t = m.get("type")
            print("  <-", t, str(m)[:130])
            if t == "FunctionCallRequest":
                for f in m["functions"]:
                    assert f["client_side"], "expected client-side"
                    # answer the way agent.html does — via the real server endpoint
                    args = json.loads(f["arguments"])
                    async with httpx.AsyncClient() as c:
                        out = (await c.get("http://127.0.0.1:8000/api/find",
                                           params={"q": args["item"]})).json()
                    print(f"  -> FunctionCallResponse {f['name']}: {out['say'][:90]}")
                    await ws.send(json.dumps({"type": "FunctionCallResponse", "id": f["id"],
                                              "name": f["name"], "content": out["say"]}))
            if t == "ConversationText" and m.get("role") == "assistant" and "table" in m.get("content", "").lower():
                print(f"PASS  agent spoke the location: \"{m['content']}\"")
                return
        print("(no assistant line asserting the location — check transcript above)")


if __name__ == "__main__":
    asyncio.run(main())
