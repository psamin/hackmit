"""Start and stop the arm's policy rollout from another process (e.g. the voice loop).

run_policy.py serves these on 127.0.0.1:8020:
    POST /preflight  POST /start  POST /stop  GET /status   -> the rollout status dict
"""
import json, urllib.request

ARM_URL = "http://127.0.0.1:8020"


def arm(command: str, url: str = ARM_URL, timeout: float = 10.0) -> dict:
    method = "GET" if command == "status" else "POST"
    req = urllib.request.Request(f"{url}/{command}", method=method, data=None if method == "GET" else b"{}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def gripper(state: str, url: str = ARM_URL) -> dict:
    """Open or close the gripper where the arm already is. state is "open" or "close"."""
    return arm(state, url)


def fetch(url: str = ARM_URL) -> dict:
    """Preflight, then start. Returns the status; last_error says why it didn't start."""
    status = arm("preflight", url)
    return arm("start", url) if status["policy_ready"] and status["observations_ready"] else status
