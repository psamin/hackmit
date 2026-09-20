"""Start and stop the arm's policy rollout from another process (e.g. the voice loop).

run_policy.py serves these on 127.0.0.1:8020:
    POST /preflight  POST /start  POST /stop  POST /home  GET /status   -> the rollout status dict
    POST /open  POST /close                                             -> the gripper alone
"""
import json, urllib.request

ARM_URL = "http://127.0.0.1:8020"


def arm(command: str, url: str = ARM_URL, timeout: float = 60.0) -> dict:
    method = "GET" if command == "status" else "POST"
    req = urllib.request.Request(f"{url}/{command}", method=method, data=None if method == "GET" else b"{}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def gripper(state: str, url: str = ARM_URL) -> dict:
    """Open or close the gripper where the arm already is. state is "open" or "close"."""
    return arm(state, url)


def home(url: str = ARM_URL) -> dict:
    """Stop whatever the arm is doing and drive it back to the taught home pose."""
    return arm("home", url)


def fetch(url: str = ARM_URL) -> dict:
    """Preflight, then start. Returns the status; last_error says why it didn't start.

    /start answers at once with {"starting": true} and drives to home on its own thread, so this returns before
    the arm has moved - blocking here used to outlast the caller's timeout and read as "the arm is unreachable".
    Progress arrives on the voice agent's push channel instead.
    """
    status = arm("preflight", url)
    return arm("start", url) if status["policy_ready"] and status["observations_ready"] else status

