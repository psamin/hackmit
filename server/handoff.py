"""Noticing that the arm has handed over the pills, so Pam can ask about them afterwards.

    outcome = await watch(item, status_fn, on_handoff)

`status_fn()` returns the arm's status dict (vla/arm_client.arm("status")); `on_handoff(item)` is called at
most once. The watcher only LOOKS. It never starts, stops or commands the arm.

The arm's rollout has no "finished" state of its own: it runs until someone stops it or it fails. So the handoff is
`active` going false with no `last_error`: the rollout was stopped cleanly, which is what happens when the bottle
has been handed over. A rollout that ends with an error is not a handoff, and neither is an arm that stopped
answering. `POST /api/handoff` is the explicit signal, for whoever or whatever knows better.

Outcomes: "handed_over", "failed" (the arm reported an error), "lost" (the arm stopped answering), "timeout".
"""
from __future__ import annotations

import asyncio
import time


async def watch(item: str, status_fn, on_handoff, *, poll_s: float = 2.0, timeout_s: float = 600.0,
                lost_after: int = 3, sleep=asyncio.sleep, clock=time.monotonic) -> str:
    started, unreachable = clock(), 0
    while clock() - started < timeout_s:
        try:
            status = await asyncio.to_thread(status_fn)
            unreachable = 0
        except Exception:  # noqa: BLE001 - the arm's server being down is an outcome, not a crash
            unreachable += 1
            if unreachable >= lost_after:
                return "lost"
            await sleep(poll_s)
            continue
        if not status.get("active"):
            if status.get("last_error"):
                return "failed"
            on_handoff(item)
            return "handed_over"
        await sleep(poll_s)
    return "timeout"
