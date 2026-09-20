"""dimOS module that runs a RunPod-hosted policy (vla/policy_server.py) on the arm.

Same inputs and safety checks as dimOS's LeRobot runtime (native/python/lerobot/dimos_lerobot/runtime.py,
Apache-2.0): missing or stale observations, missing joints and non-finite values are rejected, every action is
clipped to the checkpoint's recorded range, and stopping cancels the trajectory. Only the model call is remote.

    RemotePolicyModule.blueprint(server_url="https://<pod-id>-8000.proxy.runpod.net",
                                 task="pick up the pill bottle", joint_names=OPENYAM_JOINTS, fps=30.0)

Drive it over RPC: preflight_rollout() loads nothing and moves nothing; start_rollout(); stop_rollout().
"""
from __future__ import annotations

import base64, json, time, urllib.request
from threading import Condition, Event, RLock, Thread
from typing import Any

import cv2
import numpy as np
from pydantic import Field
from reactivex.disposable import Disposable

from dimos.control.tasks.trajectory_task.trajectory_task import JOINT_TRAJECTORY_TASK_NAME, TrajectoryExecutionStatus
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In
from dimos.imitation.policy.lerobot.module import PolicyControlSpec, RolloutStatus
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.trajectory_msgs.JointTrajectory import JointTrajectory
from dimos.msgs.trajectory_msgs.TrajectoryPoint import TrajectoryPoint
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


class RemotePolicyModuleConfig(ModuleConfig):
    server_url: str = Field(min_length=1)
    task: str = Field(min_length=1)
    joint_names: list[str] = Field(min_length=1)
    fps: float = Field(default=30.0, gt=0)  # must match the training dataset's action rate
    max_observation_age_s: float = Field(default=0.5, gt=0)
    request_timeout_s: float = Field(default=5.0, gt=0)
    jpeg_quality: int = Field(default=85, ge=10, le=100)
    # ACT predicts a whole chunk, and running one to its last action leaves the arm holding that pose while the next
    # prediction is fetched, then starting somewhere the old chunk was not heading. Both show up as a jerk every few
    # seconds. Re-predict partway through instead, and cross-fade the seam.
    exec_fraction: float = Field(default=0.6, gt=0.05, le=1.0)  # how much of a chunk to run before re-predicting
    blend_steps: int = Field(default=10, ge=0)  # actions over which a new chunk fades in over the old one


class RemotePolicyModule(Module):
    config: RemotePolicyModuleConfig

    color_image: In[Image]
    coordinator_joint_state: In[JointState]

    _control: PolicyControlSpec

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._lock = RLock()
        self._observation_changed = Condition(self._lock)
        self._info: dict[str, Any] | None = None
        self._latest_image: tuple[np.ndarray, float] | None = None
        self._latest_joint_state: JointState | None = None
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._chunks_accepted = 0
        self._pending_tail: np.ndarray | None = None
        self._last_error: str | None = None
        self._active = False

    @rpc
    def start(self) -> None:
        super().start()
        self.register_disposable(Disposable(self.color_image.subscribe(self._on_color_image)))
        self.register_disposable(Disposable(self.coordinator_joint_state.subscribe(self._on_joint_state)))

    @rpc
    def stop(self) -> None:
        self._stop_policy()
        super().stop()

    @rpc
    def preflight_rollout(self) -> RolloutStatus:
        """Check the server, the coordinator and live observations without moving."""
        try:
            with self._lock:
                if self._active:
                    raise RuntimeError("cannot preflight while a policy rollout is active")
                self._snapshot_observation(time.time())
            if JOINT_TRAJECTORY_TASK_NAME not in set(self._control.list_tasks()):
                raise RuntimeError(f"ControlCoordinator is missing trajectory task {JOINT_TRAJECTORY_TASK_NAME!r}")
            info = self._request("GET", "/info")
            if info["joint_count"] != len(self.config.joint_names):
                raise RuntimeError(f"server policy has {info['joint_count']} joints, "
                                   f"configured {len(self.config.joint_names)}")
            info["action_min"] = np.asarray(info["action_min"], np.float32)
            info["action_max"] = np.asarray(info["action_max"], np.float32)
            with self._lock:
                self._info, self._last_error = info, None
                return self._status_locked()
        except Exception as exc:
            with self._lock:
                self._info, self._last_error = None, str(exc)
                return self._status_locked()

    @rpc
    def start_rollout(self) -> RolloutStatus:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                self._last_error = "a policy rollout is already active"
                return self._status_locked()
            if self._info is None:
                self._last_error = "policy preflight has not passed"
                return self._status_locked()
            try:
                self._snapshot_observation(time.time())
            except RuntimeError as exc:
                self._last_error = str(exc)
                return self._status_locked()
            self._stop_event.clear()
            self._chunks_accepted, self._last_error, self._active = 0, None, True
            self._thread = Thread(target=self._run_rollout, name="remote-policy-rollout", daemon=True)
            self._thread.start()
            return self._status_locked()

    @rpc
    def stop_rollout(self) -> RolloutStatus:
        self._stop_policy()
        return self.rollout_status()

    @rpc
    def rollout_status(self) -> RolloutStatus:
        with self._lock:
            return self._status_locked()

    def _status_locked(self) -> RolloutStatus:
        try:
            self._snapshot_observation(time.time())
            observations_ready = True
        except RuntimeError:
            observations_ready = False
        return {"active": self._active, "policy_path": self.config.server_url, "task": self.config.task,
                "device": None if self._info is None else f"remote:{self._info['policy']}",
                "policy_ready": self._info is not None, "observations_ready": observations_ready,
                "chunks_accepted": self._chunks_accepted, "last_error": self._last_error}

    def _on_color_image(self, image: Image) -> None:
        if image.data.dtype != np.uint8 or image.data.ndim != 3 or image.format not in (ImageFormat.RGB, ImageFormat.BGR):
            return
        rgb = image.data if image.format == ImageFormat.RGB else image.data[..., ::-1]
        with self._lock:
            self._latest_image = (np.ascontiguousarray(rgb), image.ts)

    def _on_joint_state(self, state: JointState) -> None:
        with self._observation_changed:
            self._latest_joint_state = JointState(state)
            self._observation_changed.notify_all()

    def _snapshot_observation(self, now: float) -> tuple[np.ndarray, np.ndarray, float]:
        if self._latest_image is None:
            raise RuntimeError("no camera image has been received")
        if self._latest_joint_state is None:
            raise RuntimeError("no coordinator joint state has been received")
        image, image_ts = self._latest_image
        state, max_age = self._latest_joint_state, self.config.max_observation_age_s
        if now - image_ts > max_age:
            raise RuntimeError(f"camera image is stale by {now - image_ts:.2f}s")
        if now - state.ts > max_age:
            raise RuntimeError(f"joint state is stale by {now - state.ts:.2f}s")
        positions = dict(zip(state.name, state.position, strict=False))
        missing = [n for n in self.config.joint_names if n not in positions]
        if missing:
            raise RuntimeError(f"joint state is missing configured joints: {missing}")
        vector = np.asarray([positions[n] for n in self.config.joint_names], dtype=np.float32)
        if not np.all(np.isfinite(vector)):
            raise RuntimeError("joint state contains non-finite positions")
        return image.copy(), vector, state.ts

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        req = urllib.request.Request(self.config.server_url.rstrip("/") + path, method=method,
                                     data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json",
                                              # RunPod fronts the pod proxy with Cloudflare, which answers
                                              # Python's default urllib agent with 403 error 1010.
                                              "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                                                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                                                            "Chrome/126.0 Safari/537.36"})
        try:
            with urllib.request.urlopen(req, timeout=self.config.request_timeout_s) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"policy server {path}: HTTP {exc.code} {exc.read()[:300]!r}") from exc

    def _predict(self, image: np.ndarray, state: np.ndarray) -> np.ndarray:
        _, height, width = self._info["image_shape"]
        if image.shape[:2] != (height, width):
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        ok, jpeg = cv2.imencode(".jpg", image[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality])
        if not ok:
            raise RuntimeError("could not JPEG-encode the camera image")
        out = self._request("POST", "/act", {"image_jpeg_b64": base64.b64encode(jpeg).decode(),
                                             "state": state.tolist(), "task": self.config.task})
        return np.asarray(out["actions"], dtype=np.float32)

    def _run_rollout(self) -> None:
        try:
            info = self._info
            if info is None:
                raise RuntimeError("policy preflight has not passed")
            self._request("POST", "/reset", {})
            self._pending_tail = None
            steps, width = info["n_action_steps"], len(self.config.joint_names)
            while not self._stop_event.is_set():
                with self._lock:
                    image, state, state_ts = self._snapshot_observation(time.time())
                actions = self._predict(image, state)
                if actions.shape != (steps, width):
                    raise RuntimeError(f"policy returned actions {actions.shape}, expected {(steps, width)}")
                if not np.all(np.isfinite(actions)):
                    raise RuntimeError("policy returned non-finite joint targets")
                bounded = np.clip(actions, info["action_min"], info["action_max"])
                # Fade the new chunk in over the tail of the one still running, so the seam is a ramp not a step.
                tail = self._pending_tail
                if tail is not None and self.config.blend_steps:
                    n = min(self.config.blend_steps, len(tail), len(bounded))
                    if n:
                        ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
                        bounded[:n] = ramp * bounded[:n] + (1.0 - ramp) * tail[:n]
                clipped = np.any(actions != bounded, axis=0)
                if np.any(clipped):
                    logger.warning("Clipped policy actions to checkpoint range",
                                   joints=[n for n, c in zip(self.config.joint_names, clipped, strict=True) if c])
                if self._stop_event.is_set():
                    break
                result = self._control.execute_trajectory(self._trajectory(state, bounded))
                if result.status is TrajectoryExecutionStatus.START_STATE_MISMATCH:
                    self._wait_for_newer_joint_state(state_ts)
                    continue
                if result.status is not TrajectoryExecutionStatus.ACCEPTED:
                    raise RuntimeError(result.message or f"trajectory rejected: {result.status.name}")
                with self._lock:
                    self._chunks_accepted += 1
                run_steps = max(1, int(steps * self.config.exec_fraction))
                self._pending_tail = bounded[run_steps:]  # what the arm would have done had it kept going
                self._stop_event.wait(run_steps / self.config.fps)
        except Exception as exc:
            with self._lock:
                self._last_error = str(exc)
            logger.exception("Remote policy execution stopped", error=str(exc))
        finally:
            self._stop_event.set()
            error = self._cancel_trajectory()
            with self._lock:
                if error is not None:
                    self._last_error = f"{self._last_error}; {error}" if self._last_error else error
                self._active = False

    def _stop_policy(self) -> None:
        with self._observation_changed:
            self._stop_event.set()
            self._observation_changed.notify_all()
            thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(self.config.request_timeout_s + 1.0)  # a request in flight finishes or times out first
            if thread.is_alive():
                self._cancel_trajectory()

    def _trajectory(self, state: np.ndarray, actions: np.ndarray) -> JointTrajectory:
        zeros = [0.0] * len(self.config.joint_names)
        points = [TrajectoryPoint(positions=[float(v) for v in state], velocities=zeros, time_from_start=0.0)]
        points += [TrajectoryPoint(positions=[float(v) for v in action], velocities=zeros,
                                   time_from_start=(i + 1) / self.config.fps) for i, action in enumerate(actions)]
        return JointTrajectory(joint_names=list(self.config.joint_names), points=points)

    def _wait_for_newer_joint_state(self, previous_ts: float) -> None:
        with self._observation_changed:
            self._observation_changed.wait_for(lambda: self._stop_event.is_set() or (
                self._latest_joint_state is not None and self._latest_joint_state.ts > previous_ts))

    def _cancel_trajectory(self) -> str | None:
        try:
            result = self._control.cancel_trajectory()
        except Exception as exc:
            logger.exception("Failed to cancel policy trajectory")
            return f"Failed to cancel policy trajectory: {exc}"
        return None if result.safe else (result.message or "Policy trajectory cancellation was uncertain")
