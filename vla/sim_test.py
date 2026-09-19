"""End-to-end check of the RunPod policy path against dimOS's MOCK OpenYAM arm. Never touches real hardware.

    python vla/policy_server.py --dummy --host 127.0.0.1 --port 8011 &
    PYTHONPATH=. python vla/sim_test.py --server http://127.0.0.1:8011

Checks: preflight passes, the rollout sends action chunks the coordinator accepts, stop cancels cleanly.
"""
import argparse, time
from threading import Event, Thread

import numpy as np

from dimos.core.global_config import global_config

global_config.simulation = "mock"  # must be set before OpenYAM hardware is built: selects the in-memory adapter

from dimos.control.coordinator import ControlCoordinator  # noqa: E402
from dimos.control.tasks.trajectory_task.trajectory_task import joint_trajectory_task  # noqa: E402
from dimos.core.coordination.blueprints import autoconnect  # noqa: E402
from dimos.core.coordination.module_coordinator import ModuleCoordinator  # noqa: E402
from dimos.core.core import rpc  # noqa: E402
from dimos.core.module import Module  # noqa: E402
from dimos.core.stream import Out  # noqa: E402
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat  # noqa: E402
from dimos.robot.manipulators.openyam.config import OPENYAM_JOINTS, openyam_hardware  # noqa: E402

from vla.remote_policy import RemotePolicyModule  # noqa: E402


class SyntheticCamera(Module):
    """Publishes a flat grey 480x640 RGB frame at 15 Hz in place of the arm camera."""

    color_image: Out[Image]

    @rpc
    def start(self) -> None:
        super().start()
        self._done = Event()
        Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        frame = np.full((480, 640, 3), 128, np.uint8)
        while not self._done.is_set():
            self.color_image.publish(Image.from_numpy(frame, format=ImageFormat.RGB))
            self._done.wait(1 / 15)

    @rpc
    def stop(self) -> None:
        self._done.set()
        super().stop()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8011")
    ap.add_argument("--seconds", type=float, default=4.0)
    args = ap.parse_args()

    hardware = openyam_hardware()
    if hardware.adapter_type != "mock_whole_body":
        raise SystemExit(f"refusing to run: OpenYAM adapter is {hardware.adapter_type!r}, not the mock")

    blueprint = autoconnect(
        ControlCoordinator.blueprint(hardware=[hardware], tasks=[joint_trajectory_task(OPENYAM_JOINTS)]),
        SyntheticCamera.blueprint(),
        RemotePolicyModule.blueprint(server_url=args.server, task="pick up the pill bottle",
                                     joint_names=list(OPENYAM_JOINTS), fps=30.0),
    )
    coordinator = ModuleCoordinator.build(blueprint)
    try:
        policy = coordinator.get_instance(RemotePolicyModule)
        time.sleep(2.0)  # let the camera and joint-state streams start
        print("preflight:", policy.preflight_rollout(), flush=True)
        print("start:    ", policy.start_rollout(), flush=True)
        time.sleep(args.seconds)
        print("running:  ", policy.rollout_status(), flush=True)
        print("stop:     ", policy.stop_rollout(), flush=True)
    finally:
        coordinator.stop()


if __name__ == "__main__":
    from vla.sim_test import main as imported_main  # re-import so worker processes can find SyntheticCamera

    imported_main()
