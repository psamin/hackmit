"""Record OpenYAM demonstrations for VLA fine-tuning: dimOS's learning-collect-webxr blueprint for this arm.

    PYTHONPATH=. python vla/collect_openyam.py --real --camera-index 1     # ENERGIZES THE ARM: Quest teleop + arm camera
    PYTHONPATH=. python vla/collect_openyam.py --mock --test               # mock arm + synthetic camera, scripted episodes

Episodes: Quest B starts/saves and Y discards (dimOS default), or type in this terminal: Enter starts/saves, d + Enter
discards, q + Enter quits. The session DB path is printed; turn it into a LeRobot dataset with
    dimos dataprep build -s <session.db> -c vla/openyam_dataprep.json
"""
import argparse, time
from datetime import datetime

from dimos.core.core import rpc
from dimos.core.module import Module
from dimos.core.stream import Out
from dimos.imitation.collection.episode_monitor import KeyPress


class TerminalKeys(Module):
    """Feeds EpisodeMonitorModule's keyboard input, which nothing in dimOS publishes yet. Keys arrive over RPC
    because modules run in worker processes without a terminal."""

    keyboard: Out[KeyPress]

    @rpc
    def press(self, key: str) -> None:
        self.keyboard.publish(KeyPress(key=key, ts=time.time()))


def main() -> None:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--real", action="store_true", help="drive the real arm (energizes the motors)")
    mode.add_argument("--mock", action="store_true", help="dimOS's in-memory OpenYAM adapter")
    ap.add_argument("--camera-index", type=int, default=0, help="--real: OpenCV index of the arm camera")
    ap.add_argument("--db", default=None, help="session DB path (default: dimOS recordings dir)")
    ap.add_argument("--test", action="store_true", help="--mock only: record two scripted episodes and exit")
    args = ap.parse_args()

    from dimos.core.global_config import global_config

    if args.mock:
        global_config.simulation = "mock"  # before the OpenYAM blueprints are imported: they build hardware on import
    from dimos.constants import RECORDINGS_DIR
    from dimos.core.coordination.blueprints import autoconnect
    from dimos.core.coordination.module_coordinator import ModuleCoordinator
    from dimos.imitation.collection.episode_monitor import EpisodeMonitorModule
    from dimos.imitation.collection.recorder import CollectionRecorder
    from dimos.robot.manipulators.openyam.blueprints import teleop as openyam_teleop

    adapter = openyam_teleop._openyam_webxr_hw.adapter_type
    if args.mock and adapter != "mock_whole_body":
        raise SystemExit(f"refusing to run: expected the mock adapter, got {adapter!r}")
    if args.real:
        print(f"REAL ARM ({adapter}): the motors will be energized. Clear the workspace, keep the power switch in reach.")
        if input("Type 'go' to continue: ").strip() != "go":
            raise SystemExit("aborted")

    if args.mock:
        from vla.sim_test import SyntheticCamera

        camera = SyntheticCamera.blueprint()
    else:
        from dimos.hardware.sensors.camera.module import CameraModule
        from dimos.hardware.sensors.camera.webcam import Webcam

        camera = CameraModule.blueprint(hardware=Webcam(camera_index=args.camera_index, width=640, height=480, fps=30.0))

    db = args.db or str(RECORDINGS_DIR / f"session_openyam_{datetime.now():%Y%m%d_%H%M%S}.db")
    blueprint = autoconnect(
        CollectionRecorder.blueprint(db_path=db, poseless_streams=["color_image", "coordinator_joint_state", "status"],
                                     record_tf=False),
        EpisodeMonitorModule.blueprint(keyboard_map={"toggle": "enter", "discard": "d"}),
        TerminalKeys.blueprint(),
        openyam_teleop.teleop_webxr_openyam,
        camera,
    )
    coordinator = ModuleCoordinator.build(blueprint)
    keys = coordinator.get_instance(TerminalKeys)
    print(f"recording to {db}", flush=True)
    try:
        if args.test:
            time.sleep(2.0)
            for hold in (2.0, 1.5):  # two short episodes
                keys.press("enter"); time.sleep(hold); keys.press("enter"); time.sleep(0.5)
        else:
            while (line := input("Enter = start/save, d = discard, q = quit: ").strip().lower()) != "q":
                keys.press("d" if line == "d" else "enter")
    finally:
        coordinator.stop()  # the recorder flushes the DB on shutdown
    print(f"saved {db}")


if __name__ == "__main__":
    main()
