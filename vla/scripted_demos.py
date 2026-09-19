"""Record VLA demos without a VR headset: teach one pick per taped bottle spot by hand, then dimOS replays it with small
variations while the camera and joint states are recorded.

    python vla/scripted_demos.py teach spots/left.json                                    # motors stay disabled
    python vla/scripted_demos.py check spots/left.json                                    # gripper path, no hardware
    PYTHONPATH=. python vla/scripted_demos.py record spots/*.json --real --camera-index 1 --reps 10
    PYTHONPATH=. python vla/scripted_demos.py record spots/*.json --mock --auto-reset 0.5  # no hardware

Teach each spot's pick, e.g. `home o`, `above o`, `grasp c`, `lift c`, and one shared hand-over, e.g. `handover c`,
`release o`, `home o`: the gripper opens or closes in place at the pose where its state changes. `record --after
handover.json` plays the hand-over after each saved pick without recording it (before a hand-over is taught, `--release`
opens the gripper in place and returns home instead), so ACT learns only the pick (table views)
and run_policy.py --handover plays the same preset motion. Then `dimos dataprep build --source <session.db> --config
vla/openyam_dataprep.json` and `vla/runpod.sh train`.
"""
import argparse, functools, json, time
from datetime import datetime

import numpy as np

ARM_IDS = (1, 2, 3, 4, 5, 6)  # OpenYAM joints 1-3 are DM4340, 4-6 DM4310; feedback on send ID + 0x10
# Joint limits (rad) from dimOS's yam.urdf; replay noise never pushes a target past them.
LIMITS = np.array([(-2.618, 3.142), (0.0, 3.665), (0.0, 3.142), (-1.693, 1.571), (-1.571, 1.571), (-2.094, 2.094)])


def teach(path: str) -> None:
    """Refresh queries only (as arm/arm_probe.py): the motors are never enabled, so the arm can be posed by hand."""
    import can_motor_control as cmc
    from can_motor_control import damiao

    types = [damiao.MotorType.DM4340] * 3 + [damiao.MotorType.DM4310] * 3
    motors = [cmc.MotorSpec(f"yam_joint{i}", t, i, i + 0x10) for i, t in zip(ARM_IDS, types)]
    bus = cmc.GsUsbBus(vendor_id=0x1D50, product_id=0x606F)
    robot = (cmc.Robot.builder().add_bus("openyam", bus, damiao.DamiaoCodec())
             .add_arm("arm", bus="openyam", motors=motors).build())
    robot.connect()
    poses = []
    print("Motors stay disabled. Pose the arm by hand, then type the pose name and o/c for the gripper "
          "('above o', 'grasp c'). Empty line = refresh, q = save.")
    try:
        while True:
            for _ in range(5):
                robot.refresh()
                robot.tick(5000)
                time.sleep(0.01)
            q = [round(float(v), 4) for v in robot["arm"].positions()]
            words = input(f"{q}  pose + o/c: ").split()
            if words == ["q"]:
                break
            if len(words) == 2 and words[1] in ("o", "c"):
                poses.append({"name": words[0], "q": q, "gripper": 1.0 if words[1] == "o" else 0.0})
    finally:
        robot.__exit__(None, None, None)
    json.dump({"poses": poses}, open(path, "w"), indent=1)
    print(f"saved {len(poses)} poses to {path}")


def check(path: str) -> None:
    """Where the gripper tip goes during the replay, from dimOS's OpenYAM model (the one it plans with)."""
    import pinocchio as pin
    from dimos.robot.manipulators.openyam.config import OPENYAM_MODEL_PATH

    model = pin.buildModelFromUrdf(str(OPENYAM_MODEL_PATH))
    data, tip = model.createData(), model.getFrameId("gripper_tip")

    def fk(q6):
        q = pin.neutral(model)
        q[:6] = q6
        pin.framesForwardKinematics(model, data, q)
        return data.oMf[tip].translation * 100  # cm

    poses, warnings = json.load(open(path))["poses"], []
    print(f"{path}: " + " -> ".join(f"{p['name']} {'o' if p['gripper'] else 'c'}" for p in poses))
    for p in poses:
        q = np.asarray(p["q"])
        x, y, z = fk(q)
        margin = np.minimum(q - LIMITS[:, 0], LIMITS[:, 1] - q)
        print(f"  {p['name']:9s} tip x {x:6.1f}  y {y:6.1f}  height {z:6.1f} cm   nearest limit: joint {margin.argmin() + 1} "
              f"({margin.min():.2f} rad)")
        if margin.min() < 0.05:
            where = f"{-margin.min():.2f} rad past" if margin.min() < 0 else f"{margin.min():.2f} rad from"
            warnings.append(f"{p['name']}: joint {margin.argmin() + 1} is {where} its limit; move it off")
    for a, b in zip(poses, poses[1:]):
        qa, qb = np.asarray(a["q"]), np.asarray(b["q"])
        path_pts = np.array([fk(qa + t * (qb - qa)) for t in np.linspace(0, 1, 41)])
        start, end = path_pts[0], path_pts[-1]
        line = end - start
        off = max(np.linalg.norm(np.cross(pt - start, line)) / max(np.linalg.norm(line), 1e-6) for pt in path_pts)
        print(f"  {a['name']:>9s} -> {b['name']:<9s} tip moves {np.linalg.norm(line):5.1f} cm, "
              f"{line[2]:+5.1f} cm up/down, {off:4.1f} cm off a straight line")
        if a["gripper"] and not b["gripper"]:  # the approach into the grasp
            side = np.linalg.norm(line[:2])
            if side > 3.0:
                warnings.append(f"{a['name']} is {side:.0f} cm to the side of {b['name']}: put it directly over the bottle")
            if line[2] > -2.0:
                warnings.append(f"{a['name']} -> {b['name']} doesn't go down; the approach should come from above")
        if not a["gripper"] and not b["gripper"] and poses.index(a) == next(
                (i for i, p in enumerate(poses) if not p["gripper"]), None):  # the move right after the grasp
            if line[2] < 3.0:
                warnings.append(f"{b['name']} rises only {line[2]:+.1f} cm after {a['name']}; lift straight up ~10 cm")
    print("\n".join(f"  WARNING: {w}" for w in warnings) if warnings else "  OK: ready to test")


def build_trajectory(joint_names, start, poses, speed, noise, rng):
    """Joint-space path from `start` through the taught poses. The arm moves with the gripper held, then the gripper
    changes in place. Poses where the gripper changes (grasp, release) are exact; the rest get +-noise rad per joint."""
    from dimos.msgs.trajectory_msgs.JointTrajectory import JointTrajectory
    from dimos.msgs.trajectory_msgs.TrajectoryPoint import TrajectoryPoint

    zeros = [0.0] * len(joint_names)
    arm, gripper, t = np.asarray(start[:-1], float), float(start[-1]), 0.0
    points = [TrajectoryPoint(positions=[*arm, gripper], velocities=zeros, time_from_start=0.0)]
    for pose in poses:
        target = np.asarray(pose["q"], float)
        if pose["gripper"] == gripper:
            target = np.clip(target + rng.uniform(-noise, noise, target.shape), LIMITS[:, 0], LIMITS[:, 1])
        t += max(float(np.abs(target - arm).max()) / speed, 0.5)
        arm = target
        points.append(TrajectoryPoint(positions=[*arm, gripper], velocities=zeros, time_from_start=t))
        if pose["gripper"] != gripper:
            gripper, t = pose["gripper"], t + 1.0
            points.append(TrajectoryPoint(positions=[*arm, gripper], velocities=zeros, time_from_start=t))
    return JointTrajectory(joint_names=list(joint_names), points=points), t


def record(args) -> None:
    from dimos.core.global_config import global_config

    if args.mock:
        global_config.simulation = "mock"  # before building OpenYAM hardware: selects the in-memory adapter
    from dimos.constants import RECORDINGS_DIR
    from dimos.control.coordinator import ControlCoordinator, TaskConfig
    from dimos.control.tasks.trajectory_task.trajectory_task import TrajectoryExecutionStatus, joint_trajectory_task
    from dimos.core.coordination.blueprints import autoconnect
    from dimos.core.coordination.module_coordinator import ModuleCoordinator
    from dimos.imitation.collection.episode_monitor import EpisodeMonitorModule
    from dimos.imitation.collection.recorder import CollectionRecorder
    from dimos.robot.manipulators.openyam.config import (OPENYAM_GRIPPER_JOINT, OPENYAM_HARDWARE_ID, OPENYAM_JOINTS,
                                                         openyam_hardware)

    from vla.collect_openyam import TerminalKeys

    spots = {path: json.load(open(path))["poses"] for path in args.spots}
    after = json.load(open(args.after))["poses"] if args.after else None
    hardware = openyam_hardware()
    if args.mock and hardware.adapter_type != "mock_whole_body":
        raise SystemExit(f"refusing to run: expected the mock adapter, got {hardware.adapter_type!r}")
    if args.real:
        print(f"REAL ARM ({hardware.adapter_type}): the arm will replay {len(spots)} taught picks x {args.reps} at "
              f"<= {args.speed} rad/s. Clamp the base, clear the workspace, keep the power switch in reach.")
        if input("Type 'go' to continue: ").strip() != "go":
            raise SystemExit("aborted")
    if args.mock:
        from vla.sim_test import SyntheticCamera

        camera = SyntheticCamera.blueprint()
    else:
        from dimos.hardware.sensors.camera.module import CameraModule
        from dimos.hardware.sensors.camera.webcam import Webcam

        camera = CameraModule.blueprint(  # a factory: dimOS builds the webcam inside its worker process
            hardware=functools.partial(Webcam, camera_index=args.camera_index, width=640, height=480, fps=30.0))

    db = args.db or str(RECORDINGS_DIR / f"session_openyam_scripted_{datetime.now():%Y%m%d_%H%M%S}.db")
    blueprint = autoconnect(
        CollectionRecorder.blueprint(db_path=db, poseless_streams=["color_image", "coordinator_joint_state", "status"],
                                     record_tf=False),
        EpisodeMonitorModule.blueprint(keyboard_map={"toggle": "enter", "discard": "d"}),
        TerminalKeys.blueprint(),
        ControlCoordinator.blueprint(hardware=[hardware], tasks=[
            joint_trajectory_task(OPENYAM_JOINTS),
            TaskConfig(name=f"{OPENYAM_HARDWARE_ID}_gripper", type="gripper", joint_names=[OPENYAM_GRIPPER_JOINT],
                       priority=20),
        ]),
        camera,
    )
    coordinator = ModuleCoordinator.build(blueprint)
    control, keys = coordinator.get_instance(ControlCoordinator), coordinator.get_instance(TerminalKeys)
    rng = np.random.default_rng(args.seed)
    saved = 0

    def play(motion):
        """Run poses from the current state, unrecorded and without noise; wait until done."""
        positions = control.get_joint_positions()
        trajectory, duration = build_trajectory(OPENYAM_JOINTS, [positions[n] for n in OPENYAM_JOINTS], motion,
                                                args.speed, 0.0, rng)
        if control.execute_trajectory(trajectory).status is TrajectoryExecutionStatus.ACCEPTED:
            time.sleep(duration + 0.5)

    print(f"recording to {db}", flush=True)
    try:
        time.sleep(2.0)  # let the camera and joint-state streams start
        for rep in range(args.reps):
            for path, poses in spots.items():
                if args.auto_reset is None:
                    input(f"[{rep + 1}/{args.reps}] Put the bottle on {path}, then press Enter: ")
                else:
                    time.sleep(args.auto_reset)
                for _ in range(3):  # the start must match the live state; retry if the arm settled in between
                    positions = control.get_joint_positions()
                    start = [positions[name] for name in OPENYAM_JOINTS]
                    trajectory, duration = build_trajectory(OPENYAM_JOINTS, start, poses, args.speed, args.noise, rng)
                    keys.press("enter")  # start the episode just before the motion
                    result = control.execute_trajectory(trajectory)
                    if result.status is TrajectoryExecutionStatus.ACCEPTED:
                        break
                    keys.press("d")  # discard the empty episode
                    time.sleep(0.2)
                else:
                    raise RuntimeError(f"trajectory rejected: {result.status.name} {result.message or ''}")
                time.sleep(duration + 0.5)
                keys.press("enter")  # save
                saved += 1
                print(f"[{rep + 1}/{args.reps}] {path}: saved ({duration:.1f} s)", flush=True)
                if after:  # the preset hand-over: played, not recorded
                    play(after)
                elif args.release:  # no hand-over yet: open in place so the bottle can be taken, then go home
                    arm_now = [control.get_joint_positions()[n] for n in OPENYAM_JOINTS[:-1]]
                    print("gripper opening: take the bottle", flush=True)
                    play([{"name": "release", "q": arm_now, "gripper": 1.0}])
                    time.sleep(2.0)
                    play([poses[0]])
    finally:
        control.cancel_trajectory()
        coordinator.stop()  # the recorder flushes the DB on shutdown
    print(f"saved {saved} episodes to {db}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("teach", help="pose the disabled arm by hand and save one spot's pick")
    t.add_argument("spot")
    c = sub.add_parser("check", help="where the gripper tip goes for a taught file (arm model, no hardware)")
    c.add_argument("spot")
    r = sub.add_parser("record", help="replay taught picks with small variations and record demos")
    r.add_argument("spots", nargs="+")
    mode = r.add_mutually_exclusive_group(required=True)
    mode.add_argument("--real", action="store_true", help="drive the real arm (energizes the motors)")
    mode.add_argument("--mock", action="store_true", help="dimOS's in-memory OpenYAM adapter")
    r.add_argument("--reps", type=int, default=10)
    r.add_argument("--speed", type=float, default=0.4, help="rad/s cap per joint")
    r.add_argument("--noise", type=float, default=0.03, help="rad of uniform noise on poses without a gripper change")
    r.add_argument("--camera-index", type=int, default=0)
    r.add_argument("--db", default=None)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--auto-reset", type=float, default=None, help="seconds between reps instead of waiting for Enter")
    r.add_argument("--after", default=None, help="preset hand-over poses to play after each pick, not recorded")
    r.add_argument("--release", action="store_true",
                   help="no hand-over yet: after each pick open the gripper in place, then go home (not recorded)")
    args = ap.parse_args()
    {"teach": lambda: teach(args.spot), "check": lambda: check(args.spot), "record": lambda: record(args)}[args.cmd]()


if __name__ == "__main__":
    main()
