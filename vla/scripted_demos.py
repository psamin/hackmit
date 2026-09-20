"""Record VLA demos without a VR headset: teach one pick per taped bottle spot by hand, then dimOS replays it with small
variations while the camera and joint states are recorded.

    python vla/scripted_demos.py teach spots/left.json                                    # motors stay disabled
    python vla/scripted_demos.py handover spots/left.json                                 # the preset, from the grasp
    PYTHONPATH=. python vla/scripted_demos.py record spots/*.json --real --camera-index 1 --reps 10 \
        --after spots/handover.json
    PYTHONPATH=. python vla/scripted_demos.py record spots/*.json --mock --auto-reset 0.5  # no hardware

A demo ends the moment the gripper closes on the bottle. Teach each spot as `home o`, `hover o`, `above o`, `grasp c`,
then `q`: the gripper opens or closes in place at the pose where its state changes, and nothing follows the grasp. The
first pose is home, which the arm drives to before each episode starts, so every recording begins at the same pose and
only the pick itself is recorded.

What happens after the grasp is hard-coded and never recorded, so the policy is trained on the pick alone: `handover`
writes spots/handover.json (lift, swing round, open) from a taught grasp, `record --after spots/handover.json` plays it
after each pick, and run_policy.py --handover plays the same preset. `--release` just opens in place and goes home,
for testing before a hand-over exists. Then `dimos dataprep build --source <session.db> --config
vla/openyam_dataprep.json` and `vla/runpod.sh train`.
"""
import argparse, functools, json, time
from datetime import datetime

import numpy as np

ARM_IDS = (1, 2, 3, 4, 5, 6)  # OpenYAM joints 1-3 are DM4340, 4-6 DM4310; feedback on send ID + 0x10
GRIPPER_S = 2.0        # the gripper closes in place over this long: the motor is slower than the arm
GRIPPER_SETTLE_S = 0.5  # hold after it closes, before lifting
GRIPPER_EPS = 0.05      # a gripper read off the hardware settles near 0 or 1, never exactly on it
# Joint limits (rad), read off dimOS's yam_gripper_gravity.urdf, which is what the Damiao adapter clamps feedback
# against. Replay noise never pushes a target past them. Teaching can: the motors are disabled, so the arm can be
# pushed a little beyond a software limit by hand, and the adapter then clamps it back (a fault only past 0.05 rad).
LIMITS = np.array([(-3.92699, 1.57080), (0.0, 3.66519), (0.0, 4.01426),
                   (-1.65806, 1.65806), (-1.57080, 1.57080), (-2.35619, 1.83260)])


def clamp_to_limits(q: list, name: str) -> list:
    """Bring a taught pose inside the joint limits, reporting anything it had to move.

    The motors are off while teaching, so a joint can be pushed past its software limit by hand and the encoder
    reports it honestly. The Damiao adapter clamps such a command back at replay, so store what the arm can actually
    be told to do. Past its own 0.05 rad fault margin the pose is wrong, not just over the edge, so say so loudly.
    """
    rounded = np.round(np.asarray(q, float), 4)  # round first: rounding a clamped value can push it back out
    inside = np.clip(rounded, LIMITS[:, 0], LIMITS[:, 1])
    for j in np.nonzero(inside != rounded)[0]:
        over = abs(rounded[j] - inside[j])
        loud = "  THAT IS A LOT - re-teach this pose" if over > 0.05 else ""
        print(f"  clamped {name} joint {j + 1}: {rounded[j]:+.5f} -> {inside[j]:+.5f} ({over:.4f} rad past its "
              f"limit){loud}")
    return [float(v) for v in inside]


def teach(path: str) -> None:
    """Refresh queries only (as arm/arm_probe.py): the motors are never enabled, so the arm can be posed by hand."""
    import can_motor_control as cmc
    from can_motor_control import damiao

    types = [damiao.MotorType.DM4340] * 3 + [damiao.MotorType.DM4310] * 3
    motors = [cmc.MotorSpec(f"yam_joint{i}", t, i, i + 0x10) for i, t in zip(ARM_IDS, types)]
    bus = cmc.GsUsbBus(vendor_id=0x1D50, product_id=0x606F)
    robot = (cmc.Robot.builder().add_bus("openyam", bus, damiao.DamiaoCodec())
             .add_arm("arm", bus="openyam", motors=motors).build())
    def read_q():
        for _ in range(5):
            robot.refresh()
            robot.tick(5000)
            time.sleep(0.01)
        return [round(float(v), 4) for v in robot["arm"].positions()]

    robot.connect()
    poses = []
    print("Motors stay disabled. Keep holding the arm where you want it, type the pose name and o/c for the gripper "
          "('above o', 'grasp c'), and press Enter: the angles are read when you press Enter, so the arm must still "
          "be in place then. Empty line = show the angles now, q = save.\n"
          "Stop at the grasp: the lift and hand-over come from `handover`, and must not be recorded.")
    try:
        shown = read_q()
        while True:
            words = input(f"{shown}  pose + o/c: ").split()
            if words == ["q"]:
                break
            shown = read_q()  # read now, with the arm held in place, not before the prompt was printed
            if len(words) == 2 and words[1] in ("o", "c"):
                if poses and shown == poses[-1]["q"]:
                    print(f"  NOT SAVED: identical to {poses[-1]['name']}; the arm did not move. Move it and retry.")
                    continue
                q = clamp_to_limits(shown, words[0])
                poses.append({"name": words[0], "q": q, "gripper": 1.0 if words[1] == "o" else 0.0})
                print(f"  saved {words[0]} {'open' if words[1] == 'o' else 'CLOSED'} {q}")
    finally:
        robot.__exit__(None, None, None)
    json.dump({"poses": poses}, open(path, "w"), indent=1)
    print(f"\nsaved {len(poses)} poses to {path}")
    for pose in poses:
        print(f"  {pose['name']:<8} {'open' if pose['gripper'] else 'CLOSED'}  {pose['q']}")
    warn_poses(path, poses)


def warn_poses(path: str, poses: list, ends_at_grasp: bool = True) -> list:
    """What would otherwise quietly ruin a recording: a pose past a joint limit, or anything after the grasp."""
    warnings, grasp = [], next((i for i, p in enumerate(poses) if not p["gripper"]), None)
    for pose in poses:
        q = np.asarray(pose["q"], float)
        past = [j + 1 for j in range(len(LIMITS)) if not LIMITS[j, 0] <= q[j] <= LIMITS[j, 1]]
        if past:
            warnings.append(f"{pose['name']}: joint {past} past the URDF limit")
    if ends_at_grasp and grasp is not None and grasp + 1 < len(poses):
        after = ", ".join(p["name"] for p in poses[grasp + 1:])
        warnings.append(f"{after} come after the grasp, and a demo has to end when the gripper closes. Move them "
                        f"into the preset: python vla/scripted_demos.py handover {path}")
    for warning in warnings:
        print(f"  WARNING {path}: {warning}")
    return warnings


def handover(path: str, out: str, rad: float, turn: float) -> None:
    """Write the preset hand-over - lift, swing round, open - from the grasp taught in `path`.

    Hard-coded on purpose: the policy is trained on the pick alone, so none of this may appear in a recording. It is
    also derived rather than taught, because the motors are disabled while teaching and the arm drops to rest the
    moment you let go to type. Smaller joint 2 is higher on this arm (1.03 hovering, 1.53 at the bottle, 2.36
    collapsed at rest); joint 1 is the base, and `turn` swings it round to whoever is taking the bottle.
    """
    grasp = next(p for p in json.load(open(path))["poses"] if not p["gripper"])
    up = list(grasp["q"])
    up[1] = round(up[1] - rad, 4)
    up = clamp_to_limits(up, "lift")
    turned = list(up)
    turned[0] = round(turned[0] + turn, 4)
    turned = clamp_to_limits(turned, "turn")
    poses = [{"name": "lift", "q": up, "gripper": 0.0},
             {"name": "turn", "q": turned, "gripper": 0.0},
             {"name": "release", "q": turned, "gripper": 1.0}]
    json.dump({"poses": poses}, open(out, "w"), indent=1)
    print(f"from grasp {grasp['q']}")
    for pose in poses:
        print(f"  {pose['name']:<8} {pose['q']}  gripper {'open' if pose['gripper'] else 'closed'}")
    print(f"saved {out}; play it with: record ... --after {out}")
    warn_poses(out, poses, ends_at_grasp=False)


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
        holds = abs(pose["gripper"] - gripper) < GRIPPER_EPS  # a measured start value is never exactly 0.0 or 1.0
        if holds:  # bound the noise, but never move the taught pose itself
            lo, hi = np.minimum(LIMITS[:, 0], target), np.maximum(LIMITS[:, 1], target)
            target = np.clip(target + rng.uniform(-noise, noise, target.shape), lo, hi)
        t += max(float(np.abs(target - arm).max()) / speed, 0.5)
        arm = target
        points.append(TrajectoryPoint(positions=[*arm, gripper], velocities=zeros, time_from_start=t))
        if not holds:  # close/open in place, then settle before moving on
            gripper, t = pose["gripper"], t + GRIPPER_S
            points.append(TrajectoryPoint(positions=[*arm, gripper], velocities=zeros, time_from_start=t))
            t += GRIPPER_SETTLE_S
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
    for path, poses in spots.items():
        warn_poses(path, poses)
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
            # A ceiling the task enforces itself, tighter than its 1.0 rad/s default and well above the speed we
            # ask for: --speed only shapes a trajectory's timing, so nothing else stops a bad pose or a bug in this
            # script being executed as fast as the motors allow. Every configured joint has to be named, and the
            # gripper keeps the default, since it travels its whole range in GRIPPER_S and must not be throttled.
            joint_trajectory_task(OPENYAM_JOINTS, velocity_limits={
                **{n: args.speed * 2 for n in OPENYAM_JOINTS[:-1]}, OPENYAM_JOINTS[-1]: 1.0}),
            TaskConfig(name=f"{OPENYAM_HARDWARE_ID}_gripper", type="gripper", joint_names=[OPENYAM_GRIPPER_JOINT],
                       priority=20),
        ]),
        camera,
    )
    coordinator = ModuleCoordinator.build(blueprint)
    control, keys = coordinator.get_instance(ControlCoordinator), coordinator.get_instance(TerminalKeys)
    rng = np.random.default_rng(args.seed)
    saved = 0

    def wait(seconds):
        """Sleep out a trajectory, but stop the moment the arm stops reporting.

        A wedged CAN bus ("send buffer full after retries") leaves the motors energized holding a stale setpoint
        while the arm sags away from it, and the snap back when the link recovers is violent. Better to cancel and
        make someone look at the wiring than to keep pushing at a bus that is not answering.
        """
        deadline = time.time() + seconds
        while time.time() < deadline:
            time.sleep(0.1)
            try:
                live = len(control.get_joint_positions())
            except Exception as exc:  # the adapter is gone, which is the same emergency
                live, exc_text = 0, str(exc)
            else:
                exc_text = ""
            if live < len(OPENYAM_JOINTS):
                control.cancel_trajectory()
                raise SystemExit(
                    f"\nSTOPPED: the arm reported {live} of {len(OPENYAM_JOINTS)} joints. {exc_text}\n"
                    "The CAN bus stopped answering. Switch the 24 V off, reseat the CAN wires (red CANH, black "
                    "CANL) and the adapter's USB, then re-run arm/arm_probe.py before driving the arm again.")

    def play(motion):  # noqa: F811 - defined before the loop that uses it
        """Run poses from the current state, unrecorded and without noise; wait until done."""
        positions = control.get_joint_positions()
        trajectory, duration = build_trajectory(OPENYAM_JOINTS, [positions[n] for n in OPENYAM_JOINTS], motion,
                                                args.speed, 0.0, rng)
        if control.execute_trajectory(trajectory).status is TrajectoryExecutionStatus.ACCEPTED:
            wait(duration + 0.5)

    print(f"recording to {db}", flush=True)
    try:
        time.sleep(2.0)  # let the camera and joint-state streams start
        for rep in range(args.reps):
            for path, poses in spots.items():
                if args.auto_reset is None:
                    input(f"[{rep + 1}/{args.reps}] Put the bottle on {path}, then press Enter: ")
                else:
                    time.sleep(args.auto_reset)
                play([poses[0]])  # drive to home first, unrecorded: every episode then starts from the same pose
                for _ in range(3):  # the start must match the live state; retry if the arm settled in between
                    positions = control.get_joint_positions()
                    start = [positions[name] for name in OPENYAM_JOINTS]
                    trajectory, duration = build_trajectory(OPENYAM_JOINTS, start, poses[1:], args.speed, args.noise, rng)
                    keys.press("enter")  # start the episode just before the motion
                    result = control.execute_trajectory(trajectory)
                    if result.status is TrajectoryExecutionStatus.ACCEPTED:
                        break
                    keys.press("d")  # discard the empty episode
                    time.sleep(0.2)
                else:
                    raise RuntimeError(f"trajectory rejected: {result.status.name} {result.message or ''}")
                wait(duration + 0.5)
                # Last chance to throw the episode away: a rep the arm was knocked during, or one you leaned into,
                # is worse than no rep at all. The arm holds the grasp while you decide.
                drop = args.auto_reset is None and input(
                    f"[{rep + 1}/{args.reps}] Enter = keep, d = discard: ").strip().lower() == "d"
                keys.press("d" if drop else "enter")
                saved += not drop
                print(f"[{rep + 1}/{args.reps}] {path}: {'DISCARDED' if drop else f'saved ({duration:.1f} s)'}",
                      flush=True)
                if args.hold:  # stay closed on the bottle so the grasp is visible before the preset runs
                    print(f"holding at the lift for {args.hold:.0f} s", flush=True)
                    time.sleep(args.hold)
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
    h = sub.add_parser("handover", help="write the preset played after each pick; it is never recorded")
    h.add_argument("spot")
    h.add_argument("--out", default="vla/spots/handover.json")
    h.add_argument("--rad", type=float, default=0.35, help="radians to raise joint 2 above the grasp")
    h.add_argument("--turn", type=float, default=1.2, help="radians to swing joint 1 round to the person")
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
    r.add_argument("--hold", type=float, default=2.0, help="seconds to hold the closed grasp before the hand-over")
    r.add_argument("--release", action="store_true",
                   help="no hand-over yet: after each pick open the gripper in place, then go home (not recorded)")
    args = ap.parse_args()
    {"teach": lambda: teach(args.spot), "check": lambda: check(args.spot), "record": lambda: record(args),
     "handover": lambda: handover(args.spot, args.out, args.rad, args.turn)}[args.cmd]()


if __name__ == "__main__":
    main()
