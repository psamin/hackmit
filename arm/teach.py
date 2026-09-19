"""Teach and replay: the plan's fallback of a scripted pick from a fixed, marked spot.

    python arm/teach.py record poses.json --ids 1,2,3,4,5 --recv-offset 0x10   # motors limp: pose by hand, name each pose
    python arm/teach.py play poses.json --speed 0.3                            # replays the poses in order

Get --ids and --recv-offset from arm_probe.py first. Replay uses the motors' own position-velocity mode, so
--speed (rad/s) caps every joint. The arm has no brakes: disabling drops it, so support it before releasing.
"""
import argparse, json, time

import numpy as np

from arm_probe import open_robot, poll

TOL = 0.03       # rad: a pose counts as reached when every joint is this close
TIMEOUT_S = 15.0


def record(robot, path, ids):
    arm, poses = robot["motors"], []
    print("Motors are disabled. Move the arm by hand, then name the pose (empty name = skip, q = done).")
    while True:
        poll(robot, rounds=3)
        q = [round(float(v), 4) for v in arm.positions()]
        name = input(f"{q}  pose name: ").strip()
        if name == "q":
            break
        if name:
            poses.append({"name": name, "q": q})
    json.dump({"ids": ids, "poses": poses}, open(path, "w"), indent=1)
    print(f"saved {len(poses)} poses to {path}")


def play(robot, path, speed, dwell):
    arm, poses = robot["motors"], json.load(open(path))["poses"]
    input(f"The arm will move through {[p['name'] for p in poses]} at <= {speed} rad/s. "
          "Clear the workspace, keep the power switch in reach, then press Enter.")
    poll(robot)
    arm.set_mode("pos_vel")
    hold = arm.positions().copy()
    arm.pos_vel_control(np.column_stack([hold, np.full(len(hold), speed)]))  # enable in place: no jump
    arm.enable_all()
    robot.tick(5000)
    try:
        for p in poses:
            target = np.array(p["q"])
            cmd = np.column_stack([target, np.full(len(target), speed)])
            t0 = time.perf_counter()
            while True:
                arm.pos_vel_control(cmd)
                robot.tick(5000)
                err = np.abs(arm.positions() - target).max()
                if err < TOL or time.perf_counter() - t0 > TIMEOUT_S:
                    break
                time.sleep(0.02)
            print(f"{p['name']}: max joint error {err:.3f} rad after {time.perf_counter() - t0:.1f}s", flush=True)
            end = time.perf_counter() + dwell
            while time.perf_counter() < end:  # keep commanding so the motors' comm-loss timeout doesn't trip
                arm.pos_vel_control(cmd); robot.tick(5000); time.sleep(0.02)
    except KeyboardInterrupt:
        print("\nstopped")
    hold = np.column_stack([arm.positions(), np.full(len(arm), speed)])
    print("Holding. Support the arm, then press Enter to release (it goes limp).")
    import threading
    released = threading.Event()
    threading.Thread(target=lambda: (input(), released.set()), daemon=True).start()
    while not released.is_set():
        arm.pos_vel_control(hold); robot.tick(5000); time.sleep(0.02)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["record", "play"])
    ap.add_argument("poses")
    ap.add_argument("--ids", default="1,2,3,4,5")
    ap.add_argument("--recv-offset", type=lambda s: int(s, 0), default=0x10)
    ap.add_argument("--speed", type=float, default=0.3, help="rad/s cap per joint")
    ap.add_argument("--dwell", type=float, default=1.0, help="seconds to hold each pose")
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()
    ids = [int(i, 0) for i in args.ids.split(",")]
    if args.action == "play":
        ids = json.load(open(args.poses))["ids"]  # poses only make sense on the motors they were recorded on

    robot, _ = open_robot(ids, lambda s: s + args.recv_offset, args.mock)
    with robot:  # exiting disables every motor
        if args.action == "record":
            record(robot, args.poses, ids)
        else:
            play(robot, args.poses, args.speed, args.dwell)


if __name__ == "__main__":
    main()
