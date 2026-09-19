"""Check a recording against the poses it was supposed to replay: order, accuracy, and the gripper's direction.

    python vla/check_recording.py <session.db> vla/spots/left.json [--frames out.png]

Reports when each taught pose was reached and how far off it was, fails if they were not reached in the taught order,
and checks the gripper against the taught state at each one. `--frames` writes the camera frame at each pose side by
side, which is the only way to see whether the jaws were really open or shut.
"""
import argparse, json, sqlite3, sys

import numpy as np

REACHED = 0.20  # rad; a pose it stops at settles to ~0.05, one it flies through lags further behind
SETTLE_S = 3.0  # the gripper opens or closes in place after the arm arrives, so read it this long after
# The gripper reads 1 fully open and 0 fully shut. Closing on the bottle stalls it partway, so a grasp that reaches
# 0 is a grasp that closed on air - it missed. Anything in between means the jaws are holding something.
OPEN, SHUT = 0.85, 0.05


def jaws(value):
    return "open" if value > OPEN else "shut" if value < SHUT else "held"


def load(db):
    from dimos.msgs.sensor_msgs.JointState import JointState

    rows = sqlite3.connect(db).execute(
        "select j.ts, b.data from coordinator_joint_state j "
        "join coordinator_joint_state_blob b on b.id = j.id order by j.ts").fetchall()
    if not rows:
        raise SystemExit(f"{db}: no joint states recorded")
    names = list(JointState.lcm_decode(rows[0][1]).name)
    state = np.array([list(JointState.lcm_decode(data).position) for _, data in rows])
    arm = [names.index(f"yam_joint{i}") for i in range(1, 7)]
    return np.array([ts for ts, _ in rows]) - rows[0][0], state[:, arm], state[:, names.index("arm/gripper")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("spot")
    ap.add_argument("--frames", default=None, help="write the camera frame at each pose to this png")
    args = ap.parse_args()

    ts, Q, G = load(args.db)
    poses = json.load(open(args.spot))["poses"]
    print(f"{len(ts)} samples over {ts[-1]:.1f} s\n")
    print(f"{'pose':<9} {'want':>6} {'got':>6} {'err':>6} {'at':>7}  gripper")
    rows, failed = [], []
    for i, pose in enumerate(poses):
        err = np.abs(Q - np.asarray(pose["q"], float)).max(axis=1)
        k = int(err.argmin())
        want = "open" if pose["gripper"] else "held"  # a pose that closes should end up holding the bottle
        # Only a pose that changes the gripper waits for it in place; reading later at any other pose would run
        # past it into the next one.
        moves = i > 0 and pose["gripper"] != poses[i - 1]["gripper"]
        settled = int(np.abs(ts - (ts[k] + SETTLE_S)).argmin()) if moves else k
        got = jaws(G[settled])
        if got != want:
            why = " (shut on nothing - the grasp missed)" if got == "shut" else ""
            failed.append(f"{pose['name']}: jaws {got} where the pose wants {want}{why}")
        rows.append((pose["name"], ts[k], err[k]))
        flag = ""
        if err[k] > REACHED:
            failed.append(f"{pose['name']} was never reached (closest {err[k]:.2f} rad)")
            flag = "  NOT REACHED"
        print(f"{pose['name']:<9} {want:>6} {got:>6} {err[k]:6.3f} {ts[k]:6.1f}s  {G[settled]:.2f}{flag}")

    out_of_order = [b[0] for a, b in zip(rows, rows[1:]) if b[1] <= a[1]]
    if out_of_order:
        failed.append(f"reached out of order: {', '.join(out_of_order)} came too early")

    # Open on the way in, then stalled on the bottle. Fully shut means it grasped nothing.
    grasp = next((i for i, p in enumerate(poses) if not p["gripper"]), None)
    if grasp is not None:
        held = G[(ts >= rows[grasp][1]) & (ts <= rows[grasp][1] + SETTLE_S + 1.0)].min()
        print(f"\ngripper: opened to {G[ts <= rows[grasp][1]].max():.2f} on the way in, "
              f"then stalled at {held:.2f} ({jaws(held)})")
        if G[ts <= rows[grasp][1]].max() < OPEN:
            failed.append("the gripper never opened on the way in - it is inverted")

    if args.frames:
        import cv2
        from dimos.msgs.sensor_msgs.Image import Image

        con = sqlite3.connect(args.db)
        t0 = con.execute("select min(ts) from coordinator_joint_state").fetchone()[0]
        frames = con.execute("select c.ts, b.data from color_image c "
                             "join color_image_blob b on b.id = c.id order by c.ts").fetchall()
        tiles = []
        for name, t, _ in rows:
            fts, blob = min(frames, key=lambda r: abs((r[0] - t0) - t))
            tile = cv2.resize(Image.lcm_decode(blob).to_opencv(), (360, 270))
            cv2.putText(tile, f"{name} {fts - t0:.0f}s", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            tiles.append(tile)
        cv2.imwrite(args.frames, np.hstack(tiles))
        print(f"frames: {args.frames}")

    print("\n" + ("\n".join(f"FAIL: {f}" for f in failed) if failed
                  else "OK: every pose reached, in the taught order, with the gripper the right way round"))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
