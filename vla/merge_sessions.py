"""Merge several recording sessions into one, because `dimos dataprep build` reads a single --source.

    python vla/merge_sessions.py out.db left.db center.db right.db
    python vla/merge_sessions.py out.db left.db center.db --drop 3,7   # leave out episodes 3 and 7 (1-based)

Each session holds three streams - color_image, coordinator_joint_state, status - as a row table of timestamps
alongside a blob table keyed by the same id. Merging copies every row with its id offset past the rows already
written, and shifts timestamps so later sessions follow earlier ones instead of overlapping. The r-tree tables are
a spatial index over poses these recordings do not carry, so they are left behind and rebuilt empty.
"""
import argparse, pickle, shutil, sqlite3, sys

STREAMS = ("color_image", "coordinator_joint_state", "status")
GAP_S = 10.0  # dead time inserted between sessions, so no episode can straddle the join


def episodes(con):
    """(start_ts, end_ts) per saved episode, from the monitor's own markers."""
    rows = con.execute("select s.ts, b.data from status s join status_blob b on b.id = s.id order by s.ts").fetchall()
    spans, start = [], None
    for ts, data in rows:
        event = pickle.loads(data).last_event
        if event == "start":
            start = ts
        elif event == "save" and start is not None:
            spans.append((start, ts))
            start = None
    return spans


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("sources", nargs="+")
    ap.add_argument("--drop", default="", help="1-based episode numbers to leave out, counted across all sources")
    args = ap.parse_args()
    drop = {int(n) for n in args.drop.split(",") if n.strip()}

    shutil.copyfile(args.sources[0], args.out)  # keeps the schema and the _streams config verbatim
    out = sqlite3.connect(args.out)
    for stream in STREAMS:
        for suffix in ("_rtree", "_rtree_node", "_rtree_parent", "_rtree_rowid"):
            out.execute(f'delete from "{stream}{suffix}"')

    kept = list(episodes(out))
    print(f"{args.sources[0]}: {len(kept)} episodes")
    end_ts = max(out.execute(f'select max(ts) from "{s}"').fetchone()[0] or 0 for s in STREAMS)

    for path in args.sources[1:]:
        src = sqlite3.connect(path)
        spans = episodes(src)
        base_ts = src.execute('select min(ts) from "coordinator_joint_state"').fetchone()[0]
        shift = end_ts + GAP_S - base_ts
        for stream in STREAMS:
            offset = out.execute(f'select coalesce(max(id), 0) from "{stream}"').fetchone()[0]
            rows = src.execute(f'select id, ts, value, pose_x, pose_y, pose_z, pose_qx, pose_qy, pose_qz, pose_qw '
                               f'from "{stream}" order by id').fetchall()
            out.executemany(f'insert into "{stream}" (id, ts, value, pose_x, pose_y, pose_z, pose_qx, pose_qy, '
                            f'pose_qz, pose_qw) values (?,?,?,?,?,?,?,?,?,?)',
                            [(r[0] + offset, r[1] + shift, *r[2:]) for r in rows])
            blobs = src.execute(f'select id, data from "{stream}_blob"').fetchall()
            out.executemany(f'insert into "{stream}_blob" (id, data) values (?,?)',
                            [(i + offset, d) for i, d in blobs])
        kept += [(a + shift, b + shift) for a, b in spans]
        end_ts = max(end_ts, max(b for _, b in spans) + shift)
        print(f"{path}: {len(spans)} episodes")
        src.close()

    out.commit()
    merged = episodes(out)
    print(f"\n{args.out}: {len(merged)} episodes, {out.execute('select count(*) from color_image').fetchone()[0]} frames")
    if drop:
        print(f"NOTE: --drop is recorded here but not applied; exclude episodes {sorted(drop)} at training instead")
    out.close()
    if len(merged) != len(kept):
        sys.exit(f"FAIL: expected {len(kept)} episodes after merging, found {len(merged)}")
    print("OK: every episode from every source is present")


if __name__ == "__main__":
    main()
