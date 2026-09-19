"""Blocker 4: score trigger events against EPIC-KITCHENS put-down labels.

    python blockers/eval_epic.py runs/p02 P02_102
A 'placed' event is a hit if it fires after a labeled bottle put-down starts and within 4 s of its end.
Hits' frames are stacked into <run>/hits.jpg: check by eye that the boxed object is the bottle, at rest.
"""
import json, sys
import pandas as pd

run, vid = sys.argv[1], sys.argv[2]
a = pd.read_csv("data/epic/EPIC_100_train.csv")
d = a[a.video_id == vid].copy()
sec = lambda s: sum(float(x) * m for x, m in zip(s.split(":"), (3600, 60, 1)))
d["t0"], d["t1"] = d.start_timestamp.map(sec), d.stop_timestamp.map(sec)
is_put = d.verb.isin(["put", "put-down", "put-on"])
is_bottle = d.all_nouns.str.contains("bottle") | d.narration.str.contains("oil")
gt = d[is_put & is_bottle].sort_values("t0")
events = [json.loads(l) for l in open(f"{run}/events.jsonl")]
stats = json.load(open(f"{run}/stats.json"))
dur_min = stats["frames"] / 10 / 60

placed = [e for e in events if e["type"] == "placed"]
hit = set()
for _, g in gt.iterrows():
    m = [e for e in placed if g.t0 <= e["t"] <= g.t1 + 4 and e["id"] not in hit]
    if m:
        hit.add(m[0]["id"])
    print(f"GT {g.t0:6.1f}-{g.t1:6.1f}s  {g.narration:38s} -> {'HIT event %d at %.1fs' % (m[0]['id'], m[0]['t']) if m else 'MISSED'}")

print("\nOther events (not matched to a bottle put-down) with the nearest narration:")
for e in events:
    if e["id"] in hit:
        continue
    near = d.iloc[(d.t0 - e["t"]).abs().argsort()[:2]]
    print(f"  {e['type']:8s} {e['t']:6.1f}s  nearest: " + " | ".join(f"{r.t0:.1f}s {r.narration}" for _, r in near.iterrows()))

puts = d[d.verb.str.startswith("put")]
on_any_put = sum(any(g.t0 <= e["t"] <= g.t1 + 4 for _, g in puts.iterrows()) for e in placed)
n_hit = len(hit)
print(f"\nplaced events landing on ANY labeled put-down (any object): {on_any_put}/{len(placed)}")
print(f"recall {n_hit}/{len(gt)} | placed events {len(placed)} ({len(placed) - n_hit} unmatched) | "
      f"sighted {sum(e['type'] == 'sighted' for e in events)} | events/min {len(events) / dur_min:.1f} | {stats}")

episodes = [e for e in events if e["type"] == "arm_episode"]
ep_hit = set()
print("\nArm episodes (the VLM must still confirm each one):")
for _, g in gt.iterrows():
    m = [e for e in episodes if g.t0 <= e["t"] <= g.t1 + 4 and e["id"] not in ep_hit]
    if m:
        ep_hit.add(m[0]["id"])
    print(f"GT {g.t0:6.1f}-{g.t1:6.1f}s  {g.narration:38s} -> {'HIT episode %d at %.1fs' % (m[0]['id'], m[0]['t']) if m else 'MISSED'}")
print(f"arm-episode recall {len(ep_hit)}/{len(gt)} | episodes {len(episodes)} ({len(episodes) / dur_min:.1f}/min = VLM calls)")

import cv2, numpy as np
for name, ids in (("hits.jpg", hit), ("arm_hits.jpg", ep_hit)):
    if ids:
        rows = [np.hstack([cv2.resize(cv2.imread(f), (427, 240)) for f in e["frames"]]) for e in events if e["id"] in ids]
        cv2.imwrite(f"{run}/{name}", np.vstack(rows))
