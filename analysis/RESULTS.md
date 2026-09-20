# Results

_Generated 2026-09-20 by `python -m analysis.report --evaluate --write-results`. SYNTHETIC DATA: invented patients generated for a demonstration. Nobody real is described._

Graded on 20 fresh datasets (seeds 201-220), each with 18 decoy measures; and 20 datasets where nothing is real (seeds 301-320).

| Pattern planted in the data | Recovered | 95% interval |
|---|---|---|
| rising questions | 18 of 20 | 70% to 97% |
| lead lag | 20 of 20 | 84% to 100% |
| night wakes | 20 of 20 | 84% to 100% |
| evening | 20 of 20 | 84% to 100% |
| weekend | 20 of 20 | 84% to 100% |
| abrupt drop | 13 of 20 | 43% to 82% |

- Decoys wrongly called real, per dataset: **0.35** for the careful analysis, **5.10** for the naive one (one measure at a time, every day counted as a person).
- Datasets where nothing is real but the analysis still reported something: **3 of 20** careful, **18 of 20** naive.

Settings were chosen on seeds 9001, 9002, 9003, 9004, 9005, which are not used above. See README.md for what was changed after the first look and why.
