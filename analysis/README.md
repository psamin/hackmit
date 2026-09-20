# Finding patterns in patient data, and proving the method works

> **Everything here is synthetic.** The patients are invented. Nobody real is described, and every file the code writes says so.

The question this answers: *if Pam records how a person is doing every day, can we find real patterns in it, and how do we know we are not fooling ourselves?*

The method is **plant, blind, score**:

1. **Plant.** Generate fictional patients and write down, in advance, the real patterns in them (the *answer key*), plus decoys that look like patterns but are not.
2. **Blind.** Hand the analysis only the data. It is never given the key.
3. **Score.** Grade it against the key: how many planted patterns did it recover, and how many decoys did it wrongly call real? Repeat over many fresh datasets, and over datasets where *nothing* is real.

That is the only honest way to say a method "finds patterns" when you have no real ground truth. Anyone can find patterns in noise. The result that matters is how often it finds the *real* ones and how rarely it calls noise real.

## Run it

From the repo root (uses the same Python environment as the rest of the project: numpy, scipy, pandas):

```bash
python -m analysis.report                      # one synthetic cohort, in plain words
python -m analysis.report --evaluate           # the scorecard: 20 datasets + 20 where nothing is real (about 100 s)
python analysis/test_analysis.py               # 52 tests
```

Output files go to `analysis/out/` (not committed). The committed scorecard is [RESULTS.md](RESULTS.md).

## The synthetic world

240 people, 180 days each, two doses a day (morning and evening), answered on time or not. Each person has a hidden "how well are they today" state with slow ups and downs, and a quarter of them also slide steadily over the months.

| Planted pattern (the answer key) | What it means |
|---|---|
| **Rising questions** | Repeated questions rise steadily for the people on a slide |
| **Lead-lag** | Questions change **14 days before** on-time doses do |
| **Night wakes** | Each extra wake-up lowers that day's chance of an on-time dose (odds ratio 0.70) |
| **Evening** | Evening doses are answered on time less often than morning ones (OR 0.64) |
| **Weekend** | A small weekend dip (OR 0.78), faint on purpose so it is hard to catch |
| **Abrupt drop** | 15% of people have a sudden, lasting drop on one particular day |

| Decoy (not real drivers) | Why it is a trap |
|---|---|
| **Social minutes** | Higher at weekends, when adherence is lower: linked *only through the weekend* (a confounder) |
| **Sleep hours** | Falls with night wakes but does nothing on its own (a mediator) |
| **Steps** | Pure noise |
| **15 noise measures** | Ten autocorrelated, five drifting like random walks: naive tests love these |

Five percent of records are missing at random. A second world, `World.null()`, has **no real relationship at all**, to measure false alarms directly.

## What the analysis does, and why

| Question | Method | What it guards against |
|---|---|---|
| Who is on a steady slide? | Kendall trend per person on weekly means, corrected across 240 people, plus a minimum size | Testing 240 people finds "trends" by luck; bad weeks look like slides |
| What leads what? | Cross-correlation of questions vs doses per person; interval by resampling people; p-value by shifting each person's questions | Slow rhythms correlate with anything; shifting keeps the rhythm and breaks any real link |
| Who had an abrupt drop? | Best single step per person, compared with that person's own shuffled series, corrected across people; must fit better than a straight line | A slide is not a step; noise produces step-like blips |
| What drives a day's doses? | **One** logistic regression on every candidate at once, with weekday and each person's baseline, standard errors clustered by person | Confounders (weekend) and mediators (sleep) vanish when fitted together; 360 days of one person are not 360 people |
| The naive version | One correlation at a time, every patient-day independent | Kept only to show what it gets wrong |

Corrections for testing many things at once use **Benjamini-Hochberg** (controls the share of false discoveries). The lead-lag question is told the answer is *timing, not cause*.

Everything is in plain numpy/scipy so each step can be read and explained.

## Results

Graded on **fresh** seeds nothing was tuned on (full table in [RESULTS.md](RESULTS.md)):

| Planted pattern | Recovered (20 datasets) |
|---|---|
| Lead-lag (14 days) | 20 of 20 |
| Night wakes, evening, weekend | 20 of 20 each |
| Rising questions | 18 of 20 |
| Abrupt drops | 13 of 20 |

- Decoys wrongly called real, per dataset (18 decoys): **0.35** for the careful analysis vs **5.1** for the naive one.
- Datasets where *nothing* is real but it still reported something: **3 of 20** careful vs **18 of 20** naive.

The careful analysis is not perfect, and that is reported, not hidden:
- It catches only some abrupt drops. One person's daily doses are noisy, so most drops are undetectable; what it does flag it usually dates to within a few days.
- Its false-discovery rate among the decoys is about 10%, not the 5% it aims for. Random-walk noise is a known trouble spot for regression; a wild cluster bootstrap is the next improvement.

## What we changed after the first look, and why

Choosing thresholds by looking at the answer you are then graded on is how people fool themselves, so:

- Settings (`findings.Config`) were chosen on **development seeds 9001-9005** only.
- Development showed two problems that were fixed: a units mistake in the "slide size" calculation, and abrupt-drop detection that was weak. It now first removes the part of a person's doses their own earlier questions explain (using the lag found in the same data), then tests steps with shuffled-series p-values corrected across people.
- Then the scorecard was run once on **seeds 1-20**. It showed `sleep_hours` (a decoy) flagged in 5 of 20 datasets. The cause: missing night-wake counts were filled with an average, so sleep, which is recorded, was standing in for wakes. The fix is standard: use only days where every measure was recorded.
- Because that change was made after looking at seeds 1-20, those seeds are **retired**. The reported numbers come from fresh seeds **201-220** (planted) and **301-320** (nothing real). A test enforces that the seed sets never overlap.

## What this does not show

- **It is synthetic.** It proves a method works when its assumptions hold, not that it works on real, messy patient records. The generator and the analysis were written by the same team; the difficulty (noise, confounding, decoys, gaps) is real, but it is chosen difficulty.
- **Patterns are not diagnoses.** Everything is worded as "worth mentioning to the doctor". A test scans every sentence for diagnosis-like wording.
- **Real data would need consent and clinical review** before any of this touched a real person.

## If someone asks how you did it

> *"We can't use real patient data, so we built fictional patients and decided in advance which patterns were real, writing them in an answer key. The analysis never sees the key. We then graded it: across 20 fresh datasets it recovered the 14-day lead between repeated questions and missed doses every time, and the night, evening and weekend effects every time, and it called a noise measure real about once every three datasets (0.35 per dataset), where the obvious one-at-a-time approach was fooled about five times per dataset. In datasets where nothing is real, it raised a false alarm 3 times in 20; the obvious approach, 18 in 20."*

Likely follow-ups, and the honest answers:

- **"How do you know it isn't overfit?"** Settings were chosen on seeds nobody is graded on; results are on fresh seeds; and we report the seeds we used up and what we changed after looking.
- **"Why should I believe a pattern?"** Each has an estimate, an interval, and a corrected p-value, was fitted alongside every alternative explanation, and comes with a caveat. We also show what did *not* hold up.
- **"Correlation isn't causation."** Agreed, and the text says so: timing and association only.
- **"What about real data?"** Same pipeline, but it needs consent, review, and a check that its assumptions hold. The scorecard is how we would find out.
- **"Did AI write this?"** Built with an AI coding assistant (Claude Code). Be ready to explain each step yourself: the answer key, why one joint regression beats many correlations, what clustering and Benjamini-Hochberg do, and what the lead-lag shift test is.

## Files

| File | What it does |
|---|---|
| `synth.py` | The synthetic world and its answer key. Every export says it is synthetic |
| `findings.py` | The blind analysis. It only ever receives a `Dataset` |
| `stats.py` | Small tested statistics: Benjamini-Hochberg, trend, clustered logistic regression, cross-correlation |
| `scorecard.py` | Grades an analysis against the key; many seeds; the seed sets |
| `insights.py` | Turns findings into plain sentences with caveats; no number that wasn't computed |
| `report.py` | The command-line report and scoreboard |
| `test_analysis.py` | 52 tests, checked against a 22-way deliberate-breakage run |
