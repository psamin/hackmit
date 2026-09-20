"""Fictional patients, and the answer key that goes with them.

    data, key = generate(seed=7)          # 240 patients x 180 days, six real patterns, plenty of decoys
    data, key = generate(seed=7, world=World.null())    # the same, with NO real relationship in it at all

EVERYTHING HERE IS SYNTHETIC. No real person is described. Every export says so in its first line, and the data
object carries `synthetic = True`.

The point is a fair test. We choose the patterns FIRST and write them in the answer key. The analysis
(analysis/findings.py) is only ever handed `data`, never `key`, and is then scored on how many of the planted
patterns it recovered and how many decoys it wrongly called real. That is the only honest way to say a method
"finds patterns" without real ground truth.

--------------------------------------------------------------------------------
THE WORLD
--------------------------------------------------------------------------------
Each patient has a hidden "how well are they today" state z(t): slow ups and downs (bad weeks), plus, for a quarter
of patients, a steady slide over the months. From it, per day:

  repeated_questions   how often they ask the same thing again. Follows z(t) right away.
  doses                two a day (morning, evening); answered on time or not. Follows z(t - 14): the questions
                       change TWO WEEKS BEFORE the doses do.
  night_wakes          times up in the night. Every extra one lowers that day's chance of an on-time dose.
  ... and the small effects: evenings are answered on time less often than mornings; so are weekends.
  ... and 15% of patients have an abrupt, lasting drop in on-time doses on one particular day (a new medicine, an
      illness), whatever else is going on.

Decoys, on purpose:

  social_minutes   higher at weekends, and adherence is lower at weekends. So they correlate, but only through the
                   weekend (a confounder). Adjusting for the weekend should make it vanish.
  sleep_hours      falls when there are more night wakes; has no effect of its own (a mediator).
  steps_k          pure noise.
  noise_00..14     pure noise, ten of them autocorrelated and five drifting like random walks. Naive tests love
                   these: drifting series correlate with anything else that trends.

Records go missing at random (5%), like a sensor that was off or a day nobody answered.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

SYNTHETIC_NOTICE = "SYNTHETIC DATA: invented patients generated for a demonstration. Nobody real is described."


@dataclass(frozen=True)
class World:
    n_patients: int = 240
    days: int = 180
    lag_days: int = 14                 # z(t) shows in the questions now, and in the doses this many days later
    q_on_state: float = 0.55           # log-rate change in questions per unit of z
    adh_on_state: float = -0.9         # log-odds change in an on-time dose per unit of z (lagged)
    evening: float = -0.45             # log-odds, evening dose vs morning dose
    weekend: float = -0.25             # log-odds, weekend vs weekday
    night_wake: float = -0.35          # log-odds per extra night wake
    step_drop: float = -1.2            # log-odds, after an abrupt change
    p_decliner: float = 0.25           # share of patients on a steady slide
    p_step: float = 0.15               # share with an abrupt drop
    weekend_questions: float = 0.18    # weekend raises questions a little too (part of the confounding)
    missing: float = 0.05
    n_noise: int = 15

    @property
    def planted(self) -> bool:
        """False for the null world: nothing is related to anything, so there is nothing to recover."""
        effects = (self.q_on_state, self.adh_on_state, self.evening, self.weekend, self.night_wake, self.step_drop,
                   self.p_decliner, self.p_step)
        return any(e != 0 for e in effects)

    @classmethod
    def null(cls, **kw) -> "World":
        """Nothing is related to anything. Only the weekly rhythm of social_minutes and the per-patient habits remain."""
        return cls(q_on_state=0.0, adh_on_state=0.0, evening=0.0, weekend=0.0, night_wake=0.0, step_drop=0.0,
                   p_decliner=0.0, p_step=0.0, weekend_questions=0.0, **kw)


@dataclass
class Dataset:
    """All the analysis is allowed to see. There is deliberately no field here that hints at the truth."""
    n: int
    days: int
    dow: np.ndarray                    # (days,) weekday of each day, 0 = Monday
    age: np.ndarray                    # (n,)
    metrics: dict[str, np.ndarray]     # name -> (n, days), NaN = missing
    dose_am: np.ndarray                # (n, days) 1 = answered on time, 0 = not, NaN = no record
    dose_pm: np.ndarray
    synthetic: bool = True
    patient_ids: list[str] = field(default_factory=list)

    @property
    def adherence(self) -> np.ndarray:
        """Daily share of the recorded doses answered on time (NaN if neither dose was recorded)."""
        import warnings
        both = np.stack([self.dose_am, self.dose_pm])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)         # a day with neither dose recorded is NaN, not an error
            return np.nanmean(both, axis=0)

    def to_csv(self, path) -> None:
        """One row per patient-day. The first line says it is synthetic; read it back with comment='#'."""
        cols = list(self.metrics)
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# {SYNTHETIC_NOTICE}\n")
            f.write("patient,day,weekday,dose_am,dose_pm," + ",".join(cols) + "\n")
            for i in range(self.n):
                for t in range(self.days):
                    vals = [self.dose_am[i, t], self.dose_pm[i, t]] + [self.metrics[c][i, t] for c in cols]
                    f.write(f"{self.patient_ids[i]},{t},{int(self.dow[t])}," + ",".join("" if np.isnan(v) else f"{v:g}" for v in vals) + "\n")


@dataclass
class Pattern:
    id: str
    what: str
    truth: object                      # the number, or the set, the analysis should recover
    tolerance: str                     # what counts as recovering it


@dataclass
class AnswerKey:
    world: World
    decliner: np.ndarray               # (n,) bool: on a steady slide
    ramp_end: np.ndarray               # (n,) size of the slide by the last day (0 for the rest)
    step_day: np.ndarray               # (n,) day of the abrupt drop, -1 for none
    patterns: list[Pattern]
    decoys: dict[str, str]             # name -> why it is not a real driver

    def to_dict(self) -> dict:
        return {"world": asdict(self.world), "n_decliners": int(self.decliner.sum()), "n_step": int((self.step_day >= 0).sum()),
                "patterns": [{"id": p.id, "what": p.what, "tolerance": p.tolerance} for p in self.patterns],
                "decoys": self.decoys}


def _ar1(rng, shape, phi, sd):
    """Stationary AR(1) along the last axis with marginal standard deviation `sd`."""
    n, d = shape
    out = np.empty(shape)
    out[:, 0] = rng.normal(0, sd, n)
    innov = sd * np.sqrt(1 - phi ** 2)
    for t in range(1, d):
        out[:, t] = phi * out[:, t - 1] + rng.normal(0, innov, n)
    return out


def generate(seed: int = 0, world: World | None = None) -> tuple[Dataset, AnswerKey]:
    w = world or World()
    rng = np.random.default_rng(seed)
    n, d, lag = w.n_patients, w.days, w.lag_days
    dow = np.arange(d) % 7
    weekend = (dow >= 5).astype(float)

    # --- the hidden state z(t), starting `lag` days early so z(t - lag) exists for every day
    z = _ar1(rng, (n, d + lag), 0.94, 0.5)
    decliner = rng.random(n) < w.p_decliner
    ramp_end = np.where(decliner, rng.uniform(1.2, 2.4, n), 0.0)
    t0 = rng.integers(0, d // 2, n)
    t_all = np.arange(-lag, d)[None, :]
    ramp = ramp_end[:, None] * np.clip((t_all - t0[:, None]) / (d - t0[:, None]), 0, None)
    z = z + ramp
    z_now, z_lagged = z[:, lag:], z[:, :d]                 # z(t) and z(t - lag)

    # --- questions: follow z(t) now
    zeta = rng.normal(0, 0.3, n)[:, None]
    q_rate = np.exp(np.log(3.0) + zeta + w.q_on_state * z_now + w.weekend_questions * weekend[None, :])
    questions = rng.poisson(q_rate).astype(float)

    # --- night wakes: independent of everything else, but each one costs a little adherence
    wake_rate = 1.4 * np.exp(rng.normal(0, 0.25, n))[:, None]
    wakes = rng.poisson(np.broadcast_to(wake_rate, (n, d))).astype(float)

    # --- doses
    base = rng.normal(1.7, 0.5, n)[:, None]
    step_day = np.where(rng.random(n) < w.p_step, rng.integers(40, d - 40, n), -1)
    after = (np.arange(d)[None, :] >= step_day[:, None]) & (step_day[:, None] >= 0)
    core = (base + w.adh_on_state * z_lagged + w.weekend * weekend[None, :]
            + w.night_wake * (wakes - wake_rate) + w.step_drop * after)
    p_am = 1 / (1 + np.exp(-core))
    p_pm = 1 / (1 + np.exp(-(core + w.evening)))
    dose_am = (rng.random((n, d)) < p_am).astype(float)
    dose_pm = (rng.random((n, d)) < p_pm).astype(float)

    # --- the other metrics
    sleep = 7.6 - 0.4 * wakes + rng.normal(0, 0.6, (n, d))
    social = np.maximum(0, rng.normal(40, 12, (n, d)) * (1 + 0.35 * weekend[None, :]))
    steps = 3.0 + _ar1(rng, (n, d), 0.5, 0.8)
    metrics = {"repeated_questions": questions, "night_wakes": wakes, "sleep_hours": sleep,
               "social_minutes": social, "steps_k": steps}
    n_ar = w.n_noise * 2 // 3
    for j in range(w.n_noise):
        metrics[f"noise_{j:02d}"] = (_ar1(rng, (n, d), 0.6, 1.0) if j < n_ar
                                     else np.cumsum(rng.normal(0, 0.25, (n, d)), axis=1))      # a drifting random walk

    # --- missing records
    def hole(a):
        a = a.copy()
        a[rng.random(a.shape) < w.missing] = np.nan
        return a
    metrics = {k: hole(v) for k, v in metrics.items()}
    dose_am, dose_pm = hole(dose_am), hole(dose_pm)

    data = Dataset(n=n, days=d, dow=dow, age=np.round(rng.normal(78, 7, n)), metrics=metrics, dose_am=dose_am,
                   dose_pm=dose_pm, patient_ids=[f"S{i:03d}" for i in range(n)])

    patterns = [] if not w.planted else [
        Pattern("rising_questions", "Repeated questions rise steadily for the patients on a slide",
                {i for i in range(n) if decliner[i]}, "the analysis flags at least 60% of them and at least 80% of what it flags is right"),
        Pattern("lead_lag", f"Repeated questions change {lag} days before on-time doses do", lag,
                f"the estimate's 95% interval contains {lag}, excludes 0, and questions come first"),
        Pattern("night_wakes", "Each extra night wake lowers that day's chance of an on-time dose", w.night_wake,
                "significant after correction, right direction, estimate within 35% of the truth"),
        Pattern("evening", "Evening doses are answered on time less often than morning ones", w.evening,
                "significant after correction, right direction, estimate within 35% of the truth"),
        Pattern("weekend", "Weekend doses are answered on time less often (a small effect)", w.weekend,
                "significant after correction, right direction, estimate within 35% of the truth"),
        Pattern("abrupt_drop", "Some patients have an abrupt, lasting drop in on-time doses",
                {i for i in range(n) if step_day[i] >= 0},
                "at least 80% of what it flags is right, it flags at least 20% of them (one patient's noisy daily doses cannot reveal most drops), and it dates them within 7 days (median)"),
    ]
    decoys = {"social_minutes": "higher at weekends, where adherence is lower: linked only through the weekend",
              "sleep_hours": "moves with night wakes, but has no effect of its own",
              "steps_k": "pure noise"}
    decoys.update({f"noise_{j:02d}": ("pure noise, autocorrelated" if j < n_ar else "pure noise, a drifting random walk")
                   for j in range(w.n_noise)})
    key = AnswerKey(world=w, decliner=decliner, ramp_end=ramp_end, step_day=step_day, patterns=patterns, decoys=decoys)
    return data, key
