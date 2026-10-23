# ETC-NSLT-project

Code for running extra-tropical cyclones (ETCs) in GeoClaw for the NASA
Sea-Level Change Team (N-SLCT) project.

## Ensemble runs — `run_tests.py`

`run_tests.py` drives a storm-surge ensemble through the `batch` package.  It
sweeps a **balanced factorial** over these parameters for each storm and writes
one output directory per ensemble member under `$OUTPUT_PATH/ETC_NASA_SLCT/`:

| Parameter        | Sweep field                        | Values                              | Meaning                              |
|------------------|------------------------------------|-------------------------------------|--------------------------------------|
| wind strength    | `scaling`                          | 0.8, 1.0, 1.2                       | storm wind/forcing multiplier        |
| translation      | `time_dilation`                    | 0.8, 1.0, 1.2                       | storm-evolution time-scaling         |
| resolution       | forcing res                        | `0pt25`/`1pt00`/`1pt50`             | forcing grid spacing (deg)           |
| track shift      | `track_shift_lon`/`track_shift_lat` | cardinal star `{0, ±0.5°}`          | rigid N/S/E/W shift of the storm field |

The track shift is a 5-point cardinal star — center + N/S/E/W at 0.5° — applied
via GeoClaw's gridded-forcing registration shift (`x_shift`/`y_shift`, `+x` East,
`+y` North), which translates the storm field across the fixed crop window.
Crossed with the other axes the star keeps `track_lon ⟂ track_lat` and both
orthogonal to the other parameters, so the ESA slopes stay clean (≈405 runs
total).

**Sea level is no longer swept.** The ESA superposition check (below) showed peak
surge responds to sea level as a separable ~1:1 additive offset over `[0, 0.2]`
m, so it is fixed at 0.0 and re-applied post hoc rather than costing a factorial
dimension.  Older ensembles that did sweep it still analyze correctly (the
track-shift suffix in the job-directory name is optional to the parser).

Storms: `DEC1992`, `DEC2012`, `NOV2018`, each at all three forcing resolutions
(the coarse `DEC1992` files are exact subsamples of the 0.25° field, staged as
`DEC1992_1pt00.nc`/`DEC1992_1pt50.nc`).  Runs locally or on NCAR
Derecho (PBS); see the module docstring for the full CLI and Derecho workflow.
Cross-run gauge and performance figures come from `--plot-only`.

### Observed gauge data

Observed residuals (water level − tide prediction, NOAA CO-OPS, metric, GMT) are
fetched by `gauge_comparison.fetch_observed`, shared with `setplot.py`.  NOAA's
6-minute `water_level` product only exists from ~1996, so for `DEC1992` it falls
back to the `hourly_height` product with hourly predictions
(`fetch_noaa_hourly`).  1992 coverage: hourly data for every gauge except
**Kings Point (8516945)** and **New Haven (8465705)**, which have no 1992 record.
The Battery series was cross-checked against an independently supplied file
(`wlev.1992.tx`/`pred.1992.tx`, MHHW datum): identical water levels; predictions
differ by ≤3.6 cm (mean 8 mm), presumably an older harmonic-constant set.

## Ensemble Sensitivity Analysis (ESA) — `run_sensitivity.py`

A *derived statistic* on top of the ensemble, following the atmospheric-science
ESA literature (Ancell & Hakim 2007; Torn & Hakim 2008; doi:10.1175/WAF-D-25-0098.1).
It quantifies how the **peak surge** at each tide gauge responds to each sweep
parameter.  The sensitivity of a scalar response `J` (peak surge) to a parameter
`x` across the ensemble is the univariate linear-regression slope

```
S = cov(J, x) / var(x) = dJ/dx
```

Because the sweep is a *balanced orthogonal design* (a full factorial in
wind/dilation/resolution, crossed with the cardinal-star track shift), the
parameters are mutually uncorrelated, so each univariate slope equals that
parameter's main-effect coefficient — the classic ESA formula is well-posed on
this design.  Storms and gauges are kept **separate** throughout.

`run_sensitivity.py` is purely disk-driven (like `run_tests.py --plot-only`):
run it once the ensemble has finished.

```bash
python run_sensitivity.py                  # all storms, default run label
python run_sensitivity.py --storms DEC2012  # one storm
python run_sensitivity.py --window 0 4      # override the peak window (days)
```

Output lands under `$SHARED_RUNS_PATH/sensitivity/<run-label>/`:

- `peak_surge_table.csv` — tidy table, one row per (storm, gauge, member)
- `sensitivity_summary.csv` — per (storm, gauge, parameter) statistics
- `sea_level_superposition.csv` — the additivity check below
- `<storm>/` — the figures below

### Metrics (all in `sensitivity_summary.csv`)

- `slope` — `dJ/dx` in physical units (m surge per unit parameter)
- `standardized` — `slope · σ_x`, the surge change (m) per **1-σ** parameter
  perturbation; the headline plotted metric, comparable across parameters
- `corr` — correlation coefficient (and `r_squared`)
- `p` — two-tailed p-value for a zero slope; `significant` flags `p < 0.05`
- `ci_lo`/`ci_hi` — percentile bootstrap CI on the slope
- `multivariate_std` — standardized OLS coefficient controlling for the other
  parameters (a cross-check; equals `standardized` on the balanced design)

### Figures (per storm)

- **Sensitivity heatmap** — gauges (rows) × parameters (columns), colored by
  standardized sensitivity on a zero-centered diverging scale.  Cells failing
  the significance test are hatched; undefined cells (a constant parameter,
  e.g. a single-resolution storm) are grey.  The compact "at a glance" view.
- **Tornado diagram, one per gauge** — a horizontal bar chart where each bar is
  one parameter's standardized sensitivity, sorted by magnitude (largest on top)
  and extending left/right from a central zero line.  Sorting by magnitude gives
  the chart its tapering "tornado" shape; it is the standard sensitivity-analysis
  ranking plot, so **the parameter order deliberately changes from gauge to
  gauge** — that is the convention, and cross-gauge comparison of a single
  parameter is handled by the heatmap and grouped-bar figures instead.
  Bootstrap-CI error bars are drawn on each bar, and bars whose 95% CI spans zero
  are muted (their sign is not resolved by the ensemble).
- **Grouped bar, one per storm** — all gauges on one axis, clustered by
  parameter (one colored bar per gauge within each cluster), with CI error bars
  and muted non-significant bars.  The fixed layout complements the tornado: it
  makes a *single* parameter directly comparable across gauges.
- **Regression-diagnostic grid** — peak surge vs. each parameter for every
  member, with the fitted ESA slope and R² per panel.  The *verification* view:
  it exposes nonlinearity or leverage a single slope would hide.
- **Gauge maps, one per parameter** — the gauges drawn at their lon/lat, colored
  by standardized sensitivity (latitude-corrected aspect; non-significant gauges
  ringed).  Positions only — no coastline (cartopy is not a dependency).
- **Sea-level superposition check, one per storm** — see below.

### Cross-storm comparison (`comparison_<param>.png`, at the run-label root)

One figure per parameter overlays the three storms' standardized sensitivities
across gauges (marker + CI per storm).  It answers the **robustness** question —
do the storms agree on the sign and rank of a parameter's effect? — and is
deliberately *not* a pooled regression: pooling raw peak surge across storms
would conflate their different dynamical regimes and absolute surge magnitudes.
Per-storm ESA remains the primary analysis.  A storm missing a parameter (e.g.
a single-resolution storm for resolution) is simply absent from that panel.

### Sea-level superposition check

ESA gives a single averaged slope; the superposition check
(`ensemble_sensitivity.superposition_check`) answers the sharper question *"can
we drop sea level from the ensemble and add it back linearly?"*.  For every pair
of members that differ **only** in sea level (0.0 vs 0.2 m), holding wind,
translation, and resolution fixed, it computes the peak-surge difference
`Δ = J(0.2) − J(0.0)` and the effective gain `Δ / 0.2`.  It tests two things:

- **separability / no interaction** — if `Δ` barely changes as the *other*
  parameters vary (small spread), the sea-level effect is an additive offset
  independent of them;
- **gain** — `gain ≈ 1` means peak surge simply tracks the raised baseline ~1:1;
  the deviation from 1 is the physical amplification/damping.

The `superposition_<param>_<storm>.png` figure shows, per gauge, the individual
matched-pair `Δ` dots (their scatter = the interaction), the per-gauge mean, and
a dashed reference line at `Δ = 0.2` (pure 1:1 offset).

**What the current runs show (0.0 → 0.2 m).** Across all 441 matched pairs the
mean `Δ = 0.198 m` (gain **0.99**), with a within-gauge spread of only
**0.4–4 %** of the 0.2 m offset — i.e. the sea-level effect is a **near-perfect
additive offset with negligible interaction**, and surge passes through very
nearly 1:1 (a slight ≤3.5 % damping in deeper water).  **Conclusion:** over
`[0, 0.2]` sea level can be dropped from the sweep and re-applied post hoc as
`peak(SL) ≈ peak(0) + gain·SL` (flat `gain = 1.0`, or the per-gauge gain in
`sea_level_superposition.csv` for the small damping).  Doing so halves the
factorial and frees that budget for the track shifts.

**Caveat — this does *not* certify linearity.** Only two sea-level values were
run, and two points define a straight line by construction, so curvature at
intermediate values (e.g. 0.1 m) is unobservable here.  The additivity is
verified; strict linearity is a physical argument, not a measurement.  To
certify it, add one intermediate sea-level value to a small sub-sweep and check
`Δ(0→0.1) ≈ Δ(0.1→0.2)`.

### Caveats to keep in mind (see also the module docstring)

- **`time_dilation` is a proxy for translation speed**, not translation speed
  itself: it time-scales the *whole* storm evolution, and larger dilation means a
  *slower* storm, so `dJ/d(dilation)` is sign-inverted relative to forward speed.
- **Resolution is a numerical-convergence axis**, not a physical risk driver —
  its "sensitivity" indicates whether the surge has converged, not coastal risk.
- **Standardized magnitudes scale with the sweep width** through `σ_x`; the raw
  `slope` in the summary CSV is the sweep-independent physical quantity.
- **Gauge 4 (Bridgeport, CT, 8467150) is dry over the window in every run** and
  is therefore excluded from all storms.  This is a known, unexplained issue
  under investigation — not an analysis artifact.

### Method verification

`test_ensemble_sensitivity.py` verifies the estimator (not by example runs):
manufactured-solution slope recovery, algebraic identities
(`cov/var == polyfit slope`, `standardized == slope·σ_x`), the orthogonality of
the balanced design, bootstrap-CI coverage, and a round-trip against the real
`ETCJob.prefix` naming.  Run with `pytest test_ensemble_sensitivity.py` or
`python test_ensemble_sensitivity.py`.

## Track shifts (planned)

North/south and east/west track shifts are not yet implemented.  The ESA tidy
table already reserves zeroed `track_lat`/`track_lon` columns so they slot in as
additional sensitivity parameters once shifted-track runs exist.
