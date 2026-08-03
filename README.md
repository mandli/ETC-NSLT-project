# ETC-NSLT-project

Code for running extra-tropical cyclones (ETCs) in GeoClaw for the NASA
Sea-Level Change Team (N-SLCT) project.

## Ensemble runs — `run_tests.py`

`run_tests.py` drives a storm-surge ensemble through the `batch` package.  It
sweeps a **balanced full factorial** over four parameters for each storm and
writes one output directory per ensemble member under
`$OUTPUT_PATH/ETC_NASA_SLCT/`:

| Parameter        | Sweep field     | Values                | Meaning                         |
|------------------|-----------------|-----------------------|---------------------------------|
| wind strength    | `scaling`       | 0.8, 1.0, 1.2         | storm wind/forcing multiplier   |
| translation      | `time_dilation` | 0.8, 1.0, 1.2         | storm-evolution time-scaling    |
| resolution       | forcing res     | `0pt25`/`1pt00`/`1pt50` | forcing grid spacing (deg)    |
| sea level        | `sea_level`     | 0.0, 0.2              | background sea-level offset (m) |

Storms: `DEC1992` (0.25° only), `DEC2012`, `NOV2018`.  Runs locally or on NCAR
Derecho (PBS); see the module docstring for the full CLI and Derecho workflow.
Cross-run gauge and performance figures come from `--plot-only`.

## Ensemble Sensitivity Analysis (ESA) — `run_sensitivity.py`

A *derived statistic* on top of the ensemble, following the atmospheric-science
ESA literature (Ancell & Hakim 2007; Torn & Hakim 2008; doi:10.1175/WAF-D-25-0098.1).
It quantifies how the **peak surge** at each tide gauge responds to each sweep
parameter.  The sensitivity of a scalar response `J` (peak surge) to a parameter
`x` across the ensemble is the univariate linear-regression slope

```
S = cov(J, x) / var(x) = dJ/dx
```

Because the sweep is a *balanced full factorial* the parameters are mutually
orthogonal, so each univariate slope equals that parameter's main-effect
coefficient — the classic ESA formula is well-posed on this design.  Storms and
gauges are kept **separate** throughout.

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
  e.g. single-resolution `DEC1992`) are grey.  The compact "at a glance" view.
- **Tornado diagram, one per gauge** — a horizontal bar chart where each bar is
  one parameter's standardized sensitivity, sorted by magnitude (largest on top)
  and extending left/right from a central zero line.  Sorting by magnitude gives
  the chart its tapering "tornado" shape; it is the standard sensitivity-analysis
  ranking plot.  Bootstrap-CI error bars are drawn on each bar, and bars whose
  95% CI spans zero are muted (their sign is not resolved by the ensemble).
- **Regression-diagnostic grid** — peak surge vs. each parameter for every
  member, with the fitted ESA slope and R² per panel.  The *verification* view:
  it exposes nonlinearity or leverage a single slope would hide.
- **Gauge maps, one per parameter** — the gauges drawn at their lon/lat, colored
  by standardized sensitivity (latitude-corrected aspect; non-significant gauges
  ringed).  Positions only — no coastline (cartopy is not a dependency).

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
