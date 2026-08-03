#!/usr/bin/env python

"""Ensemble Sensitivity Analysis (ESA) for the ETC surge ensemble.

This module derives a *sensitivity* statistic from the storm-surge ensemble
produced by ``run_tests.py``.  Following the atmospheric-science ESA literature
(Ancell & Hakim 2007; Torn & Hakim 2008; and doi:10.1175/WAF-D-25-0098.1), the
sensitivity of a scalar response metric ``J`` to a scalar parameter ``x`` across
the ensemble is the univariate linear-regression slope

    S = cov(J, x) / var(x) = dJ/dx .

Here ``J`` is the **peak surge** (max sea-surface elevation over a per-storm
time window) at a single tide gauge and ``x`` is one of the ensemble sweep
parameters.  Storms and gauges are kept separate throughout.

Because the ``run_tests.py`` sweep is a *balanced full factorial*, the
parameters are mutually orthogonal, so each univariate slope equals that
parameter's main-effect coefficient.  A multivariate OLS fit is also computed as
a cross-check; on a balanced design the standardized univariate and multivariate
coefficients agree, and a divergence flags an imbalance (e.g. DEC1992 has only
one resolution, so its resolution sensitivity is undefined and masked).

Metrics reported per (storm, gauge, parameter):

* ``slope``        -- dJ/dx in physical units (m surge per unit parameter),
* ``standardized`` -- ``slope * sigma_x`` (m surge per 1-sigma perturbation);
                      the headline metric, comparable across parameters,
* ``corr``         -- correlation coefficient in [-1, 1],
* ``p``            -- two-tailed p-value for a zero slope,
* ``ci``           -- bootstrap confidence interval on ``slope``.

CAVEATS (see the plan file's "Things to VET"): ``dilation`` is a whole-storm
time-scaling and only a proxy for translation speed (and is sign-inverted
relative to forward speed); ``res`` is a numerical-convergence axis rather than a
physical risk driver; and the *standardized* magnitude scales with the (somewhat
arbitrary) sweep width through ``sigma_x``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
import re

import numpy as np
from scipy import stats

import gauge_comparison as gc

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
# Sweep-parameter columns treated as ESA sensitivity variables, in display
# order.  ``key`` matches the tidy-table column; ``label`` is for plot axes.
# Track-shift columns are carried in the table (zeroed) but excluded here until
# the track-shift runs exist -- a constant column has zero variance and would be
# masked anyway.
PARAMS: list[tuple[str, str]] = [
    ("scaling", "Wind strength"),
    ("dilation", "Translation (dilation)"),
    ("res", "Resolution (deg)"),
    ("track_lon", "Track E/W"),
    ("track_lat", "Track N/S"),
]
# ``sea_level`` is intentionally NOT in PARAMS: it was shown to be a separable
# ~1:1 additive offset (see ``superposition_check``) and is applied post hoc, so
# the sweep no longer varies it.  The column is retained in COLUMNS for the
# superposition diagnostic and for older ensembles that did vary it.

# Table columns, in order.  ``track_lat``/``track_lon`` are reserved for the
# future track-shift sweep so downstream code needs no structural change.
COLUMNS: list[str] = [
    "storm", "gauge", "res", "sea_level", "scaling", "dilation", "levels",
    "track_lat", "track_lon", "peak_surge",
]

# Inverse of ETCJob.prefix:
#   storm<DATE>_<RES>_sea<SEA>_scale<SCALE>_lev<LEV>_dil<DIL>[_tlat<LAT>_tlon<LON>]
# e.g. stormDEC1992_0pt25_sea0.0_scale0.80_lev5_dil0.80
#      stormDEC2012_1pt00_sea0.0_scale1.20_lev5_dil0.80_tlat+0.50_tlon-0.50
# The track-shift suffix is optional so older (pre-track) job dirs still parse,
# defaulting their track shift to zero.
_PREFIX_RE = re.compile(
    r"^storm(?P<storm>[A-Z]+\d+)_(?P<res>[0-9pt]+)"
    r"_sea(?P<sea_level>-?\d+(?:\.\d+)?)"
    r"_scale(?P<scaling>-?\d+(?:\.\d+)?)"
    r"_lev(?P<levels>\d+)"
    r"_dil(?P<dilation>-?\d+(?:\.\d+)?)"
    r"(?:_tlat(?P<track_lat>[-+]?\d+(?:\.\d+)?)"
    r"_tlon(?P<track_lon>[-+]?\d+(?:\.\d+)?))?$"
)


def res_to_degrees(res: str) -> float:
    """Convert a resolution token (``0pt25``) to grid spacing in degrees.

    The ``NpM`` form encodes ``N.M`` with ``pt`` standing in for the decimal
    point in the storm-file names, so ``0pt25`` -> 0.25 and ``1pt00`` -> 1.0.
    """
    return float(res.replace("pt", "."))


def parse_prefix(name: str) -> dict | None:
    """Parse an ``ETCJob.prefix`` job-directory name into its parameters.

    Returns a dict with numeric parameters (resolution decoded to degrees) plus
    the raw ``res`` token, or ``None`` if ``name`` is not an ETC job prefix.
    Kept as the strict inverse of ``run_tests.ETCJob.prefix`` and locked by a
    round-trip test so a naming-format change fails loudly rather than silently
    dropping runs.
    """
    m = _PREFIX_RE.match(name)
    if m is None:
        return None
    g = m.groupdict()
    return {
        "storm": g["storm"],
        "res_token": g["res"],
        "res": res_to_degrees(g["res"]),
        "sea_level": float(g["sea_level"]),
        "scaling": float(g["scaling"]),
        "levels": int(g["levels"]),
        "dilation": float(g["dilation"]),
        # Track shift defaults to zero for older prefixes without the suffix.
        "track_lat": float(g["track_lat"]) if g["track_lat"] else 0.0,
        "track_lon": float(g["track_lon"]) if g["track_lon"] else 0.0,
    }


# ----------------------------------------------------------------------------
# Table construction
# ----------------------------------------------------------------------------
@dataclass
class Table:
    """A tidy table: one row per (storm, gauge, ensemble member).

    Rows are dicts keyed by ``COLUMNS``.  ``storms`` preserves discovery order.
    """
    rows: list[dict] = field(default_factory=list)
    storms: list[str] = field(default_factory=list)
    # gauge id -> (lon, lat), captured from the gauge headers for the map plot.
    locations: dict[int, tuple[float, float]] = field(default_factory=dict)

    def column(self, key: str, rows: list[dict] | None = None) -> np.ndarray:
        rows = self.rows if rows is None else rows
        return np.array([r[key] for r in rows], dtype=float)

    def select(self, storm: str, gauge: int) -> list[dict]:
        return [r for r in self.rows
                if r["storm"] == storm and r["gauge"] == gauge]


def peak_surge(time_days: np.ndarray, surface: np.ndarray,
               window: tuple[float, float] | None) -> tuple[float, bool]:
    """Peak (max) surface elevation over ``window`` (days).

    Returns ``(peak, touched_boundary)``.  ``surface`` is a masked array (dry
    cells masked).  ``peak`` is NaN if the window contains no wet samples.
    ``touched_boundary`` is True when the max sits on the first/last in-window
    sample, a hint that a time-dilated run's peak may be truncated by the window.
    """
    if window is not None:
        in_win = (time_days >= window[0]) & (time_days <= window[1])
    else:
        in_win = np.ones(time_days.shape, dtype=bool)

    valid = in_win & ~np.ma.getmaskarray(surface)
    if not valid.any():
        return float("nan"), False

    vals = np.asarray(surface)[valid]
    idx = int(np.argmax(vals))
    peak = float(vals[idx])
    touched = idx == 0 or idx == vals.size - 1
    return peak, touched


def build_table(run_root: Path,
                storms: list[str] | None = None,
                gauges: range = range(1, 9),
                window: tuple[float, float] | None = None) -> Table:
    """Build the tidy peak-surge table by walking job directories.

    For each job directory under ``run_root`` whose name parses as an ETC job
    prefix, and each gauge, the peak surge is computed over the per-storm time
    window (``gauge_comparison.times[storm][3]``, in days relative to landfall)
    unless ``window`` overrides it.  Members whose gauge output is missing or
    dry throughout the window are logged and skipped, never silently zeroed.
    """
    table = Table()
    seen_storms: list[str] = []

    for job_dir in sorted(p for p in run_root.iterdir() if p.is_dir()):
        params = parse_prefix(job_dir.name)
        if params is None:
            continue
        storm = params["storm"]
        if storms is not None and storm not in storms:
            continue
        win = window
        if win is None:
            cfg = gc.times.get(storm)
            if cfg is None:
                logger.warning("No time window for storm %s; using full series.",
                               storm)
            else:
                win = tuple(cfg[3])
        if storm not in seen_storms:
            seen_storms.append(storm)

        for gauge in gauges:
            try:
                t, surface = gc.load_gauge_surface(gauge, str(job_dir))
            except (FileNotFoundError, OSError) as exc:
                logger.warning("Missing gauge %d in %s: %s",
                               gauge, job_dir.name, exc)
                continue
            peak, touched = peak_surge(t, surface, win)
            if not np.isfinite(peak):
                logger.warning("Gauge %d dry over window in %s; skipping.",
                               gauge, job_dir.name)
                continue
            if touched:
                logger.info("Gauge %d peak sits on window edge in %s "
                            "(possible truncation).", gauge, job_dir.name)
            if gauge not in table.locations:
                try:
                    table.locations[int(gauge)] = tuple(
                        gc.load_gauge_location(gauge, str(job_dir)))
                except (FileNotFoundError, OSError, TypeError):
                    pass
            table.rows.append({
                "storm": storm,
                "gauge": int(gauge),
                "res": params["res"],
                "sea_level": params["sea_level"],
                "scaling": params["scaling"],
                "dilation": params["dilation"],
                "levels": float(params["levels"]),
                "track_lat": params["track_lat"],
                "track_lon": params["track_lon"],
                "peak_surge": peak,
            })

    table.storms = seen_storms
    return table


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------
def sensitivity(J: np.ndarray, x: np.ndarray,
                n_boot: int = 2000, seed: int = 0,
                ci: float = 0.95) -> dict:
    """Univariate ensemble sensitivity of response ``J`` to parameter ``x``.

    Returns a dict with ``slope`` = cov(J,x)/var(x) = dJ/dx, ``standardized`` =
    slope*sigma_x, ``corr``, ``p`` (two-tailed, zero-slope null), and a
    percentile bootstrap ``ci`` = (lo, hi) on the slope.  ``defined`` is False
    (all stats NaN) when there are fewer than three finite pairs or ``x`` is
    constant (zero variance), e.g. a single-resolution storm.

    ``slope`` is computed directly as cov/var (the ESA definition); it equals a
    degree-1 ``polyfit`` slope and ``scipy.stats.linregress`` slope, an identity
    the verification suite asserts.
    """
    J = np.asarray(J, dtype=float)
    x = np.asarray(x, dtype=float)
    good = np.isfinite(J) & np.isfinite(x)
    J, x = J[good], x[good]
    n = J.size

    undefined = {
        "slope": float("nan"), "standardized": float("nan"),
        "corr": float("nan"), "p": float("nan"),
        "ci": (float("nan"), float("nan")), "sigma_x": float("nan"),
        "sigma_J": float("nan"), "n": int(n), "defined": False,
    }
    if n < 3:
        return undefined

    sigma_x = float(np.std(x))
    sigma_J = float(np.std(J))
    var_x = float(np.var(x))
    if var_x == 0.0:
        return undefined

    cov = float(np.mean((x - x.mean()) * (J - J.mean())))
    slope = cov / var_x
    standardized = slope * sigma_x
    corr = cov / (sigma_x * sigma_J) if sigma_J > 0 else float("nan")

    # Two-tailed p-value for the slope via the correlation t-statistic.
    if np.isfinite(corr) and abs(corr) < 1.0:
        t_stat = corr * np.sqrt((n - 2) / (1.0 - corr**2))
        p = float(2.0 * stats.t.sf(abs(t_stat), df=n - 2))
    elif np.isfinite(corr):
        p = 0.0  # perfect linear fit
    else:
        p = float("nan")

    # Percentile bootstrap CI on the slope (robust to the structured design).
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        xb, Jb = x[idx], J[idx]
        vb = np.var(xb)
        boot[b] = (np.mean((xb - xb.mean()) * (Jb - Jb.mean())) / vb
                   if vb > 0 else np.nan)
    lo, hi = np.nanpercentile(boot, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2])

    return {
        "slope": float(slope), "standardized": float(standardized),
        "corr": float(corr), "p": p, "ci": (float(lo), float(hi)),
        "sigma_x": sigma_x, "sigma_J": sigma_J, "n": int(n), "defined": True,
    }


def multivariate(J: np.ndarray, X: dict[str, np.ndarray]) -> dict[str, float]:
    """Standardized multivariate OLS coefficients of ``J`` on parameters ``X``.

    Predictors are z-scored (mean 0, std 1) so each coefficient is the change in
    ``J`` per 1-sigma of that parameter *controlling for the others* -- directly
    comparable to the univariate ``standardized`` sensitivity.  Constant
    predictors (zero variance) are dropped and returned as NaN.  Returns a dict
    keyed by parameter name.
    """
    keys = list(X.keys())
    cols, active = [], []
    for k in keys:
        xk = np.asarray(X[k], dtype=float)
        s = np.std(xk)
        if s > 0:
            cols.append((xk - xk.mean()) / s)
            active.append(k)

    coeffs = {k: float("nan") for k in keys}
    if not active:
        return coeffs

    design = np.column_stack([np.ones_like(J)] + cols)
    beta, *_ = np.linalg.lstsq(design, np.asarray(J, dtype=float), rcond=None)
    for k, b in zip(active, beta[1:]):
        coeffs[k] = float(b)
    return coeffs


def compute_all(table: Table,
                params: list[tuple[str, str]] = PARAMS,
                n_boot: int = 2000, seed: int = 0) -> dict:
    """Compute per (storm, gauge) univariate + multivariate sensitivities.

    Returns ``results[storm][gauge] = {"univariate": {key: sens_dict},
    "multivariate": {key: coef}, "n": n}``.  Gauges are iterated in ascending
    order; storms follow ``table.storms`` discovery order.
    """
    keys = [k for k, _ in params]
    results: dict = {}
    for storm in table.storms:
        gauges = sorted({r["gauge"] for r in table.rows if r["storm"] == storm})
        results[storm] = {}
        for gauge in gauges:
            rows = table.select(storm, gauge)
            J = table.column("peak_surge", rows)
            X = {k: table.column(k, rows) for k in keys}
            uni = {k: sensitivity(J, X[k], n_boot=n_boot, seed=seed)
                   for k in keys}
            multi = multivariate(J, X)
            results[storm][gauge] = {
                "univariate": uni, "multivariate": multi, "n": int(J.size),
            }
    return results


def superposition_check(table: Table, param: str = "sea_level",
                        match_keys: list[str] | None = None) -> dict:
    """Test whether ``param``'s effect on peak surge is a separable offset.

    ESA gives a single averaged slope; this instead asks the sharper question
    behind "can we drop this parameter and add it back linearly?".  It pairs
    ensemble members that differ *only* in ``param`` between its lowest and
    highest swept values (holding ``match_keys`` fixed) and reports, per
    (storm, gauge), the response difference ``delta = J(hi) - J(lo)`` and the
    effective gain ``delta / (hi - lo)``.

    Two things are being verified at once:

    * **separability / no interaction** -- if ``delta`` is nearly constant as
      the *other* parameters vary (small ``std``), the effect is an additive
      offset independent of them, so it can be applied post hoc;
    * **gain** -- ``gain_mean`` near 1 means the peak simply tracks the raised
      baseline ~1:1 (deviation from 1 is the physical amplification/damping).

    This does NOT test linearity *within* ``[lo, hi]``: two swept values define
    a straight line by construction, so curvature at intermediate values is
    unobservable here and needs an extra sweep point to certify.

    Returns ``result[storm][gauge] = {n, mean, std, min, max, gain_mean,
    deltas}`` plus ``result["_meta"] = {param, lo, hi, match_keys}``.
    """
    if match_keys is None:
        match_keys = [k for k, _ in PARAMS if k != param] + ["levels"]

    vals = sorted({r[param] for r in table.rows})
    if len(vals) < 2:
        logger.warning("superposition_check: %s has <2 values; skipping.", param)
        return {"_meta": {"param": param, "lo": None, "hi": None,
                          "match_keys": match_keys}}
    lo, hi = vals[0], vals[-1]
    if len(vals) > 2:
        logger.info("superposition_check: %s has %d values; pairing extremes "
                    "%.3g and %.3g only.", param, len(vals), lo, hi)

    result: dict = {"_meta": {"param": param, "lo": lo, "hi": hi,
                              "match_keys": match_keys}}
    storms = table.storms
    for storm in storms:
        gauges = sorted({r["gauge"] for r in table.rows if r["storm"] == storm})
        result[storm] = {}
        for gauge in gauges:
            rows = table.select(storm, gauge)
            # group by the fixed keys, collect {param_value: peak}
            groups: dict[tuple, dict] = {}
            for r in rows:
                mkey = tuple(r[k] for k in match_keys)
                groups.setdefault(mkey, {})[r[param]] = r["peak_surge"]
            deltas = [g[hi] - g[lo] for g in groups.values()
                      if lo in g and hi in g]
            if not deltas:
                continue
            d = np.array(deltas, dtype=float)
            result[storm][gauge] = {
                "n": int(d.size), "mean": float(d.mean()),
                "std": float(d.std()), "min": float(d.min()),
                "max": float(d.max()), "gain_mean": float(d.mean() / (hi - lo)),
                "deltas": d,
            }
    return result


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
# Significance threshold for the two-tailed zero-slope test.
_ALPHA = 0.05

# Okabe-Ito colorblind-safe categorical palette, one fixed hue per parameter
# (assigned in PARAMS order, never cycled).
_PARAM_COLORS = {
    "scaling": "#0072B2",    # blue
    "dilation": "#E69F00",   # orange
    "res": "#009E73",        # green
    "sea_level": "#CC79A7",  # reddish purple
    "track_lat": "#D55E00",  # vermillion   (reserved for future track sweep)
    "track_lon": "#56B4E9",  # sky blue
}

# Fixed gauge hues for the grouped bar chart (Okabe-Ito, colorblind-safe, one
# per gauge, never cycled).  Gauge 4 (Bridgeport) is dry and never plotted.
_GAUGE_COLORS = {
    1: "#E69F00", 2: "#56B4E9", 3: "#009E73", 4: "#F0E442",
    5: "#0072B2", 6: "#D55E00", 7: "#CC79A7", 8: "#000000",
}

# Fixed storm hues for the cross-storm comparison figure.
_STORM_COLORS = {"DEC1992": "#0072B2", "DEC2012": "#D55E00", "NOV2018": "#009E73"}


def _gauge_label(gauge: int) -> str:
    station_id, station_name = gc.gauge_mapping[gauge]
    return f"{gauge}: {station_name}"


def plot_sensitivity_heatmap(results: dict, storm: str, out_dir: Path,
                             params: list[tuple[str, str]] = PARAMS) -> Path:
    """Plot A -- standardized-sensitivity heatmap for one storm.

    Rows are gauges, columns are parameters, color is standardized sensitivity
    (m surge per 1-sigma perturbation) on a zero-centered diverging scale.
    Cells that fail the significance test are hatched; undefined cells (a
    constant parameter, e.g. single-resolution DEC1992) are drawn grey.
    """
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.colors import TwoSlopeNorm
    import matplotlib.pyplot as plt

    gauges = sorted(results[storm].keys())
    keys = [k for k, _ in params]
    labels = [lab for _, lab in params]

    M = np.full((len(gauges), len(keys)), np.nan)
    for i, g in enumerate(gauges):
        for j, k in enumerate(keys):
            s = results[storm][g]["univariate"][k]
            if s["defined"]:
                M[i, j] = s["standardized"]

    vmax = np.nanmax(np.abs(M)) if np.isfinite(M).any() else 1.0
    vmax = vmax or 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(1.6 + 1.1 * len(keys), 1.0 + 0.5 * len(gauges)))
    ax.set_facecolor("0.85")  # shows through undefined (NaN) cells
    im = ax.imshow(M, cmap="RdBu_r", norm=norm, aspect="auto")

    for i, g in enumerate(gauges):
        for j, k in enumerate(keys):
            s = results[storm][g]["univariate"][k]
            if not s["defined"]:
                continue
            val = M[i, j]
            if s["p"] > _ALPHA:  # not significant -> hatch
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           hatch="////", edgecolor="0.4",
                                           linewidth=0.0))
            txtcol = "white" if abs(val) > 0.6 * vmax else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=txtcol)

    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(gauges)))
    ax.set_yticklabels([_gauge_label(g) for g in gauges], fontsize=8)
    ax.set_xticks(np.arange(-0.5, len(keys)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(gauges)), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Standardized sensitivity\n(m surge per 1$\\sigma$)", fontsize=8)
    ax.set_title(f"Ensemble sensitivity of peak surge — {storm}\n"
                 f"(hatched: not significant at p<{_ALPHA})", fontsize=10)

    fig.tight_layout()
    out = out_dir / f"sensitivity_heatmap_{storm}.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_tornado(results: dict, storm: str, out_dir: Path,
                 params: list[tuple[str, str]] = PARAMS) -> list[Path]:
    """Plot B -- per-gauge tornado bar chart of standardized sensitivity.

    One horizontal-bar figure per gauge: bars are standardized sensitivities,
    sorted by magnitude, colored by parameter (fixed Okabe-Ito hues), with
    bootstrap CI error bars (scaled to the standardized metric).  Bars whose CI
    spans zero are drawn muted -- their sign is not resolved by the ensemble.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    label_of = dict(params)
    outs = []
    for g in sorted(results[storm].keys()):
        uni = results[storm][g]["univariate"]
        items = [(k, uni[k]) for k, _ in params if uni[k]["defined"]]
        if not items:
            continue
        items.sort(key=lambda kv: abs(kv[1]["standardized"]))

        names = [label_of[k] for k, _ in items]
        vals = np.array([s["standardized"] for _, s in items])
        # Bootstrap CI is on the raw slope; scale to standardized units.
        err_lo, err_hi = [], []
        muted = []
        for _, s in items:
            sx = s["sigma_x"]
            lo, hi = s["ci"][0] * sx, s["ci"][1] * sx
            v = s["standardized"]
            err_lo.append(abs(v - lo))
            err_hi.append(abs(hi - v))
            muted.append(lo <= 0.0 <= hi)  # CI spans zero
        colors = [_PARAM_COLORS.get(k, "0.5") for k, _ in items]
        alphas = [0.35 if m else 1.0 for m in muted]

        fig, ax = plt.subplots(figsize=(6.0, 0.6 * len(items) + 1.5))
        y = np.arange(len(items))
        for yi, (v, c, a, el, eh) in enumerate(
                zip(vals, colors, alphas, err_lo, err_hi)):
            ax.barh(yi, v, color=c, alpha=a, height=0.62, zorder=2)
            ax.errorbar(v, yi, xerr=[[el], [eh]], fmt="none", ecolor="0.3",
                        elinewidth=1.0, capsize=3, zorder=3)
        ax.axvline(0.0, color="0.3", linewidth=1.0, zorder=1)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("Standardized sensitivity (m surge per 1$\\sigma$)",
                      fontsize=9)
        ax.grid(axis="x", color="0.85", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.set_title(f"{_gauge_label(g)} — {storm}\n"
                     f"(muted bars: 95% CI spans zero, n={results[storm][g]['n']})",
                     fontsize=10)
        fig.tight_layout()
        out = out_dir / f"tornado_{storm}_gauge{g}.png"
        fig.savefig(out, dpi=300)
        plt.close(fig)
        outs.append(out)
    return outs


def plot_regression_diagnostics(results: dict, table: Table, storm: str,
                                out_dir: Path,
                                params: list[tuple[str, str]] = PARAMS) -> Path:
    """Plot C -- gauges x parameters scatter grid with fitted slope and R^2.

    The author's *vet* view: each panel shows peak surge vs a parameter for
    every ensemble member, the fitted ESA slope, and R^2, so nonlinearity and
    leverage are visible rather than hidden behind a single coefficient.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gauges = sorted(results[storm].keys())
    keys = [k for k, _ in params]
    labels = [lab for _, lab in params]

    fig, axes = plt.subplots(len(gauges), len(keys),
                             figsize=(2.4 * len(keys), 1.8 * len(gauges)),
                             squeeze=False, sharey="row")
    for i, g in enumerate(gauges):
        rows = table.select(storm, g)
        J = table.column("peak_surge", rows)
        for j, k in enumerate(keys):
            ax = axes[i][j]
            x = table.column(k, rows)
            s = results[storm][g]["univariate"][k]
            ax.scatter(x, J, s=10, color=_PARAM_COLORS.get(k, "0.5"),
                       alpha=0.6, edgecolor="none", zorder=2)
            if s["defined"]:
                xx = np.array([x.min(), x.max()])
                intercept = J.mean() - s["slope"] * x.mean()
                ax.plot(xx, s["slope"] * xx + intercept, color="0.2",
                        linewidth=1.2, zorder=3)
                ax.text(0.04, 0.92, f"$R^2$={s['corr']**2:.2f}",
                        transform=ax.transAxes, fontsize=7, va="top")
            ax.tick_params(labelsize=6)
            if i == 0:
                ax.set_title(labels[j], fontsize=8)
            if i == len(gauges) - 1:
                ax.set_xlabel(labels[j], fontsize=7)
            if j == 0:
                ax.set_ylabel(f"{g}: peak (m)", fontsize=7)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)

    fig.suptitle(f"Peak surge vs parameter (ESA fit) — {storm}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = out_dir / f"regression_diagnostics_{storm}.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_gauge_maps(results: dict, table: Table, storm: str, out_dir: Path,
                    params: list[tuple[str, str]] = PARAMS) -> list[Path]:
    """Plot D -- geographic gauge map, one per parameter, for one storm.

    Each gauge is drawn at its ``(lon, lat)`` and colored by the standardized
    sensitivity of its peak surge to the parameter (zero-centered diverging
    scale), mirroring the spatial sensitivity maps of the atmospheric ESA
    literature.  Non-significant gauges get a dashed outline; the axes use a
    latitude-corrected aspect ratio.  No basemap/coastline (cartopy absent) --
    this is a positions-only reference, not a cartographic figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.colors import TwoSlopeNorm
    import matplotlib.pyplot as plt

    if not table.locations:
        logger.warning("No gauge locations captured; skipping maps for %s.", storm)
        return []

    label_of = dict(params)
    outs = []
    for key, _ in params:
        gauges = [g for g in sorted(results[storm])
                  if results[storm][g]["univariate"][key]["defined"]
                  and g in table.locations]
        if not gauges:
            continue  # e.g. resolution for a single-resolution storm

        lons = np.array([table.locations[g][0] for g in gauges])
        lats = np.array([table.locations[g][1] for g in gauges])
        vals = np.array([results[storm][g]["univariate"][key]["standardized"]
                         for g in gauges])
        sig = [results[storm][g]["univariate"][key]["p"] < _ALPHA
               for g in gauges]

        vmax = np.nanmax(np.abs(vals)) or 1.0
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

        fig, ax = plt.subplots(figsize=(6.5, 6.0))
        sc = ax.scatter(lons, lats, c=vals, cmap="RdBu_r", norm=norm, s=260,
                        edgecolor=["black" if s else "0.5" for s in sig],
                        linewidths=[1.5 if s else 1.0 for s in sig],
                        zorder=3)
        # Dashed ring on non-significant gauges (secondary encoding, not color).
        for lon, lat, s in zip(lons, lats, sig):
            if not s:
                ax.scatter(lon, lat, s=320, facecolors="none",
                           edgecolors="0.5", linestyle=":", linewidths=1.2,
                           zorder=4)
        for g, lon, lat in zip(gauges, lons, lats):
            # Labels offset up-and-left to clear the markers; gauge 8 (Atlantic
            # City) sits in the lower-left corner where a left label runs off
            # the axis, so send its label to the upper-right instead.
            if g == 8:
                xytext, ha = (9, 7), "left"
            else:
                xytext, ha = (-9, 7), "right"
            ax.annotate(f"{g}: {gc.gauge_mapping[g][1]}", (lon, lat),
                        textcoords="offset points", xytext=xytext, ha=ha,
                        va="bottom", fontsize=7)

        mean_lat = float(np.mean(lats))
        ax.set_aspect(1.0 / np.cos(np.deg2rad(mean_lat)))
        ax.set_xlabel("Longitude (°E)", fontsize=9)
        ax.set_ylabel("Latitude (°N)", fontsize=9)
        ax.grid(color="0.9", linewidth=0.6)
        ax.set_axisbelow(True)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Standardized sensitivity (m surge per 1$\\sigma$)",
                       fontsize=8)
        ax.set_title(f"Peak-surge sensitivity to {label_of[key]} — {storm}\n"
                     f"(dashed ring: not significant at p<{_ALPHA})",
                     fontsize=10)
        fig.tight_layout()
        out = out_dir / f"gauge_map_{storm}_{key}.png"
        fig.savefig(out, dpi=300)
        plt.close(fig)
        outs.append(out)
    return outs


def plot_superposition(check: dict, storm: str, out_dir: Path,
                       seed: int = 0) -> Path | None:
    """Plot the superposition/additivity check for one storm.

    One figure per storm: gauges on the y-axis, the per-pair response
    difference ``delta = J(hi) - J(lo)`` on the x-axis.  Individual matched
    pairs are drawn as jittered dots (their scatter within a gauge = the
    interaction with the other parameters), with the per-gauge mean marked.  A
    dashed reference line at ``delta = hi - lo`` marks a pure 1:1 additive
    offset, so distance from it is the physical gain and tightness around it is
    the separability.
    """
    meta = check.get("_meta", {})
    lo, hi = meta.get("lo"), meta.get("hi")
    if lo is None or storm not in check or not check[storm]:
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    param = meta["param"]
    offset = hi - lo
    color = _PARAM_COLORS.get(param, "0.4")
    gauges = sorted(check[storm].keys())
    rng = np.random.default_rng(seed)

    fig, ax = plt.subplots(figsize=(6.5, 0.5 * len(gauges) + 1.8))
    for i, g in enumerate(gauges):
        c = check[storm][g]
        d = c["deltas"]
        jit = rng.uniform(-0.14, 0.14, size=d.size)
        ax.scatter(d, np.full(d.size, i) + jit, s=12, color=color,
                   alpha=0.35, edgecolor="none", zorder=2)
        ax.scatter(c["mean"], i, marker="|", s=400, color="0.1",
                   linewidths=1.6, zorder=4)
        ax.annotate(f"gain={c['gain_mean']:.3f}\n±{c['std']/offset*100:.1f}%",
                    (c["mean"], i), textcoords="offset points", xytext=(8, 0),
                    va="center", fontsize=7)
    ax.axvline(offset, color="0.3", linestyle="--", linewidth=1.2, zorder=1,
               label=f"pure 1:1 offset (Δ={offset:g})")
    ax.set_yticks(range(len(gauges)))
    ax.set_yticklabels([_gauge_label(g) for g in gauges], fontsize=8)
    ax.set_xlabel(f"Δ peak surge for {param}: {lo:g}→{hi:g}  (m)", fontsize=9)
    ax.grid(axis="x", color="0.9", linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title(f"Superposition check: {param} additivity — {storm}\n"
                 "dots = matched pairs (scatter = interaction); "
                 "line = per-gauge mean", fontsize=9)
    fig.tight_layout()
    out = out_dir / f"superposition_{param}_{storm}.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def _std_ci(s: dict) -> tuple[float, float]:
    """Bootstrap CI on the *standardized* sensitivity (scale slope CI by sigma_x)."""
    sx = s["sigma_x"]
    return s["ci"][0] * sx, s["ci"][1] * sx


def plot_grouped_bars(results: dict, storm: str, out_dir: Path,
                      params: list[tuple[str, str]] = PARAMS) -> Path:
    """All-gauges grouped bar chart for one storm, clustered by parameter.

    One cluster per parameter, one bar per gauge within it (colored by gauge
    from a fixed colorblind-safe palette), with bootstrap-CI error bars.  Bars
    failing the significance test are muted.  Complements the per-gauge tornado
    (which is magnitude-sorted per gauge) by making a *single* parameter
    directly comparable across gauges.  Undefined cells (a constant parameter,
    e.g. single-resolution DEC1992) simply omit their bar.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = [k for k, _ in params]
    labels = [lab for _, lab in params]
    gauges = sorted(results[storm].keys())
    n_g = len(gauges)
    width = 0.8 / n_g

    fig, ax = plt.subplots(figsize=(1.8 + 1.4 * len(keys), 4.5))
    for gi, g in enumerate(gauges):
        uni = results[storm][g]["univariate"]
        xs, ys, elo, ehi, alphas = [], [], [], [], []
        for j, k in enumerate(keys):
            s = uni[k]
            if not s["defined"]:
                continue
            v = s["standardized"]
            lo, hi = _std_ci(s)
            xs.append(j + (gi - (n_g - 1) / 2) * width)
            ys.append(v)
            elo.append(abs(v - lo))
            ehi.append(abs(hi - v))
            alphas.append(1.0 if s["p"] < _ALPHA else 0.35)
        if not xs:
            continue
        color = _GAUGE_COLORS.get(g, "0.5")
        # bar alpha must be per-bar; draw individually to vary muting
        for x, y, el, eh, a in zip(xs, ys, elo, ehi, alphas):
            ax.bar(x, y, width=width, color=color, alpha=a, zorder=2)
            ax.errorbar(x, y, yerr=[[el], [eh]], fmt="none", ecolor="0.3",
                        elinewidth=0.8, capsize=2, zorder=3)
        ax.bar(np.nan, np.nan, color=color, label=_gauge_label(g))  # legend proxy

    ax.axhline(0.0, color="0.3", linewidth=1.0, zorder=1)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Standardized sensitivity\n(m surge per 1$\\sigma$)", fontsize=9)
    ax.grid(axis="y", color="0.9", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(fontsize=7, ncol=2, loc="best", framealpha=0.9)
    ax.set_title(f"Peak-surge sensitivity by gauge — {storm}\n"
                 f"(muted bars: not significant at p<{_ALPHA})", fontsize=10)
    fig.tight_layout()
    out = out_dir / f"grouped_sensitivity_{storm}.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_storm_comparison(results: dict, out_root: Path,
                          params: list[tuple[str, str]] = PARAMS) -> list[Path]:
    """Cross-storm comparison, one figure per parameter, at ``out_root``.

    For each parameter: x = gauge, one marker-and-error-bar series per storm,
    y = standardized sensitivity.  This is the *robustness* view -- do the
    storms agree on the sign and rank of a parameter's effect across gauges? --
    and deliberately NOT a pooled regression, which would conflate storms'
    different dynamical regimes and absolute surge magnitudes.  A storm missing
    a parameter (e.g. single-resolution DEC1992 for resolution) is simply absent
    from that panel.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    storms = list(results.keys())
    all_gauges = sorted({g for s in storms for g in results[s]})
    gpos = {g: i for i, g in enumerate(all_gauges)}
    outs = []
    for key, label in params:
        fig, ax = plt.subplots(figsize=(1.2 + 0.7 * len(all_gauges), 4.2))
        drawn = False
        for si, storm in enumerate(storms):
            xs, ys, elo, ehi = [], [], [], []
            for g in results[storm]:
                s = results[storm][g]["univariate"][key]
                if not s["defined"]:
                    continue
                v = s["standardized"]
                lo, hi = _std_ci(s)
                # small horizontal dodge per storm so error bars don't overlap
                xs.append(gpos[g] + (si - (len(storms) - 1) / 2) * 0.12)
                ys.append(v)
                elo.append(abs(v - lo))
                ehi.append(abs(hi - v))
            if not xs:
                continue
            drawn = True
            ax.errorbar(xs, ys, yerr=[elo, ehi], fmt="o", markersize=5,
                        color=_STORM_COLORS.get(storm, "0.4"), label=storm,
                        elinewidth=1.0, capsize=2, linestyle="none")
        ax.axhline(0.0, color="0.3", linewidth=1.0, zorder=1)
        ax.set_xticks(range(len(all_gauges)))
        ax.set_xticklabels([_gauge_label(g) for g in all_gauges],
                           rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("Standardized sensitivity\n(m surge per 1$\\sigma$)",
                      fontsize=9)
        ax.grid(axis="y", color="0.9", linewidth=0.6)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        if drawn:
            ax.legend(fontsize=8, loc="best")
        ax.set_title(f"Cross-storm sensitivity to {label}", fontsize=10)
        fig.tight_layout()
        out = out_root / f"comparison_{key}.png"
        fig.savefig(out, dpi=300)
        plt.close(fig)
        outs.append(out)
    return outs


def plot_all(results: dict, table: Table, out_root: Path,
             params: list[tuple[str, str]] = PARAMS) -> None:
    """Write all figures under ``out_root/<storm>/`` for every storm."""
    sea_check = superposition_check(table, param="sea_level")
    for storm in results:
        out_dir = out_root / storm
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_sensitivity_heatmap(results, storm, out_dir, params)
        plot_tornado(results, storm, out_dir, params)
        plot_regression_diagnostics(results, table, storm, out_dir, params)
        plot_gauge_maps(results, table, storm, out_dir, params)
        plot_grouped_bars(results, storm, out_dir, params)
        plot_superposition(sea_check, storm, out_dir)
        logger.info("Wrote sensitivity figures for %s to %s", storm, out_dir)
    # Cross-storm comparison spans all storms; emit once at the root.
    plot_storm_comparison(results, out_root, params)
