"""Plot functions for comparing gauges across runs"""

import datetime
from pathlib import Path

from batch import JobResult
import matplotlib.pyplot as plt
import numpy as np

import clawpack.geoclaw.util as geoutil
from clawpack.pyclaw import gauges

gauge_mapping = {1: ('8518750', 'The Battery, NY'),
                 2: ('8516945', 'Kings Point, NY'),
                 3: ('8510560', 'Montauk, NY'),
                 4: ('8467150', 'Bridgeport, CT'),
                 5: ('8465705', 'New Haven, CT'),
                 6: ('8452660', 'Newport, RI'),
                 7: ('8531680', 'Sandy Hook, NJ'),
                 8: ('8534720', 'Atlantic City, NJ')}

# Per-storm time configuration.  Fields match setplot.py's storm_times:
#   [landfall_time, noaa_begin, noaa_end, gauge_xlimits]
# gauge_xlimits is [t_start, t_end] in days relative to landfall_time.
times = {"DEC1992": [np.datetime64("1992-12-08T00:00:00.00"),
                        datetime.datetime(1992, 12, 8, 0, 0),
                        datetime.datetime(1992, 12, 16, 0, 0),
                        [2, 6]],
         "DEC2012": [np.datetime64("2012-12-26T00:00"),
                        datetime.datetime(2012, 12, 25, 0, 0),
                        datetime.datetime(2012, 12, 30, 0, 0),
                        [0, 4]],
         "NOV2018": [np.datetime64("2018-11-14T08:00:00.00"),
                        datetime.datetime(2018, 11, 14, 0, 0),
                        datetime.datetime(2018, 11, 18, 0, 0),
                        [0, 4]]}

def load_gauge_surface(gauge_id, output_path, dry_tolerance=1e-16):
    """Load a single gauge's surface-elevation time series from a run directory.

    Reads ``gauge<gauge_id>.txt`` from ``output_path`` via clawpack's
    ``GaugeSolution`` and returns ``(time_days, surface)`` where ``surface`` is
    the sea-surface elevation ``eta`` (``q[3]``) masked to NaN-safe values
    wherever the water depth ``h`` (``q[0]``) falls below ``dry_tolerance``.

    This is the single definition of "surface" shared by the comparison plots
    and the ensemble-sensitivity analysis so both agree on what is plotted vs.
    what a peak surge is computed from.
    """
    gauge = gauges.GaugeSolution(gauge_id, output_path)
    time = gauge.t / (24 * 60**2)  # days
    surface = np.ma.masked_where(np.abs(gauge.q[0, :]) < dry_tolerance,
                                 gauge.q[3, :])
    return time, surface


def load_gauge_location(gauge_id, output_path):
    """Return a gauge's ``(lon, lat)`` from its header in ``output_path``."""
    return gauges.GaugeSolution(gauge_id, output_path).location


def plot_surface(ax, gauge_id, output_path, style, label=None, n_markers=15,
                 dry_tolerance=1e-16, alpha=1.0):
    """Fetch and plot surface data for a single run.

    ``style`` is a dict of matplotlib line kwargs (color, linestyle, marker,
    linewidth) encoding the run's ensemble parameters.  Markers are suppressed
    when ``alpha`` is low so they don't add clutter to ensemble backgrounds.

    Returns ``(time, surface)`` arrays so callers can aggregate across runs.
    """
    print(f"Loading data from: {output_path}")

    time, surface = load_gauge_surface(gauge_id, output_path, dry_tolerance)

    plot_style = dict(style)
    if alpha < 0.5:
        plot_style.pop("marker", None)  # suppress markers on background lines
        markevery = None
    else:
        markevery = max(1, time.size // n_markers)

    ax.plot(time, surface, markevery=markevery, alpha=alpha, label=label,
            **plot_style)
    return time, surface


def _stack_runs(run_data, n_points=500):
    """Interpolate (time, surface) runs onto a common grid over their overlap.

    Returns ``(t_common, stack)`` with ``stack`` of shape ``(n_runs, n_points)``.
    Masked (dry) samples are dropped before interpolation and runs with fewer
    than two valid samples are skipped, so ``stack`` may have fewer rows than
    ``run_data`` -- or none, in which case ``t_common`` is ``None``.
    """
    if not run_data:
        return None, np.empty((0, n_points))

    t_min = max(t[0] for t, _ in run_data)
    t_max = min(t[-1] for t, _ in run_data)
    if t_min >= t_max:
        return None, np.empty((0, n_points))
    t_common = np.linspace(t_min, t_max, n_points)

    interped = []
    for t, s in run_data:
        valid = ~np.ma.getmaskarray(s)
        if valid.sum() < 2:
            continue
        interped.append(np.interp(t_common, t[valid], np.ma.getdata(s)[valid]))

    return t_common, np.array(interped).reshape(len(interped), n_points)


def plot_ensemble_band(ax, run_data, color="steelblue", n_points=500):
    """Overlay min/max envelope and mean over a list of (time, surface) pairs.

    Interpolates all runs onto a common time grid spanning their shared overlap,
    then draws a lightly filled min/max band and a bold dashed mean line.  The
    separate low-alpha individual lines (drawn by the caller) carry clustering
    information that a pure envelope would hide.
    """
    t_common, stack = _stack_runs(run_data, n_points)
    if len(stack) < 2:
        return

    mean = np.mean(stack, axis=0)
    lo = np.min(stack, axis=0)
    hi = np.max(stack, axis=0)

    ax.fill_between(t_common, lo, hi, alpha=0.15, color=color,
                    label="ensemble range")
    ax.plot(t_common, mean, color=color, linewidth=2.0, linestyle="--",
            label="ensemble mean")

def fetch_observed(gauge_number, times):
    """Fetch a NOAA gauge record and return its detided residual.

    Returns ``(t_days, residual)`` with time in days relative to the storm's
    landfall time, or ``None`` if the fetch failed.  Both the water level and
    the tide predictions come from ``fetch_noaa_tide_data``'s defaults (station
    datum ``STND``, metric, GMT), so the datum cancels in the subtraction.  No
    baseline is removed: the residual still carries any non-tidal offset
    (e.g. sea-level rise since the tidal epoch, monthly MSL anomaly).
    """
    station_id, station_name = gauge_mapping[gauge_number]

    landfall_time = times[0]
    begin_date = times[1]
    end_date = times[2]

    # Fetch data if needed
    date_time, water_level, tide = geoutil.fetch_noaa_tide_data(station_id,
                                                                begin_date,
                                                                end_date,
                                                                verbose=False)

    if water_level is None:
        print("*** Could not fetch gauge {}.".format(station_id))
        return None

    # Convert to seconds relative to landfall
    t = (date_time - landfall_time) / np.timedelta64(1, 's')
    t /= (24 * 60**2)

    # Detide
    water_level -= tide
    return t, water_level


def plot_observed(ax, gauge_number, times):
    """Fetch and plot observed NOAA tide-gauge data."""
    observed = fetch_observed(gauge_number, times)
    if observed is not None:
        t, water_level = observed
        # Plot data
        ax.plot(t, water_level, color='lightgray', marker='x',
                                label="observed")
        # ax.set_title(station_name)
        # ax.legend()


def prestorm_baseline(t_days, residual, window=(-1.0, 0.0)):
    """Mean observed residual over ``window`` (days rel. landfall), or NaN.

    The default window is the day before the model start, which the NOAA fetch
    already covers but the comparison x-limits hide.  Missing samples (NaN)
    are ignored.
    """
    in_window = (t_days >= window[0]) & (t_days < window[1])
    values = np.asarray(residual, dtype=float)[in_window]
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float("nan")


def days_to_utc(landfall_time, t_days):
    """Convert days relative to ``landfall_time`` to UTC ``datetime64[ms]``."""
    offset_ms = np.round(np.asarray(t_days, dtype=float) * 86400e3)
    return np.datetime64(landfall_time, "ms") + offset_ms.astype("timedelta64[ms]")


def event_date_label(landfall_time, xlimits):
    """Human date span for the plotted window, e.g. ``"26–30 Dec 2012"``."""
    start, end = (days_to_utc(landfall_time, x).astype(datetime.datetime)
                  for x in xlimits)
    if (start.year, start.month) == (end.year, end.month):
        return f"{start.day}–{end.day} {start:%b %Y}"
    if start.year == end.year:
        return f"{start.day} {start:%b} – {end.day} {end:%b %Y}"
    return f"{start.day} {start:%b %Y} – {end.day} {end:%b %Y}"


# Presentation-mode styling.  Colorblind-safe (Okabe-Ito) hues in fixed order
# by resolution, fine -> coarse; validated with the dataviz palette checker.
PRESENTATION_COLORS = ["#0072B2", "#D55E00", "#009E73"]
PRESENTATION_RC = {
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
}


def plot_presentation(runs, out_path, storm_date="DEC2012", gauge_id=5,
                      style="envelope", baseline=True,
                      formats=("pdf", "png"), dpi=300):
    """Single slide-ready gauge figure with runs grouped by forcing resolution.

    ``runs`` is a list of ``(resolution_degrees, output_dir)`` pairs; every run
    sharing a resolution forms one group.  ``style`` is ``"envelope"`` (min/max
    band + median line per resolution) or ``"lines"`` (thin individual lines,
    one legend entry per resolution).  Observed residual is drawn last, in
    black, on top.  With ``baseline`` the observed pre-storm mean residual is
    drawn as a dotted reference line so any non-tidal offset stays visible.

    Writes ``<out_path>/gauge_<id>_<storm>_presentation_<style>.<fmt>`` for
    each format and returns the list of written paths.
    """
    if style not in ("envelope", "lines"):
        raise ValueError(f"Unknown presentation style: {style!r}")
    import matplotlib.dates as mdates

    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    storm_cfg = times[storm_date]
    landfall_time = storm_cfg[0]
    gauge_xlimits = storm_cfg[3]
    station_id, station_name = gauge_mapping[gauge_id]

    groups = {}
    for res, output_dir in runs:
        t, s = load_gauge_surface(gauge_id, output_dir)
        groups.setdefault(float(res), []).append((t, s))

    with plt.rc_context(PRESENTATION_RC):
        fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")

        for i, res in enumerate(sorted(groups)):
            color = PRESENTATION_COLORS[i % len(PRESENTATION_COLORS)]
            label = f"{res:g}° forcing"
            if style == "envelope":
                t_common, stack = _stack_runs(groups[res])
                if t_common is None or len(stack) == 0:
                    continue
                dates = days_to_utc(landfall_time, t_common)
                ax.fill_between(dates, stack.min(axis=0), stack.max(axis=0),
                                color=color, alpha=0.25, linewidth=0,
                                zorder=2)
                ax.plot(dates, np.median(stack, axis=0), color=color,
                        linewidth=2, label=label, zorder=3)
            else:
                for j, (t, s) in enumerate(groups[res]):
                    ax.plot(days_to_utc(landfall_time, t), s, color=color,
                            linewidth=1, alpha=0.6,
                            label=label if j == 0 else None, zorder=3)

        observed = fetch_observed(gauge_id, storm_cfg)
        if observed is not None:
            t_obs, residual = observed
            if baseline:
                level = prestorm_baseline(t_obs, residual)
                if np.isfinite(level):
                    ax.axhline(level, color="0.35", linestyle=":",
                               linewidth=1.5, zorder=4,
                               label=f"Obs. pre-storm mean ({level:+.2f} m)")
            ax.plot(days_to_utc(landfall_time, t_obs), residual,
                    color="black", linewidth=2, label="Observed", zorder=10)

        ax.set_xlim(days_to_utc(landfall_time, gauge_xlimits))
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[12]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d %HZ"))
        ax.set_ylabel("Surge (m)")
        ax.set_title(f"{station_name} ({station_id}), "
                     f"{event_date_label(landfall_time, gauge_xlimits)}")
        ax.grid(True, color="0.9", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)

        # Observed (and its baseline) first in the legend, then resolutions.
        handles, labels = ax.get_legend_handles_labels()
        order = sorted(range(len(labels)),
                       key=lambda k: (not labels[k].startswith("Obs"), k))
        ax.legend([handles[k] for k in order], [labels[k] for k in order],
                  loc="upper right", frameon=False)

        written = []
        for fmt in formats:
            path = out_path / (f"gauge_{gauge_id}_{storm_date}"
                               f"_presentation_{style}.{fmt}")
            fig.savefig(path, dpi=dpi)
            written.append(path)
        plt.close(fig)
    return written


def plot(results: list[JobResult], out_path: Path, storm_date: str,
         styles: list[dict], labels: list[str] | None = None,
         subtitle: str = "") -> None:
    """Plot gauge comparisons for a given storm date.

    All ``results`` are overlaid on each gauge's plot, styled by ``styles``
    (one matplotlib-kwargs dict per result, parallel to ``results``).

    When ``labels`` is provided each line is labelled and an in-plot legend is
    drawn; otherwise lines are drawn at low alpha with a min/max ensemble band.
    ``subtitle`` (e.g. "Dilation=1.00, Scaling=1.20") is appended to the title
    on a second line.
    """
    figure_path = out_path
    figure_path.mkdir(parents=True, exist_ok=True)

    storm_cfg = times[storm_date]
    gauge_xlimits = storm_cfg[3]
    labeled = labels is not None

    for gauge_id in range(1, 9):
        fig, ax = plt.subplots()
        plot_observed(ax, gauge_id, storm_cfg)

        run_data = []
        for i, (result, style) in enumerate(zip(results, styles)):
            lbl = labels[i] if labeled else None
            alpha = 1.0 if labeled else 0.2
            t, s = plot_surface(ax, gauge_id, result.paths.job, style,
                                label=lbl, alpha=alpha)
            run_data.append((t, s))

        if not labeled:
            plot_ensemble_band(ax, run_data)

        if labeled:
            ax.legend(fontsize=7, loc="upper left")

        ax.set_xlim(gauge_xlimits)
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Surface Elevation (m)")

        station_id, station_name = gauge_mapping[gauge_id]
        title = f"{station_name} ({station_id}) - {storm_date}"
        if subtitle:
            title = f"{title}\n{subtitle}"
        ax.set_title(title, fontsize=10)

        fig.tight_layout()
        fig.savefig(figure_path / f"gauge_{gauge_id}_surface_comparison.png", dpi=300)
        plt.close(fig)
