"""Plot functions for comparing gauges across runs"""

from pathlib import Path
import datetime

import matplotlib.pyplot as plt
import numpy as np

from batch import JobResult

from clawpack.pyclaw import gauges
import clawpack.geoclaw.util as geoutil

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

def plot_surface(ax, gauge_id, output_path, style, label=None, n_markers=15,
                 dry_tolerance=1e-16, alpha=1.0):
    """Fetch and plot surface data for a single run.

    ``style`` is a dict of matplotlib line kwargs (color, linestyle, marker,
    linewidth) encoding the run's ensemble parameters.  Markers are suppressed
    when ``alpha`` is low so they don't add clutter to ensemble backgrounds.

    Returns ``(time, surface)`` arrays so callers can aggregate across runs.
    """
    print(f"Loading data from: {output_path}")

    gauge = gauges.GaugeSolution(gauge_id, output_path)
    time = gauge.t / (24 * 60**2)  # days
    surface = np.ma.masked_where(np.abs(gauge.q[0, :]) < dry_tolerance,
                                 gauge.q[3, :])

    plot_style = dict(style)
    if alpha < 0.5:
        plot_style.pop("marker", None)  # suppress markers on background lines
        markevery = None
    else:
        markevery = max(1, time.size // n_markers)

    ax.plot(time, surface, markevery=markevery, alpha=alpha, label=label,
            **plot_style)
    return time, surface


def plot_ensemble_band(ax, run_data, color="steelblue", n_points=500):
    """Overlay min/max envelope and mean over a list of (time, surface) pairs.

    Interpolates all runs onto a common time grid spanning their shared overlap,
    then draws a lightly filled min/max band and a bold dashed mean line.  The
    separate low-alpha individual lines (drawn by the caller) carry clustering
    information that a pure envelope would hide.
    """
    if not run_data:
        return

    t_min = max(t[0] for t, _ in run_data)
    t_max = min(t[-1] for t, _ in run_data)
    if t_min >= t_max:
        return
    t_common = np.linspace(t_min, t_max, n_points)

    interped = []
    for t, s in run_data:
        valid = ~np.ma.getmaskarray(s)
        if valid.sum() < 2:
            continue
        interped.append(np.interp(t_common, t[valid], s.data[valid]))

    if len(interped) < 2:
        return

    stack = np.array(interped)
    mean = np.mean(stack, axis=0)
    lo = np.min(stack, axis=0)
    hi = np.max(stack, axis=0)

    ax.fill_between(t_common, lo, hi, alpha=0.15, color=color,
                    label="ensemble range")
    ax.plot(t_common, mean, color=color, linewidth=2.0, linestyle="--",
            label="ensemble mean")

def plot_observed(ax, gauge_number, times):
    """Fetch and plot observed NOAA tide-gauge data."""
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
    else:
        # Convert to seconds relative to landfall
        t = (date_time - landfall_time) / np.timedelta64(1, 's')
        t /= (24 * 60**2)

        # Detide
        water_level -= tide

        # Plot data
        ax.plot(t, water_level, color='lightgray', marker='x',
                                label="observed")
        # ax.set_title(station_name)
        # ax.legend()


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
