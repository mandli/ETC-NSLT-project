"""Plot functions for comparing gauges across runs"""

from pathlib import Path
import datetime

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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

times = {"DEC2012": [np.datetime64("2012-12-26T00:00"), 
                        datetime.datetime(2012, 12, 25, 0, 0), 
                        datetime.datetime(2012, 12, 30, 0, 0)],
         "NOV2018": [np.datetime64("2018-11-14T08:00:00.00"),
                        datetime.datetime(2018, 11, 14, 0, 0),
                        datetime.datetime(2018, 11, 18, 0, 0)]}

def plot_surface(ax, gauge_id, output_path, style, n_markers=15,
                 dry_tolerance=1e-16):
    """Fetch and plot surface data for gauges used.

    ``style`` is a dict of matplotlib line kwargs (color, linestyle, marker,
    linewidth) encoding the run's ensemble parameters.  Markers are placed
    sparsely (``n_markers`` of them) so the marker *shape* stays legible on a
    dense time series without smothering the line.
    """

    print(f"Loading data from: {output_path}")

    # Load gauge data
    gauge = gauges.GaugeSolution(gauge_id, output_path)
    time = gauge.t / (24 * 60**2)  # Convert to days
    surface = np.ma.masked_where(np.abs(gauge.q[0, :]) < dry_tolerance,
                              gauge.q[3, :])
    markevery = max(1, time.size // n_markers)
    ax.plot(time, surface, markevery=markevery, **style)

    return None

def plot_observed(ax, gauge_number, times):
    """Fetch and plot gauge data for gauges used."""
    station_id, station_name = gauge_mapping[gauge_number]

    # Map GeoClaw gauge number to NOAA gauge number and location/name
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


def plot_key(out_file: Path, key_channels: list[dict]) -> None:
    """Write a standalone legend figure decoding the visual channels.

    Each section in ``key_channels`` (plus an "observed" section added here)
    becomes a small captioned legend, so the data plots can stay clean for
    use in talks/papers.
    """
    sections = [{"title": "Data",
                 "entries": [("observed", {"color": "lightgray",
                                           "marker": "x", "linestyle": "-"})]}]
    sections += key_channels

    n = len(sections)
    fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 2.6))
    if n == 1:
        axes = [axes]
    for ax, section in zip(axes, sections):
        handles = [Line2D([], [], **kwargs) for _label, kwargs in section["entries"]]
        labels = [label for label, _kwargs in section["entries"]]
        ax.axis("off")
        ax.legend(handles, labels, title=section["title"], loc="center",
                  frameon=False)
    fig.tight_layout()
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot(results: list[JobResult], out_path: Path, storm_date: str,
         styles: list[dict], key_channels: list[dict]) -> None:
    """Plot gauge comparisons for a given storm date.

    All ``results`` are overlaid on each gauge's plot, styled by ``styles``
    (one matplotlib-kwargs dict per result, parallel to ``results``).  The
    per-line legend is replaced by a single standalone key figure built from
    ``key_channels``.
    """

    # Paths
    figure_path = out_path / storm_date
    figure_path.mkdir(parents=True, exist_ok=True)

    for gauge_id in range(1, 9):
        fig, ax = plt.subplots()
        plot_observed(ax, gauge_id, times[storm_date])
        for result, style in zip(results, styles):
            if storm_date == result.job.storm_path.stem.split("_")[0]:
                plot_surface(ax, gauge_id, result.paths.job, style)

        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Surface Elevation (m)")

        station_id, station_name = gauge_mapping[gauge_id]
        ax.set_title(f"{station_name} ({station_id}) - {storm_date}")
        fig.savefig(figure_path / f"gauge_{gauge_id}_surface_comparison.png", dpi=300)
        plt.close(fig)

    plot_key(figure_path / "key.png", key_channels)
