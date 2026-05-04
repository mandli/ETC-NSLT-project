"""Plot functions for comparing gauges across runs"""

from pathlib import Path
import datetime

import matplotlib.pyplot as plt
import numpy as np

from batch import JobResult

import clawpack.pyclaw.gauges
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

def plot_surface(ax, gauge_id, output_path, res, dry_tolerance=1e-16):
    """Fetch and plot surface data for gauges used."""

    print(f"Loading data from: {output_path}")

    # Load gauge data
    gauge = clawpack.pyclaw.gauges.GaugeSolution(gauge_id, output_path)
    time = gauge.t / (24 * 60**2)  # Convert to days
    surface = np.ma.masked_where(np.abs(gauge.q[0, :]) < dry_tolerance,
                              gauge.q[3, :])
    ax.plot(time, surface, label=f"{res}")
    dry = np.ma.masked_where(np.abs(gauge.q[0, :]) > dry_tolerance,
                              np.zeros(gauge.q[0, :].shape))
    # ax.plot(time, dry, color='lightcoral', linewidth=5, label="dry")

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
                                                                end_date)
    
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


def plot(results: list[JobResult], out_path: Path, storm_date: str) -> None:
    """Plot gauge comparisons for a given storm date."""

    # Paths
    figure_path = out_path / storm_date
    figure_path.mkdir(parents=True, exist_ok=True)
    
    plt.tight_layout()
    for gauge_id in range(1, 9):
        fig, ax = plt.subplots()
        plot_observed(ax, gauge_id, times[storm_date])
        for result in results:
            if storm_date == result.job.storm_path.stem.split("_")[0]:
                plot_surface(ax, gauge_id, result.paths.job, 
                                 result.job.storm_path.stem.split("_")[1])

        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Surface Elevation (m)")

        station_id, station_name = gauge_mapping[gauge_id]
        ax.set_title(f"{station_name} ({station_id}) - {storm_date}")
        ax.legend()
        fig.savefig(figure_path / f"gauge_{gauge_id}_surface_comparison.png", dpi=300)
