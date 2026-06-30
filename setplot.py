import os
import datetime

import numpy as np

import matplotlib
# Markers and line widths
matplotlib.rcParams['lines.linewidth'] = 2.0
matplotlib.rcParams['lines.markersize'] = 6
matplotlib.rcParams['lines.markersize'] = 8

# Font Sizes
matplotlib.rcParams['font.size'] = 16
matplotlib.rcParams['axes.labelsize'] = 16
matplotlib.rcParams['legend.fontsize'] = 12
matplotlib.rcParams['xtick.labelsize'] = 16
matplotlib.rcParams['ytick.labelsize'] = 16

# DPI of output images
article = False
if article:
    matplotlib.rcParams['savefig.dpi'] = 300
    matplotlib.rcParams['axes.titlesize'] = 'x-large'
    figsize_mult = 2
    add_colorbar = True
else:
    matplotlib.rcParams['savefig.dpi'] = 100
    figsize_mult = 1
    add_colorbar = True
import matplotlib.pyplot as plt

import clawpack.visclaw.gaugetools as gaugetools
import clawpack.clawutil.data as clawutil
import clawpack.geoclaw.data as geodata
import clawpack.geoclaw.surge.plot as surgeplot
import clawpack.geoclaw.util as geoutil
import clawpack.amrclaw.data as amrdata
from clawpack.geoclaw.util import fetch_noaa_tide_data
import clawpack.geoclaw.surge.storm as stormtools

def setplot(plotdata=None):
    """"""

    if plotdata is None:
        from clawpack.visclaw.data import ClawPlotData
        plotdata = ClawPlotData()

    # clear any old figures,axes,items data
    plotdata.clearfigures()
    plotdata.format = 'ascii'

    # Load data from output
    clawdata = clawutil.ClawInputData(2)
    clawdata.read(os.path.join(plotdata.outdir, 'claw.data'))
    physics = geodata.GeoClawData()
    physics.read(os.path.join(plotdata.outdir, 'geoclaw.data'))
    surge_data = geodata.SurgeData()
    surge_data.read(os.path.join(plotdata.outdir, 'surge.data'))
    friction_data = geodata.FrictionData()
    friction_data.read(os.path.join(plotdata.outdir, 'friction.data'))
    gauge_data = amrdata.GaugeData()
    gauge_data.read(plotdata.outdir)

    storm = stormtools.Storm(surge_data.storm_file, file_format="data")

    # Per-storm reference time (t=0) and NOAA fetch window, keyed by the
    # storm file's name prefix.  ``landfall_time`` must match the
    # ``time_offset`` set for the storm in setrun.py.
    # Per-storm configuration keyed by storm file name prefix.
    # Fields: (landfall_time, noaa_begin, noaa_end, gauge_xlimits)
    #   landfall_time  - np.datetime64 matching time_offset in setrun.py (t=0)
    #   noaa_begin/end - datetime.datetime window for NOAA tide data fetch
    #   gauge_xlimits  - [t_start, t_end] in days relative to landfall_time
    storm_times = {
        "DEC1992": (np.datetime64("1992-12-08T00:00:00.00"),
                    datetime.datetime(1992, 12, 8, 0, 0),
                    datetime.datetime(1992, 12, 16, 0, 0),
                    [2, 6]),
        "DEC2012": (np.datetime64("2012-12-26T00:00"),
                    datetime.datetime(2012, 12, 25, 0, 0),
                    datetime.datetime(2012, 12, 30, 0, 0),
                    [0, 4]),
        "NOV2018": (np.datetime64("2018-11-14T08:00:00.00"),
                    datetime.datetime(2018, 11, 14, 0, 0),
                    datetime.datetime(2018, 11, 18, 0, 0),
                    [0, 4]),
    }

    def get_storm_times(storm):
        name = storm.file_paths[0].name
        for prefix, times in storm_times.items():
            if name.startswith(prefix):
                return times
        raise ValueError(f"Unknown storm {storm.name}")

    landfall_time, _begin_date, _end_date, gauge_xlimits = get_storm_times(storm)
    time_label = ("Days relative to "
                  + np.datetime_as_string(landfall_time, unit='m').replace("T", " ")
                  + " UTC")

    # Load storm track
    track = surgeplot.track_data(os.path.join(plotdata.outdir, 'fort.track'))

    # Set afteraxes function
    def surge_afteraxes(cd):
        # 11:00 was 10:60 instead!
        surgeplot.surge_afteraxes(cd, track, plot_track=True, style="rX",
                                             kwargs={"markersize": 6})
        surgeplot.days_figure_title(cd, new_time=True)

    # Color limits
    surface_limits = [-1.0 + physics.sea_level, 1.0 + physics.sea_level]
    # surface_limits = [0.0, 1.0]
    speed_limits = [0.0, 2.0]
    wind_limits = (0, 30)
    pressure_limits = (980, 1013)
    friction_bounds = [0.01, 0.04]

    # ==========================================================================
    #   Plot specifications
    # ==========================================================================

    regions = {'Full Domain': {"xlimits": [clawdata.lower[0], clawdata.upper[0]],
                               "ylimits": [clawdata.lower[1], clawdata.upper[1]],
                               "shrink": 1.0,
                               "figsize": [6.4, 4.8]},
               'Tri-State Region': {"xlimits": [-74.5,-71.0],
                                    "ylimits": [40.0,41.5],
                                    "shrink": 0.75,
                                    "figsize": [6.4 * 2, 4.8]},
                'NYC': {"xlimits": [-74.2,-73.7],
                        "ylimits": [40.4,40.85],
                        "shrink": 1.0,
                        "figsize": [6.4, 4.8]}
               }
    for (name, region_dict) in regions.items():
        [size * figsize_mult for size in region_dict['figsize']]


        # Surface Figure
        plotfigure = plotdata.new_plotfigure(name="Surface - %s" % name)
        plotfigure.kwargs = {"figsize": [size * figsize_mult for size in region_dict['figsize']]}
        plotaxes = plotfigure.new_plotaxes()
        plotaxes.title = "Surface"
        plotaxes.xlimits = region_dict["xlimits"]
        plotaxes.ylimits = region_dict["ylimits"]
        plotaxes.scaled = True
        plotaxes.afteraxes = surge_afteraxes

        surgeplot.add_surface_elevation(plotaxes, bounds=surface_limits)
        surgeplot.add_land(plotaxes, bounds=[0.0, 20.0])
        surgeplot.add_bathy_contours(plotaxes)
        plotaxes.plotitem_dict['surface'].amr_patchedges_show = [1] * 10
        plotaxes.plotitem_dict['land'].amr_patchedges_show = [0] * 10
        plotaxes.plotitem_dict['bathy'].amr_contour_show = [0, 0, 0, 0, 0, 1, 1]
        plotaxes.plotitem_dict['bathy'].kwargs = {'linewidths':0.5 * figsize_mult, 'colors': 'black'}

        # Speed Figure
        plotfigure = plotdata.new_plotfigure(name="Currents - %s" % name)
        plotfigure.kwargs = {"figsize": [size * figsize_mult for size in region_dict['figsize']]}
        plotaxes = plotfigure.new_plotaxes()
        plotaxes.title = "Currents"
        plotaxes.xlimits = region_dict["xlimits"]
        plotaxes.ylimits = region_dict["ylimits"]
        plotaxes.scaled = True
        plotaxes.afteraxes = surge_afteraxes

        surgeplot.add_speed(plotaxes, bounds=speed_limits)
        surgeplot.add_land(plotaxes, bounds=[0.0, 20.0])
        surgeplot.add_bathy_contours(plotaxes)
        plotaxes.plotitem_dict['speed'].amr_patchedges_show = [1] * 10
        plotaxes.plotitem_dict['land'].amr_patchedges_show = [1] * 10
        plotaxes.plotitem_dict['bathy'].amr_contour_show = [0, 0, 0, 0, 0, 1, 1]
        plotaxes.plotitem_dict['bathy'].kwargs = {'linewidths':0.5 * figsize_mult, 'colors': 'black'}

    #
    # Friction field
    #
    def friction_after_axes(cd):
        plt.title(r"Manning's $n$ Coefficient")
        for region in friction_data.friction_regions:
            surgeplot.draw_box(plt.gca(), [region[0][0], region[1][0],
                                           region[0][1], region[1][1]],
                               style="k--")



    plotfigure = plotdata.new_plotfigure(name='Friction')
    plotfigure.show = friction_data.variable_friction and False

    plotaxes = plotfigure.new_plotaxes()
    plotaxes.xlimits = regions['Full Domain']['xlimits']
    plotaxes.ylimits = regions['Full Domain']['ylimits']
    plotaxes.title = "Manning's N Coefficient"
    plotaxes.afteraxes = friction_after_axes
    plotaxes.scaled = True

    surgeplot.add_friction(plotaxes, bounds=friction_bounds, shrink=0.9)
    plotaxes.plotitem_dict['friction'].amr_patchedges_show = [0] * 10
    plotaxes.plotitem_dict['friction'].colorbar_label = "$n$"

    #
    #  Hurricane Forcing fields
    #
    def forcing_afteraxes(cd, title=None):
        surge_afteraxes(cd)
        box = np.array([-80, 27.5, -62.5, 45])
        surgeplot.draw_box(plt.gca(), box, style='r-')
        surgeplot.draw_box(plt.gca(), np.array([box[0]-storm.ramp_width,
                                                box[1]-storm.ramp_width,
                                                box[2]+storm.ramp_width,
                                                box[3]+storm.ramp_width]),
                            style='b--')

    # Pressure field
    plotfigure = plotdata.new_plotfigure(name='Pressure')
    plotfigure.show = surge_data.pressure_forcing and True
    plotfigure.kwargs = {"figsize": [6.4 * figsize_mult,
                                     4.8 * figsize_mult]}

    plotaxes = plotfigure.new_plotaxes()
    plotaxes.xlimits = regions['Full Domain']['xlimits']
    plotaxes.ylimits = regions['Full Domain']['ylimits']
    plotaxes.title = "Pressure Field"
    plotaxes.afteraxes = forcing_afteraxes
    plotaxes.scaled = True
    surgeplot.add_pressure(plotaxes, bounds=pressure_limits)
    surgeplot.add_bathy_contours(plotaxes)
    # surgeplot.add_land(plotaxes, bounds=[0.0, 20.0])
    plotaxes.plotitem_dict['pressure'].amr_patchedges_show = [0] * 10
    # plotaxes.plotitem_dict['land'].amr_patchedges_show = [1] * 10

    # Wind field
    plotfigure = plotdata.new_plotfigure(name='Wind Speed')
    plotfigure.show = surge_data.wind_forcing and True
    plotfigure.kwargs = {"figsize": [6.4 * figsize_mult,
                                     4.8 * figsize_mult]}

    plotaxes = plotfigure.new_plotaxes()
    plotaxes.xlimits = regions['Full Domain']['xlimits']
    plotaxes.ylimits = regions['Full Domain']['ylimits']
    plotaxes.title = "Wind Field"
    plotaxes.afteraxes = forcing_afteraxes
    plotaxes.scaled = True
    surgeplot.add_wind(plotaxes, bounds=wind_limits)
    # surgeplot.add_land(plotaxes, bounds=[0.0, 20.0])
    surgeplot.add_bathy_contours(plotaxes)
    plotaxes.plotitem_dict['wind'].amr_patchedges_show = [1] * 10
    # plotaxes.plotitem_dict['land'].amr_patchedges_show = [1] * 10
    plotaxes.plotitem_dict['wind'].add_colorbar = add_colorbar
    # plotaxes.plotitem_dict['land'].add_colorbar = add_colorbar

    # ========================================================================
    #  Figures for gauges
    # ========================================================================
    gauge_mapping = {1: ('8518750', 'The Battery, NY'),
                        2: ('8516945', 'Kings Point, NY'),
                        3: ('8510560', 'Montauk, NY'),
                        4: ('8467150', 'Bridgeport, CT'),
                        5: ('8465705', 'New Haven, CT'),
                        6: ('8452660', 'Newport, RI'),
                        7: ('8531680', 'Sandy Hook, NJ'),
                        8: ('8534720', 'Atlantic City, NJ')}
    def plot_observed(current_data, storm):
        """Fetch and plot gauge data for gauges used."""

        # Map GeoClaw gauge number to NOAA gauge number and location/name
        station_id, station_name = gauge_mapping[current_data.gaugesoln.id]

        landfall_time, begin_date, end_date = get_storm_times(storm)

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
            ax = plt.gca()
            ax.plot(t, water_level, color='lightgray', marker='x',
                                    label="observed")
            ax.set_title(station_name)
            ax.legend()


    plotfigure = plotdata.new_plotfigure(name='Gauge Surfaces', figno=300,
                                         type='each_gauge')
    plotfigure.show = True
    plotfigure.clf_each_gauge = True

    plotaxes = plotfigure.new_plotaxes()
    plotaxes.time_scale = 1 / (24 * 60**2)
    plotaxes.grid = True
    plotaxes.xlimits = gauge_xlimits
    plotaxes.ylimits = [-0.5, 2.0]
    plotaxes.title = "Surface"
    plotaxes.ylabel = "Surface (m)"
    plotaxes.time_label = time_label
    plotaxes.afteraxes = lambda cd: plot_observed(cd, storm)

    plotitem = plotaxes.new_plotitem(plot_type='1d_plot')
    plotitem.plot_var = surgeplot.gauge_surface
    plotitem.kwargs = {"label": "computed"}
    # Plot red area if gauge is dry
    plotitem = plotaxes.new_plotitem(plot_type='1d_plot')
    plotitem.plot_var = surgeplot.gauge_dry_regions
    plotitem.kwargs = {"color":'lightcoral', "linewidth":5, "label": "dry"}

    # Plot wind and pressure at each gauge
    def storm_gauge_afteraxes(cd, storm):
        # Map GeoClaw gauge number to NOAA gauge number and location/name
        station_name = gauge_mapping[cd.gaugesoln.id][1]

        # Convert to seconds relative to landfall
        # t = (cd.gaugesoln.t - landfall_time) / np.timedelta64(1, 's')
        t = cd.gaugesoln.t
        t /= (24 * 60**2)

        w_ax = plt.gca()
        color ="tab:blue"
        w_ax.plot(t, surgeplot.gauge_wind(cd), label="wind", color=color)
        w_ax.set_ylabel("Wind (m/s)", color=color)
        w_ax.set_ylim(wind_limits)
        w_ax.tick_params(axis='y', labelcolor=color)

        P_ax = w_ax.twinx()
        color = "tab:red"
        pressure = surgeplot.gauge_pressure(cd) * 1e-2
        P_ax.set_ylabel("Pressure (mbar)", color=color)
        P_ax.plot(t, pressure, label="pressure", color=color)
        # Scale the pressure axis to the data (with a small margin) rather than
        # a fixed window: ETC storms span a wide pressure range and can run well
        # above the tropical-storm default, which would otherwise clip the curve.
        p_min, p_max = np.min(pressure), np.max(pressure)
        p_pad = max(1.0, 0.05 * (p_max - p_min))
        P_ax.set_ylim(p_min - p_pad, p_max + p_pad)
        P_ax.tick_params(axis='y', labelcolor=color)

        w_ax.set_title(f"Storm Fields at {station_name}")

        plt.gcf().tight_layout()

    plotfigure = plotdata.new_plotfigure(name='Gauge Wind', figno=301,
                                         type='each_gauge')
    plotfigure.show = True
    plotfigure.clf_each_gauge = True
    plotfigure.kwargs = {"figsize": [6.4 * 1.2,
                                     4.8 * 1.0]}

    plotaxes = plotfigure.new_plotaxes()
    plotaxes.time_scale = 1 / (24 * 60**2)
    plotaxes.grid = True
    # plotaxes.xlimits = [0, 3]
    plotaxes.time_label = time_label
    plotaxes.afteraxes = lambda cd: storm_gauge_afteraxes(cd, storm)

    #
    #  Gauge Location Plots
    #
    def gauge_location_afteraxes(cd):
        # plt.subplots_adjust(left=0.12, bottom=0.06, right=0.97, top=0.97)
        surge_afteraxes(cd)
        gaugetools.plot_gauge_locations(cd.plotdata, gaugenos='all',
                                        format_string='ko', add_labels=True)

    gauge_locations = {0: [[-75, -70], [38, 42]]}
    # dx = 0.125
    # for gauge in gauge_data.gauges:
    #     x, y = gauge[1:3]
    #     gauge_locations[gauge[0]] = [[x-dx, x+dx], [y-dx, y+dx]]

    for (gauge_id, gauge_loc) in gauge_locations.items():
        if gauge_id == 0:
            fig_name = "All Gauge Locations"
        else:
            fig_name = f"Gauge Location {gauge_id}"
        
        plotfigure = plotdata.new_plotfigure(name=fig_name)
        plotfigure.show = True

        # Set up for axes in this figure:
        plotaxes = plotfigure.new_plotaxes()
        plotaxes.title = fig_name
        plotaxes.scaled = True
        plotaxes.xlimits = gauge_loc[0]
        plotaxes.ylimits = gauge_loc[1]
        plotaxes.afteraxes = gauge_location_afteraxes
        surgeplot.add_surface_elevation(plotaxes, bounds=surface_limits)
        surgeplot.add_land(plotaxes, bounds=[0.0, 20.0])
        plotaxes.plotitem_dict['surface'].amr_patchedges_show = [1] * 10
        # plotaxes.plotitem_dict['surface'].pcolor_cmap = surface_cmap
        plotaxes.plotitem_dict['land'].amr_patchedges_show = [1] * 10

    # -----------------------------------------
    # Parameters used only when creating html and/or latex hardcopy
    # e.g., via pyclaw.plotters.frametools.printframes:

    plotdata.printfigs = True                # print figures
    plotdata.print_format = 'png'            # file format
    plotdata.print_framenos = 'all'          # list of frames to print
    plotdata.print_gaugenos = 'all'          # list of gauges to print
    plotdata.print_fignos = 'all'            # list of figures to print
    plotdata.html = True                     # create html files of plots?
    plotdata.latex = False                   # create latex file of plots?
    plotdata.latex_figsperline = 2           # layout of plots
    plotdata.latex_framesperline = 1         # layout of plots
    plotdata.latex_makepdf = False           # also run pdflatex?
    plotdata.parallel = True                 # parallel plotting

    return plotdata
