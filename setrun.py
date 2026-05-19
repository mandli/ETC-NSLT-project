# encoding: utf-8
"""
Module to set up run time parameters for Clawpack.

The values set in the function setrun are then written out to data files
that will be read in by the Fortran code.

"""

from pathlib import Path
import os
import datetime
import shutil
import gzip

import numpy as np

from clawpack.geoclaw.surge.storm import Storm
from clawpack.geoclaw.netcdf_utils import MetInterrogator
import clawpack.clawutil as clawutil
import clawpack.geoclaw.util as util

# Time Conversions
def days2seconds(days):
    return days * 60.0**2 * 24.0


# Scratch directory for storing topo and storm files:
CLAW = Path(os.environ["CLAW"])
scratch_dir = CLAW / 'geoclaw' / 'scratch'

# ------------------------------
def setrun(claw_pkg='geoclaw'):

    """
    Define the parameters used for running Clawpack.

    INPUT:
        claw_pkg expected to be "geoclaw" for this setrun.

    OUTPUT:
        rundata - object of class ClawRunData

    """

    from clawpack.clawutil import data

    assert claw_pkg.lower() == 'geoclaw',  "Expected claw_pkg = 'geoclaw'"

    num_dim = 2
    rundata = data.ClawRunData(claw_pkg, num_dim)

    # ------------------------------------------------------------------
    # Standard Clawpack parameters to be written to claw.data:
    #   (or to amr2ez.data for AMR)
    # ------------------------------------------------------------------
    clawdata = rundata.clawdata  # initialized when rundata instantiated

    # Set single grid parameters first.
    # See below for AMR parameters.

    # ---------------
    # Spatial domain:
    # ---------------

    # Number of space dimensions:
    clawdata.num_dim = num_dim

    # Lower and upper edge of computational domain:
    clawdata.lower[0] = -85.0      # west longitude
    clawdata.upper[0] = -60.0      # east longitude

    clawdata.lower[1] = 20.0       # south latitude
    clawdata.upper[1] = 50.0       # north latitude

    # Number of grid cells:
    degree_factor = 4  # (0.25º,0.25º) ~ (25237.5 m, 27693.2 m) resolution
    clawdata.num_cells[0] = int(clawdata.upper[0] - clawdata.lower[0]) \
        * degree_factor
    clawdata.num_cells[1] = int(clawdata.upper[1] - clawdata.lower[1]) \
        * degree_factor

    # ---------------
    # Size of system:
    # ---------------

    # Number of equations in the system:
    clawdata.num_eqn = 3

    # Number of auxiliary variables in the aux array (initialized in setaux)
    # First three are from shallow GeoClaw, fourth is friction and last 3 are
    # storm fields
    clawdata.num_aux = 3 + 1 + 3

    # Index of aux array corresponding to capacity function, if there is one:
    clawdata.capa_index = 2

    # -------------
    # Initial time:
    # -------------
    clawdata.t0 =  days2seconds(0.0)
    # clawdata.tfinal = days2seconds(4.0)
    clawdata.tfinal = days2seconds(3.0)
    # November storm 
    #   start time "2018-11-14T08"
    #   end time "2018-11-17T19" 16H + 2D + 19
    clawdata.tfinal = 16*60**2 + days2seconds(2.0) + 19*60**2

    # Restart from checkpoint file of a previous run?
    # If restarting, t0 above should be from original run, and the
    # restart_file 'fort.chkNNNNN' specified below should be in
    # the OUTDIR indicated in Makefile.

    clawdata.restart = False               # True to restart from prior results
    clawdata.restart_file = 'fort.chk00006'  # File to use for restart data

    # -------------
    # Output times:
    # --------------

    # Specify at what times the results should be written to fort.q files.
    # Note that the time integration stops after the final output time.
    # The solution at initial time t0 is always written in addition.

    clawdata.output_style = 1

    if clawdata.output_style == 1:
        # Output nout frames at equally spaced times up to tfinal:
        recurrence = 24
        clawdata.num_output_times = int((clawdata.tfinal - clawdata.t0) *
                                        recurrence / (60**2 * 24))

        clawdata.output_t0 = True  # output at initial (or restart) time?

    elif clawdata.output_style == 2:
        # Specify a list of output times.
        clawdata.output_times = [0.5, 1.0]

    elif clawdata.output_style == 3:
        # Output every iout timesteps with a total of ntot time steps:
        clawdata.output_step_interval = 1
        clawdata.total_steps = 10
        clawdata.output_t0 = True

    clawdata.output_format = 'binary'      # 'ascii' or 'binary'
    clawdata.output_q_components = 'all'   # could be list such as [True,True]
    clawdata.output_aux_components = 'all'
    clawdata.output_aux_onlyonce = False    # output aux arrays only at t0

    # ---------------------------------------------------
    # Verbosity of messages to screen during integration:
    # ---------------------------------------------------

    # The current t, dt, and cfl will be printed every time step
    # at AMR levels <= verbosity.  Set verbosity = 0 for no printing.
    #   (E.g. verbosity == 2 means print only on levels 1 and 2.)
    clawdata.verbosity = 1

    # --------------
    # Time stepping:
    # --------------

    # if dt_variable==1: variable time steps used based on cfl_desired,
    # if dt_variable==0: fixed time steps dt = dt_initial will always be used.
    clawdata.dt_variable = True

    # Initial time step for variable dt.
    # If dt_variable==0 then dt=dt_initial for all steps:
    clawdata.dt_initial = 0.016

    # Max time step to be allowed if variable dt used:
    clawdata.dt_max = 1e+99

    # Desired Courant number if variable dt used, and max to allow without
    # retaking step with a smaller dt:
    clawdata.cfl_desired = 0.75
    clawdata.cfl_max = 1.0

    # Maximum number of time steps to allow between output times:
    clawdata.steps_max = 2**16

    # ------------------
    # Method to be used:
    # ------------------

    # Order of accuracy:  1 => Godunov,  2 => Lax-Wendroff plus limiters
    clawdata.order = 2

    # Use dimensional splitting? (not yet available for AMR)
    clawdata.dimensional_split = 'unsplit'

    # For unsplit method, transverse_waves can be
    #  0 or 'none'      ==> donor cell (only normal solver used)
    #  1 or 'increment' ==> corner transport of waves
    #  2 or 'all'       ==> corner transport of 2nd order corrections too
    clawdata.transverse_waves = 2

    # Number of waves in the Riemann solution:
    clawdata.num_waves = 3

    # List of limiters to use for each wave family:
    # Required:  len(limiter) == num_waves
    # Some options:
    #   0 or 'none'     ==> no limiter (Lax-Wendroff)
    #   1 or 'minmod'   ==> minmod
    #   2 or 'superbee' ==> superbee
    #   3 or 'mc'       ==> MC limiter
    #   4 or 'vanleer'  ==> van Leer
    clawdata.limiter = ['mc', 'mc', 'mc']

    clawdata.use_fwaves = True    # True ==> use f-wave version of algorithms

    # Source terms splitting:
    #   src_split == 0 or 'none'
    #      ==> no source term (src routine never called)
    #   src_split == 1 or 'godunov'
    #      ==> Godunov (1st order) splitting used,
    #   src_split == 2 or 'strang'
    #      ==> Strang (2nd order) splitting used,  not recommended.
    clawdata.source_split = 'godunov'

    # --------------------
    # Boundary conditions:
    # --------------------

    # Number of ghost cells (usually 2)
    clawdata.num_ghost = 2

    # Choice of BCs at xlower and xupper:
    #   0 => user specified (must modify bcN.f to use this option)
    #   1 => extrapolation (non-reflecting outflow)
    #   2 => periodic (must specify this at both boundaries)
    #   3 => solid wall for systems where q(2) is normal velocity

    clawdata.bc_lower[0] = 'extrap'
    clawdata.bc_upper[0] = 'extrap'

    clawdata.bc_lower[1] = 'extrap'
    clawdata.bc_upper[1] = 'extrap'

    # Specify when checkpoint files should be created that can be
    # used to restart a computation.

    clawdata.checkpt_style = 0

    if clawdata.checkpt_style == 0:
        # Do not checkpoint at all
        pass

    elif np.abs(clawdata.checkpt_style) == 1:
        # Checkpoint only at tfinal.
        pass

    elif np.abs(clawdata.checkpt_style) == 2:
        # Specify a list of checkpoint times.
        clawdata.checkpt_times = [0.1, 0.15]

    elif np.abs(clawdata.checkpt_style) == 3:
        # Checkpoint every checkpt_interval timesteps (on Level 1)
        # and at the final time.
        clawdata.checkpt_interval = 5

    # ---------------
    # AMR parameters:
    # ---------------
    amrdata = rundata.amrdata

    # max number of refinement levels:
    amrdata.amr_levels_max = 2

    amrdata.refinement_ratios_x = [2, 2, 2, 3, 3, 3, 4, 4]
    amrdata.refinement_ratios_y = [2, 2, 2, 3, 3, 3, 4, 4]
    amrdata.refinement_ratios_t = [2, 2, 2, 3, 3, 6, 4, 4]


    # Specify type of each aux variable in amrdata.auxtype.
    # This must be a list of length maux, each element of which is one of:
    #   'center',  'capacity', 'xleft', or 'yleft'  (see documentation).

    amrdata.aux_type = ['center', 'capacity', 'yleft', 'center', 'center',
                        'center', 'center']

    # Flag using refinement routine flag2refine rather than richardson error
    amrdata.flag_richardson = False    # use Richardson?
    amrdata.flag2refine = True

    # steps to take on each level L between regriddings of level L+1:
    amrdata.regrid_interval = 3 #3

    # width of buffer zone around flagged points:
    # (typically the same as regrid_interval so waves don't escape):
    amrdata.regrid_buffer_width = 2

    # clustering alg. cutoff for (# flagged pts) / (total # of cells refined)
    # (closer to 1.0 => more small grids may be needed to cover flagged cells)
    amrdata.clustering_cutoff = 0.700000

    # print info about each regridding up to this level:
    amrdata.verbosity_regrid = 0

    #  ----- For developers -----
    # Toggle debugging print statements:
    amrdata.dprint = False      # print domain flags
    amrdata.eprint = False      # print err est flags
    amrdata.edebug = False      # even more err est flags
    amrdata.gprint = False      # grid bisection/clustering
    amrdata.nprint = False      # proper nesting output
    amrdata.pprint = False      # proj. of tagged points
    amrdata.rprint = False      # print regridding summary
    amrdata.sprint = False      # space/memory output
    amrdata.tprint = False      # time step reporting each level
    amrdata.uprint = False      # update/upbnd reporting

    # More AMR parameters can be set -- see the defaults in pyclaw/data.py

    # Battery gauge - Station ID: 8518750
    rundata.gaugedata.gauges.append([1,-74.013,40.7,clawdata.t0,clawdata.tfinal])
    # Kings point gauge - Station ID: 8516945
    rundata.gaugedata.gauges.append([2,-73.77,40.81,clawdata.t0,clawdata.tfinal])
    # Montauk, NY - Station ID: 8510560
    rundata.gaugedata.gauges.append([3,-71.96,41.04833,clawdata.t0,clawdata.tfinal])
    # Bridgeport, CT - Station ID: 8467150
    rundata.gaugedata.gauges.append([4,-73.1816666667,41.1733333333,clawdata.t0,clawdata.tfinal])
    # New Haven, CT - Station ID: 8465705
    rundata.gaugedata.gauges.append([5,-72.915152,41.2235,clawdata.t0,clawdata.tfinal])
    # Newport, RI - Station ID: 8452660 (71° 19.6 W, 41° 30.3 N)
    rundata.gaugedata.gauges.append([6, -71.326667, 41.500833,clawdata.t0,clawdata.tfinal])
    # Sandy Hook, NJ - Station ID: 8531680 (74° 0.6 W, 40° 28.0 N)
    # moified to (74.01 W, 40.46 N)
    rundata.gaugedata.gauges.append([7, -74.01, 40.46,clawdata.t0,clawdata.tfinal])
    # Atlantic City, NJ - Station ID: 8534720 (74° 25.1 W, 39° 21.4 N)
    rundata.gaugedata.gauges.append([8, -74.416944, 39.351111,clawdata.t0,clawdata.tfinal])

    # Force the gauges to also record the wind and pressure fields
    rundata.gaugedata.aux_out_fields = [4, 5, 6]

    # == setregions.data values ==
    regions = rundata.regiondata.regions
    # to specify regions of refinement append lines of the form
    #  [minlevel,maxlevel,t1,t2,x1,x2,y1,y2]
    # Entire domain
    regions.append([1, 4, clawdata.t0, clawdata.tfinal,
                          clawdata.lower[0], clawdata.upper[0],
                          clawdata.lower[1], clawdata.upper[1]])
    # NYC Region
    regions.append([1, 7, clawdata.t0, clawdata.tfinal,
                          -74.5, -73.5, 40.3, 41.0])

    # Gauges
    dx = 0.25
    for gauge in rundata.gaugedata.gauges:
        regions.append([5, 7, clawdata.t0, clawdata.tfinal,
                              gauge[1] - dx, gauge[1] + dx,
                              gauge[2] - dx, gauge[2] + dx])

    dx = 0.1
    for gauge in rundata.gaugedata.gauges:
        regions.append([6, 7, clawdata.t0, clawdata.tfinal,
                              gauge[1] - dx, gauge[1] + dx,
                              gauge[2] - dx, gauge[2] + dx])

    # regions.append([5, 6, days2seconds(1), clawdata.tfinal, -74.25,-73.5,40.5,41]) # refine gauge 1,2,3
    # regions.append([5, 6, days2seconds(1), clawdata.tfinal, -73.25,-72.75,41,41.5]) # refine gauge 4,5
    # regions.append([6, 7, days2seconds(1), clawdata.tfinal, -72.25,-72,41,41.5]) # refine gauge 6

    # ------------------------------------------------------------------
    # GeoClaw specific parameters:
    # ------------------------------------------------------------------
    rundata = setgeo(rundata)

    return rundata
    # end of function setrun
    # ----------------------


# -------------------
def setgeo(rundata):
    """
    Set GeoClaw specific runtime parameters.
    For documentation see ....
    """

    geo_data = rundata.geo_data

    # == Physics ==
    geo_data.gravity = 9.81
    geo_data.coordinate_system = 2
    geo_data.earth_radius = 6367.5e3
    geo_data.rho = 1025.0
    geo_data.rho_air = 1.15
    geo_data.ambient_pressure = 101.3e3

    # == Forcing Options
    geo_data.coriolis_forcing = True
    geo_data.friction_forcing = True
    geo_data.friction_depth = 1e10

    # == Algorithm and Initial Conditions ==
    geo_data.sea_level = 0.0
    geo_data.dry_tolerance = 1.e-2

    # Refinement Criteria
    refine_data = rundata.refinement_data
    refine_data.wave_tolerance = 1.0
    refine_data.speed_tolerance = [1.0, 2.0, 3.0, 4.0]
    refine_data.variable_dt_refinement_ratios = True

    # == settopo.data values ==
    topo_data = rundata.topo_data
    topo_data.topofiles = []
    # for topography, append lines of the form
    #   [topotype, fname]
    # See regions for control over these regions, need better bathy data for
    # the smaller domains
    topo_dir = Path(os.environ["DATA_PATH"]) / "topography"
    # topo_data.topofiles.append([3, os.path.join(topo_dir, 'atlantic_1min.tt3')])
    # topo_data.topofiles.append([3, os.path.join(topo_dir, 'newyork_3s.tt3')])
    topo_data.topofiles.append([4, topo_dir / "GEBCO" / "GEBCO_2023.nc"])
    # ncei_base_path = os.path.join(topo_dir, "ny_area", "ncei19_ny")
    ncei_base_path = topo_dir / "coastal_atlantic_9th_second"

    ncei_file_list = ["ncei19_n39x00_w075x00_2018v2.nc",
                      "ncei19_n39x00_w075x25_2014v1.nc",
                      "ncei19_n39x00_w075x50_2014v1.nc",
                      "ncei19_n39x25_w074x75_2018v2.nc",
                      "ncei19_n39x25_w075x00_2018v2.nc",
                      "ncei19_n39x25_w075x25_2018v2.nc",
                      "ncei19_n39x25_w075x50_2014v1.nc",
                      "ncei19_n39x50_w074x50_2018v2.nc",
                      "ncei19_n39x50_w074x75_2018v2.nc",
                      "ncei19_n39x50_w075x25_2018v2.nc",
                      "ncei19_n39x50_w075x50_2018v2.nc",
                      "ncei19_n39x50_w075x75_2014v1.nc",
                      "ncei19_n39x75_w074x25_2018v2.nc",
                      "ncei19_n39x75_w074x50_2018v2.nc",
                      "ncei19_n39x75_w075x50_2014v1.nc",
                      "ncei19_n39x75_w075x75_2014v1.nc",
                      "ncei19_n40x00_w074x25_2018v2.nc",
                      "ncei19_n40x00_w075x25_2014v1.nc",
                      "ncei19_n40x00_w075x50_2014v1.nc",
                      "ncei19_n40x25_w074x00_2018v2.nc",
                      "ncei19_n40x25_w074x25_2018v2.nc",
                      "ncei19_n40x25_w074x75_2014v1.nc",
                      "ncei19_n40x25_w075x00_2014v1.nc",
                      "ncei19_n40x25_w075x25_2014v1.nc",
                      "ncei19_n40x50_w074x00_2018v2.nc",
                      "ncei19_n40x50_w074x25_2018v2.nc",
                      "ncei19_n40x75_w073x00_2015v1.nc",
                      "ncei19_n40x75_w073x25_2015v1.nc",
                      "ncei19_n40x75_w073x50_2015v1.nc",
                      "ncei19_n40x75_w073x75_2015v1.nc",
                      "ncei19_n40x75_w074x00_2015v1.nc",
                      "ncei19_n40x75_w074x25_2015v1.nc",
                      "ncei19_n41x00_w072x25_2015v1.nc",
                      "ncei19_n41x00_w072x50_2015v1.nc",
                      "ncei19_n41x00_w072x75_2015v1.nc",
                      "ncei19_n41x00_w073x00_2015v1.nc",
                      "ncei19_n41x00_w073x25_2015v1.nc",
                      "ncei19_n41x00_w073x50_2015v1.nc",
                      "ncei19_n41x00_w073x75_2015v1.nc",
                      "ncei19_n41x00_w074x00_2015v1.nc",
                      "ncei19_n41x00_w074x25_2015v1.nc",
                      "ncei19_n41x25_w072x00_2015v1.nc",
                      "ncei19_n41x25_w072x25_2015v1.nc",
                      "ncei19_n41x25_w072x50_2015v1.nc",
                      "ncei19_n41x25_w072x75_2015v1.nc",
                      "ncei19_n41x25_w073x00_2016v1.nc",
                      "ncei19_n41x25_w073x25_2016v1.nc",
                      "ncei19_n41x25_w073x50_2015v1.nc",
                      "ncei19_n41x25_w073x75_2015v1.nc",
                      "ncei19_n41x25_w074x00_2015v1.nc",
                      "ncei19_n41x50_w072x00_2016v1.nc",
                      "ncei19_n41x50_w072x25_2016v1.nc",
                      "ncei19_n41x50_w072x50_2016v1.nc",
                      "ncei19_n41x50_w072x75_2016v1.nc",
                      "ncei19_n41x50_w073x00_2016v1.nc",
                      "ncei19_n41x50_w074x00_2015v1.nc",
                      "ncei19_n41x50_w074x25_2015v1.nc",
                    ]
    # Hudson: 40, 39, 49, 55
    # CT: 48, 47, 46, 54, 53,52
    # RI: 52, 51, 50
    # LI Sound: 49, 48, 47, 46, 45, 44, 43, 42, 41
    # LI 39, 38, 37, 36, 35, 34, 33, 32, 30, 29, 28, 27, 26
    # NY: 40, 39, 31, 30
    # NJ coast: 25, 24, 20, 19, 16, 13, 12, 7, 8, 3, 4, 0
    # Chesapeake: 21, 22, 23, 18, 17, 15, 14, 11, 10, 9, 6, 5, 4, 2, 1, 0
    # Sandy: range(24, 55)
    # for file_index in range(24, 55):
    #     topo_data.topofiles.append([4, 
    #                             ncei_base_path / ncei_file_list[file_index]])
    # # Add NJ coast
    # for file_index in [0, 3, 4, 8, 7, 12, 13, 16, 19, 20]:#, 24, 25]:
    #     topo_data.topofiles.append([4, 
    #                             ncei_base_path / ncei_file_list[file_index]])

    # ================a
    #  Set Surge Data
    # ================
    data = rundata.surge_data

    # Source term controls
    data.wind_forcing = True
    data.drag_law = 1
    data.pressure_forcing = True

    data.display_landfall_time = True

    # AMR parameters, m/s and m respectively
    data.wind_refine = [10.0, 20.0, 30.0]
    # data.R_refine = [60.0e3, 40e3, 20e3] # Not used

    # Storm parameters - NetCDF file
    data.storm_specification_type = 'data'
    data.storm_file = (Path() / "etc_storm.storm").resolve()

    etc_storm = Storm()
    # storm_path = (Path(os.environ['DATA_PATH']) / "storms" / "ETC_NASA_SLCT"
    #                           / "NOV2018_0pt25.nc").resolve()
    # etc_storm.time_offset = np.datetime64("2018-11-14T08:00:00.00")

    storm_path = (Path(os.environ['DATA_PATH']) / "storms" / "ETC_NASA_SLCT"
                              / "DEC2012_0pt25.nc").resolve()
    etc_storm.time_offset = np.datetime64("2012-12-26")
    etc_storm.file_paths.append(storm_path)
    
    
    etc_storm.file_format = 'netcdf'
    etc_storm.scaling = [1.0, 1.0]
    etc_storm.window_type = 'custom'
    etc_storm.ramp_width = 2
    etc_storm.window = [-80, 27.5, -62.5, 45]
    etc_storm.write(data.storm_file, file_format='data',
                                     var_mapping={'pressure': 'msl'})
    # etc_storm.write(data.storm_file, file_format='data',
    #                                  dim_mapping={"t": "valid_time"},
    #                                  var_mapping={"pressure": "msl"},
    #                                  crop=etc_storm.window,
    #                                  verbose=True
    #                                  )

    # =======================
    #  Set Variable Friction
    # =======================
    data = rundata.friction_data

    # Variable friction
    data.variable_friction = True

    # Region based friction
    # Entire domain
    data.friction_regions.append([rundata.clawdata.lower,
                                  rundata.clawdata.upper,
                                  [np.inf, 0.0, -np.inf],
                                  [0.050, 0.025]])

    # Bahamas (79.5 W, 22 N) x (73.5 W, 27.5 N)
    data.friction_regions.append([[-79.5, -73.5], [22, 27.5],
                                  [np.inf, 0.0, -50, -np.inf],
                                  [0.050, 0.050, 0.025]])

    return rundata
    # end of function setgeo
    # ----------------------


if __name__ == '__main__':
    # Set up run-time parameters and write all data files.
    import sys
    if len(sys.argv) == 2:
        rundata = setrun(sys.argv[1])
    else:
        rundata = setrun()

    rundata.write()
