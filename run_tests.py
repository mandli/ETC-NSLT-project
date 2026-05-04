#!/usr/bin/env python

"""Run storm surge ensemble jobs for NASA SLCT ETC storms."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import datetime
import numpy as np

from batch import  Job, ParallelExecutor, BatchController, ClobberPolicy
from batch.plot import plot_job

import clawpack.pyclaw.gauges
import clawpack.geoclaw.util as geoutil
import clawpack.clawutil.util as clawutil
from clawpack.geoclaw.surge.storm import Storm

gauge_mapping = {1: ('8518750', 'The Battery, NY'),
                 2: ('8516945', 'Kings Point, NY'),
                 3: ('8510560', 'Montauk, NY'),
                 4: ('8467150', 'Bridgeport, CT'),
                 5: ('8465705', 'New Haven, CT'),
                 6: ('8452660', 'Newport, RI'),
                 7: ('8531680', 'Sandy Hook, NJ'),
                 8: ('8534720', 'Atlantic City, NJ')}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)

class ETCJob(Job):

    def __init__(
            self, 
            storm_path: Path,
            sea_level: float = 0.0,
            scaling: float = 1.0,
            levels: int = 2) -> None:
            
        super().__init__()

        self.prefix = (f"storm{storm_path.stem}_" + 
                       f"sea{sea_level:.1f}_" + 
                       f"scale{scaling:.2f}_" + 
                       f"lev{levels}")
        self.executable = "xgeoclaw"

        # Job parameters
        self.storm_path = storm_path
        self.scaling = scaling
        self.sea_level = sea_level
        self.levels = levels

        setrun_path = Path(__file__).parent / "setrun.py"

        setrun = clawutil.fullpath_import(setrun_path)
        self.rundata = setrun.setrun()

        self.rundata.amrdata.amr_levels_max = self.levels
        self.rundata.geo_data.sea_level = self.sea_level

        self.rundata.clawdata.tfinal = 0.05*24*3600 # 3 days

    def write_data_objects(self, path: Path) -> None:
        etc_storm = Storm()
        etc_storm.file_paths.append(self.storm_path)
        if self.storm_path.name.startswith("DEC2012"):
            etc_storm.time_offset = np.datetime64("2012-12-26T00:00:00.00")
        elif self.storm_path.name.startswith("NOV2018"):
            etc_storm.time_offset = np.datetime64("2018-11-14T08:00:00.00")
        etc_storm.file_format = 'netcdf'
        etc_storm.scaling = [self.scaling, 1.0] # Only scale wind?
        etc_storm.window_type = 'custom'
        etc_storm.ramp_width = 2
        etc_storm.window = [-80, 27.5, -62.5, 45]

        self.rundata.surge_data.storm_file = path / f"{self.prefix}.storm"
        etc_storm.write(self.rundata.surge_data.storm_file,
                        file_format='data',
                        dim_mapping={"t": "valid_time"},
                        var_mapping={"pressure": "msl"},
                        verbose=True)
        
        return super().write_data_objects(path)

    def post_run(self, result) -> None:
        plot_job(result, setplot=Path(__file__).parent / "setplot.py")

    def __repr__(self) -> str:
        return (f"ETCJob({self.storm_path}, " +
                        f"sea_level={self.rundata.geo_data.sea_level}, " 
                        + f"scaling={self.scaling}, " 
                        + f"levels={self.rundata.amrdata.amr_levels_max }" +
                        ")")
    
    def __str__(self) -> str:
        return f"{self.prefix}"

# def plot_gauge_comparison(results, experiment):
#     """Plot gauge comparisons for all jobs"""
    
#     # Matplotlib settings
#     import matplotlib.pyplot as plt
#     plt.tight_layout()

#     plot_ensemble = clawutil.fullpath_import(
#                         Path(__file__).parent / "plot_gauge_comparison.py")

#     experiment_path = (Path(os.environ['OUTPUT_PATH']) / experiment).resolve()
#     figure_path = experiment_path / "gauge_comparisons" / ""
#     figure_path.mkdir(parents=True, exist_ok=True)

#     # Plotting parameters
#     times = [[np.datetime64("2012-12-26T00:00"), 
#                 datetime.datetime(2012, 12, 25, 0, 0), 
#                 datetime.datetime(2012, 12, 30, 0, 0)],
#             [np.datetime64("2018-11-14T08:00:00.00"),
#                 datetime.datetime(2018, 11, 14, 0, 0),
#                 datetime.datetime(2018, 11, 18, 0, 0)]]
#     gauges = range(1, 9)
    
#     # Create figures and axes
#     figs = []
#     axes = []
#     for i in range(len(times)):
#         fig, axs = plt.subplots(3, 3)
#         figs.append(fig)
#         axes.append(axs)
    
#     # Just plot the first result for now
#     for result in results:
#         if not result.success:
#             print(f"Job {result.job} did not complete successfully; skipping gauge comparison.")
#             continue

#         # Extract storm date from job prefix
#         storm_date = result.job.storm_path.stem.split("_")[0]
#         storm_resolution = result.job.storm_path.stem.split("_")[1]

#         # Set landfall time based on storm date
#         if storm_date == "DEC2012":
#             landfall_time = np.datetime64("2012-12-26T00:00:00.00")
#             plot_index = 0
#         elif storm_date == "NOV2018":
#             landfall_time = np.datetime64("2018-11-14T08:00:00.00")
#             plot_index = 1
#         else:
#             print(f"Unknown storm date {storm_date}; cannot determine landfall time.")
#             return
    
#         for gauge_number in gauges:
#             breakpoint()
#             plot_ensemble.plot_observed(axes[plot_index][gauge_number], gauge_number, times[plot_index])
#             plot_ensemble.plot_surface(axes[plot_index][gauge_number-1], gauge_number, storm_date, storm_resolution, result.paths.job)
#             axes[plot_index][gauge_number-1].set_xlabel("Time (days)")
#             axes[plot_index][gauge_number-1].set_ylabel("Surface Elevation (m)")
#             gauge_mapping = {1: ('8518750', 'The Battery, NY'),
#                     2: ('8516945', 'Kings Point, NY'),
#                     3: ('8510560', 'Montauk, NY'),
#                     4: ('8467150', 'Bridgeport, CT'),
#                     5: ('8465705', 'New Haven, CT'),
#                     6: ('8452660', 'Newport, RI'),
#                     7: ('8531680', 'Sandy Hook, NJ'),
#                     8: ('8534720', 'Atlantic City, NJ')}

#             station_id, station_name = gauge_mapping[gauge_number]
#             axes[plot_index][gauge_number-1].set_title(f"{station_name} ({station_id}) - {storm_date}")
#             axes[plot_index][gauge_number-1].legend()
#         figs[plot_index].savefig(figure_path / f"gauge_{storm_date}_surface_comparison.png", dpi=300)

#     # result.job.sea_level
#     # result.job.scaling
#     # result.job.levels
#     # result.job.storm_path
#     # result.paths.job
#     # result.paths.log
#     # result.paths.out
    



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Write .data files only; do not run the solver.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip jobs whose output directory already exists.",
    )
    parser.add_argument(
        "--storms-path",
        type=Path,
        default=Path(os.environ['DATA_PATH']) / "storms" / "ETC_NASA_SLCT",
        help="Directory containing netCDF storm files.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=int(os.environ.get("BATCH_MAX_JOBS", 4)),
        help="Maximum concurrent jobs (default: $BATCH_MAX_JOBS or 4).",
    )
    parser.add_argument(
        "--omp-num-threads",
        type=int,
        default=int(os.environ.get("OMP_NUM_THREADS", 1)),
        help="OpenMP threads per job (default: $OMP_NUM_THREADS or 1).",
    )
    args = parser.parse_args()

    # Problem setup
    base_path = args.storms_path.resolve()
    resolutions = ["0pt25", "1pt00", "1pt50"]
    storm_dates = ["DEC2012", "NOV2018"]
    storm_paths = [base_path / f"{storm_date}_{res}.nc" 
                        for storm_date in storm_dates 
                        for res in resolutions]
    sea_levels = [0.0]
    scalings = [1.0]
    amr_max_levels = [2]

    # Construct jobs for all combinations of parameters
    jobs = []
    for sea_level in sea_levels:
        for amr_max_level in amr_max_levels:
            for scaling in scalings:
                for storm_path in storm_paths:
                    jobs.append(ETCJob(storm_path, 
                                       scaling=scaling, 
                                       sea_level=sea_level,
                                       levels=amr_max_level))

    ctrl = BatchController(
        jobs=jobs,
        executor=ParallelExecutor(
            max_workers=args.max_workers,
            env={"OMP_NUM_THREADS": str(args.omp_num_threads)},
        ),
        experiment="ETC_NASA_SLCT",
        clobber=ClobberPolicy.SKIP if args.resume else ClobberPolicy.OVERWRITE,
        )

    if args.setup_only:
        paths = ctrl.setup()
        print(f"Setup complete for {len(paths)} job(s).")
        return

    results = ctrl.run(wait=True)

    n_ok = sum(1 for r in results if r.success)
    n_fail = sum(1 for r in results if not r.success and r.returncode is not None)
    print(f"\nCompleted: {n_ok}/{len(results)} successful, {n_fail} failed.")

    if n_fail:
        for r in results:
            if r.returncode is not None and r.returncode != 0:
                print(f"  FAILED: {r.job.prefix}  (see {r.paths.log})")

    # Plot gauge comparison for all finished jobs
    gauge_comparison = clawutil.fullpath_import(Path(__file__).parent 
                                                / "gauge_comparison.py")
    gauge_figs_path = Path(os.environ['OUTPUT_PATH']) / ctrl.experiment / "gauge_comparisons"
    gauge_figs_path.mkdir(parents=True, exist_ok=True)
    for storm_date in storm_dates:
        gauge_comparison.plot(results, gauge_figs_path, storm_date)

if __name__ == "__main__":
    main()
