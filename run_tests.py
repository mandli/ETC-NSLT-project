#!/usr/bin/env python

"""Run storm surge ensemble jobs for NASA SLCT ETC storms."""

from __future__ import annotations

import argparse
import itertools
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from batch import Job, ParallelExecutor, BatchController, ClobberPolicy, JobResult
from batch.plot import plot_job

import clawpack.clawutil.util as clawutil
from clawpack.geoclaw.surge.storm import Storm

import gauge_comparison as gc

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

        self.storm_path = storm_path
        self.scaling = scaling
        self.sea_level = sea_level
        self.levels = levels

        setrun_path = Path(__file__).parent / "setrun.py"
        setrun = clawutil.fullpath_import(setrun_path)
        self.rundata = setrun.setrun()

        self.rundata.amrdata.amr_levels_max = self.levels
        self.rundata.geo_data.sea_level = self.sea_level

    def write_data_objects(self, path: Path) -> None:
        etc_storm = Storm()
        etc_storm.file_paths.append(self.storm_path)
        if self.storm_path.name.startswith("DEC2012"):
            etc_storm.time_offset = np.datetime64("2012-12-26T00:00:00.00")
        elif self.storm_path.name.startswith("NOV2018"):
            etc_storm.time_offset = np.datetime64("2018-11-14T08:00:00.00")
        etc_storm.file_format = 'netcdf'
        etc_storm.scaling = [self.scaling, 1.0]
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
        return (f"ETCJob({self.storm_path}, "
                f"sea_level={self.rundata.geo_data.sea_level}, "
                f"scaling={self.scaling}, "
                f"levels={self.rundata.amrdata.amr_levels_max})")

    def __str__(self) -> str:
        return f"{self.prefix}"


@dataclass
class RunGroup:
    """Fixed-parameter group whose jobs vary only by resolution.

    These are the jobs compared together in a single gauge-comparison plot.
    """
    storm_date: str
    sea_level: float
    scaling: float
    amr_max_level: int
    # time_dilation: float          # next step
    jobs: list = field(default_factory=list)

    def label(self) -> str:
        return (f"{self.storm_date}"
                f"_sea{self.sea_level:.1f}"
                f"_scale{self.scaling:.2f}"
                f"_lev{self.amr_max_level}")
        # f"_dil{self.time_dilation:.2f}"  # next step


def build_run_groups(
    storms_path: Path,
    storm_dates: list[str],
    resolutions: list[str],
    sea_levels: list[float],
    scalings: list[float],
    amr_max_levels: list[int],
    # time_dilations: list[float],  # next step
) -> tuple[list[Job], list[RunGroup]]:
    """Build all jobs and group them for gauge comparison.

    Jobs within a group share all parameters except resolution, which is
    the dimension compared in gauge plots.  The flat job list is submitted
    to a single controller so the worker queue stays full.
    """
    all_jobs: list[Job] = []
    groups: list[RunGroup] = []

    for sea_level, amr_max_level, scaling, storm_date in itertools.product(
            sea_levels, amr_max_levels, scalings, storm_dates):
        # time_dilation,            # next step
        group = RunGroup(storm_date, sea_level, scaling, amr_max_level)
        for res in resolutions:
            job = ETCJob(storms_path / f"{storm_date}_{res}.nc",
                         sea_level=sea_level,
                         scaling=scaling,
                         levels=amr_max_level)
            all_jobs.append(job)
            group.jobs.append(job)
        groups.append(group)

    return all_jobs, groups


def report_results(results: list[JobResult]) -> None:
    n_ok = sum(1 for r in results if r.success)
    n_fail = sum(1 for r in results if not r.success and r.returncode is not None)
    print(f"\nCompleted: {n_ok}/{len(results)} successful, {n_fail} failed.")
    for r in results:
        if r.returncode is not None and r.returncode != 0:
            print(f"  FAILED: {r.job.prefix}  (see {r.paths.log})")


def plot_gauge_comparisons(
    results: list[JobResult],
    groups: list[RunGroup],
    gauge_figs_path: Path,
) -> None:
    """Call gauge_comparison.plot for each group using only that group's results."""
    job_to_result = {r.job: r for r in results}
    for group in groups:
        group_results = [job_to_result[job] for job in group.jobs
                         if job in job_to_result]
        if not group_results:
            continue
        gc.plot(group_results, gauge_figs_path / group.label(), group.storm_date)


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

    storm_dates = ["DEC2012", "NOV2018"]
    resolutions = ["0pt25", "1pt00", "1pt50"]
    sea_levels = [0.0]
    scalings = [1.0]
    amr_max_levels = [2]
    # time_dilations = [1.0]        # next step

    jobs, groups = build_run_groups(
        args.storms_path.resolve(),
        storm_dates,
        resolutions,
        sea_levels,
        scalings,
        amr_max_levels,
        # time_dilations,           # next step
    )

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
    report_results(results)

    gauge_figs_path = (Path(os.environ['OUTPUT_PATH'])
                       / ctrl.experiment / "gauge_comparisons")
    gauge_figs_path.mkdir(parents=True, exist_ok=True)
    plot_gauge_comparisons(results, groups, gauge_figs_path)


if __name__ == "__main__":
    main()
