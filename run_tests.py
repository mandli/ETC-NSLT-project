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
from batch import JobPaths
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
            levels: int = 2,
            time_dilation: float = 1.0,
            ) -> None:

        super().__init__()

        self.prefix = (f"storm{storm_path.stem}_" +
                       f"sea{sea_level:.1f}_" +
                       f"scale{scaling:.2f}_" +
                       f"lev{levels}_" +
                       f"dil{time_dilation:.2f}")
        self.executable = "xgeoclaw"

        self.storm_path = storm_path
        self.scaling = scaling
        self.sea_level = sea_level
        self.time_dilation = time_dilation
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
            self.rundata.clawdata.tfinal = 4.0 * 24.0 * 3600.0  # 4 days in seconds
        elif self.storm_path.name.startswith("NOV2018"):
            etc_storm.time_offset = np.datetime64("2018-11-14T08:00:00.00")
            self.rundata.clawdata.tfinal = 4.0 * 24.0 * 3600.0  # 4 days in seconds
        elif self.storm_path.name.startswith("DEC1992"):
            etc_storm.time_offset = np.datetime64("1992-12-08T00:00:00.00")
            self.rundata.clawdata.tfinal = 8.0 * 24.0 * 3600.0  # 8 days in seconds
        etc_storm.file_format = 'netcdf'
        etc_storm.scaling = [self.scaling, 1.0]
        etc_storm.ramp_width = 2
        # crop_extent = [lon0, lon1, lat0, lat1]
        etc_storm.crop_extent = [-80, -62.5, 27.5, 45]
        etc_storm.storm_time_scale = self.time_dilation

        self.rundata.surge_data.storm_file = path / f"{self.prefix}.storm"
        etc_storm.write(self.rundata.surge_data.storm_file,
                        file_format='data',
                        var_mapping={"pressure": "msl"},
                        verbose=True)

        return super().write_data_objects(path)

    def post_run(self, result) -> None:
        plot_job(result, setplot=Path(__file__).parent / "setplot.py")

    def __repr__(self) -> str:
        return (f"ETCJob({self.storm_path}, "
                f"sea_level={self.rundata.geo_data.sea_level}, "
                f"scaling={self.scaling}, "
                f"levels={self.rundata.amrdata.amr_levels_max}, "
                f"time_dilation={self.time_dilation})")

    def __str__(self) -> str:
        return f"{self.prefix}"


@dataclass
class RunGroup:
    """All jobs for a single storm.

    Every job for the storm is overlaid on one gauge-comparison plot, so the
    group is keyed only by storm date; all other parameters (sea level,
    scaling, resolution, levels, time dilation) vary *within* the group and
    become the distinct lines on each gauge's plot.
    """
    storm_date: str
    jobs: list = field(default_factory=list)


# Visual-encoding palettes for gauge comparison plots.  Each ensemble
# parameter is mapped to a distinct visual channel so individual runs need no
# per-line legend entry (the mapping is decoded in a separate key figure):
#   color     <- resolution
#   linestyle <- sea level
#   marker    <- time dilation
#   linewidth <- scaling
# amr_max_levels is intentionally not encoded for now.
_RES_COLORS = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]
_SEA_LINESTYLES = ["-", "--", ":", "-."]
_DIL_MARKERS = ["o", "s", "^", "D", "v", "*", "P", "X"]
_SCALE_LINEWIDTHS = [1.0, 1.75, 2.5, 3.25]


def build_styles(jobs: list["ETCJob"]) -> tuple[list[dict], list[dict]]:
    """Map ensemble parameters to per-line plot styles.

    Returns
    -------
    styles
        One matplotlib-kwargs dict per job (parallel to *jobs*).
    key_channels
        Spec for the standalone legend figure: one section per channel that
        actually varies, each a ``{"title", "entries": [(label, kwargs)]}``.
    """
    def res_of(job: "ETCJob") -> str:
        return job.storm_path.stem.split("_")[1]

    res_vals = sorted({res_of(j) for j in jobs})
    sea_vals = sorted({j.sea_level for j in jobs})
    dil_vals = sorted({j.time_dilation for j in jobs})
    scale_vals = sorted({j.scaling for j in jobs})

    color_of = {v: _RES_COLORS[i % len(_RES_COLORS)] for i, v in enumerate(res_vals)}
    style_of = {v: _SEA_LINESTYLES[i % len(_SEA_LINESTYLES)] for i, v in enumerate(sea_vals)}
    marker_of = {v: _DIL_MARKERS[i % len(_DIL_MARKERS)] for i, v in enumerate(dil_vals)}
    width_of = {v: _SCALE_LINEWIDTHS[i % len(_SCALE_LINEWIDTHS)] for i, v in enumerate(scale_vals)}

    styles = [{"color": color_of[res_of(j)],
               "linestyle": style_of[j.sea_level],
               "marker": marker_of[j.time_dilation],
               "linewidth": width_of[j.scaling]} for j in jobs]

    # Each channel: title, sorted values, value->label, value->proxy kwargs
    # (isolating that one channel against neutral defaults).
    channels = [
        ("Resolution", res_vals, lambda v: str(v),
         lambda v: {"color": color_of[v], "linestyle": "-", "linewidth": 2.0}),
        ("Sea level (m)", sea_vals, lambda v: f"{v:.1f}",
         lambda v: {"color": "black", "linestyle": style_of[v], "linewidth": 2.0}),
        ("Time dilation", dil_vals, lambda v: f"{v:.2f}",
         lambda v: {"color": "black", "linestyle": "-", "marker": marker_of[v],
                    "linewidth": 2.0}),
        ("Scaling", scale_vals, lambda v: f"{v:.2f}",
         lambda v: {"color": "black", "linestyle": "-", "linewidth": width_of[v]}),
    ]
    key_channels = [
        {"title": title, "entries": [(fmt(v), kw(v)) for v in vals]}
        for title, vals, fmt, kw in channels
        if len(vals) > 1
    ]
    return styles, key_channels


def build_run_groups(
    storms_path: Path,
    storm_dates: list[str],
    resolutions: dict[str, list[str]],
    sea_levels: list[float],
    scalings: list[float],
    amr_max_levels: list[int],
    time_dilations: list[float],
) -> tuple[list[Job], list[RunGroup]]:
    """Build all jobs and group them by storm for gauge comparison.

    ``resolutions`` maps each storm date to the resolutions available for it,
    since not every storm has every resolution.  Storm files that do not exist
    on disk are skipped with a warning rather than failing mid-run.

    All jobs for a storm share one group and are overlaid on the same gauge
    plots.  The flat job list is submitted to a single controller so the
    worker queue stays full.
    """
    all_jobs: list[Job] = []
    groups: dict[str, RunGroup] = {sd: RunGroup(sd) for sd in storm_dates}

    for storm_date in storm_dates:
        for sea_level, amr_max_level, scaling, time_dilation, res in itertools.product(
                sea_levels, amr_max_levels, scalings, time_dilations,
                resolutions.get(storm_date, [])):
            storm_file = storms_path / f"{storm_date}_{res}.nc"
            if not storm_file.exists():
                logging.warning("Storm file not found, skipping: %s", storm_file)
                continue
            job = ETCJob(storm_file,
                         sea_level=sea_level,
                         scaling=scaling,
                         levels=amr_max_level,
                         time_dilation=time_dilation)
            all_jobs.append(job)
            groups[storm_date].jobs.append(job)

    return all_jobs, list(groups.values())


def report_results(results: list[JobResult]) -> None:
    n_ok = sum(1 for r in results if r.success)
    n_fail = sum(1 for r in results if not r.success and r.returncode is not None)
    print(f"\nCompleted: {n_ok}/{len(results)} successful, {n_fail} failed.")
    for r in results:
        if r.returncode is not None and r.returncode != 0:
            print(f"  FAILED: {r.job.prefix}  (see {r.paths.log})")


def plot_gauge_comparisons(
    groups: list[RunGroup],
    gauge_figs_path: Path,
    run_root: Path,
) -> None:
    """Call gauge_comparison.plot for each group.

    Drives the comparison off each group's jobs and their canonical output
    directories rather than off the jobs submitted this invocation, so that
    ``--resume`` (which skips already-completed jobs and therefore omits them
    from ``ctrl.run()``'s results) still regenerates every comparison plot.
    """
    for group in groups:
        group_results = []
        for job in group.jobs:
            job_dir = run_root / job.prefix
            if not job_dir.exists():
                continue
            paths = JobPaths(job=job_dir,
                             plots=job_dir / "plots",
                             log=job_dir / f"{job.prefix}_log.txt")
            group_results.append(JobResult(job=job, paths=paths, returncode=0))
        if not group_results:
            logging.warning("No output found for storm %s; skipping comparison.",
                            group.storm_date)
            continue
        styles, key_channels = build_styles([r.job for r in group_results])
        gc.plot(group_results, gauge_figs_path, group.storm_date, styles, key_channels)


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

    storm_dates = ["DEC1992", "DEC2012", "NOV2018"]
    # Resolutions available per storm — not every storm has every resolution
    # (DEC1992 currently only has 0pt25 data).
    resolutions = {
        "DEC1992": ["0pt25"],
        "DEC2012": ["0pt25", "1pt00", "1pt50"],
        "NOV2018": ["0pt25", "1pt00", "1pt50"],
    }
    # sea_levels = [0.0, 0.2, 0.4]
    sea_levels = [0.0]
    # scalings = [0.8, 1.0, 1.2]
    scalings = [0.8, 1.0]
    amr_max_levels = [2]
    time_dilations = [0.8, 1.0, 1.2]

    jobs, groups = build_run_groups(
        args.storms_path.resolve(),
        storm_dates,
        resolutions,
        sea_levels,
        scalings,
        amr_max_levels,
        time_dilations,
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

    # Run output lives under OUTPUT_PATH; the cross-run comparison figures are
    # written to the shared runs directory instead.
    run_root = Path(os.environ['OUTPUT_PATH']).expanduser().resolve() / ctrl.experiment
    gauge_figs_path = (Path(os.environ['SHARED_RUNS_PATH']).expanduser().resolve()
                       / "gauge_comparisons")
    gauge_figs_path.mkdir(parents=True, exist_ok=True)
    plot_gauge_comparisons(groups, gauge_figs_path, run_root)


if __name__ == "__main__":
    main()
