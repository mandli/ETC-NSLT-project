#!/usr/bin/env python

"""Run storm surge ensemble jobs for NASA SLCT ETC storms.

The ensemble is driven through the ``batch`` package, whose pluggable executor
backends let the *same* job definitions run either locally or on an HPC
scheduler.  Two backends are wired up here, selected with ``--scheduler``:

Local runs (default)
--------------------
Runs the solver directly on this machine with a bounded worker pool, waits for
every job, then generates the cross-run comparison figures::

    python run_tests.py                       # full ensemble, local
    python run_tests.py --setup-only          # write .data files only
    python run_tests.py --resume              # skip finished job dirs

``--max-workers`` sets how many jobs run at once and ``--omp-num-threads`` sets
OpenMP threads per job; keep ``max_workers * omp_num_threads`` at or below the
core count to avoid oversubscription.

PBS runs on Derecho
-------------------
NCAR Derecho uses PBS Pro.  Each job becomes an independent ``qsub``
submission; the script submits everything and returns immediately (it does not
tie up a login-node process).  Each job self-plots on the compute node, and the
cross-run comparison figures are produced by a separate ``--plot-only`` pass
once the jobs have finished.

Typical workflow on Derecho::

    # 0. Environment (in your job/login shell)
    export DATA_PATH=...            # storm + topo inputs
    export OUTPUT_PATH=...          # scratch: per-job run directories
    export SHARED_RUNS_PATH=...     # where comparison figures are collected
    export PBS_ACCOUNT=NCAR0001     # your Derecho project code
    module load ncarenv conda       # or whatever provides clawpack

    # 1. Inspect the generated scripts without submitting
    python run_tests.py --scheduler pbs --setup-only

    # 2. Submit the ensemble (one PBS job per run)
    python run_tests.py --scheduler pbs --walltime 12:00:00

    # 3. Watch the queue
    qstat -u $USER

    # 4. After the jobs finish, build the comparison figures
    python run_tests.py --plot-only

    # Resume after a walltime kill (only unfinished jobs are resubmitted)
    python run_tests.py --scheduler pbs --resume

The Derecho project code is taken from ``--account`` or ``$PBS_ACCOUNT``; queue,
walltime, threads, and modules are all overridable on the command line.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import os
import re
import socket
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from batch import Job, ParallelExecutor, BatchController, ClobberPolicy, JobResult
from batch import JobPaths
from batch import PBSExecutor, PBSResources
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
        # Absolute setplot path so the PBS script's compute-node plotclaw call
        # (and post_run for local runs) resolves correctly regardless of cwd.
        self.setplot = str(Path(__file__).parent / "setplot.py")

        self.storm_path = storm_path
        self.scaling = scaling
        self.sea_level = sea_level
        self.time_dilation = time_dilation
        self.levels = levels
        # Per-run plotting (post_run) is on by default; --no-run-plots clears it.
        self.plot_per_run = True

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
        if self.plot_per_run:
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
# parameter is mapped to a distinct visual channel:
#   color     <- resolution
#   linestyle <- sea level
#   marker    <- time dilation
#   linewidth <- scaling
# amr_max_levels is intentionally not encoded for now.
_RES_COLORS = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]
_SEA_LINESTYLES = ["-", "--", ":", "-."]
_DIL_MARKERS = ["o", "s", "^", "D", "v", "*", "P", "X"]
_SCALE_LINEWIDTHS = [1.0, 1.75, 2.5, 3.25]


def build_styles(jobs: list["ETCJob"]) -> list[dict]:
    """Map ensemble parameters to per-line plot styles.

    Returns one matplotlib-kwargs dict per job (parallel to *jobs*).
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

    styles = []
    for j in jobs:
        s = {"color": color_of[res_of(j)],
             "linestyle": style_of[j.sea_level],
             "linewidth": width_of[j.scaling]}
        if len(dil_vals) > 1:
            s["marker"] = marker_of[j.time_dilation]
        styles.append(s)

    return styles


def build_run_groups(
    storms_path: Path,
    storm_dates: list[str],
    resolutions: dict[str, list[str]],
    sea_levels: list[float],
    scalings: list[float],
    amr_max_levels: list[int],
    time_dilations: list[float],
) -> tuple[list["ETCJob"], list[RunGroup]]:
    """Build all jobs and group them by storm for gauge comparison.

    ``resolutions`` maps each storm date to the resolutions available for it,
    since not every storm has every resolution.  Storm files that do not exist
    on disk are skipped with a warning rather than failing mid-run.

    All jobs for a storm share one group and are overlaid on the same gauge
    plots.  The flat job list is submitted to a single controller so the
    worker queue stays full.
    """
    all_jobs: list["ETCJob"] = []
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


def parse_timing(job_dir: Path) -> dict | None:
    """Parse timing.txt produced by the GeoClaw solver.

    Returns a dict with:
      levels: list of {level, wall, cpu, cells} (per AMR level)
      total_integration: {wall, cpu, cells}
      components: {stepgrid, bc, regrid, output: {wall, cpu}}
      total: {wall, cpu}
      n_threads: int
    Returns None if timing.txt is absent.
    """
    txt_path = job_dir / "timing.txt"
    if not txt_path.exists():
        return None

    text = txt_path.read_text()
    result: dict = {"levels": [], "components": {}}

    for m in re.finditer(
            r"^\s+(\d+)\s+([\d.E+]+)\s+([\d.E+]+)\s+([\d.E+]+)",
            text, re.MULTILINE):
        result["levels"].append({
            "level": int(m.group(1)),
            "wall": float(m.group(2)),
            "cpu": float(m.group(3)),
            "cells": float(m.group(4)),
        })

    m = re.search(r"^total\s+([\d.E+]+)\s+([\d.E+]+)\s+([\d.E+]+)",
                  text, re.MULTILINE)
    if m:
        result["total_integration"] = {
            "wall": float(m.group(1)), "cpu": float(m.group(2)),
            "cells": float(m.group(3)),
        }

    for key, pattern in [
        ("stepgrid", r"stepgrid\s+([\d.E+]+)\s+([\d.E+]+)"),
        ("bc",       r"BC/ghost cells\s+([\d.E+]+)\s+([\d.E+]+)"),
        ("regrid",   r"Regridding\s+([\d.E+]+)\s+([\d.E+]+)"),
        ("output",   r"Output \(valout\)\s+([\d.E+]+)\s+([\d.E+]+)"),
    ]:
        m = re.search(pattern, text)
        if m:
            result["components"][key] = {
                "wall": float(m.group(1)), "cpu": float(m.group(2))}

    m = re.search(r"Total time:\s+([\d.E+]+)\s+([\d.E+]+)", text)
    if m:
        result["total"] = {"wall": float(m.group(1)), "cpu": float(m.group(2))}

    m = re.search(r"Using (\d+) thread", text)
    result["n_threads"] = int(m.group(1)) if m else 1

    return result


def _plot_perf_group(
    timings: list[dict],
    jobs: list["ETCJob"],
    storm_date: str,
    out_path: Path,
) -> None:
    """Three-panel performance figure for one storm group.

    Left:   total wall time bars + CPU efficiency overlay
    Middle: stacked wall time by AMR level
    Right:  stacked wall time by solver component
    """
    n = len(timings)
    x = np.arange(n)

    prefix_strip = f"storm{storm_date}_"
    short = [j.prefix.replace(prefix_strip, "").replace("_", "\n") for j in jobs]
    tick_fs = max(4, 9 - n // 8)

    fig, axes = plt.subplots(1, 3, figsize=(max(14, n * 0.9 + 4), 6))
    fig.suptitle(f"Performance Analysis — {storm_date}", fontsize=13)

    # Total wall time (bars) + CPU efficiency (line on twin axis)
    ax = axes[0]
    wall_min = [t["total"]["wall"] / 60 for t in timings]
    ax.bar(x, wall_min, color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=90, fontsize=tick_fs)
    ax.set_ylabel("Wall Time (min)")
    ax.set_title("Total Wall Time")

    ax2 = ax.twinx()
    eff = [t["total"]["cpu"] / (t.get("n_threads", 1) * t["total"]["wall"]) * 100
           for t in timings]
    ax2.plot(x, eff, "o-", color="orange", label="CPU efficiency")
    ax2.set_ylabel("CPU Efficiency (%)", color="orange")
    ax2.tick_params(axis="y", labelcolor="orange")
    ax2.set_ylim(0, 110)

    # Stacked wall time by AMR level
    ax = axes[1]
    max_lev = max(len(t["levels"]) for t in timings)
    bottoms = np.zeros(n)
    for li in range(max_lev):
        vals = np.array([
            t["levels"][li]["wall"] / 60 if li < len(t["levels"]) else 0.0
            for t in timings
        ])
        ax.bar(x, vals, bottom=bottoms, label=f"Level {li + 1}")
        bottoms += vals
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=90, fontsize=tick_fs)
    ax.set_ylabel("Wall Time (min)")
    ax.set_title("Time by AMR Level")
    ax.legend(fontsize=8)

    # Stacked wall time by solver component
    ax = axes[2]
    bottoms = np.zeros(n)
    for key, lbl in [("stepgrid", "Step (PDE)"), ("bc", "BC/Ghost"),
                     ("regrid", "Regrid"), ("output", "Output")]:
        vals = np.array([
            t["components"].get(key, {}).get("wall", 0.0) / 60 for t in timings
        ])
        ax.bar(x, vals, bottom=bottoms, label=lbl)
        bottoms += vals
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=90, fontsize=tick_fs)
    ax.set_ylabel("Wall Time (min)")
    ax.set_title("Time by Component")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig_path = out_path / f"performance_{storm_date}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info("Performance plot saved: %s", fig_path)


def plot_performance_analysis(
    groups: list[RunGroup],
    perf_figs_path: Path,
    run_root: Path,
) -> None:
    """Generate timing comparison plots for each storm group.

    Reads timing.txt from each job's output directory (skipping any that are
    missing) and writes one performance figure per group to *perf_figs_path*.
    """
    for group in groups:
        timings, valid_jobs = [], []
        for job in group.jobs:
            t = parse_timing(run_root / job.prefix)
            if t is not None:
                timings.append(t)
                valid_jobs.append(job)
        if not timings:
            logging.warning("No timing data for storm %s; skipping performance plot.",
                            group.storm_date)
            continue
        _plot_perf_group(timings, valid_jobs, group.storm_date, perf_figs_path)


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

    Results are split by (time_dilation, scaling) so that each output figure
    set has at most 6 lines (3 resolutions × 2 sea levels), making individual
    runs legible with an in-plot legend.  Output lives under
    ``gauge_figs_path / storm_date / dil{d:.2f}_sc{s:.2f}/``.
    """
    def _res_of(job: "ETCJob") -> str:
        return job.storm_path.stem.split("_")[1]

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

        # Split by (time_dilation, scaling) → each subplot-set has ≤6 lines
        sub_groups: dict[tuple, list] = {}
        for result in group_results:
            key = (result.job.time_dilation, result.job.scaling)
            sub_groups.setdefault(key, []).append(result)

        for (dil, scale), sub_results in sorted(sub_groups.items()):
            out_dir = (gauge_figs_path / group.storm_date
                       / f"dil{dil:.2f}_sc{scale:.2f}")
            styles = build_styles([r.job for r in sub_results])
            labels = [f"{_res_of(r.job)}, SL={r.job.sea_level:.1f}"
                      for r in sub_results]
            subtitle = f"Dilation={dil:.2f}, Scaling={scale:.2f}"
            gc.plot(sub_results, out_dir, group.storm_date, styles,
                    labels=labels, subtitle=subtitle)


def generate_comparison_plots(groups: list[RunGroup], run_label: str) -> None:
    """Build the cross-run gauge and performance comparison figures.

    Reads each job's solver output (``fort.gauge``, ``timing.txt``) directly
    from its canonical directory under ``OUTPUT_PATH``, so this works whether
    the jobs ran locally or on PBS — it only needs the output on disk.  The
    figures are collected under ``SHARED_RUNS_PATH`` in a per-label
    subdirectory.
    """
    run_root = Path(os.environ['OUTPUT_PATH']).expanduser().resolve() / "ETC_NASA_SLCT"
    shared_runs = Path(os.environ['SHARED_RUNS_PATH']).expanduser().resolve()

    gauge_figs_path = shared_runs / "gauge_comparisons" / run_label
    gauge_figs_path.mkdir(parents=True, exist_ok=True)
    plot_gauge_comparisons(groups, gauge_figs_path, run_root)

    perf_figs_path = shared_runs / "performance" / run_label
    perf_figs_path.mkdir(parents=True, exist_ok=True)
    plot_performance_analysis(groups, perf_figs_path, run_root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scheduler",
        choices=["local", "pbs"],
        default="local",
        help="Execution backend: 'local' (subprocess pool, default) or 'pbs' "
             "(qsub on Derecho, submit-and-exit).",
    )
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Set up jobs but do not execute: local writes .data files; PBS "
             "writes .data files and the qsub scripts without submitting them.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip submission; only (re)generate comparison figures from "
             "existing output. Run this after a PBS batch finishes.",
    )
    parser.add_argument(
        "--no-run-plots",
        action="store_true",
        help="Skip per-job plotting (local post_run / PBS compute-node plotclaw). "
             "The aggregate comparison figures are unaffected.",
    )
    parser.add_argument(
        "--no-comparison",
        action="store_true",
        help="Local only: run the jobs but skip the aggregate comparison figures.",
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
        help="Local only: max concurrent jobs (default: $BATCH_MAX_JOBS or 4).",
    )
    parser.add_argument(
        "--omp-num-threads",
        type=int,
        default=int(os.environ.get("OMP_NUM_THREADS", 1)),
        help="OpenMP threads per job (default: $OMP_NUM_THREADS or 1). For PBS "
             "this also sets ncpus/ompthreads in the select request.",
    )
    parser.add_argument(
        "--account",
        default=os.environ.get("PBS_ACCOUNT", ""),
        help="PBS only: Derecho project code (#PBS -A; default: $PBS_ACCOUNT).",
    )
    parser.add_argument(
        "--queue",
        default="main",
        help="PBS only: queue name (default: main).",
    )
    parser.add_argument(
        "--walltime",
        default="12:00:00",
        help="PBS only: walltime limit HH:MM:SS (default: 12:00:00).",
    )
    parser.add_argument(
        "--pbs-modules",
        nargs="*",
        default=(os.environ.get("PBS_MODULES", "").split()),
        help="PBS only: modules to 'module load' in the job script "
             "(default: $PBS_MODULES, space-separated).",
    )
    parser.add_argument(
        "--run-label",
        default=os.environ.get("HOSTNAME") or socket.gethostname(),
        help="Label subdirectory for output plots (default: $HOSTNAME or hostname).",
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
    sea_levels = [0.0, 0.2]
    scalings = [0.8, 1.0, 1.2]
    amr_max_levels = [5]
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

    for job in jobs:
        job.plot_per_run = not args.no_run_plots

    # --plot-only: skip the controller/executor entirely and just (re)build the
    # comparison figures from whatever output already exists on disk.
    if args.plot_only:
        generate_comparison_plots(groups, args.run_label)
        return

    if args.scheduler == "pbs":
        executor = PBSExecutor(
            default_resources=PBSResources(
                queue=args.queue,
                nodes=1,
                ncpus=args.omp_num_threads,
                mpiprocs=1,
                ompthreads=args.omp_num_threads,
                walltime=args.walltime,
                account=args.account,
                env_vars={"OMP_NUM_THREADS": str(args.omp_num_threads)},
                modules=args.pbs_modules,
                plot=not args.no_run_plots,
                setplot=str(Path(__file__).parent / "setplot.py"),
            ),
            # --setup-only writes the qsub scripts without submitting them.
            dry_run=args.setup_only,
        )
    else:
        executor = ParallelExecutor(
            max_workers=args.max_workers,
            env={"OMP_NUM_THREADS": str(args.omp_num_threads)},
        )

    ctrl = BatchController(
        jobs=jobs,
        executor=executor,
        experiment="ETC_NASA_SLCT",
        clobber=ClobberPolicy.SKIP if args.resume else ClobberPolicy.OVERWRITE,
    )

    if args.setup_only:
        if args.scheduler == "pbs":
            # dry_run executor: writes .data files and qsub scripts, no submit.
            results = ctrl.run(wait=False)
            print(f"Setup complete: {len(results)} PBS script(s) written; "
                  "none submitted.")
        else:
            paths = ctrl.setup()
            print(f"Setup complete for {len(paths)} job(s).")
        return

    if args.scheduler == "pbs":
        # Submit-and-exit: qsub returns immediately. Each job self-plots on the
        # compute node; run `--plot-only` afterward for the comparison figures.
        results = ctrl.run(wait=False)
        print(f"Submitted {len(results)} job(s) to PBS.")
        for r in results:
            print(f"  {r.job.prefix}  ->  PBS job {r.job_id}")
        print("Run `python run_tests.py --plot-only` once the jobs finish.")
        return

    # Local: block until every job completes, then build comparison figures.
    results = ctrl.run(wait=True)
    report_results(results)
    if not args.no_comparison:
        generate_comparison_plots(groups, args.run_label)


if __name__ == "__main__":
    main()
