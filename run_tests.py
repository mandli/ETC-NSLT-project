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
NCAR Derecho uses PBS Pro.  The ``batch`` scheduler backend sources a
per-machine ``env_file`` in the (non-login) compute-node shell to set up
modules + the venv, so the job does *not* depend on interactive rc files.
Point ``--env-file`` / ``$BATCH_ENV_FILE`` at your copy (on this setup,
``~/.config/sci/activate.sh``) and ``--python`` at that env's interpreter.

``--scheduler pbs-packed`` (used here) fans the sweep across ``--nodes``
exclusive nodes; each node re-invokes this script to self-pack its shard with
the local pool.  Submit-and-exit — cross-run comparison figures come from a
separate ``--plot-only`` pass once the jobs finish.

Typical workflow on Derecho::

    # 0. Environment (login shell; DATA_PATH/OUTPUT_PATH/SHARED_RUNS_PATH come
    #    from your dotfiles and the env_file so packed re-runs see them too)
    export BATCH_ACCOUNT=NCAR0001                    # your Derecho project code
    export BATCH_ENV_FILE=~/.config/sci/activate.sh  # sourced on the compute node

    # 1. Inspect the generated wrappers without submitting
    python run_tests.py --scheduler pbs-packed --nodes 8 --setup-only \
        --python ~/.venvs/sci/bin/python

    # 2. Submit the ensemble (one PBS job per node, each self-packing its shard)
    python run_tests.py --scheduler pbs-packed --nodes 8 \
        --node-cpus 128 --max-workers 16 --omp-num-threads 8 \
        --walltime 12:00:00 --python ~/.venvs/sci/bin/python

    # 3. Watch the queue
    qstat -u $USER

    # 4. After the jobs finish, build the comparison figures
    python run_tests.py --plot-only

    # Resume after a walltime kill (only unfinished job dirs are re-packed)
    python run_tests.py --scheduler pbs-packed --nodes 8 --resume \
        --python ~/.venvs/sci/bin/python

The non-packed ``--scheduler pbs`` backend (one ``qsub`` per run) also works and
likewise requires ``--env-file`` / ``$BATCH_ENV_FILE``.  The Derecho project code
is taken from ``--account`` or ``$BATCH_ACCOUNT``; queue, walltime, threads, env
file, python, and modules are all overridable on the command line.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import os
import socket
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from batch import Job, JobResult, JobPaths
from batch import add_execution_args, execute, submit_packed, PackedResources
from batch import plot_performance
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


def plot_performance_analysis(
    groups: list[RunGroup],
    perf_figs_path: Path,
    run_root: Path,
) -> None:
    """Generate timing comparison plots for each storm group.

    Delegates the three-panel figure to ``batch.analysis.plot_performance``,
    which parses ``timing.txt`` from each job directory (skipping any that are
    missing).  Labels strip the shared ``storm<date>_`` prefix so each bar shows
    only the run parameters.
    """
    for group in groups:
        prefix_strip = f"storm{group.storm_date}_"
        job_dirs = [run_root / job.prefix for job in group.jobs]
        labels = [
            job.prefix.replace(prefix_strip, "").replace("_", "\n")
            for job in group.jobs
        ]
        plot_performance(
            job_dirs,
            labels=labels,
            out_path=perf_figs_path / f"performance_{group.storm_date}.png",
            title=f"Performance Analysis — {group.storm_date}",
        )


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


def submit_packed_ensemble(args) -> None:
    """Fan the sweep across ``--nodes`` exclusive nodes, packing each shard.

    ``batch.submit_packed`` renders one wrapper per node (PBS or SLURM) and
    submits it; each wrapper re-invokes this script in ``--scheduler local
    --shard i/n --pin-cpus`` mode so the local pool packs that shard onto the
    node.  Submit-and-exit — run ``--plot-only`` afterward for the cross-run
    figures.  The oversubscription check and the trailer print are project
    conveniences batch does not provide.
    """
    # The packed wrapper `source`s this file on the compute node to set up
    # modules + venv; an empty value would render `source ''` and fail obscurely.
    if not args.env_file:
        sys.exit("--env-file (or $BATCH_ENV_FILE) is required for packed "
                 "submission; point it at your machine env_file "
                 "(e.g. ~/.config/sci/activate.sh).")

    if args.max_workers * args.omp_num_threads > args.node_cpus:
        logging.warning(
            "max-workers(%d) * omp-num-threads(%d) = %d exceeds node-cpus(%d); "
            "packed jobs will oversubscribe the node.",
            args.max_workers, args.omp_num_threads,
            args.max_workers * args.omp_num_threads, args.node_cpus,
        )

    here = Path(__file__).resolve()
    scheduler = args.scheduler.split("-")[0]  # "pbs" or "slurm"
    script_dir = (Path(os.environ['OUTPUT_PATH']).expanduser().resolve()
                  / "ETC_NASA_SLCT" / "_pack_scripts")

    def inner(shard_i: int, n_shards: int) -> list[str]:
        cmd = [
            args.python, str(here),
            "--scheduler", "local",
            "--shard", f"{shard_i}/{n_shards}",
            "--max-workers", str(args.max_workers),
            "--omp-num-threads", str(args.omp_num_threads),
            "--node-cpus", str(args.node_cpus),
            "--pin-cpus",
            "--no-comparison",
            "--run-label", args.run_label,
            "--storms-path", str(args.storms_path),
        ]
        if args.resume:
            cmd.append("--resume")
        if args.no_run_plots:
            cmd.append("--no-run-plots")
        return cmd

    resources = PackedResources(
        queue=args.queue,
        walltime=args.walltime,
        account=args.account,
        node_cpus=args.node_cpus,
        modules=args.modules,
    )
    job_ids = submit_packed(
        args.nodes,
        inner,
        resources,
        scheduler,
        script_dir,
        env_file=args.env_file,
        python=args.python,
        dry_run=args.setup_only,
        name_prefix="etc_pack",
        workdir=here.parent,
    )

    if args.setup_only:
        print(f"Setup complete: {len(job_ids)} packed wrapper(s) written; "
              "none submitted.")
    else:
        print(f"Submitted {len(job_ids)} packed node job(s).")
    print("Run `python run_tests.py --plot-only` once the jobs finish.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # Shared execution flags: --scheduler {local,pbs,slurm,pbs-packed,
    # slurm-packed}, --setup-only, --resume, --max-workers, --omp-num-threads,
    # --account ($BATCH_ACCOUNT), --queue, --walltime, --modules ($BATCH_MODULES),
    # and the packing flags --nodes/--node-cpus/--shard/--pin-cpus.
    add_execution_args(parser)
    # Project-specific flags (no collision with the shared group).
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip submission; only (re)generate comparison figures from "
             "existing output. Run this after a batch finishes.",
    )
    parser.add_argument(
        "--no-run-plots",
        action="store_true",
        help="Skip per-job plotting (local post_run / compute-node plotclaw). "
             "The aggregate comparison figures are unaffected.",
    )
    parser.add_argument(
        "--no-comparison",
        action="store_true",
        help="Local only: run the jobs but skip the aggregate comparison figures.",
    )
    parser.add_argument(
        "--storms-path",
        type=Path,
        default=Path(os.environ['DATA_PATH']) / "storms" / "ETC_NASA_SLCT",
        help="Directory containing netCDF storm files.",
    )
    parser.add_argument(
        "--run-label",
        default=os.environ.get("HOSTNAME") or socket.gethostname(),
        help="Label subdirectory for output plots (default: $HOSTNAME or hostname).",
    )
    args = parser.parse_args()

    # Packed: fan the sweep across --nodes exclusive nodes as independent
    # submissions, each self-packing its shard.  Nothing to build here.
    if args.scheduler.endswith("-packed"):
        submit_packed_ensemble(args)
        return

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
    # comparison figures from whatever output already exists on disk.  Runs before
    # execute() and always sees the full job set (sharding, applied inside
    # execute() on the local path, does not affect this disk-driven pass).
    if args.plot_only:
        generate_comparison_plots(groups, args.run_label)
        return

    # execute() dispatches on --scheduler: local blocks and reports results;
    # pbs/slurm submit-and-exit; --shard restricts the local job set; --setup-only
    # writes .data (local) or submission scripts (scheduler); plot=/setplot= turn
    # on compute-node self-plotting for the scheduler backends.
    execute(
        args,
        jobs,
        experiment="ETC_NASA_SLCT",
        plot=not args.no_run_plots,
        setplot=str(Path(__file__).parent / "setplot.py"),
    )

    if args.scheduler == "local":
        if not args.setup_only and not args.no_comparison:
            generate_comparison_plots(groups, args.run_label)
    elif not args.setup_only:
        # Scheduler submit-and-exit: execute() already reported the submitted
        # job IDs; build the comparison figures with --plot-only once they finish.
        print("Run `python run_tests.py --plot-only` once the jobs finish.")


if __name__ == "__main__":
    main()
