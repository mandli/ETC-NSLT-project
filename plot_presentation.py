#!/usr/bin/env python

"""Slide-ready single-gauge comparison figure from existing ensemble output.

Presentation counterpart to the per-(dilation, scaling) gauge comparisons that
``run_tests.py --plot-only`` writes.  Like ``run_sensitivity.py`` it is purely
disk-driven -- it never builds, submits, or runs a job -- and it discovers runs
by parsing the job-directory names under the output root rather than from
``run_tests.py``'s sweep lists, so it sees the whole ensemble that is actually
on disk.  All discovered runs (every wind scaling, dilation, and track shift)
are grouped by forcing resolution::

    python plot_presentation.py                          # DEC2012, New Haven
    python plot_presentation.py --style lines
    python plot_presentation.py --output-path $OUTPUT_PATH/ETC_NASA_SLCT_pike

Only runs at ``--sea-level`` (default 0.0) are used, since the plotted surface
includes the initial sea level and the legend does not report it.  Figures
land under ``$SHARED_RUNS_PATH/gauge_comparisons/<run-label>/presentation/``.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import socket

import ensemble_sensitivity as es
import gauge_comparison as gc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("plot_presentation")


def discover_runs(run_root: Path, storm: str, gauge_id: int,
                  sea_level: float | None) -> list[tuple[float, Path]]:
    """Return ``(resolution_degrees, job_dir)`` for each matching run on disk.

    Skips directories that do not parse as an ``ETCJob.prefix``, belong to
    another storm, are at another sea level (unless ``sea_level`` is None), or
    lack the requested gauge's output file.
    """
    runs = []
    for job_dir in sorted(p for p in run_root.iterdir() if p.is_dir()):
        params = es.parse_prefix(job_dir.name)
        if params is None or params["storm"] != storm:
            continue
        if sea_level is not None and params["sea_level"] != sea_level:
            continue
        if not (job_dir / f"gauge{gauge_id:05d}.txt").exists():
            logger.warning("No gauge %d output in %s; skipping.",
                           gauge_id, job_dir.name)
            continue
        runs.append((params["res"], job_dir))
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--output-path", type=Path,
        default=Path(os.environ["OUTPUT_PATH"]).expanduser() / "ETC_NASA_SLCT",
        help="Ensemble output root (default: $OUTPUT_PATH/ETC_NASA_SLCT).")
    parser.add_argument("--storm", default="DEC2012", choices=sorted(gc.times))
    parser.add_argument("--gauge", type=int, default=5,
                        choices=sorted(gc.gauge_mapping),
                        help="Gauge number (default: 5, New Haven, CT).")
    parser.add_argument("--style", default="envelope",
                        choices=["envelope", "lines"],
                        help="Min/max envelope + median, or thin lines.")
    parser.add_argument("--sea-level", type=float, default=0.0,
                        help="Only use runs at this initial sea level.")
    parser.add_argument("--no-baseline", action="store_true",
                        help="Omit the observed pre-storm mean reference line.")
    parser.add_argument(
        "--run-label",
        default=os.environ.get("HOSTNAME") or socket.gethostname(),
        help="Label subdirectory for output plots (default: $HOSTNAME or hostname).")
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Override the figure directory "
             "(default: $SHARED_RUNS_PATH/gauge_comparisons/<label>/presentation).")
    args = parser.parse_args()

    run_root = args.output_path.expanduser().resolve()
    if not run_root.is_dir():
        raise SystemExit(f"Output root not found: {run_root}")

    runs = discover_runs(run_root, args.storm, args.gauge, args.sea_level)
    if not runs:
        raise SystemExit(f"No {args.storm} runs at sea level {args.sea_level} "
                         f"under {run_root}")
    counts = {}
    for res, _ in runs:
        counts[res] = counts.get(res, 0) + 1
    logger.info("Using %d runs from %s: %s", len(runs), run_root,
                ", ".join(f"{r:g} deg x{n}" for r, n in sorted(counts.items())))

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = (Path(os.environ["SHARED_RUNS_PATH"]).expanduser().resolve()
                   / "gauge_comparisons" / args.run_label / "presentation")

    written = gc.plot_presentation(runs, out_dir, storm_date=args.storm,
                                   gauge_id=args.gauge, style=args.style,
                                   baseline=not args.no_baseline)
    for path in written:
        logger.info("Wrote %s", path)


if __name__ == "__main__":
    main()
