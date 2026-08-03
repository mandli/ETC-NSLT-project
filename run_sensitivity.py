#!/usr/bin/env python

"""Compute Ensemble Sensitivity Analysis (ESA) figures from ensemble output.

Post-processing counterpart to ``run_tests.py``: it reads the peak surge at each
tide gauge from the already-computed ensemble output on disk and derives the
sensitivity of peak surge to each sweep parameter (wind strength, translation,
resolution, sea level), keeping storms and gauges separate.  See
``ensemble_sensitivity.py`` for the method.

Like ``run_tests.py --plot-only`` this is purely disk-driven — it never submits
or runs the solver — so run it once the ensemble has finished::

    python run_sensitivity.py                       # all storms, default label
    python run_sensitivity.py --storms DEC2012      # one storm
    python run_sensitivity.py --window 0 4          # override the peak window (days)

Output figures land under
``$SHARED_RUNS_PATH/sensitivity/<run-label>/<storm>/`` alongside the tidy
peak-surge table and a sensitivity summary, both written as CSV for auditing.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
from pathlib import Path
import socket

import ensemble_sensitivity as es

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_sensitivity")


def write_table_csv(table: es.Table, path: Path) -> None:
    """Write the tidy peak-surge table (one row per storm/gauge/member)."""
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=es.COLUMNS)
        writer.writeheader()
        writer.writerows(table.rows)


def write_summary_csv(results: dict, path: Path,
                      params=es.PARAMS) -> None:
    """Write per (storm, gauge, parameter) sensitivity statistics."""
    fields = ["storm", "gauge", "parameter", "n", "slope", "standardized",
              "corr", "r_squared", "p", "ci_lo", "ci_hi", "significant",
              "multivariate_std", "defined"]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for storm in results:
            for gauge in sorted(results[storm]):
                cell = results[storm][gauge]
                for key, _ in params:
                    s = cell["univariate"][key]
                    writer.writerow({
                        "storm": storm, "gauge": gauge, "parameter": key,
                        "n": s["n"], "slope": s["slope"],
                        "standardized": s["standardized"], "corr": s["corr"],
                        "r_squared": s["corr"]**2 if s["defined"] else float("nan"),
                        "p": s["p"], "ci_lo": s["ci"][0], "ci_hi": s["ci"][1],
                        "significant": (s["defined"] and s["p"] < es._ALPHA),
                        "multivariate_std": cell["multivariate"][key],
                        "defined": s["defined"],
                    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--output-path", type=Path,
        default=Path(os.environ["OUTPUT_PATH"]).expanduser() / "ETC_NASA_SLCT",
        help="Ensemble output root (default: $OUTPUT_PATH/ETC_NASA_SLCT).",
    )
    parser.add_argument(
        "--shared-runs-path", type=Path,
        default=Path(os.environ["SHARED_RUNS_PATH"]).expanduser(),
        help="Root for shared figures (default: $SHARED_RUNS_PATH).",
    )
    parser.add_argument(
        "--run-label",
        default=os.environ.get("HOSTNAME") or socket.gethostname(),
        help="Label subdirectory for output (default: $HOSTNAME or hostname).",
    )
    parser.add_argument(
        "--storms", nargs="+", default=None,
        help="Restrict to these storm dates (default: all found).",
    )
    parser.add_argument(
        "--window", nargs=2, type=float, metavar=("T0", "T1"), default=None,
        help="Peak-surge time window in days (default: per-storm from "
             "gauge_comparison.times).",
    )
    parser.add_argument("--n-boot", type=int, default=2000,
                        help="Bootstrap resamples for the slope CI.")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for the bootstrap (reproducibility).")
    args = parser.parse_args()

    run_root = args.output_path.expanduser().resolve()
    if not run_root.is_dir():
        raise SystemExit(f"Output root not found: {run_root}")

    window = tuple(args.window) if args.window is not None else None
    logger.info("Building peak-surge table from %s", run_root)
    table = es.build_table(run_root, storms=args.storms, window=window)
    if not table.rows:
        raise SystemExit("No ensemble output parsed; nothing to analyze.")
    logger.info("Table: %d rows across storms %s", len(table.rows), table.storms)

    results = es.compute_all(table, n_boot=args.n_boot, seed=args.seed)

    out_root = (args.shared_runs_path.expanduser().resolve()
                / "sensitivity" / args.run_label)
    out_root.mkdir(parents=True, exist_ok=True)
    write_table_csv(table, out_root / "peak_surge_table.csv")
    write_summary_csv(results, out_root / "sensitivity_summary.csv")
    es.plot_all(results, table, out_root)

    logger.info("Wrote CSVs + figures under %s", out_root)


if __name__ == "__main__":
    main()
