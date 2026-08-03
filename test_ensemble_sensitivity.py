#!/usr/bin/env python

"""Verification suite for ensemble_sensitivity.py.

These are *verification* checks, not example runs: a manufactured solution with
known regression slopes, algebraic identities the estimator must satisfy, the
orthogonality property of the balanced factorial, and a round-trip against the
real ``ETCJob.prefix`` naming.  Runnable with pytest or directly::

    pytest test_ensemble_sensitivity.py
    python test_ensemble_sensitivity.py
"""

from __future__ import annotations

import itertools

import numpy as np

import ensemble_sensitivity as es

# Design used by run_tests.py (balanced full factorial).
SCALINGS = [0.8, 1.0, 1.2]
DILATIONS = [0.8, 1.0, 1.2]
SEA_LEVELS = [0.0, 0.2]
RESOLUTIONS = [0.25, 1.0, 1.5]

# Manufactured linear response: peak = A*scale + B*dil + C*sea + D*res (+noise).
A, B, C, D = 2.0, -0.5, 1.5, -0.3


def _make_table(noise_std=0.0, seed=0, resolutions=RESOLUTIONS, storm="TEST"):
    rng = np.random.default_rng(seed)
    rows = []
    for scale, dil, sea, res in itertools.product(
            SCALINGS, DILATIONS, SEA_LEVELS, resolutions):
        peak = A * scale + B * dil + C * sea + D * res
        if noise_std:
            peak += rng.normal(0.0, noise_std)
        rows.append({
            "storm": storm, "gauge": 1, "res": res, "sea_level": sea,
            "scaling": scale, "dilation": dil, "levels": 5.0,
            "track_lat": 0.0, "track_lon": 0.0, "peak_surge": peak,
        })
    t = es.Table(rows=rows, storms=[storm])
    return t


def test_manufactured_slopes_exact():
    """cov/var recovers the exact known slope for each parameter (no noise)."""
    t = _make_table()
    res = es.compute_all(t, n_boot=200)["TEST"][1]["univariate"]
    truth = {"scaling": A, "dilation": B, "sea_level": C, "res": D}
    for k, expected in truth.items():
        got = res[k]["slope"]
        assert np.isclose(got, expected, atol=1e-9), \
            f"{k}: slope {got} != {expected}"
        assert np.sign(got) == np.sign(expected), f"{k}: wrong sign"


def test_multivariate_matches_univariate_on_balanced_design():
    """Orthogonality: standardized univariate == multivariate coefficient."""
    t = _make_table()
    cell = es.compute_all(t, n_boot=200)["TEST"][1]
    for k, _ in es.PARAMS:
        uni = cell["univariate"][k]["standardized"]
        multi = cell["multivariate"][k]
        assert np.isclose(uni, multi, atol=1e-9), \
            f"{k}: univariate {uni} != multivariate {multi}"


def test_identities():
    """slope==polyfit slope; standardized==slope*sigma_x; corr==slope*sx/sJ."""
    t = _make_table(noise_std=0.05, seed=1)
    rows = t.select("TEST", 1)
    J = t.column("peak_surge", rows)
    for k, _ in es.PARAMS:
        x = t.column(k, rows)
        s = es.sensitivity(J, x, n_boot=100)
        poly = np.polyfit(x, J, 1)[0]
        assert np.isclose(s["slope"], poly, atol=1e-9)
        assert np.isclose(s["standardized"], s["slope"] * np.std(x), atol=1e-12)
        assert np.isclose(s["corr"],
                          s["slope"] * np.std(x) / np.std(J), atol=1e-9)


def test_constant_parameter_is_masked():
    """A single-resolution storm has undefined (masked) resolution sensitivity."""
    t = _make_table(resolutions=[0.25])  # mimic DEC1992
    s = es.compute_all(t, n_boot=50)["TEST"][1]["univariate"]["res"]
    assert s["defined"] is False
    assert np.isnan(s["slope"]) and np.isnan(s["standardized"])


def test_bootstrap_ci_contains_truth_under_noise():
    """With noise the 95% bootstrap CI on each slope brackets the true slope."""
    t = _make_table(noise_std=0.05, seed=7)
    res = es.compute_all(t, n_boot=4000, seed=7)["TEST"][1]["univariate"]
    truth = {"scaling": A, "dilation": B, "sea_level": C, "res": D}
    for k, expected in truth.items():
        lo, hi = res[k]["ci"]
        assert lo <= expected <= hi, f"{k}: {expected} not in CI ({lo}, {hi})"


def test_prefix_roundtrip_against_etcjob():
    """parse_prefix inverts the real ETCJob.prefix for a spread of parameters.

    Locks the naming coupling: if ETCJob.prefix changes format, this fails
    rather than silently dropping ensemble members from the analysis.
    """
    try:
        from pathlib import Path

        import run_tests
    except Exception as exc:  # heavy clawpack/batch import unavailable
        import pytest
        pytest.skip(f"run_tests import unavailable: {exc}")

    cases = [
        ("DEC2012_1pt00.nc", 0.2, 1.2, 5, 0.8),
        ("NOV2018_0pt25.nc", 0.0, 0.8, 5, 1.0),
        ("DEC1992_1pt50.nc", 0.2, 1.0, 5, 1.2),
    ]
    for stem, sea, scale, lev, dil in cases:
        job = run_tests.ETCJob(Path(stem), sea_level=sea, scaling=scale,
                               levels=lev, time_dilation=dil)
        p = es.parse_prefix(job.prefix)
        assert p is not None, f"failed to parse {job.prefix!r}"
        storm, res_token = stem[:-3].split("_")
        assert p["storm"] == storm
        assert p["res"] == es.res_to_degrees(res_token)
        assert np.isclose(p["sea_level"], sea)
        assert np.isclose(p["scaling"], scale)
        assert p["levels"] == lev
        assert np.isclose(p["dilation"], dil)


def test_parse_prefix_rejects_non_jobs():
    assert es.parse_prefix("_pack_scripts") is None
    assert es.parse_prefix("timing.txt") is None


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as exc:  # noqa: BLE001 - report and continue
            failed += 1
            print(f"FAIL  {fn.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
