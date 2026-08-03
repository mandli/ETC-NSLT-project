#!/usr/bin/env python

"""Verification suite for ensemble_sensitivity.py.

These are *verification* checks, not example runs: a manufactured solution with
known regression slopes, algebraic identities the estimator must satisfy, the
orthogonality of the balanced factorial (including the cardinal-star track
design), a round-trip against the real ``ETCJob.prefix`` naming, and a check
that a requested track shift reaches the written ``.storm`` descriptor.  Runnable
with pytest or directly::

    pytest test_ensemble_sensitivity.py
    python test_ensemble_sensitivity.py
"""

from __future__ import annotations

import itertools

import numpy as np

import ensemble_sensitivity as es

# Design used by run_tests.py: wind x dilation x resolution x sea level, crossed
# with the 5-point cardinal-star track shift.  Sea level is swept here (not in
# the production sweep) only so the superposition check has two values to pair.
SCALINGS = [0.8, 1.0, 1.2]
DILATIONS = [0.8, 1.0, 1.2]
SEA_LEVELS = [0.0, 0.2]
RESOLUTIONS = [0.25, 1.0, 1.5]
TRACK_STAR = [(0.0, 0.0), (0.5, 0.0), (-0.5, 0.0), (0.0, 0.5), (0.0, -0.5)]

# Manufactured linear response:
#   peak = A*scale + B*dil + C*sea + D*res + E*track_lon + F*track_lat (+noise)
A, B, C, D, E, F = 2.0, -0.5, 1.5, -0.3, 0.7, -0.9

# Truth for the ESA parameters (matches es.PARAMS; sea level is excluded there).
TRUTH = {"scaling": A, "dilation": B, "res": D, "track_lon": E, "track_lat": F}


def _make_table(noise_std=0.0, seed=0, resolutions=RESOLUTIONS, storm="TEST"):
    rng = np.random.default_rng(seed)
    rows = []
    for scale, dil, sea, res, (tlon, tlat) in itertools.product(
            SCALINGS, DILATIONS, SEA_LEVELS, resolutions, TRACK_STAR):
        peak = (A * scale + B * dil + C * sea + D * res
                + E * tlon + F * tlat)
        if noise_std:
            peak += rng.normal(0.0, noise_std)
        rows.append({
            "storm": storm, "gauge": 1, "res": res, "sea_level": sea,
            "scaling": scale, "dilation": dil, "levels": 5.0,
            "track_lat": tlat, "track_lon": tlon, "peak_surge": peak,
        })
    return es.Table(rows=rows, storms=[storm])


def test_manufactured_slopes_exact():
    """cov/var recovers the exact known slope for each parameter (no noise)."""
    res = es.compute_all(_make_table(), n_boot=200)["TEST"][1]["univariate"]
    for k, expected in TRUTH.items():
        got = res[k]["slope"]
        assert np.isclose(got, expected, atol=1e-9), \
            f"{k}: slope {got} != {expected}"
        assert np.sign(got) == np.sign(expected), f"{k}: wrong sign"


def test_multivariate_matches_univariate_on_balanced_design():
    """Orthogonality: standardized univariate == multivariate coefficient."""
    cell = es.compute_all(_make_table(), n_boot=200)["TEST"][1]
    for k, _ in es.PARAMS:
        uni = cell["univariate"][k]["standardized"]
        multi = cell["multivariate"][k]
        assert np.isclose(uni, multi, atol=1e-9), \
            f"{k}: univariate {uni} != multivariate {multi}"


def test_track_axes_orthogonal_and_recovered():
    """The cardinal star keeps track_lon ⟂ track_lat and recovers both slopes."""
    t = _make_table()
    rows = t.select("TEST", 1)
    tlon, tlat = t.column("track_lon", rows), t.column("track_lat", rows)
    cov = np.mean((tlon - tlon.mean()) * (tlat - tlat.mean()))
    assert abs(cov) < 1e-12, f"track axes not orthogonal: cov={cov}"
    res = es.compute_all(t, n_boot=100)["TEST"][1]["univariate"]
    assert np.isclose(res["track_lon"]["slope"], E, atol=1e-9)
    assert np.isclose(res["track_lat"]["slope"], F, atol=1e-9)


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
    res = es.compute_all(_make_table(noise_std=0.05, seed=7),
                         n_boot=4000, seed=7)["TEST"][1]["univariate"]
    for k, expected in TRUTH.items():
        lo, hi = res[k]["ci"]
        assert lo <= expected <= hi, f"{k}: {expected} not in CI ({lo}, {hi})"


def test_superposition_exact_offset_when_additive():
    """A purely additive model gives constant delta == C*(hi-lo), zero interaction."""
    chk = es.superposition_check(_make_table(), param="sea_level")
    lo, hi = chk["_meta"]["lo"], chk["_meta"]["hi"]
    cell = chk["TEST"][1]
    assert np.isclose(cell["mean"], C * (hi - lo), atol=1e-9)
    assert np.isclose(cell["gain_mean"], C, atol=1e-9)
    assert np.isclose(cell["std"], 0.0, atol=1e-9)  # no interaction -> no spread


def test_superposition_detects_interaction():
    """A sea_level*scaling cross term shows up as nonzero delta spread."""
    t = _make_table()
    for r in t.rows:  # inject interaction: extra 0.5*sea_level*scaling
        r["peak_surge"] += 0.5 * r["sea_level"] * r["scaling"]
    cell = es.superposition_check(t, param="sea_level")["TEST"][1]
    assert cell["std"] > 1e-3  # separability broken -> detectable spread


def test_prefix_roundtrip_against_etcjob():
    """parse_prefix inverts the real ETCJob.prefix, including the track shift.

    Locks the naming coupling: if ETCJob.prefix changes format, this fails
    rather than silently dropping ensemble members from the analysis.
    """
    try:
        from pathlib import Path

        import run_tests
    except Exception as exc:  # heavy clawpack/batch import unavailable
        import pytest
        pytest.skip(f"run_tests import unavailable: {exc}")

    # (stem, sea, scale, lev, dil, track_lon, track_lat)
    cases = [
        ("DEC2012_1pt00.nc", 0.0, 1.2, 5, 0.8, 0.5, 0.0),
        ("NOV2018_0pt25.nc", 0.0, 0.8, 5, 1.0, 0.0, -0.5),
        ("DEC1992_1pt50.nc", 0.0, 1.0, 5, 1.2, 0.0, 0.0),
    ]
    for stem, sea, scale, lev, dil, tlon, tlat in cases:
        job = run_tests.ETCJob(Path(stem), sea_level=sea, scaling=scale,
                               levels=lev, time_dilation=dil,
                               track_shift_lon=tlon, track_shift_lat=tlat)
        p = es.parse_prefix(job.prefix)
        assert p is not None, f"failed to parse {job.prefix!r}"
        storm, res_token = stem[:-3].split("_")
        assert p["storm"] == storm
        assert p["res"] == es.res_to_degrees(res_token)
        assert np.isclose(p["sea_level"], sea)
        assert np.isclose(p["scaling"], scale)
        assert p["levels"] == lev
        assert np.isclose(p["dilation"], dil)
        assert np.isclose(p["track_lon"], tlon)
        assert np.isclose(p["track_lat"], tlat)


def test_old_prefix_without_track_defaults_to_zero():
    """A pre-track job-dir name still parses, with track shift defaulting to 0."""
    p = es.parse_prefix("stormDEC2012_1pt00_sea0.2_scale1.00_lev5_dil1.00")
    assert p is not None
    assert p["track_lon"] == 0.0 and p["track_lat"] == 0.0


def test_track_shift_reaches_storm_descriptor():
    """A requested (x,y) shift is written verbatim into the .storm descriptor.

    Exercises the full ETCJob.write_data_objects path so the ETCJob -> Storm
    x_shift/y_shift wiring is locked, not just asserted in isolation.
    """
    import os
    from pathlib import Path
    import shutil
    import tempfile
    try:
        import run_tests
    except Exception as exc:
        import pytest
        pytest.skip(f"run_tests import unavailable: {exc}")

    nc = (Path(os.environ["DATA_PATH"]) / "storms" / "ETC_NASA_SLCT"
          / "DEC2012_1pt50.nc")
    if not nc.exists():
        import pytest
        pytest.skip(f"storm file not available: {nc}")

    job = run_tests.ETCJob(nc, scaling=1.0, levels=5, time_dilation=1.0,
                           track_shift_lon=0.5, track_shift_lat=-0.5)
    tmp = Path(tempfile.mkdtemp())
    try:
        job.write_data_objects(tmp)
        txt = (tmp / f"{job.prefix}.storm").read_text()
        xs = ys = None
        for line in txt.splitlines():
            if "x_shift" in line:
                xs = float(line.split()[0])
            elif "y_shift" in line:
                ys = float(line.split()[0])
        assert xs == 0.5, f"x_shift in descriptor = {xs}"
        assert ys == -0.5, f"y_shift in descriptor = {ys}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


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
            if type(exc).__name__ == "Skipped":  # pytest.skip outside pytest
                print(f"SKIP  {fn.__name__}: {exc}")
                continue
            failed += 1
            print(f"FAIL  {fn.__name__}: {exc}")
    print(f"\n{len([t for t in tests]) - failed} of {len(tests)} did not fail")
    raise SystemExit(1 if failed else 0)
