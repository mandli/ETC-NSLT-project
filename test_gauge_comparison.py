#!/usr/bin/env python

"""Verification checks for the presentation-mode helpers in gauge_comparison.py.

Synthetic inputs with known answers: the envelope/median stack, the day ->
UTC conversion that drives the date axis, the pre-storm baseline, and the
run discovery that decides which job directories feed the figure.  Runnable
with pytest or directly::

    pytest test_gauge_comparison.py
    python test_gauge_comparison.py
"""

from __future__ import annotations

import numpy as np

import gauge_comparison as gc
import plot_presentation as pp

LANDFALL = np.datetime64("2012-12-26T00:00")


def test_stack_runs_known_envelope():
    """Linear runs on staggered grids: interpolation is exact, stats known."""
    t1 = np.linspace(0.0, 4.0, 41)
    t2 = np.linspace(-0.5, 3.5, 17)
    t3 = np.linspace(0.2, 4.0, 77)
    runs = [(t, np.ma.masked_array(a * t + b))
            for t, a, b in ((t1, 1.0, 0.0), (t2, 2.0, -1.0), (t3, 0.5, 0.5))]
    t_common, stack = gc._stack_runs(runs, n_points=101)
    assert np.isclose(t_common[0], 0.2) and np.isclose(t_common[-1], 3.5)
    expected = np.array([1.0 * t_common, 2.0 * t_common - 1.0,
                         0.5 * t_common + 0.5])
    np.testing.assert_allclose(stack, expected, atol=1e-12)
    np.testing.assert_allclose(np.median(stack, axis=0),
                               np.median(expected, axis=0), atol=1e-12)


def test_stack_runs_drops_masked_and_degenerate():
    """Dry (masked) samples are skipped; an all-dry run is dropped."""
    t = np.linspace(0.0, 1.0, 11)
    s = np.ma.masked_array(2.0 * t, mask=False)
    s[5] = np.ma.masked
    s.data[5] = 1e6  # garbage under the mask must not leak through
    dry = np.ma.masked_array(np.zeros_like(t), mask=True)
    t_common, stack = gc._stack_runs([(t, s), (t, dry)], n_points=21)
    assert stack.shape == (1, 21)
    np.testing.assert_allclose(stack[0], 2.0 * t_common, atol=1e-12)


def test_stack_runs_empty():
    t_common, stack = gc._stack_runs([], n_points=5)
    assert t_common is None and stack.shape == (0, 5)


def test_days_to_utc():
    got = gc.days_to_utc(LANDFALL, [0.0, 1.5, 4.0, -1.0])
    want = np.array(["2012-12-26T00:00", "2012-12-27T12:00",
                     "2012-12-30T00:00", "2012-12-25T00:00"],
                    dtype="datetime64[ms]")
    np.testing.assert_array_equal(got, want)
    # Six-minute NOAA sampling survives the round trip exactly.
    six_min = gc.days_to_utc(LANDFALL, 0.1 / 24)
    assert six_min == np.datetime64("2012-12-26T00:06", "ms")


def test_event_date_label():
    assert gc.event_date_label(LANDFALL, [0, 4]) == "26–30 Dec 2012"
    assert (gc.event_date_label(np.datetime64("2018-11-14T08:00"), [0, 4])
            == "14–18 Nov 2018")
    assert (gc.event_date_label(LANDFALL, [0, 7])
            == "26 Dec 2012 – 2 Jan 2013")


def test_prestorm_baseline():
    """Mean over [-1, 0) only, NaNs ignored, NaN when the window is empty."""
    t = np.arange(-1.0, 2.0, 0.1 / 24)  # 6-minute samples
    resid = np.where(t < 0, 0.15, 1.0)
    resid[3] = np.nan
    assert np.isclose(gc.prestorm_baseline(t, resid), 0.15)
    assert np.isnan(gc.prestorm_baseline(t[t >= 0], resid[t >= 0]))


def test_fetch_noaa_hourly_from_cache(tmp_path):
    """Hourly fallback parses cached NOAA CSV (no network) incl. a gap."""
    import datetime
    begin = datetime.datetime(1992, 12, 7)
    end = datetime.datetime(1992, 12, 7, 2)
    dates = "199212070000_199212070200"
    for product, header, values in (
            ("hourly_height", "Date Time, Water Level, Sigma, I, L",
             ["1.5,0.000,0,0", ",,,", "2.5,0.000,0,0"]),
            ("predictions", "Date Time, Prediction", ["1.0", "1.25", "2.0"])):
        path = (tmp_path / f"{product}_hourly" / "8518750" / dates
                / "GMT_STND_metric")
        path.parent.mkdir(parents=True)
        rows = [f"1992-12-07 0{h}:00,{v}" for h, v in enumerate(values)]
        path.write_text("\n".join([header] + rows) + "\n")

    t, wl, pred = gc.fetch_noaa_hourly("8518750", begin, end,
                                       cache_dir=tmp_path)
    np.testing.assert_array_equal(
        t, np.array(["1992-12-07T00:00", "1992-12-07T01:00",
                     "1992-12-07T02:00"], dtype="datetime64[m]"))
    np.testing.assert_allclose(wl - pred, [0.5, np.nan, 0.5])


def test_discover_runs(tmp_path):
    """Only parseable dirs for the storm, at the sea level, with the gauge."""
    def make(name, gauge=True):
        d = tmp_path / name
        d.mkdir()
        if gauge:
            (d / "gauge00005.txt").write_text("")
    make("stormDEC2012_0pt25_sea0.0_scale0.80_lev5_dil1.00")
    make("stormDEC2012_1pt50_sea0.0_scale1.20_lev5_dil0.80_tlat+0.50_tlon+0.00")
    make("stormDEC2012_1pt00_sea0.2_scale1.00_lev5_dil1.00")  # wrong sea level
    make("stormNOV2018_1pt00_sea0.0_scale1.00_lev5_dil1.00")  # wrong storm
    make("stormDEC2012_1pt00_sea0.0_scale1.00_lev5_dil1.00", gauge=False)
    make("_pack_scripts")                                     # not a job dir

    runs = pp.discover_runs(tmp_path, "DEC2012", 5, sea_level=0.0)
    assert sorted(r for r, _ in runs) == [0.25, 1.5]
    runs_all = pp.discover_runs(tmp_path, "DEC2012", 5, sea_level=None)
    assert sorted(r for r, _ in runs_all) == [0.25, 1.0, 1.5]


if __name__ == "__main__":
    import tempfile
    from pathlib import Path
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            if "tmp_path" in fn.__code__.co_varnames[:fn.__code__.co_argcount]:
                with tempfile.TemporaryDirectory() as d:
                    fn(Path(d))
            else:
                fn()
            print(f"ok  {name}")
