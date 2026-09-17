"""Regressions found on the round-one review pass, pinned here.

Two of them were misread as data problems: a fixture yield map that lost
two thirds of its records to "null values" turned out to be the flow delay
shifting six thousand records, because the time-delta helper read pandas 3's
microsecond timestamps as nanoseconds; and the demo's "wrong units" alert
was the app correctly noticing that ten-tonne yields are not canola's.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agrosuite.clean import pipeline as clean_pipeline
from agrosuite.core import preflight
from agrosuite.core import schema as sch
from agrosuite.core.dataset import Dataset, DatasetMeta, apply_source_units
from agrosuite.demo import synthetic_harvest, synthetic_trial
from agrosuite.formats import registry

import fixtures as fx

M_PER_DEG_LAT = 111_320.0


def _track_from_a_file(n: int = 600, every_s: float = 2.0, step_m: float = 4.0) -> Dataset:
    """A straight track whose timestamps arrive as text, as they do from any
    CSV or DBF: pandas 3 parses those to microseconds, not nanoseconds."""
    stamps = pd.date_range("2025-09-08 13:30:00", periods=n, freq=f"{every_s:g}s")
    df = pd.DataFrame({
        sch.LON: -103.0,
        sch.LAT: 52.0 + np.arange(n) * step_m / M_PER_DEG_LAT,
        sch.TIMESTAMP: stamps.strftime("%Y-%m-%d %H:%M:%S"),
        sch.VALUE: 3_000.0 + np.arange(n) % 7,
        sch.SWATH: 9.0,
    })
    ds = Dataset(df, DatasetMeta(name="track", operation="harvest", crop="wheat"))
    # Pin the resolution so the test means the same under pandas 2 and 3.
    ds.df[sch.TIMESTAMP] = ds.df[sch.TIMESTAMP].astype("datetime64[us]")
    return ds


# --------------------------------------------------------- time deltas ---

def test_time_deltas_do_not_depend_on_the_timestamp_resolution():
    ds = _track_from_a_file(every_s=2.0)
    assert ds.df[sch.TIMESTAMP].dtype == "datetime64[us]"
    dt = ds._time_delta_seconds()
    assert dt is not None
    assert float(np.nanmedian(dt)) == pytest.approx(2.0)
    assert dt[0] == dt[1], "the first record borrows the interval of the second"


def test_time_deltas_survive_a_timezone():
    ds = _track_from_a_file()
    ds.df[sch.TIMESTAMP] = ds.df[sch.TIMESTAMP].dt.tz_localize("America/Regina")
    assert float(np.nanmedian(ds._time_delta_seconds())) == pytest.approx(2.0)


def test_flow_delay_shifts_by_records_not_by_ticks():
    """12 s at one record every 2 s is six records — not six thousand."""
    ds = _track_from_a_file(every_s=2.0)
    message = clean_pipeline.apply_flow_delay(ds, 12.0)
    assert message is not None and "6 records at 2.00 s" in message
    assert int(ds.df[sch.VALUE].isna().sum()) == 6


def test_speed_reconstructed_from_file_timestamps_is_plausible():
    """4 m every 2 s is 7.2 km/h; read as nanoseconds it came out at 7 200."""
    ds = _track_from_a_file(every_s=2.0, step_m=4.0)
    ds.ensure_derived()
    assert float(ds.df[sch.SPEED].median()) == pytest.approx(7.2, rel=0.05)


def test_fixture_yield_map_is_not_emptied_by_the_flow_delay(tmp_path):
    """The reviewer's repro: the trial fixture's yield shapefile, converted
    the way the first-look button does it, cleaned with the harvest preset."""
    paths = fx.difm_project(tmp_path / "project")
    ds = registry.read_any(paths["yield"])
    apply_source_units(ds, {sch.VALUE: "bu/ac", sch.SPEED: "mph", sch.SWATH: "ft"}, "canola")
    ds.ensure_derived()

    result = clean_pipeline.run(ds, clean_pipeline.PRESETS["harvest"])

    assert any("6 records at 2.00 s" in note for note in result.report["corrections"])
    nulls = next(s for s in result.report["steps"] if s["key"] == "null_value")
    assert nulls["removed"] <= 6, "only the records the shift has no reading for"
    assert result.report["totals"]["removed_pct"] < 40


# ---------------------------------------------------------------- demos ---

@pytest.mark.parametrize("make", [synthetic_harvest, synthetic_trial])
def test_the_demo_passes_its_own_first_look(make):
    """Sample data the app's own check calls wrong is a bad first contact."""
    ds = make()
    ds.ensure_derived()
    report = preflight.run(ds)
    titles = {f["title"] for f in report["findings"] if f["level"] == "alert"}
    assert "Units look wrong" not in titles
    assert report["proposed_units"] is None
    assert report["verdict"] != "alert"
