"""Reading SoMat eDAQ logs (.sie).

The file is made up by :mod:`agrosuite.machine.synthetic`, which writes it
with every habit of the real logger — the XML over many blocks with its root
never closed, a distance counter that wraps, the CAN channels at their
"missing" values with the engine off, a clock 23 s behind the GPS, a second
data mode — and knows the true totals, so every number read can be checked.
"""

from __future__ import annotations

import struct

import numpy as np
import pandas as pd
import pytest

from agrosuite.core import schema as sch
from agrosuite.formats import registry
from agrosuite.formats import somat
from agrosuite.machine import synthetic as syn


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    folder = tmp_path_factory.mktemp("somat")
    path, day = syn.demo_log(folder / "sprayer_day.sie")
    return path, day


@pytest.fixture(scope="module")
def read(demo):
    path, day = demo
    return registry.read_any(path), day


def test_the_registry_recognises_a_sie_and_reads_it_as_telemetry(read, demo):
    dataset, _ = read
    assert ".sie" in registry.ALL_IMPORT_EXT
    assert registry.detect(demo[0]).kind == "somat"
    assert dataset.meta.operation == "telemetry"
    assert dataset.meta.brand == "somat"
    assert dataset.meta.value_label == "Fuel rate" and dataset.meta.value_unit == "L/h"


def test_every_sample_becomes_a_row_with_the_canonical_columns(read):
    dataset, day = read
    assert len(dataset) == day.truth["samples"]
    for column in (sch.LON, sch.LAT, sch.TIMESTAMP, sch.SPEED, sch.HEADING, sch.ELEVATION,
                   sch.DISTANCE, "fuel_rate_lh", "fuel_l", "def_level_pct", "engine_rpm",
                   "engine_load_pct", "coolant_c", "battery_v", "engine_hours", "satellites"):
        assert column in dataset.df, column
    # The GPS clock is read to set the logger's clock, then dropped.
    assert not any(c.startswith("utc_") for c in dataset.df.columns)
    assert "RunTrigger" not in dataset.df and "ch_runtrigger" not in dataset.df


def test_units_are_converted_to_metric(read):
    dataset, day = read
    moving = day.frame["speed_kmh"] > 0
    speed = dataset.df.loc[moving.to_numpy(), sch.SPEED]
    assert speed.median() == pytest.approx(day.frame.loc[moving, "speed_kmh"].median(), rel=1e-6)
    # DeltaD is written in feet; the column is metres per sample.
    assert dataset.df[sch.DISTANCE].sum() == pytest.approx(day.truth["distance_m"], rel=1e-9)


def test_missing_readings_become_nan_from_the_values_the_channels_declare(read):
    dataset, day = read
    off = (day.frame["activity"] == "off").to_numpy()
    for column in ("engine_rpm", "fuel_rate_lh", "engine_load_pct", "def_level_pct",
                   "coolant_c", "battery_v", "engine_hours", "fuel_l"):
        assert dataset.df.loc[off, column].isna().all(), column
        assert dataset.df.loc[~off, column].notna().all(), column
    invalid = dataset.meta.extra["somat"]["invalid"]
    assert invalid["EngCoolantTemp"] == off.sum()        # declared as 1000
    assert invalid["ElectricalPotential"] == off.sum()   # declared as -500
    assert invalid["DeltaF"] == off.sum()                # outside its declared range
    assert any("marked as missing" in note for note in dataset.meta.notes)


def test_a_wrapped_distance_counter_is_stitched_back(read):
    dataset, day = read
    facts = dataset.meta.extra["somat"]
    assert facts["distance_counter_m_wraps"] == 3
    counter = dataset.df["distance_counter_m"].dropna().to_numpy()
    assert (np.diff(counter) >= -1e-6).all(), "the stitched counter must never go back"
    assert facts["distance_from_counter"] == pytest.approx(day.truth["distance_m"], rel=1e-4)
    assert any("went back to zero 3 times" in note for note in dataset.meta.notes)


def test_fuel_from_the_three_sources_agrees(read):
    dataset, day = read
    facts = dataset.meta.extra["somat"]
    assert facts["fuel_from_increments"] == pytest.approx(day.truth["fuel_l"], rel=1e-9)
    assert facts["fuel_from_counter"] == pytest.approx(day.truth["fuel_l"], rel=1e-6)
    assert facts["fuel_from_rate"] == pytest.approx(day.truth["fuel_l"], rel=1e-6)
    assert not any("disagree" in note for note in dataset.meta.notes)


def test_the_logger_clock_is_set_to_gps_time(read):
    dataset, _ = read
    facts = dataset.meta.extra["somat"]
    assert facts["clock_drift_s"] == pytest.approx(syn.CLOCK_LAG_S, abs=0.01)
    assert facts["utc_offset_h"] == syn.UTC_OFFSET_H
    first = pd.Timestamp(dataset.df[sch.TIMESTAMP].iloc[0])
    assert first == syn.DEMO_START + pd.Timedelta(seconds=syn.CLOCK_LAG_S)
    assert any("23 s behind the GPS" in note and "UTC-6" in note for note in dataset.meta.notes)


def test_the_second_data_mode_is_named_and_left_out(read):
    dataset, _ = read
    assert dataset.meta.extra["somat"]["datamode"] == "__dm"
    assert any("'Omni_2022', which repeats" in note for note in dataset.meta.notes)


def test_positions_are_projected_for_the_map(read):
    dataset, _ = read
    assert dataset.metric_crs == "EPSG:32612"
    assert dataset.df[sch.X].notna().all()


def test_the_logger_is_named_in_the_first_note(read):
    dataset, day = read
    assert dataset.meta.notes[0] == (
        f"SoMat log from edaqxr_lite 560999: {day.truth['samples']:,} samples at 2 Hz.")


# ==========================================================================
# Files that are not what they should be
# ==========================================================================

def _channels(n=40):
    t = np.arange(n, dtype=float)
    return [syn.Channel("EngFuelRate", 20 + t, "L/hr", -1, 0, 3000),
            syn.Channel("latitude", 51.8 + t * 1e-6, "degrees", 91, -90, 90),
            syn.Channel("longitude", -114.0 + t * 1e-6, "degrees", 181, -180, 180)]


def test_an_unfinished_last_block_is_reported_and_the_rest_read(tmp_path):
    path = syn.write_sie(tmp_path / "cut.sie", _channels(), "2026-09-02T10:00:00", truncate_bytes=64)
    dataset = registry.read_any(path)
    assert len(dataset) == 40
    assert dataset.meta.extra["somat"]["truncated_bytes"] == 64
    assert any("unfinished block" in note for note in dataset.meta.notes)


def test_a_broken_block_in_the_middle_is_refused(tmp_path):
    path = syn.write_sie(tmp_path / "broken.sie", _channels(), "2026-09-02T10:00:00")
    data = bytearray(path.read_bytes())
    # Break the sync word of the third block.
    pos = 0
    for _ in range(2):
        pos += struct.unpack_from(">I", data, pos)[0]
    data[pos + 8:pos + 12] = b"\0\0\0\0"
    path.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="damaged at byte"):
        registry.read_any(path)


def test_a_file_that_only_has_the_extension_is_refused(tmp_path):
    path = tmp_path / "not_a_log.sie"
    path.write_bytes(b"hello, this is not a SoMat log at all" * 10)
    with pytest.raises(ValueError, match="does not start"):
        registry.read_any(path)


def test_a_channel_this_reader_does_not_know_is_kept(tmp_path):
    channels = _channels() + [syn.Channel("BoomPressure", np.full(40, 3.1), "bar", -1, 0, 20)]
    dataset = registry.read_any(syn.write_sie(tmp_path / "extra.sie", channels, "2026-09-02T10:00:00"))
    assert dataset.df["ch_boompressure"].tolist() == [3.1] * 40


def test_a_channel_at_another_rate_is_matched_by_time(tmp_path):
    channels = _channels(40) + [syn.Channel("EngSpeed", np.arange(20, dtype=float) * 10 + 1000,
                                            "RPM", -1, 0, 8000, rate_hz=1.0)]
    dataset = registry.read_any(syn.write_sie(tmp_path / "rates.sie", channels, "2026-09-02T10:00:00"))
    rpm = dataset.df["engine_rpm"].to_numpy()
    assert len(rpm) == 40
    # Two 2 Hz samples per 1 Hz reading, never averaged.
    assert rpm[0] == rpm[1] == 1000 and rpm[2] == rpm[3] == 1010


def test_a_log_without_gps_time_keeps_the_logger_clock_and_says_so(tmp_path):
    dataset = registry.read_any(syn.write_sie(tmp_path / "noutc.sie", _channels(), "2026-09-02T10:00:00"))
    assert pd.Timestamp(dataset.df[sch.TIMESTAMP].iloc[0]) == pd.Timestamp("2026-09-02T10:00:00")
    assert any("could not be checked" in note for note in dataset.meta.notes)


def test_is_sie_reads_only_the_first_block_header(tmp_path, demo):
    assert somat.is_sie(demo[0])
    (tmp_path / "empty.sie").write_bytes(b"")
    assert not somat.is_sie(tmp_path / "empty.sie")
    assert not somat.is_sie(tmp_path / "missing.sie")
