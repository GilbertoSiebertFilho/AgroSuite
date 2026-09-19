"""A machine's day from its telemetry: activities, fuel, DEF, and the crop
under the tyres.

The day is the made-up one in :mod:`agrosuite.machine.synthetic`, whose true
time, distance and fuel per activity are known; the geometry tests use paths
whose crushed area can be worked out by hand.
"""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Polygon

from agrosuite import machine
from agrosuite.core.units import UNIT_PRESETS, Phrase
from agrosuite.formats import registry
from agrosuite.machine import analysis as ma
from agrosuite.machine import synthetic as syn
from agrosuite.machine import trampling as tr

TYRE = {"tyre_size": "380/90R46", "track_width_m": 3.048}


class _Boundary:
    """The shape of a loaded boundary file, for :func:`machine.boundary_polygon`."""

    def __init__(self, ring):
        self.geometry = [Polygon(ring)]
        self.meta = type("Meta", (), {"name": "boundary"})()


@pytest.fixture(scope="module")
def day(tmp_path_factory):
    path, truth_day = syn.demo_log(tmp_path_factory.mktemp("machine") / "day.sie")
    dataset = registry.read_any(path)
    _, metric, area = machine.boundary_polygon(_Boundary(truth_day.boundary_lonlat()), dataset.metric_crs)
    return dataset, truth_day, metric, area


@pytest.fixture(scope="module")
def result(day):
    dataset, truth_day, metric, area = day
    settings = {"boom_width_m": truth_day.truth["boom_m"], "def_tank_l": 60, "fuel_price": 1.6,
                "yield_kg_ha": 4000, "crop_price_per_kg": 0.3, **TYRE}
    return machine.analyse(dataset, settings, boundary=metric, boundary_area_ha=area)


# ==========================================================================
# Activities
# ==========================================================================

def test_time_distance_and_fuel_split_as_the_day_happened(result, day):
    truth = day[1].truth
    rows = {r["key"]: r for r in result.summary["activities"]}
    assert set(rows) == {"field", "idle", "road", "off"}
    for key in rows:
        assert rows[key]["time_s"] == pytest.approx(truth[key]["time_s"], abs=1.0), key
        assert rows[key]["distance_m"] == pytest.approx(truth[key]["distance_m"], rel=1e-3, abs=1.0), key
        assert rows[key]["fuel_l"] == pytest.approx(truth[key]["fuel_l"], rel=1e-3, abs=1e-3), key
    assert result.summary["basis"] == "boundary"


def test_the_split_adds_up(result):
    s = result.summary
    rows = s["activities"]
    assert sum(r["time_s"] for r in rows) == pytest.approx(s["span"]["duration_s"])
    assert sum(r["fuel_l"] for r in rows) == pytest.approx(s["fuel"]["total_l"])
    assert sum(r["share_fuel"] for r in rows if r["share_fuel"]) == pytest.approx(1.0)
    engine = [r["share_engine_time"] for r in rows if r["key"] != "off"]
    assert sum(engine) == pytest.approx(1.0)


def test_a_short_pause_in_the_field_stays_field_work():
    settings = ma.MachineSettings()
    activity = np.array(["field"] * 50 + ["idle"] * 6 + ["field"] * 50 + ["idle"] * 60 + ["road"] * 10,
                        dtype=object)
    out = ma._absorb_short_stops(activity, samples=20)
    assert (out[50:56] == "field").all(), "a 3 s pause is part of the pass"
    assert (out[106:166] == "idle").all(), "a 30 s stop is a stop"
    assert settings.min_stop_s == 10.0


def test_without_a_boundary_the_road_is_what_is_faster(day):
    dataset, truth_day, _, _ = day
    result = machine.analyse(dataset, {"road_speed_kmh": 25})
    rows = {r["key"]: r for r in result.summary["activities"]}
    assert result.summary["basis"] == "speed"
    # The demo drives the road at 45 km/h and sprays at 14: speed alone
    # splits it the same way here.
    assert rows["road"]["distance_m"] == pytest.approx(truth_day.truth["road"]["distance_m"], rel=0.01)
    texts = [f["text"] for f in result.summary["findings"]]
    assert any("No field boundary" in t for t in texts)


def test_field_runs_split_at_gaps_in_the_gps():
    import pandas as pd

    df = pd.DataFrame({"x": [0, 1, 2, np.nan, 4, 5, 200, 201], "y": [0] * 8})
    runs = ma.field_runs(df, np.array(["field"] * 8, dtype=object))
    assert [len(r) for r in runs] == [3, 2, 2]


# ==========================================================================
# Fuel, DEF, engine
# ==========================================================================

def test_fuel_per_hectare_worked_and_its_cost(result, day):
    truth = day[1].truth
    fuel = result.summary["fuel"]
    covered = result.summary["area"]["covered_ha"]
    assert covered == pytest.approx(truth["field_path_m"] * truth["boom_m"] / 1e4, rel=0.15)
    assert fuel["per_ha_field"] == pytest.approx(truth["field"]["fuel_l"] / covered)
    assert fuel["cost"] == pytest.approx(truth["fuel_l"] * 1.6)
    assert fuel["road_l_per_km"] == pytest.approx(
        truth["road"]["fuel_l"] / (truth["road"]["distance_m"] / 1000), rel=1e-3)


def test_def_follows_the_level_through_slosh(result, day):
    truth = day[1].truth
    d = result.summary["def"]
    assert d["available"] and d["refills"] == 0
    assert d["drop_pct"] == pytest.approx(truth["def_used_pct"], abs=0.5)
    assert d["litres"] == pytest.approx(truth["def_used_l"], abs=0.3)
    assert d["share_of_fuel"] == pytest.approx(0.03, abs=0.02)
    assert d["step_pct"] == pytest.approx(0.4, abs=1e-6)


def test_a_def_refill_is_not_counted_as_negative_use():
    """8 points over two hours, a refill to 95 %, then 5 over an hour and a
    half — the pace a DEF tank really falls at, a point or two an hour."""
    import pandas as pd

    level = pd.Series(np.concatenate([np.linspace(80, 72, 14_400), np.linspace(95, 90, 10_800)]))
    out = ma.def_usage(level, np.ones(len(level), dtype=bool), dt=0.5, tank_l=50)
    assert out["refills"] == 1
    assert out["drop_pct"] == pytest.approx(13.0, abs=0.1)
    assert out["litres"] == pytest.approx(6.5, abs=0.05)


def test_engine_and_speed_are_summarised(result):
    s = result.summary
    assert s["engine"]["rpm"]["max"] == 2100
    assert s["engine"]["load"]["median"] == pytest.approx(55, abs=10)
    assert s["speed"]["field"]["median"] == pytest.approx(14.0, abs=0.01)
    assert s["speed"]["road"]["max"] == pytest.approx(45.0, abs=0.01)
    assert s["gps"]["satellites_median"] == 10


def test_the_statistics_table_is_infield_shaped(result):
    rows = result.statistics
    assert {r["selection"] for r in rows} == {"all", "field", "idle", "road", "off"}
    fuel = next(r for r in rows if r["selection"] == "field" and r["channel"] == "fuel_rate_lh")
    assert fuel["mean"] == pytest.approx(28.0) and fuel["std"] == pytest.approx(0.0, abs=1e-9)
    for key in ("n", "duration_s", "min", "t_min", "max", "t_max", "peak_to_peak", "mean",
                "median", "rms", "std", "variance", "skewness", "kurtosis", "crest_factor"):
        assert key in fuel


def test_channel_stats_match_the_textbook():
    x = np.array([1.0, 2.0, 3.0, 4.0, 10.0])
    t = np.array(["2026-09-02T10:00:00"] * 5, dtype="datetime64[ns]")
    s = ma.channel_stats(x, t, dt=0.5)
    assert s["n"] == 5 and s["duration_s"] == 2.5
    assert s["mean"] == 4.0 and s["median"] == 3.0
    assert s["std"] == pytest.approx(np.std(x, ddof=1))
    assert s["rms"] == pytest.approx(np.sqrt(np.mean(x ** 2)))
    assert s["crest_factor"] == pytest.approx(10.0 / np.sqrt(np.mean(x ** 2)))
    m2 = np.mean((x - 4) ** 2)
    assert s["kurtosis"] == pytest.approx(np.mean((x - 4) ** 4) / m2 ** 2)


# ==========================================================================
# Trampling
# ==========================================================================

@pytest.mark.parametrize("text, mm", [
    ("380/90R46", 380), ("VF 380/90 R46 173D", 380), ("320/85-38", 320), ("IF 480/80R50", 480),
    ("18.4R38", 467.36), ("13.6-38", 345.44), ("15.5/80-24", 393.7),
    ("380 mm", 380), ("15 in", 381), ("0.38 m", 380),
])
def test_a_tyre_size_reads_as_its_section_width(text, mm):
    assert tr.parse_tyre(text)["width_m"] * 1000 == pytest.approx(mm, abs=0.01)


@pytest.mark.parametrize("text", ["", "banana", "5000 mm", "2 mm"])
def test_what_is_not_a_tyre_is_refused_in_words(text):
    with pytest.raises(ValueError):
        tr.parse_tyre(text)


SETUP = tr.TyreSetup(0.38, 3.048, True)
LINE = np.column_stack([np.linspace(0, 1000, 2001), np.zeros(2001)])


def test_a_straight_kilometre_crushes_two_tyre_widths():
    assert tr.footprint([LINE], SETUP).area == pytest.approx(760.0, abs=0.5)


def test_rear_wheels_off_the_front_tracks_crush_twice_as_much():
    assert tr.footprint([LINE], tr.TyreSetup(0.38, 3.048, False)).area == pytest.approx(1520.0, abs=0.5)


def test_a_track_driven_again_is_counted_once():
    assert tr.footprint([LINE, LINE.copy()], SETUP).area == pytest.approx(760.0, abs=0.5)


def test_one_pass_does_not_erase_the_tracks_of_another():
    """The gap between one pass's wheels must not take away ground that a
    neighbouring pass crushed — the reason the strips are drawn per piece."""
    shifted = LINE + [0, 0.05]
    assert tr.footprint([LINE, shifted], SETUP).area == pytest.approx(860.0, abs=0.5)


def test_a_turn_of_the_boom_s_radius_overlaps_nothing():
    t = np.linspace(-np.pi / 2, np.pi / 2, 200)
    arc = np.column_stack([1000 + 18 * np.cos(t), 18 + 18 * np.sin(t)])
    back = np.column_stack([np.linspace(1000, 0, 2001), np.full(2001, 36.0)])
    run = np.vstack([LINE, arc[1:], back[1:]])
    length = np.hypot(*np.diff(run, axis=0).T).sum()
    assert tr.footprint([run], SETUP).area == pytest.approx(length * 0.76, rel=0.005)


def test_setup_problems_are_named():
    assert tr.TyreSetup(0.38, 0.30).problems()
    assert tr.TyreSetup(0.0, 3.0).problems()
    assert not SETUP.problems()


def test_the_day_s_trampling_and_its_cost(result, day):
    truth = day[1].truth
    t = result.summary["trampling"]
    assert t["available"]
    assert t["area_ha"] == pytest.approx(truth["field_path_m"] * 0.76 / 1e4, rel=0.03)
    assert t["share_of_field"] == pytest.approx(t["area_ha"] / day[3])
    assert t["lost_kg"] == pytest.approx(t["area_ha"] * 4000)
    assert t["lost_value"] == pytest.approx(t["lost_kg"] * 0.3)
    assert t["lost_kg_per_field_ha"] == pytest.approx(t["lost_kg"] / day[3])
    assert t["tyre_size"] == "380/90R46" and t["yield_source"] == "typed"
    assert result.footprint is not None and result.footprint.area / 1e4 == pytest.approx(t["area_ha"])


def test_the_share_lost_under_the_tyre_scales_the_loss(day):
    dataset, truth_day, metric, area = day
    half = machine.analyse(dataset, {"yield_kg_ha": 4000, "loss_fraction": 50, **TYRE},
                           boundary=metric, boundary_area_ha=area).summary["trampling"]
    assert half["loss_fraction"] == 0.5
    assert half["lost_kg"] == pytest.approx(half["area_ha"] * 4000 * 0.5)


def test_a_yield_map_supplies_the_yield(day):
    dataset, _, metric, area = day
    r = machine.analyse(dataset, dict(TYRE), boundary=metric, boundary_area_ha=area,
                        yield_kg_ha=3500, yield_source="yield map 'Wheat 2026'")
    t = r.summary["trampling"]
    assert t["yield_kg_ha"] == 3500 and t["yield_source"] == "yield map 'Wheat 2026'"


def test_no_tyre_no_trampling_and_it_says_what_is_missing(day):
    dataset, _, metric, area = day
    r = machine.analyse(dataset, {}, boundary=metric, boundary_area_ha=area)
    assert r.summary["trampling"]["available"] is False
    assert any("No trampling estimate" in f["text"] for f in r.summary["findings"])


# ==========================================================================
# Settings
# ==========================================================================

def test_settings_are_read_tolerantly_and_checked():
    s = ma.MachineSettings.from_dict({"tyre_size": "18.4R38", "loss_fraction": "80",
                                      "rear_follows_front": "false", "unknown": 1, "fuel_price": ""})
    assert s.tyre_width_m == pytest.approx(0.46736)
    assert s.loss_fraction == pytest.approx(0.8)
    assert s.rear_follows_front is False and s.fuel_price is None
    with pytest.raises(ValueError, match="cannot be negative"):
        ma.MachineSettings.from_dict({"boom_width_m": -1})
    with pytest.raises(ValueError, match="must be a number"):
        ma.MachineSettings.from_dict({"boom_width_m": "wide"})
    with pytest.raises(ValueError, match="at most 100"):
        ma.MachineSettings.from_dict({"loss_fraction": 250})


# ==========================================================================
# What it says
# ==========================================================================

def test_the_findings_are_written_in_the_reader_s_units(result):
    canada = " ".join(f["text"] for f in machine.findings(result.summary, UNIT_PRESETS["canada"]))
    metric = " ".join(f["text"] for f in machine.findings(result.summary, UNIT_PRESETS["metric"]))
    assert " mi " in canada and " mph" in canada and "C$" in canada and " ac " in canada
    assert " km " in metric and " km/h" in metric and "€" in metric and " ha " in metric
    assert "L of diesel" in canada and "DEF fell" in canada


def test_the_gps_caveat_always_goes_with_the_trampling(result):
    texts = [f["text"] for f in result.summary["findings"]]
    assert any("no RTK correction" in t and "upper bound" in t for t in texts)


def test_a_long_idle_is_a_warning(result):
    idle = next(f for f in result.summary["findings"] if "stood still" in f["text"])
    assert idle["level"] == "warning"


def test_restate_rewrites_only_the_sentences(result):
    again = machine.restate(result.summary, UNIT_PRESETS["usa"])
    assert again["fuel"] is result.summary["fuel"]
    assert any(" gal " in f["text"] for f in again["findings"])


def test_phrase_writes_money_durations_and_fuel():
    say = Phrase(UNIT_PRESETS["canada"])
    assert say.money(3516.22) == "C$ 3 516"   # thin space, as every number in the app
    assert say.money(12.4) == "C$ 12.40"
    assert say.duration(23_880) == "6 h 38 min"
    assert say.duration(40) == "40 s" and say.duration(7200) == "2 h"
    assert say.liquid(132.49) == "132 L"
    assert Phrase(UNIT_PRESETS["usa"]).liquid(132.49) == "35.0 gal"
    assert say.distance(79_590) == "49.5 mi"
    assert Phrase().liquid_per_area(8.4) == "8.4 L/ha"


# ==========================================================================
# The machine profile carries the tyres and the DEF tank
# ==========================================================================

def test_a_sprayer_profile_keeps_its_tyres_and_def_tank():
    from agrosuite.core.profiles import MachineProfile

    p = MachineProfile.from_dict({
        "name": "Sprayer 120 ft", "kind": "sprayer", "implement_width_m": 36.576,
        "speed_min_kmh": 8, "speed_max_kmh": 25, "tyre_size": "380/90R46",
        "track_width_m": 3.048, "rear_follows_front": "true", "def_tank_l": 60})
    assert p.problems() == []
    assert p.tyre_width_m == pytest.approx(0.38)
    data = p.to_dict()
    assert data["tyre_size"] == "380/90R46" and data["def_tank_l"] == 60
    assert MachineProfile.from_dict(data).tyre_width_m == pytest.approx(0.38)


def test_a_profile_written_before_the_tyres_existed_still_opens():
    from agrosuite.core.profiles import MachineProfile

    p = MachineProfile.from_dict({"name": "Combine", "kind": "combine", "implement_width_m": 9.1,
                                  "speed_min_kmh": 3, "speed_max_kmh": 10})
    assert p.problems() == []
    assert p.tyre_size == "" and p.tyre_width_m == 0.0 and p.rear_follows_front is True


def test_a_profile_with_an_impossible_tyre_is_not_saved():
    from agrosuite.core.profiles import MachineProfile

    base = {"name": "X", "kind": "sprayer", "implement_width_m": 30,
            "speed_min_kmh": 8, "speed_max_kmh": 25}
    assert any("tyre size" in m for m in MachineProfile.from_dict({**base, "tyre_size": "banana"}).problems())
    narrow = MachineProfile.from_dict({**base, "tyre_size": "380/90R46", "track_width_m": 0.3})
    assert any("narrower than the tyres" in m for m in narrow.problems())


# ==========================================================================
# Units
# ==========================================================================

def test_every_preset_names_a_fuel_and_a_distance_unit():
    from agrosuite.core import units

    for key, preset in units.UNIT_PRESETS.items():
        units.unit_factor("liquid", preset["liquid_unit"])
        units.unit_factor("distance", preset["distance_unit"])
    assert units.UNIT_PRESETS["canada"]["liquid_unit"] == "L"      # diesel is sold by the litre
    assert units.UNIT_PRESETS["usa"]["liquid_unit"] == "gal"
    catalog = units.unit_catalog()
    assert {"liquid", "distance"} <= set(catalog["groups"])
