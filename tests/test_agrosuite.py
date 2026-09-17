"""AgroSuite test suite.

The tests are written against behaviours that matter in practice, not against
implementation details: a map in bu/ac has to become kg/ha with the right test
weight, cleaning has to recover the signal from a dataset with known defects,
the analysis has to find the optimum rate that was planted in the data, and
the generated file has to pass the same verification the app shows the user.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agrosuite.clean import pipeline as clean_pipeline
from agrosuite.core import schema as sch
from agrosuite.core import units as units_mod
from agrosuite.core.guidance import ab_line_from_direction, ab_line_from_points
from agrosuite.demo import synthetic_harvest, synthetic_trial
from agrosuite.difm import analysis as difm_analysis
from agrosuite.difm import response as difm_response
from agrosuite.difm.design import design_strips
from agrosuite.formats import brands, isoxml, registry, validate, writers

import fixtures as fx

QUARTER_SECTION = [
    (-105.8340, 50.4520), (-105.8229, 50.4520),
    (-105.8229, 50.4592), (-105.8340, 50.4592),
]


@pytest.fixture(scope="session")
def sample_data(tmp_path_factory) -> dict[str, Path]:
    """Generate the sample files for every monitor once."""
    target = tmp_path_factory.mktemp("sample_data")
    return fx.build_all(target)


# ==========================================================================
# Schema and units
# ==========================================================================

@pytest.mark.parametrize("raw,expected", [
    ("Yld_Vol_Dr", sch.VALUE),
    ("VRYIELDVOL", sch.VALUE),
    ("Swth_Wdth_", sch.SWATH),
    ("Arbeitsbreite (m)", sch.SWATH),
    ("Geschwindigkeit (km/h)", sch.SPEED),
    ("Longitude", sch.LON),
    ("Umidade", sch.MOISTURE),
    ("Tgt_Rate_O", sch.TARGET_RATE),
    ("coluna_inventada_xyz", None),
])
def test_alias_resolution(raw, expected):
    assert sch.resolve_column(raw) == expected


def test_map_columns_does_not_duplicate_targets():
    """Two columns pointing at the same canonical name: the first one wins."""
    mapping = sch.map_columns(["Yield", "Dry Yield", "Latitude", "Longitude"])
    assert list(mapping.values()).count(sch.VALUE) == 1


@pytest.mark.parametrize("value,unit,crop,expected", [
    (200, "bu/ac", "corn", 12553.5),
    (60, "bu/ac", "soybean", 4035.1),
    (50, "bu/ac", "canola", 2802.0),
    (90, "lb/ac", None, 100.9),
    (100, "sc/ha", None, 6000.0),
])
def test_rate_to_internal(value, unit, crop, expected):
    result = units_mod.to_internal(value, "rate_mass", unit, crop)
    assert result == pytest.approx(expected, rel=1e-3)


def test_rate_round_trip():
    """A round trip must not accumulate noticeable error."""
    for unit in ("kg/ha", "t/ha", "sc/ha", "lb/ac", "bu/ac"):
        internal = units_mod.to_internal(137.5, "rate_mass", unit, "wheat")
        back = units_mod.from_internal(internal, "rate_mass", unit, "wheat")
        assert back == pytest.approx(137.5, rel=1e-9)


def test_quarter_section_in_acres():
    """160 acres is the Prairie land unit; it has to come out right."""
    assert units_mod.to_internal(160, "area", "ac") == pytest.approx(64.75, rel=1e-3)


def test_canadian_preset_is_default():
    catalog = units_mod.unit_catalog()
    assert catalog["default_preset"] == "canada"
    preset = catalog["presets"]["canada"]
    assert preset["yield_unit"] == "bu/ac"
    assert preset["input_rate_unit"] == "lb/ac"
    assert preset["area_unit"] == "ac"
    assert preset["length_unit"] == "ft"
    assert preset["currency"] == "CAD"


# ==========================================================================
# Monitor identification
# ==========================================================================

@pytest.mark.parametrize("columns,path,expected", [
    (["VRYIELDVOL", "Machine", "Variety"], "colheita.shp", "john_deere"),
    (["Obj__Id", "Swth_Wdth_", "Track_deg_"], "export.shp", "ag_leader"),
    (["Longitude", "Prod Rate", "Boom Section"], "viper4/log.csv", "raven"),
    (["Longitude", "Tgt_Rate_O"], "gfx/coverage.csv", "trimble"),
    (["vigor", "session_id", "ndvi"], "augmenta/s.geojson", "augmenta"),
])
def test_brand_detection(columns, path, expected):
    key, confidence = brands.detect_brand(columns, path)
    assert key == expected
    assert confidence > 0.3


# ==========================================================================
# Reading the monitor formats
# ==========================================================================

@pytest.mark.parametrize("key,brand,operation", [
    ("john_deere_shp", "john_deere", "harvest"),
    ("john_deere_zip", "john_deere", "harvest"),
    ("ag_leader_shp", "ag_leader", "harvest"),
    ("raven_csv", "raven", "application"),
    ("trimble_csv", "trimble", "application"),
    ("bourgault_csv", "bourgault", "planting"),
    ("semicolon_csv", "vaderstad", "planting"),
    ("isoxml", "case_ih", "application"),
    ("augmenta", "augmenta", "application"),
])
def test_reads_monitor_formats(sample_data, key, brand, operation):
    dataset = registry.read_any(sample_data[key])
    dataset.ensure_derived()
    assert dataset.meta.brand == brand
    assert dataset.meta.operation == operation
    assert len(dataset) > 100
    assert sch.LON in dataset.df.columns and sch.LAT in dataset.df.columns
    assert dataset.df[sch.LAT].between(-90, 90).all()


def test_isoxml_binary_log_decodes_values(sample_data):
    """The binary log has to return the right rate, speed and width.

    The values written into the sample are 90 and 130 kg/ha, 8 km/h and
    18.29 m; if the binary unpacking is off by a byte, none of that matches.
    """
    dataset = registry.read_any(sample_data["isoxml"])
    dataset.ensure_derived()
    rates = dataset.df[sch.APPLIED_RATE].dropna().unique()
    assert set(np.round(rates)) == {90.0, 130.0}
    assert dataset.df[sch.SPEED].median() == pytest.approx(8.0, abs=0.5)
    assert dataset.df[sch.SWATH].median() == pytest.approx(18.29, abs=0.05)


def test_isoxml_carries_boundary(sample_data):
    dataset = registry.read_any(sample_data["isoxml"])
    setup = dataset.meta.extra.get("field_setup")
    assert setup and setup["fields"]
    assert setup["fields"][0]["boundaries"]


def test_semicolon_csv_decimal_comma(sample_data):
    """European CSV: ';' as separator and ',' as decimal mark."""
    dataset = registry.read_any(sample_data["semicolon_csv"])
    dataset.ensure_derived()
    assert dataset.df[sch.SPEED].median() == pytest.approx(8.0, abs=0.5)
    assert dataset.df[sch.SWATH].median() == pytest.approx(18.29, abs=0.05)


def test_augmenta_pairs_vigor_and_rate(sample_data):
    from agrosuite.formats.augmenta import vigor_rate_summary

    dataset = registry.read_any(sample_data["augmenta"])
    assert "vigor_index" in dataset.df.columns
    summary = vigor_rate_summary(dataset)
    assert summary["available"]
    # The sample applies more where vigour is higher: correlation must be strong.
    assert summary["correlation"] > 0.8


def test_unsupported_extension_explains_itself(tmp_path):
    bad = tmp_path / "dados.xyz"
    bad.write_bytes(b"\x00\x01")
    with pytest.raises(ValueError, match="not supported"):
        registry.read_any(bad)


# ==========================================================================
# Cleaning
# ==========================================================================

def test_cleaning_recovers_the_signal():
    """Cleaning has to move the map closer to the known true value."""
    dataset = synthetic_harvest()
    before = float(np.corrcoef(dataset.df["value"], dataset.df["truth_kg_ha"])[0, 1])

    result = clean_pipeline.run(dataset, clean_pipeline.PRESETS["harvest"])
    clean = result.clean.df
    after = float(np.corrcoef(clean["value"], clean["truth_kg_ha"])[0, 1])

    assert after > 0.75, "cleaning should recover the signal"
    assert after > before + 0.3, "the gain has to be substantial"
    assert result.report["totals"]["removed_pct"] < 40


def test_overlap_filter_finds_the_planted_overlap():
    """The synthetic set has one re-run pass; the filter has to find it."""
    dataset = synthetic_harvest()
    result = clean_pipeline.run(dataset, clean_pipeline.PRESETS["harvest"])
    overlap = next(s for s in result.report["steps"] if s["key"] == "overlap")
    assert overlap["removed"] > 200


def test_cleaning_preserves_the_original():
    dataset = synthetic_harvest()
    original_rows = len(dataset)
    clean_pipeline.run(dataset, clean_pipeline.PRESETS["harvest"])
    assert len(dataset) == original_rows


def test_minimal_preset_removes_little():
    dataset = synthetic_harvest()
    result = clean_pipeline.run(dataset, clean_pipeline.PRESETS["minimal"])
    assert result.report["totals"]["removed_pct"] < 5


def test_report_flags_excessive_removal():
    """Cleaning that is too aggressive has to show up as an alert in the report."""
    dataset = synthetic_harvest()
    config = {
        "corrections": {"flow_delay_s": 0},
        "steps": {
            "null_value": {"enabled": True},
            "global_outlier": {"enabled": True, "method": "percentile",
                               "lower_pct": 30.0, "upper_pct": 70.0},
        },
    }
    result = clean_pipeline.run(dataset, config)
    levels = {f["level"] for f in result.report["findings"]}
    assert "alert" in levels


# ==========================================================================
# Response and economics
# ==========================================================================

def test_quadratic_eonr_matches_theory():
    """Y = 7600 + 34R - 0.070R^2; with a price ratio of 5, the optimum is 207.1."""
    rates = np.repeat([0, 60, 120, 180, 240], 40).astype(float)
    yields = 7600 + 34 * rates - 0.070 * rates**2
    fit = difm_response.fit_quadratic(rates, yields)
    optimum = difm_response.optimum_rate(fit, crop_price=1.2, input_cost=6.0, rate_max=300)
    assert optimum["optimum_rate"] == pytest.approx(207.1, abs=1.0)


def test_optimum_is_below_agronomic_maximum():
    """The maximum-profit rate has to sit below the maximum-yield rate."""
    rates = np.repeat([0, 60, 120, 180, 240], 40).astype(float)
    yields = 7600 + 34 * rates - 0.070 * rates**2
    fit = difm_response.fit_quadratic(rates, yields)
    optimum = difm_response.optimum_rate(fit, crop_price=1.2, input_cost=6.0, rate_max=300)
    assert optimum["optimum_rate"] < optimum["agronomic_maximum"]


def test_free_input_optimum_equals_agronomic_maximum():
    """Free input: the economic optimum coincides with the agronomic maximum."""
    rates = np.repeat([0, 60, 120, 180, 240], 40).astype(float)
    yields = 7600 + 34 * rates - 0.070 * rates**2
    fit = difm_response.fit_quadratic(rates, yields)
    optimum = difm_response.optimum_rate(fit, crop_price=1.2, input_cost=0.0, rate_max=300)
    assert optimum["optimum_rate"] == pytest.approx(fit.plateau_rate, abs=1.0)


def test_needs_three_distinct_rates():
    with pytest.raises(ValueError, match="3 distinct rates"):
        difm_response.fit_best(np.array([0.0, 0, 100, 100]), np.array([1.0, 2, 3, 4]))


def test_difm_recovers_zone_optima():
    """The synthetic trial's zones have optima of 207 and 169 kg/ha."""
    dataset = synthetic_trial()
    report = difm_analysis.analyze(
        dataset, crop_price=1.2, input_cost=6.0,
        cell_m=20, edge_margin_m=6, zone_column="zone",
    )
    optima = {z["zone"]: z["optimum_rate"] for z in report["zones"]["by_zone"] if "optimum_rate" in z}
    assert optima["0"] == pytest.approx(207.1, abs=15)
    assert optima["1"] == pytest.approx(169.4, abs=15)


def test_aggregation_never_invents_a_dose():
    """A cell straddling two strips must not become an in-between rate."""
    dataset = synthetic_trial()
    applied = set(np.round(dataset.df["applied_rate"].unique(), 1))
    report = difm_analysis.analyze(
        dataset, crop_price=1.2, input_cost=6.0, cell_m=20, edge_margin_m=6,
    )
    assert set(report["rates_tested"]) <= applied


# ==========================================================================
# Trial layout and AB lines
# ==========================================================================

def test_design_is_balanced():
    result = design_strips(QUARTER_SECTION, rates=[0, 50, 100, 150, 200],
                           implement_width_m=18.29, passes_per_strip=2,
                           blocks=4, buffer_m=20)
    counts = set(result["summary"]["reps_per_rate"].values())
    assert len(counts) == 1, "each rate has to appear the same number of times"
    assert result["summary"]["blocks"] >= 2


def test_design_refuses_impossible_layout():
    with pytest.raises(ValueError, match="only fits"):
        design_strips(QUARTER_SECTION, rates=[0, 50, 100, 150, 200],
                      implement_width_m=200.0, passes_per_strip=2, blocks=4)


def test_design_needs_three_rates():
    with pytest.raises(ValueError, match="3 distinct rates"):
        design_strips(QUARTER_SECTION, rates=[0, 100], implement_width_m=18.29)


def test_ab_line_heading_is_compass_bearing():
    line = ab_line_from_direction(QUARTER_SECTION, angle_deg=0.0)
    assert line["heading"] == pytest.approx(90.0, abs=0.1)   # east
    line = ab_line_from_direction(QUARTER_SECTION, angle_deg=90.0)
    assert line["heading"] == pytest.approx(0.0, abs=0.1)    # north


def test_ab_line_rejects_coincident_points():
    with pytest.raises(ValueError, match="less than a metre"):
        ab_line_from_points((-105.834, 50.452), (-105.834, 50.452))


# ==========================================================================
# ISOXML round trips
# ==========================================================================

def test_prescription_round_trip(tmp_path):
    """A rate written in mg/m2 has to come back as the same kg/ha."""
    grid = np.array([[100.0, 120.0, 140.0], [110.0, 130.0, 150.0]])
    isoxml.write_prescription(tmp_path, grid, -105.834, 50.452, 0.0001, 0.0001,
                              rate_kind="mass", field_name="NW-14-32-W2")
    dataset = isoxml.read_isoxml(tmp_path / "TASKDATA")
    recovered = sorted(dataset.df[sch.TARGET_RATE].round(1).unique())
    assert recovered == [100.0, 110.0, 120.0, 130.0, 140.0, 150.0]


def test_field_setup_round_trip(tmp_path):
    lines = [{"name": "AB Principal", "type": 1,
              "a": (-105.8335, 50.4525), "b": (-105.8285, 50.4525)}]
    out = isoxml.write_field_setup(tmp_path, field_name="NW-14-32-W2",
                                   boundary=QUARTER_SECTION, guidance_lines=lines)
    back = isoxml.read_field_setup(out / "TASKDATA.XML")
    field = back["fields"][0]
    assert field["name"] == "NW-14-32-W2"
    assert field["boundaries"]
    assert field["guidance_lines"][0]["a"] == pytest.approx((-105.8335, 50.4525))
    assert field["guidance_lines"][0]["b"] == pytest.approx((-105.8285, 50.4525))


# ==========================================================================
# Packages and verification
# ==========================================================================

@pytest.fixture
def built_package(tmp_path):
    design = design_strips(QUARTER_SECTION, rates=[0, 50, 100, 150, 200],
                           implement_width_m=18.29, passes_per_strip=2,
                           blocks=4, buffer_m=20)
    line = ab_line_from_direction(QUARTER_SECTION, design["summary"]["direction_deg"])
    return design, line, tmp_path


@pytest.mark.parametrize("monitor", ["raven", "john_deere", "case_ih", "trimble", "vaderstad"])
def test_package_passes_its_own_verification(built_package, monitor):
    design, line, tmp_path = built_package
    folder = tmp_path / monitor
    writers.build_package(folder, monitor=monitor, prescription=design["features"],
                          boundary=QUARTER_SECTION, guidance_lines=[line],
                          rate_unit="lb/ac", cell_m=10, field_name="NW-14-32-W2")
    report = validate.validate_package(folder, monitor)
    assert report["verdict"] == "ok", report["summary"]
    assert report["totals"]["fail"] == 0


def test_package_uses_the_expected_rate_field(built_package):
    design, line, tmp_path = built_package
    folder = tmp_path / "trimble"
    writers.build_package(folder, monitor="trimble", prescription=design["features"],
                          rate_unit="lb/ac", field_name="T")
    shapefiles = list(folder.rglob("*.shp"))
    assert shapefiles
    import geopandas as gpd

    columns = set(gpd.read_file(shapefiles[0]).columns)
    assert "TGT_RATE" in columns, "Trimble looks for the rate in TGT_RATE"


def test_package_readme_is_written(built_package):
    design, line, tmp_path = built_package
    folder = tmp_path / "raven"
    result = writers.build_package(folder, monitor="raven",
                                   prescription=design["features"],
                                   boundary=QUARTER_SECTION)
    readme = Path(result["readme"]).read_text(encoding="utf-8")
    assert "RAVEN VIPER 4" in readme
    assert "HOW TO LOAD IT" in readme


def test_validator_catches_missing_sidecar(tmp_path):
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import Polygon

    poly = Polygon([(-105.83, 50.45), (-105.82, 50.45), (-105.82, 50.46)])
    gpd.GeoDataFrame(pd.DataFrame({"RATE": [100.0]}), geometry=[poly],
                     crs="EPSG:4326").to_file(tmp_path / "rx.shp")
    (tmp_path / "rx.prj").unlink()
    checks = validate.validate_shapefile(tmp_path / "rx.shp", rate_field="RATE")
    assert any(c.status == "fail" and "prj" in c.message for c in checks)


def test_validator_catches_points_used_as_prescription(tmp_path):
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import Point

    gpd.GeoDataFrame(pd.DataFrame({"RATE": [100.0]}), geometry=[Point(-105.83, 50.45)],
                     crs="EPSG:4326").to_file(tmp_path / "rx.shp")
    checks = validate.validate_shapefile(tmp_path / "rx.shp", rate_field="RATE")
    assert any(c.status == "fail" and "polygons" in c.message for c in checks)


def test_validator_catches_grid_length_mismatch(tmp_path):
    grid = np.array([[100.0, 120.0], [110.0, 130.0]])
    isoxml.write_prescription(tmp_path, grid, -105.834, 50.452, 0.0001, 0.0001)
    binary = tmp_path / "TASKDATA" / "GRD00001.BIN"
    binary.write_bytes(binary.read_bytes()[:-4])  # drop one cell
    checks = validate.validate_taskdata(tmp_path / "TASKDATA")
    assert any(c.status == "fail" and "bytes" in c.message for c in checks)


# ==========================================================================
# John Deere card
# ==========================================================================

def test_greenstar_card_is_recognized(tmp_path):
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import Polygon

    from agrosuite.formats import johndeere as jd

    card = tmp_path / "GS3_2630"
    (card / "SETUP").mkdir(parents=True)
    (card / "RCD").mkdir()
    poly = Polygon(QUARTER_SECTION)
    gpd.GeoDataFrame(pd.DataFrame([{"NAME": "NW-14-32-W2"}]), geometry=[poly],
                     crs="EPSG:4326").to_file(card / "SETUP" / "Boundary.shp")
    (card / "SETUP" / "Setup.jdf").write_bytes(b"JDF" + bytes(64))

    inventory = jd.inventory(tmp_path)
    assert "GreenStar 3" in inventory.generation
    assert inventory.readable and inventory.proprietary
    assert jd.readable_layers(inventory)[0]["role"] == "boundary"

    dataset = registry.read_any(tmp_path)
    assert dataset.meta.brand == "john_deere"
    assert dataset.meta.extra["field_setup"]["fields"][0]["boundaries"]


def test_card_with_only_proprietary_files_explains_the_way_out(tmp_path):
    card = tmp_path / "GS3_2630"
    (card / "SETUP").mkdir(parents=True)
    (card / "SETUP" / "Setup.jdf").write_bytes(b"JDF" + bytes(64))
    with pytest.raises(ValueError, match="shapefile instead of GreenStar"):
        registry.read_any(tmp_path)


# ==========================================================================
# Preliminary analysis
# ==========================================================================

def test_preflight_catches_undeclared_imperial_units(sample_data):
    """A bu/ac file imported as kg/ha has to be caught, not quietly accepted.

    This is the error with the worst consequences in the whole app: every
    number downstream inherits a factor of fifty, and nothing else looks wrong.
    """
    from agrosuite.core import preflight

    dataset = registry.read_any(sample_data["john_deere_shp"])
    dataset.ensure_derived()
    report = preflight.run(dataset)

    assert report["verdict"] == "alert"
    unit_finding = next(f for f in report["findings"] if f["title"] == "Units look wrong")
    assert "bu/ac" in unit_finding["detail"]
    assert report["next_step"]["step"] == "units"


def test_preflight_accepts_a_metric_file(sample_data):
    from agrosuite.core import preflight

    dataset = registry.read_any(sample_data["isoxml"])
    dataset.ensure_derived()
    report = preflight.run(dataset)
    assert report["verdict"] == "ok"
    assert report["suggested_role"] == "as_applied"


def test_preflight_warns_when_only_width_looks_imperial(sample_data):
    """A rate is plausible at both 96 kg/ha and 96 lb/ac.

    The value alone cannot settle it, so the check has to lean on the speed and
    the width, which a monitor writes in the same system as the rate.
    """
    from agrosuite.core import preflight

    dataset = registry.read_any(sample_data["raven_csv"])
    dataset.ensure_derived()
    report = preflight.run(dataset)
    assert any(f["title"] == "Possibly imperial units" for f in report["findings"])


def test_preflight_reads_the_crop_from_the_file(sample_data):
    from agrosuite.core import preflight

    dataset = registry.read_any(sample_data["john_deere_shp"])
    dataset.ensure_derived()
    report = preflight.run(dataset)
    assert report["info"]["crop_detected"] == "canola"


# ==========================================================================
# Treatment levels
# ==========================================================================

@pytest.mark.parametrize("noise", [0.0, 0.01, 0.03, 0.05])
def test_rate_levels_recovers_the_design(noise):
    """Machine execution error must not turn five rates into twelve."""
    rng = np.random.default_rng(4)
    nominal = np.repeat([0.0, 60, 120, 180, 240], 1200)
    observed = nominal + rng.normal(0, noise * np.maximum(nominal, 20))
    _, levels = difm_analysis.rate_levels(observed)
    assert len(levels) == 5
    assert levels == pytest.approx([0, 60, 120, 180, 240], abs=max(3.0, 240 * noise))


def test_rate_levels_does_not_invent_structure():
    """A genuinely continuous rate map has no treatments to find."""
    rng = np.random.default_rng(4)
    _, levels = difm_analysis.rate_levels(rng.uniform(50, 250, 3000))
    assert len(levels) == 12  # falls back to binning, which preserves order


def test_rate_levels_handles_a_single_rate():
    _, levels = difm_analysis.rate_levels(np.full(500, 90.0))
    assert levels == [90.0]


# ==========================================================================
# Joining layers
# ==========================================================================

@pytest.fixture(scope="session")
def difm_files(tmp_path_factory) -> dict[str, Path]:
    return fx.difm_project(tmp_path_factory.mktemp("difm"))


def _load_project(difm_files) -> dict[str, Any]:
    """Load the three files the way the app does, declaring imperial units."""
    from agrosuite.core.dataset import apply_source_units

    imperial = {
        sch.SPEED: "mph", sch.SWATH: "ft",
        sch.TARGET_RATE: "lb/ac", sch.APPLIED_RATE: "lb/ac",
    }
    layers = {}
    for role, path, value_unit in [
        ("plan", difm_files["plan"], "lb/ac"),
        ("as_applied", difm_files["as_applied"], "lb/ac"),
        ("yield", difm_files["yield"], "bu/ac"),
    ]:
        dataset = registry.read_any(path)
        apply_source_units(dataset, {**imperial, sch.VALUE: value_unit}, "canola")
        dataset.ensure_derived()
        layers[role] = dataset
    return layers


def test_the_three_files_get_the_right_roles(difm_files):
    """A plan, an as-applied log and a yield map must not be confused.

    A plan carries only a target; an as-applied log carries what actually went
    out. Reading one as the other changes which numbers the response is fitted
    against.
    """
    from agrosuite.core import preflight

    expected = {"plan": "plan", "as_applied": "as_applied", "yield": "yield"}
    for role, path in difm_files.items():
        dataset = registry.read_any(path)
        dataset.ensure_derived()
        assert preflight.run(dataset)["suggested_role"] == expected[role]


def test_join_keeps_the_trial_rates_intact(difm_files):
    """Cells straddling two strips must not average into a rate nobody applied."""
    from agrosuite.difm.join import join_layers

    layers = _load_project(difm_files)
    joined, report = join_layers(layers, cell_m=20, carry=["Zone"])

    rates = sorted(joined.df[sch.APPLIED_RATE].round(0).unique())
    assert len(rates) == 5
    assert rates == pytest.approx([0, 45, 90, 135, 180], abs=3)
    assert report["carried"] == ["Zone"]


def test_join_reports_how_far_the_machine_drifted(difm_files):
    from agrosuite.difm.join import join_layers

    layers = _load_project(difm_files)
    _, report = join_layers(layers, cell_m=20)
    assert "plan_vs_applied" in report
    assert report["plan_vs_applied"]["median_relative_pct"] is not None


def test_join_refuses_without_a_yield_layer(difm_files):
    from agrosuite.difm.join import join_layers

    layers = _load_project(difm_files)
    with pytest.raises(ValueError, match="needs a yield layer"):
        join_layers({"as_applied": layers["as_applied"]})


def test_join_refuses_without_a_rate(difm_files):
    from agrosuite.difm.join import join_layers

    layers = _load_project(difm_files)
    with pytest.raises(ValueError, match="rate that was applied"):
        join_layers({"yield": layers["yield"]})


def test_joined_layers_recover_the_zone_optima(difm_files):
    """The whole point of the exercise: three files in, two optima out.

    The trial's true economic optima, at these prices, are about 181 kg/ha in
    the poorer zone and 170 in the richer one.
    """
    from agrosuite.difm.join import join_layers

    layers = _load_project(difm_files)
    joined, _ = join_layers(layers, cell_m=20, carry=["Zone"])

    crop_price = 16.50 / 22.6796      # C$ per kg of canola
    input_cost = 0.62 / 0.45359237    # C$ per kg of N
    report = difm_analysis.analyze(
        joined, crop_price=crop_price, input_cost=input_cost,
        cell_m=20, edge_margin_m=0, zone_column="Zone",
    )
    optima = {z["zone"]: z["optimum_rate"] for z in report["zones"]["by_zone"]
              if "optimum_rate" in z}
    assert len(optima) == 2
    for value in optima.values():
        assert 140 <= value <= 200
    for zone in report["zones"]["by_zone"]:
        assert zone["r2"] > 0.85


# ==========================================================================
# Workflow
# ==========================================================================

def test_workflow_names_what_is_missing():
    from agrosuite.core import workflow

    evaluation = workflow.evaluate("difm", {"yield"}, has_prices=False)
    assert not evaluation["ready"]
    missing = {r["key"] for r in evaluation["missing"]}
    assert missing == {"rate", "prices"}


def test_workflow_accepts_a_plan_when_there_is_no_as_applied():
    from agrosuite.core import workflow

    evaluation = workflow.evaluate("difm", {"yield", "plan"}, has_prices=True)
    assert evaluation["ready"]


def test_workflow_next_action_follows_the_order():
    from agrosuite.core import workflow

    ready = workflow.evaluate("difm", {"yield", "as_applied"}, has_prices=True)
    assert workflow.next_action("review", "difm", ready)["step"] == "review"
    assert workflow.next_action("clean", "difm", ready)["step"] == "clean"
    assert workflow.next_action("analyse", "difm", ready)["step"] == "analyse"

    incomplete = workflow.evaluate("difm", {"yield"}, has_prices=False)
    assert workflow.next_action("clean", "difm", incomplete)["step"] == "load"


# ==========================================================================
# Writing to a USB drive
# ==========================================================================

def test_usb_never_overwrites_without_permission(tmp_path):
    """A stick carries other jobs. Wiping them because the app assumed it was
    scratch space would be unforgivable."""
    from agrosuite.formats import usb

    stick = tmp_path / "stick"
    (stick / "TASKDATA").mkdir(parents=True)
    (stick / "TASKDATA" / "last_year.xml").write_text("old")
    (stick / "photos").mkdir()
    (stick / "photos" / "a.jpg").write_bytes(b"0" * 10)

    package = tmp_path / "package"
    (package / "TASKDATA").mkdir(parents=True)
    (package / "TASKDATA" / "TASKDATA.XML").write_text("new")
    (package / "README.txt").write_text("how to load it")

    plan = usb.plan_write(package, stick)
    assert plan["needs_confirmation"]
    assert [c["name"] for c in plan["conflicts"]] == ["TASKDATA"]

    result = usb.write_to_drive(package, stick)
    assert result["copied"] == ["README.txt"]
    assert [s["name"] for s in result["skipped"]] == ["TASKDATA"]
    assert (stick / "TASKDATA" / "last_year.xml").exists()

    result = usb.write_to_drive(package, stick, replace=["TASKDATA"])
    assert result["copied"] == ["TASKDATA"]
    assert (stick / "TASKDATA" / "TASKDATA.XML").exists()
    assert not (stick / "TASKDATA" / "last_year.xml").exists()
    assert (stick / "photos" / "a.jpg").exists(), "unrelated files must survive"


def test_usb_refuses_a_missing_drive(tmp_path):
    from agrosuite.formats import usb

    package = tmp_path / "package"
    package.mkdir()
    with pytest.raises(ValueError, match="Drive not found"):
        usb.plan_write(package, str(tmp_path / "no-such-drive"))


# ==========================================================================
# QGIS
# ==========================================================================

def test_qgis_round_trip(tmp_path):
    """Layers out to a GeoPackage and a project, and back in by name.

    The GeoPackage is the reliable half of the bridge, so the test asserts on
    it: the project file is a convenience whose schema belongs to QGIS.
    """
    from agrosuite.formats import qgis

    layers = {"yield 2025": synthetic_harvest(), "difm trial": synthetic_trial()}
    result = qgis.export_for_qgis(layers, tmp_path, name="NW-14-32-W2")

    assert Path(result["geopackage"]["path"]).exists()
    assert len(result["geopackage"]["layers"]) == 2
    assert all(layer["crs"] == "EPSG:4326" for layer in result["geopackage"]["layers"])

    project = qgis.read_project(Path(result["project"]["path"]))
    assert len(project["importable"]) == 2

    dataset = qgis.read_project_layer(project["importable"][0])
    assert len(dataset) == len(layers["yield 2025"])
    assert "QGIS project layer" in dataset.meta.notes[-1]


def test_qgis_layer_names_survive_the_round_trip(tmp_path):
    """A name with spaces has to come back recognizable, not as 'layer1'."""
    from agrosuite.formats import qgis

    result = qgis.export_for_qgis(
        {"NW 14-32 W2 yield": synthetic_harvest()}, tmp_path, name="field"
    )
    project = qgis.read_project(Path(result["project"]["path"]))
    assert project["layers"][0]["name"] == "NW_14-32_W2_yield"


def test_qgis_reports_a_layer_stored_elsewhere(tmp_path):
    """A project written on another machine points at paths that are not here."""
    from agrosuite.formats import qgis

    project = tmp_path / "elsewhere.qgs"
    project.write_text(
        '<?xml version="1.0"?><qgis version="3.28.0"><title>Elsewhere</title>'
        '<projectlayers><maplayer><datasource>/not/here/soil.shp</datasource>'
        '<layername>Soil</layername><provider>ogr</provider></maplayer>'
        "</projectlayers></qgis>",
        encoding="utf-8",
    )
    read = qgis.read_project(project)
    assert read["layers"][0]["exists"] is False
    assert read["importable"] == []


def test_qgis_project_is_refused_as_a_data_file(tmp_path):
    """A project names layers; it does not hold them. Saying so beats a parse error."""
    project = tmp_path / "field.qgs"
    project.write_text("<qgis/>", encoding="utf-8")
    with pytest.raises(ValueError, match="QGIS project"):
        registry.read_any(project)


# ==========================================================================
# The MCP surface
# ==========================================================================

def test_mcp_declares_its_tools():
    from agrosuite import mcp_server

    listing = mcp_server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    tools = listing["result"]["tools"]
    assert {t["name"] for t in tools} >= {
        "open_file", "project_status", "clean_dataset", "analyse_economics",
        "list_usb_drives", "plan_usb_write", "write_to_usb",
    }
    for tool in tools:
        assert tool["description"].strip()
        assert tool["inputSchema"]["type"] == "object"


def test_mcp_initialize_declares_the_protocol():
    from agrosuite import mcp_server

    response = mcp_server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    assert response["result"]["protocolVersion"] == mcp_server.PROTOCOL_VERSION
    assert response["result"]["serverInfo"]["name"] == "agrosuite"


def test_mcp_reports_a_bad_tool_as_an_error_not_a_crash():
    from agrosuite import mcp_server

    response = mcp_server.handle({
        "jsonrpc": "2.0", "id": 2, "method": "tools/call",
        "params": {"name": "no_such_tool", "arguments": {}},
    })
    assert "error" in response


def test_mcp_ignores_notifications():
    from agrosuite import mcp_server

    assert mcp_server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None


def test_mcp_write_to_usb_requires_explicit_replacement():
    """The assistant must not be able to overwrite a stick by accident."""
    from agrosuite import mcp_server

    tool = mcp_server.TOOLS_BY_NAME["write_to_usb"]
    assert "replace" in tool["schema"]["properties"]
    assert "replace" not in tool["schema"]["required"]
    assert "left alone" in tool["description"].lower()

    planner = mcp_server.TOOLS_BY_NAME["plan_usb_write"]
    assert "writes nothing" in planner["description"].lower()


# ==========================================================================
# Seeing what the cleaning did
# ==========================================================================

def test_removed_records_carry_their_reason():
    """A percentage removed is not judgeable; a map of where it went is."""
    from agrosuite.app import session as session_mod

    dataset = synthetic_harvest()
    result = clean_pipeline.run(dataset, clean_pipeline.PRESETS["harvest"])

    assert "removal_reason" in result.removed.df.columns
    payload = session_mod.map_payload(result.removed, group_column="removal_reason")
    assert payload["group_column"] == "removal_reason"
    assert len(payload["groups"]) == payload["count"]

    reasons = set(filter(None, payload["groups"]))
    assert "Swath overlap" in reasons, "the planted overlap has to be visible on the map"


def test_first_look_proposes_the_whole_unit_set(sample_data):
    """One click, not a dialog with three pickers: the app already worked the
    units out from the file."""
    from agrosuite.core import preflight

    dataset = registry.read_any(sample_data["john_deere_shp"])
    dataset.ensure_derived()
    proposed = preflight.run(dataset)["proposed_units"]

    assert proposed["rate"] == "bu/ac"
    assert proposed["length"] == "ft"
    assert proposed["crop"] == "canola"


def test_applying_the_proposed_units_fixes_the_magnitude(sample_data):
    """The proposal has to be right, not merely plausible."""
    from agrosuite.core import preflight
    from agrosuite.core.dataset import apply_source_units

    dataset = registry.read_any(sample_data["john_deere_shp"])
    dataset.ensure_derived()
    proposed = preflight.run(dataset)["proposed_units"]

    converted = dataset.copy()
    apply_source_units(converted, {
        sch.VALUE: proposed["rate"],
        sch.SPEED: proposed.get("speed", "km/h"),
        sch.SWATH: proposed.get("length", "m"),
    }, proposed.get("crop"))
    converted.ensure_derived()

    after = preflight.run(converted)
    assert after["verdict"] == "ok"
    assert 800 <= converted.stats()["mean"] <= 6000, "canola yield in kg/ha"
