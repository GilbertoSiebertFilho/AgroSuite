"""The elevation rasters people actually hand the terrain analyser.

A DEM is rarely the clean float GeoTIFF the synthetic tests write. It comes
in feet, in whole metres (SRTM, and anything QGIS converted from an integer
raster), clipped to the wrong place so every cell is nodata, or it is not a
DEM at all but the orthophoto next to it. Each case here once produced a
confident wrong answer — a 700 m field reported at 2 296 m, ten hills on a
field with two, an "elevation layer" of zero rows, a photo analysed as
relief — and now either refuses in a sentence or answers honestly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.warp import Resampling, reproject, transform_bounds

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agrosuite.core import preflight
from agrosuite.core import schema as sch
from agrosuite.core.dataset import apply_source_units
from agrosuite.formats import raster as raster_mod
from agrosuite.formats import registry
from agrosuite.terrain import synthetic_dem
from agrosuite.terrain.analysis import TerrainOptions, analyze

FT = 0.3048


@pytest.fixture(scope="module")
def dem():
    return synthetic_dem()


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from agrosuite.app import server as server_mod

    return TestClient(server_mod.app)


def _write_raw(path: Path, data: np.ndarray, crs: str, transform: Affine, nodata=None) -> Path:
    """Write a raster the way a third-party tool would: no help from the app."""
    bands = data if data.ndim == 3 else data[None]
    with rasterio.open(
        path, "w", driver="GTiff", height=bands.shape[1], width=bands.shape[2],
        count=bands.shape[0], dtype=bands.dtype.name, crs=crs, transform=transform, nodata=nodata,
    ) as dst:
        dst.write(bands)
    return path


def _int16_dem(grid, path: Path) -> Path:
    data = np.where(np.isfinite(grid.z), np.round(grid.z), -32768).astype("int16")
    return _write_raw(path, data, grid.crs, grid.transform(), nodata=-32768)


def _brief(summary: dict) -> dict:
    return {
        "hills": len(summary["features"]["hills"]),
        "lows": len(summary["features"]["lows"]),
        "p95": summary["slope"]["p95_pct"],
        "contours": summary["contours"]["count"],
    }


# ==========================================================================
# 1. A declared elevation unit reaches the raster the analysis re-reads
# ==========================================================================

def test_declared_feet_reach_the_raster_analysis(dem, tmp_path, client):
    grid, _ = dem
    path = raster_mod.write_geotiff(grid, tmp_path / "dem_ft.tif", values=grid.z / FT, dtype="float64")

    body = client.post(
        "/api/import/path", json={"path": str(path), "source_units": {"elev_m": "ft"}}
    ).json()
    assert body["meta"]["extra"]["elevation_factor"] == pytest.approx(FT)
    assert body["meta"]["extra"]["elevation_unit_in"] == "ft"
    assert body["meta"]["source_value_unit"] == "ft"
    # The map colours come from `value`, which mirrors the elevation.
    assert 690 < body["stats"]["median"] < 710

    summary = client.post("/api/terrain/analyze", json={"dataset_id": body["id"]}).json()["summary"]
    assert summary["source"]["kind"] == "dem"
    assert summary["elevation"]["mean_m"] == pytest.approx(float(np.nanmean(grid.z)), abs=0.05)
    assert summary["elevation"]["relief_m"] == pytest.approx(
        float(np.nanmax(grid.z) - np.nanmin(grid.z)), abs=0.05)
    assert summary["character"]["key"] == "gently_undulating"
    assert any("converted from ft to metres" in f["text"] for f in summary["findings"])


def test_units_declared_after_import_convert_the_raster_too(dem, tmp_path, client):
    grid, _ = dem
    path = raster_mod.write_geotiff(grid, tmp_path / "dem_ft2.tif", values=grid.z / FT, dtype="float64")
    raw = client.post("/api/import/path", json={"path": str(path)}).json()
    assert client.post("/api/terrain/analyze", json={"dataset_id": raw["id"]}).json()[
        "summary"]["elevation"]["mean_m"] > 2000  # feet read as metres, as declared: none

    converted = client.post(f"/api/datasets/{raw['id']}/units", json={"source_units": {"elev_m": "ft"}}).json()
    summary = client.post("/api/terrain/analyze", json={"dataset_id": converted["dataset"]["id"]}).json()["summary"]
    assert summary["elevation"]["mean_m"] == pytest.approx(float(np.nanmean(grid.z)), abs=0.05)


def test_elevation_factor_accumulates_with_the_column(dem):
    grid, _ = dem
    ds = raster_mod.dataset_from_dem(grid, {"path": "", "notes": []})
    ds.df[sch.ELEVATION] = ds.df[sch.ELEVATION] / FT
    ds.df[sch.VALUE] = ds.df[sch.VALUE] / FT
    apply_source_units(ds, {"elev_m": "ft"})
    assert ds.meta.extra["elevation_factor"] == pytest.approx(FT)
    assert ds.df[sch.VALUE].median() == pytest.approx(ds.df[sch.ELEVATION].median())
    assert 690 < ds.df[sch.ELEVATION].median() < 710
    # A second declaration on the same dataset converts the column again and
    # the factor follows it, so the raster and the points never disagree.
    apply_source_units(ds, {"elev_m": "ft"})
    assert ds.meta.extra["elevation_factor"] == pytest.approx(FT * FT)


# ==========================================================================
# 2. Whole-metre DEMs: terraces are not hills
# ==========================================================================

def test_value_step_is_measured_on_the_raw_values():
    rng = np.random.default_rng(0)
    z = 700.0 + rng.normal(0.0, 2.0, (12, 12))
    whole = np.round(z)
    whole[3, 4] = np.nan
    assert raster_mod._value_step(whole) == 1.0
    assert raster_mod._value_step(np.round(z * 2.0) / 2.0) == 0.5
    assert raster_mod._value_step(np.round(z, 1)) == 0.1
    assert raster_mod._value_step(z) == 0.0
    assert raster_mod._value_step(np.full((12, 12), np.nan)) is None  # nothing to measure
    assert raster_mod._value_step(np.round(z).astype("float32")) == 1.0
    # Too few values to tell chance from quantisation.
    assert raster_mod._value_step(np.array([700.0, 701.0, 703.0])) == 0.0


@pytest.mark.parametrize("dtype", ["int16", "float32"])
def test_whole_metre_dem_is_smoothed_and_says_so(dem, tmp_path, dtype):
    grid, _ = dem
    if dtype == "int16":
        path = _int16_dem(grid, tmp_path / "dem_i16.tif")
    else:  # SRTM after QGIS converted it: float values, still whole metres
        path = raster_mod.write_geotiff(grid, tmp_path / "dem_whole.tif", values=np.round(grid.z), dtype="float32")
    ds = registry.read_any(path)
    assert ds.meta.extra["dem_info"]["value_step_m"] == 1.0

    result = analyze(ds)
    s = result.summary()
    assert s["source"]["value_step_m"] == 1.0
    assert s["options"]["min_feature_height_m"] == 1.0
    assert s["options"]["smooth_m"] == 10.0
    brief = _brief(s)
    assert brief["hills"] == 2 and brief["lows"] == 2, brief
    assert brief["p95"] < 5.0, brief  # 10.6 % on the raw staircase
    assert brief["contours"] < 40, brief  # 255 loops around rounding jitter without the micro-rounding
    assert not any("steeper than" in f["text"] for f in s["findings"])
    assert any("steps of 1 m" in f["text"] and "smoothed over 10 m" in f["text"] for f in s["findings"])

    # An explicit 0 keeps the raw surface: the user asked for it.
    raw = analyze(ds, TerrainOptions(smooth_m=0.0)).summary()
    assert raw["options"]["smooth_m"] == 0.0 and raw["slope"]["p95_pct"] > 8.0


def test_float_dem_is_not_taken_for_a_quantised_one(dem, tmp_path):
    grid, _ = dem
    path = raster_mod.write_geotiff(grid, tmp_path / "dem_f.tif", dtype="float64")
    ds = registry.read_any(path)
    assert ds.meta.extra["dem_info"]["value_step_m"] == 0.0
    s = analyze(ds).summary()
    assert s["options"]["smooth_m"] == 0.0 and s["options"]["min_feature_height_m"] == 0.5
    assert any("no GPS noise or pass offsets apply" in f["text"] for f in s["findings"])


@pytest.mark.parametrize("dtype", ["float32", "int16"])
def test_constant_dem_is_flat_not_quantised(dem, tmp_path, dtype):
    """One value lies on every lattice. Calling it 'whole metres' forced a
    10 m smoothing and a 1 m minimum feature height on a surface with no
    terraces; the raster now says it cannot tell (None) and the analysis
    keeps its defaults. A terraced field is a different thing: the whole
    metre synthetic DEM holds eleven distinct values and is still caught."""
    grid, _ = dem
    path = _write_raw(tmp_path / f"flat_{dtype}.tif", np.full(grid.shape, 700, dtype=dtype),
                      grid.crs, grid.transform())
    _, info = raster_mod.read_dem(path)
    assert info["value_step_m"] is None
    ds = registry.read_any(path)
    assert ds.meta.extra["dem_info"]["value_step_m"] is None

    s = analyze(ds).summary()
    assert s["source"]["value_step_m"] == 0.0
    assert s["options"]["smooth_m"] == 0.0 and s["options"]["min_feature_height_m"] == 0.5
    assert not any("steps of" in f["text"] for f in s["findings"])
    assert any("no GPS noise or pass offsets apply" in f["text"] for f in s["findings"])

    whole = np.round(grid.z)
    assert np.unique(whole[np.isfinite(whole)]).size == 11
    assert raster_mod._value_step(whole) == 1.0
    assert raster_mod._value_step(np.full(200, 700.0)) is None
    assert raster_mod._value_step(np.full(200, 700.0), integer=True) is None
    assert raster_mod._value_step(np.array([700.0] * 100 + [701.0] * 100)) == 1.0


def test_integer_dem_in_degrees_keeps_its_step_through_the_warp(dem, tmp_path):
    """A 1 arc-second SRTM-like tile: the warper interpolates between the
    plateaus, so the step has to be read before it — and it is."""
    grid, _ = dem
    west, south, east, north = transform_bounds(grid.crs, "EPSG:4326", *grid.bounds_metric())
    cell = 1.0 / 3600.0
    cols, rows = int(np.ceil((east - west) / cell)), int(np.ceil((north - south) / cell))
    transform = Affine(cell, 0.0, west, 0.0, -cell, north)
    dst = np.full((rows, cols), np.nan)
    reproject(grid.z, dst, src_transform=grid.transform(), src_crs=grid.crs, src_nodata=np.nan,
              dst_transform=transform, dst_crs="EPSG:4326", dst_nodata=np.nan,
              resampling=Resampling.average)
    path = _write_raw(tmp_path / "srtm.tif", np.where(np.isfinite(dst), np.round(dst), -32768).astype("int16"),
                      "EPSG:4326", transform, nodata=-32768)
    back, info = raster_mod.read_dem(path)
    assert info["reprojected"] and info["value_step_m"] == 1.0
    assert not np.allclose(back.z[back.mask], np.round(back.z[back.mask]))  # the warp did blur them

    s = analyze(registry.read_any(path)).summary()
    brief = _brief(s)
    assert brief["hills"] == 2 and brief["lows"] == 2, brief
    assert brief["contours"] < 40, brief  # 2 049 before
    assert s["options"]["smooth_m"] == pytest.approx(back.cell)  # one cell: ten steps is less than that


# ==========================================================================
# 3. Rasters that are not elevation models
# ==========================================================================

def test_all_nodata_raster_is_refused_at_import(dem, tmp_path, client):
    grid, _ = dem
    path = _write_raw(tmp_path / "nothing.tif", np.full((40, 40), -9999.0, dtype="float32"),
                      grid.crs, grid.transform(), nodata=-9999.0)
    with pytest.raises(ValueError, match="holds no elevation.*nodata \\(-9999\\)"):
        raster_mod.read_dem(path)
    response = client.post("/api/import/path", json={"path": str(path)})
    assert response.status_code == 400
    assert "holds no elevation" in response.json()["detail"]
    assert "Transparency" in response.json()["detail"]


def test_rgb_image_is_refused_and_flagged_by_the_inspector(dem, tmp_path, client):
    grid, _ = dem
    rng = np.random.default_rng(0)
    path = _write_raw(tmp_path / "photo.tif", rng.integers(0, 255, (3, 40, 40), dtype="uint8").astype("uint8"),
                      grid.crs, grid.transform())
    info = raster_mod.describe(path)
    assert info["looks_like_image"] and "an image, not an elevation model" in info["summary"]
    with pytest.raises(ValueError, match="is an image"):
        raster_mod.read_dem(path)
    response = client.post("/api/import/path", json={"path": str(path)})
    assert response.status_code == 400 and "single-band GeoTIFF" in response.json()["detail"]
    inspected = client.get("/api/inspect", params={"path": str(path)}).json()
    assert "an image, not an elevation model" in inspected["detail"]


def test_doubtful_rasters_are_read_with_a_note(dem, tmp_path):
    grid, _ = dem
    z = np.where(np.isfinite(grid.z), grid.z, -9999.0).astype("float32")
    two_band = _write_raw(tmp_path / "two.tif", np.stack([z, np.zeros_like(z)]), grid.crs,
                          grid.transform(), nodata=-9999.0)
    ds = registry.read_any(two_band)
    assert any("2 bands; only the first was read" in n for n in ds.meta.notes)
    assert ds.df[sch.ELEVATION].median() == pytest.approx(float(np.nanmedian(grid.z)), abs=0.5)

    small = np.clip(np.round(np.nan_to_num(grid.z[:40, :40], nan=650.0) - 650.0), 0, 255).astype("uint8")
    byte = _write_raw(tmp_path / "byte.tif", small, grid.crs, grid.transform())
    _, info = raster_mod.read_dem(byte)
    assert any("8-bit values" in n for n in info["notes"])
    assert info["value_step_m"] is None  # that corner is one value: flat, not terraced
    assert not raster_mod.describe(byte)["looks_like_image"]


# ==========================================================================
# 4. One area for an elevation layer
# ==========================================================================

def test_dem_area_is_the_same_figure_everywhere(dem, tmp_path, client):
    grid, _ = dem
    path = raster_mod.write_geotiff(grid, tmp_path / "area.tif", dtype="float64")
    ds = registry.read_any(path)
    ds.ensure_derived()
    footprint = round(grid.area_ha(), 2)
    assert ds.area_ha() == pytest.approx(footprint)
    assert ds.summary()["area_ha"] == pytest.approx(footprint)
    report = preflight.run(ds)
    assert report["info"]["area_ha"] == pytest.approx(footprint)

    body = client.post("/api/import/path", json={"path": str(path)}).json()
    assert body["area_ha"] == pytest.approx(footprint)
    # Over the wire the sentence is written in the units the app opens in —
    # acres, on the Canadian default — and it is the same figure.
    assert (f"{footprint / 0.40468564224:.1f} ac"
            in body["preflight"]["findings"][0]["detail"])
    listed = next(d for d in client.get("/api/datasets").json()["datasets"] if d["id"] == body["id"])
    assert listed["area_ha"] == pytest.approx(footprint)


# ==========================================================================
# 5. Fill values the raster never declared as nodata
# ==========================================================================

def _undeclared(grid, path: Path, fill, crs=None, transform=None, dtype="float32") -> Path:
    """A DEM whose empty cells hold ``fill`` with no nodata tag: what a clip
    or a format conversion leaves behind when the tag is lost."""
    z = np.where(np.isfinite(grid.z), grid.z, fill).astype(dtype)
    return _write_raw(path, z, crs or grid.crs, transform or grid.transform())


def test_undeclared_minus_9999_is_read_as_nodata(dem, tmp_path, client):
    """Before: the first look said 'clean and complete' and the analysis then
    refused with a contour message about 10 703 m of relief."""
    grid, _ = dem
    path = _undeclared(grid, tmp_path / "undeclared.tif", -9999.0)
    back, info = raster_mod.read_dem(path)
    assert info["nodata"] is None and info["nodata_detected"] == -9999.0
    assert np.array_equal(back.mask, grid.mask)
    assert any("declares no nodata value, but -9999 fills 6.2 %" in n and "Transparency" in n
               for n in info["notes"])

    body = client.post("/api/import/path", json={"path": str(path)}).json()
    assert body["preflight"]["verdict"] == "ok"
    assert any("declares no nodata value" in n for n in body["meta"]["notes"])
    summary = client.post("/api/terrain/analyze", json={"dataset_id": body["id"]}).json()["summary"]
    assert summary["elevation"]["relief_m"] == pytest.approx(
        float(np.nanmax(grid.z) - np.nanmin(grid.z)), abs=0.05)
    assert summary["contours"]["count"] < 40


def test_undeclared_fill_is_masked_before_the_warp(dem, tmp_path):
    """Through the reprojection path the fill has to be known before the
    warper runs, or bilinear interpolation smears -9999 into the cells next
    to it and the field edge sinks by thousands of metres."""
    grid, _ = dem
    west, south, east, north = transform_bounds(grid.crs, "EPSG:4326", *grid.bounds_metric())
    cell = 1.0 / 3600.0
    cols, rows = int(np.ceil((east - west) / cell)), int(np.ceil((north - south) / cell))
    transform = Affine(cell, 0.0, west, 0.0, -cell, north)
    dst = np.full((rows, cols), np.nan)
    reproject(grid.z, dst, src_transform=grid.transform(), src_crs=grid.crs, src_nodata=np.nan,
              dst_transform=transform, dst_crs="EPSG:4326", dst_nodata=np.nan,
              resampling=Resampling.average)
    path = _write_raw(tmp_path / "geo_undeclared.tif", np.where(np.isfinite(dst), dst, -9999.0).astype("float32"),
                      "EPSG:4326", transform)
    back, info = raster_mod.read_dem(path)
    assert info["reprojected"] and info["nodata_detected"] == -9999.0
    valid = back.z[back.mask]
    assert valid.min() > np.nanmin(grid.z) - 1.0 and valid.max() < np.nanmax(grid.z) + 1.0
    assert 0.9 < back.valid_count * back.cell ** 2 / (grid.valid_count * grid.cell ** 2) < 1.1


def test_other_undeclared_fills_are_recognised(dem, tmp_path):
    grid, _ = dem
    # The float32 extreme a tool writes for "no value".
    _, info = raster_mod.read_dem(_undeclared(grid, tmp_path / "f32min.tif", np.finfo("float32").min))
    assert info["nodata_detected"] < -1e38 and info["nodata"] is None
    # The int16 minimum SRTM voids carry, in a converted float raster.
    _, info = raster_mod.read_dem(_undeclared(grid, tmp_path / "i16min.tif", -32768.0))
    assert info["nodata_detected"] == -32768.0
    # An unlisted value gives itself away: one exact value far below every
    # height, repeated over more than half a percent of the cells.
    back, info = raster_mod.read_dem(_undeclared(grid, tmp_path / "odd.tif", -1234.5))
    assert info["nodata_detected"] == -1234.5 and np.array_equal(back.mask, grid.mask)
    # A declared nodata is taken as declared: nothing is guessed on top of it.
    _, info = raster_mod.read_dem(raster_mod.write_geotiff(grid, tmp_path / "declared.tif"))
    assert info["nodata"] == -9999.0 and info["nodata_detected"] is None
    assert not any("declares no nodata" in n for n in info["notes"])
    # Every cell filled and nothing declared: still refused, naming the fill.
    with pytest.raises(ValueError, match="holds no elevation.*\\(-9999\\)"):
        raster_mod.read_dem(_write_raw(tmp_path / "all_fill.tif", np.full((40, 40), -9999.0, dtype="float32"),
                                       grid.crs, grid.transform()))


def test_preflight_warns_about_impossible_relief(dem, tmp_path):
    """A fill too rare to be recognised (two cells of -1234.5) stays in the
    heights; the first look then says the range is not a field's, instead
    of 'clean and complete'."""
    grid, _ = dem
    z = np.where(np.isfinite(grid.z), grid.z, np.nan).astype("float32")
    z[80, 80] = z[81, 81] = -1234.5
    path = raster_mod.write_geotiff(grid, tmp_path / "two_cells.tif", values=z)
    ds = registry.read_any(path)
    ds.ensure_derived()
    report = preflight.run(ds)
    assert ds.meta.extra["dem_info"]["nodata_detected"] is None
    assert report["verdict"] == "warning"
    finding = next(f for f in report["findings"] if f["title"] == "Elevation range")
    assert "more relief than any field has" in finding["detail"] and "Transparency" in finding["action"]
    assert report["info"]["elevation_range_m"] > 1000.0
    assert report["next_step"]["step"] == "terrain"

    clean = registry.read_any(raster_mod.write_geotiff(grid, tmp_path / "clean.tif"))
    clean.ensure_derived()
    report = preflight.run(clean)
    assert report["verdict"] == "ok" and 5.0 < report["info"]["elevation_range_m"] < 50.0

