"""Tests for reading and writing elevation rasters.

A DEM is the one input the terrain analyser takes that is not a monitor
file, and it arrives in whatever shape the elevation service chose:
geographic degrees, feet, Web Mercator, a nodata marker, or no CRS at all.
These tests write small rasters in each of those shapes and check that
``read_dem`` turns every one into the same thing — an ElevationGrid in
ground metres whose plane slope is the one that was written — and that a
grid written by ``write_geotiff`` reads back cell for cell. The rest checks
that a GeoTIFF is understood by the registry, the preflight and the HTTP
import like any other file.
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agrosuite.core import preflight
from agrosuite.core import schema as sch
from agrosuite.core.dataset import Dataset, DatasetMeta, OPERATION_LABELS
from agrosuite.formats import raster as raster_mod
from agrosuite.formats import registry
from agrosuite.terrain import ElevationGrid, synthetic_dem

#: A plane in the field's UTM frame: z = 700 + GX (x - xc) + GY (y - yc).
GX, GY = 0.010, 0.005


def _write_raw(path: Path, data: np.ndarray, crs, transform: Affine, nodata=None, dtype=None) -> Path:
    """Write a raster the way a third-party tool would: no help from the app."""
    dtype = dtype or data.dtype.name
    with rasterio.open(
        path, "w", driver="GTiff", height=data.shape[0], width=data.shape[1], count=1,
        dtype=dtype, crs=crs, transform=transform, nodata=nodata,
    ) as dst:
        dst.write(data.astype(dtype), 1)
    return path


def _plane_raster(path: Path, crs: str, cell: float, lon0: float, lat0: float,
                  ncols: int, nrows: int, grid: ElevationGrid, centre) -> Path:
    """A plane sampled on a grid of ``crs`` whose cell is ``cell`` CRS units,
    with the plane defined in the reference grid's metric frame."""
    from pyproj import Transformer

    to_crs = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    ox, oy = to_crs.transform(lon0, lat0)
    xs = ox + (np.arange(ncols) + 0.5) * cell
    ys = oy + nrows * cell - (np.arange(nrows) + 0.5) * cell
    CX, CY = np.meshgrid(xs, ys)
    lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(CX, CY)
    X, Y = grid.from_lonlat(lon, lat)
    xc, yc = centre
    Z = 700.0 + GX * (X - xc) + GY * (Y - yc)
    return _write_raw(path, Z.astype("float32"), crs, Affine(cell, 0, ox, 0, -cell, oy + nrows * cell),
                      nodata=-9999.0)


def _fit_plane(grid: ElevationGrid, centre) -> tuple[float, float, float]:
    """Least-squares gradient of the grid and the residual RMSE."""
    X, Y = grid.xy()
    m = grid.mask
    xc, yc = centre
    A = np.c_[X[m] - xc, Y[m] - yc, np.ones(int(m.sum()))]
    coef, *_ = np.linalg.lstsq(A, grid.z[m], rcond=None)
    rmse = float(np.std(grid.z[m] - A @ coef))
    return float(coef[0]), float(coef[1]), rmse


@pytest.fixture(scope="module")
def dem():
    return synthetic_dem()


@pytest.fixture(scope="module")
def dem_tif(dem, tmp_path_factory) -> Path:
    grid, _ = dem
    return raster_mod.write_geotiff(grid, tmp_path_factory.mktemp("dem") / "field_dem.tif",
                                    dtype="float64", tags={"source": "synthetic"})


# ==========================================================================
# Writing and reading back
# ==========================================================================

def test_float64_geotiff_round_trips_exactly(dem, dem_tif):
    grid, _ = dem
    back, info = raster_mod.read_dem(dem_tif)
    assert np.array_equal(back.z, grid.z, equal_nan=True)
    assert (back.x0, back.y0, back.cell, back.crs) == (grid.x0, grid.y0, grid.cell, grid.crs)
    assert np.isnan(back.z[0, 0])          # the notch stays NaN through nodata
    assert info["reprojected"] is False and info["resampled"] is False and info["coarsened"] is False
    assert info["crs_in"] == "EPSG:32612" and info["cell_in_m"] == 5.0
    assert info["nodata"] == -9999.0 and info["path"] == str(dem_tif)
    json.dumps(info)

    with rasterio.open(dem_tif) as src:
        assert src.tags()["source"] == "synthetic"
        assert src.profile["compress"] == "lzw" and src.profile["tiled"] is True
        assert src.nodata == -9999.0
        assert src.transform == grid.transform()


def test_float32_default_keeps_millimetres(dem, tmp_path):
    grid, _ = dem
    path = raster_mod.write_geotiff(grid, tmp_path / "dem32.tif")
    back, _ = raster_mod.read_dem(path)
    assert np.nanmax(np.abs(back.z - grid.z)) < 1e-3
    assert np.array_equal(np.isnan(back.z), np.isnan(grid.z))


def test_write_geotiff_refuses_mismatched_values(dem, tmp_path):
    grid, _ = dem
    with pytest.raises(ValueError, match="do not fit"):
        raster_mod.write_geotiff(grid, tmp_path / "bad.tif", values=np.zeros((3, 3)))
    with pytest.raises(ValueError, match="write_geotiff_int"):
        raster_mod.write_geotiff(grid, tmp_path / "bad.tif", dtype="uint8")


def test_write_geotiff_int_writes_classes_with_labels(dem, tmp_path):
    grid, _ = dem
    classes = np.where(grid.mask, (grid.z > np.nanmedian(grid.z)).astype(float) + 1, np.nan)
    path = raster_mod.write_geotiff_int(
        grid, tmp_path / "classes.tif", classes, labels={1: "Low ground", 2: "High ground"},
        tags={"layer": "landform"},
    )
    with rasterio.open(path) as src:
        data = src.read(1)
        assert src.dtypes[0] == "uint8" and src.nodata == 255
        assert set(np.unique(data)) == {1, 2, 255}
        assert (data == 255).sum() == int((~grid.mask).sum())
        tags = src.tags()
        assert tags["class_1"] == "Low ground" and tags["class_2"] == "High ground"
        assert tags["layer"] == "landform"

    with pytest.raises(ValueError, match="nodata"):
        raster_mod.write_geotiff_int(grid, tmp_path / "bad.tif", np.full(grid.shape, 255.0))
    with pytest.raises(ValueError, match="wider dtype"):
        raster_mod.write_geotiff_int(grid, tmp_path / "bad.tif", np.full(grid.shape, 300.0))


# ==========================================================================
# Rasters that are not yet what the terrain modules need
# ==========================================================================

def test_geographic_dem_is_reprojected_to_a_metric_plane(dem, tmp_path):
    grid, truth = dem
    # 1e-4 degree cells: 6.9 m by 11.1 m at 51.7 N — a rectangle on the ground.
    path = _plane_raster(tmp_path / "plane4326.tif", "EPSG:4326", 1e-4, -113.56, 51.744,
                         180, 120, grid, truth["centre_m"])
    back, info = raster_mod.read_dem(path)
    assert back.crs == "EPSG:32612" and info["reprojected"] is True and info["crs_in"] == "EPSG:4326"
    assert back.cell == 9.0                       # sqrt(6.9 * 11.1) = 8.8, rounded to a figure a person would pick
    assert 6.5 < info["cell_in_xy_m"][0] < 7.2 and 11.0 < info["cell_in_xy_m"][1] < 11.3
    assert back.x0 % back.cell == 0 and back.y0 % back.cell == 0
    gx, gy, rmse = _fit_plane(back, truth["centre_m"])
    assert abs(gx - GX) < 0.02 * GX and abs(gy - GY) < 0.02 * GY
    assert rmse < 0.02
    assert any("Reprojected" in note for note in info["notes"])
    assert 0.55 < back.valid_count / back.z.size   # a rotated footprint, not a hole


@pytest.mark.parametrize("crs, cell, reason", [
    ("EPSG:2223", 30.0, "foot"),       # Arizona Central, US survey feet
    ("EPSG:3857", 15.0, "Mercator"),   # Web Mercator "metres"
])
def test_non_metre_projections_are_reprojected(dem, tmp_path, crs, cell, reason):
    grid, truth = dem
    path = _plane_raster(tmp_path / "plane.tif", crs, cell, -113.56, 51.744, 100, 80,
                         grid, truth["centre_m"])
    back, info = raster_mod.read_dem(path)
    assert info["reprojected"] is True and back.crs == "EPSG:32612"
    assert reason in info["notes"][0]
    gx, gy, _ = _fit_plane(back, truth["centre_m"])
    assert abs(gx - GX) < 0.02 * GX and abs(gy - GY) < 0.02 * GY
    # The native cell is reported in ground metres, not in the CRS's unit.
    expected = cell * 0.3048 if reason == "foot" else cell * np.cos(np.radians(51.75))
    assert abs(info["cell_in_m"] - expected) < 0.05 * expected


def test_nodata_cells_become_nan(tmp_path):
    data = np.full((30, 40), 500, dtype="int16")
    data[5:10, 5:10] = -32768
    path = _write_raw(tmp_path / "srtm.tif", data, "EPSG:32612",
                      Affine(10, 0, 300000, 0, -10, 5740000), nodata=-32768)
    back, info = raster_mod.read_dem(path)
    assert back.z.dtype == np.float64
    assert int(np.isnan(back.z).sum()) == 25 and np.isnan(back.z[7, 7])
    assert back.z[0, 0] == 500.0 and info["nodata"] == -32768.0

    # The same hole survives a resampling, without smearing into its neighbours.
    coarse, _ = raster_mod.read_dem(path, target_cell_m=20)
    assert coarse.cell == 20.0
    assert np.nanmin(coarse.z) == 500.0 and np.nanmax(coarse.z) == 500.0


def test_raster_without_crs_names_the_fix(tmp_path):
    path = _write_raw(tmp_path / "nocrs.tif", np.ones((10, 10), "float32"), None,
                      Affine(10, 0, 300000, 0, -10, 5740000))
    with pytest.raises(ValueError, match="QGIS"):
        raster_mod.read_dem(path)
    assert raster_mod.describe(path)["crs"] is None


def test_not_a_raster_is_refused_with_a_hint(tmp_path):
    path = tmp_path / "notes.tif"
    path.write_text("this is not a GeoTIFF")
    with pytest.raises(ValueError, match="could not be opened as a raster"):
        raster_mod.read_dem(path)


def test_target_cell_resamples_a_metric_raster(dem, dem_tif):
    grid, truth = dem
    back, info = raster_mod.read_dem(dem_tif, target_cell_m=10)
    assert back.cell == 10.0 and info["resampled"] is True and info["coarsened"] is False
    assert back.crs == grid.crs
    X, Y = back.xy()
    m = back.mask
    err = back.z[m] - truth["surface"](X[m], Y[m])
    assert np.sqrt(np.mean(err ** 2)) < 0.05
    with pytest.raises(ValueError, match="positive"):
        raster_mod.read_dem(dem_tif, target_cell_m=0)


def test_oversized_raster_is_coarsened_with_a_note(dem_tif, monkeypatch):
    monkeypatch.setattr(raster_mod, "MAX_CELLS", 5_000)
    back, info = raster_mod.read_dem(dem_tif)
    assert back.rows * back.cols <= 5_000
    assert info["coarsened"] is True and back.cell == 15.0
    assert any("coarsened" in note for note in info["notes"])
    assert 690 < np.nanmean(back.z) < 710


def test_describe_reads_the_header_only(dem_tif):
    info = raster_mod.describe(dem_tif)
    assert info["rows"] == 160 and info["cols"] == 160 and info["cell_m"] == 5.0
    assert info["crs"] == "EPSG:32612" and info["needs_reprojection"] is False
    assert "160 × 160" in info["summary"] and "EPSG:32612" in info["summary"]
    json.dumps(info)


# ==========================================================================
# Into the app
# ==========================================================================

def test_dataset_from_dem_respects_max_points(dem, dem_tif):
    grid, _ = dem
    _, info = raster_mod.read_dem(dem_tif)
    ds = raster_mod.dataset_from_dem(grid, info, max_points=5_000)
    assert 0 < len(ds) <= 5_000
    assert (ds.df[sch.VALUE] == ds.df[sch.ELEVATION]).all()
    assert ds.df[sch.ELEVATION].notna().all()
    assert ds.meta.operation == "elevation" and ds.meta.source_format == "geotiff"
    assert ds.meta.value_label == "Elevation" and ds.meta.value_unit == "m"
    assert ds.meta.geometry_type == "point" and ds.meta.brand == "generic"
    extra = ds.meta.extra
    assert extra["dem_path"] == str(dem_tif) and extra["dem_cell_m"] == 5.0
    assert extra["dem_full_cells"] == grid.valid_count and extra["dem_stride"] == 3
    assert ds.metric_crs == grid.crs
    # The sample sits exactly on cell centres of the grid it came from.
    x = ds.df[sch.X].to_numpy()
    y = ds.df[sch.Y].to_numpy()
    assert np.allclose(grid.sample(x, y), ds.df[sch.ELEVATION].to_numpy(), atol=1e-6)
    assert any("Every 3rd cell" in note for note in ds.meta.notes)
    json.dumps(ds.summary())

    full = raster_mod.dataset_from_dem(grid, None)
    assert len(full) == grid.valid_count and full.meta.name == "elevation"


def test_registry_detects_and_reads_a_geotiff(dem_tif):
    assert ".tif" in registry.ALL_IMPORT_EXT and ".tiff" in registry.ALL_IMPORT_EXT
    source = registry.detect(dem_tif)
    assert source.kind == "raster" and "GeoTIFF" in source.label

    ds = registry.read_any(dem_tif)
    assert isinstance(ds, Dataset)
    assert ds.meta.operation == "elevation"
    assert ds.meta.extra["dem_path"] == str(dem_tif)
    assert ds.meta.to_dict()["operation_label"] == "Elevation (DEM)"

    info = registry.inspect(dem_tif)
    assert info["kind"] == "raster" and info["raster"]["cell_m"] == 5.0
    assert "EPSG:32612" in info["detail"]


_QUARTER = [(-105.8340, 50.4520), (-105.8229, 50.4520), (-105.8229, 50.4592), (-105.8340, 50.4592)]


def test_inspect_reads_the_taskdata_inside_an_isoxml_folder(tmp_path):
    """detect() hands back the folder; the inspector used to pass it straight
    to the XML parser and report 'Is a directory'."""
    from fastapi.testclient import TestClient

    from agrosuite.app import server as server_mod
    from agrosuite.formats import isoxml

    isoxml.write_field_setup(tmp_path, field_name="NW-14-32-W2", boundary=_QUARTER)
    for path in (tmp_path, tmp_path / "TASKDATA", tmp_path / "TASKDATA" / "TASKDATA.XML"):
        info = registry.inspect(path)
        assert info["kind"] == "isoxml", path
        assert "unreadable" not in info["detail"], info["detail"]
        assert info["isoxml"]["fields"] == ["NW-14-32-W2"]
        assert info["isoxml"]["version"]

    inspected = TestClient(server_mod.app).get("/api/inspect", params={"path": str(tmp_path)})
    assert inspected.status_code == 200, inspected.text
    assert inspected.json()["isoxml"]["fields"] == ["NW-14-32-W2"]

    (tmp_path / "TASKDATA" / "TASKDATA.XML").write_text("<not xml")
    assert "unreadable" in registry.inspect(tmp_path)["detail"]


def test_isoxml_default_field_designator_is_english(tmp_path):
    from agrosuite.formats import isoxml

    out = isoxml.write_field_setup(tmp_path / "setup", boundary=_QUARTER)
    assert isoxml.read_field_setup(out / "TASKDATA.XML")["fields"][0]["name"] == "Field"

    isoxml.write_prescription(tmp_path / "rx", np.full((3, 3), 100.0), -105.834, 50.452, 0.0001, 0.0001)
    catalog = isoxml.parse_taskdata(tmp_path / "rx" / "TASKDATA" / "TASKDATA.XML")
    assert [f["name"] for f in catalog["fields"]] == ["Field"]
    assert "Talhao" not in (tmp_path / "rx" / "TASKDATA" / "TASKDATA.XML").read_text()


def test_zipped_geotiff_is_found_inside_the_archive(dem_tif, tmp_path):
    archive = tmp_path / "dem.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(dem_tif, dem_tif.name)
    ds = registry.read_any(archive)
    assert ds.meta.operation == "elevation"
    assert any("Extracted from dem.zip" in note for note in ds.meta.notes)


def test_preflight_accepts_an_elevation_layer(dem_tif):
    ds = registry.read_any(dem_tif)
    ds.ensure_derived()
    report = preflight.run(ds)
    assert report["verdict"] == "ok"
    assert report["suggested_role"] == "terrain"
    assert report["role_label"] == "Elevation / terrain"
    assert report["next_step"]["step"] == "terrain"
    titles = [f["title"] for f in report["findings"]]
    assert not any(t.startswith("Missing") or t == "No timestamp" for t in titles)
    assert all(f["level"] == "ok" for f in report["findings"])
    text = " ".join(f["detail"] for f in report["findings"])
    assert "terrain analyser" in text and "60.0 ha" in text
    json.dumps(report)

    assert list(preflight.ROLES).index("terrain") < list(preflight.ROLES).index("other")
    assert preflight.ROLE_BY_OPERATION["elevation"] == "terrain"
    assert OPERATION_LABELS["elevation"] == "Elevation (DEM)"
    assert DatasetMeta().brand_label == "Unknown"


def test_http_import_of_a_geotiff(dem_tif):
    from fastapi.testclient import TestClient

    from agrosuite.app import server as server_mod

    client = TestClient(server_mod.app)
    response = client.post("/api/import/path", json={"path": str(dem_tif)})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["meta"]["operation"] == "elevation"
    assert body["preflight"]["suggested_role"] == "terrain"
    assert body["preflight"]["verdict"] == "ok"
    assert body["meta"]["extra"]["dem_path"] == str(dem_tif)

    inspected = client.get("/api/inspect", params={"path": str(dem_tif)})
    assert inspected.status_code == 200 and inspected.json()["kind"] == "raster"

    catalogue = client.get("/api/catalog").json()
    assert ".tif" in catalogue["import_extensions"]
    assert catalogue["roles"]["terrain"] == "Elevation / terrain"
