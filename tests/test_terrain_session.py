"""The terrain feature inside the rest of the session.

An elevation layer is a new kind of dataset, and the steps built for machine
tracks — cleaning, the economic fit — must refuse it in a sentence rather than
run over a raster's scan order. The QGIS export and the monitor package
also come up on the way from a relief to the cab, and both had an older
habit that the terrain round trip exposed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fixtures  # noqa: E402  (tests/fixtures.py)
from agrosuite.core.dataset import Dataset, DatasetMeta  # noqa: E402
from agrosuite.formats import qgis, raster  # noqa: E402
from agrosuite.terrain.synthetic import synthetic_dem  # noqa: E402


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from agrosuite.app import server as server_mod

    return TestClient(server_mod.app)


@pytest.fixture(scope="module")
def dem_dataset(client, tmp_path_factory):
    grid, _ = synthetic_dem()
    path = raster.write_geotiff(grid, tmp_path_factory.mktemp("dem") / "field_dem.tif", dtype="float64")
    loaded = client.post("/api/import/path", json={"path": str(path)})
    assert loaded.status_code == 200, loaded.text
    return loaded.json()


# ==========================================================================
# Steps built for tracks stop at an elevation layer
# ==========================================================================

def test_dem_dataset_has_no_track_columns(client, dem_dataset):
    from agrosuite.app import server as server_mod

    columns = set(server_mod.state.get(dem_dataset["id"]).dataset.df.columns)
    assert not {"distance_m", "heading_deg", "pass_id", "speed_kmh"} & columns


def test_cleaning_refuses_an_elevation_layer(client, dem_dataset):
    before = len(client.get("/api/datasets").json()["datasets"])
    response = client.post(f"/api/datasets/{dem_dataset['id']}/clean", json={})
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "elevation layer" in detail and "/api/terrain/analyze" in detail
    # Nothing registered: no 'clean' or 'removed' copy of a raster.
    assert len(client.get("/api/datasets").json()["datasets"]) == before


def test_difm_refuses_an_elevation_layer(client, dem_dataset):
    response = client.post(f"/api/datasets/{dem_dataset['id']}/difm", json={})
    assert response.status_code == 400
    assert "elevation layer" in response.json()["detail"]


def test_terrain_zones_are_refused_by_the_cleaning_too(client, dem_dataset):
    analysed = client.post("/api/terrain/analyze", json={"dataset_id": dem_dataset["id"]})
    assert analysed.status_code == 200, analysed.text
    zones = client.post(f"/api/terrain/{dem_dataset['id']}/zones", json={"by": "slope_class"}).json()
    response = client.post(f"/api/datasets/{zones['dataset']['id']}/clean", json={})
    assert response.status_code == 400
    assert "elevation layer" in response.json()["detail"]


def test_a_harvest_track_still_cleans(client, tmp_path):
    shp = fixtures.john_deere_shapefile(tmp_path / "jd")
    loaded = client.post("/api/import/path", json={"path": str(shp)}).json()
    response = client.post(f"/api/datasets/{loaded['id']}/clean", json={})
    assert response.status_code == 200, response.text
    assert response.json()["report"]["totals"]["kept"] > 0


# ==========================================================================
# GeoPackage: a source column that differs from a canonical one only by case
# ==========================================================================

def _dataset_with_case_twins() -> Dataset:
    rng = np.random.default_rng(3)
    n = 30
    df = pd.DataFrame({
        "lon": -113.55 + rng.normal(0, 1e-3, n),
        "lat": 51.75 + rng.normal(0, 1e-3, n),
        "value": rng.uniform(2000, 3000, n),
        "product": "Canola",
        "Product": "Canola",
    })
    return Dataset(df, DatasetMeta(name="twins", operation="harvest"))


def test_geopackage_keeps_columns_that_differ_only_by_case(tmp_path):
    """SQLite folds the case of column names; the second column was making
    GDAL refuse the layer and the whole export with it."""
    path = tmp_path / "twins.gpkg"
    info = qgis.write_geopackage({"twins": _dataset_with_case_twins()}, path)
    assert info["layers"][0]["renamed"] == {"Product": "Product_2"}
    back = gpd.read_file(path, layer="twins")
    assert {"product", "Product_2"} <= set(back.columns)
    assert (back["Product_2"] == "Canola").all()


def test_qgis_export_works_with_a_john_deere_file_loaded(client, tmp_path):
    shp = fixtures.john_deere_shapefile(tmp_path / "jd")
    loaded = client.post("/api/import/path", json={"path": str(shp)}).json()
    response = client.post("/api/qgis/export", json={})
    assert response.status_code == 200, response.text
    layers = {layer["layer"]: layer for layer in response.json()["geopackage"]["layers"]}
    jd = next(layer for name, layer in layers.items() if name.startswith("JD_Colheita"))
    assert jd["features"] == loaded["rows"]
    assert jd["renamed"] == {"Product": "Product_2"}


# ==========================================================================
# The monitor package speaks English
# ==========================================================================

def test_package_response_and_files_are_named_in_english(client, dem_dataset):
    west, south, east, north = dem_dataset["bounds"]
    ring = [[west, south], [east, south], [east, north], [west, north], [west, south]]
    design = client.post("/api/design", json={
        "boundary": ring, "rates": [50, 100, 150], "implement_width_m": 18.29,
    })
    assert design.status_code == 200, design.text
    response = client.post("/api/export/package", json={
        "monitor": "generic", "features": design.json()["features"], "boundary": ring,
        "field_name": "North quarter",
    })
    assert response.status_code == 200, response.text
    body = response.json()
    assert "notes" in body and "observacoes" not in body
    assert Path(body["folder"]).name.startswith("package_")
    paths = [item["path"] for item in body["contents"]]
    boundary = [p for p in paths if "boundary" in p.lower()]
    assert boundary, paths
    assert not any("contorno" in p.lower() or "pacote" in p.lower() for p in paths)


# ==========================================================================
# A polygon layer's area is its polygons, not a swath
# ==========================================================================

def test_polygon_dataset_area_is_the_polygons_area():
    """A boundary or a zone polygon has no swath and no advance; its area
    is the ground the polygons cover, measured in the metric CRS."""
    from shapely.geometry import Polygon

    from agrosuite.core import preflight

    # Two 200 m x 100 m rectangles near Calgary: 2 ha each.
    lon0, lat0 = -113.55, 51.75
    dlon = 200.0 / (111_320.0 * np.cos(np.radians(lat0)))
    dlat = 100.0 / 111_320.0
    rects = []
    for k in range(2):
        west = lon0 + k * 2 * dlon
        rects.append(Polygon([
            (west, lat0), (west + dlon, lat0), (west + dlon, lat0 + dlat), (west, lat0 + dlat),
        ]))
    df = pd.DataFrame({
        "lon": [r.representative_point().x for r in rects],
        "lat": [r.representative_point().y for r in rects],
        "value": [1.0, 2.0],
    })
    ds = Dataset(df, DatasetMeta(name="rects", operation="prescription", geometry_type="polygon"),
                 geometry=rects)
    assert ds.area_ha() == pytest.approx(4.0, rel=0.01)
    assert ds.summary()["area_ha"] == pytest.approx(4.0, abs=0.05)
    # Never projected: the area is still measured in metres, not degrees.
    ds.metric_crs = None
    assert ds.area_ha() == pytest.approx(4.0, rel=0.01)

    report = preflight.run(ds)
    coverage = [f for f in report["findings"] if f["title"] == "Coverage"]
    assert coverage and coverage[0]["detail"] == "2 polygons covering 4.0 ha."
    assert not any(f["title"] == "Sparse data" for f in report["findings"])
