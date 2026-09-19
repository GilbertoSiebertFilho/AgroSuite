"""A public elevation model fetched for a field whose file carries no heights.

No test here touches the network: the two sources are stood in for by
GeoTIFFs written on the spot — one that covers the field, one whose tile is
empty over it (as the 1 m LiDAR model is over many Prairie fields), and two
halves of a field split across a tile edge — through the ``readers`` hook
:func:`agrosuite.terrain.fetch.fetch_dem` takes for exactly this.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon, box

from agrosuite.core.dataset import Dataset, DatasetMeta
from agrosuite.terrain import fetch

# A made-up prairie field: about 700 m by 1 400 m.
FIELD = box(-113.861, 51.950, -113.851, 51.963)


def _boundary(polygons, name="trial-design-seed-2025"):
    centre = polygons[0].centroid
    return Dataset(pd.DataFrame({"lon": [centre.x], "lat": [centre.y], "value": [np.nan]}),
                   DatasetMeta(name=name, operation="boundary", geometry_type="polygon"),
                   geometry=polygons)


def _raster(path, west, south, east, north, *, cell_deg=0.0003, empty=False, rise_m=12.0):
    """A geographic GeoTIFF over the box: a plane rising to the east."""
    import rasterio
    from rasterio.transform import from_origin

    cols = int(round((east - west) / cell_deg))
    rows = int(round((north - south) / cell_deg))
    z = np.tile(np.linspace(1070.0, 1070.0 + rise_m, cols, dtype="float32"), (rows, 1))
    if empty:
        z[:] = -9999.0
    with rasterio.open(path, "w", driver="GTiff", width=cols, height=rows, count=1,
                       dtype="float32", crs="EPSG:4326", nodata=-9999.0,
                       transform=from_origin(west, north, cell_deg, cell_deg)) as dst:
        dst.write(z, 1)
    return str(path)


@pytest.fixture
def outline():
    shape, _ = fetch.field_outline(_boundary([FIELD]))
    return shape


# ==========================================================================
# The field's outline
# ==========================================================================

def test_a_trial_layout_s_plots_make_one_field_with_a_margin():
    """254 plots with alleys between them are one field: their hull, grown
    by the margin the slope at the edge needs."""
    plots = [box(-113.861 + i * 0.001, 51.950, -113.8605 + i * 0.001, 51.963) for i in range(10)]
    shape, area_ha = fetch.field_outline(_boundary(plots))
    assert shape.geom_type == "Polygon"
    assert shape.contains(Polygon(plots[0].exterior)) and shape.contains(plots[-1])
    bare, bare_ha = fetch.field_outline(_boundary(plots), margin_m=0)
    assert area_ha > bare_ha > 0
    assert shape.area > bare.area


def test_points_are_outlined_by_their_hull():
    df = pd.DataFrame({"lon": [-113.86, -113.85, -113.85, -113.86], "lat": [51.95, 51.95, 51.96, 51.96]})
    shape, area_ha = fetch.field_outline(Dataset(df, DatasetMeta(name="points")))
    assert 70 < area_ha < 120


def test_a_file_that_is_not_a_field_is_refused_in_words():
    with pytest.raises(fetch.FetchError, match="too few positions"):
        fetch.field_outline(Dataset(pd.DataFrame({"lon": [-113.8], "lat": [51.9]}),
                                    DatasetMeta(name="one point")))
    huge = box(-116.0, 50.0, -113.0, 53.0)
    with pytest.raises(fetch.FetchError, match="more than a field"):
        fetch.field_outline(_boundary([huge]))


# ==========================================================================
# Sources
# ==========================================================================

def test_copernicus_tiles_are_named_by_their_south_west_corner():
    assert fetch.copernicus_tiles(-113.86, 51.95, -113.85, 51.96) == [
        "Copernicus_DSM_COG_10_N51_00_W114_00_DEM"]
    assert fetch.copernicus_tiles(-51.3, -23.6, -51.1, -23.4) == [
        "Copernicus_DSM_COG_10_S24_00_W052_00_DEM"]
    assert fetch.copernicus_tiles(10.2, 45.9, 10.4, 46.1) == [
        "Copernicus_DSM_COG_10_N45_00_E010_00_DEM", "Copernicus_DSM_COG_10_N46_00_E010_00_DEM"]


def test_hrdem_assets_are_the_terrain_models_the_catalogue_lists():
    asked = []

    def catalogue(url):
        asked.append(url)
        return {"features": [
            {"assets": {"dtm": {"href": "https://example/a-dtm.tif"}, "dsm": {"href": "x"}}},
            {"assets": {"dsm": {"href": "no-terrain-here"}}},
        ]}

    assert fetch.hrdem_assets(-113.86, 51.95, -113.85, 51.96, get_json=catalogue) == [
        "https://example/a-dtm.tif"]
    assert "collections=hrdem-mosaic-1m" in asked[0] and "bbox=-113.86" in asked[0]


# ==========================================================================
# Fetching
# ==========================================================================

def test_lidar_that_covers_the_field_is_taken_first(tmp_path, outline):
    lidar = _raster(tmp_path / "lidar.tif", -113.87, 51.94, -113.84, 51.97, cell_deg=0.0001)
    world = _raster(tmp_path / "world.tif", -113.87, 51.94, -113.84, 51.97)
    got = fetch.fetch_dem(outline, cache_dir=tmp_path / "cache",
                          readers={"hrdem": lambda o: [lidar], "copernicus": lambda o: [world]})
    assert got.source == "hrdem" and got.coverage >= 0.99
    assert got.path.exists() and got.path.parent == tmp_path / "cache"
    assert "HRDEM" in got.notes[0] and "Open Government Licence" in got.notes[-1]


def test_an_empty_lidar_tile_falls_back_to_copernicus(tmp_path, outline):
    """The LiDAR catalogue lists a tile for the area, but it holds nothing
    over this field — the case of the first field it was tried on."""
    empty = _raster(tmp_path / "lidar.tif", -113.87, 51.94, -113.84, 51.97, empty=True)
    world = _raster(tmp_path / "world.tif", -113.87, 51.94, -113.84, 51.97)
    got = fetch.fetch_dem(outline, cache_dir=tmp_path / "cache",
                          readers={"hrdem": lambda o: [empty], "copernicus": lambda o: [world]})
    assert got.source == "copernicus"
    assert got.coverage >= 0.99
    assert any("coarse for a single field" in n for n in got.notes)
    assert "Copernicus DEM GLO-30" in got.notes[-1]


def test_a_catalogue_that_does_not_answer_falls_back_too(tmp_path, outline):
    def offline(o):
        raise OSError("no route to host")

    world = _raster(tmp_path / "world.tif", -113.87, 51.94, -113.84, 51.97)
    got = fetch.fetch_dem(outline, cache_dir=tmp_path / "cache",
                          readers={"hrdem": offline, "copernicus": lambda o: [world]})
    assert got.source == "copernicus"


def test_a_field_across_two_tiles_is_joined(tmp_path, outline):
    west = _raster(tmp_path / "west.tif", -113.87, 51.94, -113.856, 51.97)
    east = _raster(tmp_path / "east.tif", -113.856, 51.94, -113.84, 51.97)
    got = fetch.fetch_dem(outline, source="copernicus", cache_dir=tmp_path / "cache",
                          readers={"copernicus": lambda o: [west, east]})
    assert got.coverage >= 0.99


def test_the_second_ask_is_read_from_this_computer(tmp_path, outline):
    world = _raster(tmp_path / "world.tif", -113.87, 51.94, -113.84, 51.97)
    calls = []

    def copernicus(o):
        calls.append(1)
        return [world]

    readers = {"hrdem": lambda o: [], "copernicus": copernicus}
    first = fetch.fetch_dem(outline, cache_dir=tmp_path / "cache", readers=readers)
    again = fetch.fetch_dem(outline, cache_dir=tmp_path / "cache", readers=readers)
    assert not first.cached and again.cached and again.path == first.path
    assert len(calls) == 1


def test_the_raster_is_cut_to_the_field(tmp_path, outline):
    import rasterio

    world = _raster(tmp_path / "world.tif", -113.90, 51.92, -113.80, 52.00)
    got = fetch.fetch_dem(outline, source="copernicus", cache_dir=tmp_path / "cache",
                          readers={"copernicus": lambda o: [world]})
    with rasterio.open(got.path) as src:
        z = src.read(1, masked=True)
        left, bottom, right, top = src.bounds
    # A window around the field, not the whole tile, with the corners of
    # that window — outside the outline — left empty.
    assert right - left < 0.02 and top - bottom < 0.02
    assert np.ma.getmaskarray(z)[0, 0] and not np.ma.getmaskarray(z)[z.shape[0] // 2, z.shape[1] // 2]


def test_nothing_that_covers_the_field_says_what_was_tried(tmp_path, outline):
    empty = _raster(tmp_path / "empty.tif", -113.87, 51.94, -113.84, 51.97, empty=True)
    with pytest.raises(fetch.FetchError) as refused:
        fetch.fetch_dem(outline, cache_dir=tmp_path / "cache",
                        readers={"hrdem": lambda o: [empty], "copernicus": lambda o: [empty]})
    message = str(refused.value)
    assert "HRDEM 1 m" in message and "Copernicus 30 m" in message
    assert "internet connection" in message


def test_an_unknown_source_is_refused():
    with pytest.raises(fetch.FetchError, match="auto, hrdem or copernicus"):
        fetch.fetch_dem(FIELD, source="lidar-from-space")


# ==========================================================================
# Through the app
# ==========================================================================

@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from agrosuite.app import server as server_mod

    return TestClient(server_mod.app)


@pytest.fixture
def registered():
    """Registers datasets in the shared session and takes them out after:
    later tests export the whole session and must not meet these."""
    from agrosuite.app import server as server_mod

    ids = []

    def register(dataset, label):
        entry = server_mod._register(dataset, label, "upload")
        ids.append(entry["id"])
        return entry

    yield register, ids
    for dataset_id in ids:
        try:
            server_mod.state.remove(dataset_id)
        except KeyError:
            pass


def test_a_boundary_gets_its_elevation_and_keeps_its_role(client, tmp_path, monkeypatch, registered):
    """From a file without heights to the relief of its field in two calls,
    the way the Terrain tab does it — the boundary staying the boundary."""
    from agrosuite.app import server as server_mod

    world = _raster(tmp_path / "world.tif", -113.87, 51.94, -113.84, 51.97, rise_m=12.0)
    real_fetch = fetch.fetch_dem
    monkeypatch.setattr(fetch, "fetch_dem", lambda outline, source="auto": real_fetch(
        outline, source=source, cache_dir=tmp_path / "cache",
        readers={"hrdem": lambda o: [], "copernicus": lambda o: [world]}))

    register, ids = registered
    boundary = register(_boundary([FIELD]), "trial-design-seed-2025")
    response = client.post("/api/terrain/fetch-dem", json={"dataset_id": boundary["id"]})
    assert response.status_code == 200, response.text
    body = response.json()
    ids.append(body["dataset"]["id"])
    assert body["source"] == "copernicus" and body["source_label"] == "Copernicus 30 m"
    elevation = server_mod.state.get(body["dataset"]["id"])
    assert elevation.label == "Elevation · trial-design-seed-2025"
    assert elevation.dataset.meta.operation == "elevation"
    assert elevation.dataset.meta.extra["dem_fetched"]["for_id"] == boundary["id"]
    assert any("Copernicus DEM GLO-30" in n for n in elevation.dataset.meta.notes)
    assert server_mod.state.get(boundary["id"]).dataset.meta.operation == "boundary"

    relief = client.post("/api/terrain/analyze", json={"dataset_id": body["dataset"]["id"]})
    assert relief.status_code == 200, relief.text
    # The plane rises 12 m across the 0.03° raster; the field and its margin
    # take about 0.0115° of that: some 4.5 m of relief.
    heights = relief.json()["summary"]["elevation"]
    assert heights["max_m"] - heights["min_m"] == pytest.approx(4.5, abs=1.0)


def test_a_file_with_no_position_is_refused_in_words(client, registered):
    """A log whose GPS never found a fix: the columns are there, empty."""
    register, _ = registered
    nowhere = Dataset(pd.DataFrame({"lon": [np.nan] * 3, "lat": [np.nan] * 3, "value": [1.0, 2.0, 3.0]}),
                      DatasetMeta(name="no fix"))
    entry = register(nowhere, "A log with no GPS fix")
    response = client.post("/api/terrain/fetch-dem", json={"dataset_id": entry["id"]})
    assert response.status_code == 400
    assert "too few positions" in response.json()["detail"]


def test_the_interface_files_are_revalidated_every_time(client):
    """After an update the browser must not run yesterday's script."""
    response = client.get("/static/app.js")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers.get("etag")
