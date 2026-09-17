"""HTTP tests for the terrain routes.

They drive the app the way the interface does — load the demo, analyse,
fetch the summary, the layer images, the contours, a profile, make zones,
export — through :class:`fastapi.testclient.TestClient`, and check the
shapes the interface relies on: the summary keys of the contract, PNGs that
decode with an alpha channel and sit over the field, contours and features
inside the field, zone datasets that the existing export route can write,
and errors that say what to do.
"""

from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agrosuite.app.routes import terrain as terrain_routes
from agrosuite.formats import raster
from agrosuite.terrain.analysis import LAYER_SPECS, ZONE_KINDS
from agrosuite.terrain.grid import ElevationGrid
from agrosuite.terrain.synthetic import synthetic_terrain

SUMMARY_KEYS = {
    "source", "grid", "elevation", "slope", "aspect", "trend", "character",
    "landforms", "features", "wetness", "contours", "layers", "findings", "options",
}
LAYER_KEYS = [spec[0] for spec in LAYER_SPECS]


# ==========================================================================
# Fixtures
# ==========================================================================

@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from agrosuite.app import server as server_mod

    return TestClient(server_mod.app)


@pytest.fixture(scope="module")
def demo(client):
    """The terrain demo loaded and analysed once: ``(dataset_id, summary)``."""
    loaded = client.post("/api/terrain/demo")
    assert loaded.status_code == 200, loaded.text
    dataset_id = loaded.json()["id"]
    analysed = client.post("/api/terrain/analyze", json={"dataset_id": dataset_id})
    assert analysed.status_code == 200, analysed.text
    return dataset_id, analysed.json()["summary"]


def _inside(lon: float, lat: float, bounds: list[float], tol: float = 1e-5) -> bool:
    west, south, east, north = bounds
    return west - tol <= lon <= east + tol and south - tol <= lat <= north + tol


def _coordinates(geometry: dict):
    """Every (lon, lat) pair of a GeoJSON geometry, whatever its nesting."""
    def walk(coords):
        if isinstance(coords[0], (int, float)):
            yield coords
        else:
            for part in coords:
                yield from walk(part)
    yield from walk(geometry["coordinates"])


# ==========================================================================
# Demo, analysis, summary
# ==========================================================================

def test_demo_registers_a_dataset_with_its_truth(client):
    response = client.post("/api/terrain/demo")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["label"] == "Terrain demo"
    assert body["origin"] == "demo"
    truth = body["meta"]["extra"]["terrain_truth"]
    assert len(truth["hills"]) == 2
    for hill in truth["hills"]:
        lon, lat = hill["summit_lonlat"]
        assert _inside(lon, lat, body["bounds"])
    assert truth["depression"]["bottom_lonlat"]
    assert truth["valley"]["start_lonlat"] and truth["valley"]["end_lonlat"]
    # The whole entry must survive JSON: no numpy, no callables.
    json.dumps(body)
    listed = {d["id"] for d in client.get("/api/datasets").json()["datasets"]}
    assert body["id"] in listed


def test_analysis_summary_follows_the_contract(client, demo):
    dataset_id, summary = demo
    assert set(summary) == SUMMARY_KEYS
    assert summary["source"]["kind"] == "points"
    assert summary["source"]["points_used"] > 0
    assert summary["grid"]["rows"] > 0 and summary["grid"]["cell_m"] > 0
    assert len(summary["grid"]["bounds_lonlat"]) == 4
    assert [layer["key"] for layer in summary["layers"]] == LAYER_KEYS
    assert summary["character"]["key"] in ("flat", "gently_undulating", "rolling", "hilly")
    assert len(summary["features"]["hills"]) == 2
    assert summary["findings"] and all(
        f["level"] in ("ok", "info", "warning") and f["text"] for f in summary["findings"]
    )
    # The resolved options are echoed: the interface shows what was used.
    assert summary["options"]["cell_m"] == summary["grid"]["cell_m"]
    assert summary["options"]["tpi_large_m"] > summary["options"]["tpi_small_m"]

    fetched = client.get(f"/api/terrain/{dataset_id}")
    assert fetched.status_code == 200
    assert fetched.json() == {"dataset_id": dataset_id, "summary": summary}

    # The summary also lives with the dataset, where a saved report reads it.
    entry = client.get(f"/api/datasets/{dataset_id}").json()
    assert entry["reports"]["terrain"] is True
    report = client.get(f"/api/datasets/{dataset_id}/report/terrain")
    assert report.status_code == 200 and report.json()["grid"] == summary["grid"]


def test_get_before_analysis_says_to_run_it(client):
    loaded = client.post("/api/import/demo", json={"kind": "harvest"}).json()
    response = client.get(f"/api/terrain/{loaded['id']}")
    assert response.status_code == 404
    assert "Run the terrain analysis first" in response.json()["detail"]
    for path in ("layers", "layer/elevation.png", "contours", "features"):
        assert client.get(f"/api/terrain/{loaded['id']}/{path}").status_code == 404
    assert client.post(
        f"/api/terrain/{loaded['id']}/profile", json={"points": [[0, 0], [1, 1]]}
    ).status_code == 404
    assert client.post(
        f"/api/terrain/{loaded['id']}/zones", json={"by": "landform"}
    ).status_code == 404
    assert client.post(f"/api/terrain/{loaded['id']}/export", json={}).status_code == 404


def test_unknown_dataset_is_404_everywhere(client):
    assert client.get("/api/terrain/no-such-id").status_code == 404
    assert client.get("/api/terrain/no-such-id/layers").status_code == 404
    response = client.post("/api/terrain/analyze", json={"dataset_id": "no-such-id"})
    assert response.status_code == 404
    assert "no-such-id" in response.json()["detail"]


@pytest.mark.parametrize("method, route, body", [
    ("get", "", None),
    ("get", "/layers", None),
    ("get", "/layer/elevation.png", None),
    ("get", "/contours", None),
    ("get", "/features", None),
    ("post", "/profile", {"points": [[0, 0], [1, 1]]}),
    ("post", "/zones", {"by": "landform"}),
    ("post", "/export", {}),
])
def test_missing_dataset_message_is_a_sentence_not_a_quoted_string(client, method, route, body):
    """str(KeyError) wraps its argument in quotation marks; the interface
    would print the 404 with them. The detail is the plain sentence."""
    url = f"/api/terrain/no-such-id{route}"
    response = client.get(url) if method == "get" else client.post(url, json=body)
    assert response.status_code == 404
    detail = response.json()["detail"]
    assert detail == "Dataset 'no-such-id' is not loaded in this session."
    assert not detail.startswith('"')


def test_harvest_demo_has_no_elevation_and_says_so(client):
    """The harvest demo carries no GPS altitude, so the analysis refuses it
    with the message that says what file to open instead."""
    loaded = client.post("/api/import/demo", json={"kind": "harvest"}).json()
    response = client.post("/api/terrain/analyze", json={"dataset_id": loaded["id"]})
    assert response.status_code == 400
    assert "no usable elevation" in response.json()["detail"]
    assert client.get(f"/api/terrain/{loaded['id']}").status_code == 404


def test_unknown_option_is_refused(client, demo):
    dataset_id, _ = demo
    response = client.post("/api/terrain/analyze", json={"dataset_id": dataset_id, "cell": 5})
    assert response.status_code == 422
    bad = client.post("/api/terrain/analyze", json={"dataset_id": dataset_id, "cell_m": -1})
    assert bad.status_code == 400
    assert "positive" in bad.json()["detail"]


# ==========================================================================
# Layers
# ==========================================================================

def test_layers_list_carries_urls_bounds_and_legends(client, demo):
    dataset_id, summary = demo
    response = client.get(f"/api/terrain/{dataset_id}/layers")
    assert response.status_code == 200
    layers = response.json()["layers"]
    assert [layer["key"] for layer in layers] == LAYER_KEYS
    dataset_bounds = client.get(f"/api/datasets/{dataset_id}").json()["bounds"]
    west, south, east, north = dataset_bounds
    for layer in layers:
        for key in ("key", "label", "kind", "unit", "min", "max", "palette", "url", "bounds", "legend"):
            assert key in layer, layer["key"]
        assert layer["url"] == f"/api/terrain/{dataset_id}/layer/{layer['key']}.png"
        (b_south, b_west), (b_north, b_east) = layer["bounds"]
        assert b_west <= west and b_east >= east and b_south <= south and b_north >= north
        legend = layer["legend"]
        if layer["key"] == "landform":
            assert layer["kind"] == "categorical" and legend["kind"] == "categorical"
            assert [c["code"] for c in legend["classes"]] == [1, 2, 3, 4, 5, 6]
            assert legend["classes"][0]["label"] == "Hilltop / ridge"
        else:
            assert legend["kind"] == "continuous" and len(legend["stops"]) == 7
            assert legend["vmax"] > legend["vmin"]
    by_key = {layer["key"]: layer for layer in layers}
    assert by_key["hillshade"]["legend"]["vmin"] == 0.0
    assert by_key["hillshade"]["legend"]["vmax"] == 1.0
    assert by_key["depression_depth"]["legend"]["vmin"] == 0.0
    assert by_key["aspect_deg"]["legend"]["vmax"] == 360.0


@pytest.mark.parametrize("key", LAYER_KEYS)
def test_every_layer_png_decodes_with_alpha(client, demo, key):
    from PIL import Image

    dataset_id, summary = demo
    response = client.get(f"/api/terrain/{dataset_id}/layer/{key}.png", params={"hillshade": 1})
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "image/png"
    assert response.headers["cache-control"] == "no-store"
    image = Image.open(io.BytesIO(response.content))
    assert image.mode == "RGBA"
    alpha = np.asarray(image)[..., 3]
    # The field is not its own bounding box (notch, mercator rotation), so
    # some pixels are transparent and most are opaque.
    assert (alpha == 0).any() and (alpha == 255).any()
    assert (alpha == 255).mean() > 0.5
    width, height = image.size
    assert width >= summary["grid"]["cols"] - 2 and height >= summary["grid"]["rows"] - 2


def test_png_stretch_and_blend_options(client, demo):
    from PIL import Image

    dataset_id, _ = demo
    base = f"/api/terrain/{dataset_id}/layer/slope_pct.png"
    plain = client.get(base)
    stretched = client.get(base, params={"vmin": "0", "vmax": "2"})
    blank = client.get(base, params={"vmin": "", "vmax": ""})
    assert plain.status_code == stretched.status_code == blank.status_code == 200
    assert plain.content == blank.content
    assert plain.content != stretched.content
    shaded = client.get(base, params={"hillshade": 1})
    assert shaded.status_code == 200 and shaded.content != plain.content
    # Blending keeps the mask: the same pixels are transparent.
    alpha_plain = np.asarray(Image.open(io.BytesIO(plain.content)))[..., 3]
    alpha_shaded = np.asarray(Image.open(io.BytesIO(shaded.content)))[..., 3]
    assert np.array_equal(alpha_plain == 0, alpha_shaded == 0)
    # The hillshade layer is never blended with itself.
    hs = f"/api/terrain/{dataset_id}/layer/hillshade.png"
    assert client.get(hs).content == client.get(hs, params={"hillshade": 1}).content

    bad = client.get(base, params={"vmin": "abc"})
    assert bad.status_code == 400 and "vmin" in bad.json()["detail"]
    missing = client.get(f"/api/terrain/{dataset_id}/layer/nope.png")
    assert missing.status_code == 404
    for key in LAYER_KEYS:
        assert key in missing.json()["detail"]


# ==========================================================================
# Contours, features, profile
# ==========================================================================

def test_contours_match_the_summary_and_fall_inside_the_field(client, demo):
    dataset_id, summary = demo
    response = client.get(f"/api/terrain/{dataset_id}/contours")
    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "FeatureCollection"
    assert body["interval_m"] == summary["contours"]["interval_m"]
    assert body["count"] == len(body["features"]) == summary["contours"]["count"]
    bounds = summary["grid"]["bounds_lonlat"]
    for feature in body["features"]:
        assert feature["geometry"]["type"] == "LineString"
        assert set(feature["properties"]) == {"level_m", "index", "major"}
        assert all(_inside(lon, lat, bounds) for lon, lat in _coordinates(feature["geometry"]))

    finer = client.get(f"/api/terrain/{dataset_id}/contours", params={"interval_m": "0.5"})
    assert finer.status_code == 200
    assert finer.json()["interval_m"] == 0.5
    assert finer.json()["count"] > body["count"]
    levels = {f["properties"]["level_m"] for f in finer.json()["features"]}
    assert all(abs(level / 0.5 - round(level / 0.5)) < 1e-6 for level in levels)

    assert client.get(f"/api/terrain/{dataset_id}/contours", params={"interval_m": "0"}).status_code == 400
    assert client.get(f"/api/terrain/{dataset_id}/contours", params={"interval_m": "x"}).status_code == 400
    # A blank box means the default interval, not an error.
    blank = client.get(f"/api/terrain/{dataset_id}/contours", params={"interval_m": ""})
    assert blank.status_code == 200 and blank.json()["count"] == body["count"]


def test_features_are_polygons_with_kinds_plus_drainage(client, demo):
    dataset_id, summary = demo
    response = client.get(f"/api/terrain/{dataset_id}/features")
    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "FeatureCollection"
    kinds = [f["properties"]["kind"] for f in body["features"]]
    expected = summary["features"]
    assert kinds.count("hill") == len(expected["hills"])
    assert kinds.count("low") == len(expected["lows"])
    assert kinds.count("depression") == len(expected["depressions"])
    bounds = summary["grid"]["bounds_lonlat"]
    for feature in body["features"]:
        assert feature["geometry"]["type"] in ("Polygon", "MultiPolygon")
        assert all(_inside(lon, lat, bounds) for lon, lat in _coordinates(feature["geometry"]))
        assert feature["properties"]["label"]
    drainage = body["drainage"]
    assert drainage["type"] == "FeatureCollection"
    assert all(f["geometry"]["type"] == "LineString" for f in drainage["features"])
    assert all(f["properties"]["length_m"] > 0 for f in drainage["features"])
    assert len(drainage["features"]) > 0


def test_profile_returns_n_stations(client, demo):
    dataset_id, summary = demo
    west, south, east, north = summary["grid"]["bounds_lonlat"]
    response = client.post(
        f"/api/terrain/{dataset_id}/profile",
        json={"points": [[west, south], [east, north]], "n": 60},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["distance_m"]) == len(body["elev_m"]) == len(body["slope_pct"]) == 60
    assert body["length_m"] > 500
    assert body["distance_m"][0] == 0.0 and abs(body["distance_m"][-1] - body["length_m"]) < 1e-6
    finite = [v for v in body["elev_m"] if v is not None]
    assert len(finite) > 30
    assert summary["elevation"]["min_m"] - 1 <= min(finite) <= max(finite) <= summary["elevation"]["max_m"] + 1

    default_n = client.post(
        f"/api/terrain/{dataset_id}/profile", json={"points": [[west, south], [east, north]]}
    )
    assert len(default_n.json()["distance_m"]) == 200
    one_point = client.post(f"/api/terrain/{dataset_id}/profile", json={"points": [[west, south]]})
    assert one_point.status_code == 400
    assert "two points" in one_point.json()["detail"]


def test_profile_station_count_is_capped(client, demo):
    """One typed zero too many must not tie the server up building a body of
    hundreds of MB: the cap answers at once and says what to ask for."""
    dataset_id, summary = demo
    west, south, east, north = summary["grid"]["bounds_lonlat"]
    line = {"points": [[west, south], [east, north]]}
    cap = terrain_routes.MAX_PROFILE_STATIONS

    too_many = client.post(f"/api/terrain/{dataset_id}/profile", json={**line, "n": 5_000_000})
    assert too_many.status_code == 400
    assert f"at most {cap:,}" in too_many.json()["detail"]
    assert "5,000,000" in too_many.json()["detail"]

    at_cap = client.post(f"/api/terrain/{dataset_id}/profile", json={**line, "n": cap})
    assert at_cap.status_code == 200, at_cap.text
    assert len(at_cap.json()["distance_m"]) == cap


def test_profile_outside_the_field_is_refused(client, demo):
    dataset_id, summary = demo
    west, south, east, north = summary["grid"]["bounds_lonlat"]
    width = east - west
    outside = {"points": [[west - 2 * width, south], [west - width, north]]}
    response = client.post(f"/api/terrain/{dataset_id}/profile", json=outside)
    assert response.status_code == 400
    assert "does not cross the field" in response.json()["detail"]


def test_level_field_has_no_contours_at_any_interval(client, tmp_path):
    """Redrawing a level field's contours at 0.1 m must not bring back the
    noise loops the analysis refused to draw."""
    grid = ElevationGrid(np.full((60, 60), 700.0), 500_000.0, 5_730_000.0, 5.0, "EPSG:32612")
    path = raster.write_geotiff(grid, tmp_path / "level.tif", dtype="float64")
    loaded = client.post("/api/import/path", json={"path": str(path)})
    assert loaded.status_code == 200, loaded.text
    dataset_id = loaded.json()["id"]
    analysed = client.post("/api/terrain/analyze", json={"dataset_id": dataset_id})
    assert analysed.status_code == 200, analysed.text
    assert analysed.json()["summary"]["contours"] == {"interval_m": None, "count": 0}
    for query in ("", "?interval_m=0.1"):
        body = client.get(f"/api/terrain/{dataset_id}/contours{query}").json()
        assert body["count"] == 0 and body["features"] == [] and body["interval_m"] is None
    exported = client.post(f"/api/terrain/{dataset_id}/export", json={"include": ["contours"]})
    assert exported.status_code == 200, exported.text
    assert any("no contours file" in note for note in exported.json()["notes"])


# ==========================================================================
# Zones
# ==========================================================================

@pytest.mark.parametrize("by", ZONE_KINDS)
@pytest.mark.parametrize("kind", ["points", "polygons"])
def test_zones_register_a_dataset_the_export_route_can_write(client, demo, by, kind):
    dataset_id, summary = demo
    before = client.get("/api/datasets").json()["datasets"]
    source_role = client.get(f"/api/datasets/{dataset_id}").json()["role"]
    response = client.post(f"/api/terrain/{dataset_id}/zones", json={"by": by, "kind": kind})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["by"] == by and body["kind"] == kind and body["source_id"] == dataset_id
    assert body["zone_labels"] and all(isinstance(k, str) for k in body["zone_labels"])
    zone = body["dataset"]
    assert zone["label"].startswith(f"Terrain zones ({terrain_routes.ZONE_LABELS[by]})")
    assert zone["origin"] == "terrain_zones"
    assert zone["parent_id"] is None
    assert zone["meta"]["geometry_type"] == ("polygon" if kind == "polygons" else "point")
    if kind == "polygons":
        # One row per zone that has any cell; a class absent from the field
        # (no steep ground, say) has no polygon.
        assert 1 <= zone["rows"] <= len(body["zone_labels"])
    else:
        assert zone["rows"] > 1000

    after = client.get("/api/datasets").json()["datasets"]
    assert len(after) == len(before) + 1
    assert zone["id"] in {d["id"] for d in after}
    # The source keeps its role: the zones are a new layer, not a replacement.
    assert client.get(f"/api/datasets/{dataset_id}").json()["role"] == source_role
    assert source_role is not None
    roles = {layer["dataset_id"]: layer["role"] for layer in client.get("/api/project").json()["layers"]}
    assert roles[dataset_id] == source_role
    assert zone["id"] not in roles

    exported = client.post("/api/export", json={"dataset_id": zone["id"], "formats": ["shapefile"]})
    assert exported.status_code == 200, exported.text
    entries = exported.json()["bundle"]["entries"]
    assert any(name.endswith(".shp") for name in entries)
    assert any(name.endswith(".dbf") for name in entries)
    assert exported.json()["outputs"][0]["features"] == zone["rows"]


@pytest.mark.parametrize("kind", ["points", "polygons"])
def test_zone_datasets_carry_no_track_columns(client, demo, kind):
    """A zone row is a piece of ground, not a record along a track: a
    distance, heading and pass number computed from its row order would be
    numbers about nothing, and they were travelling into every export."""
    from agrosuite.app import server as server_mod

    dataset_id, _summary = demo
    body = client.post(f"/api/terrain/{dataset_id}/zones", json={"by": "landform", "kind": kind}).json()
    zone = server_mod.state.get(body["dataset"]["id"]).dataset
    track_columns = {"distance_m", "heading_deg", "pass_id", "speed_kmh", "swath_m"}
    assert not track_columns & set(zone.df.columns)

    exported = client.post("/api/export", json={"dataset_id": body["dataset"]["id"], "formats": ["geojson"]})
    assert exported.status_code == 200, exported.text
    path = Path(exported.json()["outputs"][0]["path"])
    properties = json.loads(path.read_text(encoding="utf-8"))["features"][0]["properties"]
    assert not track_columns & set(properties)
    assert {"zone", "zone_label"} <= set(properties)


def test_zone_errors_say_what_to_pick(client, demo):
    dataset_id, _ = demo
    response = client.post(f"/api/terrain/{dataset_id}/zones", json={"by": "colour"})
    assert response.status_code == 400
    for kind in ZONE_KINDS:
        assert kind in response.json()["detail"]
    response = client.post(f"/api/terrain/{dataset_id}/zones", json={"by": "landform", "kind": "lines"})
    assert response.status_code == 400 and "points" in response.json()["detail"]
    response = client.post(
        f"/api/terrain/{dataset_id}/zones", json={"by": "elevation_bands", "bands": 1}
    )
    assert response.status_code == 400 and "between 2 and 20" in response.json()["detail"]


def test_zone_codes_are_not_scaled_as_a_rate_on_export(client, demo):
    """The Export tab always sends its rate unit; the zone code in 'value'
    is not a rate, and 1, 2, 3 were coming out as 0.9, 1.8, 2.7 'lb/ac'."""
    dataset_id, _ = demo
    zone = client.post(
        f"/api/terrain/{dataset_id}/zones", json={"by": "landform", "kind": "polygons"}
    ).json()["dataset"]
    exported = client.post("/api/export", json={
        "dataset_id": zone["id"], "formats": ["geojson"], "rate_unit": "lb/ac",
    })
    assert exported.status_code == 200, exported.text
    body = exported.json()
    features = json.loads(Path(body["outputs"][0]["path"]).read_text(encoding="utf-8"))["features"]
    values = sorted(f["properties"]["value"] for f in features)
    assert values == sorted(f["properties"]["zone"] for f in features)
    assert {1, 2, 3} <= set(values)
    assert all(float(v).is_integer() for v in values)
    # No conversion happened, so no note claims one did.
    assert not any("converted" in note for note in body["notes"])

    # A yield map's value is a rate, and it still converts.
    harvest = client.post("/api/import/demo", json={"kind": "harvest"}).json()
    exported = client.post("/api/export", json={
        "dataset_id": harvest["id"], "formats": ["csv"], "rate_unit": "lb/ac",
    })
    assert exported.status_code == 200, exported.text
    assert any("converted from kg/ha to lb/ac" in note for note in exported.json()["notes"])


@pytest.mark.parametrize("kind", ["points", "polygons"])
def test_zone_first_look_describes_zones_and_their_area(client, demo, kind):
    """The first look said 'An elevation raster of 6 cells, covering 0.0 ha'
    of six landform polygons over 60 ha."""
    from agrosuite.app import server as server_mod

    dataset_id, summary = demo
    zone = client.post(
        f"/api/terrain/{dataset_id}/zones", json={"by": "landform", "kind": kind}
    ).json()["dataset"]
    field_ha = summary["grid"]["area_ha"]
    assert zone["area_ha"] == pytest.approx(field_ha, rel=0.02)

    report = server_mod.state.get(zone["id"]).reports["preflight"]
    coverage = [f for f in report["findings"] if f["title"] == "Coverage"]
    assert len(coverage) == 1
    detail = coverage[0]["detail"]
    zones = report["info"]["zones"]
    assert zones >= 2
    if kind == "polygons":
        assert zones == zone["rows"]
    assert detail.startswith(f"Terrain zones (landform) with {zones} zones over ")
    assert "elevation raster" not in detail
    assert report["info"]["area_ha"] == pytest.approx(field_ha, rel=0.02)


@pytest.mark.parametrize("kind", ["points", "polygons"])
def test_analysing_a_zone_dataset_points_back_to_its_source(client, demo, kind):
    """The zone points are grid-cell centres; re-gridded as a GPS track they
    would give back a coarser copy of the relief under a new analysis."""
    dataset_id, _ = demo
    zone = client.post(
        f"/api/terrain/{dataset_id}/zones", json={"by": "wetness", "kind": kind}
    ).json()["dataset"]
    response = client.post("/api/terrain/analyze", json={"dataset_id": zone["id"]})
    assert response.status_code == 400, response.text
    detail = response.json()["detail"]
    assert detail.startswith("These are terrain zones derived from 'Terrain demo'")
    assert "analyse 'Terrain demo' instead" in detail and dataset_id in detail
    # Nothing was registered as a report of the zones.
    assert client.get(f"/api/terrain/{zone['id']}").status_code == 404


# ==========================================================================
# Export
# ==========================================================================

def test_export_writes_rasters_vectors_readme_and_a_zip(client, demo):
    import rasterio

    dataset_id, summary = demo
    response = client.post(f"/api/terrain/{dataset_id}/export", json={})
    assert response.status_code == 200, response.text
    body = response.json()
    folder = Path(body["path"])
    assert folder.is_dir() and folder.name.endswith("_terrain")
    expected = {
        "elevation.tif", "slope_pct.tif", "aspect_deg.tif", "twi.tif", "hillshade.tif",
        "landform.tif", "depression_depth.tif", "contours.shp", "features.shp",
        "drainage.shp", "README.txt",
    }
    assert set(body["files"]) == expected
    for name in body["files"]:
        assert (folder / name).is_file(), name

    with rasterio.open(folder / "landform.tif") as src:
        assert src.dtypes[0] == "uint8" and src.nodata == 255
        codes = src.read(1)
        assert set(np.unique(codes)) - {255} <= {1, 2, 3, 4, 5, 6}
        assert src.tags()["class_1"] == "Hilltop / ridge"
    with rasterio.open(folder / "elevation.tif") as src:
        assert src.dtypes[0] == "float32" and src.nodata == -9999.0
        z = src.read(1)
        valid = z[z != -9999.0]
        assert abs(float(valid.min()) - summary["elevation"]["min_m"]) < 0.01
        assert abs(float(valid.max()) - summary["elevation"]["max_m"]) < 0.01
        assert src.crs.to_string() == summary["grid"]["crs"]

    import geopandas as gpd

    contours = gpd.read_file(folder / "contours.shp")
    assert len(contours) == summary["contours"]["count"]
    assert {"level_m", "index", "major"} <= set(contours.columns)
    features = gpd.read_file(folder / "features.shp")
    assert len(features) == sum(len(summary["features"][k]) for k in ("hills", "lows", "depressions"))
    assert set(features["kind"]) <= {"hill", "low", "depression"}

    readme = (folder / "README.txt").read_text(encoding="utf-8")
    for name in expected - {"README.txt"}:
        assert name in readme
    assert "1  Hilltop / ridge" in readme and "6  Valley / hollow" in readme
    assert "nodata 255" in readme
    assert summary["findings"][0]["text"] in readme

    download = client.get(body["download_url"])
    assert download.status_code == 200 and len(download.content) > 10_000
    archive = zipfile.ZipFile(io.BytesIO(download.content))
    names = {Path(name).name for name in archive.namelist()}
    assert expected <= names
    assert "features.dbf" in names and "contours.prj" in names
    assert set(body["entries"]) == set(archive.namelist())

    # A second export goes beside the first, never over it.
    again = client.post(f"/api/terrain/{dataset_id}/export", json={"include": ["contours"], "vector_format": "geojson"})
    assert again.status_code == 200
    assert Path(again.json()["path"]) != folder
    assert set(again.json()["files"]) == {"contours.geojson", "README.txt"}
    geojson = json.loads((Path(again.json()["path"]) / "contours.geojson").read_text(encoding="utf-8"))
    assert len(geojson["features"]) == summary["contours"]["count"]


def test_export_refuses_unknown_parts(client, demo):
    dataset_id, _ = demo
    response = client.post(f"/api/terrain/{dataset_id}/export", json={"include": ["rasters"]})
    assert response.status_code == 400 and "geotiff" in response.json()["detail"]
    response = client.post(f"/api/terrain/{dataset_id}/export", json={"vector_format": "kml"})
    assert response.status_code == 400 and "shapefile" in response.json()["detail"]
    response = client.post(f"/api/terrain/{dataset_id}/export", json={"include": []})
    assert response.status_code == 400


# ==========================================================================
# Re-analysis and the result cache
# ==========================================================================

def test_reanalysis_replaces_the_report_without_a_new_dataset(client, demo):
    dataset_id, first = demo
    before = client.get("/api/datasets").json()["datasets"]
    response = client.post(
        "/api/terrain/analyze", json={"dataset_id": dataset_id, "cell_m": 8.0}
    )
    assert response.status_code == 200, response.text
    second = response.json()["summary"]
    assert second["grid"]["cell_m"] == 8.0 and first["grid"]["cell_m"] != 8.0
    assert client.get(f"/api/terrain/{dataset_id}").json()["summary"]["grid"]["cell_m"] == 8.0
    assert client.get(f"/api/datasets/{dataset_id}/report/terrain").json()["grid"]["cell_m"] == 8.0
    layers = client.get(f"/api/terrain/{dataset_id}/layers").json()["layers"]
    assert layers[0]["legend"]["vmax"] <= second["elevation"]["max_m"]
    after = client.get("/api/datasets").json()["datasets"]
    assert len(after) == len(before)
    # Back to the default grid for whatever runs next.
    assert client.post("/api/terrain/analyze", json={"dataset_id": dataset_id}).status_code == 200


def test_cache_keeps_the_most_recent_results_and_the_summaries_survive(client, demo, monkeypatch):
    from agrosuite.app import server as server_mod

    monkeypatch.setattr(terrain_routes, "MAX_RESULTS", 2)
    ids = []
    for seed in (11, 12, 13):
        dataset, _ = synthetic_terrain(size_m=300, spacing_m=3.0, seed=seed)
        ids.append(server_mod._register(dataset, f"Small field {seed}", "demo")["id"])
        response = client.post("/api/terrain/analyze", json={"dataset_id": ids[-1]})
        assert response.status_code == 200, response.text

    # The first of the three was pushed out: its arrays are gone, its summary is not.
    evicted, kept = ids[0], ids[-1]
    assert client.get(f"/api/terrain/{evicted}").status_code == 200
    response = client.get(f"/api/terrain/{evicted}/layers")
    assert response.status_code == 404
    assert "no longer in memory" in response.json()["detail"]
    assert client.get(f"/api/terrain/{kept}/layers").status_code == 200

    # Analysing again brings it back, and the newest is still there.
    assert client.post("/api/terrain/analyze", json={"dataset_id": evicted}).status_code == 200
    assert client.get(f"/api/terrain/{evicted}/layers").status_code == 200
    assert client.get(f"/api/terrain/{kept}/layers").status_code == 200

    # A deleted dataset leaves the cache too.
    client.delete(f"/api/datasets/{kept}")
    assert client.get(f"/api/terrain/{kept}/layers").status_code == 404
    assert kept not in terrain_routes._cache

    monkeypatch.undo()
    dataset_id, _ = demo
    assert client.post("/api/terrain/analyze", json={"dataset_id": dataset_id}).status_code == 200
