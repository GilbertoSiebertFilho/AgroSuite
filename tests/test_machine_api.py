"""The Machine tab's API, driven the way the interface drives it.

The demo loads a made-up sprayer day — written as a real SoMat log and read
back through the same reader — with its field's boundary; everything the tab
does is then asked of the server and checked against what the day holds.
"""

from __future__ import annotations

import io
import zipfile

import pytest

TYRES = {"boom_width_m": 36.576, "tyre_size": "380/90R46", "track_width_m": 3.048, "def_tank_l": 60}


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from agrosuite.app import server as server_mod

    return TestClient(server_mod.app)


@pytest.fixture(scope="module")
def demo(client):
    response = client.post("/api/machine/demo")
    assert response.status_code == 200, response.text
    return response.json()


@pytest.fixture(scope="module")
def analysed(client, demo):
    body = {"dataset_id": demo["log"]["id"], "boundary_id": demo["boundary"]["id"],
            "yield_kg_ha": 4000, "crop_price_per_kg": 0.3, "fuel_price": 1.6, **TYRES}
    response = client.post("/api/machine/analyze", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def test_the_demo_loads_a_log_and_its_boundary(client, demo):
    from agrosuite.app import server as server_mod

    log = server_mod.state.get(demo["log"]["id"])
    assert log.dataset.meta.operation == "telemetry"
    assert server_mod.state.get(demo["boundary"]["id"]).dataset.meta.operation == "boundary"
    assert demo["suggested"]["tyre_size"] == "380/90R46"


def test_the_preflight_reads_a_log_as_a_log(demo):
    from agrosuite.app import server as server_mod

    report = server_mod.state.get(demo["log"]["id"]).reports["preflight"]
    assert report["verdict"] == "ok"
    assert report["suggested_role"] == "telemetry"
    assert report["next_step"]["step"] == "machine"
    titles = [f["title"] for f in report["findings"]]
    assert "Repeated positions" not in titles and "Speed out of range" not in titles
    assert not any(t.startswith("Missing:") for t in titles)


def test_the_analysis_answers_with_the_day_and_its_layers(analysed):
    summary = analysed["summary"]
    assert summary["basis"] == "boundary"
    assert summary["boundary"]["label"] == "Demo field boundary"
    assert {r["key"] for r in summary["activities"]} == {"field", "idle", "road", "off"}
    assert summary["trampling"]["available"] and summary["trampling"]["lost_kg"] > 0
    layers = analysed["layers"]
    activities = {f["properties"]["activity"] for f in layers["track"]["features"]}
    assert activities == {"field", "road"}
    assert {f["properties"]["activity"] for f in layers["stops"]["features"]} == {"idle", "off"}
    strip = layers["strips"]["features"][0]
    assert strip["geometry"]["type"] in ("Polygon", "MultiPolygon")
    lon, lat = strip["geometry"]["coordinates"][0][0][0][:2] if strip["geometry"]["type"] == "MultiPolygon" \
        else strip["geometry"]["coordinates"][0][0][:2]
    assert -115 < lon < -113 and 51 < lat < 53, "the layers are in WGS84"


def test_the_crop_price_comes_from_the_project_when_not_given(client, demo):
    from agrosuite.app import server as server_mod

    server_mod.state.project["prices"]["crop_price"] = 0.25
    try:
        body = {"dataset_id": demo["log"]["id"], "boundary_id": demo["boundary"]["id"],
                "yield_kg_ha": 4000, **TYRES}
        t = client.post("/api/machine/analyze", json=body).json()["summary"]["trampling"]
        assert t["crop_price_per_kg"] == 0.25
        assert t["lost_value"] == pytest.approx(t["lost_kg"] * 0.25)
    finally:
        server_mod.state.project["prices"].pop("crop_price", None)


def test_the_summary_is_said_again_in_the_units_on_screen(client, demo, analysed):
    from agrosuite.app import server as server_mod
    from agrosuite.core.units import UNIT_PRESETS

    saved = dict(server_mod.state.display_units)
    try:
        server_mod.state.display_units = dict(UNIT_PRESETS["metric"])
        metric = client.get(f"/api/machine/{demo['log']['id']}").json()
        server_mod.state.display_units = dict(UNIT_PRESETS["usa"])
        again = client.get(f"/api/machine/{demo['log']['id']}").json()
        assert any(" L " in f["text"] for f in metric["summary"]["findings"])
        assert any(" gal " in f["text"] for f in again["summary"]["findings"])
        assert again["summary"]["fuel"] == metric["summary"]["fuel"], "no number moves"
        assert again["layers"] is not None
    finally:
        server_mod.state.display_units = saved


def test_statistics_come_as_rows_and_as_csv(client, demo, analysed):
    rows = client.get(f"/api/machine/{demo['log']['id']}/statistics").json()["rows"]
    assert {r["selection"] for r in rows} == {"all", "field", "idle", "road", "off"}
    response = client.get(f"/api/machine/{demo['log']['id']}/statistics.csv")
    assert response.status_code == 200 and response.headers["content-type"].startswith("text/csv")
    lines = response.text.splitlines()
    assert lines[0].startswith("selection,channel,unit,n,duration_s,min,t_min,max,t_max")
    assert len(lines) == len(rows) + 1


def test_the_export_writes_every_file_and_a_zip(client, demo, analysed):
    response = client.post(f"/api/machine/{demo['log']['id']}/export", json={})
    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body["files"]) == {"activity_track.shp", "stops.shp", "tyre_strips.shp",
                                  "samples.csv", "statistics.csv", "summary.json", "README.txt"}
    archive = zipfile.ZipFile(io.BytesIO(client.get(body["download_url"]).content))
    names = {name.split("/")[-1] for name in archive.namelist()}
    assert {"tyre_strips.dbf", "tyre_strips.prj", "samples.csv", "README.txt"} <= names
    readme = next(n for n in archive.namelist() if n.endswith("README.txt"))
    assert "What the analysis found" in archive.read(readme).decode("utf-8")
    samples = next(n for n in archive.namelist() if n.endswith("samples.csv"))
    assert archive.read(samples).decode("utf-8").splitlines()[0].startswith("activity,")


def test_the_export_can_write_geojson(client, demo, analysed):
    body = client.post(f"/api/machine/{demo['log']['id']}/export",
                       json={"vector_format": "geojson"}).json()
    assert "tyre_strips.geojson" in body["files"]


def test_a_file_that_is_not_a_log_is_refused_in_words(client, demo):
    response = client.post("/api/machine/analyze", json={"dataset_id": demo["boundary"]["id"]})
    assert response.status_code == 400
    assert "is not a machine log" in response.json()["detail"]


def test_a_boundary_that_is_not_a_polygon_is_refused(client, demo):
    response = client.post("/api/machine/analyze", json={"dataset_id": demo["log"]["id"],
                                                         "boundary_id": demo["log"]["id"]})
    assert response.status_code == 400
    assert "holds no polygon" in response.json()["detail"]


def test_a_bad_tyre_is_refused_in_words(client, demo):
    response = client.post("/api/machine/analyze",
                           json={"dataset_id": demo["log"]["id"], "tyre_size": "banana"})
    assert response.status_code == 400
    assert "does not read as a tyre size" in response.json()["detail"]


def test_unknown_settings_are_refused(client, demo):
    response = client.post("/api/machine/analyze",
                           json={"dataset_id": demo["log"]["id"], "tyre_colour": "black"})
    assert response.status_code == 422


def test_an_unanalysed_log_says_what_to_do(client):
    from agrosuite.app import server as server_mod
    from agrosuite.machine import synthetic
    from agrosuite.formats import registry

    path, _ = synthetic.demo_log(server_mod.state.uploads / "fresh.sie")
    registered = server_mod._register(registry.read_any(path), "Fresh log", "import")
    response = client.get(f"/api/machine/{registered['id']}")
    assert response.status_code == 404
    assert "has not been analysed" in response.json()["detail"]


def test_a_profile_keeps_the_tyres_and_the_def_tank(client, tmp_path, monkeypatch):
    monkeypatch.setenv("AGROSUITE_HOME", str(tmp_path / "home"))
    body = {"name": "Sprayer 120 ft", "kind": "sprayer", "implement_width_m": 36.576,
            "speed_min_kmh": 8, "speed_max_kmh": 25, "tyre_size": "380/90R46",
            "track_width_m": 3.048, "rear_follows_front": True, "def_tank_l": 60}
    response = client.post("/api/profiles", json=body)
    assert response.status_code == 200, response.text
    saved = next(p for p in client.get("/api/profiles").json()["profiles"] if p["name"] == "Sprayer 120 ft")
    assert saved["tyre_width_m"] == pytest.approx(0.38)
    assert saved["track_width_m"] == 3.048 and saved["def_tank_l"] == 60


def test_the_catalog_carries_the_new_units(client):
    catalog = client.get("/api/catalog").json()["units"]
    assert {"liquid", "distance"} <= set(catalog["groups"])
    assert catalog["presets"]["canada"]["liquid_unit"] == "L"


# ==========================================================================
# The MCP tool, over the same app
# ==========================================================================

class _LocalApp:
    """``App`` speaking to the app in this process instead of over a port."""

    def __init__(self, client) -> None:
        self.client = client

    def call(self, method: str, path: str, body=None):
        response = self.client.request(method, path, json=body)
        if response.status_code >= 400:
            raise RuntimeError(response.json().get("detail", response.reason_phrase))
        return response.json()


def test_the_mcp_tool_says_the_day_in_sentences(client, demo, monkeypatch):
    from agrosuite import mcp_server

    monkeypatch.setattr(mcp_server, "APP", _LocalApp(client))
    assert any(t["name"] == "analyse_machine" for t in mcp_server.TOOLS)
    text = mcp_server.tool_analyse_machine(
        demo["log"]["id"], boundary_id=demo["boundary"]["id"], yield_kg_ha=4000, **TYRES)
    assert text.startswith("Machine log 'Sprayer day (demo)':")
    assert "of diesel" in text and "The tyres crossed" in text
    assert "By activity (metric):" in text and "- Working in the field:" in text
    assert "Crop crossed by the tyres:" in text and "kg lost" in text


def test_the_mcp_tool_passes_a_refusal_through(client, demo, monkeypatch):
    from agrosuite import mcp_server

    monkeypatch.setattr(mcp_server, "APP", _LocalApp(client))
    with pytest.raises(RuntimeError, match="is not a machine log"):
        mcp_server.tool_analyse_machine(demo["boundary"]["id"])
