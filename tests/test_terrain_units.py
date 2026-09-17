"""The GPS altitude of a monitor export is in the unit the rest of the
file is in.

A John Deere Operations Center export writes feet: 60 for the width, 1 920
for the altitude of a 585 m field. The preliminary check reads the width
and proposes feet, but the terrain analyser is the first consumer of the
altitude column, and when the accepted unit did not reach it the report
multiplied every height, noise figure and pond volume by 3.28 with a
straight face. Two guards: the declared length unit covers the altitude,
and an analysis of a file whose width says feet opens with a warning.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fixtures  # noqa: E402  (tests/fixtures.py)
from agrosuite.terrain import analysis as analysis_mod  # noqa: E402
from agrosuite.terrain.synthetic import synthetic_terrain  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FT = 0.3048


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from agrosuite.app import server as server_mod

    return TestClient(server_mod.app)


@pytest.fixture(scope="module")
def john_deere(client, tmp_path_factory):
    """The Operations Center fixture imported as it comes: feet undeclared."""
    shp = fixtures.john_deere_shapefile(tmp_path_factory.mktemp("jd"))
    loaded = client.post("/api/import/path", json={"path": str(shp)})
    assert loaded.status_code == 200, loaded.text
    return loaded.json()


def test_undeclared_feet_open_the_report_with_a_warning(client, john_deere):
    assert john_deere["preflight"]["proposed_units"]["length"] == "ft"
    summary = client.post("/api/terrain/analyze", json={"dataset_id": john_deere["id"]}).json()["summary"]
    # The numbers are what the file says, in the unit it does not name...
    assert summary["elevation"]["min_m"] > 1900
    # ...and the first thing the reader sees says so.
    first = summary["findings"][0]
    assert first["level"] == "warning"
    assert "feet" in first["text"] and "3.3 times" in first["text"]
    assert summary["source"]["unit_doubt"] == first["text"]


def test_declared_length_unit_covers_the_altitude(client, john_deere):
    """What the interface's one-click button and Claude's open_file send:
    the length unit for the width, the distance and the altitude."""
    converted = client.post(
        f"/api/datasets/{john_deere['id']}/units",
        json={"source_units": {"swath_m": "ft", "distance_m": "ft", "elev_m": "ft"}, "crop": "canola"},
    )
    assert converted.status_code == 200, converted.text
    assert any(c.startswith("Elevation: converted from ft") for c in converted.json()["conversions"])

    summary = client.post(
        "/api/terrain/analyze", json={"dataset_id": converted.json()["dataset"]["id"]}
    ).json()["summary"]
    assert 584 < summary["elevation"]["min_m"] < summary["elevation"]["max_m"] < 586
    assert summary["source"]["vertical_noise_m"] == pytest.approx(0.4, abs=0.15)
    assert summary["source"]["unit_doubt"] is None
    assert not any("feet" in f["text"] for f in summary["findings"])


def test_no_doubt_when_the_width_is_in_metres():
    ds, _ = synthetic_terrain()
    assert analysis_mod._elevation_unit_doubt(ds) is None
    # A 12 m header written as 39 ft: a width only feet explain. (The demo's
    # own 9 m would read 30, which is a fair width in either unit, and an
    # ambiguous width is no evidence — the preliminary check says nothing then either.)
    ds.df["swath_m"] = 12.0 / FT
    assert "feet" in analysis_mod._elevation_unit_doubt(ds)


def test_the_clients_send_the_length_unit_for_the_altitude_too():
    """No JavaScript test runner here, so the interface and the MCP tool are
    read as text: wherever a length unit goes to ``swath_m`` it goes to
    ``elev_m`` as well, or the terrain analyser reads feet as metres."""
    app_js = (ROOT / "agrosuite" / "app" / "static" / "app.js").read_text(encoding="utf-8")
    assert app_js.count("source_units.swath_m =") == app_js.count("source_units.elev_m =") >= 2

    mcp = (ROOT / "agrosuite" / "mcp_server.py").read_text(encoding="utf-8")
    block = re.search(r"if length_unit:\n(.*?)\n\S", mcp, re.S).group(1)
    assert "swath_m" in block and "elev_m" in block
