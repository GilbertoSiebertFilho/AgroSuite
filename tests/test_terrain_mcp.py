"""The MCP tools that read the relief, and the one that was renamed.

The MCP server is a translation layer over the local HTTP API, so a test
that only inspects the TOOLS table proves nothing about what a handler
would say. These tests run the handlers against a real analysis, with
:class:`fastapi.testclient.TestClient` standing in for the HTTP round trip:
the module's own :class:`~agrosuite.mcp_server.App` would have to find or
start a server on a port, which means a subprocess, up to twenty seconds of
waiting and a port that may already be taken by whatever else the machine
is running. The substitute answers exactly as ``App.call`` does — the
parsed JSON body, or ``RuntimeError`` carrying the ``detail`` of an error —
so the handlers, and the error text a caller finally reads, are the real
ones.

What matters here is the text, not the JSON: ``analyse_terrain`` exists to
turn a dozen nested tables into sentences a person can act on, and a number
whose unit was left for the assistant to guess is worse than no number.

The unit is the one the app is showing — the session opens on the Canadian
default, so these tests read in feet and acres. The tool asks the app for
the set rather than choosing one, because the findings it quotes are
already written in it and an answer in two systems about one field is
worse than either.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agrosuite import mcp_server

TERRAIN_TOOLS = ("analyse_terrain", "terrain_zones")


# ==========================================================================
# Fixtures
# ==========================================================================

class _LocalApp:
    """``App`` speaking to the app in this process instead of over a port."""

    def __init__(self, client) -> None:
        self.client = client

    def call(self, method: str, path: str, body=None):
        response = self.client.request(method, path, json=body)
        if response.status_code >= 400:
            # The same unwrapping App.call does, so a refusal reaches the
            # caller as the sentence the app wrote, not as a status code.
            try:
                detail = response.json().get("detail", response.reason_phrase)
            except Exception:
                detail = response.reason_phrase
            raise RuntimeError(str(detail))
        return response.json()


@pytest.fixture(scope="module")
def app(request):
    from fastapi.testclient import TestClient

    from agrosuite.app import server as server_mod

    client = TestClient(server_mod.app)
    original = mcp_server.APP
    mcp_server.APP = _LocalApp(client)
    request.addfinalizer(lambda: setattr(mcp_server, "APP", original))
    return client


@pytest.fixture(scope="module")
def terrain(app):
    """The terrain demo, loaded and analysed once: ``(dataset_id, text)``.

    ``min_feature_height_m`` is dropped to 0.2 m so the closed depression
    the synthetic field really carries clears the threshold: at the default
    (twice the measured GPS noise) it is one of the hollows the analysis
    counts but does not list, and the depression wording would go untested.
    """
    dataset_id = app.post("/api/terrain/demo").json()["id"]
    text = mcp_server.tool_analyse_terrain(dataset_id, min_feature_height_m=0.2)
    return dataset_id, text


@pytest.fixture(scope="module")
def trial(app):
    """The strip-trial demo: a rate that was applied and a yield to fit."""
    return app.post("/api/import/demo", json={"kind": "trial"}).json()["id"]


# ==========================================================================
# The tool table
# ==========================================================================

def test_the_tools_are_declared_under_their_new_names():
    names = {tool["name"] for tool in mcp_server.TOOLS}
    assert {"analyse_economics", *TERRAIN_TOOLS} <= names
    # The old name is gone everywhere, not shadowed by an alias: a caller
    # that still asks for it should get "No such tool", not a silent hit.
    assert "analyse_difm" not in names
    assert "analyse_difm" not in mcp_server.TOOLS_BY_NAME
    assert not hasattr(mcp_server, "tool_analyse_difm")

    listing = mcp_server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    listed = {tool["name"] for tool in listing["result"]["tools"]}
    assert {"analyse_economics", *TERRAIN_TOOLS} <= listed
    assert "analyse_difm" not in listed


def test_the_economics_tool_is_described_without_its_old_jargon():
    """The tool is offered to anyone with a rate and a yield, not only to
    someone who knows what the acronym stood for."""
    description = mcp_server.TOOLS_BY_NAME["analyse_economics"]["description"]
    assert "difm" not in description.lower()
    assert "on-farm precision experiment" not in description.lower()
    for promised in ("economic optimum", "agronomic maximum", "zone"):
        assert promised in description.lower()


@pytest.mark.parametrize("name", ["analyse_economics", *TERRAIN_TOOLS])
def test_each_schema_matches_its_handler(name):
    """A schema that promises an argument the handler will not take turns
    into "Wrong arguments for ..." in the middle of a conversation."""
    import inspect

    tool = mcp_server.TOOLS_BY_NAME[name]
    schema = tool["schema"]
    assert schema["type"] == "object"
    properties = schema["properties"]
    assert properties, "a tool with no properties cannot be given a dataset"
    required = schema.get("required", [])
    assert "dataset_id" in required
    assert set(required) <= set(properties)

    parameters = inspect.signature(tool["handler"]).parameters
    assert set(properties) <= set(parameters)
    for field, spec in properties.items():
        assert spec["type"] in ("string", "number", "integer", "boolean", "array")
        if field not in required:
            assert parameters[field].default is not inspect.Parameter.empty


def test_the_zone_choices_are_the_ones_the_route_accepts():
    """An enum copied out of step with the route turns a typo into a 400
    the caller only meets after the analysis has run."""
    from agrosuite.terrain.analysis import ZONE_KINDS

    properties = mcp_server.TOOLS_BY_NAME["terrain_zones"]["schema"]["properties"]
    assert tuple(properties["by"]["enum"]) == tuple(ZONE_KINDS)
    assert properties["kind"]["enum"] == ["points", "polygons"]


# ==========================================================================
# analyse_terrain against a real analysis
# ==========================================================================

def test_the_terrain_summary_reads_as_sentences_with_their_units(terrain):
    _dataset_id, text = terrain

    assert "{" not in text and "[]" not in text, "the summary must not leak raw JSON"
    # The set is named once, at the top, so the conversation can convert
    # from it if it is asked for something else.
    assert text.startswith("Measurements below are in the units AgroSuite is showing:")
    # The character of the field and its total relief — 60.5 ha and 10.5 m
    # in the store, read to someone working in acres and feet.
    assert "Gently undulating field of 149.5 ac" in text
    assert "34.6 ft of total relief" in text
    # The trend: how much it falls, which way, how steeply.
    assert "falls 26.6 ft towards the south-west" in text
    assert "average gradient of 0.7 %" in text
    # Every unit is named rather than left to be guessed, and all of them
    # belong to the one set: a percentage is the same in every set.
    for unit in (" ft", " ac", " %", " ft³"):
        assert unit in text
    # ...and not one quantity is left in the store's own units. A number
    # then the unit as a word of its own: " ha" alone is inside "had".
    assert not re.search(r"\d\s?(?:ha|m|m³|km|cm)\b", text), text


def test_the_hills_lows_and_depressions_are_placed_and_measured(terrain):
    _dataset_id, text = terrain

    assert "Hill 1 in the south-west part: 11.5 ft over 6.5 ac" in text
    assert "Hill 2 in the eastern part" in text
    assert "Low 1 in the southern part: 7.5 ft below over 13.3 ac" in text
    assert "water ponds there" in text
    # A closed depression is quoted with the volume it holds before it
    # spills, in the cube of the reader's own length unit.
    assert "Depression 1 in the north-east part" in text
    assert "ft³ before it spills" in text
    # And the hollows too shallow to list are still accounted for.
    assert "within the measurement noise" in text


def test_the_steep_share_and_the_wet_area_are_reported(terrain):
    _dataset_id, text = terrain

    assert "is above 10 % slope" in text
    assert "Likely wet: 14.6 ac" in text
    # The shares leave out the outer ring of cells, and say so, because a
    # ring whose slope leans on copied values would inflate them.
    assert "outermost ring of cells" in text


def test_the_findings_travel_verbatim(app, terrain):
    """The findings are already the sentences a farmer acts on; a tool that
    paraphrased them would be re-deciding what matters.

    Word for word against what the screen is showing, which is what the
    tool speaking the app's own unit set buys: the same sentence reaches
    the person whether they read it in the browser or hear it back from
    the assistant.
    """
    dataset_id, text = terrain
    summary = app.get(f"/api/terrain/{dataset_id}").json()["summary"]

    findings = summary["findings"]
    assert findings
    for finding in findings:
        assert f"[{finding['level']}] {finding['text']}" in text


def test_the_options_are_passed_through_in_metres(app):
    """A cell size the caller asked for must reach the analyser; a 0 m
    smoothing must mean "no smoothing", not "choose for me"."""
    dataset_id = app.post("/api/terrain/demo").json()["id"]
    mcp_server.tool_analyse_terrain(dataset_id, cell_m=10.0, smooth_m=0.0,
                                   contour_interval_m=2.0)
    options = app.get(f"/api/terrain/{dataset_id}").json()["summary"]["options"]

    assert options["cell_m"] == 10.0
    assert options["smooth_m"] == 0.0
    assert options["contour_interval_m"] == 2.0


def test_a_file_without_altitude_comes_back_with_the_app_s_own_message(app):
    """The analyser's refusal names what to open instead; wrapping it in a
    summary of nothing would bury the one useful sentence."""
    from agrosuite.app import server as server_mod
    from agrosuite.core import schema as sch
    from agrosuite.demo import synthetic_harvest

    dataset = synthetic_harvest()
    dataset.df.drop(columns=[sch.ELEVATION], inplace=True, errors="ignore")
    dataset_id = server_mod._register(dataset, "Harvest without altitude", "demo")["id"]

    response = mcp_server.handle({
        "jsonrpc": "2.0", "id": 2, "method": "tools/call",
        "params": {"name": "analyse_terrain", "arguments": {"dataset_id": dataset_id}},
    })
    result = response["result"]
    assert result["isError"] is True
    assert result["content"][0]["text"] == (
        "This file carries no usable elevation, so its relief cannot be analysed. "
        "Open a yield or as-applied export that includes the GPS altitude, or a "
        "DEM GeoTIFF of the field."
    )


# ==========================================================================
# terrain_zones
# ==========================================================================

@pytest.mark.parametrize("by,first_label", [
    ("landform", "Hilltop / ridge"),
    ("slope_class", "Flat (< 2 %)"),
    ("wetness", "Well drained"),
])
def test_zones_report_the_new_dataset_and_its_labels(terrain, by, first_label):
    dataset_id, _text = terrain

    result = mcp_server.tool_terrain_zones(dataset_id, by=by)

    assert result["dataset_id"] != dataset_id
    assert result["label"].startswith("Terrain zones (")
    assert result["zones"][0] == f"1: {first_label}"
    # Numbered in code order, which is the order the codes mean something
    # in: 1 is the hilltop and the last code the hollow.
    assert [z.split(":")[0] for z in result["zones"]] == \
        [str(i + 1) for i in range(len(result["zones"]))]


def test_zone_polygons_carry_one_row_per_zone(terrain):
    dataset_id, _text = terrain

    points = mcp_server.tool_terrain_zones(dataset_id, by="landform", kind="points")
    polygons = mcp_server.tool_terrain_zones(dataset_id, by="landform", kind="polygons")

    assert points["geometry"] == "point"
    assert polygons["geometry"] == "polygon"
    assert polygons["rows"] == len(polygons["zones"])
    assert points["rows"] > polygons["rows"]


def test_the_new_zone_dataset_is_one_the_session_can_export(app, terrain):
    """The point of the tool: the id it hands back is an ordinary dataset,
    so the conversation goes straight on to the export tools."""
    dataset_id, _text = terrain

    result = mcp_server.tool_terrain_zones(dataset_id, by="elevation_bands",
                                           kind="polygons")
    listed = {d["dataset_id"]: d for d in mcp_server.tool_list_datasets()}

    assert result["dataset_id"] in listed
    assert listed[result["dataset_id"]]["label"] == result["label"]
    # It starts with no role: zones are an output of the analysis, not a
    # layer of the field, and the user says what they are for.
    assert listed[result["dataset_id"]]["role"] is None


def test_zones_before_an_analysis_say_which_call_is_missing(app):
    dataset_id = app.post("/api/terrain/demo").json()["id"]

    response = mcp_server.handle({
        "jsonrpc": "2.0", "id": 3, "method": "tools/call",
        "params": {"name": "terrain_zones",
                   "arguments": {"dataset_id": dataset_id, "by": "landform"}},
    })
    result = response["result"]
    assert result["isError"] is True
    assert "Run the terrain analysis first" in result["content"][0]["text"]


# ==========================================================================
# analyse_economics
# ==========================================================================

def test_the_renamed_economics_tool_still_fits_the_curve(app, trial):
    """The rename is a rename: the same call, the same report."""
    report = mcp_server.tool_analyse_economics(
        trial, crop_price_per_kg=0.73, input_cost_per_kg=1.37, zone_column="zone",
    )

    assert report["model"]
    assert 0.0 <= report["r2"] <= 1.0
    assert report["rates_tested_kg_ha"] == [0.0, 60.0, 120.0, 180.0, 240.0]
    economics = report["economics_internal_units"]
    assert economics["optimum_rate"] > 0
    assert economics["yield_at_optimum"] > 0
    assert economics["agronomic_maximum"] > economics["optimum_rate"]
    assert len(report["by_rate"]) == len(report["rates_tested_kg_ha"])
    # A zone column brings one curve per zone and the comparison that
    # decides whether the map is worth building.
    assert len(report["by_zone"]) == 2
    assert "reading" in report["uniform_vs_variable"]


def test_the_old_tool_name_is_refused(app, trial):
    response = mcp_server.handle({
        "jsonrpc": "2.0", "id": 4, "method": "tools/call",
        "params": {"name": "analyse_difm",
                   "arguments": {"dataset_id": trial, "crop_price_per_kg": 0.73,
                                 "input_cost_per_kg": 1.37}},
    })
    assert response["error"]["message"] == "No such tool: analyse_difm"
