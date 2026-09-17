"""The app's own sentences, in the units the reader works in.

Every number this app stores is metric, and the interface converts it for
the screen. A *sentence* cannot be converted for the screen — its numbers
are written inside it, and restating one in another unit means deciding
again what it says — so the unit set travels into the writing of it, and a
stored report's prose is written again whenever someone reads it.

What these tests protect is that split:

* the same analysis, said in two unit sets, carries the same numbers and
  different words;
* nothing a reader on the Canadian default is shown quotes a metre, a
  hectare, a cubic metre or kg/ha, except where the sentence is *about*
  the file's own units and naming one is the point;
* the imperial sentence carries the converted value, not a relabelled one;
* an unknown unit is refused by name rather than quietly ignored;
* a project saved in one unit set opens in the one in force when it is
  opened, not the one it was saved in.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agrosuite.clean import pipeline as clean_pipeline
from agrosuite.core import preflight
from agrosuite.core import schema as sch
from agrosuite.core.dataset import Dataset, DatasetMeta
from agrosuite.core.units import UNIT_PRESETS, Phrase
from agrosuite.demo import synthetic_harvest
from agrosuite.terrain import analysis as terrain
from agrosuite.terrain.synthetic import synthetic_terrain

CANADA = UNIT_PRESETS["canada"]
METRIC = UNIT_PRESETS["metric"]

#: A metric quantity inside a sentence: a number, then the unit as a word of
#: its own. Written this way rather than as a substring search because " ha"
#: is inside "it has" and " m " inside "12 m of relief" alike — only the
#: second is a measurement, and only a boundary after the unit tells them
#: apart ("chamber", "hard", "metres" are words, not units).
METRIC_QUANTITY = re.compile(
    r"\d(?:[\d .,]*)\s?(?:ha|m|m²|m³|km|cm|kg/ha|km/h|kg)\b"
)


def assert_no_metric(text: str) -> None:
    found = [m.group(0) for m in METRIC_QUANTITY.finditer(text)]
    assert not found, f"metric quantity {found} in a Canadian sentence: {text}"


# ==========================================================================
# Fixtures
# ==========================================================================

@pytest.fixture(scope="module")
def relief():
    """One analysed relief, kept for the whole module: it takes seconds."""
    dataset, _ = synthetic_terrain()
    return terrain.analyze(dataset)


@pytest.fixture(scope="module")
def harvest():
    dataset = synthetic_harvest()
    dataset.ensure_derived()
    return dataset


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from agrosuite.app import server as server_mod

    return TestClient(server_mod.app)


def _imperial_yield_map() -> Dataset:
    """A canola map written in bu/ac, mph and feet and imported as metric.

    The median reads 55, which is a fair yield in bu/ac and an impossible
    one in kg/ha: the file the unit check exists for.
    """
    rng = np.random.default_rng(3)
    n = 900
    frame = pd.DataFrame({
        sch.LON: -105.83 + rng.normal(0, 0.002, n),
        sch.LAT: 50.45 + rng.normal(0, 0.002, n),
        sch.VALUE: rng.normal(55.0, 4.0, n),
        sch.SPEED: rng.normal(5.4, 0.3, n),
        sch.SWATH: np.full(n, 40.0),
    })
    dataset = Dataset(frame, DatasetMeta(
        name="canola.csv", operation="harvest", crop="canola", value_label="Yield"))
    dataset.ensure_derived()
    return dataset


# ==========================================================================
# The same analysis, said twice
# ==========================================================================

def test_the_relief_says_the_same_numbers_in_two_unit_sets(relief):
    """Same analysis, two readers: identical payload, different prose."""
    stored = relief.summary()
    canadian = terrain.restate(stored, CANADA)
    metric = terrain.restate(stored, METRIC)

    def numbers(summary):
        """Every number in the payload, prose left out."""
        out = []

        def walk(value, path=""):
            if isinstance(value, dict):
                for key, item in value.items():
                    walk(item, f"{path}.{key}")
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    walk(item, f"{path}[{i}]")
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                out.append((path, float(value)))

        walk(summary)
        return out

    assert numbers(canadian) == numbers(metric)
    assert numbers(canadian) == numbers(stored)

    canadian_text = [f["text"] for f in canadian["findings"]]
    metric_text = [f["text"] for f in metric["findings"]]
    assert len(canadian_text) == len(metric_text)
    assert canadian_text != metric_text
    # Every sentence the analyser wrote and that measures something differs;
    # the ones that measure nothing ("2 distinct hills rise above the
    # surrounding ground") read the same in every unit set, and should.
    # The notes the grid builder wrote while it read the file are in here
    # too, and are held to the same standard: they measure ground.
    measured = [
        (a, b) for a, b in zip(canadian_text, metric_text)
        if METRIC_QUANTITY.search(b)
    ]
    assert len(measured) >= 5
    assert all(a != b for a, b in measured)
    # The findings are the analyser's own words, so the level and the order
    # of the sentences do not move with the units either.
    assert [f["level"] for f in canadian["findings"]] == [
        f["level"] for f in metric["findings"]]


def test_the_canadian_reader_is_never_shown_a_metre_or_a_hectare(relief):
    """Nothing in the Terrain tab's sentences quotes a metric unit.

    The notes the grid builder writes while it reads the file are part of
    the findings, so they are checked too: this analysis is re-run with the
    Canadian set, which is what the app does.
    """
    dataset, _ = synthetic_terrain()
    canadian = terrain.analyze(dataset, units=CANADA).summary()
    canadian = terrain.restate(canadian, CANADA)
    for finding in canadian["findings"]:
        assert_no_metric(finding["text"])
    assert_no_metric(canadian["grid"]["shares_note"])
    assert_no_metric(canadian["character"]["why"])
    # And the same sentences in the metric set do quote them, or the check
    # above would pass on prose with no numbers in it at all.
    metric = terrain.restate(relief.summary(), METRIC)
    assert any(METRIC_QUANTITY.search(f["text"]) for f in metric["findings"])


def test_an_imperial_sentence_carries_the_converted_value(relief):
    """10.536 m of relief is 34.6 ft, not 10.5 of anything."""
    stored = relief.summary()
    relief_m = float(stored["elevation"]["relief_m"])
    assert 10.0 < relief_m < 11.0, relief_m
    feet = relief_m / 0.3048

    canadian = terrain.restate(stored, CANADA)["findings"][0]["text"]
    assert f"{feet:.1f} ft" in canadian
    assert canadian.startswith("Total relief is ")
    assert "ac" in canadian

    metric = terrain.restate(stored, METRIC)["findings"][0]["text"]
    assert f"{relief_m:.1f} m" in metric
    # The relief itself never moved.
    assert terrain.restate(stored, CANADA)["elevation"]["relief_m"] == relief_m


def test_the_notes_from_reading_the_file_follow_the_reader(relief):
    """The last sentences on the Terrain tab that did not move with the picker.

    A note is written while the file is gridded — the smoothing, the
    spacing between readings, a strip, two blocks of ground in one export
    — and it used to be finished prose the moment it was written:
    analysed on the Canadian preset and read back in metric, every
    finding around it changed and the smoothing note stayed in feet. It
    is stored as the fact it states now, so it is written again like
    every other sentence, from the same metric numbers.
    """
    stored = relief.summary()  # analysed with no unit set: metric throughout
    facts = stored["source"]["note_facts"]
    assert facts, "a file with GPS noise in it is smoothed, and says so"

    canadian = terrain.restate(stored, CANADA)
    metric = terrain.restate(stored, METRIC)
    assert canadian["source"]["notes"] != metric["source"]["notes"]
    assert metric["source"]["notes"] == stored["source"]["notes"]
    for note in canadian["source"]["notes"]:
        assert_no_metric(note)
    # They reach the reader as findings, and those carry the same words.
    said = [f["text"] for f in canadian["findings"]]
    for note in canadian["source"]["notes"]:
        assert note in said
    # Saying them again moved nothing: the facts are what they were.
    assert canadian["source"]["note_facts"] == facts == stored["source"]["note_facts"]


def test_the_relief_notes_move_with_the_picker_over_the_wire(client):
    """Analysed in one preset, read in the other: one analysis, no re-run."""
    client.post("/api/session/new")
    client.put("/api/units/display", json={"units": CANADA})
    dataset_id = client.post("/api/terrain/demo").json()["id"]
    analysed = client.post("/api/terrain/analyze", json={"dataset_id": dataset_id})
    assert analysed.status_code == 200, analysed.text
    canadian = analysed.json()["summary"]
    smoothing = next(f for f in canadian["source"]["note_facts"] if f["key"] == "smoothing")
    assert f"smoothed over {smoothing['smooth_m'] / 0.3048:.1f} ft" in \
        " ".join(canadian["source"]["notes"])

    # The picker moves. Nothing is gridded, smoothed or measured again.
    assert client.put("/api/units/display", json={"units": METRIC}).status_code == 200
    metric = client.get(f"/api/terrain/{dataset_id}").json()["summary"]

    assert metric["source"]["note_facts"] == canadian["source"]["note_facts"]
    assert f"smoothed over {smoothing['smooth_m']:.1f} m" in " ".join(metric["source"]["notes"])
    assert metric["grid"]["cell_m"] == canadian["grid"]["cell_m"]
    client.put("/api/units/display", json={"units": CANADA})


# ==========================================================================
# The first look
# ==========================================================================

def test_the_first_look_follows_the_reader(harvest):
    """Same file, two readers: the same verdict, different sentences."""
    canadian = preflight.run(harvest, CANADA)
    metric = preflight.run(harvest, METRIC)

    assert canadian["verdict"] == metric["verdict"]
    assert canadian["info"] == metric["info"]
    assert canadian["next_step"] == metric["next_step"]

    coverage_ca = next(f for f in canadian["findings"] if f["title"] == "Coverage")
    coverage_m = next(f for f in metric["findings"] if f["title"] == "Coverage")
    assert coverage_ca["detail"] != coverage_m["detail"]
    area_ha = float(canadian["info"]["area_ha"])
    assert f"{area_ha / 0.40468564224:.1f} ac" in coverage_ca["detail"]
    assert f"{area_ha:.1f} ha" in coverage_m["detail"]
    assert_no_metric(coverage_ca["detail"])

    magnitude_ca = next(f for f in canadian["findings"] if f["title"] == "Magnitude")
    magnitude_m = next(f for f in metric["findings"] if f["title"] == "Magnitude")
    assert "bu/ac" in magnitude_ca["detail"] and "kg/ha" not in magnitude_ca["detail"]
    assert "kg/ha" in magnitude_m["detail"]


def test_the_unit_doubt_reads_right_in_bushels():
    """The findings that are *about* units are written, not substituted.

    For a reader in bu/ac, "read as bu/ac this becomes 1 247 kg/ha" would
    come out as "read as bu/ac this becomes 55 bu/ac" — true, circular and
    useless. What the reader needs is the two quantities side by side: what
    the app is holding now, and what the file would be worth declared
    otherwise.
    """
    dataset = _imperial_yield_map()
    canadian = preflight.run(dataset, CANADA)
    metric = preflight.run(dataset, METRIC)

    doubt_ca = next(f for f in canadian["findings"] if f["title"] == "Units look wrong")
    doubt_m = next(f for f in metric["findings"] if f["title"] == "Units look wrong")

    # The file's median, read as kg/ha, is about 55 kg/ha — which is one
    # bushel an acre, and the sentence says so rather than repeating 55.
    median = float(canadian["info"]["median_value"])
    bushel_kg_ha = 22.6796 / 0.40468564224  # canola, kg/ha in one bu/ac
    assert f"{median / bushel_kg_ha:.2f} bu/ac as the file is being read" in doubt_ca["detail"]
    assert f"read as bu/ac would put it at {median:.1f} bu/ac" in doubt_ca["detail"]
    assert "14.3 bu/ac to 107 bu/ac" in doubt_ca["detail"]
    assert "kg/ha" not in doubt_ca["detail"]
    # The claim still names the unit the monitor may have written in.
    assert "bu/ac" in doubt_ca["action"]

    assert f"{median:.1f} kg/ha as the file is being read" in doubt_m["detail"]
    assert "800 kg/ha to 6\u2009000 kg/ha" in doubt_m["detail"]
    # Both readers are told the same thing about the same file.
    assert canadian["verdict"] == metric["verdict"] == "alert"
    assert canadian["proposed_units"] == metric["proposed_units"]


def test_a_rate_in_pounds_is_not_told_it_is_plausible_in_pounds():
    """The imperial doubt on a rate that looks fine either way.

    Its whole content is a comparison — what the app holds against what the
    file would mean — so both have to be quantities in the reader's unit.
    """
    rng = np.random.default_rng(5)
    n = 600
    frame = pd.DataFrame({
        sch.LON: -105.83 + rng.normal(0, 0.002, n),
        sch.LAT: 50.45 + rng.normal(0, 0.002, n),
        sch.VALUE: rng.normal(120.0, 4.0, n),
        sch.SPEED: rng.normal(5.4, 0.3, n),
        sch.SWATH: np.full(n, 40.0),
    })
    dataset = Dataset(frame, DatasetMeta(
        name="urea.csv", operation="application", value_label="Rate"))
    dataset.ensure_derived()

    canadian = preflight.run(dataset, CANADA)
    doubt = next(f for f in canadian["findings"] if f["title"] == "Possibly imperial units")
    held, real = re.findall(r"([\d .]+) lb/ac", doubt["detail"])
    assert float(held.replace(" ", "")) < float(real.replace(" ", ""))
    assert "holding it at" in doubt["detail"] and "the real rate is" in doubt["detail"]


# ==========================================================================
# Cleaning
# ==========================================================================

def test_the_cleaning_report_is_said_again_without_running_again(harvest):
    """Restating a stored cleaning report changes its words, not its counts."""
    stored = clean_pipeline.run(harvest).report
    canadian = clean_pipeline.restate(stored, CANADA, "harvest")
    metric = clean_pipeline.restate(stored, METRIC, "harvest")

    assert canadian["totals"] == stored["totals"] == metric["totals"]
    assert canadian["statistics"] == stored["statistics"]
    assert canadian["histogram"] == stored["histogram"]
    assert [s["removed"] for s in canadian["steps"]] == [s["removed"] for s in stored["steps"]]

    edge_ca = next(s for s in canadian["steps"] if s["key"] == "pass_ends")
    edge_m = next(s for s in metric["steps"] if s["key"] == "pass_ends")
    assert "ft" in edge_ca["detail"] and "m at the start" in edge_m["detail"]
    for step in canadian["steps"]:
        assert_no_metric(step["detail"])

    # The step that could not run keeps the sentence it already had: it has
    # no number in it, and it means the same in every unit set.
    skipped = [s for s in canadian["steps"] if s["skipped"]]
    for step in skipped:
        assert step["detail"] == next(
            s["detail"] for s in stored["steps"] if s["key"] == step["key"])


# ==========================================================================
# Refusals
# ==========================================================================

def test_an_unknown_unit_is_refused_by_name():
    with pytest.raises(ValueError) as excinfo:
        Phrase({"length_unit": "furlong"})
    message = str(excinfo.value)
    assert "'furlong' is not a valid length unit" in message
    assert "ft" in message and "m" in message  # what would have been accepted

    with pytest.raises(ValueError) as excinfo:
        Phrase({"area_unit": "football pitches"})
    assert "area unit" in str(excinfo.value)

    # And through the endpoint the interface uses, with the same sentence.
    from fastapi.testclient import TestClient

    from agrosuite.app import server as server_mod

    client = TestClient(server_mod.app)
    refused = client.put("/api/units/display", json={"units": {"length_unit": "furlong"}})
    assert refused.status_code == 400
    assert "furlong" in refused.json()["detail"]
    # Refused means unchanged: the session is still in the units it was in.
    assert server_mod.state.display_units["length_unit"] != "furlong"


# ==========================================================================
# Over the wire, and across a save
# ==========================================================================

def test_changing_the_picker_rewrites_every_sentence_on_screen(client):
    """The behaviour the whole mechanism exists for.

    One import, no re-import and no re-analysis: the first look and the
    cleaning report are asked for again and come back in the units now on
    screen, with the same numbers underneath.
    """
    client.post("/api/session/new")
    client.put("/api/units/display", json={"units": CANADA})
    loaded = client.post("/api/import/demo", json={"kind": "harvest"})
    assert loaded.status_code == 200, loaded.text
    dataset_id = loaded.json()["id"]
    assert "ac" in loaded.json()["preflight"]["findings"][0]["detail"]

    cleaned = client.post(f"/api/datasets/{dataset_id}/clean", json={})
    assert cleaned.status_code == 200, cleaned.text
    canadian_steps = {s["key"]: s["detail"] for s in cleaned.json()["report"]["steps"]}
    assert "ft" in canadian_steps["pass_ends"]

    canadian_look = client.get(f"/api/datasets/{dataset_id}").json()
    area_ha = float(canadian_look["area_ha"])

    # The picker moves. Nothing is imported, cleaned or analysed again.
    assert client.put("/api/units/display", json={"units": METRIC}).status_code == 200

    metric_look = client.get(f"/api/datasets/{dataset_id}").json()
    assert metric_look["area_ha"] == area_ha
    canadian_text = [f["detail"] for f in canadian_look["reports_data"]["preflight"]["findings"]]
    metric_text = [f["detail"] for f in metric_look["reports_data"]["preflight"]["findings"]]
    assert canadian_text != metric_text
    assert any(" ha" in t for t in metric_text)
    assert not any(METRIC_QUANTITY.search(t) for t in canadian_text if "kg/ha" not in t)

    metric_steps = {
        s["key"]: s["detail"]
        for s in client.get(f"/api/datasets/{dataset_id}/report/clean").json()["steps"]
    }
    assert "m at the start" in metric_steps["pass_ends"]
    assert metric_steps.keys() == canadian_steps.keys()


def test_a_project_opens_in_the_units_of_the_day_it_is_opened(client, tmp_path):
    """Saved in acres, reopened in hectares: the sentences follow the reader."""
    client.post("/api/session/new")
    client.put("/api/units/display", json={"units": CANADA})
    dataset_id = client.post("/api/import/demo", json={"kind": "harvest"}).json()["id"]
    saved = client.post("/api/session/save", json={
        "name": "phrasing", "path": str(tmp_path / "phrasing.agrosuite")})
    assert saved.status_code == 200, saved.text
    path = saved.json()["path"]

    # A new session, a reader who works in metric, the same file.
    client.post("/api/session/new")
    client.put("/api/units/display", json={"units": METRIC})
    opened = client.post("/api/session/open", json={"path": path})
    assert opened.status_code == 200, opened.text
    reopened_id = opened.json()["datasets"][0]["id"]
    assert reopened_id == dataset_id

    detail = client.get(f"/api/datasets/{reopened_id}").json()
    coverage = next(
        f for f in detail["reports_data"]["preflight"]["findings"]
        if f["title"] == "Coverage"
    )
    assert " ha (" in coverage["detail"] and " ac" not in coverage["detail"]

    # And back again, without reopening anything.
    client.put("/api/units/display", json={"units": CANADA})
    coverage_again = next(
        f for f in client.get(f"/api/datasets/{reopened_id}").json()
        ["reports_data"]["preflight"]["findings"]
        if f["title"] == "Coverage"
    )
    assert " ac (" in coverage_again["detail"]
    client.put("/api/units/display", json={"units": CANADA})


def test_a_raster_layer_s_own_notes_follow_the_reader(tmp_path):
    """The remarks a reader made about a file move with the unit picker.

    They were the last prose written once and left alone: a raster opened
    while the screen showed acres kept saying metres after the picker moved,
    among sentences that had all changed.
    """
    import numpy as np
    import rasterio
    from fastapi.testclient import TestClient
    from rasterio.transform import Affine

    from agrosuite.app import server as server_mod
    from agrosuite.core import units as units_mod

    # Rectangular cells, so the reader has to resample and says so — a
    # remark that carries a length, which is the point of the test.
    z = np.fromfunction(lambda r, c: 700.0 + 0.04 * c - 0.02 * r,
                        (60, 60)).astype("float32")
    path = tmp_path / "field.tif"
    with rasterio.open(
        path, "w", driver="GTiff", height=60, width=60, count=1,
        dtype="float32", crs="EPSG:32612", nodata=-9999.0,
        transform=Affine(7.0, 0, 500_000.0, 0, -11.0, 5_700_000.0),
    ) as handle:
        handle.write(z, 1)

    client = TestClient(server_mod.app)
    # The unit set belongs to the reader, not to the project, so starting a
    # new project deliberately leaves it alone — which means this test has to
    # put it back itself or every test after it reads in the wrong units.
    was = dict(server_mod.state.display_units)
    try:
        client.post("/api/session/new")
        client.put("/api/units/display",
                   json={"units": units_mod.UNIT_PRESETS["canada"]})
        opened = client.post("/api/import/path", json={"path": str(path)})
        assert opened.status_code == 200, opened.text
        dataset_id = opened.json()["id"]

        def notes_now():
            listing = client.get("/api/datasets").json()["datasets"]
            row = next(d for d in listing if d["id"] == dataset_id)
            return row["meta"]["notes"]

        imperial = notes_now()
        assert imperial, "the reader said something about the raster"
        assert any(" ft" in note for note in imperial), imperial
        assert not any(" m " in note or note.endswith(" m") for note in imperial), imperial

        client.put("/api/units/display",
                   json={"units": units_mod.UNIT_PRESETS["metric"]})
        metric = notes_now()
        assert any(" m" in note for note in metric), metric
        assert not any(" ft" in note for note in metric), metric

        # The same remarks, one for one: only the unit they are written in
        # changed, never which remarks were made.
        assert len(metric) == len(imperial)
        assert metric != imperial
    finally:
        client.put("/api/units/display", json={"units": was})
        client.post("/api/session/new")
