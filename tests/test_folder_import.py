"""Dropping a folder, or several files, on the app at once.

What these tests protect: a shapefile's four files arrive as one dataset,
an ISOXML tree keeps its structure and reads as ISOXML, unrelated files each
become their own dataset, whatever cannot be read is named with a reason,
and a path that tries to climb out of the upload folder is refused before
anything is written.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fixtures as fx  # noqa: E402

from agrosuite.app import server as server_mod  # noqa: E402
from agrosuite.app import session as session_mod  # noqa: E402
from agrosuite.app.routes import folder_import  # noqa: E402
from agrosuite.formats import registry  # noqa: E402

ENDPOINT = "/api/import/files"


@pytest.fixture
def client(monkeypatch):
    """The real app over a fresh session, so datasets never leak between tests."""
    fresh = session_mod.Session()
    monkeypatch.setattr(server_mod, "state", fresh)
    with TestClient(server_mod.app) as client:
        yield client
    fresh.cleanup()


def _drop(client: TestClient, entries: list[tuple[str, Path]], with_paths: bool = True):
    """POST files as the browser does: one 'files' part per file, plus 'paths'."""
    files = [("files", (Path(relative).name, path.read_bytes())) for relative, path in entries]
    data = {"paths": [relative for relative, _ in entries]} if with_paths else {}
    return client.post(ENDPOINT, files=files, data=data)


def _shapefile_set(shp: Path) -> list[Path]:
    return [p for p in shp.parent.iterdir() if p.stem == shp.stem]


# ==========================================================================
# The four cases the feature exists for
# ==========================================================================

def test_shapefile_set_becomes_one_dataset(client, tmp_path):
    shp = fx.john_deere_shapefile(tmp_path / "john_deere")
    members = _shapefile_set(shp)
    assert len(members) >= 4, "the fixture should write .shp, .shx, .dbf and .prj"

    response = _drop(client, [(f"john_deere/{p.name}", p) for p in members])
    assert response.status_code == 200, response.text
    body = response.json()

    assert len(body["imported"]) == 1
    assert body["skipped"] == []
    summary = body["imported"][0]
    assert summary["label"] == shp.stem
    assert summary["origin"] == "upload"
    assert summary["meta"]["brand"] == "john_deere"
    assert summary["rows"] > 100
    # Registered through the server, so the preliminary check and role are there.
    assert "preflight" in summary and summary["role"]
    assert [item["id"] for item in server_mod.state.list()] == [summary["id"]]


def test_taskdata_folder_reads_as_isoxml(client, tmp_path):
    taskdata = fx.isoxml_with_log(tmp_path / "cnh")
    members = sorted(taskdata.iterdir())
    assert [p.name for p in members] == ["TASKDATA.XML", "TLG00001.BIN", "TLG00001.XML"]
    expected = registry.read_any(taskdata).meta

    response = _drop(client, [(f"TASKDATA/{p.name}", p) for p in members])
    assert response.status_code == 200, response.text
    body = response.json()

    assert len(body["imported"]) == 1
    assert body["skipped"] == []
    meta = body["imported"][0]["meta"]
    assert meta["source_format"] == "isoxml"
    assert meta["brand"] == expected.brand == "case_ih"
    assert body["imported"][0]["rows"] > 100
    # The tree was kept: the log was decoded, which needs the .BIN beside its header.
    assert body["imported"][0]["label"] == expected.name == "NW-14-32-W2"


def test_three_unrelated_csvs_become_three_datasets(client, tmp_path):
    sources = [
        fx.raven_viper_csv(tmp_path / "raven"),
        fx.trimble_csv(tmp_path / "trimble"),
        fx.bourgault_csv(tmp_path / "bourgault"),
    ]
    response = _drop(client, [(f"logs/{p.name}", p) for p in sources])
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["skipped"] == []
    assert sorted(item["label"] for item in body["imported"]) == sorted(p.stem for p in sources)
    assert {item["meta"]["brand"] for item in body["imported"]} == {"raven", "trimble", "bourgault"}
    assert len(server_mod.state.list()) == 3


@pytest.mark.parametrize("bad", ["../escape.csv", "logs/../../escape.csv", "/etc/escape.csv", "C:/escape.csv"])
def test_path_outside_the_drop_is_refused_before_writing(client, tmp_path, bad):
    csv = fx.raven_viper_csv(tmp_path / "raven")
    before = set(server_mod.state.uploads.iterdir())

    response = _drop(client, [("logs/fine.csv", csv), (bad, csv)])
    assert response.status_code == 400
    assert "refused" in response.json()["detail"] or "absolute" in response.json()["detail"]
    assert server_mod.state.list() == [], "a refused drop must import nothing"
    assert set(server_mod.state.uploads.iterdir()) == before, "and must write nothing"


# ==========================================================================
# Around the edges
# ==========================================================================

def test_john_deere_card_is_opened_as_the_card(client, tmp_path):
    """The card's SETUP folder alone is not a card: the walk into the drop
    must stop at GS3_2630, or the boundary loses the card that explains it
    and the proprietary setup file is reported as if it were a stray."""
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import Polygon

    card = tmp_path / "GS3_2630"
    (card / "SETUP").mkdir(parents=True)
    quarter = Polygon([(-105.8340, 50.4520), (-105.8229, 50.4520),
                       (-105.8229, 50.4592), (-105.8340, 50.4592)])
    gpd.GeoDataFrame(pd.DataFrame([{"NAME": "NW-14-32-W2"}]), geometry=[quarter],
                     crs="EPSG:4326").to_file(card / "SETUP" / "Boundary.shp")
    (card / "SETUP" / "Setup.jdf").write_bytes(b"JDF" + bytes(64))

    entries = [(f"GS3_2630/{p.relative_to(card).as_posix()}", p) for p in card.rglob("*") if p.is_file()]
    response = _drop(client, entries)
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["skipped"] == []
    assert len(body["imported"]) == 1
    summary = body["imported"][0]
    assert summary["label"] == "GS3_2630"
    assert summary["meta"]["brand"] == "john_deere"
    assert "jd_card" in summary["meta"]["extra"]


def test_two_shapefile_sets_become_two_datasets(client, tmp_path):
    first = fx.john_deere_shapefile(tmp_path / "sets")
    second = fx.ag_leader_shapefile(tmp_path / "sets")
    entries = [(f"sets/{p.name}", p) for shp in (first, second) for p in _shapefile_set(shp)]

    response = _drop(client, entries)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["skipped"] == []
    assert sorted(item["label"] for item in body["imported"]) == sorted([first.stem, second.stem])
    assert {item["meta"]["brand"] for item in body["imported"]} == {"john_deere", "ag_leader"}


def test_loose_files_without_paths_use_their_names(client, tmp_path):
    csv = fx.trimble_csv(tmp_path / "trimble")
    response = _drop(client, [(csv.name, csv)], with_paths=False)
    assert response.status_code == 200, response.text
    assert [item["label"] for item in response.json()["imported"]] == [csv.stem]


def test_siblings_of_a_taskdata_folder_are_not_lost(client, tmp_path):
    """A field folder holding TASKDATA plus a yield CSV yields both, and the
    unreadable extra is named rather than dropped."""
    taskdata = fx.isoxml_with_log(tmp_path / "field")
    csv = fx.raven_viper_csv(tmp_path / "field")
    note = tmp_path / "field" / "notes.pdf"
    note.write_bytes(b"%PDF-1.4 not really")

    entries = [(f"Field 12/TASKDATA/{p.name}", p) for p in taskdata.iterdir()]
    entries += [(f"Field 12/{csv.name}", csv), ("Field 12/notes.pdf", note)]
    response = _drop(client, entries)
    assert response.status_code == 200, response.text
    body = response.json()

    assert sorted(item["meta"]["source_format"] for item in body["imported"]) == ["csv", "isoxml"]
    assert body["skipped"] == [{
        "name": "Field 12/notes.pdf",
        "reason": "Extension '.pdf' is not supported on import.",
    }]


def test_lone_shp_is_named_not_swallowed(client, tmp_path):
    shp = fx.john_deere_shapefile(tmp_path / "jd")
    csv = fx.trimble_csv(tmp_path / "trimble")
    response = _drop(client, [("drop/JD_Colheita_Canola.shp", shp), (f"drop/{csv.name}", csv)])
    assert response.status_code == 200, response.text
    body = response.json()

    assert [item["label"] for item in body["imported"]] == [csv.stem]
    assert len(body["skipped"]) == 1
    assert body["skipped"][0]["name"] == "drop/JD_Colheita_Canola.shp"
    assert ".shx and .dbf" in body["skipped"][0]["reason"]


def test_nothing_importable_is_a_clear_400(client, tmp_path):
    note = tmp_path / "notes.pdf"
    note.write_bytes(b"%PDF-1.4 not really")
    response = _drop(client, [("drop/notes.pdf", note)])
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "Nothing in the drop could be imported" in detail
    assert "drop/notes.pdf" in detail and "'.pdf' is not supported" in detail


def test_duplicate_paths_are_refused(client, tmp_path):
    csv = fx.trimble_csv(tmp_path / "trimble")
    response = _drop(client, [("a/log.csv", csv), ("a/log.csv", csv)])
    assert response.status_code == 400
    assert "appears twice" in response.json()["detail"]


def test_mismatched_counts_are_refused(client, tmp_path):
    csv = fx.trimble_csv(tmp_path / "trimble")
    response = client.post(
        ENDPOINT,
        files=[("files", (csv.name, csv.read_bytes())), ("files", (csv.name, csv.read_bytes()))],
        data={"paths": ["only/one.csv"]},
    )
    assert response.status_code == 400
    assert "2 file(s) arrived with 1 path(s)" in response.json()["detail"]


@pytest.mark.parametrize("raw,expected", [
    ("TASKDATA/TASKDATA.XML", "TASKDATA/TASKDATA.XML"),
    ("./a/./b.csv", "a/b.csv"),
    ("a\\b\\c.shp", "a/b/c.shp"),
    ("  folder/file.csv  ", "folder/file.csv"),
])
def test_relative_paths_are_normalised(raw, expected):
    assert folder_import.safe_relative_path(raw).as_posix() == expected


@pytest.mark.parametrize("raw", ["", "   ", "..", "a/../b", "/abs.csv", "\\\\server\\share.csv", "D:\\x.csv", "bad\x00.csv"])
def test_unsafe_paths_raise(raw):
    with pytest.raises(ValueError):
        folder_import.safe_relative_path(raw)


def test_router_is_mounted():
    assert server_mod.app.url_path_for("import_files") == ENDPOINT
