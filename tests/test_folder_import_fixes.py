"""What a drop must never do: lose a file quietly, name a temporary path, or
hand the browser a file to open in place of the app.

Server side: a corrupt shapefile is named by the file the user dropped, not
by the upload folder; sidecars without their .shp get the shapefile advice;
a boundary lying beside a TASKDATA.XML is imported, not swallowed by the
ISOXML; an ArcGIS ``.shp.xml`` is part of the shapefile; a readme with no
records is a skip, not a dataset. Client side, the contract the page has to
keep: the drag listeners cover the whole document (a busy #app has pointer
events off, and a drop that reaches the body unhandled opens the file in the
tab), only a file drag lights the outline, an empty drop says so, and the
skipped-files fold never describes a drop other than the last one.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fixtures as fx  # noqa: E402

from agrosuite.app import server as server_mod  # noqa: E402
from agrosuite.app import session as session_mod  # noqa: E402

ENDPOINT = "/api/import/files"
STATIC = Path(__file__).resolve().parents[1] / "agrosuite" / "app" / "static"
APP_JS = (STATIC / "app.js").read_text(encoding="utf-8")


@pytest.fixture
def client(monkeypatch):
    fresh = session_mod.Session()
    monkeypatch.setattr(server_mod, "state", fresh)
    with TestClient(server_mod.app) as client:
        yield client
    fresh.cleanup()


def _drop(client: TestClient, entries: list[tuple[str, bytes]]):
    """POST files as the browser does: one 'files' part per file, plus 'paths'."""
    files = [("files", (Path(relative).name, data)) for relative, data in entries]
    return client.post(ENDPOINT, files=files, data={"paths": [relative for relative, _ in entries]})


def _shapefile_set(shp: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in shp.parent.iterdir() if p.stem == shp.stem}


def _boundary(folder: Path, name: str = "boundary") -> Path:
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import Polygon

    folder.mkdir(parents=True, exist_ok=True)
    quarter = Polygon([(-105.8340, 50.4520), (-105.8229, 50.4520),
                       (-105.8229, 50.4592), (-105.8340, 50.4592)])
    gpd.GeoDataFrame(pd.DataFrame([{"NAME": "NW-14-32-W2"}]), geometry=[quarter],
                     crs="EPSG:4326").to_file(folder / f"{name}.shp")
    return folder / f"{name}.shp"


# ==========================================================================
# Server: names and reasons
# ==========================================================================

@pytest.mark.parametrize("prefix", ["", "Field/"])
def test_a_corrupt_shp_in_a_complete_set_is_named_by_its_file_not_the_upload_folder(
        client, tmp_path, prefix):
    parts = _shapefile_set(fx.john_deere_shapefile(tmp_path / "jd"))
    entries = [(f"{prefix}bad{Path(name).suffix}", b"garbage" * 16 if name.endswith(".shp") else data)
               for name, data in parts.items()]
    response = _drop(client, entries)
    assert response.status_code == 400, response.text
    detail = response.json()["detail"]
    assert detail.startswith(f"Nothing in the drop could be imported. {prefix}bad.shp: ")
    assert ". .:" not in detail and "the dropped files" not in detail
    assert str(server_mod.state.uploads) not in detail, "a temporary path the user never saw"
    assert server_mod.state.list() == []


@pytest.mark.parametrize("prefix", ["", "Field/"])
def test_sidecars_without_their_shp_get_the_shapefile_advice(client, tmp_path, prefix):
    parts = _shapefile_set(fx.john_deere_shapefile(tmp_path / "jd"))
    entries = [(f"{prefix}a{Path(name).suffix}", data) for name, data in parts.items()
               if not name.endswith(".shp")]
    assert len(entries) >= 3
    response = _drop(client, entries)
    assert response.status_code == 400, response.text
    detail = response.json()["detail"]
    assert f"{prefix}a.dbf: 'a.dbf' belongs to a shapefile whose .shp is not in the drop" in detail
    assert "holds nothing that can be imported" not in detail
    assert ". .:" not in detail


def test_a_drop_of_only_hidden_files_says_so_without_a_dot_name(client):
    response = _drop(client, [(".DS_Store", b"\x00" * 8)])
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "The drop holds nothing that can be imported" in detail
    assert ". .:" not in detail


# ==========================================================================
# Server: what sits beside a TASKDATA.XML, and what belongs to a shapefile
# ==========================================================================

def test_files_beside_a_taskdata_xml_are_imported_or_named_never_swallowed(client, tmp_path):
    taskdata = fx.isoxml_with_log(tmp_path / "cnh")
    shp = _boundary(tmp_path / "bnd")
    entries = [(f"Field/{p.name}", p.read_bytes()) for p in taskdata.iterdir()]
    entries += [(f"Field/{name}", data) for name, data in _shapefile_set(shp).items()]
    entries += [("Field/notes.pdf", b"%PDF-1.4 not really")]

    response = _drop(client, entries)
    assert response.status_code == 200, response.text
    body = response.json()
    assert sorted((i["label"], i["meta"]["source_format"]) for i in body["imported"]) == [
        ("Field", "isoxml"), ("boundary", "shapefile")]
    # The TLG header is part of the ISOXML: walked on its own it would import
    # the same task a second time.
    assert [i["meta"]["source_format"] for i in body["imported"]].count("isoxml") == 1
    assert body["skipped"] == [{
        "name": "Field/notes.pdf", "reason": "Extension '.pdf' is not supported on import."}]


def test_a_taskdata_folder_holding_nothing_else_is_still_one_clean_dataset(client, tmp_path):
    taskdata = fx.isoxml_with_log(tmp_path / "cnh")
    response = _drop(client, [(f"Field/{p.name}", p.read_bytes()) for p in taskdata.iterdir()])
    assert response.status_code == 200, response.text
    body = response.json()
    assert [i["label"] for i in body["imported"]] == ["Field"]
    assert body["skipped"] == [], "the folder was imported; it does not 'hold nothing'"


def test_arcgis_metadata_beside_a_shapefile_is_part_of_it(client, tmp_path):
    shp = fx.john_deere_shapefile(tmp_path / "jd")
    entries = [(f"Field/{name}", data) for name, data in _shapefile_set(shp).items()]
    entries.append((f"Field/{shp.name}.xml", b"<metadata><Esri/></metadata>"))
    response = _drop(client, entries)
    assert response.status_code == 200, response.text
    body = response.json()
    assert [i["label"] for i in body["imported"]] == [shp.stem]
    assert body["skipped"] == []


def test_arcgis_metadata_without_its_shp_is_named_as_a_shapefile_part(client, tmp_path):
    csv = fx.trimble_csv(tmp_path / "trimble")
    response = _drop(client, [("drop/lonely.shp.xml", b"<metadata/>"), (f"drop/{csv.name}", csv.read_bytes())])
    assert response.status_code == 200, response.text
    body = response.json()
    assert [i["label"] for i in body["imported"]] == [csv.stem]
    assert body["skipped"] == [{
        "name": "drop/lonely.shp.xml",
        "reason": "'lonely.shp.xml' belongs to a shapefile whose .shp is not in the drop. "
                  "Drop the folder holding all of its files (.shp, .shx, .dbf, .prj).",
    }]


# ==========================================================================
# Server: a readme is not a dataset
# ==========================================================================

def test_a_text_file_with_no_records_is_a_skip_not_a_dataset(client, tmp_path):
    """A header with columns and nothing under it: the walk lets it through
    to the reader (it is shaped like a table), and the reader names it."""
    csv = fx.raven_viper_csv(tmp_path / "raven")
    response = _drop(client, [("Field/notes.txt", b"Longitude,Latitude,Yield\n"),
                              (f"Field/{csv.name}", csv.read_bytes())])
    assert response.status_code == 200, response.text
    body = response.json()
    assert [i["label"] for i in body["imported"]] == [csv.stem]
    assert [i["name"] for i in body["skipped"]] == ["Field/notes.txt"]
    assert "holds no records" in body["skipped"][0]["reason"]
    assert all(len(server_mod.state.get(item["id"]).dataset.df) > 0 for item in server_mod.state.list())


def test_a_header_only_export_is_refused_by_the_reader_too(tmp_path):
    from agrosuite.formats import registry

    empty = tmp_path / "Yield.csv"
    empty.write_text("Longitude,Latitude,Yield\n", encoding="utf-8")
    with pytest.raises(ValueError, match="holds no records"):
        registry.read_any(empty)


# ==========================================================================
# Client: the contract the page keeps
# ==========================================================================

def _block(name: str) -> str:
    match = re.search(rf"^  (?:async )?{name}\(", APP_JS, flags=re.MULTILINE)
    assert match, f"App.{name} is not defined"
    return APP_JS[match.start():APP_JS.index("\n  },\n", match.start())]


def test_drag_listeners_cover_the_document_not_only_the_app():
    """#app loses pointer events while busy, so a drag over it lands on the
    body; unhandled there, a dropped file replaces the page."""
    imports = _block("bindImport")
    for kind in ("dragenter", "dragover", "drop"):
        assert f'document.addEventListener("{kind}"' in imports, f"{kind} must be on the document"
        assert f'dropZone.addEventListener("{kind}"' not in imports
    style = (STATIC / "style.css").read_text(encoding="utf-8")
    assert re.search(r"\.busy\s*\{[^}]*pointer-events:\s*none", style), "the reason the listeners moved"
    # A drop while a request is out, or a dialog is up, is refused with a
    # word — before the drop is read, never handed back to the browser.
    blocked = imports[imports.index("const blocked = ()"):]
    assert 'querySelector(".busy")' in blocked and 'querySelector("dialog[open]")' in blocked
    drop = imports[imports.index('document.addEventListener("drop"'):]
    assert drop.index("blocked()") < drop.index("this.collectDropped(")
    assert "e.preventDefault()" in drop


def test_only_a_file_drag_lights_the_outline():
    imports = _block("bindImport")
    assert 'includes("Files")' in imports
    enter = imports[imports.index('document.addEventListener("dragenter"'):]
    assert enter.index("carriesFiles(e)") < enter.index('classList.add("dropping")')


def test_an_empty_drop_or_folder_is_answered():
    imports = _block("bindImport")
    assert imports.count('this.toast("Nothing to import"') == 2, "the drop zone and the folder picker"


def test_the_skipped_fold_never_describes_an_older_drop():
    block = _block("importFiles")
    cleared = block.index("this.renderDropReport([])")
    assert cleared < block.index("maxFiles"), "cleared before the cap can refuse the drop"
    assert cleared < block.index('"/api/import/files"'), "and before a 400 can"
    assert "this.renderDropReport([])" in _block("resetSessionState"), "New project and Open project"
