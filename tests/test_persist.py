"""Saving a session and reopening it.

What these tests protect is the promise that closing the app loses nothing:
the datasets come back with the same ids, types and geometry, the cleaning
report and removal reasons are still there, the roles and prices are what
they were, and a file the app did not write is refused before it can do
any harm to the session that is open.
"""

from __future__ import annotations

import asyncio
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agrosuite.app import persist
from agrosuite.app import session as session_mod
from agrosuite.app.routes import persist as persist_routes
from agrosuite.clean import pipeline as clean_pipeline
from agrosuite.core import preflight
from agrosuite.core import schema as sch
from agrosuite.demo import synthetic_harvest
from agrosuite.formats import registry

SAVED_AT = "2026-09-17T10:00:00-06:00"


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch) -> Path:
    """Keep the recent list out of the real home folder."""
    target = tmp_path / "agrosuite-home"
    monkeypatch.setenv("AGROSUITE_HOME", str(target))
    return target


def _polygon_shapefile(folder: Path) -> Path:
    """A three-cell prescription: enough to carry geometry and a text column."""
    import geopandas as gpd
    from shapely.geometry import Polygon

    cells = [
        Polygon([
            (-105.8340 + i * 0.001, 50.4520), (-105.8330 + i * 0.001, 50.4520),
            (-105.8330 + i * 0.001, 50.4530), (-105.8340 + i * 0.001, 50.4530),
        ])
        for i in range(3)
    ]
    frame = pd.DataFrame({"Tgt_Rate_O": [60.0, 120.0, 180.0], "zone": ["low", "mid", "high"]})
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "rx.shp"
    gpd.GeoDataFrame(frame, geometry=cells, crs="EPSG:4326").to_file(path, driver="ESRI Shapefile")
    return path


def _populated_session(tmp_path: Path) -> tuple[session_mod.Session, dict[str, str]]:
    """A session as the app would leave it after a morning's work."""
    state = session_mod.Session()

    harvest = state.add(synthetic_harvest(), "Harvest with defects (demo)", "demo")
    harvest.reports["preflight"] = preflight.run(harvest.dataset)
    harvest.reports["preflight"]["marker"] = "kept from the saved file"
    harvest.role = "yield"

    result = clean_pipeline.run(harvest.dataset, clean_pipeline.PRESETS["harvest"])
    clean = state.add(result.clean, f"{harvest.label} · clean", "clean", harvest.id)
    removed = state.add(result.removed, f"{harvest.label} · removed", "clean_removed", harvest.id)
    clean.reports["clean"] = result.report
    harvest.reports["clean"] = result.report
    clean.role = "yield"

    polygons = state.add(registry.read_any(_polygon_shapefile(tmp_path / "rx")), "rx", "path")
    polygons.role = "plan"

    state.project.update({
        "name": "North quarter / 2025",
        "goal": "difm",
        "roles": {clean.id: "yield", polygons.id: "plan"},
        "prices": {"crop_price": 0.55, "input_cost": 1.35, "currency": "CAD", "crop": "canola"},
        "reviewed": {clean.id, polygons.id},
        "exported": True,
    })
    ids = {"harvest": harvest.id, "clean": clean.id, "removed": removed.id, "polygons": polygons.id}
    return state, ids


# ==========================================================================
# The round trip
# ==========================================================================

def test_round_trip_restores_the_session(tmp_path):
    state, ids = _populated_session(tmp_path)
    target = tmp_path / "north.agrosuite"
    result = persist.save_session(state, target, SAVED_AT)
    assert result["datasets"] == 4 and result["saved_at"] == SAVED_AT

    fresh = session_mod.Session()
    loaded = persist.load_session(fresh, target)

    # Ids, order and lineage are what the reports and roles point at.
    assert [item["id"] for item in fresh.list()] == [item["id"] for item in state.list()]
    assert [item["id"] for item in loaded["datasets"]] == list(ids.values())
    for key, dataset_id in ids.items():
        before, after = state.get(dataset_id), fresh.get(dataset_id)
        assert after.label == before.label
        assert after.origin == before.origin
        assert after.parent_id == before.parent_id
        assert after.role == before.role
        assert sorted(after.reports) == sorted(before.reports), key
        assert len(after.dataset) == len(before.dataset), key
        assert after.dataset.metric_crs == before.dataset.metric_crs
        assert after.dataset.meta.to_dict() == before.dataset.meta.to_dict()
        assert dict(after.dataset.df.dtypes.astype(str)) == dict(before.dataset.df.dtypes.astype(str)), key

    clean_after = fresh.get(ids["clean"]).dataset.df
    clean_before = state.get(ids["clean"]).dataset.df
    assert pd.api.types.is_datetime64_any_dtype(clean_after[sch.TIMESTAMP])
    assert clean_after[sch.TIMESTAMP].equals(clean_before[sch.TIMESTAMP])
    assert np.array_equal(clean_after[sch.VALUE].to_numpy(), clean_before[sch.VALUE].to_numpy(), equal_nan=True)
    # Derived fields are restored, not recomputed: the same x/y to the bit.
    assert np.array_equal(clean_after[[sch.X, sch.Y]].to_numpy(), clean_before[[sch.X, sch.Y]].to_numpy())

    removed_after = fresh.get(ids["removed"]).dataset.df
    assert removed_after["removal_reason"].tolist() == state.get(ids["removed"]).dataset.df["removal_reason"].tolist()
    assert "Swath overlap" in set(removed_after["removal_reason"])

    # The saved report is the record; nothing was re-run on the way in.
    assert fresh.get(ids["harvest"]).reports["preflight"]["marker"] == "kept from the saved file"
    assert fresh.get(ids["clean"]).reports["clean"]["totals"] == state.get(ids["clean"]).reports["clean"]["totals"]

    project = fresh.project
    assert project["name"] == "North quarter / 2025"
    assert project["prices"] == state.project["prices"]
    assert project["roles"] == state.project["roles"]
    assert isinstance(project["reviewed"], set) and project["reviewed"] == {ids["clean"], ids["polygons"]}
    assert project["exported"] is True
    assert loaded["project"] == "North quarter / 2025"


def test_geometry_round_trips_exactly(tmp_path):
    state, ids = _populated_session(tmp_path)
    target = tmp_path / "north.agrosuite"
    persist.save_session(state, target, SAVED_AT)

    fresh = session_mod.Session()
    persist.load_session(fresh, target)

    before = state.get(ids["polygons"]).dataset
    after = fresh.get(ids["polygons"]).dataset
    assert after.geometry is not None and len(after.geometry) == len(before.geometry) == 3
    for original, restored in zip(before.geometry, after.geometry):
        assert restored.equals_exact(original, 0)
        assert restored.wkb == original.wkb
    assert after.df["zone"].tolist() == ["low", "mid", "high"]
    assert after.meta.operation == "prescription"
    # Point datasets stay point datasets.
    assert fresh.get(ids["harvest"]).dataset.geometry is None


def test_project_file_layout(tmp_path):
    state, ids = _populated_session(tmp_path)
    target = tmp_path / "north.agrosuite"
    persist.save_session(state, target, SAVED_AT)

    with zipfile.ZipFile(target) as zf:
        names = set(zf.namelist())
        manifest = json.loads(zf.read("manifest.json"))
    assert "manifest.json" in names
    assert {f"data/{dataset_id}.parquet" for dataset_id in ids.values()} <= names
    assert f"data/{ids['polygons']}.geometry.parquet" in names
    assert f"data/{ids['harvest']}.geometry.parquet" not in names

    assert manifest["app"] == "AgroSuite"
    assert manifest["format"] == persist.FORMAT_VERSION
    assert manifest["saved_at"] == SAVED_AT
    assert sorted(manifest["project"]["reviewed"]) == sorted([ids["clean"], ids["polygons"]])
    records = {record["id"]: record for record in manifest["datasets"]}
    assert records[ids["clean"]]["parent_id"] == ids["harvest"]
    assert records[ids["polygons"]]["has_geometry"] is True
    assert records[ids["harvest"]]["meta"]["crop"] == state.get(ids["harvest"]).dataset.meta.crop
    assert "clean" in records[ids["clean"]]["reports"]


def test_load_replaces_the_current_session(tmp_path):
    state, ids = _populated_session(tmp_path)
    target = tmp_path / "north.agrosuite"
    persist.save_session(state, target, SAVED_AT)

    other = session_mod.Session()
    stray = other.add(synthetic_harvest(seed=1), "stray", "demo")
    other.project["name"] = "Something else"
    other.project["reviewed"].add(stray.id)

    persist.load_session(other, target)
    assert stray.id not in {item["id"] for item in other.list()}
    assert set(item["id"] for item in other.list()) == set(ids.values())
    assert other.project["name"] == "North quarter / 2025"
    assert stray.id not in other.project["reviewed"]


def test_damaged_files_are_refused_before_touching_the_session(tmp_path):
    state = session_mod.Session()
    kept = state.add(synthetic_harvest(seed=2), "kept", "demo")
    state.project["name"] = "Still here"

    not_zip = tmp_path / "notes.agrosuite"
    not_zip.write_text("this is not a project", encoding="utf-8")
    with pytest.raises(ValueError, match="not a ZIP archive"):
        persist.load_session(state, not_zip)

    no_manifest = tmp_path / "data.agrosuite"
    with zipfile.ZipFile(no_manifest, "w") as zf:
        zf.writestr("readme.txt", "monitor export")
    with pytest.raises(ValueError, match="no manifest"):
        persist.load_session(state, no_manifest)

    newer = tmp_path / "future.agrosuite"
    with zipfile.ZipFile(newer, "w") as zf:
        zf.writestr("manifest.json", json.dumps({
            "app": "AgroSuite", "format": persist.FORMAT_VERSION + 1, "datasets": [],
        }))
    with pytest.raises(ValueError, match="newer AgroSuite"):
        persist.load_session(state, newer)

    with pytest.raises(FileNotFoundError):
        persist.load_session(state, tmp_path / "missing.agrosuite")

    assert [item["id"] for item in state.list()] == [kept.id]
    assert state.project["name"] == "Still here"


def test_reused_ids_are_refused():
    """Two entries under one id would silently swap a dataset for another."""
    state = session_mod.Session()
    first = state.add(synthetic_harvest(seed=3), "first", "demo", dataset_id="abc123")
    assert first.id == "abc123"
    with pytest.raises(ValueError, match="already in use"):
        state.add(synthetic_harvest(seed=4), "second", "demo", dataset_id="abc123")


# ==========================================================================
# Recent files
# ==========================================================================

def test_recent_files_keep_the_last_ten_deduplicated(tmp_path, home):
    state, _ = _populated_session(tmp_path)
    saved = tmp_path / "real.agrosuite"
    persist.save_session(state, saved, SAVED_AT)

    for index in range(12):
        persist.remember_recent(tmp_path / f"project-{index:02d}.agrosuite", f"2026-09-{index + 1:02d}T08:00:00")
    recent = persist.recent_files()
    assert len(recent) == persist.RECENT_LIMIT == 10
    assert recent[0]["name"] == "project-11" and recent[-1]["name"] == "project-02"
    assert all(item["exists"] is False for item in recent)

    # Reopening an old one moves it to the top without duplicating it.
    persist.remember_recent(tmp_path / "project-05.agrosuite")
    recent = persist.recent_files()
    assert recent[0]["name"] == "project-05"
    assert [item["name"] for item in recent].count("project-05") == 1
    assert len(recent) == 10

    # A real file reports that it exists, and its saved_at is read from the manifest.
    persist.remember_recent(saved)
    top = persist.recent_files()[0]
    assert top["path"] == str(saved.resolve())
    assert top["exists"] is True and top["saved_at"] == SAVED_AT

    stored = json.loads((home / "recent.json").read_text(encoding="utf-8"))
    assert len(stored["recent"]) == 10


def test_recent_list_starts_empty_and_survives_damage(home):
    assert persist.recent_files() == []
    home.mkdir(parents=True)
    (home / "recent.json").write_text("{not json", encoding="utf-8")
    assert persist.recent_files() == []


# ==========================================================================
# The registry knows the file, and refuses to import it
# ==========================================================================

def test_registry_refuses_project_files(tmp_path):
    state, _ = _populated_session(tmp_path)
    target = tmp_path / "north.agrosuite"
    persist.save_session(state, target, SAVED_AT)

    assert ".agrosuite" in registry.ALL_IMPORT_EXT, "the file browser must list project files"
    with pytest.raises(ValueError, match="Open project"):
        registry.detect(target)
    with pytest.raises(ValueError, match="saved AgroSuite project"):
        registry.read_any(target)


# ==========================================================================
# The endpoints
# ==========================================================================

@pytest.fixture
def server(tmp_path):
    """The real app state, emptied before and after so tests do not leak."""
    from agrosuite.app import server as server_mod

    persist_routes.new_session()
    yield server_mod
    persist_routes.new_session()


def test_router_is_mounted(server):
    assert server.app.url_path_for("save_session") == "/api/session/save"
    assert server.app.url_path_for("open_session") == "/api/session/open"
    assert server.app.url_path_for("upload_session") == "/api/session/upload"
    assert server.app.url_path_for("recent_sessions") == "/api/session/recent"
    assert server.app.url_path_for("new_session") == "/api/session/new"


def test_save_open_and_new_through_the_endpoints(server, tmp_path):
    from fastapi import HTTPException

    state = server.state
    with pytest.raises(HTTPException) as refused:
        persist_routes.save_session(persist_routes.SaveRequest())
    assert refused.value.status_code == 400 and "Nothing to save" in refused.value.detail

    harvest = server._register(synthetic_harvest(), "Harvest with defects (demo)", "demo")
    cleaned = server.clean_dataset(harvest["id"], server.CleanRequest())
    state.project["name"] = "Quarter: NW 14-32-W2"
    state.project["prices"] = {"crop_price": 0.5, "input_cost": 1.2, "currency": "CAD", "crop": "canola"}
    ids_before = [item["id"] for item in state.list()]
    # Cleaning hands the parent's role to the clean copy; what must survive
    # the file is the roles as they stood at save time, whatever they were.
    roles_before = {item["id"]: state.get(item["id"]).role for item in state.list()}
    assert cleaned["clean"]["id"] in ids_before

    saved = persist_routes.save_session(persist_routes.SaveRequest())
    path = Path(saved["path"])
    assert path.parent == state.exports and path.name == "Quarter- NW 14-32-W2.agrosuite"
    assert saved["datasets"] == 3 and saved["name"] == "Quarter- NW 14-32-W2"
    token = saved["download_url"].rsplit("/", 1)[-1]
    assert state.file_for(token) == path
    assert persist_routes.recent_sessions()["recent"][0]["path"] == str(path.resolve())

    # Re-saving the file the session is on is the normal flow and asks nothing.
    again = persist_routes.save_session(persist_routes.SaveRequest())
    assert again["path"] == saved["path"]

    # A path ending in .agrosuite is the file (and needs no name); anything
    # else is a folder — there yet or not — with a file named after the
    # project inside it.
    elsewhere = persist_routes.save_session(
        persist_routes.SaveRequest(path=str(tmp_path / "copy.agrosuite")))
    assert elsewhere["path"] == str(tmp_path / "copy.agrosuite")
    in_folder = persist_routes.save_session(persist_routes.SaveRequest(path=str(tmp_path)))
    assert Path(in_folder["path"]) == tmp_path / "Quarter- NW 14-32-W2.agrosuite"
    new_folder = persist_routes.save_session(persist_routes.SaveRequest(path=str(tmp_path / "copy")))
    assert Path(new_folder["path"]) == tmp_path / "copy" / "Quarter- NW 14-32-W2.agrosuite"

    # Saving over any other existing file is a decision, not a default.
    with pytest.raises(HTTPException) as conflict:
        persist_routes.save_session(persist_routes.SaveRequest())
    assert conflict.value.status_code == 409
    replaced = persist_routes.save_session(persist_routes.SaveRequest(overwrite=True))
    assert replaced["path"] == saved["path"]

    persist_routes.new_session()
    assert state.list() == []
    assert state.project == session_mod.default_project()

    opened = persist_routes.open_session(persist_routes.OpenRequest(path=f'"{path}"'))
    assert [item["id"] for item in opened["datasets"]] == ids_before
    assert opened["project"] == "Quarter: NW 14-32-W2"
    # The file was last written by the confirmed replacement.
    assert opened["saved_at"] == replaced["saved_at"]
    assert state.project["prices"]["crop_price"] == 0.5
    assert state.get(cleaned["clean"]["id"]).reports["clean"]["totals"] == cleaned["report"]["totals"]
    assert {item["id"]: state.get(item["id"]).role for item in state.list()} == roles_before
    assert state.project["roles"] == {k: v for k, v in roles_before.items() if v}

    with pytest.raises(HTTPException) as not_found:
        persist_routes.open_session(persist_routes.OpenRequest(path=str(tmp_path / "gone.agrosuite")))
    assert not_found.value.status_code == 404
    # A folder exists; "not found" would send the person looking for it.
    with pytest.raises(HTTPException) as folder:
        persist_routes.open_session(persist_routes.OpenRequest(path=str(tmp_path)))
    assert folder.value.status_code == 400 and "That is a folder" in folder.value.detail
    assert ".agrosuite file inside it" in folder.value.detail

    bogus = tmp_path / "bogus.agrosuite"
    bogus.write_bytes(b"not a zip")
    with pytest.raises(HTTPException) as bad:
        persist_routes.open_session(persist_routes.OpenRequest(path=str(bogus)))
    assert bad.value.status_code == 400 and "not a ZIP" in bad.value.detail
    assert [item["id"] for item in state.list()] == ids_before, "a bad file must not empty the session"


def test_upload_endpoint_loads_the_file(server, tmp_path):
    from fastapi import HTTPException
    from starlette.datastructures import UploadFile

    state = server.state
    harvest = server._register(synthetic_harvest(), "Harvest with defects (demo)", "demo")
    saved = persist_routes.save_session(persist_routes.SaveRequest(name="uploaded"))
    payload = Path(saved["path"]).read_bytes()
    persist_routes.new_session()

    upload = UploadFile(io.BytesIO(payload), filename="uploaded.agrosuite")
    result = asyncio.run(persist_routes.upload_session(upload))
    assert [item["id"] for item in result["datasets"]] == [harvest["id"]]
    assert Path(result["path"]).parent == state.uploads
    # Uploads live in the session's temporary folder, so they are not "recent".
    assert all(item["name"] != "uploaded" for item in persist_routes.recent_sessions()["recent"]) or \
        persist_routes.recent_sessions()["recent"][0]["path"] == str(Path(saved["path"]).resolve())

    wrong = UploadFile(io.BytesIO(b"lon,lat\n1,2\n"), filename="yield.csv")
    with pytest.raises(HTTPException) as refused:
        asyncio.run(persist_routes.upload_session(wrong))
    assert "not a project file" in refused.value.detail


# ==========================================================================
# The file the session is on
# ==========================================================================

def _save(**kwargs):
    return persist_routes.save_session(persist_routes.SaveRequest(**kwargs))


def test_resaving_the_open_file_asks_nothing_whatever_the_name(server, tmp_path):
    """'North / 2025' is written as 'North - 2025.agrosuite'. The server, not
    the interface, knows that the two are the same file, so a plain Save of
    the file that is open must never come back as 'already exists'."""
    from fastapi import HTTPException

    state = server.state
    server._register(synthetic_harvest(), "Harvest with defects (demo)", "demo")
    first = _save(name="North / 2025", path=str(tmp_path))
    assert Path(first["path"]) == tmp_path / "North - 2025.agrosuite"
    assert state.project["name"] == "North / 2025"
    assert first["file"]["path"] == first["path"] and first["file"]["name"] == "North - 2025"
    assert first["file"]["folder"] == str(tmp_path)
    assert first["file"]["on_disk"] is True and first["file"]["named_after_project"] is True
    assert first["file"]["download_url"] == first["download_url"]
    assert first["warning"] is None

    # No overwrite flag, same target: the open file is re-saved.
    again = _save(name="North / 2025", path=str(tmp_path))
    assert again["path"] == first["path"]
    # The same file through another spelling of the folder is still it.
    (tmp_path / "sub").mkdir()
    via_dots = _save(name="North / 2025", path=f"{tmp_path}/sub/../")
    assert via_dots["path"] == first["path"]

    # The recent endpoint says which file is open, so a reloaded page can
    # pick it up; the temporary folder is never "on disk".
    recent = persist_routes.recent_sessions()
    assert recent["current"] == via_dots["file"]
    assert recent["recent"][0]["path"] == first["path"]
    in_temp = _save(name="Scratch")
    assert Path(in_temp["path"]).parent == state.exports
    assert in_temp["file"]["on_disk"] is False
    assert persist_routes.recent_sessions()["current"]["path"] == in_temp["path"]

    # Any other existing file still asks.
    with pytest.raises(HTTPException) as conflict:
        _save(name="North / 2025", path=str(tmp_path))
    assert conflict.value.status_code == 409

    # A file saved under a name of its own is reported as such: the dialog
    # proposes the full path, so re-saving it is still one press of Save.
    custom = _save(name="Field A", path=str(tmp_path / "custom_name.agrosuite"))
    assert custom["file"]["name"] == "custom_name"
    assert custom["file"]["named_after_project"] is False
    assert _save(name="Field A", path=custom["path"])["path"] == custom["path"]

    # Starting afresh forgets the file: even the one just saved asks again.
    persist_routes.new_session()
    assert state.project_file is None
    assert persist_routes.recent_sessions()["current"] is None
    server._register(synthetic_harvest(seed=1), "Other", "demo")
    with pytest.raises(HTTPException) as conflict:
        _save(name="Field A", path=custom["path"])
    assert conflict.value.status_code == 409

    # Opening a file makes it the one the session is on.
    opened = persist_routes.open_session(persist_routes.OpenRequest(path=custom["path"]))
    assert opened["file"]["path"] == custom["path"] and opened["file"]["on_disk"] is True
    assert opened["file"]["download_url"] is None and opened["warning"] is None
    assert persist_routes.recent_sessions()["current"] == opened["file"]
    assert _save(name="Field A", path=custom["path"])["path"] == custom["path"]


def test_a_refused_save_renames_nothing(server, tmp_path):
    """The name travels with the save and becomes the project's only once the
    file holds it; otherwise the panel, the report title and the server
    would disagree after a save that did nothing."""
    from fastapi import HTTPException

    state = server.state
    server._register(synthetic_harvest(), "Harvest with defects (demo)", "demo")
    state.project["name"] = "Before"

    with pytest.raises(HTTPException) as relative:
        _save(name="Renamed but failed", path="somewhere/sub")
    assert "relative path" in relative.value.detail
    assert state.project["name"] == "Before"

    blocker = tmp_path / "notes.txt"
    blocker.write_text("a file where a folder is needed", encoding="utf-8")
    with pytest.raises(HTTPException) as in_the_way:
        _save(name="Renamed but failed", path=str(blocker / "x.agrosuite"))
    assert in_the_way.value.status_code == 400 and "is a file, not a folder" in in_the_way.value.detail
    assert state.project["name"] == "Before"

    saved = _save(name="  After  ", path=str(tmp_path))
    assert state.project["name"] == "After"
    assert persist.read_manifest(saved["path"])["project"]["name"] == "After"
    # No name: the project keeps its own, and names the file.
    again = _save(path=str(tmp_path / "again"))
    assert state.project["name"] == "After"
    assert Path(again["path"]) == tmp_path / "again" / "After.agrosuite"


def test_an_unwritable_home_does_not_fail_the_save_or_the_open(server, tmp_path, monkeypatch):
    """The recent list is a convenience; the file and the session are not."""
    blocker = tmp_path / "not-a-folder"
    blocker.write_text("AGROSUITE_HOME points at a file", encoding="utf-8")
    monkeypatch.setenv("AGROSUITE_HOME", str(blocker))

    harvest = server._register(synthetic_harvest(), "Harvest with defects (demo)", "demo")
    saved = _save(name="x", path=str(tmp_path))
    assert Path(saved["path"]).is_file()
    assert "recent list could not be updated" in saved["warning"]
    listing = persist_routes.recent_sessions()
    assert listing["recent"] == [] and listing["current"]["path"] == saved["path"]

    opened = persist_routes.open_session(persist_routes.OpenRequest(path=saved["path"]))
    assert [item["id"] for item in opened["datasets"]] == [harvest["id"]]
    assert "recent list could not be updated" in opened["warning"]
    assert opened["file"]["path"] == saved["path"]
    assert persist_routes.recent_sessions()["current"]["path"] == saved["path"]


def test_a_manifest_with_bad_ids_is_refused_before_the_session_is_touched(tmp_path):
    state, _ = _populated_session(tmp_path)
    target = tmp_path / "north.agrosuite"
    persist.save_session(state, target, SAVED_AT)

    def rewritten(name, mutate):
        with zipfile.ZipFile(target) as zf:
            members = {member: zf.read(member) for member in zf.namelist()}
        manifest = json.loads(members["manifest.json"])
        mutate(manifest)
        members["manifest.json"] = json.dumps(manifest).encode("utf-8")
        out = tmp_path / name
        with zipfile.ZipFile(out, "w") as zf:
            for member, payload in members.items():
                zf.writestr(member, payload)
        return out

    no_id = rewritten("no-id.agrosuite", lambda m: m["datasets"][0].pop("id"))
    duplicated = rewritten("dup.agrosuite", lambda m: m["datasets"].append(dict(m["datasets"][0])))

    other = session_mod.Session()
    kept = other.add(synthetic_harvest(seed=5), "kept", "demo")
    other.project["name"] = "Still here"
    with pytest.raises(ValueError, match="record 1 has no id"):
        persist.load_session(other, no_id)
    with pytest.raises(ValueError, match="share the id"):
        persist.load_session(other, duplicated)
    assert [item["id"] for item in other.list()] == [kept.id]
    assert other.project["name"] == "Still here"


def test_the_extension_decides_between_file_and_folder(server, tmp_path):
    """'/x/nofolder' means the folder nofolder — with or without a trailing
    separator, there yet or not — never a file called nofolder.agrosuite in
    /x under a name nobody chose. Only a path ending in .agrosuite is the
    file, and its folder is made too."""
    server._register(synthetic_harvest(), "Harvest with defects (demo)", "demo")

    with_separator = _save(name="x", path=f"{tmp_path / 'nofolder'}/")
    assert Path(with_separator["path"]) == tmp_path / "nofolder" / "x.agrosuite"
    without = _save(name="x", path=str(tmp_path / "other"))
    assert Path(without["path"]) == tmp_path / "other" / "x.agrosuite"
    assert not (tmp_path / "other.agrosuite").exists()
    assert not (tmp_path / "nofolder.agrosuite").exists()

    nested = _save(name="x", path=str(tmp_path / "deep" / "er" / "north.AGROSUITE"))
    assert Path(nested["path"]) == tmp_path / "deep" / "er" / "north.AGROSUITE"
    assert (tmp_path / "deep" / "er").is_dir()

    saved = _save(name="x", path=f"{tmp_path}/")
    assert Path(saved["path"]) == tmp_path / "x.agrosuite"


def test_relative_and_blocked_save_paths_are_refused(server, tmp_path, monkeypatch):
    """A relative path would land beside wherever the app was started from;
    a file where the folder should be cannot be made into one. Both are
    refused in words, and nothing appears anywhere."""
    from fastapi import HTTPException

    monkeypatch.chdir(tmp_path)
    server._register(synthetic_harvest(), "Harvest with defects (demo)", "demo")
    for relative in ("projects", "./north.agrosuite", "projects\\north.agrosuite", '"north"'):
        with pytest.raises(HTTPException) as refused:
            _save(name="x", path=relative)
        assert refused.value.status_code == 400, relative
        assert "relative path" in refused.value.detail and "full path" in refused.value.detail
    assert not list(tmp_path.rglob("*.agrosuite")) and not (tmp_path / "projects").exists()
    # '~' is expanded before the check: it is a full path to the person.
    home = tmp_path / "home"
    # Windows expands '~' from USERPROFILE and ignores HOME; setting only
    # HOME would send the save to the real profile of whoever runs the
    # tests. Both are set, so '~' is the temporary folder on every platform.
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    at_home = _save(name="x", path="~/projects")
    assert Path(at_home["path"]) == home / "projects" / "x.agrosuite"

    blocker = tmp_path / "notes.txt"
    blocker.write_text("in the way", encoding="utf-8")
    with pytest.raises(HTTPException) as as_folder:
        _save(name="x", path=str(blocker))
    assert "is a file, not a folder" in as_folder.value.detail
    assert ".agrosuite" in as_folder.value.detail
    with pytest.raises(HTTPException) as under_file:
        _save(name="x", path=str(blocker / "deeper"))
    assert under_file.value.status_code == 400
    assert blocker.read_text(encoding="utf-8") == "in the way"


# ==========================================================================
# Looking at a file before opening it
# ==========================================================================

def test_peek_describes_a_file_without_loading_it(server, tmp_path):
    """The interface asks 'close the N datasets?' only for a file that will
    then open. A folder — what the browser puts in the box on a click — a
    missing file or a ZIP of monitor data get a reason instead, and the
    session that is open is not touched either way."""
    state = server.state
    harvest = server._register(synthetic_harvest(), "Harvest with defects (demo)", "demo")
    saved = _save(name="North", path=str(tmp_path))
    server._register(synthetic_harvest(seed=1), "Second", "demo")
    ids_before = [item["id"] for item in state.list()]

    good = persist_routes.peek_session(f'"{tmp_path}/./North.agrosuite"')
    assert good["ok"] is True
    assert good["name"] == "North" and good["project"] == "North"
    assert good["datasets"] == 1 and good["saved_at"] == saved["saved_at"]
    assert good["path"] == saved["path"]

    folder = persist_routes.peek_session(str(tmp_path))
    assert folder["ok"] is False
    assert "That is a folder" in folder["reason"] and ".agrosuite file inside it" in folder["reason"]

    missing = persist_routes.peek_session(str(tmp_path / "gone.agrosuite"))
    assert missing["ok"] is False and "File not found" in missing["reason"]

    text = tmp_path / "notes.agrosuite"
    text.write_text("not a zip", encoding="utf-8")
    assert "not a ZIP" in persist_routes.peek_session(str(text))["reason"]
    data = tmp_path / "monitor.zip"
    with zipfile.ZipFile(data, "w") as zf:
        zf.writestr("yield.csv", "lon,lat\n1,2\n")
    assert "no manifest" in persist_routes.peek_session(str(data))["reason"]

    # Nothing was opened: both datasets are still there, and the recent list
    # and the open file are as the save left them.
    assert [item["id"] for item in state.list()] == ids_before
    assert harvest["id"] in ids_before
    assert persist_routes.recent_sessions()["current"]["path"] == saved["path"]
    assert server.app.url_path_for("peek_session") == "/api/session/peek"


# ==========================================================================
# The trial layout is part of the project
# ==========================================================================

FIELD = [(-105.8340, 50.4520), (-105.8229, 50.4520), (-105.8229, 50.4592), (-105.8340, 50.4592)]


def test_the_trial_layout_round_trips_with_the_project(server, tmp_path):
    """The export offers the layout as the prescription; a file that lost it
    would hand back a session with the trial to lay out again — and the
    dialog promises the whole session."""
    from agrosuite.difm.design import design_strips

    state = server.state
    server._register(synthetic_harvest(), "Harvest with defects (demo)", "demo")
    assert state.project["design"] is None

    request = server.DesignRequest(boundary=[list(p) for p in FIELD], rates=[0, 50, 100, 150],
                                   implement_width_m=18.29, blocks=3, buffer_m=20, seed=3)
    result = server.design(request)
    stored = state.project["design"]
    assert stored["result"] == result
    assert stored["request"] == request.model_dump() and stored["dataset_id"] is None
    assert stored["created_at"]
    # The same layout, made the same way, is what the file must give back.
    assert result["features"] == design_strips(FIELD, rates=[0, 50, 100, 150], implement_width_m=18.29,
                                               blocks=3, buffer_m=20, seed=3)["features"]

    saved = _save(name="Trial", path=str(tmp_path))
    manifest = persist.read_manifest(saved["path"])
    assert manifest["project"]["design"]["result"]["summary"] == result["summary"]

    persist_routes.new_session()
    assert state.project["design"] is None
    opened = persist_routes.open_session(persist_routes.OpenRequest(path=saved["path"]))
    assert state.project["design"] == stored
    # Handed back with the datasets, so the interface can draw the strips.
    assert opened["design"] == stored
    assert opened["design"]["result"]["features"]["features"][0]["properties"]["rate"] in (0, 50, 100, 150)

    # A file from before the layout was kept, or one without a layout, opens
    # with none — not with the layout of the project that was open before.
    server.design(request)
    fresh = session_mod.Session()
    fresh.add(synthetic_harvest(), "x", "demo")
    without = tmp_path / "without.agrosuite"
    persist.save_session(fresh, without, SAVED_AT)
    assert persist_routes.open_session(persist_routes.OpenRequest(path=str(without)))["design"] is None
    assert state.project["design"] is None


def test_paths_are_reported_resolved(server, tmp_path):
    """The status line, the recent list and the 'current' mark compare paths
    as text; '..' or './' in what was typed must not make one file two."""
    (tmp_path / "agro").mkdir()
    server._register(synthetic_harvest(), "Harvest with defects (demo)", "demo")
    saved = _save(name="North / 2025", path=f"{tmp_path}/agro/../agro/")
    expected = str(tmp_path / "agro" / "North - 2025.agrosuite")
    assert saved["path"] == expected == saved["file"]["path"]
    listing = persist_routes.recent_sessions()
    assert listing["recent"][0]["path"] == expected == listing["current"]["path"]

    persist_routes.new_session()
    opened = persist_routes.open_session(
        persist_routes.OpenRequest(path=f"{tmp_path}/agro/./North - 2025.agrosuite"))
    assert opened["path"] == expected == opened["file"]["path"]
    assert persist_routes.recent_sessions()["current"]["path"] == expected
