"""Saving the session without being asked, and picking it up again.

What these tests protect is a promise made to someone who closes the app by
closing the window: the work is on disk, in a folder he can find, and the
app opens again on what he left. The promise is only worth making if the
saving cannot itself destroy anything — so most of what is checked here is
what auto-save refuses to do. An empty session does not overwrite a project.
A file this session did not write is not adopted. The version before the
last save stays beside it. A save that fails says so once and does not break
the request that triggered it.

The writer is a thread, so the tests either drive it directly with
``flush()`` — which is what the shutdown does — or shorten the idle delay
and wait for the file to appear.
"""

from __future__ import annotations

import json
import os
import sys
import time
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agrosuite.app import autosave  # noqa: E402
from agrosuite.app import persist  # noqa: E402
from agrosuite.app import server as server_mod  # noqa: E402
from agrosuite.app import session as session_mod  # noqa: E402
from agrosuite.app import settings as settings_mod  # noqa: E402
from agrosuite.demo import synthetic_harvest  # noqa: E402

STATIC = ROOT / "agrosuite" / "app" / "static"
APP_JS = (STATIC / "app.js").read_text(encoding="utf-8")
INDEX = (STATIC / "index.html").read_text(encoding="utf-8")


# ==========================================================================
# Fixtures
# ==========================================================================

@pytest.fixture
def projects(tmp_path, monkeypatch) -> Path:
    """A home folder and a projects folder of this test's own.

    Nothing here may touch the real ``~/.agrosuite`` or the real Documents
    folder: the default is deliberately a place people keep things in.
    """
    monkeypatch.setenv("AGROSUITE_HOME", str(tmp_path / "home"))
    folder = tmp_path / "projects"
    settings_mod.write(projects_dir=folder, autosave=True)
    return folder


@pytest.fixture
def state(monkeypatch) -> session_mod.Session:
    """A session of this test's own, in the place the routes read from."""
    fresh = session_mod.Session()
    monkeypatch.setattr(server_mod, "state", fresh)
    yield fresh
    fresh.cleanup()


@pytest.fixture
def quick(monkeypatch) -> None:
    """The writer, with the waiting taken out of it."""
    monkeypatch.setattr(autosave, "IDLE_SECONDS", 0.05)
    monkeypatch.setattr(autosave, "POLL_SECONDS", 0.05)
    monkeypatch.setenv(autosave.ENV_FLAG, "on")


def _saver(state: session_mod.Session) -> autosave.AutoSaver:
    """A writer that is not a thread: every write happens where it is asked
    for, which is what makes the refusals testable one at a time."""
    return autosave.AutoSaver(lambda: state)


def _harvest(state: session_mod.Session, label: str = "Harvest") -> str:
    entry = state.add(synthetic_harvest(), label, "demo")
    state.touch()
    return entry.id


def _until(predicate, timeout: float = 8.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def _datasets_in(path: Path) -> int:
    return len(persist.read_manifest(path)["datasets"])


# ==========================================================================
# A change ends up in a file
# ==========================================================================

def test_a_change_leads_to_a_file_in_the_projects_folder(projects, state, quick):
    """The whole feature in one line: work, and the file appears."""
    with TestClient(server_mod.app) as client:
        loaded = client.post("/api/import/demo", json={"kind": "harvest"})
        assert loaded.status_code == 200, loaded.text

        target = projects / "Untitled project.agrosuite"
        assert _until(target.exists), f"nothing was written to {projects}"
        assert _datasets_in(target) == 1

        status = client.get("/api/session/autosave").json()
        assert status["enabled"] and status["running"]
        assert _until(lambda: client.get("/api/session/autosave").json()["state"] == "saved")
        assert client.get("/api/session/autosave").json()["path"] == str(target)


def test_every_way_the_session_changes_marks_it_for_saving(projects, state):
    """A mutation the writer never hears about is work that is not saved.

    The routes that change state are asked to prove they say so; the
    counter is the only thing the writer looks at.
    """
    with TestClient(server_mod.app) as client:
        dataset_id = client.post("/api/import/demo", json={"kind": "harvest"}).json()["id"]
        calls = [
            ("POST", "/api/project", {"name": "North quarter"}),
            ("POST", "/api/project/role", {"dataset_id": dataset_id, "role": "yield"}),
            ("POST", "/api/project/prices", {"crop_price": 0.55, "input_cost": 1.2}),
            ("PUT", "/api/session/view", {"dataset_id": dataset_id, "tab": "limpeza"}),
            ("POST", f"/api/datasets/{dataset_id}/units", {"source_units": {"value": "bu/ac"},
                                                           "crop": "canola"}),
            ("POST", f"/api/datasets/{dataset_id}/clean", {}),
            ("POST", f"/api/datasets/{dataset_id}/difm", {}),
            ("POST", "/api/terrain/analyze", {"dataset_id": dataset_id}),
        ]
        for method, path, body in calls:
            before = state.revision
            response = client.request(method, path, json=body)
            # Some of these refuse this particular dataset — a demo harvest
            # has no relief worth analysing — and a refusal changes nothing,
            # so only the ones that worked have to have said so.
            if response.status_code != 200:
                continue
            assert state.revision > before, f"{method} {path} changed the session in silence"

        # Deleting is a change like any other: a layer removed and not saved
        # comes back the next time the project is opened.
        before = state.revision
        assert client.delete(f"/api/datasets/{dataset_id}").status_code == 200
        assert state.revision > before, "a removed dataset did not mark the session"


# ==========================================================================
# What auto-save refuses to do
# ==========================================================================

def test_an_empty_session_never_overwrites_a_project(projects, state):
    """The file holds an afternoon's work; the session holds nothing. That
    is the moment just after 'New project', and it must not be the moment
    the afternoon is lost."""
    _harvest(state)
    saver = _saver(state)
    saver.flush()
    target = state.autosave_path
    assert target is not None and _datasets_in(target) == 1
    before = target.read_bytes()

    # The session is emptied but still pointed at the file — the worst case,
    # worse than what 'New project' actually does, which is to let go of it.
    state.clear()
    state.autosave_path = target
    state.touch()
    saver.flush()

    assert target.read_bytes() == before, "an empty session was written over a project"
    assert saver.status()["state"] != "failed"


def test_a_file_this_session_did_not_write_is_not_adopted(projects, state):
    """Two projects are called 'Untitled project' soon enough. The second one
    gets a file of its own rather than the first one's."""
    settings_mod.ensure(projects)
    stranger = projects / "Untitled project.agrosuite"
    other = session_mod.Session()
    try:
        other.add(synthetic_harvest(), "Somebody else's morning", "demo")
        persist.save_session(other, stranger, "2026-09-16T09:00:00-06:00")
    finally:
        other.cleanup()
    before = stranger.read_bytes()

    _harvest(state)
    _saver(state).flush()

    assert state.autosave_path == projects / "Untitled project 2.agrosuite"
    assert stranger.read_bytes() == before


def test_the_backup_holds_the_version_before_the_last_save(projects, state):
    """One step back, which is what it takes to undo an auto-save made over
    a mistake."""
    first = _harvest(state, "First file")
    saver = _saver(state)
    saver.flush()
    target = state.autosave_path
    assert target is not None

    state.add(synthetic_harvest(), "Second file", "demo")
    state.touch()
    saver.flush()

    backup = target.with_name(target.name + autosave.BACKUP_SUFFIX)
    assert _datasets_in(target) == 2
    assert backup.exists(), "the previous version was not kept"
    assert _datasets_in(backup) == 1
    assert [d["id"] for d in persist.read_manifest(backup)["datasets"]] == [first]

    # One backup, not a history: the next save replaces it.
    state.add(synthetic_harvest(), "Third file", "demo")
    state.touch()
    saver.flush()
    assert _datasets_in(backup) == 2
    assert not list(projects.glob("*" + autosave.PENDING_SUFFIX)), "a half-written file was left"


def test_a_failing_save_is_reported_and_breaks_nothing(projects, state, quick):
    """A folder that has gone — a USB stick pulled out, a path that cannot
    be made — must reach the person as a line in the panel, not as a failed
    import."""
    blocked = projects.parent / "a-file"
    blocked.write_text("not a folder", encoding="utf-8")
    settings_mod.write(projects_dir=blocked / "projects", autosave=True)

    with TestClient(server_mod.app) as client:
        loaded = client.post("/api/import/demo", json={"kind": "harvest"})
        assert loaded.status_code == 200, "a failing auto-save broke the request"

        assert _until(lambda: client.get("/api/session/autosave").json()["state"] == "failed")
        status = client.get("/api/session/autosave").json()
        assert str(blocked / "projects") in status["message"]
        assert "choose another folder" in status["message"]

        # Reported once: the writer does not come back to a revision it has
        # already failed on until something else changes.
        first = status["message"]
        time.sleep(0.3)
        assert client.get("/api/session/autosave").json()["message"] == first

        # And it tries again on the next change, into a folder that works.
        settings_mod.write(projects_dir=projects, autosave=True)
        assert client.post("/api/project", json={"name": "Second try"}).status_code == 200
        assert _until(lambda: (projects / "Second try.agrosuite").exists())


def test_auto_save_off_writes_nothing(projects, state, quick):
    """Off means the app behaves as it did before any of this."""
    settings_mod.write(autosave=False)
    with TestClient(server_mod.app) as client:
        assert client.post("/api/import/demo", json={"kind": "harvest"}).status_code == 200
        time.sleep(0.5)
        assert not list(projects.glob("*")), "auto-save is off and something was written"

        status = client.get("/api/session/autosave").json()
        assert status["enabled"] is False and status["state"] == "off"
        assert state.autosave_path is None

        # Switched back on, what was done in the meantime is saved.
        assert client.put("/api/settings", json={"autosave": True}).status_code == 200
        assert _until(lambda: (projects / "Untitled project.agrosuite").exists())


# ==========================================================================
# The name of the file, and the folder it is in
# ==========================================================================

def test_renaming_the_project_moves_the_file(projects, state):
    """Renamed, not copied: a second file under the old name is a second
    project as far as 'pick up where I left off' is concerned."""
    _harvest(state)
    saver = _saver(state)
    saver.flush()
    old = state.autosave_path
    assert old == projects / "Untitled project.agrosuite"
    # A second save, so there is a backup to move as well.
    state.touch()
    saver.flush()
    assert old.with_name(old.name + autosave.BACKUP_SUFFIX).exists()

    state.project["name"] = "North quarter / 2026"
    state.touch()
    saver.flush()

    new = projects / f"{persist.slugify('North quarter / 2026')}.agrosuite"
    assert state.autosave_path == new
    assert new.exists() and not old.exists(), "the old file was left behind"
    assert not old.with_name(old.name + autosave.BACKUP_SUFFIX).exists()
    assert new.with_name(new.name + autosave.BACKUP_SUFFIX).exists()
    assert persist.read_manifest(new)["project"]["name"] == "North quarter / 2026"
    assert sorted(p.name for p in projects.glob("*.agrosuite")) == [new.name]

    # The recent list follows too: the old name would show as "not found"
    # and lead nowhere, for the project that is open under the new one.
    listed = [item["path"] for item in persist.recent_files()]
    assert str(new) in listed and str(old) not in listed


def test_changing_the_folder_takes_the_project_with_it(projects, state, quick, tmp_path):
    """The file already written moves; the old folder is not left holding a
    copy that quietly goes stale."""
    with TestClient(server_mod.app) as client:
        client.post("/api/import/demo", json={"kind": "harvest"})
        assert _until(lambda: (projects / "Untitled project.agrosuite").exists())

        elsewhere = tmp_path / "on the desktop"
        answer = client.put("/api/settings", json={"projects_dir": str(elsewhere)})
        assert answer.status_code == 200, answer.text
        payload = answer.json()
        assert payload["projects_dir"] == str(elsewhere.resolve())
        assert payload["moved"] is True

        moved = elsewhere / "Untitled project.agrosuite"
        assert moved.exists() and _datasets_in(moved) == 1
        assert not list(projects.glob("*.agrosuite")), "a copy was left in the old folder"

        # And the next save goes there too.
        assert client.post("/api/project", json={"name": "Moved project"}).status_code == 200
        assert _until(lambda: (elsewhere / "Moved project.agrosuite").exists())
        assert not moved.exists()


def test_the_folder_is_refused_with_a_way_out(projects, state, tmp_path):
    """Three refusals, each saying what to do rather than what went wrong."""
    with TestClient(server_mod.app) as client:
        relative = client.put("/api/settings", json={"projects_dir": "projects/2026"})
        assert relative.status_code == 400
        assert "relative path" in relative.json()["detail"]
        assert "full path" in relative.json()["detail"]

        a_file = tmp_path / "harvest.csv"
        a_file.write_text("x", encoding="utf-8")
        refused = client.put("/api/settings", json={"projects_dir": str(a_file)})
        assert refused.status_code == 400
        assert "is a file, not a folder" in refused.json()["detail"]

        impossible = client.put("/api/settings", json={"projects_dir": str(a_file / "inside")})
        assert impossible.status_code == 400
        assert "Could not create the folder" in impossible.json()["detail"]

        # Refused means nothing changed.
        assert client.get("/api/settings").json()["projects_dir"] == str(projects)

        accepted = client.put("/api/settings", json={"projects_dir": str(tmp_path / "chosen")})
        assert accepted.status_code == 200
        assert (tmp_path / "chosen").is_dir(), "the folder was not created"


def test_the_default_folder_is_one_a_person_can_find(monkeypatch, tmp_path):
    """Not inside the hidden .agrosuite: a project nobody can see is a
    project nobody can copy to a stick."""
    home = tmp_path / "home"
    (home / "Documents").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    assert settings_mod.default_projects_dir() == home / "Documents" / "AgroSuite"

    bare = tmp_path / "bare"
    bare.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: bare))
    assert settings_mod.default_projects_dir() == bare / "AgroSuite"
    assert ".agrosuite" not in str(settings_mod.default_projects_dir())


# ==========================================================================
# Picking it up again
# ==========================================================================

def test_the_app_opens_on_what_was_left(projects, monkeypatch, quick):
    """The heart of it: close the app, open it again, and the work is there
    — with the same dataset selected and the same tab open."""
    first = session_mod.Session()
    monkeypatch.setattr(server_mod, "state", first)
    with TestClient(server_mod.app) as client:
        dataset_id = client.post("/api/import/demo", json={"kind": "harvest"}).json()["id"]
        client.post("/api/project", json={"name": "Picked up"})
        client.put("/api/session/view", json={"dataset_id": dataset_id, "tab": "difm"})
        assert _until(lambda: (projects / "Picked up.agrosuite").exists())
    first.cleanup()

    # A new session, as a restart gives.
    second = session_mod.Session()
    monkeypatch.setattr(server_mod, "state", second)
    try:
        with TestClient(server_mod.app) as client:
            datasets = client.get("/api/datasets").json()["datasets"]
            assert [d["id"] for d in datasets] == [dataset_id]
            assert client.get("/api/project").json()["name"] == "Picked up"

            recent = client.get("/api/session/recent").json()
            assert recent["view"] == {"dataset_id": dataset_id, "tab": "difm"}
            resumed = recent["resumed"]
            assert resumed["ok"] and resumed["project"] == "Picked up"
            assert resumed["datasets"] == 1 and resumed["saved_at"]

            # Said once. A reloaded page is not a second start-up.
            assert client.get("/api/session/recent").json()["resumed"] is None

            # It carries on with the same file rather than making a second one.
            assert second.autosave_path == projects / "Picked up.agrosuite"
            client.post("/api/project", json={"name": "Picked up"})
            time.sleep(0.4)
            assert sorted(p.name for p in projects.glob("*.agrosuite")) == \
                ["Picked up.agrosuite"]
    finally:
        second.cleanup()


def test_starting_a_new_project_leaves_the_last_one_alone(projects, state, quick):
    """'Start a new project instead' is the one click back out of an
    auto-resume, and it must not be a click that deletes anything."""
    with TestClient(server_mod.app) as client:
        client.post("/api/import/demo", json={"kind": "harvest"})
        assert _until(lambda: (projects / "Untitled project.agrosuite").exists())
        kept = (projects / "Untitled project.agrosuite").read_bytes()

        assert client.post("/api/session/new", json={}).status_code == 200
        assert state.autosave_path is None
        time.sleep(0.3)
        assert (projects / "Untitled project.agrosuite").read_bytes() == kept

        # And the panel stops naming that file: an empty session reported as
        # "saved 12:04" would promise that the file holds this nothing.
        status = client.get("/api/session/autosave").json()
        assert status["path"] is None and status["saved_at"] is None

        # And the project that follows gets a file of its own.
        client.post("/api/import/demo", json={"kind": "trial"})
        assert _until(lambda: (projects / "Untitled project 2.agrosuite").exists())
        assert (projects / "Untitled project.agrosuite").read_bytes() == kept


def test_latest_says_what_would_be_picked_up_without_loading_it(projects, state, quick):
    with TestClient(server_mod.app) as client:
        empty = client.get("/api/session/latest").json()
        assert empty["available"] is False and empty["folder"] == str(projects)

        client.post("/api/import/demo", json={"kind": "harvest"})
        client.post("/api/project", json={"name": "Named project"})
        assert _until(lambda: (projects / "Named project.agrosuite").exists())

        latest = client.get("/api/session/latest").json()
        assert latest["available"] is True
        # Not resumable while a session is loaded: it is what is loaded.
        assert latest["resumable"] is False
        assert latest["project"]["project"] == "Named project"
        assert latest["project"]["datasets"] == 1 and latest["project"]["saved_at"]
        # Nothing was loaded to answer that.
        assert len(state.list()) == 1


def test_a_damaged_project_does_not_stop_the_app(projects, state, quick):
    """Reopening is a convenience; failing to start is not an acceptable
    price for it."""
    settings_mod.ensure(projects)
    (projects / "Broken.agrosuite").write_bytes(b"this is not a zip")

    with TestClient(server_mod.app) as client:
        assert client.get("/api/health").json()["ok"] is True
        assert client.get("/api/datasets").json()["datasets"] == []
        resumed = client.get("/api/session/recent").json()["resumed"]
        assert resumed and resumed["ok"] is False
        assert "Broken.agrosuite" in resumed["reason"]
        assert "started empty" in resumed["reason"]


# ==========================================================================
# The file itself
# ==========================================================================

def test_the_view_travels_in_the_project_file(projects, state):
    dataset_id = _harvest(state)
    state.view = {"dataset_id": dataset_id, "tab": "terrain"}
    _saver(state).flush()

    manifest = persist.read_manifest(state.autosave_path)
    assert manifest["view"] == {"dataset_id": dataset_id, "tab": "terrain"}
    assert manifest["saved_at"], "the file does not say when it was saved"

    fresh = session_mod.Session()
    try:
        result = persist.load_session(fresh, state.autosave_path)
        assert fresh.view == {"dataset_id": dataset_id, "tab": "terrain"}
        assert result["view"] == fresh.view
    finally:
        fresh.cleanup()


def test_a_project_file_from_before_the_view_still_opens(projects, state, tmp_path):
    """He has files from before this change, and they have to open."""
    dataset_id = _harvest(state)
    state.view = {"dataset_id": dataset_id, "tab": "limpeza"}
    saved = tmp_path / "last season.agrosuite"
    persist.save_session(state, saved, "2026-03-01T08:00:00-06:00")

    older = tmp_path / "older.agrosuite"
    _without_view(saved, older)
    assert "view" not in persist.read_manifest(older)

    fresh = session_mod.Session()
    try:
        result = persist.load_session(fresh, older)
        assert [d["id"] for d in result["datasets"]] == [dataset_id]
        assert fresh.view == session_mod.default_view()
        assert result["view"] == {"dataset_id": None, "tab": None}
    finally:
        fresh.cleanup()


def test_a_view_pointing_at_a_dataset_the_file_lost_is_dropped(projects, state, tmp_path):
    """A stale id would select nothing and leave the panel empty."""
    _harvest(state)
    state.view = {"dataset_id": "not-in-this-file", "tab": "difm"}
    saved = tmp_path / "stale.agrosuite"
    persist.save_session(state, saved, "2026-03-01T08:00:00-06:00")

    fresh = session_mod.Session()
    try:
        persist.load_session(fresh, saved)
        assert fresh.view == {"dataset_id": None, "tab": "difm"}
    finally:
        fresh.cleanup()


def _without_view(source: Path, target: Path) -> None:
    """A copy of a project file with the view key taken out of its manifest,
    which is what every file written before this change looks like."""
    with zipfile.ZipFile(source) as zin, zipfile.ZipFile(target, "w") as zout:
        for item in zin.infolist():
            payload = zin.read(item.filename)
            if item.filename == "manifest.json":
                manifest = json.loads(payload.decode("utf-8"))
                manifest.pop("view", None)
                payload = json.dumps(manifest, indent=2).encode("utf-8")
            zout.writestr(item, payload)


# ==========================================================================
# Settings on disk
# ==========================================================================

def test_an_unreadable_settings_file_is_left_alone(projects, monkeypatch, tmp_path):
    """The folder someone typed into it is not worth losing to a comma."""
    path = settings_mod.settings_path()
    path.write_text("{ this is not json", encoding="utf-8")

    config = settings_mod.read()
    assert config.projects_dir == settings_mod.default_projects_dir()
    assert config.autosave is True
    assert "could not be read" in config.warning and str(path) in config.warning
    assert path.read_text(encoding="utf-8") == "{ this is not json"

    # Changing a setting puts it aside rather than writing over it.
    settings_mod.write(autosave=False)
    assert path.with_suffix(".json.broken").read_text(encoding="utf-8") == "{ this is not json"
    assert settings_mod.read().autosave is False


def test_settings_keep_what_this_version_does_not_know(projects):
    path = settings_mod.settings_path()
    data = json.loads(path.read_text(encoding="utf-8"))
    data["something_from_later"] = {"keep": "me"}
    path.write_text(json.dumps(data), encoding="utf-8")

    settings_mod.write(autosave=False)
    assert json.loads(path.read_text(encoding="utf-8"))["something_from_later"] == {"keep": "me"}


def test_the_writer_stays_out_of_a_process_that_only_imported_the_app(monkeypatch):
    """A test, a script or the MCP tools must not start a thread that writes
    to somebody's Documents folder."""
    monkeypatch.delenv(autosave.ENV_FLAG, raising=False)
    assert autosave.enabled_for_process() is False
    assert autosave.start(lambda: server_mod.state) is None
    monkeypatch.setenv(autosave.ENV_FLAG, "on")
    assert autosave.enabled_for_process() is True


# ==========================================================================
# The interface
# ==========================================================================

def test_the_panel_has_the_three_things_and_no_dialog():
    """The folder, the switch and one line — in the Project file block, in
    the left panel, with nothing to dismiss."""
    for node_id in ("autosave", "autosave-toggle", "autosave-folder",
                    "autosave-status", "btn-projects-folder"):
        assert f'id="{node_id}"' in INDEX, f"index.html has no #{node_id}"
    panel = INDEX[INDEX.index("<h4>Project file</h4>"):INDEX.index("<h3>Project</h3>")]
    assert 'id="autosave"' in panel, "auto-save is not in the Project file block"
    assert "<dialog" not in panel


def test_the_page_calls_the_endpoints_the_server_serves():
    from agrosuite.app.routes import persist as persist_routes
    from agrosuite.app.routes import settings as settings_routes

    served = {route.path for route in persist_routes.router.routes}
    served |= {route.path for route in settings_routes.router.routes}
    for path in ("/api/settings", "/api/session/view", "/api/session/autosave"):
        assert path in served, f"the router does not serve {path}"
        assert f'"{path}"' in APP_JS, f"the page never calls {path}"


def test_the_view_is_sent_when_it_changes_and_not_before():
    """Cheap on purpose: no post on every mouse move, none at all for a tab
    that is already open."""
    note = APP_JS[APP_JS.index("  noteView() {"):]
    note = note[:note.index("\n  },\n")]
    assert "if (view.dataset_id === known.dataset_id && view.tab === known.tab) return;" in note
    assert "clearTimeout(this.state.viewTimer)" in note and "setTimeout(" in note
    for name in ("  openTab(name) {", "  async selectDataset(id) {"):
        block = APP_JS[APP_JS.index(name):]
        assert "this.noteView();" in block[:block.index("\n  },\n")], f"{name} does not note the view"


def test_the_tab_is_read_before_anything_is_selected():
    """Selecting a dataset notes the view, so the tab has to be read out of
    the file's view before that: read after, it is the tab the page opened
    on, and the session comes back on Data every time. It cost the restored
    tab once, in a browser, which is the only place it shows."""
    for name in ("  async restoreView() {", "  async sessionReplaced(result) {"):
        block = APP_JS[APP_JS.index(name):]
        block = block[:block.index("\n  },\n")]
        assert "const openOn" in block, f"{name} does not read the tab out first"
        assert block.index("const openOn") < block.index("this.selectDataset(")
        assert "this.openTab(openOn)" in block


def test_the_resume_toast_offers_the_way_back_out():
    block = APP_JS[APP_JS.index("  announceResume(resumed) {"):]
    block = block[:block.index("\n  },\n")]
    assert "Start a new project instead" in block
    assert "this.newProject(false)" in block
    assert "resumed.reason" in block, "a failed resume must say why"
