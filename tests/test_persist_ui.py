"""The project-file controls in the interface, checked against the contract.

The interface is vanilla JavaScript and has no unit tests of its own; what
can be checked here without a browser is the contract it relies on. Every
endpoint the page calls must be one the router serves, the request it
POSTs must carry only fields the route accepts, the controls it binds must
exist in the page, and the two routes a project file can take in — the
path dialog and the upload input — must both turn away from import. A
rename on either side would otherwise only show up in a browser.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agrosuite.app import session as session_mod
from agrosuite.app.routes import persist as persist_routes

STATIC = Path(__file__).resolve().parents[1] / "agrosuite" / "app" / "static"
APP_JS = (STATIC / "app.js").read_text(encoding="utf-8")
INDEX = (STATIC / "index.html").read_text(encoding="utf-8")


def _block(name: str) -> str:
    """The source of one App method, up to its closing brace at column two."""
    match = re.search(rf"^  (?:async )?{name}\(", APP_JS, flags=re.MULTILINE)
    assert match, f"App.{name} is not defined"
    end = APP_JS.index("\n  },\n", match.start())
    return APP_JS[match.start():end]


def test_controls_and_dialog_are_in_the_page():
    for node_id in ("btn-project-save", "btn-project-open", "btn-project-new",
                    "recent-projects", "project-file", "dlg-save-project",
                    "save-name", "save-path", "btn-save-project-confirm", "dlg-path-title"):
        assert f'id="{node_id}"' in INDEX, f"index.html has no #{node_id}"
    # The row sits in the "Load data" panel, beside the other ways in.
    panel = INDEX[INDEX.index("<h3>Load data</h3>"):INDEX.index("<h3>Project</h3>")]
    for node_id in ("btn-project-save", "btn-project-open", "btn-project-new", "recent-projects"):
        assert f'id="{node_id}"' in panel, f"#{node_id} is not in the Load data panel"


def test_the_page_calls_every_session_endpoint():
    served = {route.path for route in persist_routes.router.routes}
    for path in ("/api/session/save", "/api/session/open", "/api/session/upload",
                 "/api/session/recent", "/api/session/new"):
        assert path in served, f"the router does not serve {path}"
        assert f'"{path}"' in APP_JS, f"the page never calls {path}"
    assert "/api/session/peek" in served
    assert "`/api/session/peek?path=${encodeURIComponent(path)}`" in _block("peekProject")


def test_save_posts_only_fields_the_route_accepts():
    """A key the route ignores would be a silently dropped choice."""
    literal = re.search(r"const request = \{(.*?)\};", _block("saveProject"), flags=re.DOTALL)
    assert literal, "saveProject builds no request literal"
    posted = {part.strip().split(":")[0].strip() for part in literal.group(1).split(",") if part.strip()}
    accepted = set(persist_routes.SaveRequest.model_fields)
    assert posted == accepted, f"posted {sorted(posted)} vs accepted {sorted(accepted)}"


def test_an_existing_file_is_replaced_only_after_asking():
    """The server answers 409 until told to overwrite; the page must carry the
    status through and ask rather than retry on its own."""
    assert "error.status = response.status" in _block("api")
    save = _block("saveProject")
    assert "err.status !== 409" in save
    assert "window.confirm(" in save
    assert "overwrite: true" in save


def test_a_project_file_never_goes_to_import():
    """Both ways a file comes in — the path dialog and the file input or drop
    zone — must route .agrosuite to the session, before any import call."""
    dialogs = _block("bindDialogs")
    assert "this.isProjectFile(path)" in dialogs
    assert dialogs.index("this.openProject(path, peek)") < dialogs.index('"/api/import/path"')
    imports = _block("bindImport")
    assert imports.count("this.isProjectFile(file.name)") == 2, "file input and drop zone"
    assert imports.index("this.uploadProject(file)") < imports.index("this.importFiles(")
    assert re.search(r"/\\\.agrosuite\$/i", _block("isProjectFile"))


def test_replacing_the_session_asks_and_resets_the_client():
    for name in ("openProject", "uploadProject", "newProject"):
        assert "this.confirmReplace(" in _block(name), f"{name} does not ask first"
    reset = _block("resetSessionState")
    # Reports are cached by dataset id and a reopened project keeps its ids:
    # a stale cache would pass off a later cleaning as the saved one.
    assert "this.state.reports = {}" in reset
    after = _block("sessionReplaced")
    for call in ("this.refreshDatasets()", "this.refreshProject()", "this.selectDataset("):
        assert call in after, f"sessionReplaced does not call {call}"


def test_tabs_fall_back_to_the_stored_reports():
    """After a reopen the client cache is empty while the server holds the
    reports; the tabs must ask for them through the report endpoint."""
    assert "d.has_clean_report" in _block("tabLimpeza")
    assert "d.has_difm_report" in _block("tabDifm")
    assert "`/api/datasets/${id}/report/${kind}`" in _block("storedReport")
    shape = _block("cleanResultFor")
    # renderCleanReport expects {report, clean: {id}, removed: {id}|null}.
    assert 'd.origin === "clean"' in shape and 'd.origin === "clean_removed"' in shape
    assert "d.parent_id === id" in shape
    assert "removed: removed ? { id: removed.id } : null" in shape


def test_a_project_is_looked_at_before_the_session_is_given_up(tmp_path, monkeypatch):
    """The 'closes the N datasets' question must come after the file has been
    checked, or a folder in the box — what the browser puts there on a click
    — confirms and then fails. The picker stays open on a refused path."""
    from agrosuite.demo import synthetic_harvest

    opening = _block("openProject")
    assert opening.index("this.peekProject(path)") < opening.index("this.confirmReplace(")
    assert opening.index("this.confirmReplace(") < opening.index('"/api/session/open"')
    dialogs = _block("bindDialogs")
    picker = dialogs[dialogs.index("this.isProjectFile(path)"):dialogs.index('"/api/import/path"')]
    assert picker.index("this.peekProject(path)") < picker.index('.close()')
    assert "if (!peek) return;" in picker
    peeking = _block("peekProject")
    assert "peek.ok" in peeking and "peek.reason" in peeking and "return null;" in peeking

    # Every field the page reads off the answer is one the server sends.
    monkeypatch.setenv("AGROSUITE_HOME", str(tmp_path / "home"))
    state = session_mod.Session()
    state.add(synthetic_harvest(), "x", "demo")
    from agrosuite.app import persist
    persist.save_session(state, tmp_path / "x.agrosuite", "2026-09-17T10:00:00-06:00")
    accepted = persist_routes._peek(tmp_path / "x.agrosuite")
    refused = persist_routes._peek(tmp_path)
    assert accepted["ok"] is True and refused["ok"] is False and "reason" in refused
    carried = set(accepted) | set(refused)
    read = set(re.findall(r"\bpeek\.([a-z_]+)", opening + peeking))
    assert read and read <= carried, sorted(read - carried)


def test_the_trial_layout_comes_back_with_the_file():
    """The dialog promises the whole session; the layout is what Export
    offers as the prescription, so it must be on the map again, not only in
    the server's copy of the project."""
    assert "this.state.design = result.design?.result || null" in _block("sessionReplaced")
    assert "this.state.design = null" in _block("resetSessionState")
    tab = _block("tabEnsaio")
    assert "this.renderDesignReport(this.state.design)" in tab
    assert "this.drawDesignWithGuidance()" in tab
    assert "MapView.setFeatures(this.state.design.features" in _block("drawDesignWithGuidance")
    # What the open route hands back is what the page reads.
    assert '"design": result["design"]' in Path(persist_routes.__file__).read_text(encoding="utf-8")
    assert "design" in session_mod.default_project()


def test_the_save_hint_states_the_rule():
    """One rule on both sides: the extension decides, and only a full path."""
    dialog = INDEX[INDEX.index('id="dlg-save-project"'):INDEX.index("</dialog>", INDEX.index('id="dlg-save-project"'))]
    assert "Ending in .agrosuite it is the file" in dialog
    assert "folder" in dialog and "created if it is not there" in dialog
    assert "full path" in dialog.lower()


def test_a_missing_recent_file_is_shown_not_opened():
    recent = _block("renderRecentProjects")
    assert '" unavailable"' in recent
    assert "if (item.exists) { this.openProject(item.path); return; }" in recent


def test_the_save_dialog_cannot_submit_twice():
    """busy() blocks the pointer only; Enter in the inputs still reaches
    saveProject, and two saves in flight race the server's exists() check."""
    save = _block("saveProject")
    assert "if (this.state.saving) return;" in save
    assert "this.state.saving = true;" in save and "this.state.saving = false;" in save
    assert save.index("this.state.saving = true;") < save.index('"/api/session/save"')
    assert "saving: false" in APP_JS[:APP_JS.index("async api(")]


def test_the_name_travels_with_the_save():
    """One request, not a rename followed by a save: a refused save must
    leave the project's name as it was, on both sides."""
    save = _block("saveProject")
    assert '"/api/project"' not in save
    assert "name" in set(persist_routes.SaveRequest.model_fields)
    # Which file is open is the server's knowledge; the page never guesses
    # it from the name, which the file name only resembles.
    assert "overwrite: false" in save
    assert "sameFile" not in save


def test_the_open_file_comes_from_the_server():
    """After a reload the page must know where the next save goes."""
    recent = _block("refreshRecent")
    assert "this.state.projectFile = payload.current" in recent
    assert "this.renderProjectFile()" in recent
    assert "this.refreshRecent()" in _block("init")
    for name in ("saveProject", "sessionReplaced"):
        assert "this.state.projectFile = result.file" in _block(name), name
    dialog = _block("openSaveDialog")
    assert "current.named_after_project ? current.folder : current.path" in dialog

    # Every field the page reads off the file is one the server sends.
    state = session_mod.Session()
    state.project_file = {"path": Path("/x/y.agrosuite"), "saved_at": None, "token": None}
    carried = set(persist_routes._file_payload(state))
    read = set()
    for name in ("openSaveDialog", "renderProjectFile", "saveProject"):
        read |= set(re.findall(r"\b(?:current|file|result\.file)\.([a-z_]+)", _block(name)))
    assert read and read <= carried, sorted(read - carried)
