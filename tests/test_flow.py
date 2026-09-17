"""Two tracks, and nothing that blocks.

The app serves two jobs a season apart: planning a trial from a boundary,
and evaluating one after harvest. Neither is a gate. What these tests
protect is that promise in every layer it lives in — the workflow module
that models it, the first look that suggests the next step, the project
state the server reports, the tabs the interface draws — plus the rename
that took the acronym out of everything the user reads while leaving the
wire names alone, so a project file saved yesterday still opens.
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from agrosuite.app import persist  # noqa: E402
from agrosuite.app import session as session_mod  # noqa: E402
from agrosuite.clean import pipeline as clean_pipeline  # noqa: E402
from agrosuite.core import preflight  # noqa: E402
from agrosuite.core import workflow  # noqa: E402
from agrosuite.demo import synthetic_harvest, synthetic_trial  # noqa: E402

STATIC = ROOT / "agrosuite" / "app" / "static"
APP_JS = (STATIC / "app.js").read_text(encoding="utf-8")
INDEX_HTML = (STATIC / "index.html").read_text(encoding="utf-8")
SAVED_AT = "2026-09-17T10:00:00-06:00"


def _block(name: str) -> str:
    """The source of one App method, up to its closing brace at column two."""
    match = re.search(rf"^  (?:async )?{name}\(", APP_JS, flags=re.MULTILINE)
    assert match, f"App.{name} is not defined"
    return APP_JS[match.start():APP_JS.index("\n  },\n", match.start())]


# ==========================================================================
# The two tracks
# ==========================================================================

def test_the_two_tracks_are_the_two_jobs():
    """Plan a trial before the season, evaluate it after harvest. Each track
    names the goal it sets, and the goal decides the track — there is no
    third place for the two to disagree."""
    assert set(workflow.TRACKS) == {"plan", "evaluate"}
    assert workflow.TRACKS["plan"]["label"] == "Plan a trial"
    assert workflow.TRACKS["evaluate"]["label"] == "Evaluate a trial"
    assert workflow.track_of("trial_design") == "plan"
    assert workflow.track_of("difm") == "evaluate"
    # A goal from a file this version never wrote still lands somewhere.
    assert workflow.track_of("something-else") == "evaluate"


def test_planning_a_trial_needs_a_field_and_nothing_else():
    """No yield map, no prices, no cleaning: a boundary — or any layer whose
    outline can stand for the field — is the whole requirement."""
    spec = workflow.GOALS["trial_design"]
    assert spec["label"] == "Plan a trial"
    assert [key for key, _, _ in spec["requires"]] == ["field"]

    assert not workflow.evaluate("trial_design", set())["ready"]
    assert workflow.evaluate("trial_design", {"boundary"})["ready"]
    # A yield map is a field outline too; so is anything loaded at all.
    assert workflow.evaluate("trial_design", {"yield"})["ready"]
    assert workflow.evaluate("trial_design", set(), has_layers=True)["ready"]

    evaluation = workflow.evaluate("trial_design", {"boundary"})
    assert evaluation["track"] == "plan"
    assert "prices" not in {r["key"] for r in evaluation["requirements"]}


def test_the_terrain_hint_is_only_offered_to_a_project_that_has_one():
    """Slope matters when laying strips out, but naming a terrain analysis to
    someone who has loaded no elevation layer is one more line of noise in a
    panel that exists to say what is missing."""
    without = workflow.evaluate("trial_design", {"boundary"})
    assert "terrain" not in {r["key"] for r in without["optional"]}
    with_terrain = workflow.evaluate("trial_design", {"boundary", "terrain"})
    optional = {r["key"]: r for r in with_terrain["optional"]}
    assert "guidance" in optional
    assert optional["terrain"]["satisfied"] is True


def test_the_economic_analysis_kept_its_key_and_lost_the_acronym():
    """The label is what the user reads; the key travels in project files,
    in stored reports and over HTTP, so it stays as it was."""
    assert workflow.GOALS["difm"]["label"] == "Economic analysis"
    assert workflow.evaluate("difm", {"yield"})["label"] == "Economic analysis"
    goals = workflow.describe()["goals"]
    assert goals["difm"]["label"] == "Economic analysis"
    assert goals["trial_design"]["track"] == "plan"
    assert "difm" not in json.dumps(
        [g["label"] for g in goals.values()] + [
            s["label"] for s in workflow.describe()["stages"]
        ]
    ).lower()


# ==========================================================================
# Nothing blocks
# ==========================================================================

def test_a_file_that_arrives_clean_still_reaches_the_analysis():
    """Cleaning is a suggestion, not a gate. A project whose files are
    reviewed and whose requirements are met is at the analysis, with no
    cleaning report anywhere in the session."""
    ready = workflow.evaluate("difm", {"yield", "as_applied"}, has_prices=True)["ready"]
    assert ready
    assert workflow.stage_of(2, reviewed=True, cleaned=False, analysed=False,
                             exported=False, goal="difm", ready=ready) == "analyse"

    # Not yet reviewed, or not yet able to run: cleaning is what is worth
    # doing next — still only a suggestion.
    assert workflow.stage_of(2, False, False, False, False, goal="difm", ready=ready) == "review"
    assert workflow.stage_of(2, True, False, False, False, goal="difm", ready=False) == "clean"
    # Cleaned, and the analysis can run: past it either way.
    assert workflow.stage_of(2, True, True, False, False, goal="difm", ready=False) == "analyse"


def test_the_old_positional_call_still_works():
    """server.py and anything written against the five-argument form keep
    working: the new arguments have defaults."""
    assert workflow.stage_of(0, False, False, False, False) == "load"
    assert workflow.stage_of(1, False, False, False, False) == "review"
    assert workflow.stage_of(1, True, True, True, False) == "export"


def test_the_plan_track_goes_field_design_export():
    assert workflow.stages_for("trial_design") == ("load", "design", "export")
    assert workflow.stage_of(0, False, False, False, False, goal="trial_design") == "load"
    assert workflow.stage_of(1, False, False, False, False, goal="trial_design") == "design"
    assert workflow.stage_of(1, False, False, False, False, goal="trial_design",
                             designed=True) == "export"


def test_every_stage_is_reachable_and_a_skipped_one_says_so():
    stages = workflow.stage_states("difm", "analyse", reviewed=True, cleaned=False)
    by_key = {s["key"]: s for s in stages}
    assert [s["key"] for s in stages] == list(workflow.STAGES)
    assert all(s["reachable"] for s in stages)
    assert all(s["state"] != "blocked" for s in stages)
    assert by_key["load"]["state"] == "done"
    assert by_key["review"]["state"] == "done"
    # Passed over, not failed, and one click away.
    assert by_key["clean"]["state"] == "skippable"
    assert by_key["analyse"]["state"] == "current"
    assert by_key["export"]["state"] == "ahead"


def test_loading_is_never_reported_as_skipped():
    """Every later stage is reached with something open, so 'load' behind the
    current stage is done — reporting it as passed over would be a lie about
    the one thing that certainly happened."""
    for stage in ("review", "clean", "analyse", "export"):
        states = {s["key"]: s["state"] for s in workflow.stage_states("difm", stage)}
        assert states["load"] == "done", stage


def test_every_next_step_says_what_stopping_here_would_give():
    """The next step is an offer. What the stage already produces, and how to
    take it away, travels with it."""
    ready = workflow.evaluate("difm", {"yield", "as_applied"}, has_prices=True)
    for stage in ("load", "review", "clean", "analyse", "export"):
        action = workflow.next_action(stage, "difm", ready)
        assert action["blocks"] is False
        stop = action["stop"]
        assert stop["produces"] and stop["export"], stage
        assert "difm" not in (stop["produces"] + stop["export"]).lower()

    design = workflow.next_action("design", "trial_design",
                                  workflow.evaluate("trial_design", {"boundary"}))
    assert design["step"] == "design"
    assert "monitor" in design["stop"]["export"].lower()


def test_what_is_missing_names_the_gap_without_closing_anything():
    incomplete = workflow.evaluate("difm", {"yield"}, has_prices=False)
    assert not incomplete["ready"]
    assert incomplete["blocks"] is False
    assert {r["key"] for r in incomplete["missing"]} == {"rate", "prices"}


# ==========================================================================
# The first look points at what the file is for
# ==========================================================================

def test_a_boundary_is_offered_the_trial_design():
    import geopandas as gpd
    from shapely.geometry import Polygon

    from agrosuite.core.dataset import Dataset, DatasetMeta

    ring = Polygon([(-105.83, 50.45), (-105.82, 50.45), (-105.82, 50.46), (-105.83, 50.46)])
    frame = gpd.GeoDataFrame({"name": ["NW 14"]}, geometry=[ring], crs="EPSG:4326")
    dataset = Dataset(
        frame.drop(columns="geometry").assign(lon=-105.825, lat=50.455),
        DatasetMeta(name="NW 14 boundary", operation="boundary", geometry_type="polygon"),
        list(frame.geometry),
    )
    report = preflight.run(dataset)
    assert report["suggested_role"] == "boundary"
    assert report["next_step"]["step"] == "design"
    assert "trial" in report["next_step"]["label"].lower()


def test_a_prescription_is_offered_the_export():
    trial = synthetic_trial()
    trial.meta.operation = "prescription"
    report = preflight.run(trial)
    assert report["suggested_role"] == "plan"
    assert report["next_step"]["step"] == "export"


def test_a_clean_file_is_offered_the_economics_and_a_raw_one_the_cleaning():
    """Read from the data: the clean copy carries the cleaning's own mark, and
    a file with no track columns at all was never a machine log."""
    harvest = synthetic_harvest()
    result = clean_pipeline.run(harvest)

    assert not preflight.looks_cleaned(harvest)
    assert preflight.looks_cleaned(result.clean)
    assert result.clean.meta.extra["cleaning"]["removed"] > 0

    assert preflight.run(harvest)["next_step"]["step"] == "clean"
    clean_step = preflight.run(result.clean)["next_step"]
    assert clean_step["step"] == "analyse"
    assert "difm" not in (clean_step["label"] + clean_step["why"]).lower()


def test_an_aggregated_table_is_not_a_machine_log():
    """A joined grid — one row per cell, no timestamp, no speed, no swath —
    has nothing left for the filters to catch."""
    joined = synthetic_trial()
    joined.df = joined.df.drop(columns=["timestamp", "speed_kmh", "swath_m"],
                               errors="ignore")
    assert preflight.looks_cleaned(joined)
    assert preflight.run(joined)["next_step"]["step"] == "analyse"


# ==========================================================================
# The project state the server reports
# ==========================================================================

@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from agrosuite.app import server as server_mod

    with TestClient(server_mod.app) as test_client:
        yield test_client


@pytest.fixture
def clean_session():
    """A session of its own for each test that changes the project."""
    from agrosuite.app import server as server_mod

    server_mod.state.clear()
    server_mod.state.project = session_mod.default_project()
    yield server_mod.state
    server_mod.state.clear()
    server_mod.state.project = session_mod.default_project()


def test_the_project_reports_its_track_and_its_stages(client, clean_session):
    project = client.get("/api/project").json()
    assert project["track"] == "evaluate"
    assert project["track_label"] == "Evaluate a trial"
    assert [s["key"] for s in project["stages"]] == list(workflow.STAGES)
    assert all(s["reachable"] for s in project["stages"])
    assert project["next_action"]["stop"]["produces"]


def test_choosing_the_plan_track_changes_the_goal_and_the_stages(client, clean_session):
    project = client.post("/api/project", json={"goal": "trial_design"}).json()
    assert project["goal"] == "trial_design"
    assert project["track"] == "plan"
    assert [s["key"] for s in project["stages"]] == ["load", "design", "export"]
    assert project["evaluation"]["label"] == "Plan a trial"


def test_a_demo_trial_can_reach_the_economics_without_being_cleaned(client, clean_session):
    """The path a grower with an already clean file takes: open it, say what
    it is, and the project stands at the analysis with nothing cleaned."""
    loaded = client.post("/api/import/demo", json={"kind": "trial"}).json()
    assert "DIFM" not in loaded["label"]

    client.post("/api/project/role", json={"dataset_id": loaded["id"], "role": "yield"})
    client.post("/api/project/prices",
                json={"crop_price": 0.55, "input_cost": 1.35, "currency": "CAD",
                      "crop": "canola"})
    project = client.get("/api/project").json()
    assert project["evaluation"]["ready"]
    assert project["stage"] == "analyse"
    assert not any(d["origin"] == "clean" for d in client.get("/api/datasets").json()["datasets"])

    report = client.post(f"/api/datasets/{loaded['id']}/difm",
                         json={"crop_price": 0.55, "input_cost": 1.35}).json()
    assert report["economics"]["optimum_rate"] > 0


def test_the_analysis_refusal_no_longer_names_the_acronym(client, clean_session):
    loaded = client.post("/api/import/demo", json={"kind": "harvest"}).json()
    response = client.post(f"/api/datasets/{loaded['id']}/difm",
                           json={"rate_column": "not_a_column"})
    assert response.status_code == 400
    assert "DIFM" not in response.json()["detail"]
    assert "economic analysis" in response.json()["detail"].lower()


# ==========================================================================
# A project file written before any of this still opens
# ==========================================================================

def test_a_project_saved_before_the_two_tracks_still_opens(tmp_path, monkeypatch):
    """The saved project dict carries goal 'difm' and no track at all. The
    track is derived from the goal, so the file needs nothing added to it."""
    monkeypatch.setenv("AGROSUITE_HOME", str(tmp_path / "home"))

    state = session_mod.Session()
    harvest = state.add(synthetic_harvest(), "Harvest with defects (demo)", "demo")
    harvest.role = "yield"
    harvest.reports["preflight"] = preflight.run(harvest.dataset)
    state.project.update({
        "name": "North quarter / 2025",
        "goal": "difm",                       # the wire name, as it was saved
        "roles": {harvest.id: "yield"},
        "prices": {"crop_price": 0.55, "input_cost": 1.35, "currency": "CAD",
                   "crop": "canola"},
        "reviewed": {harvest.id},
    })
    target = tmp_path / "north.agrosuite"
    persist.save_session(state, target, SAVED_AT)

    # What is on the disk is the old shape: a goal, and no mention of a track.
    manifest = persist.read_manifest(target)
    assert manifest["project"]["goal"] == "difm"
    assert "track" not in manifest["project"]

    fresh = session_mod.Session()
    loaded = persist.load_session(fresh, target)
    assert loaded["project"] == "North quarter / 2025"
    assert fresh.project["goal"] == "difm"
    assert fresh.project["roles"] == {harvest.id: "yield"}
    assert fresh.project["reviewed"] == {harvest.id}
    # And it lands on the evaluate track, with the renamed label.
    assert workflow.track_of(fresh.project["goal"]) == "evaluate"
    assert workflow.GOALS[fresh.project["goal"]]["label"] == "Economic analysis"


def test_a_project_saved_on_the_plan_track_round_trips(tmp_path, monkeypatch):
    monkeypatch.setenv("AGROSUITE_HOME", str(tmp_path / "home"))

    state = session_mod.Session()
    state.add(synthetic_harvest(), "Field", "demo")
    state.project["goal"] = "trial_design"
    target = tmp_path / "plan.agrosuite"
    persist.save_session(state, target, SAVED_AT)

    fresh = session_mod.Session()
    persist.load_session(fresh, target)
    assert fresh.project["goal"] == "trial_design"
    assert workflow.track_of(fresh.project["goal"]) == "plan"


# ==========================================================================
# What the interface says
# ==========================================================================

def test_the_tabs_are_in_order_and_carry_no_numbers():
    """Data, then Trial design, then Cleaning, then Economics, then Export.
    The order is asserted as an order, not as a list, so a tab added between
    two of them — Terrain belongs after Data — does not break this."""
    tabs = re.findall(r'<button data-tab="(\w+)"[^>]*>([^<]+)</button>', INDEX_HTML)
    labels = [label.strip() for _, label in tabs]
    expected = ["Data", "Trial design", "Cleaning", "Economics", "Export"]
    assert set(expected) <= set(labels), labels
    positions = [labels.index(label) for label in expected]
    assert positions == sorted(positions), labels
    # The numbers promised an order that is not required.
    assert not any(re.match(r"\d", label) for label in labels)


def test_nothing_the_user_reads_says_difm():
    """The acronym is gone from every string on screen; the keys it named —
    the tab, the endpoint, the cache and the stored report — are not."""
    for path in (STATIC / "index.html", STATIC / "app.js", STATIC / "style.css",
                 ROOT / "README.md", ROOT / "agrosuite" / "report.py",
                 ROOT / "agrosuite" / "core" / "workflow.py"):
        assert "DIFM" not in path.read_text(encoding="utf-8"), path.name

    # What is left in app.js is element ids, the endpoint, the cache key and
    # the stored report key — plus the comments that explain why they stayed.
    for line in APP_JS.splitlines():
        stripped = line.strip()
        if "difm" not in line or stripped.startswith(("//", "*", "/*")):
            continue
        assert re.search(r'"difm"|difm-|-difm|:difm|/difm|difm:|Difm|has_difm_report',
                         line), line


def test_every_tab_says_what_it_needs_and_what_it_gives():
    """A tab that cannot work yet names the missing piece and carries the one
    button that fixes it; a tab that can ends with what it produces alone."""
    for name in ("tabDados", "tabTerrain", "tabLimpeza", "tabDifm", "tabEnsaio",
                 "tabExportar"):
        assert "this.missingPanel(" in _block(name), name
    for name in ("tabDados", "tabTerrain", "tabLimpeza", "tabDifm", "tabEnsaio"):
        assert "this.producesPanel(" in _block(name), name
    # And none of them is closed off: no tab button is ever disabled.
    nav = re.search(r'<nav class="steps".*?</nav>', INDEX_HTML, re.S)
    assert nav and "disabled" not in nav.group()


def test_the_economics_panel_reads_as_an_economic_analysis():
    tab = _block("tabDifm")
    assert "Economic analysis: optimum rate from your trial" in tab
    assert "Run the economic analysis" in tab
    # And the wire names it keeps are commented where they are used.
    assert "wire names" in tab


def test_the_stage_strip_offers_both_tracks_and_keeps_every_stage_clickable():
    strip = _block("renderStageStrip")
    assert "data-track=" in strip and "this.setTrack(" in _block("renderStageStrip")
    assert 'button class="stage' in strip and "data-stage=" in strip
    assert "goToStep(button.dataset.stage)" in strip
    # With nothing loaded the strip invites either track rather than one.
    assert "Either way" in strip
    # And the step the strip proposes says what stopping there would give.
    assert "next.stop" in strip


def test_the_trial_design_step_has_a_tab_to_go_to():
    assert 'design: "ensaio"' in _block("goToStep")
    assert 'analyse: "difm"' in _block("goToStep")


# ==========================================================================
# In a real browser
# ==========================================================================

sync_api = pytest.importorskip("playwright.sync_api")
CHROMIUM = Path(os.environ.get("AGROSUITE_TEST_CHROMIUM", "/opt/pw-browsers/chromium"))


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    """The app on a free port, stopped when the module is done."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    log = tmp_path_factory.mktemp("flow-server") / "server.log"
    with log.open("w") as handle:
        proc = subprocess.Popen(
            [sys.executable, "-m", "agrosuite", "--port", str(port), "--no-browser"],
            cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
        )
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(60):
            try:
                with urllib.request.urlopen(f"{base}/api/health", timeout=1) as response:
                    if response.status == 200:
                        break
            except OSError:
                if proc.poll() is not None:
                    pytest.fail(f"the app did not start:\n{log.read_text()}")
                time.sleep(0.5)
        else:
            pytest.fail(f"the app did not answer on {base}:\n{log.read_text()}")
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as playwright:
        options = {"executable_path": str(CHROMIUM)} if CHROMIUM.exists() else {}
        try:
            instance = playwright.chromium.launch(**options)
        except Exception as exc:  # no browser on this machine
            pytest.skip(f"Chromium is not available: {exc}")
        yield instance
        instance.close()


@pytest.fixture
def page(browser, served):
    page = browser.new_page(viewport={"width": 1400, "height": 900})
    page.errors = []
    page.on("pageerror", lambda err: page.errors.append(str(err)))
    # "New project" asks before closing loaded datasets, and a dismissed
    # prompt would leave the last test's files in this one's session.
    page.on("dialog", lambda dialog: dialog.accept())
    page.goto(served)
    page.wait_for_selector("#btn-demo-harvest")
    page.evaluate("async () => { await App.newProject(); }")
    page.wait_for_function("() => App.state.datasets.length === 0")
    yield page
    assert page.errors == [], page.errors
    page.close()


TABS = ["dados", "ensaio", "limpeza", "difm", "exportar"]


def _tab(page, name):
    page.click(f'#steps button[data-tab="{name}"]')


def test_every_tab_explains_itself_with_nothing_loaded(page):
    """A fresh app: each tab says what it is for and offers one way in. None
    throws, and none is closed."""
    for name in TABS:
        _tab(page, name)
        panel = page.locator("#right-panel")
        assert panel.locator("button[data-cta]").count() >= 1, name
        assert len(panel.inner_text()) > 60, name
    assert page.locator("#steps button[disabled]").count() == 0
    assert page.errors == []


def test_a_trial_is_planned_and_exported_without_ever_opening_cleaning(page):
    """Track A end to end: a field, a layout, an AB line, a package for a
    Raven terminal — no yield map, no prices, no cleaning."""
    page.click("#btn-demo-trial")
    page.wait_for_selector("#dataset-list .card")
    page.wait_for_function("() => App.state.selected !== null")
    page.click('#stage-strip button[data-track="plan"]')
    page.wait_for_selector('#stage-strip button[data-track="plan"].on')
    assert page.evaluate("() => App.state.project.goal") == "trial_design"

    _tab(page, "ensaio")
    page.click("#btn-run-design")
    page.wait_for_selector("#btn-make-ab")
    page.click("#btn-make-ab")
    page.wait_for_selector("#ab-status .note")

    page.click("#btn-design-export")
    page.wait_for_selector("#pkg-monitor")
    page.select_option("#pkg-monitor", "raven")
    page.click("#btn-run-package")
    page.wait_for_selector('#toasts .toast:has-text("Package built"), #export-report .panel',
                           timeout=30000)
    assert page.locator("#export-report").inner_text().strip()
    # The cleaning tab was never opened, and nothing asked for it.
    assert page.evaluate("() => App.state.tab") == "exportar"


def test_the_data_tab_prints_and_exports_on_its_own(page):
    """Track B stopped at the first step: a printed first look and the file
    as it stands are both complete results."""
    page.click("#btn-demo-harvest")
    page.wait_for_selector("#dataset-list .card")
    page.wait_for_function("() => App.state.selected !== null")
    page.evaluate("() => { window.__opened = []; window.open = (url) => "
                  "{ window.__opened.push(url); return true; }; }")

    _tab(page, "dados")
    page.click('#right-panel .produces button[data-cta="print"]')
    page.wait_for_selector('#toasts .toast:has-text("Report written")')
    assert page.evaluate("() => window.__opened").pop().startswith("/api/download/")

    page.click('#right-panel .produces button[data-cta="export-file"]')
    page.wait_for_selector("#exp-source")
    assert page.evaluate("() => App.state.exportMode") == "files"


def test_cleaning_ends_with_the_clean_copy_as_a_file(page):
    page.click("#btn-demo-harvest")
    page.wait_for_selector("#dataset-list .card")
    page.wait_for_function("() => App.state.selected !== null")
    _tab(page, "limpeza")
    page.click("#btn-run-clean")
    page.wait_for_selector('#toasts .toast:has-text("Cleaning done")', timeout=60000)

    page.click('#right-panel .produces button[data-cta="export-clean"]')
    page.wait_for_selector("#exp-source")
    assert page.evaluate("() => App.state.selected.origin") == "clean"


def test_the_economics_tab_runs_without_a_cleaning(page):
    """The already-clean file's path: straight to Economics, prices typed in
    there, and an optimum out."""
    page.click("#btn-demo-trial")
    page.wait_for_selector("#dataset-list .card")
    page.wait_for_function("() => App.state.selected !== null")
    _tab(page, "difm")
    page.fill("#difm-price", "0.55")
    page.fill("#difm-cost", "1.35")
    page.click("#btn-run-difm")
    page.wait_for_selector("#difm-report .panel", timeout=60000)

    # The panel heading is upper-cased by the stylesheet, not by the markup.
    text = page.locator("#right-panel").inner_text()
    assert "economic analysis: optimum rate from your trial" in text.lower()
    assert "difm" not in text.lower()
    assert page.evaluate("() => Object.keys(App.state.reports).length") == 1
