"""Machine profiles: what the second review found, kept from coming back.

Each test here stands for one way a saved machine could be lost, bent or
misread without the interface saying so: a name typed twice replacing a
machine with no question, a suggestion stored with 9.000134 m where the
file said 9, a profiles.json missing its list read as empty and written
over, a hand-edited profile the app would refuse poured into the cleaning
anyway, a pick forgotten on a tab switch, a refusal quoting km/h nobody
typed, and no way to edit a machine but retyping it.

What can be checked without a browser is checked in Python; what only shows
in a browser — the exact numbers a suggestion saves, the pick surviving a
tab switch, the question before a replace — runs in one, and is skipped
where Playwright or Chromium is not installed.
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
from fastapi import HTTPException

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agrosuite.core import profiles as profiles_mod
from agrosuite.core.profiles import MachineProfile

APP_JS = (ROOT / "agrosuite" / "app" / "static" / "app.js").read_text(encoding="utf-8")


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch) -> Path:
    """Every test gets an empty AGROSUITE_HOME, so none touches the real one."""
    target = tmp_path / "agrosuite_home"
    monkeypatch.setenv("AGROSUITE_HOME", str(target))
    return target


@pytest.fixture
def routes():
    from agrosuite.app import server as server_mod  # noqa: F401 - registers the routers
    from agrosuite.app.routes import profiles as profiles_routes

    return profiles_routes


def _block(name: str) -> str:
    """The source of one App method, up to its closing brace at column two."""
    match = re.search(rf"^  (?:async )?{name}\(", APP_JS, flags=re.MULTILINE)
    assert match, f"App.{name} is not defined"
    end = APP_JS.index("\n  },\n", match.start())
    return APP_JS[match.start():end]


def _write(home: Path, payload) -> Path:
    home.mkdir(parents=True, exist_ok=True)
    path = home / "profiles.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ==========================================================================
# 1. An existing name is replaced only once that is agreed
# ==========================================================================

def test_saving_under_an_existing_name_answers_409_until_replace_is_agreed(routes):
    """'My Combine' typed on another day is the same machine; the one entered
    months ago is not written over by a name that merely matches."""
    routes.save_profile(routes.ProfileRequest(
        name="My Combine", implement_width_m=18.288, notes="60 ft draper"))

    again = dict(name="my  combine", implement_width_m=12.192)
    with pytest.raises(HTTPException) as info:
        routes.save_profile(routes.ProfileRequest(**again))
    assert info.value.status_code == 409
    # The saved spelling, so the question names the machine as the list does.
    assert "A machine named 'My Combine' already exists" in info.value.detail
    assert "confirm replacing it" in info.value.detail
    kept = routes.list_profiles()["profiles"]
    assert [(p["name"], p["notes"]) for p in kept] == [("My Combine", "60 ft draper")]

    replaced = routes.save_profile(routes.ProfileRequest(**again, replace=True))
    assert [(p["name"], p["implement_width_m"]) for p in replaced["profiles"]] == \
        [("my  combine", pytest.approx(12.192))]


def test_the_core_asks_only_when_told():
    """Replacing stays the default in Python — that is how a machine is
    corrected — and the refusal carries the profile as it is saved."""
    profiles_mod.save_profile(MachineProfile(name="Drill", kind="seeder", implement_width_m=15))
    with pytest.raises(profiles_mod.ProfileExists) as info:
        profiles_mod.save_profile(MachineProfile(name="DRILL", kind="seeder", implement_width_m=12),
                                  replace=False)
    assert info.value.existing.name == "Drill"
    assert "'Drill' already exists" in str(info.value)
    assert profiles_mod.get_profile("drill").implement_width_m == 15

    profiles_mod.save_profile(MachineProfile(name="DRILL", kind="seeder", implement_width_m=12))
    assert profiles_mod.get_profile("drill").implement_width_m == 12


def test_a_bad_profile_is_refused_before_the_question_is_asked(routes):
    """Being asked 'replace?' and then told the width is missing is two
    round trips where one will do."""
    routes.save_profile(routes.ProfileRequest(name="Sprayer", kind="sprayer", implement_width_m=36))
    with pytest.raises(HTTPException) as info:
        routes.save_profile(routes.ProfileRequest(name="sprayer", kind="sprayer", implement_width_m=0))
    assert info.value.status_code == 400


def test_the_page_asks_before_replacing_and_not_when_updating():
    save = _block("saveMachine")
    assert "err.status !== 409" in save
    assert "window.confirm(" in save
    assert "replace: true" in save
    # Updating the machine that was picked is what the dialog opened for.
    assert "replace: !!editing && this.machineKey(name) === this.machineKey(editing)" in save


# ==========================================================================
# 2. A suggestion is saved with the numbers it came from
# ==========================================================================

def test_what_the_app_placed_in_a_field_is_read_back_exactly():
    """9 m shown as 29.528 ft and converted back is 9.000134 m; the dialog
    and the tabs remember the metric number behind what they show."""
    dialog = _block("renderMachineDialog")
    for field in ("mp-width", "mp-speed-min", "mp-speed-max"):
        assert f'this.machinePlaced("{field}"' in dialog, f"{field} is placed unremembered"
    scope = _block("machineScope")
    for field in ("p-speed_range-min", "p-speed_range-max", "design-width"):
        assert f'this.machinePlaced("{field}"' in scope, f"{field} is placed unremembered"
    reader = _block("machineFieldMetric")
    assert "placed.display === value ? placed.metric" in reader
    # Both readers go through it, so a typed value is still converted.
    assert 'this.machineFieldMetric("length", "mp-width")' in _block("saveMachine")
    assert "this.machineFieldMetric(kind, id)" in scope


# ==========================================================================
# 3. A profiles.json without its list is refused, not written over
# ==========================================================================

@pytest.mark.parametrize("payload,holds", [
    ({"version": 1}, "'version'"),
    ({}, "no keys at all"),
    ({"version": 1, "machines": []}, "'version', 'machines'"),
])
def test_a_file_without_its_profiles_list_is_refused_not_overwritten(home, payload, holds):
    """Read as empty, the next save wrote {"version": 1, "profiles": [one]}
    over whatever the file did hold."""
    path = _write(home, payload)
    before = path.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="has no 'profiles' list") as info:
        profiles_mod.list_profiles()
    assert holds in str(info.value) and str(path) in str(info.value)
    with pytest.raises(ValueError, match="Fix or move it away"):
        profiles_mod.save_profile(MachineProfile(name="x", implement_width_m=9))
    with pytest.raises(ValueError, match="Fix or move it away"):
        profiles_mod.delete_profile("x")
    assert path.read_text(encoding="utf-8") == before


def test_the_api_names_the_file_and_what_is_wrong(routes, home):
    path = _write(home, {"version": 1})
    with pytest.raises(HTTPException) as info:
        routes.list_profiles()
    assert info.value.status_code == 500
    assert "no 'profiles' list" in info.value.detail and str(path) in info.value.detail


def test_a_bare_list_is_still_taken_as_the_profiles(home):
    """A file that is the list itself is not broken, only terse."""
    _write(home, [{"name": "Bare", "implement_width_m": 9}])
    assert [p.name for p in profiles_mod.list_profiles()] == ["Bare"]


def test_the_pickers_say_a_refusal_once():
    """Every tab asks for the list again; the same refusal is not a toast at
    every tab switch, and a click on the save button hears it again."""
    load = _block("loadMachines")
    assert "this.state.machinesRefused !== err.message" in load
    assert "always ||" in load
    assert "this.loadMachines({ always: true })" in _block("openMachineDialog")
    assert "could not be read" in _block("machineOptions")


# ==========================================================================
# 4. A hand-edited profile the app would refuse is marked, not applied
# ==========================================================================

def test_a_hand_edited_profile_is_listed_with_its_problems(home, routes):
    """The file is edited by hand, so the list is not always what the app
    would have saved: the listing says what is wrong with each one."""
    _write(home, {"version": 1, "profiles": [
        {"name": "Fine", "kind": "combine", "implement_width_m": 9,
         "speed_min_kmh": 3, "speed_max_kmh": 8},
        {"name": "Upside down", "kind": "hovercraft", "monitor": "acme",
         "implement_width_m": 9, "speed_min_kmh": 9, "speed_max_kmh": 3},
    ]})
    listed = {p["name"]: p["problems"] for p in routes.list_profiles()["profiles"]}
    assert listed["Fine"] == []
    assert len(listed["Upside down"]) == 3
    assert any("hovercraft" in p for p in listed["Upside down"])
    assert any("acme" in p for p in listed["Upside down"])
    assert any("minimum speed must be below" in p for p in listed["Upside down"])
    # It can still be deleted, or saved again once corrected.
    assert [p["name"] for p in routes.delete_profile("Upside down")["profiles"]] == ["Fine"]


def test_a_profile_without_a_name_refuses_the_file(home):
    """Every other problem is reported under the profile's name; this one
    could be neither picked nor deleted."""
    path = _write(home, {"version": 1, "profiles": [{"name": "  ", "implement_width_m": 9}]})
    with pytest.raises(ValueError, match="empty name") as info:
        profiles_mod.list_profiles()
    assert str(path) in str(info.value)


def test_the_page_marks_them_and_does_not_fill_from_them():
    assert "needs attention" in _block("machineOptions")
    apply = _block("applyMachine")
    assert apply.index("profile.problems?.length") < apply.index(".fill(profile)")
    assert "return;" in apply[:apply.index(".fill(profile)")]
    assert "Update machine" in apply


# ==========================================================================
# 5. The pick survives a tab switch
# ==========================================================================

def test_the_picked_machine_is_remembered_per_tab():
    assert "machinePicked" in _block("pickMachine")
    bind = _block("bindMachinePicker")
    assert "this.machinePickedIn(scope)" in bind
    assert "this.applyMachine(scope, picked, { quiet: true })" in bind
    assert "this.state.machinePicked?.[scope]" in _block("machinePickedIn")
    # Deleted, it is forgotten on every tab; saved, it is the pick.
    assert "this.state.machinePicked[tab] = \"\"" in _block("deleteMachine")
    assert "[scope]: saved.name" in _block("saveMachine")


# ==========================================================================
# 6. A refused speed range is said in the units it was typed in
# ==========================================================================

def test_a_defaulted_speed_is_named_rather_than_quoted(routes):
    """12 km/h typed as the minimum with the maximum left empty: the refusal
    says the default filled the maximum instead of quoting 10 as typed."""
    with pytest.raises(HTTPException) as info:
        routes.save_profile(routes.ProfileRequest(
            name="Fast", kind="combine", implement_width_m=9, speed_min_kmh=12))
    assert info.value.status_code == 400
    assert "got 12 km/h and the typical maximum for a combine, 10 km/h" in info.value.detail

    with pytest.raises(HTTPException) as info:
        routes.save_profile(routes.ProfileRequest(
            name="Slow", kind="seeder", implement_width_m=9, speed_max_kmh=4))
    assert "got the typical minimum for a seeder, 5 km/h and 4 km/h" in info.value.detail

    # Both typed: both quoted, each with its unit.
    with pytest.raises(HTTPException) as info:
        routes.save_profile(routes.ProfileRequest(
            name="Both", implement_width_m=9, speed_min_kmh=9, speed_max_kmh=8))
    assert "(got 9 km/h and 8 km/h)" in info.value.detail


def test_the_page_checks_the_range_on_screen_before_posting():
    save = _block("saveMachine")
    check = save.index("low >= high")
    assert check < save.index('this.api("/api/profiles"')
    assert "Units.label.speed()" in save[check:save.index('this.api("/api/profiles"')]


# ==========================================================================
# 7. A picked machine is updated, not retyped
# ==========================================================================

def test_the_dialog_opens_on_the_picked_machine():
    opened = _block("openMachineDialog")
    assert "const picked = this.machinePickedIn(scope)" in opened
    assert "picked ? { ...picked }" in opened
    assert "editing: picked?.name || null" in opened
    assert '"Update machine…"' in _block("machineSaveLabel")
    # A suggestion into a machine being updated keeps its name.
    assert "if (this.state.machineDialog?.editing) profile.name" in _block("suggestMachine")


# ==========================================================================
# In a browser
# ==========================================================================

sync_api = pytest.importorskip("playwright.sync_api")

CHROMIUM = Path(os.environ.get("AGROSUITE_TEST_CHROMIUM", "/opt/pw-browsers/chromium"))


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    """The app on a free port, with profiles in a folder of its own."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    folder = tmp_path_factory.mktemp("server")
    home = folder / "home"
    log = folder / "server.log"
    with log.open("w") as handle:
        proc = subprocess.Popen(
            [sys.executable, "-m", "agrosuite", "--port", str(port), "--no-browser"],
            cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
            env={**os.environ, "AGROSUITE_HOME": str(home)},
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
        yield base, home / "profiles.json"
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
    """A fresh page on the Cleaning tab with the demo harvest selected and
    no machine saved; uncaught errors are collected."""
    base, stored = served
    if stored.exists():
        stored.unlink()
    page = browser.new_page(viewport={"width": 1400, "height": 900})
    page.errors = []
    page.on("pageerror", lambda err: page.errors.append(str(err)))
    page.goto(base)
    page.wait_for_selector("#btn-demo-harvest")
    page.request.post(f"{base}/api/import/demo", data=json.dumps({"kind": "harvest"}),
                      headers={"content-type": "application/json"})
    page.reload()
    page.wait_for_function("() => App.state.selectedId !== null")
    page.stored = stored
    _tab(page, "limpeza")
    yield page
    assert page.errors == [], page.errors
    page.close()


def _tab(page, name):
    page.click(f'#steps button[data-tab="{name}"]')
    page.wait_for_selector("select[data-machine-picker]")


def _saved(page):
    return json.loads(page.stored.read_text(encoding="utf-8"))["profiles"]


def _open_dialog(page, scope="clean"):
    page.click(f"#btn-machine-save-{scope}")
    page.wait_for_selector("#dlg-machine[open]")


def _save_dialog(page):
    page.click("#btn-machine-save")
    page.wait_for_function("() => !document.getElementById('dlg-machine').open")


def test_a_suggestion_is_saved_with_the_numbers_it_came_from(page):
    """The demo is a 9 m combine at 2.5–6 km/h. Shown in feet and mph and
    converted back it was saved as 9.000134 m and 2.499311 km/h."""
    _open_dialog(page)
    page.click("#btn-machine-suggest")
    page.wait_for_selector("#mp-problems .note")
    assert page.evaluate("() => App.value('mp-width')") == "29.528", "the Canadian default shows feet"
    _save_dialog(page)

    (profile,) = _saved(page)
    assert profile["implement_width_m"] == 9.0
    assert (profile["speed_min_kmh"], profile["speed_max_kmh"]) == (2.5, 6.0)
    # And a machine picked on the tab is read back the same way.
    page.select_option("#machine-clean", profile["name"])
    _open_dialog(page)
    page.fill("#mp-notes", "checked")
    _save_dialog(page)
    (profile,) = _saved(page)
    assert (profile["implement_width_m"], profile["speed_min_kmh"], profile["notes"]) == (9.0, 2.5, "checked")


def test_the_pick_survives_a_tab_switch_and_is_updated_in_place(page):
    _open_dialog(page)
    page.fill("#mp-name", "S780")
    page.fill("#mp-width", "40")
    _save_dialog(page)
    assert page.evaluate("() => App.value('machine-clean')") == "S780"
    assert page.text_content("#btn-machine-save-clean") == "Update machine…"

    _tab(page, "ensaio")
    assert page.evaluate("() => App.value('machine-design')") == ""
    _tab(page, "limpeza")
    assert page.evaluate("() => App.value('machine-clean')") == "S780"
    assert page.evaluate("() => App.checked('en-speed_range')") is True

    # No question: the picked machine is the one being updated.
    asked = []
    page.on("dialog", lambda dialog: (asked.append(dialog.message), dialog.accept()))
    _open_dialog(page)
    assert page.text_content("#dlg-machine h3") == "Update 'S780'"
    assert page.evaluate("() => App.value('mp-name')") == "S780"
    page.fill("#mp-notes", "new draper")
    _save_dialog(page)
    assert asked == []
    assert [(p["name"], p["notes"]) for p in _saved(page)] == [("S780", "new draper")]


def test_an_existing_name_is_replaced_only_after_asking(page):
    _open_dialog(page)
    page.fill("#mp-name", "S780")
    page.fill("#mp-width", "40")
    _save_dialog(page)
    page.select_option("#machine-clean", "")

    # Playwright dismisses a confirm nobody handles: the answer is "no".
    _open_dialog(page)
    page.fill("#mp-name", "s780")
    page.fill("#mp-width", "30")
    page.click("#btn-machine-save")
    page.wait_for_timeout(300)
    assert page.evaluate("() => document.getElementById('dlg-machine').open")
    assert [(p["name"], round(p["implement_width_m"], 3)) for p in _saved(page)] == [("S780", 12.192)]

    asked = []
    page.on("dialog", lambda dialog: (asked.append(dialog.message), dialog.accept()))
    _save_dialog(page)
    assert len(asked) == 1 and "'S780' already exists" in asked[0]
    assert [(p["name"], round(p["implement_width_m"], 3)) for p in _saved(page)] == [("s780", 9.144)]


def test_an_inverted_range_is_refused_in_the_units_on_screen(page):
    _open_dialog(page)
    page.fill("#mp-name", "Inverted")
    page.fill("#mp-width", "30")
    page.fill("#mp-speed-min", "7")
    page.fill("#mp-speed-max", "3")
    page.click("#btn-machine-save")
    page.wait_for_selector(".toast.warn")
    assert "got 7 and 3 mph" in page.text_content(".toast.warn .m")
    assert page.evaluate("() => document.getElementById('dlg-machine').open")
    assert not page.stored.exists()


def test_a_broken_profile_is_marked_and_not_applied(page):
    page.stored.parent.mkdir(parents=True, exist_ok=True)
    page.stored.write_text(json.dumps({"version": 1, "profiles": [
        {"name": "Broken", "kind": "hovercraft", "implement_width_m": 9,
         "speed_min_kmh": 9, "speed_max_kmh": 3},
    ]}), encoding="utf-8")
    _tab(page, "ensaio")
    page.wait_for_function(
        "() => [...document.querySelectorAll('#machine-design option')]"
        ".some((o) => o.textContent.includes('needs attention'))")
    before = page.evaluate("() => App.value('design-width')")
    page.select_option("#machine-design", "Broken")
    page.wait_for_selector(".toast.warn")
    assert "hovercraft" in page.text_content(".toast.warn .m")
    assert page.evaluate("() => App.value('design-width')") == before
    assert page.text_content("#btn-machine-save-design") == "Update machine…"


def test_an_unreadable_file_is_said_once_and_left_alone(page):
    page.stored.parent.mkdir(parents=True, exist_ok=True)
    page.stored.write_text('{"version": 1}', encoding="utf-8")
    _tab(page, "ensaio")
    page.wait_for_selector(".toast.error")
    _tab(page, "limpeza")
    _tab(page, "ensaio")
    page.wait_for_timeout(500)
    assert page.locator(".toast.error").count() == 1
    assert "no 'profiles' list" in page.text_content(".toast.error .m")
    assert page.evaluate("() => document.querySelector('#machine-design option').textContent") == \
        "— profiles.json could not be read —"
    assert page.stored.read_text(encoding="utf-8") == '{"version": 1}'
