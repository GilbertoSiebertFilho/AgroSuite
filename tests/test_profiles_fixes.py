"""Machine profiles: what the first review found, kept from coming back.

Each test here stands for one way a profile could be saved and then be
wrong, unusable or undeletable without the interface saying so: a name the
picker could not hold, a 0 that turned into the kind's default, a fraction
of a pass, a value rounded on its way to disk, a name a browser folds away,
a disk error shown as a traceback, and a suggestion for the wrong machine.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agrosuite.core import profiles as profiles_mod
from agrosuite.core.profiles import MachineProfile
from agrosuite.demo import synthetic_harvest

STATIC = Path(__file__).resolve().parents[1] / "agrosuite" / "app" / "static"
APP_JS = (STATIC / "app.js").read_text(encoding="utf-8")


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


# ==========================================================================
# Names
# ==========================================================================

def test_a_name_with_quotes_survives_the_page_markup():
    """'S780 "40 ft"' is a normal way to name a header. Put unescaped into a
    value="…" attribute it ended at the first quote, so the picker held
    'S780 ' and could neither fill nor delete the machine."""
    escape = _block("escape")
    assert "&quot;" in escape and "&#39;" in escape
    # The picker and the dialog both put the name into an attribute through it.
    assert 'value="${this.escape(value)}"' in _block("machineOptions")
    assert 'id="mp-name" value="${this.escape(p.name)}"' in _block("renderMachineDialog")


@pytest.mark.parametrize("name", [".", "..", " . ", ".. "])
def test_dot_names_are_refused_because_a_browser_folds_them_away(name):
    """encodeURIComponent leaves '.' and '..' as they are, and the browser
    turns DELETE /api/profiles/.. into DELETE /api/: saved, never deletable."""
    with pytest.raises(ValueError, match="cannot be a name"):
        profiles_mod.save_profile(MachineProfile(name=name, implement_width_m=9))
    assert profiles_mod.list_profiles() == []


@pytest.mark.parametrize("name", ["...", "a/..", "./b", "S780 \"40 ft\""])
def test_names_that_merely_look_odd_are_fine(name):
    profiles_mod.save_profile(MachineProfile(name=name, implement_width_m=9))
    assert [p.name for p in profiles_mod.list_profiles()] == [name]
    profiles_mod.delete_profile(name)


def test_api_refuses_a_dot_name_with_a_message(routes):
    with pytest.raises(HTTPException) as info:
        routes.save_profile(routes.ProfileRequest(name="..", implement_width_m=9))
    assert info.value.status_code == 400
    assert "'..' cannot be a name" in info.value.detail
    assert routes.list_profiles()["profiles"] == []


# ==========================================================================
# Numbers
# ==========================================================================

@pytest.mark.parametrize("passes", [1.5, "2.5", 0.5])
def test_a_fraction_of_a_pass_is_refused_not_truncated(passes):
    """int(1.5) is 1, silently; the file would then hold a number nobody typed."""
    with pytest.raises(ValueError, match="'passes_per_strip' must be a whole number"):
        MachineProfile.from_dict({"name": "x", "implement_width_m": 9, "passes_per_strip": passes})


@pytest.mark.parametrize("passes,expected", [(2.0, 2), ("3", 3), (True, 1)])
def test_a_whole_number_of_passes_is_read_as_an_int(passes, expected):
    profile = MachineProfile.from_dict({"name": "x", "implement_width_m": 9, "passes_per_strip": passes})
    assert profile.passes_per_strip == expected and isinstance(profile.passes_per_strip, int)


def test_api_answers_a_fractional_pass_with_a_sentence_not_a_422(routes):
    """The request model must let 1.5 through so the refusal is a 400 in
    plain words, like every other one, rather than pydantic's error list —
    which the page showed as '[object Object]'."""
    request = routes.ProfileRequest(name="Frac", implement_width_m=9, passes_per_strip=1.5)
    with pytest.raises(HTTPException) as info:
        routes.save_profile(request)
    assert info.value.status_code == 400
    assert "'passes_per_strip' must be a whole number, not 1.5" in info.value.detail
    assert routes.list_profiles()["profiles"] == []


def test_the_page_reads_out_a_validation_list_instead_of_object_object():
    """Belt and braces: should any route still answer a 422, the toast names
    the field and pydantic's message rather than stringifying the list."""
    api = _block("api")
    assert "Array.isArray(payload?.detail)" in api
    assert "e.msg" in api and "e.loc" in api


def test_the_dialog_shows_a_speed_minimum_of_zero():
    """A 0 typed in the speed filter was drawn as an empty box, sent as null
    and filled with the kind's default: 0 mph became 1.86 mph unannounced."""
    dialog = _block("renderMachineDialog")
    assert "value === 0" not in dialog
    assert 'shown("speed", p.speed_min_kmh)' in dialog
    # Only the width keeps 0 for "unknown": that is what a suggestion sends
    # when the file has no swath, and the placeholder is the better prompt.
    assert 'shown("length", p.implement_width_m || null)' in dialog


def test_typed_values_are_not_rounded_away_in_metric():
    """33 ft cut to 10.058 m came back as 32.999 ft while the toast said 33.
    Six metric decimals are invisible at the three the forms show."""
    metric = _block("machineMetric")
    assert "1e6" in metric
    for name in ("machineScope", "saveMachine"):
        assert re.search(r"Math\.round\(.*factor", _block(name)) is None, f"{name} still rounds"
    # The same arithmetic, in Python: what the page sends round-trips.
    for typed, factor in ((33, 0.3048), (35, 0.3048), (10, 1.609344), (4.5, 1.609344)):
        stored = round(typed * factor, 6)
        assert round(stored / factor, 3) == typed


# ==========================================================================
# The file on disk
# ==========================================================================

def test_a_home_that_cannot_be_written_is_refused_with_the_path(tmp_path, monkeypatch):
    """AGROSUITE_HOME pointing at a file, a read-only folder or a full disk
    used to surface as a bare 500 with a traceback."""
    not_a_dir = tmp_path / "notadir"
    not_a_dir.write_text("", encoding="utf-8")
    monkeypatch.setenv("AGROSUITE_HOME", str(not_a_dir))

    with pytest.raises(ValueError, match="could not be written.*AGROSUITE_HOME") as info:
        profiles_mod.save_profile(MachineProfile(name="x", implement_width_m=9))
    assert str(not_a_dir / "profiles.json") in str(info.value)


def test_api_tells_a_disk_problem_from_a_bad_profile(routes, tmp_path, monkeypatch):
    """What is wrong with the profile is a 400; what is wrong with the disk
    is a 500 — both with a sentence, neither with a trace."""
    not_a_dir = tmp_path / "notadir"
    not_a_dir.write_text("", encoding="utf-8")
    monkeypatch.setenv("AGROSUITE_HOME", str(not_a_dir))

    with pytest.raises(HTTPException) as info:
        routes.save_profile(routes.ProfileRequest(name="x", implement_width_m=9))
    assert info.value.status_code == 500
    assert "could not be written" in info.value.detail
    assert "profiles.json" in info.value.detail

    with pytest.raises(HTTPException) as info:
        routes.save_profile(routes.ProfileRequest(name="", implement_width_m=9))
    assert info.value.status_code == 400


def test_a_hand_written_zero_speed_minimum_is_kept(home):
    """0 is a real minimum; only a missing speed gets the kind's default."""
    home.mkdir(parents=True)
    (home / "profiles.json").write_text(json.dumps({"version": 1, "profiles": [
        {"name": "Zero", "kind": "combine", "implement_width_m": 9,
         "speed_min_kmh": 0, "speed_max_kmh": 10},
    ]}), encoding="utf-8")
    assert profiles_mod.get_profile("Zero").speed_min_kmh == 0.0


# ==========================================================================
# Suggesting for the machine the form is about
# ==========================================================================

def test_a_suggestion_for_another_kind_keeps_that_kind_and_drops_foreign_evidence():
    """On the trial-layout tab a yield map is selected but the profile is
    the seeder's: the combine's header width, speed and flow delay are
    facts about another machine and must not land in it."""
    dataset = synthetic_harvest()
    profile = profiles_mod.suggest_from_dataset(dataset, kind="seeder")

    assert profile.kind == "seeder"
    assert profile.flow_delay_s == 0.0
    assert profile.implement_width_m == 0.0, "the header width is not the drill's"
    assert (profile.speed_min_kmh, profile.speed_max_kmh) == profiles_mod.DEFAULT_SPEED_KMH["seeder"]
    assert "seeder" in profile.name.lower() and "combine" not in profile.name.lower()
    assert "written by a combine" in profile.notes
    assert "enter the implement width" in profile.notes
    assert any("width" in p for p in profile.problems())


def test_a_suggestion_for_the_files_own_kind_is_unchanged():
    dataset = synthetic_harvest()
    assert profiles_mod.suggest_from_dataset(dataset, kind="combine") == \
        profiles_mod.suggest_from_dataset(dataset)
    # An unknown kind is ignored rather than refused: the file decides.
    assert profiles_mod.suggest_from_dataset(dataset, kind="hovercraft").kind == "combine"


def test_a_file_of_unknown_operation_lends_its_evidence_to_any_kind():
    dataset = synthetic_harvest()
    dataset.meta.operation = "unknown"
    profile = profiles_mod.suggest_from_dataset(dataset, kind="sprayer")
    assert profile.kind == "sprayer"
    assert profile.implement_width_m == pytest.approx(9.0, abs=0.1)
    assert "median swath" in profile.notes


def test_api_suggests_for_the_requested_kind(routes):
    from agrosuite.app import server as server_mod

    summary = server_mod._register(synthetic_harvest(), "Harvest (demo)", "demo")
    result = routes.suggest_profile(routes.SuggestRequest(dataset_id=summary["id"], kind="seeder"))
    assert result["profile"]["kind"] == "seeder"
    assert result["profile"]["flow_delay_s"] == 0.0
    assert result["problems"], "the width is left for the user to enter"


def test_the_dialog_sends_its_kind_and_keeps_the_passes():
    suggest = _block("suggestMachine")
    assert "body: { dataset_id: this.state.selectedId, kind }" in suggest
    assert 'this.value("mp-passes")' in suggest


# ==========================================================================
# A unit change keeps the picked machine
# ==========================================================================

def test_a_unit_change_refills_the_picked_machine():
    """The whole tab is redrawn on a unit change; the machine that was
    picked is remembered by name and filled in again, in the new units,
    instead of vanishing — the same memory that survives a tab switch."""
    bind = _block("bindMachinePicker")
    assert "this.machinePickedIn(scope)" in bind
    assert "this.applyMachine(scope, picked, { quiet: true })" in bind
    assert ".fill(profile)" in _block("applyMachine")
    # Both unit pickers — the preset and the ⚙ panel — just redraw the tab.
    assert "this.renderTab()" in _block("buildUnitPreset")
    assert "this.renderTab()" in _block("openUnitsDialog")
    assert "rerenderKeepingMachines" not in APP_JS
