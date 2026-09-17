"""Machine profiles.

What matters in practice: a profile entered once comes back exactly as
entered, saving it twice does not make two, the file on disk can be opened
in a text editor, a suggestion from a file is right and is *not* saved, and
a bad profile is refused with a message that says what to fix.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agrosuite.core import profiles as profiles_mod
from agrosuite.core import schema as sch
from agrosuite.core.dataset import Dataset
from agrosuite.core.profiles import MachineProfile
from agrosuite.demo import synthetic_harvest


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch) -> Path:
    """Every test gets an empty AGROSUITE_HOME, so none touches the real one."""
    target = tmp_path / "agrosuite_home"
    monkeypatch.setenv("AGROSUITE_HOME", str(target))
    return target


def _combine(**overrides) -> MachineProfile:
    base = dict(
        name="My combine", kind="combine", monitor="john_deere",
        implement_width_m=18.288, flow_delay_s=12.0, passes_per_strip=2,
        speed_min_kmh=3.0, speed_max_kmh=8.0, notes="60 ft draper",
    )
    base.update(overrides)
    return MachineProfile(**base)


# ==========================================================================
# Storage
# ==========================================================================

def test_home_honours_the_environment(home):
    assert profiles_mod.profiles_path() == home / "profiles.json"


def test_round_trip(home):
    """What goes in is what comes out — every field, metric, untouched."""
    assert profiles_mod.list_profiles() == []

    profiles_mod.save_profile(_combine())
    listed = profiles_mod.list_profiles()
    assert [p.name for p in listed] == ["My combine"]
    assert listed[0] == _combine()
    assert profiles_mod.get_profile("My combine") == _combine()
    assert (home / "profiles.json").exists()


def test_upsert_replaces_instead_of_duplicating():
    profiles_mod.save_profile(_combine())
    profiles_mod.save_profile(_combine(implement_width_m=12.192, notes="40 ft"))

    listed = profiles_mod.list_profiles()
    assert len(listed) == 1
    assert listed[0].implement_width_m == pytest.approx(12.192)
    assert listed[0].notes == "40 ft"


def test_names_differing_only_in_case_are_the_same_profile():
    """'My Combine' and 'my combine' typed on different days are one machine."""
    profiles_mod.save_profile(_combine(name="My Combine"))
    profiles_mod.save_profile(_combine(name="my  combine", flow_delay_s=10.0))

    listed = profiles_mod.list_profiles()
    assert len(listed) == 1
    assert listed[0].flow_delay_s == 10.0
    assert profiles_mod.get_profile("MY COMBINE").name == "my  combine"


def test_list_is_sorted_by_name():
    for name in ("Sprayer", "air drill", "Combine"):
        profiles_mod.save_profile(_combine(name=name))
    assert [p.name for p in profiles_mod.list_profiles()] == ["air drill", "Combine", "Sprayer"]


def test_delete():
    profiles_mod.save_profile(_combine())
    profiles_mod.save_profile(_combine(name="Sprayer", kind="sprayer", flow_delay_s=0))

    remaining = profiles_mod.delete_profile("My combine")
    assert [p.name for p in remaining] == ["Sprayer"]
    assert [p.name for p in profiles_mod.list_profiles()] == ["Sprayer"]

    with pytest.raises(KeyError, match="No profile named 'My combine'.*Sprayer"):
        profiles_mod.delete_profile("My combine")
    with pytest.raises(KeyError, match="No profile named"):
        profiles_mod.get_profile("My combine")


def test_file_on_disk_is_plain_readable_json(home):
    """The file has to survive a text editor and a copy to another computer."""
    profiles_mod.save_profile(_combine())

    text = (home / "profiles.json").read_text(encoding="utf-8")
    assert "\n  " in text, "indented, so it can be read and edited by hand"
    data = json.loads(text)
    assert isinstance(data["profiles"], list)
    record = data["profiles"][0]
    assert record["name"] == "My combine"
    assert record["implement_width_m"] == pytest.approx(18.288), "stored metric, never in feet"
    assert record["flow_delay_s"] == 12.0
    assert record["monitor"] == "john_deere"
    assert not list(home.glob("*.tmp")), "the temporary file is renamed away, not left behind"


def test_hand_edited_file_is_read_back(home):
    """A profile typed straight into the file, with a key from a newer version."""
    home.mkdir(parents=True)
    (home / "profiles.json").write_text(json.dumps({
        "version": 1,
        "profiles": [{
            "name": "Typed by hand", "kind": "seeder", "implement_width_m": "15.24",
            "speed_min_kmh": 5, "speed_max_kmh": 12, "future_key": True,
        }],
    }), encoding="utf-8")

    profile = profiles_mod.get_profile("Typed by hand")
    assert profile.implement_width_m == pytest.approx(15.24)
    assert profile.kind == "seeder"
    assert profile.monitor == "generic"


def test_a_corrupt_file_is_refused_not_overwritten(home):
    """Reading nothing and then saving would wipe the user's profiles."""
    home.mkdir(parents=True)
    path = home / "profiles.json"
    path.write_text("{ this is not json", encoding="utf-8")

    with pytest.raises(ValueError, match="could not be read.*profiles.json"):
        profiles_mod.list_profiles()
    with pytest.raises(ValueError, match="Fix or move it away"):
        profiles_mod.save_profile(_combine())
    assert path.read_text(encoding="utf-8") == "{ this is not json"


# ==========================================================================
# Validation
# ==========================================================================

@pytest.mark.parametrize("overrides,message", [
    ({"name": "   "}, "Give the profile a name"),
    ({"implement_width_m": 0}, "width must be greater than zero"),
    ({"implement_width_m": -3}, "width must be greater than zero"),
    ({"speed_min_kmh": 8, "speed_max_kmh": 8}, "minimum speed must be below the maximum"),
    ({"speed_min_kmh": 9, "speed_max_kmh": 8}, "minimum speed must be below the maximum"),
    ({"speed_min_kmh": -1}, "minimum speed cannot be negative"),
    ({"kind": "tractor"}, "Unknown machine kind 'tractor'"),
    ({"monitor": "acme"}, "Unknown monitor 'acme'"),
    ({"flow_delay_s": -2}, "flow delay cannot be negative"),
    ({"passes_per_strip": 0}, "Passes per strip must be at least 1"),
])
def test_validation_refuses_with_a_message(overrides, message):
    with pytest.raises(ValueError, match=message):
        profiles_mod.save_profile(_combine(**overrides))
    assert profiles_mod.list_profiles() == [], "a refused profile leaves no trace"


def test_validation_reports_every_problem_at_once():
    problems = _combine(name="", implement_width_m=0, speed_min_kmh=10, speed_max_kmh=5).problems()
    assert len(problems) == 3


def test_non_numeric_values_name_the_field():
    with pytest.raises(ValueError, match="'implement_width_m' must be a number"):
        MachineProfile.from_dict({"name": "x", "implement_width_m": "sixty"})
    with pytest.raises(ValueError, match="has no name"):
        MachineProfile.from_dict({"implement_width_m": 9})


# ==========================================================================
# Suggestion from a file
# ==========================================================================

def test_suggestion_from_the_demo_harvest():
    """The demo is a 9 m combine at about 5.6 km/h with a 12 s flow delay."""
    dataset = synthetic_harvest()
    before = profiles_mod.list_profiles()

    profile = profiles_mod.suggest_from_dataset(dataset)

    assert profile.kind == "combine"
    assert profile.implement_width_m == pytest.approx(9.0, abs=0.1)
    assert profile.flow_delay_s == 12.0
    assert profile.monitor == "generic"
    assert "9 m" in profile.name
    # The range brackets the working speed but leaves out the headland ramps.
    assert 2.0 <= profile.speed_min_kmh <= 3.0
    assert 5.5 <= profile.speed_max_kmh <= 6.5
    assert profile.speed_min_kmh < profile.speed_max_kmh
    assert profile.speed_min_kmh % 0.5 == 0 and profile.speed_max_kmh % 0.5 == 0
    assert profile.problems() == [], "a suggestion from a complete file is ready to save"
    assert "median swath" in profile.notes

    assert profiles_mod.list_profiles() == before, "suggesting must not save"


def test_suggestion_names_the_brand():
    dataset = synthetic_harvest()
    dataset.meta.brand = "john_deere"
    profile = profiles_mod.suggest_from_dataset(dataset)
    assert profile.monitor == "john_deere"
    assert profile.name.startswith("John Deere")


def test_suggestion_without_a_swath_asks_for_the_width():
    """Inventing a width would be worse than leaving the box empty."""
    source = synthetic_harvest()
    dataset = Dataset(source.df.drop(columns=[sch.SWATH]), source.meta)

    profile = profiles_mod.suggest_from_dataset(dataset)
    assert profile.implement_width_m == 0.0
    assert any("width" in p for p in profile.problems())
    assert "enter the implement width" in profile.notes


def test_suggestion_for_an_application_has_no_flow_delay():
    dataset = synthetic_harvest()
    dataset.meta.operation = "application"
    profile = profiles_mod.suggest_from_dataset(dataset)
    assert profile.kind == "sprayer"
    assert profile.flow_delay_s == 0.0


# ==========================================================================
# The API
# ==========================================================================

@pytest.fixture
def api():
    from agrosuite.app import server as server_mod
    from agrosuite.app.routes import profiles as routes

    return server_mod, routes


def test_router_is_included_in_the_app(api):
    """The feature is a file in routes/; the server has to pick it up on its own."""
    server_mod, _ = api
    paths = server_mod.app.openapi()["paths"]
    assert set(paths["/api/profiles"]) == {"get", "post"}
    assert set(paths["/api/profiles/suggest"]) == {"post"}
    assert set(paths["/api/profiles/{name}"]) == {"delete"}


def test_api_round_trip(api, home):
    _, routes = api

    listing = routes.list_profiles()
    assert listing["profiles"] == []
    assert ["combine", "seeder", "sprayer", "spreader", "other"] == [k for k, _ in listing["kinds"]]
    assert listing["storage"] == str(home / "profiles.json")

    saved = routes.save_profile(routes.ProfileRequest(
        name="My combine", kind="combine", monitor="john_deere",
        implement_width_m=18.288, flow_delay_s=12, speed_min_kmh=3, speed_max_kmh=8,
    ))
    assert [p["name"] for p in saved["profiles"]] == ["My combine"]
    record = saved["profiles"][0]
    assert record["kind_label"] == "Combine"
    assert record["monitor_label"] == "John Deere"
    assert record["implement_width_m"] == pytest.approx(18.288)

    # The same machine, spelt differently: replaced only once that is agreed.
    again = dict(name="my combine", implement_width_m=12.192, speed_min_kmh=3, speed_max_kmh=8)
    with pytest.raises(HTTPException) as info:
        routes.save_profile(routes.ProfileRequest(**again))
    assert info.value.status_code == 409
    assert "'My combine' already exists" in info.value.detail
    assert routes.list_profiles()["profiles"][0]["implement_width_m"] == pytest.approx(18.288)

    replaced = routes.save_profile(routes.ProfileRequest(**again, replace=True))
    assert len(replaced["profiles"]) == 1
    assert replaced["profiles"][0]["implement_width_m"] == pytest.approx(12.192)

    remaining = routes.delete_profile("my combine")
    assert remaining["profiles"] == []


def test_api_fills_the_speed_range_from_the_kind(api):
    _, routes = api
    saved = routes.save_profile(routes.ProfileRequest(
        name="Sprayer", kind="sprayer", implement_width_m=36.576,
    ))
    record = saved["profiles"][0]
    assert (record["speed_min_kmh"], record["speed_max_kmh"]) == profiles_mod.DEFAULT_SPEED_KMH["sprayer"]


@pytest.mark.parametrize("request_kwargs,message", [
    ({"name": "", "implement_width_m": 9}, "Give the profile a name"),
    ({"name": "x", "implement_width_m": 0}, "width must be greater than zero"),
    ({"name": "x", "implement_width_m": 9, "speed_min_kmh": 9, "speed_max_kmh": 4},
     "minimum speed must be below the maximum"),
    ({"name": "x", "implement_width_m": 9, "kind": "drone"}, "Unknown machine kind"),
])
def test_api_refuses_a_bad_profile_with_a_message(api, request_kwargs, message):
    _, routes = api
    with pytest.raises(HTTPException) as info:
        routes.save_profile(routes.ProfileRequest(**request_kwargs))
    assert info.value.status_code == 400
    assert message in info.value.detail
    assert routes.list_profiles()["profiles"] == []


def test_api_delete_of_a_missing_profile_is_404(api):
    _, routes = api
    with pytest.raises(HTTPException) as info:
        routes.delete_profile("nobody")
    assert info.value.status_code == 404
    assert "No profile named 'nobody'" in info.value.detail


def test_api_suggests_from_a_loaded_dataset(api):
    server_mod, routes = api
    summary = server_mod._register(synthetic_harvest(), "Harvest (demo)", "demo")

    result = routes.suggest_profile(routes.SuggestRequest(dataset_id=summary["id"]))
    profile = result["profile"]
    assert profile["kind"] == "combine"
    assert profile["implement_width_m"] == pytest.approx(9.0, abs=0.1)
    assert profile["flow_delay_s"] == 12.0
    assert result["problems"] == []
    assert result["dataset_id"] == summary["id"]
    assert result["dataset_label"] == "Harvest (demo)"
    assert routes.list_profiles()["profiles"] == [], "suggesting must not save"

    with pytest.raises(HTTPException) as info:
        routes.suggest_profile(routes.SuggestRequest(dataset_id="no_such_dataset"))
    assert info.value.status_code == 404
