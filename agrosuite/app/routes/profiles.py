"""Machine profiles over HTTP.

The profiles themselves live in :mod:`agrosuite.core.profiles`; this module
only maps them onto the API and turns refusals into messages the interface
can show. Everything the handlers return is in the app's internal metric
units — the interface converts, as it does for every other number.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from ...core import profiles as profiles_mod

router = APIRouter(prefix="/api/profiles", tags=["profiles"])


class ProfileRequest(BaseModel):
    """A profile as the interface sends it. Metric, like the stored one.

    Speeds are optional on purpose: left out, the profile gets the typical
    range for its kind of machine, which is a better start than an empty box.
    """

    name: str
    kind: str = "combine"
    monitor: str = "generic"
    implement_width_m: float = 0.0
    flow_delay_s: float = 0.0
    # Any number is let through so that 1.5 is refused by the profile itself,
    # in a sentence, rather than by pydantic's list of error objects.
    passes_per_strip: float = 2
    speed_min_kmh: float | None = None
    speed_max_kmh: float | None = None
    notes: str = ""
    #: A profile already saved under this name is replaced only when this is
    #: set; otherwise the save is refused with a 409 so the interface can ask.
    replace: bool = False


class SuggestRequest(BaseModel):
    dataset_id: str
    #: The kind the form already shows, so the suggestion is for that machine
    #: rather than for whichever one wrote the file.
    kind: str | None = None


def _listing() -> dict[str, Any]:
    """The full list, plus what the interface needs to draw the form."""
    from agrosuite.app import server as server_mod

    try:
        profiles = profiles_mod.list_profiles()
    except ValueError as exc:
        raise server_mod._fail(str(exc), 500)
    return {
        "profiles": [p.to_dict() for p in profiles],
        "kinds": [[key, label] for key, label in profiles_mod.KINDS.items()],
        "storage": str(profiles_mod.profiles_path()),
    }


@router.get("")
def list_profiles() -> dict[str, Any]:
    return _listing()


@router.post("")
def save_profile(request: ProfileRequest) -> dict[str, Any]:
    """Create the profile, or replace the one with this name once agreed.

    Like a project file, an existing name answers 409 until the request
    says ``replace``: a machine entered months ago is not overwritten by a
    name typed the same way twice.
    """
    from agrosuite.app import server as server_mod

    data = request.model_dump()
    replace = data.pop("replace")
    defaults = profiles_mod.DEFAULT_SPEED_KMH.get(
        str(data.get("kind") or "").strip().lower(), profiles_mod.DEFAULT_SPEED_KMH["other"]
    )
    # Which speeds the default filled: a refusal should say "the typical
    # maximum for a combine" rather than quote a number nobody typed.
    defaulted = {key for key in ("speed_min_kmh", "speed_max_kmh") if data[key] is None}
    if data["speed_min_kmh"] is None:
        data["speed_min_kmh"] = defaults[0]
    if data["speed_max_kmh"] is None:
        data["speed_max_kmh"] = defaults[1]

    # What is wrong with the profile is the user's to fix (400); trouble with
    # the file on disk is the machine's (500). Both arrive as ValueError.
    try:
        profile = profiles_mod.MachineProfile.from_dict(data).validate(defaulted=defaulted)
    except ValueError as exc:
        raise server_mod._fail(str(exc))
    try:
        profiles_mod.save_profile(profile, replace=replace)
    except profiles_mod.ProfileExists as exc:
        raise server_mod._fail(str(exc), 409)
    except ValueError as exc:
        raise server_mod._fail(str(exc), 500)
    return _listing()


@router.post("/suggest")
def suggest_profile(request: SuggestRequest) -> dict[str, Any]:
    """A profile prefilled from a loaded dataset. Nothing is saved."""
    from agrosuite.app import server as server_mod

    try:
        entry = server_mod.state.get(request.dataset_id)
    except KeyError as exc:
        # str() of a KeyError adds its own quotes around the message.
        raise server_mod._fail(str(exc.args[0] if exc.args else exc), 404)
    profile = profiles_mod.suggest_from_dataset(entry.dataset, kind=request.kind)
    return {
        "profile": profile.to_dict(),
        # Width may be 0 when the file has no swath column: the user must fill
        # it in, and saving as-is is refused with that message.
        "problems": profile.problems(),
        "dataset_id": entry.id,
        "dataset_label": entry.label,
    }


@router.delete("/{name:path}")
def delete_profile(name: str) -> dict[str, Any]:
    """Remove a profile by name, and return the list that remains."""
    from agrosuite.app import server as server_mod

    try:
        profiles_mod.delete_profile(name)
    except KeyError as exc:
        raise server_mod._fail(str(exc.args[0] if exc.args else exc), 404)
    except ValueError as exc:
        raise server_mod._fail(str(exc), 500)
    return _listing()
