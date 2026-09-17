"""Machine profiles: the numbers about a machine that never change between jobs.

A combine's header width, its flow delay and its working speed are properties
of the machine, not of the field, yet the app asked for them on every
cleaning and every trial layout. Typing "60 ft, 12 s" for the third time this
season is how a 6 sneaks in for a 60. A profile is entered once, checked once
and reused.

Profiles outlive the session, so they live on disk rather than in
``server.state``: ``$AGROSUITE_HOME/profiles.json``, defaulting to
``~/.agrosuite/profiles.json``. The file is plain, indented JSON on purpose —
it can be read, edited or copied to another computer with nothing but a text
editor.

Every physical value is stored **metric** (m, s, km/h), like everything else
in the app. The interface converts at display and at entry, so switching the
unit set never changes what is on disk.
"""

from __future__ import annotations

import json
import math
import os
import threading
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Collection

import numpy as np
import pandas as pd

from . import schema as sch
from ..formats import brands as brands_mod

#: What the machine does. The kind decides the defaults that cannot be read
#: from a file: a combine has a flow delay, a sprayer does not.
KINDS: dict[str, str] = {
    "combine": "Combine",
    "seeder": "Seeder / planter",
    "sprayer": "Sprayer",
    "spreader": "Spreader",
    "other": "Other",
}

#: Which kind of machine produced each operation the app recognizes.
OPERATION_KIND: dict[str, str] = {
    "harvest": "combine",
    "planting": "seeder",
    "application": "sprayer",
}

#: Typical working speed per kind, in km/h, for when the file carries no
#: speed. Wide on purpose: a range that is too tight silently throws away
#: good records in the speed filter, which is worse than one that is loose.
DEFAULT_SPEED_KMH: dict[str, tuple[float, float]] = {
    "combine": (3.0, 10.0),
    "seeder": (5.0, 12.0),
    "sprayer": (8.0, 25.0),
    "spreader": (8.0, 25.0),
    "other": (2.0, 20.0),
}

#: Seconds between the cut and the yield sensor on a combine. The same
#: default the harvest cleaning preset uses, so a suggested profile and the
#: preset never disagree.
COMBINE_FLOW_DELAY_S = 12.0

#: Below this many speed records the percentiles say more about noise than
#: about the machine, and the kind's typical range is the safer suggestion.
MIN_SPEED_RECORDS = 20

#: Suggested speeds are rounded to this step, in km/h. A range of
#: "2.513–5.841" reads like a measurement; "2.5–6.0" reads like a setting.
SPEED_STEP_KMH = 0.5

_FILE_VERSION = 1
_lock = threading.Lock()


# ==========================================================================
# The profile
# ==========================================================================

@dataclass
class MachineProfile:
    """One machine, as the operator knows it. All physical values metric."""

    name: str
    kind: str = "combine"
    #: A brand key from :mod:`agrosuite.formats.brands`, or ``generic``.
    monitor: str = "generic"
    implement_width_m: float = 0.0
    flow_delay_s: float = 0.0
    passes_per_strip: int = 2
    speed_min_kmh: float = DEFAULT_SPEED_KMH["other"][0]
    speed_max_kmh: float = DEFAULT_SPEED_KMH["other"][1]
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["kind_label"] = KINDS.get(self.kind, self.kind)
        data["monitor_label"] = brands_mod.get_brand(self.monitor).label
        # The file is edited by hand, so a listed profile is not always one
        # the app would have saved: the interface marks these and refuses to
        # pour an inverted speed range into the cleaning.
        data["problems"] = self.problems()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MachineProfile":
        """Build a profile from a plain dict, ignoring keys it does not know.

        Unknown keys are dropped rather than refused so that a file written by
        a newer version of the app still opens in an older one.
        """
        if not isinstance(data, dict):
            raise ValueError("A profile must be an object with at least a 'name'.")
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in known}
        if "name" not in kwargs:
            raise ValueError("The profile has no name. Give it one, e.g. 'My combine'.")
        try:
            profile = cls(**kwargs)
        except TypeError as exc:
            raise ValueError(f"The profile could not be read: {exc}") from exc
        profile._coerce()
        return profile

    def _coerce(self) -> None:
        """Force the numeric fields to numbers, with a message naming the field."""
        self.name = str(self.name or "").strip()
        self.kind = str(self.kind or "").strip().lower()
        self.monitor = str(self.monitor or "generic").strip().lower() or "generic"
        self.notes = str(self.notes or "")
        for name in ("implement_width_m", "flow_delay_s", "speed_min_kmh", "speed_max_kmh"):
            value = getattr(self, name)
            try:
                number = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"'{name}' must be a number, not {value!r}.")
            if not math.isfinite(number):
                raise ValueError(f"'{name}' must be a finite number.")
            setattr(self, name, number)
        # int() would quietly turn 1.5 passes into 1; a fraction is refused
        # instead, so what is stored is always what was meant.
        try:
            passes = float(self.passes_per_strip)
        except (TypeError, ValueError):
            passes = math.nan
        if not passes.is_integer():
            raise ValueError(
                f"'passes_per_strip' must be a whole number, not {self.passes_per_strip!r}."
            )
        self.passes_per_strip = int(passes)

    def problems(self, *, defaulted: Collection[str] = ()) -> list[str]:
        """Everything that stops this profile from being saved, in plain words.

        ``defaulted`` names the speed fields the caller filled with the kind's
        typical range because they were left empty: a refusal then says so
        instead of quoting a number nobody typed.
        """
        found: list[str] = []
        if not self.name:
            found.append("Give the profile a name, e.g. 'My combine'.")
        elif self.name in (".", ".."):
            # The interface addresses a profile by its name in a web address,
            # where '.' and '..' mean "here" and "up one level": a browser
            # folds them away before the request is sent, and the profile
            # could be saved but never deleted from the interface.
            found.append(
                f"'{self.name}' cannot be a name: a web address reads it as a folder. "
                "Use a real name, e.g. 'My combine'."
            )
        if self.kind not in KINDS:
            found.append(
                f"Unknown machine kind '{self.kind}'. "
                f"Choose one of: {', '.join(KINDS)}."
            )
        if self.monitor not in brands_mod.BRANDS_BY_KEY:
            found.append(
                f"Unknown monitor '{self.monitor}'. "
                f"Choose one of: {', '.join(brands_mod.BRANDS_BY_KEY)}."
            )
        if self.implement_width_m <= 0:
            found.append("The implement width must be greater than zero.")
        if self.flow_delay_s < 0:
            found.append("The flow delay cannot be negative. Use 0 for no delay.")
        if self.passes_per_strip < 1:
            found.append("Passes per strip must be at least 1.")
        if self.speed_min_kmh < 0:
            found.append("The minimum speed cannot be negative.")
        if self.speed_min_kmh >= self.speed_max_kmh:
            kind = KINDS.get(self.kind, self.kind).split(" /")[0].lower()
            low = f"{self.speed_min_kmh:g} km/h"
            high = f"{self.speed_max_kmh:g} km/h"
            if "speed_min_kmh" in defaulted:
                low = f"the typical minimum for a {kind}, {low}"
            if "speed_max_kmh" in defaulted:
                high = f"the typical maximum for a {kind}, {high}"
            found.append(
                f"The minimum speed must be below the maximum speed (got {low} and {high})."
            )
        return found

    def validate(self, *, defaulted: Collection[str] = ()) -> "MachineProfile":
        """Raise ``ValueError`` listing every problem at once, or return self.

        All problems come back together: fixing them one round trip at a time
        is the kind of friction profiles exist to remove.
        """
        self._coerce()
        found = self.problems(defaulted=defaulted)
        if found:
            raise ValueError(" ".join(found))
        return self


def _key(name: str) -> str:
    """Two names that differ only in case or spacing are the same profile."""
    return " ".join(str(name or "").split()).casefold()


class ProfileExists(Exception):
    """A profile is already saved under this name and replacing was not agreed.

    Carries the profile as it is saved, so the question the interface asks
    names the machine the way the list shows it, not the way it was just
    typed.
    """

    def __init__(self, existing: MachineProfile):
        super().__init__(
            f"A machine named '{existing.name}' already exists. "
            "Choose another name, or confirm replacing it."
        )
        self.existing = existing


# ==========================================================================
# Storage
# ==========================================================================

def home_dir() -> Path:
    """Where AgroSuite keeps what outlives a session.

    ``AGROSUITE_HOME`` overrides the default so that tests, and anyone who
    keeps their settings on a shared drive, can point it elsewhere.
    """
    override = os.environ.get("AGROSUITE_HOME", "").strip()
    return Path(override).expanduser() if override else Path.home() / ".agrosuite"


def profiles_path() -> Path:
    return home_dir() / "profiles.json"


def _read_all() -> list[MachineProfile]:
    path = profiles_path()
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        # Refusing to read beats reading nothing and then writing an empty
        # list over the user's profiles at the next save.
        raise ValueError(
            f"The profiles file could not be read ({exc}). Fix or move it away: {path}"
        ) from exc
    # A file that is an object but has lost its 'profiles' key is not empty,
    # it is broken: read as empty, the next save would write over whatever
    # it does hold. A bare list is still taken as the list of profiles.
    if isinstance(raw, dict) and "profiles" not in raw:
        keys = ", ".join(f"'{k}'" for k in raw) or "no keys at all"
        raise ValueError(
            f"The profiles file has no 'profiles' list (it holds {keys}). "
            f"Fix or move it away: {path}"
        )
    items = raw["profiles"] if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        raise ValueError(
            f"The profiles file does not hold a list of profiles. Fix or move it away: {path}"
        )
    profiles = []
    for item in items:
        try:
            profile = MachineProfile.from_dict(item)
        except ValueError as exc:
            raise ValueError(f"A profile in {path} could not be read: {exc}") from exc
        # Any other problem is reported under the profile's name; a profile
        # without one could be neither picked nor deleted, so the file is.
        if not profile.name:
            raise ValueError(
                f"A profile in {path} has an empty name, so it could not be listed "
                f"or deleted. Give it one, or move the file away: {path}"
            )
        profiles.append(profile)
    return profiles


def _write_all(profiles: list[MachineProfile]) -> None:
    path = profiles_path()
    payload = {
        "version": _FILE_VERSION,
        "profiles": [asdict(p) for p in sorted(profiles, key=lambda p: _key(p.name))],
    }
    # Written beside the file and renamed into place, so a crash mid-write
    # leaves the previous file intact rather than a truncated one.
    tmp = path.with_suffix(".json.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    except OSError as exc:
        # A read-only home, a full disk or an AGROSUITE_HOME that points at a
        # file all end here; like the read side, the message names the path
        # so it can be fixed rather than leaving a bare traceback.
        raise ValueError(
            f"The profiles file could not be written ({exc}). "
            f"Make the folder writable, or point AGROSUITE_HOME at one that is: {path}"
        ) from exc


def list_profiles() -> list[MachineProfile]:
    """Every saved profile, sorted by name."""
    with _lock:
        return sorted(_read_all(), key=lambda p: _key(p.name))


def get_profile(name: str) -> MachineProfile:
    """The profile with this name, or ``KeyError`` naming what exists."""
    for profile in list_profiles():
        if _key(profile.name) == _key(name):
            return profile
    raise KeyError(_missing(name))


def save_profile(profile: MachineProfile, *, replace: bool = True) -> list[MachineProfile]:
    """Save the profile and return the list.

    A profile already saved under the same name is replaced — that is how a
    machine gets corrected — unless ``replace`` is False, when
    :class:`ProfileExists` is raised so the caller can ask first. The check
    and the write share one lock: two windows saving the same name at the
    same moment cannot both find it free.
    """
    profile.validate()
    with _lock:
        profiles = _read_all()
        same = [p for p in profiles if _key(p.name) == _key(profile.name)]
        if same and not replace:
            raise ProfileExists(same[0])
        profiles = [p for p in profiles if _key(p.name) != _key(profile.name)]
        profiles.append(profile)
        _write_all(profiles)
        return sorted(profiles, key=lambda p: _key(p.name))


def delete_profile(name: str) -> list[MachineProfile]:
    """Remove the profile with this name; ``KeyError`` if there is none."""
    with _lock:
        profiles = _read_all()
        kept = [p for p in profiles if _key(p.name) != _key(name)]
        if len(kept) == len(profiles):
            raise KeyError(_missing(name))
        _write_all(kept)
        return sorted(kept, key=lambda p: _key(p.name))


def _missing(name: str) -> str:
    names = [p.name for p in _read_all()]
    return (
        f"No profile named '{name}'. "
        + (f"Saved profiles: {', '.join(names)}." if names else "No profile has been saved yet.")
    )


# ==========================================================================
# Suggesting a profile from a file
# ==========================================================================

def suggest_from_dataset(dataset, kind: str | None = None) -> MachineProfile:
    """A profile prefilled from what the file shows, for the user to confirm.

    Nothing here is saved. The file is evidence, not truth: a yield map made
    with a 30-foot header at 4 mph says what the machine was, but it is the
    operator who knows whether that is *their* combine.

    ``kind`` is the kind of machine the profile is for, when the caller has
    already decided it; left out, it follows the file's operation.
    """
    meta = dataset.meta
    df = dataset.df
    file_kind = OPERATION_KIND.get(meta.operation, "other")
    kind = kind if kind in KINDS else file_kind
    # A yield map on the trial-layout tab describes the combine that cut the
    # field, not the drill that will seed the trial: its swath and speed are
    # facts about another machine, and a seeder "30 ft wide" because the
    # header was is exactly the wrong number profiles exist to prevent.
    foreign = file_kind != "other" and kind != file_kind
    monitor = meta.brand if meta.brand in brands_mod.BRANDS_BY_KEY else "generic"
    evidence: list[str] = []

    # Width: the median swath, not the mean — a few partial-swath records at
    # the field edge would drag a mean below the real header width.
    width = 0.0
    swath = None if foreign else _positive(df.get(sch.SWATH))
    if swath is not None:
        width = round(float(swath.median()), 2)
        evidence.append(f"median swath {width:g} m")
    elif foreign:
        evidence.append(
            f"the file was written by a {KINDS[file_kind].lower()}, whose swath and speed "
            f"say nothing about a {KINDS[kind].lower()} — enter the implement width"
        )
    else:
        evidence.append("no swath width in the file — enter the implement width")

    # Speed: the 2nd and 98th percentiles leave out the headland turns and the
    # stray GPS jumps that the minimum and maximum would faithfully report.
    speed_min, speed_max = DEFAULT_SPEED_KMH[kind]
    speed = None if foreign else _positive(df.get(sch.SPEED))
    if speed is not None and len(speed) >= MIN_SPEED_RECORDS:
        low, high = np.percentile(speed.to_numpy(dtype="float64"), [2, 98])
        speed_min = _round_down(float(low))
        speed_max = _round_up(float(high))
        if speed_max <= speed_min:
            speed_max = speed_min + SPEED_STEP_KMH
        evidence.append(f"speed {speed_min:g}–{speed_max:g} km/h (2nd–98th percentile)")
    elif not foreign:
        evidence.append(f"speed range is the typical one for a {KINDS[kind].lower()}")

    flow_delay = COMBINE_FLOW_DELAY_S if kind == "combine" else 0.0

    brand_label = _brand_label(meta)
    name_parts = [brand_label, KINDS[kind].split(" /")[0].lower() if brand_label else KINDS[kind]]
    if width > 0:
        name_parts.append(f"{width:g} m")
    name = " ".join(part for part in name_parts if part)

    return MachineProfile(
        name=name,
        kind=kind,
        monitor=monitor,
        implement_width_m=width,
        flow_delay_s=flow_delay,
        passes_per_strip=2,
        speed_min_kmh=speed_min,
        speed_max_kmh=speed_max,
        notes=f"Suggested from '{meta.name}': " + "; ".join(evidence) + ".",
    )


def _positive(series) -> pd.Series | None:
    """Finite, positive values of a column, or ``None`` when there are none."""
    if series is None:
        return None
    values = pd.to_numeric(series, errors="coerce")
    values = values[np.isfinite(values) & (values > 0)]
    return values if len(values) else None


def _round_down(value: float) -> float:
    return math.floor(value / SPEED_STEP_KMH) * SPEED_STEP_KMH


def _round_up(value: float) -> float:
    return math.ceil(value / SPEED_STEP_KMH) * SPEED_STEP_KMH


def _brand_label(meta) -> str:
    """The manufacturer's name for the suggested profile name, if there is one."""
    if meta.brand in brands_mod.BRANDS_BY_KEY and meta.brand != "generic":
        return brands_mod.get_brand(meta.brand).label
    label = (meta.brand_label or "").strip()
    # "Desconhecido" is DatasetMeta's placeholder for "no brand", not a brand.
    if label and label.lower() not in ("desconhecido", "unknown", "generic"):
        return label
    return ""
