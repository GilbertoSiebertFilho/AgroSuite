"""Export packages, one per monitor.

Producing the right file is half the job; the other half is handing it over
in the folder structure the terminal looks for. This module describes, for
each platform, what it accepts and how the USB stick should end up, and
assembles the folder ready to copy.

Two conventions are standardized and hold for any ISOBUS terminal:

* the folder is named ``TASKDATA`` and sits at the **root** of the stick;
* a shapefile only opens with ``.shp``, ``.shx``, ``.dbf`` and ``.prj``
  together.

Everything else varies by manufacturer and by firmware version, so each
package ships with an instruction file naming the import path and what to
confirm on screen — rather than the app pretending certainty about menus
that change with every update.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Artifacts a package may contain.
ARTIFACT_LABELS = {
    "prescription": "Prescription (variable rate)",
    "boundary": "Field boundary",
    "guidance": "Guidance lines (AB)",
    "data": "Field data (points)",
}


@dataclass(frozen=True)
class MonitorProfile:
    """How to prepare files for a specific monitor."""

    key: str
    label: str
    #: Artifacts this platform is able to receive.
    accepts: tuple[str, ...]
    #: Preferred format for each artifact.
    preferred: dict[str, str]
    #: Subfolder inside the package, per format.
    layout: dict[str, str] = field(default_factory=dict)
    #: Name of the rate field the monitor looks for in the shapefile.
    rate_field: str = "RATE"
    instructions: tuple[str, ...] = ()


ISOBUS_STEPS = (
    "Copy the whole TASKDATA folder to the ROOT of the USB stick — not inside "
    "another folder, and without renaming it.",
    "Use a stick formatted as FAT32. Many terminals will not read exFAT or NTFS.",
    "On the terminal, open Import / Data Manager and select the USB stick.",
    "Check on screen that the field, the guidance lines and the task showed up "
    "before heading out.",
)

SHAPEFILE_STEPS = (
    "Copy all four files together (.shp, .shx, .dbf, .prj). Miss any one of "
    "them and the monitor will not open the map.",
    "During import the monitor asks which column holds the rate: pick the "
    "column named in this package's README file.",
    "Confirm the rate unit on the monitor screen — a shapefile stores the "
    "number, not the unit.",
)


PROFILES: tuple[MonitorProfile, ...] = (
    MonitorProfile(
        key="raven",
        label="Raven Viper 4",
        accepts=("prescription", "boundary", "guidance"),
        preferred={"prescription": "shapefile", "boundary": "shapefile", "guidance": "isoxml"},
        layout={"shapefile": "Raven", "isoxml": "."},
        rate_field="RATE",
        instructions=SHAPEFILE_STEPS + (
            "The Viper 4 imports a prescription from File Manager -> USB, picking "
            "the polygon shapefile and then the rate column.",
            "A field boundary also goes in as a polygon shapefile.",
            "If your Viper 4 has ISOBUS enabled, the TASKDATA folder in this "
            "package carries the AB lines and the boundary in one go.",
        ),
    ),
    MonitorProfile(
        key="john_deere",
        label="John Deere (Gen 4 / Operations Center)",
        accepts=("prescription", "boundary", "guidance", "data"),
        preferred={"prescription": "shapefile", "boundary": "shapefile", "guidance": "isoxml"},
        layout={"shapefile": "Rx", "isoxml": "."},
        rate_field="RATE",
        instructions=SHAPEFILE_STEPS + (
            "Most reliable route: upload the shapefile to Operations Center "
            "(Files -> Upload) and send the map to the machine over Data Sync.",
            "With no connectivity, import the shapefile straight from the USB "
            "stick on the Gen 4 display, under File Manager.",
            "Older GreenStar displays (2600/2630) need the map to pass through "
            "desktop software first.",
        ),
    ),
    MonitorProfile(
        key="case_ih",
        label="Case IH AFS Pro / AFS Connect",
        accepts=("prescription", "boundary", "guidance"),
        preferred={"prescription": "isoxml", "boundary": "isoxml", "guidance": "isoxml"},
        layout={"isoxml": ".", "shapefile": "Shapefile"},
        instructions=ISOBUS_STEPS + (
            "AFS Pro 700 and AFS Connect read ISOXML natively — that is the "
            "preferred route, because it carries boundary, AB lines and "
            "prescription in a single set of files.",
            "The Shapefile folder in this package is the fallback, in case the "
            "terminal refuses the TASKDATA.",
        ),
    ),
    MonitorProfile(
        key="new_holland",
        label="New Holland IntelliView / PLM",
        accepts=("prescription", "boundary", "guidance"),
        preferred={"prescription": "isoxml", "boundary": "isoxml", "guidance": "isoxml"},
        layout={"isoxml": ".", "shapefile": "Shapefile"},
        instructions=ISOBUS_STEPS + (
            "IntelliView IV and XCN are ISOBUS, so TASKDATA is the direct route.",
        ),
    ),
    MonitorProfile(
        key="bourgault",
        label="Bourgault X30 / X35",
        accepts=("prescription", "boundary", "guidance"),
        preferred={"prescription": "isoxml", "boundary": "isoxml", "guidance": "isoxml"},
        layout={"isoxml": ".", "shapefile": "Shapefile"},
        instructions=ISOBUS_STEPS + (
            "On the X35 the prescription shows up inside the imported task; map "
            "each tank to its product before starting.",
        ),
    ),
    MonitorProfile(
        key="vaderstad",
        label="Väderstad E-Control",
        accepts=("prescription", "boundary", "guidance"),
        preferred={"prescription": "isoxml", "boundary": "isoxml", "guidance": "isoxml"},
        layout={"isoxml": "."},
        instructions=ISOBUS_STEPS,
    ),
    MonitorProfile(
        key="topcon",
        label="Topcon / Müller",
        accepts=("prescription", "boundary", "guidance"),
        preferred={"prescription": "isoxml", "boundary": "isoxml", "guidance": "isoxml"},
        layout={"isoxml": "."},
        instructions=ISOBUS_STEPS,
    ),
    MonitorProfile(
        key="trimble",
        label="Trimble GFX / TMX / FmX",
        accepts=("prescription", "boundary", "guidance"),
        preferred={"prescription": "shapefile", "boundary": "shapefile", "guidance": "isoxml"},
        layout={"shapefile": "AgData", "isoxml": "."},
        rate_field="TGT_RATE",
        instructions=SHAPEFILE_STEPS + (
            "On GFX/TMX displays, import through the data manager, choosing the "
            "shapefile and then the TGT_RATE column.",
            "Models with ISOBUS enabled also accept the TASKDATA folder.",
        ),
    ),
    MonitorProfile(
        key="ag_leader",
        label="Ag Leader InCommand / SMS",
        accepts=("prescription", "boundary", "data"),
        preferred={"prescription": "shapefile", "boundary": "shapefile"},
        layout={"shapefile": "AgLeader"},
        instructions=SHAPEFILE_STEPS + (
            "On InCommand, import under Setup -> Field -> Prescription.",
        ),
    ),
    MonitorProfile(
        key="augmenta",
        label="Augmenta",
        accepts=("prescription", "boundary", "data"),
        preferred={"prescription": "geojson", "boundary": "geojson"},
        layout={"geojson": "Augmenta", "shapefile": "Shapefile"},
        instructions=(
            "Augmenta works with GeoJSON and shapefile: use the GeoJSON when "
            "uploading through the web console, and the shapefile over USB.",
            "The system decides the rate in real time from the camera; the "
            "prescription comes in as a cap or as a base map, depending on how "
            "the machine is configured.",
        ),
    ),
    MonitorProfile(
        key="generic",
        label="Generic (any monitor)",
        accepts=("prescription", "boundary", "guidance", "data"),
        preferred={"prescription": "shapefile", "boundary": "shapefile", "guidance": "isoxml"},
        layout={},
        instructions=SHAPEFILE_STEPS + ISOBUS_STEPS,
    ),
)

PROFILES_BY_KEY = {p.key: p for p in PROFILES}


def get_profile(key: str | None) -> MonitorProfile:
    """Export profile by key, falling back to the generic one."""
    return PROFILES_BY_KEY.get(key or "", PROFILES_BY_KEY["generic"])


def profile_catalog() -> list[dict[str, Any]]:
    """Serializable catalogue for the interface."""
    return [
        {
            "key": p.key,
            "label": p.label,
            "accepts": list(p.accepts),
            "preferred": dict(p.preferred),
            "rate_field": p.rate_field,
            "instructions": list(p.instructions),
        }
        for p in PROFILES
    ]


def write_readme(
    folder: Path,
    profile: MonitorProfile,
    contents: list[dict[str, Any]],
    rate_field: str | None = None,
    rate_unit: str | None = None,
) -> Path:
    """Write the README that ships inside the package."""
    lines = [
        f"PACKAGE FOR {profile.label.upper()}",
        "=" * (12 + len(profile.label)),
        "",
        "Generated by AgroSuite.",
        "",
        "CONTENTS",
        "--------",
    ]
    for item in contents:
        detail = item.get("detail", "")
        lines.append(f"  {item['path']}")
        lines.append(f"      {ARTIFACT_LABELS.get(item['artifact'], item['artifact'])}"
                     + (f" - {detail}" if detail else ""))
    lines.append("")

    if rate_field:
        lines += [
            "RATE COLUMN",
            "-----------",
            f"  In the shapefile, the rate lives in field: {rate_field}",
            f"  Unit written: {rate_unit or 'as chosen at export time'}",
            "  A shapefile stores only the number - confirm the unit on the monitor.",
            "",
        ]

    lines += ["HOW TO LOAD IT", "--------------"]
    for index, step in enumerate(profile.instructions, start=1):
        lines.append(f"  {index}. {step}")
    lines += [
        "",
        "BEFORE HEADING OUT",
        "------------------",
        "  - Confirm the right field is selected on the monitor.",
        "  - Check the rate shown over a part of the field you know well.",
        "  - Make sure the product and the unit match what is in the tank.",
        "",
    ]

    path = folder / "README.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
