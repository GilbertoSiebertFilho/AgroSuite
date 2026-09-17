"""John Deere card structures.

When SMS exports through the **GreenStar** option it does not write a loose
file: it writes the folder tree the display expects to find on the card or
stick. Recognizing that tree is what lets AgroSuite open what the user
already has in hand, instead of demanding they export everything again as
shapefile.

The three generations have different roots::

    GS2_2600/   GreenStar 2 (2600 display)
    GS3_2630/   GreenStar 3 (2630 display)
    JD-Data/    Gen 4 (4200, 4600, 4640 displays) and Gen 5

and inside them, the same division of roles::

    SETUP/      data going TO the display: clients, farms, fields,
                boundaries, guidance lines, prescriptions
    RCD/        data coming FROM the display: operation records

What this module does and does not do
-------------------------------------
The setup files SMS writes inside ``SETUP/`` are in John Deere's proprietary
format, and AgroSuite does **not** try to reconstruct them. What it does is
inventory the card, read everything in an open format — boundary, guidance
or prescription shapefiles, which is how a good share of SMS exports store
the geometry — and state plainly what it found and what it cannot interpret,
rather than failing silently or, worse, producing a file the display refuses
out in the field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Recognized card roots, newest to oldest.
CARD_ROOTS = {
    "JD-Data": "Gen 4 / Gen 5 (4200, 4600, 4640)",
    "GS3_2630": "GreenStar 3 (2630 display)",
    "GS3_2600": "GreenStar 3 (2600 display)",
    "GS2_2600": "GreenStar 2 (2600 display)",
    "GS2_1800": "GreenStar 2 (1800 display)",
}

#: Subfolders with a defined meaning inside the card root.
CARD_SUBFOLDERS = {
    "SETUP": "Data going to the display (fields, boundaries, lines, prescriptions)",
    "RCD": "Data recorded by the display during operation",
    "DOCUMENTATION": "Exported operation documentation",
    "BOUNDARIES": "Field boundaries",
    "GUIDANCE": "Guidance lines",
    "RX": "Variable rate prescriptions",
    "SHAPEFILES": "Shapefile layers",
}

#: Open-format extensions we can interpret from inside the card.
READABLE_EXT = {".shp", ".geojson", ".json", ".kml", ".kmz", ".csv", ".txt", ".xml"}

#: John Deere proprietary extensions. Listed so the inventory can say what
#: they are, instead of calling them "unknown file".
PROPRIETARY_EXT = {
    ".gsd": "GreenStar documentation (proprietary binary)",
    ".fdd": "Field Doc Data (proprietary binary)",
    ".fdl": "Field Doc Log (proprietary binary)",
    ".jdp": "John Deere data package",
    ".jdf": "John Deere setup file",
    ".ver": "Card version control",
    ".dat": "Display binary data",
    ".bin": "Display binary data",
}


@dataclass
class CardInventory:
    """What was found on a John Deere card."""

    root: Path
    card_root: Path | None = None
    generation: str = "desconhecida"
    readable: list[dict[str, Any]] = field(default_factory=list)
    proprietary: list[dict[str, Any]] = field(default_factory=list)
    folders: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "card_root": str(self.card_root) if self.card_root else None,
            "generation": self.generation,
            "folders": self.folders,
            "readable": self.readable,
            "proprietary": self.proprietary,
            "summary": self.summary(),
        }

    def summary(self) -> str:
        if not self.card_root:
            return "No John Deere card structure found."
        parts = [f"{self.generation} card."]
        if self.readable:
            parts.append(f"{len(self.readable)} file(s) in an open format, readable here.")
        if self.proprietary:
            parts.append(
                f"{len(self.proprietary)} file(s) in John Deere's proprietary format, "
                "which only their software or the display itself can convert."
            )
        if not self.readable and self.proprietary:
            parts.append(
                "To bring that content into AgroSuite, export again from SMS choosing "
                "shapefile instead of GreenStar."
            )
        return " ".join(parts)


def find_card_root(path: Path) -> tuple[Path | None, str]:
    """Locate the card root and say which generation it belongs to."""
    path = Path(path)
    candidates = [path, *[p for p in path.iterdir() if p.is_dir()]] if path.is_dir() else [path]

    for candidate in candidates:
        if candidate.name in CARD_ROOTS:
            return candidate, CARD_ROOTS[candidate.name]

    # The card may sit deeper, inside a backup folder.
    if path.is_dir():
        for name, label in CARD_ROOTS.items():
            for found in path.rglob(name):
                if found.is_dir():
                    return found, label

    # Structure without the named root folder, but with SETUP/RCD side by side.
    if path.is_dir():
        children = {p.name.upper() for p in path.iterdir() if p.is_dir()}
        if {"SETUP", "RCD"} & children:
            return path, "GreenStar (unnamed root)"

    return None, "unknown"


def inventory(path: Path) -> CardInventory:
    """Inventory a card: what can be read and what is proprietary."""
    path = Path(path)
    card_root, generation = find_card_root(path)
    result = CardInventory(root=path, card_root=card_root, generation=generation)
    if card_root is None:
        return result

    result.folders = sorted(
        p.name for p in card_root.iterdir() if p.is_dir()
    )

    for item in sorted(card_root.rglob("*")):
        if not item.is_file():
            continue
        suffix = item.suffix.lower()
        relative = str(item.relative_to(card_root))
        entry = {
            "path": str(item),
            "relative": relative,
            "name": item.name,
            "size": item.stat().st_size,
            "folder_role": CARD_SUBFOLDERS.get(item.parent.name.upper(), ""),
        }
        if suffix in READABLE_EXT:
            result.readable.append({**entry, "kind": suffix.lstrip(".")})
        elif suffix in PROPRIETARY_EXT:
            result.proprietary.append({**entry, "kind": PROPRIETARY_EXT[suffix]})

    return result


def readable_layers(inv: CardInventory) -> list[dict[str, Any]]:
    """Card layers AgroSuite can import, already classified.

    Classifying by keyword in the path is what lets us say "this is the
    boundary" without opening every file: SMS and Operations Center exports
    name their folders and files consistently.
    """
    layers = []
    for entry in inv.readable:
        if entry["kind"] not in ("shp", "geojson", "json", "kml", "kmz"):
            continue
        text = entry["relative"].lower()
        if any(k in text for k in ("boundary", "bound", "field_border", "contorno")):
            role = "boundary"
        elif any(k in text for k in ("guidance", "abline", "ab_line", "track", "swath")):
            role = "guidance"
        elif any(k in text for k in ("rx", "prescription", "target")):
            role = "prescription"
        else:
            role = "data"
        layers.append({**entry, "role": role})
    return layers


def describe_export_paths(generation_key: str = "gen4") -> dict[str, Any]:
    """Where each display generation looks for files on the stick.

    These paths describe the platform's card structure; the import menu
    changes between firmware versions, so the generated package always ships
    with instructions asking the operator to confirm on screen.
    """
    paths = {
        "gen4": {
            "label": "Gen 4 / Gen 5 (4200, 4600, 4640)",
            "card_root": "JD-Data",
            "notes": (
                "A shapefile prescription is imported by the display itself, through "
                "the USB file manager. Boundaries and guidance lines arrive more "
                "reliably through Operations Center, synced to the machine."
            ),
        },
        "gs3": {
            "label": "GreenStar 3 (2630)",
            "card_root": "GS3_2630",
            "notes": (
                "The card separates SETUP (what goes to the display) from RCD (what "
                "the display recorded). The setup files are proprietary and come out "
                "of SMS or Apex, not out of AgroSuite."
            ),
        },
        "gs2": {
            "label": "GreenStar 2 (2600)",
            "card_root": "GS2_2600",
            "notes": "Same SETUP/RCD split as GreenStar 3.",
        },
    }
    return paths.get(generation_key, paths["gen4"])
