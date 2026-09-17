"""Writing straight to the USB stick.

The last step of every job is the same: copy a folder to a stick and carry it
to the machine. Doing it by hand is where the mistakes happen — the TASKDATA
ends up one folder deep, or the old prescription is still sitting there and
the operator loads last year's map.

This module finds the removable drives, reports what is already on them, and
copies the package to the root. It never deletes anything that was not named
in an explicit replace list: a stick usually carries other jobs, and wiping it
because the app assumed it was scratch space would be unforgivable.
"""

from __future__ import annotations

import ctypes
import os
import platform
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

#: Names that mark a folder as belonging to a monitor, so the app can say what
#: is already on the stick instead of listing dozens of unrelated files.
KNOWN_FOLDERS = {
    "TASKDATA": "ISOXML task data",
    "JD-Data": "John Deere Gen 4 / Gen 5",
    "GS3_2630": "John Deere GreenStar 3",
    "GS2_2600": "John Deere GreenStar 2",
    "AgGPS": "Trimble",
    "AgData": "Trimble / Ag Leader",
    "Raven": "Raven",
    "Rx": "Prescriptions",
}

#: Windows drive type for a removable volume.
_DRIVE_REMOVABLE = 2
_DRIVE_FIXED = 3


@dataclass
class Drive:
    """A drive the app could write a package to."""

    path: str
    label: str
    removable: bool
    total_bytes: int | None = None
    free_bytes: int | None = None
    contents: list[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["free_mb"] = round(self.free_bytes / 1e6) if self.free_bytes else None
        data["total_mb"] = round(self.total_bytes / 1e6) if self.total_bytes else None
        return data


def _usage(path: Path) -> tuple[int | None, int | None]:
    try:
        usage = shutil.disk_usage(path)
        return usage.total, usage.free
    except OSError:
        return None, None


def _describe_contents(root: Path, limit: int = 12) -> list[dict[str, Any]]:
    """What is already on the drive, with the monitor folders called out."""
    entries: list[dict[str, Any]] = []
    try:
        for child in sorted(root.iterdir()):
            if child.name.startswith(".") or child.name.upper() == "SYSTEM VOLUME INFORMATION":
                continue
            entries.append({
                "name": child.name,
                "is_dir": child.is_dir(),
                "known": KNOWN_FOLDERS.get(child.name, ""),
            })
            if len(entries) >= limit:
                break
    except (PermissionError, OSError):
        pass
    return entries


def _windows_drives() -> list[Drive]:
    drives: list[Drive] = []
    kernel32 = ctypes.windll.kernel32
    bitmask = kernel32.GetLogicalDrives()
    for index in range(26):
        if not (bitmask >> index) & 1:
            continue
        letter = f"{chr(ord('A') + index)}:\\"
        kind = kernel32.GetDriveTypeW(ctypes.c_wchar_p(letter))
        if kind not in (_DRIVE_REMOVABLE, _DRIVE_FIXED):
            continue
        root = Path(letter)
        if not root.exists():
            continue
        # Read the volume label so the user recognizes their own stick.
        buffer = ctypes.create_unicode_buffer(1024)
        kernel32.GetVolumeInformationW(
            ctypes.c_wchar_p(letter), buffer, ctypes.sizeof(buffer),
            None, None, None, None, 0,
        )
        total, free = _usage(root)
        drives.append(Drive(
            path=str(root),
            label=buffer.value or letter.rstrip("\\"),
            removable=kind == _DRIVE_REMOVABLE,
            total_bytes=total, free_bytes=free,
            contents=_describe_contents(root),
        ))
    return drives


def _unix_drives() -> list[Drive]:
    """Mount points where a removable volume normally appears."""
    candidates: list[Path] = []
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or ""

    if platform.system() == "Darwin":
        volumes = Path("/Volumes")
        if volumes.is_dir():
            candidates.extend(p for p in volumes.iterdir() if p.is_dir())
    else:
        for base in (Path("/media") / user, Path("/run/media") / user,
                     Path("/media"), Path("/mnt")):
            if base.is_dir():
                candidates.extend(p for p in base.iterdir() if p.is_dir())

    drives: list[Drive] = []
    seen: set[str] = set()
    for path in candidates:
        resolved = str(path.resolve())
        if resolved in seen or resolved == "/":
            continue
        seen.add(resolved)
        total, free = _usage(path)
        drives.append(Drive(
            path=str(path),
            label=path.name,
            removable=True,
            total_bytes=total, free_bytes=free,
            contents=_describe_contents(path),
        ))
    return drives


def list_drives() -> list[dict[str, Any]]:
    """Drives the app could write to, removable ones first."""
    try:
        drives = _windows_drives() if platform.system() == "Windows" else _unix_drives()
    except Exception:
        drives = []
    drives.sort(key=lambda d: (not d.removable, d.label.lower()))
    return [d.to_dict() for d in drives]


def plan_write(package_folder: Path, drive_path: str) -> dict[str, Any]:
    """Say what copying would do, before anything is written.

    Nothing on the stick is touched by this call. It exists so the app can
    show, in advance, which existing folders the copy would replace — the
    moment at which last season's prescription gets silently overwritten is
    the moment worth stopping at.
    """
    package_folder = Path(package_folder)
    drive = Path(drive_path)

    if not package_folder.is_dir():
        raise ValueError(f"Package folder not found: {package_folder}")
    if not drive.is_dir():
        raise ValueError(
            f"Drive not found: {drive_path}. If the stick was just plugged in, "
            "give the system a moment and refresh the list."
        )

    items = [p for p in package_folder.iterdir()]
    conflicts, new_items, total_bytes = [], [], 0
    for item in items:
        target = drive / item.name
        size = sum(f.stat().st_size for f in item.rglob("*") if f.is_file()) \
            if item.is_dir() else item.stat().st_size
        total_bytes += size
        record = {"name": item.name, "is_dir": item.is_dir(), "bytes": size}
        (conflicts if target.exists() else new_items).append(record)

    _, free = _usage(drive)
    return {
        "drive": str(drive),
        "package": str(package_folder),
        "new": new_items,
        "conflicts": conflicts,
        "total_bytes": total_bytes,
        "total_mb": round(total_bytes / 1e6, 2),
        "free_mb": round(free / 1e6) if free else None,
        "fits": (free is None) or (free > total_bytes * 1.1),
        "needs_confirmation": bool(conflicts),
    }


def write_to_drive(
    package_folder: Path,
    drive_path: str,
    replace: list[str] | None = None,
) -> dict[str, Any]:
    """Copy the package to the drive root.

    Parameters
    ----------
    replace:
        Names the caller has explicitly agreed to overwrite. Anything already
        on the stick and not named here is left alone and reported as skipped —
        a stick normally carries other jobs, and the app has no business
        deciding they are expendable.
    """
    plan = plan_write(package_folder, drive_path)
    if not plan["fits"]:
        raise ValueError(
            f"The package needs {plan['total_mb']} MB and the drive has "
            f"{plan['free_mb']} MB free."
        )

    package_folder = Path(package_folder)
    drive = Path(drive_path)
    allowed = set(replace or [])
    copied, skipped = [], []

    for item in package_folder.iterdir():
        target = drive / item.name
        if target.exists() and item.name not in allowed:
            skipped.append({
                "name": item.name,
                "reason": "already on the drive and not marked for replacement",
            })
            continue
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)
        copied.append(item.name)

    # Flush to the device: a stick pulled out before the OS finishes writing
    # arrives at the machine with a truncated file.
    try:
        os.sync()
    except AttributeError:  # Windows has no os.sync
        pass

    return {
        "drive": str(drive),
        "copied": copied,
        "skipped": skipped,
        "bytes": plan["total_bytes"],
        "message": (
            f"{len(copied)} item(s) copied to {drive}. "
            "Eject the drive through the operating system before pulling it out, "
            "so the write finishes."
        ),
    }
