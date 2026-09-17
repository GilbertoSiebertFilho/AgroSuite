"""What the app remembers between runs: where projects live, and whether it
keeps them by itself.

A session used to live in a temporary folder and go with the app unless
someone remembered to press "Save project". Saving by itself needs one thing
the app never had — a folder chosen once that survives a restart — and that
is what this file holds: ``$AGROSUITE_HOME/settings.json``, beside the
machine profiles and the recent list, plain indented JSON so it can be read,
edited or copied with a text editor.

The default folder is deliberately **not** inside ``.agrosuite``. That folder
is hidden, and a project the person cannot see in Explorer is a project they
cannot copy to a USB stick, attach to an e-mail or include in a backup. So:
``Documents/AgroSuite`` where there is a Documents folder, and
``<home>/AgroSuite`` where there is not — created the first time something
needs it, not at install time, so an app that is only being tried out leaves
nothing behind.

A folder is checked when it is chosen, not at midnight when the first save
fails: a relative path, a path that is really a file and a path that cannot
be created are each refused with the sentence that says what to do about it.

Nothing here is overwritten in silence. A settings file this version cannot
read is left exactly as it is and the defaults are used instead — the folder
someone typed into it is not worth losing to a stray comma — and if the
person then changes a setting, the unreadable file is moved aside rather
than written over.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import persist as persist_mod

#: Bumped when a file written today would confuse the version before it.
FILE_VERSION = 1

#: What the projects folder is called under Documents (or under the home
#: folder). The app's own name, because that is what the person will look
#: for when they go hunting for their files.
FOLDER_NAME = "AgroSuite"

_lock = threading.Lock()


@dataclass(frozen=True)
class Settings:
    """The settings as they are in force, whatever the file on disk says."""

    projects_dir: Path
    autosave: bool
    #: What was wrong with the stored file, in a sentence the person can act
    #: on, or ``None``. It travels to the interface rather than to a log: a
    #: settings file that is being ignored is something to be told about.
    warning: str | None = None

    def payload(self) -> dict[str, Any]:
        """The shape the interface reads."""
        return {
            "projects_dir": str(self.projects_dir),
            "autosave": self.autosave,
            "default_projects_dir": str(default_projects_dir()),
            "settings_file": str(settings_path()),
            "warning": self.warning,
        }


def settings_path() -> Path:
    """The settings file, under whatever ``AGROSUITE_HOME`` names now.

    Read on every call, like the rest of the home folder: the tests point it
    at a temporary folder, and a packaged install may set it per user.
    """
    return persist_mod.home_dir() / "settings.json"


def default_projects_dir() -> Path:
    """Where projects go until the person says otherwise.

    ``Documents`` is checked rather than assumed: it is absent on a fresh
    Linux account and renamed on a localized Windows in some setups, and a
    folder created inside a Documents that does not exist would be a second
    hiding place rather than a visible one.
    """
    home = Path.home()
    documents = home / "Documents"
    return (documents if documents.is_dir() else home) / FOLDER_NAME


def read() -> Settings:
    """The settings in force. Never raises: the app has to open."""
    path = settings_path()
    if not path.exists():
        return Settings(default_projects_dir(), True)

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("it does not hold a settings object")
    except (OSError, ValueError) as exc:
        return Settings(
            default_projects_dir(), True,
            f"The settings file could not be read ({exc}), so the defaults are in "
            f"use and nothing in it was changed. Fix it or move it away: {path}",
        )

    warning: str | None = None
    folder = default_projects_dir()
    stored_folder = raw.get("projects_dir")
    if isinstance(stored_folder, str) and stored_folder.strip():
        candidate = Path(stored_folder.strip()).expanduser()
        if candidate.is_absolute():
            folder = candidate
        else:
            # A relative folder would be resolved against wherever the app
            # was started from — a place nobody chose — so it is refused
            # here exactly as it is refused when it is typed.
            warning = (
                f"'{stored_folder}' in {path} is a relative path, and the app cannot "
                f"tell where it is meant from, so projects go to {folder} instead. "
                "Put the full path in, from the drive or the root."
            )

    autosave = raw.get("autosave")
    if not isinstance(autosave, bool):
        autosave = True
    return Settings(folder, autosave, warning)


def write(projects_dir: Path | str | None = None, autosave: bool | None = None) -> Settings:
    """Change what is stored and return the settings that are now in force.

    Only what is passed is changed, and every other key the file holds — one
    from a later version, one typed by hand — is kept: the file belongs to
    the person, not to this function.
    """
    with _lock:
        path = settings_path()
        data: dict[str, Any] = {}
        if path.exists():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                data = loaded if isinstance(loaded, dict) else {}
            except (OSError, ValueError):
                # Unreadable and about to be replaced: it is put aside, not
                # written over, so whatever is in it can still be looked at.
                try:
                    os.replace(path, path.with_suffix(".json.broken"))
                except OSError:
                    pass

        data["version"] = FILE_VERSION
        if projects_dir is not None:
            data["projects_dir"] = str(projects_dir)
        if autosave is not None:
            data["autosave"] = bool(autosave)

        try:
            persist_mod.write_atomically(
                path, (json.dumps(data, indent=2) + "\n").encode("utf-8")
            )
        except OSError as exc:
            raise ValueError(
                f"The settings could not be saved ({exc.strerror or exc}). Make that "
                f"folder writable, or point AGROSUITE_HOME at one that is: {path}"
            ) from exc
    return read()


def validate_projects_dir(raw: str | Path) -> Path:
    """The folder a typed path means, created if it is not there yet.

    Three refusals, each saying what to do instead of what went wrong. They
    are raised as ``ValueError`` so the route can turn them into a 400
    without knowing which one it caught.
    """
    text = str(raw).strip().strip('"').strip("'")
    if not text:
        raise ValueError(
            "Give the folder to keep projects in — the full path, from the drive or "
            "the root (e.g. C:\\Users\\you\\Documents\\AgroSuite)."
        )

    folder = Path(text).expanduser()
    if not folder.is_absolute():
        raise ValueError(
            f"'{folder}' is a relative path, and the app cannot tell where it is meant "
            "from. Give the full path of a folder, from the drive or the root "
            "(e.g. C:\\Data\\Projects or /home/you/projects)."
        )
    if folder.exists() and not folder.is_dir():
        raise ValueError(
            f"'{folder}' is a file, not a folder. Choose a folder to keep projects in, "
            "or make a new one beside it."
        )
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ValueError(
            f"Could not create the folder '{folder}': {exc.strerror or exc}. Check that "
            "the drive is connected and that its parent folder is writable, or choose "
            "another folder."
        ) from exc
    return folder.resolve()


def ensure(folder: Path) -> Path:
    """Make the projects folder, on the first save that needs it.

    Raises ``OSError`` — the caller is the auto-saver, which reports a folder
    it cannot make the same way it reports a disk it cannot write to.
    """
    folder.mkdir(parents=True, exist_ok=True)
    return folder
