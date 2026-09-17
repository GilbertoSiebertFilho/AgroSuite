"""The settings the app keeps between runs, over HTTP.

Two of them: the folder projects are kept in, and whether the app saves the
session into it by itself. :mod:`agrosuite.app.settings` owns the file and
the rules; this module maps them onto the API, and makes sure that changing
the folder takes the project that is already written along with it — a new
folder that stays empty until the next change would leave the person looking
at two places for one project.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from .. import autosave as autosave_mod
from .. import projectlock as lock_mod
from .. import settings as settings_mod

router = APIRouter(prefix="/api/settings", tags=["settings"])


class SettingsRequest(BaseModel):
    """What to change. What is not sent is not changed."""

    projects_dir: str | None = None
    autosave: bool | None = None


def _payload(config: settings_mod.Settings, **extra: Any) -> dict[str, Any]:
    from agrosuite.app import server as server_mod

    return {**config.payload(), "autosave_status": autosave_mod.status(server_mod.state), **extra}


@router.get("")
def get_settings() -> dict[str, Any]:
    """Where projects are kept, and whether they are kept automatically.

    Nothing is created here. "Created on first use" means the first save
    that needs it, not the first look at the panel: opening the app, or
    merely asking what the settings are, should leave nothing behind on a
    machine where nobody saves anything. ``exists`` says whether it is there
    yet, which is all the folder picker needs to know before it opens.

    A stored folder that turns out to be a file is reported as a problem
    rather than as a failed request: the app has to open, and the panel is
    where it says what to fix.
    """
    config = settings_mod.read()
    folder = config.projects_dir
    exists = folder.is_dir()
    problem = None
    if folder.exists() and not exists:
        problem = (
            f"'{folder}' is a file, not a folder, so nothing can be saved into it. "
            "Choose another one with 'Change folder…'."
        )
    return _payload(config, exists=exists, problem=problem)


@router.put("")
def put_settings(request: SettingsRequest) -> dict[str, Any]:
    """Change the folder, the switch, or both.

    The folder is checked before it is stored — a relative path, a path that
    is a file, a path that cannot be created — so a save at midnight cannot
    be the first time anyone finds out about it.
    """
    from agrosuite.app import server as server_mod

    folder = None
    if request.projects_dir is not None:
        try:
            folder = settings_mod.validate_projects_dir(request.projects_dir)
        except ValueError as exc:
            raise server_mod._fail(str(exc))

    try:
        config = settings_mod.write(projects_dir=folder, autosave=request.autosave)
    except ValueError as exc:
        raise server_mod._fail(str(exc), 500)

    if request.autosave is False:
        # Nothing is written from here on, so the claim on the project goes
        # at once rather than a minute later when its heartbeat runs out:
        # the window beside this one can have the file back immediately.
        lock_mod.release()

    # The project already written moves to the new folder, rather than being
    # left behind under the old one as a copy that will quietly go stale.
    moved: dict[str, Any] = {"moved": False}
    if folder is not None:
        moved = autosave_mod.relocate(server_mod.state)
    return _payload(config, **moved)
