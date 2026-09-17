"""Saving the session without being asked.

The work of an afternoon used to live in a temporary folder and go with the
app unless someone remembered to press "Save project". This module is the
promise that it does not: the session is written to the projects folder as
it is worked on, and the app opens again on what was left open.

**When it writes.** Not on every change: a session with a million records in
it is a few seconds of parquet, and writing it again for every tick box
would make the app feel like it was thinking about something else. Not on a
timer alone either: a timer that fires every thirty seconds loses the last
thirty seconds, which is exactly the part nobody remembers doing. So the two
halves are split. Routes that change anything call ``state.touch()``, which
costs a counter and a clock reading; one daemon thread watches that counter
and writes when it has moved and the session has then been still for
:data:`IDLE_SECONDS`. Typing a price, clicking through three tabs and
setting a role is one write, a couple of seconds after the last of them —
and a clean shutdown writes once more, so what the last two seconds changed
goes in the file too.

**What it will not do.** An auto-save the person did not ask for must never
be the thing that loses their work, so:

* an empty session is never written over a file that holds datasets — the
  session that has nothing in it is not a project, and "New project" is not
  a command to delete the last one;
* a file this session did not write is never adopted: a new project whose
  name happens to match an existing file gets a file of its own beside it;
* every write goes through the atomic path in :mod:`agrosuite.app.persist`,
  with the file that was there kept beside it as one rolling backup, so an
  auto-save made over a mistake is one file rename away from being undone;
* a write that fails — a full disk, a USB stick pulled out, a folder gone
  read-only — is reported once, quietly, with the reason and what to do, and
  is retried when something changes rather than every half second;
* a project another running app is writing is not written here at all. The
  launcher takes the next free port, so the app can be open twice, and two
  windows auto-saving one file would leave the loser's afternoon in a single
  rolling backup. The file is claimed while it is being written to (see
  :mod:`agrosuite.app.projectlock`); the second window starts empty, says
  which project is open elsewhere, and saves under a name of its own.

**Where it runs.** In the process that serves the app, and only there: the
launcher sets :data:`ENV_FLAG`, and an app object merely imported — by a
test, a script, the MCP tools — starts no thread and writes nothing. An
import should never start writing to somebody's Documents folder.
"""

from __future__ import annotations

import os
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from . import persist as persist_mod
from . import projectlock as lock_mod
from . import session as session_mod
from . import settings as settings_mod

#: How long the session must sit still before it is written. Two seconds is
#: past the end of a burst of clicks and well short of the pause between two
#: thoughts, so a person working steadily gets one write per action rather
#: than one per keystroke, and never waits for one.
IDLE_SECONDS = 2.0

#: How often the writer looks at the counter. Short enough that the delay
#: the person feels is IDLE_SECONDS and not IDLE_SECONDS plus a tick.
POLL_SECONDS = 0.5

#: The previous version of the project, kept beside it. One, not a history:
#: it exists so that an auto-save made over a mistake — a cleaning run on
#: the wrong file, a dataset deleted — can be undone by renaming this file
#: back. It is replaced by every successful save.
BACKUP_SUFFIX = ".backup"

#: The name a save is assembled under before it takes the project's place.
#: Visible in the folder for the microseconds between two renames.
PENDING_SUFFIX = ".writing"

#: Set by the launcher on the process that serves the app; set to "off" to
#: keep the writer out of the way even then.
ENV_FLAG = "AGROSUITE_AUTOSAVE"

_ON = {"1", "on", "true", "yes"}


def enabled_for_process() -> bool:
    """Whether this process is the app, rather than something importing it."""
    return os.environ.get(ENV_FLAG, "").strip().lower() in _ON


# ==========================================================================
# Writing one file
# ==========================================================================

def save_with_backup(
    state: session_mod.Session, target: Path, saved_at: str
) -> dict[str, Any]:
    """Write the session over ``target``, keeping the previous file beside it.

    Two renames, both atomic and both cheap. The new file is assembled in
    full under a temporary name — :func:`persist.save_session` does that part
    and nothing else here knows the format — then the file that was there
    becomes the rolling backup and the new one takes its place. Nothing is
    copied, so the cost of keeping a backup does not grow with the size of
    the project, and at no point in the sequence is the only copy of the
    work a half-written one.
    """
    pending = target.with_name(target.name + PENDING_SUFFIX)
    backup = target.with_name(target.name + BACKUP_SUFFIX)
    try:
        result = persist_mod.save_session(state, pending, saved_at)
        if target.exists():
            os.replace(target, backup)
        os.replace(pending, target)
    finally:
        # A save that failed leaves nothing of its own behind.
        pending.unlink(missing_ok=True)
    result["path"] = str(target)
    result["backup"] = str(backup) if backup.exists() else None
    return result


def free_name(folder: Path, name: str, keep: Path | None = None) -> Path:
    """``folder/name``, or the next free name beside it.

    The one file this rule protects is somebody else's: two projects can
    easily be called "Untitled project", and an auto-save that adopted the
    file of the same name would write a session with one dataset in it over
    a season's work. ``keep`` is the file this session already writes, which
    is not somebody else's and is reused rather than stepped around.

    A name another running app has claimed is taken too, even before that
    app has written its first byte: two windows started within a second of
    each other both want "Untitled project", and the file not existing yet
    is precisely the moment the collision is invisible.
    """
    def taken(candidate: Path) -> bool:
        if keep is not None and candidate == keep:
            return False
        return candidate.exists() or lock_mod.busy(candidate)

    candidate = folder / name
    if not taken(candidate):
        return candidate
    stem = name[: -len(persist_mod.EXTENSION)] if name.endswith(persist_mod.EXTENSION) else name
    for suffix in range(2, 1000):
        candidate = folder / f"{stem} {suffix}{persist_mod.EXTENSION}"
        if not taken(candidate):
            return candidate
    # A thousand projects of one name is not a case worth a better answer
    # than a name nobody else can have.
    return folder / f"{stem} {int(time.time())}{persist_mod.EXTENSION}"


def move_project(source: Path, target: Path) -> Path:
    """Move an auto-saved project, and its backup, to ``target``.

    A rename of the project and a change of the projects folder both end
    here, and both mean the same thing: the file that is already written
    goes to the new name or the new folder rather than staying behind as a
    second copy of the work under the old one. The backup travels with it,
    because a backup beside the project it no longer belongs to is a trap.

    If the move cannot be done — the old drive is gone, the new folder is
    read-only — the project keeps the file it has and the caller is told so
    by the returned path: a rename is not worth losing a save over, and the
    next write tries again.
    """
    if source == target:
        return source
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        # shutil.move rather than os.replace: a new projects folder can be on
        # another drive, where a rename is not allowed at all.
        shutil.move(str(source), str(target))
    except OSError:
        return source

    backup = source.with_name(source.name + BACKUP_SUFFIX)
    if backup.exists():
        try:
            shutil.move(str(backup), str(target.with_name(target.name + BACKUP_SUFFIX)))
        except OSError:
            # The project moved; a backup left behind is untidy, not lost
            # work, and saying so would be noise over something the next
            # save replaces anyway.
            pass
    return target


def target_for(state: session_mod.Session, folder: Path) -> Path:
    """The file this session auto-saves to, claimed, and moved if it has to be.

    The name follows the project's: renaming the project in the panel
    renames the file, and the old one is moved rather than left behind.

    Claiming happens here because this is the one place that decides which
    file the session writes to, and the claim has to be in hand *before* the
    file is moved: a project renamed onto a name another window is already
    writing would be the collision this whole mechanism exists to prevent.
    Raises :class:`projectlock.Busy` when that file belongs to another
    running app, which the writer reports rather than writing anyway.
    """
    wanted = f"{persist_mod.slugify(state.project['name'])}{persist_mod.EXTENSION}"
    current = state.autosave_path
    name = state.project["name"]

    if current is not None and lock_mod.busy(current):
        # The claim on the file this session was writing has gone to another
        # window — a machine asleep for longer than the heartbeat lasts is
        # how that happens, and it is the one case where holding a claim is
        # not proof of still having it. The session does not write into
        # their file and does not move it either: it takes a file of its
        # own, below, with everything it holds in it.
        current = None
        state.autosave_path = None

    if current is not None and current.parent == folder and current.name == wanted:
        lock_mod.hold(current, name)
        return current

    target = free_name(folder, wanted, keep=current)
    lock_mod.hold(target, name)
    if current is None:
        return target

    moved = move_project(current, target)
    if moved == current:
        # The move could not be made and the session stays on the file it
        # has, so the claim goes back to it: the new name is not this app's
        # to hold when nothing of this project is in it.
        lock_mod.hold(current, name)
        return moved

    if (state.project_file or {}).get("path") == current:
        # The panel names this same file; it followed the project rather
        # than being left pointing at a name that no longer exists.
        state.project_file["path"] = moved
    try:
        persist_mod.forget_recent(current)
    except OSError:
        pass
    return moved


# ==========================================================================
# What would be picked up again
# ==========================================================================

def scan(folder: Path) -> dict[str, Any]:
    """What the projects folder holds: ``{"project", "skipped"}``.

    ``project`` is the most recently written one that can be read, described
    from its manifest — a small member of the ZIP, so naming a project costs
    nothing next to loading it. ``skipped`` is every newer file that could
    not be read, with the reason.

    A damaged file does not hide the project behind it, and it is not passed
    over in silence either: someone whose newest project will not open has
    to be told that, or an app that opens on last week's work looks like an
    app that lost this week's. Backups are not candidates at all — their
    name keeps them out of the ``*.agrosuite`` listing on purpose.
    """
    found: dict[str, Any] | None = None
    skipped: list[dict[str, str]] = []
    try:
        files = sorted(
            (item for item in folder.glob(f"*{persist_mod.EXTENSION}") if item.is_file()),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return {"project": None, "skipped": skipped}

    for path in files:
        try:
            manifest = persist_mod.read_manifest(path)
        except (OSError, ValueError) as exc:
            skipped.append({"name": path.name, "reason": str(exc)})
            continue
        found = {
            "path": str(path),
            "name": path.stem,
            "project": (manifest.get("project") or {}).get("name") or path.stem,
            "saved_at": manifest.get("saved_at"),
            "datasets": len(manifest["datasets"]),
        }
        break
    return {"project": found, "skipped": skipped}


def latest_project(folder: Path) -> dict[str, Any] | None:
    """The project that would be reopened, or ``None``."""
    return scan(folder)["project"]


# ==========================================================================
# The writer
# ==========================================================================

class AutoSaver:
    """The daemon that writes the session once it has gone quiet.

    One instance per process, held below as :data:`_saver`. It reads the
    session through a callable rather than holding it, because the tests
    swap the server's session and the writer must follow the one that is
    actually being served.
    """

    def __init__(self, session_of: Callable[[], session_mod.Session]) -> None:
        self._session_of = session_of
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        #: One write at a time. The shutdown does not wait for ever on the
        #: thread it is stopping, so its own last write can meet a save
        #: already under way, and two writers sharing one temporary name
        #: would leave the pair of them with half a file.
        self._writing = threading.RLock()
        self._watched: session_mod.Session | None = None
        #: The revision already dealt with — written, or deliberately not
        #: written. It is what keeps a failure from being retried, and
        #: reported, every half second.
        self._handled = 0
        self._status: dict[str, Any] = {"state": "idle", "path": None,
                                        "saved_at": None, "message": None}

    # -- lifecycle -------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        # A daemon thread: closing the app must not wait for the next tick.
        # The write that matters on the way out is the one stop() makes.
        self._thread = threading.Thread(
            target=self._loop, name="agrosuite-autosave", daemon=True
        )
        self._thread.start()

    def stop(self, final_write: bool = True) -> None:
        """Stop watching, and write once more on the way out.

        The last few seconds of work are the ones the idle delay has not got
        to yet, and they are also the ones the person is most sure they did.
        """
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=POLL_SECONDS * 4)
        if final_write:
            self.flush()

    def flush(self) -> None:
        """Write now, if there is anything to write.

        The idle delay is what makes the writer cheap; this is the way past
        it for the two callers that cannot wait — the shutdown, and a test.
        """
        self._tick(final=True)

    # -- what the interface shows ----------------------------------------
    def status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._status)

    def _set(self, **fields: Any) -> None:
        with self._lock:
            self._status.update(fields)

    # -- the loop --------------------------------------------------------
    def _loop(self) -> None:
        while not self._stop.wait(POLL_SECONDS):
            try:
                self._tick()
            except Exception as exc:  # pragma: no cover - the thread must not die
                self._set(state="failed", message=(
                    f"The project could not be saved automatically: {exc}. "
                    "Use 'Save project' to keep this session while it is looked at."
                ))

    def _tick(self, final: bool = False) -> None:
        with self._writing:
            self._tick_once(final)

    def _tick_once(self, final: bool) -> None:
        state = self._session_of()
        if state is not self._watched:
            # A different session — the tests swap one in — is a session
            # whose changes have never been written by this writer.
            self._watched = state
            self._handled = 0

        config = settings_mod.read()
        if not config.autosave:
            # Off means off: nothing is written, nothing is moved, and the
            # revision is left alone so that turning it back on saves what
            # was done in the meantime. The claim goes as well — an app that
            # writes nothing has no business holding a file against the
            # window beside it.
            lock_mod.release()
            self._set(state="off")
            return

        # The claim this app holds says "still here" from this tick, which
        # runs anyway, rather than from a timer of its own: two timers
        # doing one job is two things to stop on the way out. It writes at
        # most every few seconds; the rest of the ticks cost a comparison.
        lock_mod.beat()

        revision = state.revision
        if revision == self._handled:
            return
        if not final and time.monotonic() - state.touched_at < IDLE_SECONDS:
            self._set(state="pending")
            return
        self._write(state, config, revision)

    def _write(
        self, state: session_mod.Session, config: settings_mod.Settings, revision: int
    ) -> None:
        if not state.list():
            # An empty session is not a project. Writing it would replace an
            # afternoon's work with a file holding nothing — and the empty
            # session is exactly what the app holds just after "New project"
            # and just before the first file is opened. The revision counts
            # as handled so the writer does not come back every half second.
            self._handled = revision
            return

        self._set(state="saving")
        try:
            folder = settings_mod.ensure(config.projects_dir)
            target = target_for(state, folder)
            saved_at = datetime.now().astimezone().isoformat(timespec="seconds")
            save_with_backup(state, target, saved_at)
        except lock_mod.Busy as busy:
            # Another window took the file between the last save and this
            # one — a rename onto its project, or the two apps starting
            # together. Nothing is written into it, and the revision counts
            # as handled so the panel says this once rather than every half
            # second. The next change looks again, and by then the name is
            # usually free or a free one is chosen beside it.
            self._handled = revision
            self._set(state="locked", path=None, saved_at=None,
                      message=busy.holder.sentence())
            return
        except Exception as exc:
            if state.revision != revision:
                # The session moved while the file was being written, which
                # is the one way a save fails without anything being wrong
                # with the disk. The next tick writes the newer session.
                self._set(state="pending")
                return
            self._handled = revision
            self._set(state="failed", path=None,
                      message=_failure(exc, config.projects_dir))
            return

        state.autosave_path = target
        # The panel names "the file the session is on", and until the person
        # has saved one somewhere themselves, this is it. Where they have
        # chosen a place of their own — a USB stick, a shared drive — this
        # does not argue with it; it only keeps its clock right.
        if state.project_file is None:
            state.project_file = {"path": target, "saved_at": saved_at, "token": None}
        elif state.project_file.get("path") == target:
            state.project_file["saved_at"] = saved_at

        # A project saved by hand goes on the recent list; one saved by the
        # app is no less recent, and without this the only project he never
        # pressed Save on would be the one the panel could not offer him a
        # way back to. The list is a convenience under the home folder: a
        # home folder that cannot be written must not turn a good save into
        # a reported failure.
        try:
            persist_mod.remember_recent(target, saved_at)
        except OSError:
            pass

        self._handled = revision
        self._set(state="saved", path=str(target), saved_at=saved_at, message=None)

def _failure(exc: BaseException, folder: Path) -> str:
    """Why a save failed and what to do about it, in one sentence each."""
    reason = getattr(exc, "strerror", None) or str(exc) or type(exc).__name__
    return (
        f"The project could not be saved to '{folder}': {reason}. Check that the "
        "folder is still there and can be written to — on a USB stick, that the "
        "drive is still plugged in — or choose another folder. The next change "
        "will try again, and 'Save project' still writes wherever you point it."
    )


# ==========================================================================
# The one in this process
# ==========================================================================

_saver: AutoSaver | None = None


def start(session_of: Callable[[], session_mod.Session]) -> AutoSaver | None:
    """Start the writer, if this process is the one serving the app."""
    global _saver
    if not enabled_for_process():
        return None
    if _saver is None:
        _saver = AutoSaver(session_of)
    _saver.start()
    return _saver


def stop() -> None:
    """Stop the writer, write the session one last time, and let the file go.

    The claim is released after the last write, not before it: the file is
    this app's until the last byte of it is written. A clean shutdown
    therefore leaves nothing beside the project, and the next launch — this
    one or the other window — picks it up without waiting for anything.
    """
    global _saver
    saver, _saver = _saver, None
    if saver is not None:
        saver.stop()
    lock_mod.release()


def current() -> AutoSaver | None:
    return _saver


def relocate(state: session_mod.Session) -> dict[str, Any]:
    """Take the session's file to wherever the settings now say.

    Called when the person changes the projects folder, so that the project
    already written moves with it instead of the new folder staying empty
    until the next change and the old one keeping a copy that goes stale.
    Nothing is copied and nothing is left behind under the old path.
    """
    config = settings_mod.read()
    source = state.autosave_path
    if source is None or not config.autosave:
        return {"moved": False}
    try:
        folder = settings_mod.ensure(config.projects_dir)
    except OSError as exc:
        return {"moved": False, "problem": _failure(exc, config.projects_dir)}
    if source.parent == folder:
        return {"moved": False}

    try:
        target = target_for(state, folder)
    except lock_mod.Busy as busy:
        # The new folder holds that project, open in another window. The
        # file stays where it is rather than being written into theirs.
        return {"moved": False, "problem": busy.holder.sentence()}
    if target == source:
        # target_for hands the old path back when the move could not be
        # made: the file is still where it was, which is the answer that
        # loses nothing, and the next save tries again.
        return {"moved": False, "problem": (
            f"'{source}' could not be moved to {folder}, so it is still where it was. "
            "The next change will try again."
        )}

    state.autosave_path = target
    if _saver is not None:
        _saver._set(state="saved", path=str(target))
    return {"moved": True, "from": str(source), "to": str(target)}


def status(state: session_mod.Session) -> dict[str, Any]:
    """What the status line in the panel says, whoever is asking.

    It answers with the settings even when no writer is running — an app
    being driven by a script still has a projects folder — so the interface
    has one place to read from.
    """
    config = settings_mod.read()
    saver = _saver
    payload: dict[str, Any] = {
        "enabled": config.autosave,
        "running": saver is not None,
        "folder": str(config.projects_dir),
        "state": "off" if not config.autosave else "idle",
        "path": str(state.autosave_path) if state.autosave_path else None,
        "saved_at": None,
        "message": config.warning,
        # Who has the project this app would otherwise have picked up, while
        # they still have it. None the rest of the time, which is almost
        # always: one window is the ordinary case.
        "lock": None,
    }
    if saver is not None:
        reported = saver.status()
        # What the writer last did, but only where it is still about this
        # session. A new project lets go of its file, and reporting the time
        # and the name of the project before it would promise that the empty
        # session on screen is saved in a file that holds something else. A
        # failure outlives the project it happened to — the folder is still
        # broken — and so does a write in flight.
        on_this = state.autosave_path is not None
        # Only "saved" is about a particular file; "off", "failed", "saving"
        # and "pending" are about the writer, and they outlive the project
        # they happened to.
        stale = reported["state"] == "saved" and not on_this
        payload.update({
            "state": "idle" if stale else reported["state"],
            "path": str(state.autosave_path) if on_this else None,
            "saved_at": reported["saved_at"] if on_this else None,
            "message": reported["message"] or payload["message"],
        })

    # A project open in another window outranks "idle": the line's job at
    # that moment is to say why nothing is being saved here, and it says it
    # only while this session has no file of its own — once this window is
    # writing its own project, "Saved 12:07" is the true answer and the
    # other window is no longer any of its business.
    refused = lock_mod.refusal()
    if config.autosave and refused is not None and state.autosave_path is None:
        payload.update({
            "state": "locked",
            "path": None,
            "saved_at": None,
            "lock": refused.payload(),
            "message": refused.sentence(),
        })
    return payload
