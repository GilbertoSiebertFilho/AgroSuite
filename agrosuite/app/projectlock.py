"""One project file, one running app.

The launcher takes the next free port rather than refusing to start, so
double-clicking ``run.bat`` a second time does not fail — it gives a second
live app. Before auto-save that was harmless: two windows, two sessions,
and nothing on disk unless somebody pressed Save. With auto-save it is the
one way this feature can lose work. Both apps reopen the newest project in
the folder, both write it a couple of seconds after each change, and the
last writer wins with a single rolling backup between them. An afternoon
done in the first window disappears without a message.

So a project file is claimed by the app that auto-saves into it, with a
small JSON file beside it — ``<name>.agrosuite.lock`` — and a second app
that finds a live claim does not adopt that project. It starts empty and
says which project is open elsewhere. It is not refused the *app*: it works
exactly as it always did, and saves under a name of its own.

**What the claim says.** The process id, the host, when it was taken, and a
heartbeat the owning app refreshes every :data:`HEARTBEAT_SECONDS` from the
thread that was already ticking (see :mod:`agrosuite.app.autosave`) — one
timer, not two. The project's name travels too, so the other window can be
named on screen rather than described as a path.

**When a claim is stale.** A crash, a pulled power lead or a killed process
leaves the file behind, and a lock nobody can clear without deleting a file
by hand is worse than the problem it solves. Two rules, either of which is
enough:

* the heartbeat is older than :data:`STALE_AFTER` — twelve missed beats, a
  minute, which is far longer than any pause the writer takes (it holds its
  own lock while a large project is written, and that write is seconds, not
  minutes) and short enough that relaunching after a crash is a wait nobody
  measures;
* the process named in it is *plainly* gone, on this host, which frees the
  file at once in the ordinary crash case.

The reverse is deliberately not a rule: a process that exists does **not**
make a claim live. Process ids are reused — on Windows within hours of a
reboot — and a reused id would otherwise hold a project for ever. Liveness
always rests on the heartbeat; the process check can only ever make a claim
stale *sooner*.

**Asking whether a process is alive.** ``os.kill(pid, 0)`` is the usual
answer and it is the wrong one here, because this runs on Windows: CPython
maps ``os.kill`` on Windows onto ``OpenProcess(PROCESS_ALL_ACCESS)`` plus
``TerminateProcess(handle, sig)``, so ``os.kill(pid, 0)`` does not ask a
question — it kills the other AgroSuite window and gives it exit code 0.
Windows is served by ``OpenProcess`` with the two rights that grant no
control over the process, and ``WaitForSingleObject(handle, 0)``: a handle
that is already signalled is a process that has exited. POSIX keeps
``os.kill(pid, 0)``, which there really is the question and nothing else.
Anywhere neither is available the answer is "cannot tell", and staleness
falls back to the heartbeat alone — a minute's wait, never a wrong answer.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

#: Marker that tells this file apart from any other JSON that happens to sit
#: beside a project.
APP_MARKER = "AgroSuite"

#: Bumped when a lock written today would mislead the version before it.
FILE_VERSION = 1

#: Appended to the project's own file name, so a claim is obviously about the
#: project it sits next to. It also keeps locks out of the ``*.agrosuite``
#: listing the folder is scanned with, which is what decides what is reopened.
SUFFIX = ".lock"

#: How often the owning app rewrites its heartbeat. Longer than the writer's
#: tick on purpose: the tick is half a second, and rewriting a file twice a
#: second to say "still here" is noise on a disk and on a USB stick.
HEARTBEAT_SECONDS = 5.0

#: How long a heartbeat is believed after it was written. Twelve missed
#: beats: generous enough that a long save, a suspended laptop lid or a
#: machine that stalls under another program's load never loses a project to
#: the window beside it, short enough that a crash costs a minute rather
#: than a support call.
STALE_AFTER = 60.0


class Busy(Exception):
    """Raised when the project is claimed by another running app.

    It carries the claim, because every caller wants the same two things out
    of it: the sentence to show, and who to name in it.
    """

    def __init__(self, holder: "Holder") -> None:
        super().__init__(holder.sentence())
        self.holder = holder


@dataclass(frozen=True)
class Holder:
    """What a lock file says, read at one moment.

    ``live`` is the answer the app acts on; the rest is what it says while
    explaining itself. ``free_at`` is when the claim goes stale if nothing
    refreshes it, which is the one thing somebody looking at a locked
    project actually wants to know.
    """

    project: str
    path: Path
    pid: int | None
    host: str
    taken_at: str | None
    heartbeat: str | None
    age: float | None
    live: bool
    free_at: datetime | None
    #: True when the process named in the claim is gone for certain, False
    #: when it is certainly there, None when this platform cannot say.
    running: bool | None

    def is_me(self) -> bool:
        return self.pid == os.getpid() and self.host == this_host()

    def sentence(self) -> str:
        """Why this project is not being saved here, and what to do instead.

        Named, not described: "the project in the folder" is not something
        anybody can go and close.
        """
        where = ("another AgroSuite window" if self.host == this_host()
                 else f"AgroSuite on {self.host}")
        which = f" (process {self.pid})" if self.pid and self.host == this_host() else ""
        when = ""
        if self.free_at is not None:
            when = (" — if that window has already gone, this project frees itself "
                    f"by {self.free_at.strftime('%H:%M:%S')}")
        return (
            f"'{self.project}' is open in {where}{which}, so nothing is saved into it "
            f"here. Work here and save under another name, or close the other window "
            f"and reopen it{when}."
        )

    def payload(self) -> dict[str, Any]:
        """The shape the interface reads."""
        return {
            "project": self.project,
            "path": str(self.path),
            "pid": self.pid,
            "host": self.host,
            "taken_at": self.taken_at,
            "heartbeat": self.heartbeat,
            "live": self.live,
            "free_at": self.free_at.isoformat(timespec="seconds") if self.free_at else None,
            "message": self.sentence(),
        }


def this_host() -> str:
    """This machine's name, as the claim records it.

    A projects folder can live on a shared drive, and a process id from
    another computer says nothing about this one.
    """
    try:
        return socket.gethostname() or "this computer"
    except OSError:  # pragma: no cover - a host without a name
        return "this computer"


def lock_path(project: Path | str) -> Path:
    """The claim that belongs to ``project``."""
    project = Path(project)
    return project.with_name(project.name + SUFFIX)


# ==========================================================================
# Is that process still there?
# ==========================================================================

def process_running(pid: int | None) -> bool | None:
    """``True`` running, ``False`` certainly gone, ``None`` cannot tell.

    Only ever used to call a claim stale sooner than its heartbeat would —
    never to keep one alive, because a process id can belong to something
    else entirely by the time it is read. See the module docstring for why
    ``os.kill`` is not used on Windows.
    """
    if not isinstance(pid, int) or pid <= 1:
        # 0 and negative ids mean "this process group" and "every process" to
        # kill(2); they are not a process to ask about.
        return None
    if sys.platform == "win32":  # pragma: no cover - answered on Windows only
        return _running_windows(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # It exists and belongs to another user. On a machine with one person
        # on it this is rare, and "there" is the safe answer either way.
        return True
    except OSError:
        return None
    return True


def _running_windows(pid: int) -> bool | None:  # pragma: no cover - Windows only
    """The Windows answer, asked with rights that cannot harm the process.

    ``PROCESS_QUERY_LIMITED_INFORMATION`` and ``SYNCHRONIZE`` allow waiting
    on the handle and nothing else — no terminating, no reading memory — and
    they are granted across user accounts, so the answer does not depend on
    who started the other window.
    """
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    except (ImportError, OSError, AttributeError, ValueError):
        return None

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    SYNCHRONIZE = 0x00100000
    ERROR_INVALID_PARAMETER = 87
    WAIT_TIMEOUT = 0x00000102

    # Spelled out rather than left to ctypes' defaults: a handle is 64 bits
    # on a 64-bit Windows and the default return type is a 32-bit int, which
    # would hand back a truncated handle and close somebody else's.
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL

    handle = kernel32.OpenProcess(
        PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE, False, pid
    )
    if not handle:
        # "No such process" is the one error that is an answer; access denied
        # and everything else mean the question could not be asked.
        return False if ctypes.get_last_error() == ERROR_INVALID_PARAMETER else None
    try:
        # A process object is signalled when the process has exited, so a
        # wait that times out immediately is a process still running.
        return kernel32.WaitForSingleObject(handle, 0) == WAIT_TIMEOUT
    finally:
        kernel32.CloseHandle(handle)


# ==========================================================================
# Reading a claim
# ==========================================================================

def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def holder_of(project: Path | str) -> Holder | None:
    """Who claims ``project``, live or stale, or ``None`` if nobody does.

    A lock file that cannot be read, or that does not hold what this version
    writes, counts as nobody: a damaged byte beside a project must never be
    the reason somebody cannot open their own work. It is replaced by the
    next claim rather than left to puzzle over.
    """
    project = Path(project)
    path = lock_path(project)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict) or raw.get("app") != APP_MARKER:
        return None

    pid = raw.get("pid")
    host = raw.get("host") if isinstance(raw.get("host"), str) else ""
    beat = _parse_time(raw.get("heartbeat"))
    now = datetime.now().astimezone()

    age: float | None = None
    if beat is not None:
        if beat.tzinfo is None:
            beat = beat.astimezone()
        # A heartbeat in the future means the clock moved, not that the
        # claim is fresher than fresh; it is treated as this instant, which
        # errs towards leaving the other window alone.
        age = max((now - beat).total_seconds(), 0.0)

    running = process_running(pid) if host == this_host() else None
    live = age is not None and age < STALE_AFTER and running is not False
    named = raw.get("project")
    return Holder(
        project=named if isinstance(named, str) and named else project.stem,
        path=project,
        pid=pid if isinstance(pid, int) else None,
        host=host or "another computer",
        taken_at=raw.get("taken_at") if isinstance(raw.get("taken_at"), str) else None,
        heartbeat=raw.get("heartbeat") if isinstance(raw.get("heartbeat"), str) else None,
        age=age,
        live=live,
        free_at=(beat + timedelta(seconds=STALE_AFTER)) if beat is not None else None,
        running=running,
    )


def busy(project: Path | str) -> bool:
    """Whether another running app is auto-saving into ``project``.

    The question the name of a new file is chosen against: a project this
    app holds is not busy, and neither is one whose claim has gone stale.
    """
    holder = holder_of(project)
    return bool(holder and holder.live and not holder.is_me())


# ==========================================================================
# Holding one
# ==========================================================================

@dataclass
class Lock:
    """The claim this process holds on one project file."""

    project: Path
    path: Path
    taken_at: str
    #: False when the lock file could not be written at all. The app carries
    #: on saving: a folder that refuses a 200-byte JSON file is about to
    #: refuse the project too, and reporting the project as unsaveable
    #: because its claim could not be written would be the tail wagging the
    #: dog. What is lost is the protection, not the work.
    recorded: bool = True
    name: str = ""
    _beat_at: float = 0.0

    def payload(self, heartbeat: str) -> dict[str, Any]:
        return {
            "app": APP_MARKER,
            "version": FILE_VERSION,
            "pid": os.getpid(),
            "host": this_host(),
            "taken_at": self.taken_at,
            "heartbeat": heartbeat,
            # So a reader from another version knows how long a silence has
            # to be before this claim means nothing.
            "heartbeat_seconds": HEARTBEAT_SECONDS,
            "stale_after_seconds": STALE_AFTER,
            "project": self.name or self.project.stem,
        }

    def beat(self, force: bool = False) -> None:
        """Say the app is still here, at most every ``HEARTBEAT_SECONDS``.

        Called from the auto-saver's tick, which runs anyway; rate-limited
        here rather than there so that the cadence is a property of the
        claim and not of whoever happens to call it.
        """
        if not self.recorded:
            return
        now = time.monotonic()
        if not force and now - self._beat_at < HEARTBEAT_SECONDS:
            return
        stamp = datetime.now().astimezone().isoformat(timespec="seconds")
        try:
            _write_payload(self.path, self.payload(stamp))
        except OSError:
            # A folder that has gone — the USB stick is out — is reported by
            # the writer, in the words of the save that failed. A heartbeat
            # has nothing to add to that.
            return
        self._beat_at = now

    def release(self) -> None:
        """Let the project go, leaving nothing behind to clear up.

        Only a claim that is still this app's is removed: if it went stale
        and another window took it over, that window's file belongs to it.
        """
        self.recorded = False
        holder = holder_of(self.project)
        if holder is not None and not holder.is_me():
            return
        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            # A lock that cannot be deleted goes stale on its own within the
            # minute, which is exactly the crash case and needs nobody.
            pass


_guard = threading.Lock()
_held: Lock | None = None
#: The claim this app was last refused, so the panel can say why it is not
#: saving into the project the folder holds. Cleared the moment this app
#: claims anything, and ignored once the other window has gone.
_refused: Holder | None = None


def held() -> Lock | None:
    """The project this app has claimed, or ``None``."""
    return _held


def hold(project: Path | str, name: str = "") -> Lock:
    """Claim ``project`` for this app, letting go of whatever was held.

    Raises :class:`Busy` when another running app has it, in which case what
    was held before is kept: being refused a new file is not a reason to
    stop protecting the one already being written.
    """
    global _held, _refused
    project = Path(project)
    with _guard:
        if _held is not None and _held.project == project:
            if name:
                _held.name = name
            _held.beat(force=True)
            _refused = None
            return _held

        claimed = _claim(project, name)
        if _held is not None:
            _held.release()
        _held = claimed
        _refused = None
        return claimed


def release() -> None:
    """Let go of the project this app holds, if it holds one."""
    global _held, _refused
    with _guard:
        if _held is not None:
            _held.release()
        _held = None
        _refused = None


def beat() -> None:
    """Refresh the heartbeat of whatever is held. Cheap enough to call on
    every tick of the writer; it writes at most every HEARTBEAT_SECONDS."""
    lock = _held
    if lock is not None:
        lock.beat()


def refusal() -> Holder | None:
    """The claim this app was refused, if it still stands.

    Read again rather than remembered: the other window closing is the end
    of the story, and a panel still explaining a lock that has gone would be
    worse than one that never mentioned it.
    """
    global _refused
    refused = _refused
    if refused is None:
        return None
    current = holder_of(refused.path)
    if current is None or not current.live or current.is_me():
        _refused = None
        return None
    return current


def _claim(project: Path, name: str) -> Lock:
    """Take the lock file beside ``project``, or say who has it.

    Exclusive creation first, which is the whole of the ordinary case and is
    atomic on every filesystem the app runs on. A file that is already there
    is read: a live claim belonging to somebody else is refused, and a stale
    or damaged one is replaced and then read back, because two apps starting
    at the same second can both find the same stale claim and only one of
    them may come away holding it.
    """
    global _refused
    path = lock_path(project)
    lock = Lock(
        project=project,
        path=path,
        taken_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        name=name or project.stem,
    )
    payload = lock.payload(lock.taken_at)

    for attempt in range(3):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            pass
        except OSError:
            return _hollow(lock)
        else:
            with os.fdopen(handle, "w", encoding="utf-8") as out:
                json.dump(payload, out, indent=2)
            lock._beat_at = time.monotonic()
            return lock

        holder = holder_of(project)
        if holder is not None and holder.live and not holder.is_me():
            _refused = holder
            raise Busy(holder)
        try:
            _write_payload(path, payload)
        except OSError:
            return _hollow(lock)
        settled = holder_of(project)
        if settled is not None and settled.is_me():
            lock._beat_at = time.monotonic()
            return lock
        if settled is not None and settled.live:
            _refused = settled
            raise Busy(settled)
    # Three rounds of losing a race to a claim that is then gone again is
    # not a state worth a fourth: the file is written and the app carries on.
    return _hollow(lock)


def _hollow(lock: Lock) -> Lock:
    lock.recorded = False
    return lock


def _write_payload(path: Path, payload: dict[str, Any]) -> None:
    """Replace the lock file in one step.

    A heartbeat half written is a claim that reads as damaged, which is a
    claim that reads as free — so it is assembled beside itself and moved
    over, the same way the project it guards is.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial",
                                         dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as out:
            json.dump(payload, out, indent=2)
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise
