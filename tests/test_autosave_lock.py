"""One project file, one running app.

The launcher takes the next free port instead of refusing to start, so
double-clicking ``run.bat`` twice gives two live apps rather than an error.
Before auto-save that cost nothing. With it, both windows reopen the newest
project, both write it a couple of seconds after every change, and the
loser's afternoon survives only as the single rolling backup — until the
next save replaces that too. Nobody is told anything.

So the app claims the project it auto-saves into, and these tests are the
two halves of that promise: a project another window is writing is never
adopted here, and a claim left behind by a crash never needs anybody to
delete a file. In between sit the ordinary courtesies — the claim is given
back on the way out, on "New project", on a rename, when the switch goes
off — and the rule that a damaged claim stops nothing at all.

The other window is a real process: a claim is only as good as its answer
to "is that process still there", and a pid invented for a test would never
exercise it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agrosuite.app import autosave  # noqa: E402
from agrosuite.app import persist  # noqa: E402
from agrosuite.app import projectlock  # noqa: E402
from agrosuite.app import server as server_mod  # noqa: E402
from agrosuite.app import session as session_mod  # noqa: E402
from agrosuite.app import settings as settings_mod  # noqa: E402
from agrosuite.demo import synthetic_harvest  # noqa: E402

APP_JS = (ROOT / "agrosuite" / "app" / "static" / "app.js").read_text(encoding="utf-8")
LOCK_SOURCE = (ROOT / "agrosuite" / "app" / "projectlock.py").read_text(encoding="utf-8")


# ==========================================================================
# Fixtures
# ==========================================================================

@pytest.fixture(autouse=True)
def no_claim_left_behind():
    """Every test gives the project back.

    The claim this process holds is module state, like the writer itself, and
    a test that walked off with one would hand the next test a project it
    cannot have.
    """
    yield
    projectlock.release()


@pytest.fixture
def projects(tmp_path, monkeypatch) -> Path:
    """A home folder and a projects folder of this test's own."""
    monkeypatch.setenv("AGROSUITE_HOME", str(tmp_path / "home"))
    folder = tmp_path / "projects"
    settings_mod.write(projects_dir=folder, autosave=True)
    return folder


@pytest.fixture
def state(monkeypatch) -> session_mod.Session:
    """A session of this test's own, in the place the routes read from."""
    fresh = session_mod.Session()
    monkeypatch.setattr(server_mod, "state", fresh)
    yield fresh
    fresh.cleanup()


@pytest.fixture
def quick(monkeypatch) -> None:
    """The writer, with the waiting taken out of it."""
    monkeypatch.setattr(autosave, "IDLE_SECONDS", 0.05)
    monkeypatch.setattr(autosave, "POLL_SECONDS", 0.05)
    monkeypatch.setenv(autosave.ENV_FLAG, "on")


@pytest.fixture
def other_window():
    """A process standing in for the app in the other window.

    It does nothing but stay alive, which is all the claim asks of it. The
    tests that need the other window *gone* kill it and then ask again,
    which is the crash this whole mechanism is measured against.
    """
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(600)"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        yield process
    finally:
        process.kill()
        process.wait(timeout=10)


# ==========================================================================
# Helpers
# ==========================================================================

def _their_project(folder: Path, name: str, label: str = "Their morning") -> Path:
    """A project in the folder, written by a session that is not this one."""
    settings_mod.ensure(folder)
    path = folder / f"{name}.agrosuite"
    other = session_mod.Session()
    try:
        other.project["name"] = name
        other.add(synthetic_harvest(), label, "demo")
        persist.save_session(other, path, "2026-09-17T09:00:00-06:00")
    finally:
        other.cleanup()
    return path


def _claimed_by(project: Path, pid: int, age_seconds: float = 0.0,
                name: str | None = None) -> Path:
    """Write the claim the other window would have left beside ``project``."""
    beat = datetime.now().astimezone() - timedelta(seconds=age_seconds)
    payload = {
        "app": projectlock.APP_MARKER,
        "version": projectlock.FILE_VERSION,
        "pid": pid,
        "host": projectlock.this_host(),
        "taken_at": beat.isoformat(timespec="seconds"),
        "heartbeat": beat.isoformat(timespec="seconds"),
        "heartbeat_seconds": projectlock.HEARTBEAT_SECONDS,
        "stale_after_seconds": projectlock.STALE_AFTER,
        "project": name or project.stem,
    }
    path = projectlock.lock_path(project)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _claim_of(project: Path) -> dict:
    return json.loads(projectlock.lock_path(project).read_text(encoding="utf-8"))


def _until(predicate, timeout: float = 8.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


# ==========================================================================
# The second window
# ==========================================================================

def test_a_project_another_window_is_writing_is_not_picked_up(
    projects, state, quick, other_window
):
    """The case that costs work: the app open twice, both on one file.

    The second window starts empty and names the project instead — it does
    not reopen it, does not write it, and does not leave it looking as
    though the work had gone.
    """
    theirs = _their_project(projects, "North quarter")
    _claimed_by(theirs, other_window.pid)
    untouched = theirs.read_bytes()

    with TestClient(server_mod.app) as client:
        assert client.get("/api/datasets").json()["datasets"] == []

        resumed = client.get("/api/session/recent").json()["resumed"]
        assert resumed and resumed["locked"] is True and resumed["ok"] is False
        assert "North quarter" in resumed["reason"]
        assert "open in another AgroSuite window" in resumed["reason"]
        assert "save under another name" in resumed["reason"]
        assert "started empty" in resumed["reason"]

        # The panel says the same thing in its own quiet line.
        status = client.get("/api/session/autosave").json()
        assert status["state"] == "locked"
        assert status["lock"]["project"] == "North quarter"
        assert status["path"] is None and status["saved_at"] is None

        # And the offer to reopen is withdrawn rather than left to fail.
        latest = client.get("/api/session/latest").json()
        assert latest["available"] is True and latest["locked"] is True
        assert latest["resumable"] is False
        assert latest["lock"]["pid"] == other_window.pid

    assert theirs.read_bytes() == untouched, "the other window's project was written"
    assert _claim_of(theirs)["pid"] == other_window.pid, "their claim was taken over"


def test_the_second_window_still_works_and_saves_under_another_name(
    projects, state, quick, other_window, tmp_path
):
    """A refusal to auto-save into one file, not a refusal to run.

    Everything works; the project this window makes gets a file of its own
    beside the one it could not have.
    """
    theirs = _their_project(projects, "Untitled project")
    _claimed_by(theirs, other_window.pid)
    untouched = theirs.read_bytes()

    with TestClient(server_mod.app) as client:
        assert client.post("/api/import/demo", json={"kind": "harvest"}).status_code == 200

        mine = projects / "Untitled project 2.agrosuite"
        assert _until(mine.exists), "the second window could not save at all"
        assert _until(lambda: client.get("/api/session/autosave").json()["state"] == "saved")
        assert state.autosave_path == mine
        assert _claim_of(mine)["pid"] == os.getpid(), "its own file is not claimed"

        # Saving by hand, where he points it, works as it always did.
        elsewhere = tmp_path / "by hand"
        saved = client.post("/api/session/save",
                            json={"name": "By hand", "path": str(elsewhere)})
        assert saved.status_code == 200, saved.text
        assert (elsewhere / "By hand.agrosuite").exists()

    assert theirs.read_bytes() == untouched


def test_opening_their_project_by_hand_reads_it_without_taking_it(
    projects, state, quick, other_window
):
    """'Open project' on a file another window has: it opens, and it says
    that nothing here will be written into it."""
    theirs = _their_project(projects, "Shared field")
    _claimed_by(theirs, other_window.pid)

    with TestClient(server_mod.app) as client:
        opened = client.post("/api/session/open", json={"path": str(theirs)})
        assert opened.status_code == 200, opened.text
        payload = opened.json()
        assert len(payload["datasets"]) == 1, "the project did not open"
        assert "open in another AgroSuite window" in payload["locked"]
        assert payload["locked"] in payload["warning"]
        assert state.autosave_path is None, "it was adopted for auto-saving anyway"
        assert _claim_of(theirs)["pid"] == other_window.pid


# ==========================================================================
# A claim that nobody is behind
# ==========================================================================

def test_a_claim_with_an_old_heartbeat_is_taken(projects, state, quick, other_window):
    """The crash case, and the rule that keeps it from needing a human.

    The process in the claim is alive — it is the pid of a running program —
    and the claim is still stale, because nothing has refreshed it. A pid is
    reused within hours of a reboot; a heartbeat is not.
    """
    theirs = _their_project(projects, "Left behind")
    _claimed_by(theirs, other_window.pid, age_seconds=projectlock.STALE_AFTER + 30)

    holder = projectlock.holder_of(theirs)
    assert holder.running is True, "the stand-in process is not running"
    assert holder.live is False, "a live pid alone held the project"

    with TestClient(server_mod.app) as client:
        resumed = client.get("/api/session/recent").json()["resumed"]
        assert resumed and resumed["ok"] is True
        assert resumed["project"] == "Left behind"
        assert _claim_of(theirs)["pid"] == os.getpid(), "the stale claim was not taken"


def test_a_claim_whose_process_has_gone_is_taken_at_once(projects, state, quick,
                                                         other_window):
    """A killed app frees its project immediately, rather than in a minute.

    The heartbeat is seconds old, which is what it would be at the moment of
    a crash; the process is the thing that is gone.
    """
    theirs = _their_project(projects, "Crashed out")
    _claimed_by(theirs, other_window.pid, age_seconds=1.0)
    other_window.kill()
    other_window.wait(timeout=10)

    assert projectlock.process_running(other_window.pid) is False
    assert projectlock.holder_of(theirs).live is False
    assert projectlock.busy(theirs) is False

    with TestClient(server_mod.app) as client:
        resumed = client.get("/api/session/recent").json()["resumed"]
        assert resumed and resumed["ok"] is True and resumed["project"] == "Crashed out"


def test_a_stale_claim_is_never_a_file_to_delete_by_hand(projects, other_window):
    """What the person is promised: nothing to clean up, ever.

    A claim from a process that has gone, and one whose heartbeat has run
    out, are both taken by the next app that wants the project — the second
    of those without asking the operating system anything at all.
    """
    project = projects / "Any project.agrosuite"
    settings_mod.ensure(projects)
    _claimed_by(project, other_window.pid, age_seconds=projectlock.STALE_AFTER + 1)
    lock = projectlock.hold(project, "Any project")
    assert lock.recorded and _claim_of(project)["pid"] == os.getpid()


def test_the_message_says_when_the_project_frees_itself(projects, other_window):
    """While the claim is still warm, the one thing worth knowing is how
    long it stays that way."""
    project = projects / "Still warm.agrosuite"
    settings_mod.ensure(projects)
    _claimed_by(project, other_window.pid, age_seconds=5.0)

    holder = projectlock.holder_of(project)
    assert holder.live is True
    free_at = datetime.fromisoformat(holder.payload()["free_at"])
    waiting = (free_at - datetime.now().astimezone()).total_seconds()
    assert 0 < waiting <= projectlock.STALE_AFTER
    assert holder.sentence().endswith(f"by {free_at.strftime('%H:%M:%S')}.")
    assert "frees itself" in holder.sentence()


# ==========================================================================
# Giving it back
# ==========================================================================

def test_the_claim_is_released_when_the_app_closes(projects, state, quick):
    """A clean shutdown leaves nothing beside the project."""
    with TestClient(server_mod.app) as client:
        assert client.post("/api/import/demo", json={"kind": "harvest"}).status_code == 200
        target = projects / "Untitled project.agrosuite"
        assert _until(target.exists)
        assert projectlock.lock_path(target).exists(), "the file it writes is not claimed"

    assert not projectlock.lock_path(target).exists(), "a claim outlived the app"
    assert target.exists(), "the project went with it"
    assert projectlock.held() is None


def test_a_new_project_gives_the_last_one_back(projects, state, quick):
    """'New project' lets go of the file, so the claim goes with it."""
    with TestClient(server_mod.app) as client:
        client.post("/api/import/demo", json={"kind": "harvest"})
        target = projects / "Untitled project.agrosuite"
        assert _until(lambda: projectlock.lock_path(target).exists())
        kept = target.read_bytes()

        assert client.post("/api/session/new", json={}).status_code == 200
        assert not projectlock.lock_path(target).exists()
        assert projectlock.held() is None
        assert target.read_bytes() == kept, "'New project' touched the file"


def test_switching_auto_save_off_gives_the_file_back(projects, state, quick):
    """Off means nothing is written, so there is nothing to hold."""
    with TestClient(server_mod.app) as client:
        client.post("/api/import/demo", json={"kind": "harvest"})
        target = projects / "Untitled project.agrosuite"
        assert _until(lambda: projectlock.lock_path(target).exists())

        assert client.put("/api/settings", json={"autosave": False}).status_code == 200
        assert not projectlock.lock_path(target).exists()
        assert projectlock.held() is None

        # And switching it back on takes it again.
        assert client.put("/api/settings", json={"autosave": True}).status_code == 200
        assert client.post("/api/project", json={"name": "On again"}).status_code == 200
        renamed = projects / "On again.agrosuite"
        assert _until(lambda: projectlock.lock_path(renamed).exists())


def test_a_rename_moves_the_claim_with_the_file(projects, state, quick):
    """The project file follows the project's name; a claim left beside the
    old name would hold a file that no longer exists."""
    with TestClient(server_mod.app) as client:
        client.post("/api/import/demo", json={"kind": "harvest"})
        old = projects / "Untitled project.agrosuite"
        assert _until(lambda: projectlock.lock_path(old).exists())

        assert client.post("/api/project", json={"name": "South block"}).status_code == 200
        new = projects / "South block.agrosuite"
        assert _until(new.exists)
        assert _until(lambda: projectlock.lock_path(new).exists())
        assert not projectlock.lock_path(old).exists(), "the old claim stayed behind"
        assert _claim_of(new)["project"] == "South block"
        assert sorted(p.name for p in projects.glob("*.lock")) == [new.name + ".lock"]


def test_changing_the_folder_moves_the_claim_too(projects, state, quick, tmp_path):
    """The project moves to the new folder; so does the claim on it."""
    with TestClient(server_mod.app) as client:
        client.post("/api/import/demo", json={"kind": "harvest"})
        here = projects / "Untitled project.agrosuite"
        assert _until(lambda: projectlock.lock_path(here).exists())

        elsewhere = tmp_path / "on the desktop"
        answer = client.put("/api/settings", json={"projects_dir": str(elsewhere)})
        assert answer.status_code == 200 and answer.json()["moved"] is True

        there = elsewhere / "Untitled project.agrosuite"
        assert there.exists()
        assert _until(lambda: projectlock.lock_path(there).exists())
        assert not projectlock.lock_path(here).exists()
        assert not list(projects.glob("*")), "something was left in the old folder"


def test_a_project_taken_while_this_window_slept_is_not_written_into(
    projects, state, quick, other_window
):
    """Holding a claim is not proof of still having it.

    A laptop shut for a couple of minutes misses every heartbeat, and the
    window next to it takes the project fairly. When the first one wakes it
    must not write into that file on the strength of what it remembers: it
    takes a file of its own, with everything it holds in it.
    """
    with TestClient(server_mod.app) as client:
        client.post("/api/import/demo", json={"kind": "harvest"})
        mine = projects / "Untitled project.agrosuite"
        assert _until(lambda: client.get("/api/session/autosave").json()["state"] == "saved")
        theirs_now = mine.read_bytes()

        # What the other window leaves behind when it takes the project over.
        _claimed_by(mine, other_window.pid, name="Untitled project")

        assert client.post("/api/project/prices",
                           json={"crop_price": 0.6, "input_cost": 1.1}).status_code == 200
        second = projects / "Untitled project 2.agrosuite"
        assert _until(second.exists), "the woken window saved nothing at all"
        assert mine.read_bytes() == theirs_now, "it wrote into the other window's file"
        assert state.autosave_path == second
        assert _claim_of(second)["pid"] == os.getpid()
        assert _claim_of(mine)["pid"] == other_window.pid


# ==========================================================================
# The claim can never be the thing that breaks
# ==========================================================================

def test_a_damaged_claim_stops_nothing(projects, state, quick):
    """Bytes nobody can parse beside a project must not be the reason
    somebody cannot open their own work."""
    theirs = _their_project(projects, "Readable project")
    projectlock.lock_path(theirs).write_text("{ this is not json", encoding="utf-8")

    assert projectlock.holder_of(theirs) is None
    assert projectlock.busy(theirs) is False

    with TestClient(server_mod.app) as client:
        resumed = client.get("/api/session/recent").json()["resumed"]
        assert resumed and resumed["ok"] is True
        assert _claim_of(theirs)["pid"] == os.getpid(), "the damaged claim was not replaced"


def test_a_claim_from_a_stranger_is_ignored(projects, state):
    """JSON that is not this app's claim is not this app's business, and
    not a reason to refuse a project either."""
    project = projects / "Odd neighbour.agrosuite"
    settings_mod.ensure(projects)
    projectlock.lock_path(project).write_text(
        json.dumps({"app": "something else", "pid": 1}), encoding="utf-8")

    assert projectlock.holder_of(project) is None
    lock = projectlock.hold(project, "Odd neighbour")
    assert lock.recorded and _claim_of(project)["app"] == projectlock.APP_MARKER


def test_a_claim_that_cannot_be_written_does_not_stop_the_save(projects, state, quick):
    """A folder that refuses the claim still gets the project.

    What is lost is the protection, not the work: a file the app cannot
    write beside a project is no reason to stop writing the project.
    """
    settings_mod.ensure(projects)
    target = projects / "Untitled project.agrosuite"
    # A folder where the lock file should be: it cannot be read as a claim,
    # cannot be created and cannot be replaced.
    projectlock.lock_path(target).mkdir(parents=True)

    with TestClient(server_mod.app) as client:
        assert client.post("/api/import/demo", json={"kind": "harvest"}).status_code == 200
        assert _until(target.exists), "the project was not saved"
        assert _until(lambda: client.get("/api/session/autosave").json()["state"] == "saved")
        assert projectlock.held() is not None
        assert projectlock.held().recorded is False, "it claims to hold what it never wrote"


# ==========================================================================
# The heartbeat
# ==========================================================================

def test_the_heartbeat_is_refreshed_while_the_app_runs(projects, state, quick, monkeypatch):
    """Without this every project would free itself a minute after it was
    claimed, which is worse than no claim at all."""
    monkeypatch.setattr(projectlock, "HEARTBEAT_SECONDS", 0.05)
    with TestClient(server_mod.app) as client:
        client.post("/api/import/demo", json={"kind": "harvest"})
        target = projects / "Untitled project.agrosuite"
        assert _until(lambda: projectlock.lock_path(target).exists())
        first = _claim_of(target)["heartbeat"]

        # The stamps are whole seconds, like every other time the app writes.
        assert _until(lambda: _claim_of(target)["heartbeat"] != first, timeout=6.0), \
            "the claim went stale under a running app"
        assert projectlock.holder_of(target).age < projectlock.STALE_AFTER


def test_the_heartbeat_rides_the_writer_rather_than_a_timer_of_its_own():
    """One thread ticking, not two: the writer's loop is where the claim
    says it is still here, so there is one thing to stop on the way out."""
    source = (ROOT / "agrosuite" / "app" / "autosave.py").read_text(encoding="utf-8")
    tick = source[source.index("    def _tick_once(self"):]
    tick = tick[:tick.index("\n    def ")]
    assert "lock_mod.beat()" in tick, "the writer's tick does not refresh the claim"
    assert "threading.Timer" not in source
    assert source.count("threading.Thread(") == 1, "a second timer was started for it"


def test_the_writer_beats_before_it_decides_there_is_nothing_to_do():
    """An app left open on an unchanged project is the commonest state
    there is, and its claim has to hold while it sits there."""
    source = (ROOT / "agrosuite" / "app" / "autosave.py").read_text(encoding="utf-8")
    tick = source[source.index("    def _tick_once(self"):]
    tick = tick[:tick.index("\n    def ")]
    assert tick.index("lock_mod.beat()") < tick.index("if revision == self._handled")


# ==========================================================================
# Asking whether a process is still there
# ==========================================================================

def test_the_windows_answer_is_never_os_kill(monkeypatch):
    """The trap this feature would have walked into.

    CPython maps ``os.kill`` on Windows onto ``TerminateProcess``, so
    ``os.kill(pid, 0)`` there does not ask whether the other window is
    running — it closes it, with exit code 0. The Windows branch must reach
    for ``OpenProcess`` and a wait on the handle, and never for ``os.kill``.
    """
    def never(*args, **kwargs):  # pragma: no cover - the point is that it is not called
        raise AssertionError("os.kill was used to ask a question on Windows")

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(os, "kill", never)
    # No ctypes for Windows on this machine, so the honest answer is "cannot
    # tell", which falls back to the heartbeat. What matters is that asking
    # cost the other window nothing.
    assert projectlock.process_running(4321) in (True, False, None)

    windows = LOCK_SOURCE[LOCK_SOURCE.index("def _running_windows"):]
    windows = windows[:windows.index("\n\n\n")]
    assert "OpenProcess" in windows and "WaitForSingleObject" in windows
    assert "os.kill" not in windows
    assert "PROCESS_ALL_ACCESS" not in windows, "rights that can kill the other window"


def test_a_process_id_that_is_not_a_process_is_not_an_answer():
    """0 and 1 mean 'this process group' and 'everything' to kill(2); they
    are not something to ask about."""
    for pid in (None, 0, -1, 1, "4321"):
        assert projectlock.process_running(pid) is None
    assert projectlock.process_running(os.getpid()) is True


def test_a_claim_on_another_computer_rests_on_its_heartbeat_alone(projects):
    """A projects folder on a shared drive: a pid from another machine says
    nothing about this one, so only the heartbeat can speak."""
    project = projects / "Shared drive.agrosuite"
    settings_mod.ensure(projects)
    _claimed_by(project, os.getpid(), age_seconds=2.0)
    claim = json.loads(projectlock.lock_path(project).read_text(encoding="utf-8"))
    claim["host"] = "the other tractor"
    projectlock.lock_path(project).write_text(json.dumps(claim), encoding="utf-8")

    holder = projectlock.holder_of(project)
    assert holder.running is None, "a pid on another host was asked about here"
    assert holder.live is True and holder.is_me() is False
    assert "AgroSuite on the other tractor" in holder.sentence()

    claim["heartbeat"] = (datetime.now().astimezone()
                          - timedelta(seconds=projectlock.STALE_AFTER + 5)
                          ).isoformat(timespec="seconds")
    projectlock.lock_path(project).write_text(json.dumps(claim), encoding="utf-8")
    assert projectlock.holder_of(project).live is False


def test_a_heartbeat_from_the_future_leaves_the_other_window_alone(projects):
    """A clock that moved is not a claim that has expired."""
    project = projects / "Clock skew.agrosuite"
    settings_mod.ensure(projects)
    _claimed_by(project, os.getpid(), age_seconds=-3600)
    claim = json.loads(projectlock.lock_path(project).read_text(encoding="utf-8"))
    claim["pid"] = os.getppid()
    projectlock.lock_path(project).write_text(json.dumps(claim), encoding="utf-8")

    holder = projectlock.holder_of(project)
    assert holder.age == 0.0 and holder.live is True


# ==========================================================================
# The interface
# ==========================================================================

def test_the_status_line_carries_the_locked_state():
    """The same quiet line that says "Saved 12:04" says who has the file."""
    block = APP_JS[APP_JS.index("  autosaveLine(status) {"):]
    block = block[:block.index("\n  },\n")]
    assert 'status.state === "locked"' in block
    assert "status.message" in block


def test_the_resume_toast_names_the_other_window_and_offers_the_new_project():
    block = APP_JS[APP_JS.index("  announceResume(resumed) {"):]
    block = block[:block.index("\n  },\n")]
    assert "resumed.locked" in block
    assert "another window" in block
    assert "this.newProject(false)" in block
    assert block.index("resumed.locked") < block.index("if (!resumed.ok)"), \
        "a locked project would be reported as a failure to reopen"
