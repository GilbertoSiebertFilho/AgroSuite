"""Second round of what a drop must get right.

Server side: a drop bigger than Starlette's default part count is parsed,
not uploaded whole and then refused; a loose ``.xml`` one level above a
TASKDATA folder does not import the task a second time; an AppleDouble
``._name.shp`` fork is not the shapefile; a drive letter hidden in a later
path part is refused; system files are counted rather than dropped without
a word; a table named X/Y gets its latitude; a table with no coordinates is
named with the reason rather than the bare column name; and a prose note is
not a table. In the browser: a second drop while the first is still being
enumerated is refused, and the toast says how many system files were left
out. The browser tests are skipped where Playwright or Chromium is missing.
"""

from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fixtures as fx  # noqa: E402

from agrosuite.app import server as server_mod  # noqa: E402
from agrosuite.app import session as session_mod  # noqa: E402
from agrosuite.app.routes import folder_import  # noqa: E402
from agrosuite.core import schema as sch  # noqa: E402
from agrosuite.core.dataset import Dataset  # noqa: E402
from agrosuite.formats import registry  # noqa: E402

ENDPOINT = "/api/import/files"
STATIC = ROOT / "agrosuite" / "app" / "static"
APP_JS = (STATIC / "app.js").read_text(encoding="utf-8")
CHROMIUM = Path(os.environ.get("AGROSUITE_TEST_CHROMIUM", "/opt/pw-browsers/chromium"))


@pytest.fixture
def client(monkeypatch):
    fresh = session_mod.Session()
    monkeypatch.setattr(server_mod, "state", fresh)
    with TestClient(server_mod.app) as client:
        yield client
    fresh.cleanup()


def _drop(client: TestClient, entries: list[tuple[str, bytes]]):
    """POST files as the browser does: one 'files' part per file, plus 'paths'."""
    files = [("files", (Path(relative).name, data)) for relative, data in entries]
    return client.post(ENDPOINT, files=files, data={"paths": [relative for relative, _ in entries]})


def _shapefile_set(shp: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in shp.parent.iterdir() if p.stem == shp.stem}


def _block(name: str) -> str:
    match = re.search(rf"^  (?:async )?{name}\(", APP_JS, flags=re.MULTILINE)
    assert match, f"App.{name} is not defined"
    return APP_JS[match.start():APP_JS.index("\n  },\n", match.start())]


# ==========================================================================
# 1. More files than Starlette parses by default
# ==========================================================================

def test_a_drop_of_1100_files_is_parsed_not_refused(client, tmp_path):
    """Starlette stops at 1,000 parts unless told otherwise; the page lets
    2,000 through, and a drop in between was uploaded whole and then refused."""
    csv = fx.raven_viper_csv(tmp_path / "raven")
    entries = [(f"Field/{csv.name}", csv.read_bytes())]
    entries += [(f"Field/photos/IMG_{i:04d}.jpg", b"\xff\xd8\xff") for i in range(1099)]
    assert len(entries) == 1100 > 1000

    response = _drop(client, entries)
    assert response.status_code == 200, response.text[:300]
    body = response.json()
    assert [item["label"] for item in body["imported"]] == [csv.stem]
    assert len(body["skipped"]) == 1099
    assert all("'.jpg' is not supported" in item["reason"] for item in body["skipped"])


def test_more_than_the_cap_is_refused_with_advice(client):
    entries = [(f"Field/photos/IMG_{i:04d}.jpg", b"\xff\xd8\xff")
               for i in range(folder_import.MAX_FILES + 1)]
    response = _drop(client, entries)
    assert response.status_code == 400, response.text[:300]
    detail = response.json()["detail"]
    assert f"more than {folder_import.MAX_FILES:,} files" in detail
    assert "Drop one field's folder at a time, or zip it." in detail
    assert server_mod.state.list() == []


def test_the_client_cap_matches_the_server_cap():
    assert f"maxFiles = {folder_import.MAX_FILES}," in _block("importFiles")
    # One 'paths' field rides with every file, and the parser counts them apart.
    assert folder_import.MAX_FIELDS >= 2 * folder_import.MAX_FILES


# ==========================================================================
# 2. A loose .xml one level above a TASKDATA folder
# ==========================================================================

def test_a_stray_xml_above_a_taskdata_folder_does_not_import_the_task_twice(client, tmp_path):
    taskdata = fx.isoxml_with_log(tmp_path / "cnh")
    entries = [(f"Field/TASKDATA/{p.name}", p.read_bytes()) for p in taskdata.iterdir()]
    entries.append(("Field/manifest.xml", b"<manifest><export software='AFS'/></manifest>"))

    response = _drop(client, entries)
    assert response.status_code == 200, response.text
    body = response.json()
    assert [(i["label"], i["meta"]["source_format"]) for i in body["imported"]] == [
        ("NW-14-32-W2", "isoxml")]
    assert [i["name"] for i in body["skipped"]] == ["Field/manifest.xml"]
    reason = body["skipped"][0]["reason"]
    assert "'manifest.xml' is not an ISOXML task on its own" in reason
    assert "Field/TASKDATA/TASKDATA.XML" in reason
    assert str(server_mod.state.uploads) not in reason


def test_a_loose_xml_with_no_taskdata_near_it_keeps_the_registry_reason(client, tmp_path):
    csv = fx.trimble_csv(tmp_path / "trimble")
    response = _drop(client, [("Field/config.xml", b"<config/>"), (f"Field/{csv.name}", csv.read_bytes())])
    assert response.status_code == 200, response.text
    body = response.json()
    assert [i["label"] for i in body["imported"]] == [csv.stem]
    assert body["skipped"] == [{"name": "Field/config.xml",
                                "reason": "XML not recognized as ISOXML: config.xml"}]


# ==========================================================================
# 3. AppleDouble forks and __MACOSX
# ==========================================================================

def test_an_appledouble_fork_is_not_the_shapefile(tmp_path):
    shp = fx.john_deere_shapefile(tmp_path / "jd")
    fork = shp.parent / f"._{shp.name}"
    fork.write_bytes(b"\x00\x05\x16\x07" + bytes(60))
    assert fork.name < shp.name, "the fork sorts first, which is how it got picked"

    assert registry.detect(shp.parent).path == shp
    assert registry.read_any(shp.parent).meta.brand == "john_deere"


def test_a_shapefile_under_macosx_is_not_a_source(tmp_path):
    shp = fx.john_deere_shapefile(tmp_path / "__MACOSX")
    # The list of formats the message names grows with the readers; what this
    # test is about is that the folder is refused, not what it offers instead.
    with pytest.raises(ValueError, match="holds neither a TASKDATA.XML"):
        registry.detect(shp.parent)


def test_a_drop_carrying_the_forks_is_still_one_dataset(client, tmp_path):
    shp = fx.john_deere_shapefile(tmp_path / "jd")
    parts = _shapefile_set(shp)
    entries = [(f"Field/{name}", data) for name, data in parts.items()]
    entries += [(f"Field/._{name}", b"\x00\x05\x16\x07" + bytes(60)) for name in parts]
    response = _drop(client, entries)
    assert response.status_code == 200, response.text
    body = response.json()
    assert [i["label"] for i in body["imported"]] == [shp.stem]
    assert body["skipped"] == []
    assert body["ignored"] == len(parts)


# ==========================================================================
# 4. A drive letter in a later part
# ==========================================================================

@pytest.mark.parametrize("raw", ["field/D:evil.csv", "field/D:/evil.csv", "a/b/c:d.csv"])
def test_a_drive_letter_after_the_first_part_is_refused(raw):
    with pytest.raises(ValueError, match="drive letter, and was refused"):
        folder_import.safe_relative_path(raw)


def test_a_drive_letter_in_a_later_part_writes_nothing(client, tmp_path):
    csv = fx.raven_viper_csv(tmp_path / "raven")
    before = set(server_mod.state.uploads.iterdir())
    response = _drop(client, [("field/fine.csv", csv.read_bytes()), ("field/D:evil.csv", csv.read_bytes())])
    assert response.status_code == 400
    assert "refused" in response.json()["detail"]
    assert server_mod.state.list() == []
    assert set(server_mod.state.uploads.iterdir()) == before


# ==========================================================================
# 5. System files are counted
# ==========================================================================

def test_system_files_are_counted_not_listed(client, tmp_path):
    csv = fx.raven_viper_csv(tmp_path / "raven")
    entries = [
        (f"Field/{csv.name}", csv.read_bytes()),
        ("Field/Thumbs.db", bytes(16)),
        ("Field/.DS_Store", bytes(16)),
        ("Field/desktop.ini", b"[.ShellClassInfo]\n"),
        ("Field/__MACOSX/Field/._log.csv", bytes(16)),
        ("Field/.git/config", b"[core]\n"),
    ]
    response = _drop(client, entries)
    assert response.status_code == 200, response.text
    body = response.json()
    assert [i["label"] for i in body["imported"]] == [csv.stem]
    assert body["skipped"] == [], "a system file is not the user's data, and not a skip"
    assert body["ignored"] == 5


def test_a_drop_of_only_system_files_says_how_many(client):
    response = _drop(client, [(".DS_Store", bytes(8)), ("Thumbs.db", bytes(8))])
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "The drop holds nothing that can be imported" in detail
    assert "2 system file(s) such as .DS_Store or Thumbs.db ignored" in detail


def test_the_toast_names_the_ignored_count():
    block = _block("importFiles")
    assert "result.ignored" in block
    assert "system file(s) ignored" in block


# ==========================================================================
# 6. A second drop during enumeration
# ==========================================================================

def test_the_drop_is_busy_from_the_moment_it_lands():
    imports = _block("bindImport")
    assert imports.index("let reading = false") < imports.index("const blocked = ()")
    blocked = imports[imports.index("const blocked = ()"):imports.index("let depth = 0")]
    assert 'reading || document.querySelector(".busy")' in blocked
    drop = imports[imports.index('document.addEventListener("drop"'):]
    on = drop.index("reading = true")
    assert on < drop.index('dropZone.classList.add("busy")') < drop.index("this.collectDropped(")
    assert drop.index("blocked()") < on, "a blocked drop is refused before it takes the flag"
    off = drop.index("reading = false")
    assert drop.index("} finally {") < off, "cleared whatever the drop did"
    assert 'dropZone.classList.remove("busy")' in drop[off:]


# ==========================================================================
# 7. X/Y columns, and a table with no coordinates
# ==========================================================================

def test_a_table_named_x_y_gets_its_latitude(client):
    rows = "x,y,yield\n" + "".join(
        f"{-105.83 - i * 1e-4:.5f},{50.45 + i * 1e-4:.5f},{50 + i % 7}\n" for i in range(40))
    response = _drop(client, [("Field/xy.csv", rows.encode())])
    assert response.status_code == 200, response.text
    body = response.json()
    assert [i["label"] for i in body["imported"]] == ["xy"]
    assert body["skipped"] == []
    frame = server_mod.state.get(body["imported"][0]["id"]).dataset.df
    assert sch.LON in frame.columns and sch.LAT in frame.columns
    assert abs(float(frame[sch.LAT].iloc[0]) - 50.45) < 1e-6


@pytest.mark.parametrize("header", ["a,b,c", "Longitude,Yield", "Latitude,Yield"])
def test_a_table_without_coordinates_is_named_with_the_reason(client, tmp_path, header):
    csv = fx.trimble_csv(tmp_path / "trimble")
    width = len(header.split(","))
    rows = header + "\n" + "".join(
        ",".join(str(i * k) for k in range(1, width + 1)) + "\n" for i in range(1, 6))
    response = _drop(client, [("Field/prices.csv", rows.encode()), (f"Field/{csv.name}", csv.read_bytes())])
    assert response.status_code == 200, response.text
    body = response.json()
    assert [i["label"] for i in body["imported"]] == [csv.stem]
    assert [i["name"] for i in body["skipped"]] == ["Field/prices.csv"]
    reason = body["skipped"][0]["reason"]
    assert "No coordinates (lon/lat or x/y) found in 'prices.csv'" in reason
    assert reason != "'lat'" and "'lat'" not in reason


def test_bounds_of_half_a_coordinate_pair_is_none():
    import pandas as pd

    dataset = Dataset(pd.DataFrame({sch.LON: [-105.8, -105.9], sch.VALUE: [1.0, 2.0]}))
    assert dataset.bounds() is None
    assert dataset.summary()["bounds"] is None


# ==========================================================================
# 8. A prose note is not a table
# ==========================================================================

@pytest.mark.parametrize("name", ["notes.txt", "README.csv"])
def test_a_prose_note_is_not_a_table(client, tmp_path, name):
    csv = fx.raven_viper_csv(tmp_path / "raven")
    note = b"Rained on the 12th\nHail on the north side\nCombine ran late\n"
    response = _drop(client, [(f"Field/{name}", note), (f"Field/{csv.name}", csv.read_bytes())])
    assert response.status_code == 200, response.text
    body = response.json()
    assert [i["label"] for i in body["imported"]] == [csv.stem]
    assert [i["name"] for i in body["skipped"]] == [f"Field/{name}"]
    assert f"'{name}' is not a table" in body["skipped"][0]["reason"]
    assert all(len(server_mod.state.get(item["id"]).dataset.df) > 0 for item in server_mod.state.list())


def test_an_empty_text_file_is_named_as_empty_not_as_prose(client, tmp_path):
    csv = fx.raven_viper_csv(tmp_path / "raven")
    response = _drop(client, [("Field/empty.txt", b""), (f"Field/{csv.name}", csv.read_bytes())])
    assert response.status_code == 200, response.text
    body = response.json()
    assert [i["name"] for i in body["skipped"]] == ["Field/empty.txt"]
    assert "Empty file" in body["skipped"][0]["reason"]


def test_a_tab_separated_note_is_still_a_table(tmp_path):
    path = tmp_path / "log.txt"
    path.write_text("Longitude\tLatitude\tYield\n-105.8\t50.45\t52\n", encoding="utf-8")
    assert folder_import._looks_tabular(path)
    path.write_text("Field notes\nnothing here\n", encoding="utf-8")
    assert not folder_import._looks_tabular(path)


# ==========================================================================
# In the browser: the guard and the toast
# ==========================================================================

@pytest.fixture(scope="module")
def server(tmp_path_factory):
    """The app on a free port, with its home in a scratch folder so the
    profiles and recent files on this machine are left alone."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    home = tmp_path_factory.mktemp("home")
    log = tmp_path_factory.mktemp("server") / "server.log"
    with log.open("w") as handle:
        proc = subprocess.Popen(
            [sys.executable, "-m", "agrosuite", "--port", str(port), "--no-browser"],
            cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
            env={**os.environ, "AGROSUITE_HOME": str(home)},
        )
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(60):
            try:
                with urllib.request.urlopen(f"{base}/api/health", timeout=1) as response:
                    if response.status == 200:
                        break
            except OSError:
                if proc.poll() is not None:
                    pytest.fail(f"the app did not start:\n{log.read_text()}")
                time.sleep(0.5)
        else:
            pytest.fail(f"the app did not answer on {base}:\n{log.read_text()}")
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture(scope="module")
def browser():
    sync_api = pytest.importorskip("playwright.sync_api")
    with sync_api.sync_playwright() as playwright:
        options = {"executable_path": str(CHROMIUM)} if CHROMIUM.exists() else {}
        try:
            instance = playwright.chromium.launch(**options)
        except Exception as exc:  # no browser on this machine
            pytest.skip(f"Chromium is not available: {exc}")
        yield instance
        instance.close()


@pytest.fixture
def page(browser, server):
    page = browser.new_page(viewport={"width": 1400, "height": 900})
    page.errors = []
    page.on("pageerror", lambda err: page.errors.append(str(err)))
    page.goto(server)
    page.wait_for_selector("#btn-demo-harvest")
    yield page
    assert page.errors == [], page.errors
    page.close()


# A drop dispatched from the page: a DataTransfer built by hand has no entry
# API, so the handler falls back to the plain file list, which is enough to
# reach the server and the toast.
DROP_JS = """
(files) => {
  const transfer = new DataTransfer();
  for (const [name, text] of files) transfer.items.add(new File([text], name));
  document.dispatchEvent(new DragEvent("drop", { dataTransfer: transfer, bubbles: true, cancelable: true }));
}
"""


def _toasts(page):
    return page.evaluate(
        "() => Array.from(document.querySelectorAll('#toasts .toast')).map((t) => "
        "[t.querySelector('.t').textContent, t.querySelector('.m').textContent])")


def test_a_second_drop_while_the_first_is_read_is_refused(page, tmp_path):
    """Reading a big folder's entries takes a while; a drop landing meanwhile
    used to pass the busy check and race the first. The enumeration is
    slowed here so the second drop lands inside it."""
    csv = fx.raven_viper_csv(tmp_path / "raven")
    page.evaluate("""() => {
      const original = App.collectDropped;
      App.collectDropped = (transfer) => new Promise((resolve) =>
        setTimeout(() => resolve(original.call(App, transfer)), 1500));
    }""")
    page.evaluate(DROP_JS, [[csv.name, csv.read_text(encoding="utf-8")]])
    page.wait_for_timeout(300)
    assert page.evaluate("() => document.getElementById('app').classList.contains('busy')")
    page.evaluate(DROP_JS, [["Thumbs.db", "x"]])
    page.wait_for_selector("#toasts .toast")
    titles = [title for title, _ in _toasts(page)]
    assert titles == ["Not now"], titles
    assert "Something is still running" in _toasts(page)[0][1]
    # The first drop runs to its end, and the window is released after it.
    page.wait_for_function("() => document.querySelectorAll('#toasts .toast').length === 2", timeout=8000)
    assert [title for title, _ in _toasts(page)] == ["Not now", "Imported"]
    assert not page.evaluate("() => document.getElementById('app').classList.contains('busy')")
    assert page.evaluate("() => App.state.datasets.map((d) => d.label)") == [csv.stem]


def test_the_toast_counts_the_system_files_left_out(page, tmp_path):
    csv = fx.raven_viper_csv(tmp_path / "raven")
    page.evaluate(DROP_JS, [[csv.name, csv.read_text(encoding="utf-8")],
                            ["Thumbs.db", "x"], [".DS_Store", "x"]])
    page.wait_for_selector("#toasts .toast")
    toasts = _toasts(page)
    assert toasts[0][0] == "Imported", toasts
    assert toasts[0][1].startswith("3 file(s) → 1 dataset(s) (2 system file(s) ignored).")
    assert page.evaluate("() => document.getElementById('drop-report').innerHTML") == ""
