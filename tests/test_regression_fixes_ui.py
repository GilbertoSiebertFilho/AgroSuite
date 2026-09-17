"""A page reload keeps the session, so it should keep a selection too.

Driven in a real browser because the behaviour lives in ``App.init()``: the
server still holds the datasets, the list comes back, and what the map does
next is what the user judges the reload by. Skipped where Playwright or a
browser is not installed.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

sync_api = pytest.importorskip("playwright.sync_api")

CHROMIUM = Path(os.environ.get("AGROSUITE_TEST_CHROMIUM", "/opt/pw-browsers/chromium"))


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    """The app on a free port, stopped when the module is done."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    log = tmp_path_factory.mktemp("server") / "server.log"
    with log.open("w") as handle:
        proc = subprocess.Popen(
            [sys.executable, "-m", "agrosuite", "--port", str(port), "--no-browser"],
            cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
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
    """A fresh page on the app, with uncaught errors collected."""
    page = browser.new_page(viewport={"width": 1400, "height": 900})
    page.errors = []
    page.on("pageerror", lambda err: page.errors.append(str(err)))
    page.goto(server)
    page.wait_for_selector("#btn-demo-harvest")
    yield page
    assert page.errors == [], page.errors
    page.close()


def _post(page, base, path, body):
    return page.request.post(f"{base}{path}", data=json.dumps(body),
                             headers={"content-type": "application/json"}).json()


def test_an_empty_session_reloads_to_nothing_selected(page):
    """Nothing to select is not an error: the empty Data tab stays."""
    page.reload()
    page.wait_for_selector("#btn-demo-harvest")
    page.wait_for_function("() => document.querySelector('#dataset-list .empty')")
    assert page.evaluate("() => App.state.selectedId") is None


def test_a_reload_selects_the_first_dataset(page, server):
    """The list came back after a reload but the map stayed empty and no card
    was selected — the same session looked like lost data."""
    imported = _post(page, server, "/api/import/demo", {"kind": "harvest"})

    page.reload()
    page.wait_for_selector("#btn-demo-harvest")
    page.wait_for_function("() => App.state.selectedId !== null")

    assert page.evaluate("() => App.state.selectedId") == imported["id"]
    page.wait_for_function(
        "() => document.getElementById('map-status').textContent.trim() !== ''")
    assert page.locator('#dataset-list .card[aria-selected="true"]').count() == 1
    assert page.evaluate("() => App.state.selected?.meta?.crop") == "corn"
