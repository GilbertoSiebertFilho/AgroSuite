"""The compare panes and the hover tooltip, driven in a real browser.

These run the app as the user does — a server on a free port, Chromium
through Playwright — because the behaviour under test lives in the canvas
and in Leaflet: two panes on one scale that pan together, a tooltip that
follows the unit picker, a comparison that ends when its context changes,
an overlay that does not outlive its dataset. None of that is visible to a
test that only reads the source. Skipped where Playwright or a browser is
not installed.
"""

from __future__ import annotations

import json
import os
import re
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

from agrosuite.app.session import MAP_POINT_LIMIT  # noqa: E402
from agrosuite.demo import synthetic_harvest  # noqa: E402

CHROMIUM = Path(os.environ.get("AGROSUITE_TEST_CHROMIUM", "/opt/pw-browsers/chromium"))
COUNT = r"[\d,]+"
COMPARE_STATUS = re.compile(
    rf"^Before: ({COUNT})(?: of ({COUNT}) points \(sampled\)| points)"
    rf" · After: ({COUNT})(?: of ({COUNT}) points \(sampled\)| points)$")


def _as_int(text: str) -> int:
    return int(text.replace(",", ""))


# ------------------------------------------------------------- fixtures ---

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


# -------------------------------------------------------------- helpers ---

def _get(page, base, path):
    return page.request.get(f"{base}{path}").json()


def _post(page, base, path, body):
    return page.request.post(f"{base}{path}", data=json.dumps(body),
                             headers={"content-type": "application/json"}).json()


def _select(page, dataset_id):
    """Select as the dataset list does, and wait until the map is painted."""
    page.evaluate("async (id) => { await App.selectDataset(id); }", dataset_id)
    assert page.evaluate("() => App.state.selectedId") == dataset_id


def _cleaned_demo(page, base):
    """The harvest demo, cleaned from the Cleaning tab: the original stays
    selected with the report on screen. Returns (original, clean, removed)."""
    imported = _post(page, base, "/api/import/demo", {"kind": "harvest"})
    page.evaluate("async () => { await App.refreshDatasets(); }")
    _select(page, imported["id"])
    page.click('button[data-tab="limpeza"]')
    page.click("#btn-run-clean")
    page.wait_for_selector("#btn-compare")
    result = page.evaluate("(id) => App.state.reports[`${id}:clean`]", imported["id"])
    return imported["id"], result["clean"]["id"], result["removed"]["id"]


def _status(page):
    return page.locator("#map-status").text_content().strip()


def _enter_compare(page):
    page.click("#btn-compare")
    page.wait_for_function("() => document.getElementById('map-status').textContent.startsWith('Before:')")
    assert page.evaluate("() => !!App.state.compare")


def _exit_compare(page):
    page.click("#btn-compare")
    page.wait_for_function("() => !document.getElementById('map-status').textContent.startsWith('Before:')")
    assert not page.evaluate("() => !!App.state.compare")


def _hover(page, pane, view, lon, lat):
    """Centre `view` on the point, move the pointer onto it in `pane` and
    return the tooltip text, or None when no tooltip is showing."""
    page.evaluate(f"([lat, lon]) => {view}.instance().setView([lat, lon], 19, {{ animate: false }})",
                  [lat, lon])
    x, y = page.evaluate(
        f"([lat, lon]) => {{ const p = {view}.instance().latLngToContainerPoint([lat, lon]);"
        f" return [p.x, p.y]; }}", [lat, lon])
    box = page.locator(pane).bounding_box()
    # Two moves: the second is what lands on the point, the first makes sure
    # a mousemove fires even when the pointer was already there.
    page.mouse.move(box["x"] + x + 40, box["y"] + y + 40)
    page.mouse.move(box["x"] + x, box["y"] + y)
    page.wait_for_timeout(250)
    tip = page.locator(f"{pane} .map-tooltip")
    if tip.count() and tip.first.is_visible():
        return tip.first.text_content().strip()
    return None


def _expected_tooltip(page, raw):
    """What the tooltip should say for an internal value, in the units shown."""
    return page.evaluate(
        "(raw) => { const { conv, unit } = Units.forColumn('value', App.state.selected.meta.operation);"
        " return `${Units.num(conv(raw))} ${unit}`; }", raw)


def _pane_counts(page):
    return page.evaluate("() => [document.querySelectorAll('.leaflet-container').length,"
                         " document.querySelectorAll('.map-tooltip').length,"
                         " document.querySelectorAll('.map-caption').length,"
                         " document.querySelectorAll('#map-pane-b').length,"
                         " document.getElementById('map-panes').classList.contains('compare'),"
                         " document.getElementById('color-column').disabled]")


# ---------------------------------------------------------------- tests ---

def test_compare_shows_two_linked_panes_on_one_scale(page, server):
    original, clean, _ = _cleaned_demo(page, server)
    _enter_compare(page)

    assert _pane_counts(page) == [2, 2, 2, 1, True, True]
    assert [c.text_content() for c in page.locator(".map-caption").all()] == ["Before", "After"]
    assert page.locator("#btn-compare").text_content().strip() == "Back to a single map"

    match = COMPARE_STATUS.match(_status(page))
    assert match, _status(page)
    assert match.group(2) is None and match.group(4) is None, "the demo is not sampled"
    assert _as_int(match.group(3)) < _as_int(match.group(1)), "cleaning removes points"

    # One scale for both: the legend shows the original's range, not the
    # clean copy's, which is narrower once the outliers are gone.
    before = _get(page, server, f"/api/datasets/{original}/map?column=value")
    after = _get(page, server, f"/api/datasets/{clean}/map?column=value")
    assert after["scale"] != before["scale"]
    low, high = page.evaluate(
        "(s) => { const { conv } = Units.forColumn('value', App.state.selected.meta.operation);"
        " return [Units.num(conv(s.low)), Units.num(conv(s.high))]; }", before["scale"])
    assert page.locator("#legend-min").text_content() == low
    assert page.locator("#legend-max").text_content() == high

    # Locked together: move either pane and the other follows.
    lat, lon = before["lat"][0], before["lon"][0]
    page.evaluate("([lat, lon]) => MapView.instance().setView([lat, lon], 18, { animate: false })",
                  [lat + 0.001, lon])
    centre_b = page.evaluate("() => { const c = App.state.compare.view.instance().getCenter();"
                             " return [c.lat, c.lng, App.state.compare.view.instance().getZoom()]; }")
    assert centre_b[0] == pytest.approx(lat + 0.001, abs=1e-5)
    assert centre_b[1] == pytest.approx(lon, abs=1e-5)
    assert centre_b[2] == 18
    page.evaluate("([lat, lon]) => App.state.compare.view.instance().setView([lat, lon], 17, { animate: false })",
                  [lat, lon + 0.002])
    centre_a = page.evaluate("() => { const c = MapView.instance().getCenter();"
                             " return [c.lat, c.lng, MapView.instance().getZoom()]; }")
    assert centre_a[0] == pytest.approx(lat, abs=1e-5)
    assert centre_a[1] == pytest.approx(lon + 0.002, abs=1e-5)
    assert centre_a[2] == 17

    _exit_compare(page)
    assert _pane_counts(page) == [1, 1, 0, 0, False, False]
    assert re.fullmatch(rf"{COUNT} points", _status(page))


def test_tooltip_reads_the_value_in_the_chosen_unit(page, server):
    imported = _post(page, server, "/api/import/demo", {"kind": "harvest"})
    page.evaluate("async () => { await App.refreshDatasets(); }")
    _select(page, imported["id"])
    payload = _get(page, server, f"/api/datasets/{imported['id']}/map?column=value")
    i = next(k for k, v in enumerate(payload["values"]) if v is not None)
    lon, lat, raw = payload["lon"][i], payload["lat"][i], payload["values"][i]

    imperial = _hover(page, "#map-pane", "MapView", lon, lat)
    assert imperial == _expected_tooltip(page, raw)
    assert imperial.endswith(" bu/ac"), imperial

    page.select_option("#unit-preset", "metric")
    page.wait_for_function("() => document.getElementById('legend-title').textContent.includes('kg/ha')")
    metric = _hover(page, "#map-pane", "MapView", lon, lat)
    assert metric == _expected_tooltip(page, raw)
    assert metric.endswith(" kg/ha"), metric
    assert metric.split(" ")[0] != imperial.split(" ")[0]

    # Off the points there is no tooltip.
    box = page.locator("#map-pane").bounding_box()
    page.mouse.move(box["x"] + 3, box["y"] + 3)
    page.wait_for_timeout(250)
    assert page.locator("#map-pane .map-tooltip").first.is_hidden()


def test_the_removed_overlay_does_not_outlive_its_dataset(page, server):
    original, clean, removed = _cleaned_demo(page, server)
    page.click("#btn-show-all-removed")
    page.wait_for_function("() => document.getElementById('removed-legend').textContent.includes('removed record')")
    marks = _get(page, server, f"/api/datasets/{removed}/map?group_column=removal_reason")
    lon, lat, reason = marks["lon"][0], marks["lat"][0], marks["groups"][0]
    assert _hover(page, "#map-pane", "MapView", lon, lat) == f"Removed {reason}"

    # The clean copy does not carry the parent's removals.
    _select(page, clean)
    tip = _hover(page, "#map-pane", "MapView", lon, lat)
    assert tip is None or not tip.startswith("Removed"), tip
    assert page.locator("#removed-legend").count() == 0 \
        or page.locator("#removed-legend").text_content().strip() == ""

    # Nor does the 'Before' pane: the comparison shows values on one scale.
    _select(page, original)
    page.click('button[data-tab="limpeza"]')
    page.wait_for_selector("#btn-show-all-removed")
    page.click("#btn-show-all-removed")
    page.wait_for_function("() => document.getElementById('removed-legend').textContent.includes('removed record')")
    _enter_compare(page)
    assert page.locator("#removed-legend").text_content().strip() == ""
    tip = _hover(page, "#map-pane", "MapView", lon, lat)
    assert tip is None or not tip.startswith("Removed"), tip

    # Asking for the overlay while comparing brings the single map back.
    page.click("#btn-show-all-removed")
    page.wait_for_function("() => document.getElementById('removed-legend').textContent.includes('removed record')")
    assert not page.evaluate("() => !!App.state.compare")
    assert _pane_counts(page) == [1, 1, 0, 0, False, False]
    assert _hover(page, "#map-pane", "MapView", lon, lat) == f"Removed {reason}"

    # Another field on the same ground: the dots and their legend are gone.
    trial = _post(page, server, "/api/import/demo", {"kind": "trial"})
    page.evaluate("async () => { await App.refreshDatasets(); }")
    _select(page, trial["id"])
    tip = _hover(page, "#map-pane", "MapView", lon, lat)
    assert tip is None or not tip.startswith("Removed"), tip


def test_the_comparison_ends_when_its_context_changes(page, server):
    original, clean, _ = _cleaned_demo(page, server)

    _enter_compare(page)
    page.click('button[data-tab="dados"]')
    assert not page.evaluate("() => !!App.state.compare")
    assert _pane_counts(page) == [1, 1, 0, 0, False, False]

    page.click('button[data-tab="limpeza"]')
    page.wait_for_selector("#btn-compare")
    _enter_compare(page)
    _select(page, clean)
    assert not page.evaluate("() => !!App.state.compare")
    assert _pane_counts(page) == [1, 1, 0, 0, False, False]
    assert re.fullmatch(rf"{COUNT} points", _status(page))

    _select(page, original)
    page.click('button[data-tab="limpeza"]')
    page.wait_for_selector("#btn-compare")
    _enter_compare(page)
    page.click("#btn-run-clean")
    page.wait_for_function("() => document.getElementById('btn-compare')?.textContent.includes('Compare before')")
    assert not page.evaluate("() => !!App.state.compare")
    assert _pane_counts(page) == [1, 1, 0, 0, False, False]


def test_panes_come_and_go_without_leaking(page, server):
    _cleaned_demo(page, server)
    for _ in range(4):
        _enter_compare(page)
        assert _pane_counts(page) == [2, 2, 2, 1, True, True]
        _exit_compare(page)
        assert _pane_counts(page) == [1, 1, 0, 0, False, False]
    # The main map still answers after every pane it was linked to is gone.
    page.evaluate("() => { window.dispatchEvent(new Event('resize')); MapView.invalidateSize(); }")
    assert page.evaluate("() => MapView.instance().getContainer().id") == "map"


def test_compare_counts_sampled_points_honestly(page, server, tmp_path):
    """Past the map limit the server sends a sample; the status must say
    'N of M (sampled)' on that side, not a sample beside a full count."""
    big = synthetic_harvest(n_passes=200)
    assert len(big.df) > MAP_POINT_LIMIT
    path = tmp_path / "big_harvest.csv"
    big.df.to_csv(path, index=False)
    imported = _post(page, server, "/api/import/path", {"path": str(path)})
    original = imported["imported"][0]["id"] if "imported" in imported else imported["id"]
    cleaned = _post(page, server, f"/api/datasets/{original}/clean", {})
    kept = cleaned["report"]["totals"]["kept"]
    assert kept <= MAP_POINT_LIMIT, "the clean side must be the unsampled one for this check"

    page.evaluate("async () => { await App.refreshDatasets(); }")
    _select(page, original)
    assert re.fullmatch(rf"{COUNT} of {len(big.df):,} points \(sampled\)", _status(page))
    page.click('button[data-tab="limpeza"]')
    page.wait_for_selector("#btn-compare")   # the stored report, restored
    _enter_compare(page)

    match = COMPARE_STATUS.match(_status(page))
    assert match, _status(page)
    assert _as_int(match.group(2)) == len(big.df), "the 'Before' side names the whole dataset"
    assert _as_int(match.group(1)) < len(big.df)
    assert match.group(4) is None and _as_int(match.group(3)) == kept


def test_compare_rolls_back_when_the_clean_copy_is_gone(page, server):
    original, clean, _ = _cleaned_demo(page, server)
    # Removed behind the client's back: another tab, the MCP server.
    assert page.request.delete(f"{server}/api/datasets/{clean}").status == 200
    page.click("#btn-compare")
    page.wait_for_function("() => [...document.querySelectorAll('#toasts .toast')]"
                           ".some((t) => t.textContent.includes('is not loaded'))")
    page.wait_for_function("() => !App.state.compare")
    assert _pane_counts(page) == [1, 1, 0, 0, False, False]
    assert page.locator("#btn-compare").text_content().strip() == "Compare before and after on the map"
    assert re.fullmatch(rf"{COUNT} points", _status(page))

    # Known to the client, the button says what to do instead of building a pane.
    page.evaluate("(id) => { App.state.datasets = App.state.datasets.filter((d) => d.id !== id); }", clean)
    page.evaluate("() => { document.getElementById('toasts').innerHTML = ''; }")
    page.click("#btn-compare")
    page.wait_for_function("() => [...document.querySelectorAll('#toasts .toast')]"
                           ".some((t) => t.textContent.includes('Run the cleaning again'))")
    assert not page.evaluate("() => !!App.state.compare")
    assert _pane_counts(page) == [1, 1, 0, 0, False, False]
