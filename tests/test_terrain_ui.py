"""The Terrain tab: the relief with a face on it.

The analyser had a complete backend and no interface at all. What these
tests protect is what the interface promises on top of it:

* the tab is wired by its key alone, sits where the workflow expects it, and
  says what it needs and what it gives like every other tab;
* the findings come first and are printed as the analyser wrote them —
  metric numbers and all — while every table, legend and chart around them
  follows the unit set on screen, and the panel says so rather than mixing
  the two in silence;
* the map layers, their legends and the contour labels come from the API
  rather than from a list copied into the front end;
* the two things the tab produces for the rest of the app — a zone dataset
  the monitor can read and a zip for QGIS — actually arrive;
* a file with no usable altitude is answered with the analyser's own
  sentence about what to open instead, and a field with no relief is
  reported as level instead of being contoured into noise.

The static half reads the sources; the browser half drives the app the way
a person does. The map tiles are blocked in the sandbox, which costs the
imagery and nothing else: the relief is served by the app itself.
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

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from agrosuite import report as report_mod  # noqa: E402
from agrosuite.app.session import Entry  # noqa: E402
from agrosuite.core import preflight  # noqa: E402
from agrosuite.core import units as units_mod  # noqa: E402
from agrosuite.formats import raster  # noqa: E402
from agrosuite.terrain import analysis as terrain_analysis  # noqa: E402
from agrosuite.terrain.grid import ElevationGrid  # noqa: E402
from agrosuite.terrain.synthetic import synthetic_terrain  # noqa: E402

STATIC = ROOT / "agrosuite" / "app" / "static"
APP_JS = (STATIC / "app.js").read_text(encoding="utf-8")
MAPVIEW_JS = (STATIC / "mapview.js").read_text(encoding="utf-8")
CHARTS_JS = (STATIC / "charts.js").read_text(encoding="utf-8")
INDEX_HTML = (STATIC / "index.html").read_text(encoding="utf-8")

UNIT_KEYS = ("yield_unit", "input_rate_unit", "area_unit", "length_unit",
             "speed_unit", "currency", "crop")
CANADA = {k: units_mod.UNIT_PRESETS["canada"][k] for k in UNIT_KEYS}
METRIC = {k: units_mod.UNIT_PRESETS["metric"][k] for k in UNIT_KEYS}
STAMP = "2026-09-17 10:42"
PROJECT = {"name": "Home quarter 2026", "goal": "difm", "roles": {}, "prices": {},
           "reviewed": set(), "exported": False}


def _block(name: str) -> str:
    """The source of one App method, up to its closing brace at column two."""
    match = re.search(rf"^  (?:async )?{name}\(", APP_JS, flags=re.MULTILINE)
    assert match, f"App.{name} is not defined"
    return APP_JS[match.start():APP_JS.index("\n  },\n", match.start())]


# ==========================================================================
# Where the tab sits and how it is wired
# ==========================================================================

def test_the_terrain_tab_sits_between_data_and_trial_design():
    tabs = re.findall(r'<button data-tab="(\w+)"[^>]*>([^<]+)</button>', INDEX_HTML)
    labels = [label.strip() for _, label in tabs]
    assert labels.index("Data") < labels.index("Terrain") < labels.index("Trial design")
    # Numbers would promise an order that is not required — of this tab least
    # of all, since the relief is read before or after anything else.
    assert not any(re.match(r"\d", label) for label in labels)


def test_the_tab_is_wired_by_its_key_and_nothing_else():
    """`data-tab` plus one line in the renderers map is the whole wiring —
    nothing counts the tabs."""
    assert 'terrain: () => this.tabTerrain(panel)' in _block("renderTab")
    # And the step the first look proposes for an elevation layer lands here.
    assert 'terrain: "terrain"' in _block("goToStep")


def test_the_relief_demo_is_offered_beside_the_other_demos():
    assert 'id="btn-demo-terrain"' in INDEX_HTML
    assert '"/api/terrain/demo"' in _block("bindImport")


def test_the_layer_list_is_never_copied_into_the_front_end():
    """The picker is built from `/layers`, so a layer added to the analyser
    appears without a change here. Only the two keys the interface has to
    reason about are named: the hillshade, whose toggle greys out on itself,
    and the elevation, which is the layer the map opens on."""
    for key in ("slope_pct", "aspect_deg", "tpi_small", "tpi_large", "twi",
                "curvature_profile", "curvature_plan", "depression_depth", "flow_acc"):
        assert f'"{key}"' not in APP_JS, key
    # 'landform' is written down once, as a kind of zone the analyser cuts —
    # never as a layer the picker offers.
    assert "landform" not in _block("terrainMapPanel")
    assert '"landform"' in _block("terrainZonesPanel")


# ==========================================================================
# The findings first, and the units not mixed
# ==========================================================================

def test_the_findings_are_the_first_panel_and_are_printed_as_they_came():
    tab = _block("tabTerrain")
    panels = tab[tab.index("panel.innerHTML =\n      this.terrainFindingsPanel"):]
    assert panels.index("terrainFindingsPanel") < panels.index("terrainMapPanel")
    findings = _block("terrainFindingsPanel")
    # The sentence goes through escaping and nothing else: no conversion, no
    # rewriting of the numbers written into it.
    assert "this.escape(f.text)" in findings
    assert "Units.convert" not in findings
    # And the levels the analyser uses reach the note styling.
    assert '{ ok: "ok", warning: "warning" }' in findings


def test_the_panel_shows_the_findings_in_the_units_on_screen():
    """The panel used to warn that the sentences were metric while the
    tables beside them were not. They are not metric any more: the server
    writes them in the unit set the reader is in, and a change of units
    asks for them again — so the warning is gone, and what replaces it is
    the refresh that makes it unnecessary."""
    findings = _block("terrainFindingsPanel")
    assert "carry metric numbers" not in findings
    assert 'Units.label.length() === "m"' not in findings
    refresh = _block("refreshPhrasing")
    assert "/api/units/display" in _block("sendDisplayUnits")
    assert "this.sendDisplayUnits()" in refresh
    assert "refreshTerrainPhrasing()" in refresh
    # The relief's own summary is asked for again, not only the comparison.
    terrain = _block("refreshTerrainPhrasing")
    assert "`/api/terrain/${analysis.id}`" in terrain


def test_a_volume_is_the_length_unit_cubed_on_both_sides():
    """There is no volume group in the unit catalogue, so the length factor
    is cubed — in the panel and on the printed page alike."""
    js = _block("terrainVolume")
    assert "factor * factor * factor" in js
    units = report_mod._Units(CANADA)
    # 1 m³ is 35.3147 ft³; the pond's volume is the one number nobody can
    # check by eye, so a factor applied once instead of three times would
    # never be noticed.
    assert units.volume(1.0) == pytest.approx(1 / 0.3048 ** 3, rel=1e-9)
    assert units.volume_unit == "ft³"
    assert report_mod._Units(METRIC).volume(1.0) == pytest.approx(1.0)


# ==========================================================================
# What the map view and the charts had to learn
# ==========================================================================

def test_the_map_view_places_a_raster_under_the_vectors():
    """The relief is an image and the contours are lines over it: in one
    pane the image would cover them, because Leaflet stacks panes whole."""
    assert "function setImage(" in MAPVIEW_JS
    assert "createPane(IMAGE_PANE).style.zIndex = 350" in MAPVIEW_JS
    for name in ("setImageOpacity", "clearImage", "setGeoJson", "clearGeoJson",
                 "setClickThrough"):
        assert f"function {name}(" in MAPVIEW_JS, name
    # Named layers, because one bucket cleared as a whole would take the
    # hills away with the contours.
    assert "named.set(name, layer)" in MAPVIEW_JS


def test_the_profile_chart_leaves_a_gap_where_the_line_left_the_field():
    """Stations off the field come back null, and a straight segment across
    them would invent ground that was never measured."""
    assert "function profile(" in CHARTS_JS
    assert "function rose(" in CHARTS_JS
    assert "run = []" in CHARTS_JS and "flush()" in CHARTS_JS
    assert "marks" in CHARTS_JS  # the slope-class breaks on the histogram


# ==========================================================================
# The printed page
# ==========================================================================

@pytest.fixture(scope="module")
def analysed_entry():
    """The demo field analysed once, as a session entry with its reports."""
    dataset, _truth = synthetic_terrain()
    entry = Entry(id="terrain01", dataset=dataset, label="Terrain demo", origin="demo")
    entry.reports["preflight"] = preflight.run(dataset)
    # Half the default floor, so the demo's shallow hollow is listed and the
    # page has a depression to print.
    entry.reports["terrain"] = terrain_analysis.analyze(
        dataset, {"min_feature_height_m": 0.2}).summary()
    return entry


def _pdf_strings(path: Path) -> str:
    from test_report import _pdf_text

    return _pdf_text(path.read_bytes()).decode("latin-1")


def test_the_report_carries_the_relief_where_its_tab_sits(analysed_entry, tmp_path):
    out = tmp_path / "relief.pdf"
    summary = report_mod.build_pdf(analysed_entry, out, CANADA, STAMP, PROJECT)
    assert summary["sections"] == ["header", "preflight", "terrain", "caveat"]
    # One page: the relief section is a summary, not the tab on paper.
    assert summary["pages"] == 1

    text = _pdf_strings(out)
    for phrase in ("Relief", "Gently undulating", "Falls to the", "south-west",
                   "Slope class", "Hill 1", "Low 1", "Depression 1"):
        assert phrase in text, phrase


def test_the_printed_relief_is_in_the_units_that_were_asked_for(analysed_entry, tmp_path):
    report = analysed_entry.reports["terrain"]
    relief_m = report["elevation"]["relief_m"]

    canadian = tmp_path / "ca.pdf"
    report_mod.build_pdf(analysed_entry, canadian, CANADA, STAMP, PROJECT)
    feet = _pdf_strings(canadian)
    assert f"{relief_m / 0.3048:,.1f}" in feet
    # The pond, in cubic feet — the length factor cubed, not applied once.
    volume_m3 = report["features"]["depressions"][0]["volume_m3"]
    assert f"{volume_m3 / 0.3048 ** 3:,.0f}" in feet
    # reportlab writes a byte outside ASCII as an octal escape inside the
    # string literal, so the superscript three of "ft³" reads as \263.
    assert "ft\\263" in feet
    # The sentences are in feet too, so there is nothing left to warn about:
    # the page used to carry a line saying the prose was metric.
    assert "the tables are in ft and ac" not in feet

    metric = tmp_path / "metric.pdf"
    report_mod.build_pdf(analysed_entry, metric, METRIC, STAMP, PROJECT)
    metres = _pdf_strings(metric)
    assert f"{relief_m:,.1f}" in metres
    assert "the tables are in" not in metres


def test_the_printed_findings_are_the_analysers_own_sentences(analysed_entry, tmp_path):
    """The analyser's words, written in the units the page was asked for.

    Their numbers are inside the sentence, so the page cannot convert them
    — it asks the analyser to say them again from the stored numbers. What
    it must never do is print one sentence in metres beside a table in
    feet, which is what it used to do.
    """
    from agrosuite.terrain import analysis as terrain_analysis

    out = tmp_path / "findings.pdf"
    report_mod.build_pdf(analysed_entry, out, CANADA, STAMP, PROJECT)
    text = _pdf_strings(out)
    stored = analysed_entry.reports["terrain"]
    assert stored["findings"][0]["text"].startswith("Total relief is 10.5 m")
    first = terrain_analysis.restate(stored, CANADA)["findings"][0]["text"]
    assert first.split(":")[0] in text
    assert "34.6 ft" in first and "10.5 m" not in first
    assert "10.5 m" not in text


# ==========================================================================
# In a real browser
# ==========================================================================

sync_api = pytest.importorskip("playwright.sync_api")
CHROMIUM = Path(os.environ.get("AGROSUITE_TEST_CHROMIUM", "/opt/pw-browsers/chromium"))


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    """The app on a free port, stopped when the module is done."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    log = tmp_path_factory.mktemp("terrain-server") / "server.log"
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
def page(browser, served):
    page = browser.new_page(viewport={"width": 1400, "height": 950})
    page.errors = []
    page.on("pageerror", lambda err: page.errors.append(str(err)))
    # "New project" asks before closing loaded datasets, and a dismissed
    # prompt would leave the last test's files in this one's session.
    page.on("dialog", lambda dialog: dialog.accept())
    page.goto(served)
    page.wait_for_selector("#btn-demo-terrain")
    page.evaluate("async () => { await App.newProject(); }")
    page.wait_for_function("() => App.state.datasets.length === 0")
    yield page
    assert page.errors == [], page.errors
    page.close()


def _analyse_the_demo(page):
    page.click("#btn-demo-terrain")
    page.wait_for_function("() => App.state.tab === 'terrain'")
    page.click("#btn-run-terrain")
    page.wait_for_selector("#terrain-layer", timeout=60000)


def test_the_demo_analyses_and_every_layer_carries_its_own_legend(page):
    _analyse_the_demo(page)
    findings = page.locator("#right-panel .panel").first.inner_text()
    assert "gently undulating" in findings.lower()

    keys = page.eval_on_selector_all("#terrain-layer option", "els => els.map(e => e.value)")
    assert len(keys) >= 12 and keys[0] == "elevation"
    for key in keys:
        page.select_option("#terrain-layer", key)
        page.wait_for_function(
            "key => document.querySelector('img.leaflet-image-layer')?.src.includes(key)",
            arg=key)
        legend = page.locator("#terrain-legend").inner_text()
        assert legend.strip(), key
    # The landform legend is the six classes with their colours, not a bar.
    page.select_option("#terrain-layer", "landform")
    page.wait_for_timeout(200)
    assert "Hilltop / ridge" in page.locator("#terrain-legend").inner_text()


def test_the_opacity_and_the_hillshade_reach_the_image(page):
    _analyse_the_demo(page)
    page.eval_on_selector("#terrain-opacity",
                          "el => { el.value = 0.4; el.dispatchEvent(new Event('input')); }")
    assert float(page.eval_on_selector(
        "img.leaflet-image-layer", "el => el.style.opacity")) == pytest.approx(0.4)

    page.check("#terrain-hillshade")
    page.wait_for_function(
        "() => document.querySelector('img.leaflet-image-layer').src.includes('hillshade=1')")
    page.uncheck("#terrain-hillshade")
    page.wait_for_function(
        "() => !document.querySelector('img.leaflet-image-layer').src.includes('hillshade=1')")
    # The blend is the server's, at the strength each layer kind needs: one
    # query parameter, not a second overlay at a guessed opacity.
    assert page.locator(".leaflet-pane img.leaflet-image-layer").count() == 1


def test_the_contours_redraw_at_the_interval_that_was_asked_for(page):
    _analyse_the_demo(page)
    page.check("#terrain-contours")
    page.wait_for_function("() => App.state.terrain.contours?.features.length > 0")
    labels = page.eval_on_selector_all(".leaflet-tooltip", "els => els.map(e => e.textContent)")
    # Every fifth line labels itself where it is drawn, in the length unit.
    assert labels and all(label.endswith(" ft") for label in labels), labels

    page.fill("#terrain-contour-interval", "10")
    page.click("#btn-terrain-contours")
    page.wait_for_function(
        "() => Math.abs(App.state.terrain.contourInterval - 3.048) < 1e-6", timeout=30000)
    assert "10.00 ft" in page.locator("#terrain-contour-status").inner_text()


def test_a_profile_across_the_field_is_drawn_and_ends_on_escape(page):
    _analyse_the_demo(page)
    page.click("#btn-terrain-profile")
    box = page.locator("#map").bounding_box()
    page.mouse.click(box["x"] + box["width"] * 0.45, box["y"] + box["height"] * 0.6)
    page.mouse.click(box["x"] + box["width"] * 0.6, box["y"] + box["height"] * 0.4)
    page.wait_for_selector("#terrain-profile-chart svg", timeout=30000)

    panel = page.locator("#terrain-profile-chart").locator("xpath=..").inner_text()
    assert "Length" in panel and "Fall along the line" in panel
    # A third click extends the same line rather than starting another.
    page.mouse.click(box["x"] + box["width"] * 0.75, box["y"] + box["height"] * 0.5)
    page.wait_for_function("() => App.state.terrain.profile.points.length === 3", timeout=30000)
    page.keyboard.press("Escape")
    page.wait_for_function("() => App.state.terrainDraw === null")


def test_zones_of_every_kind_register_a_dataset_for_the_monitor(page):
    _analyse_the_demo(page)
    before = page.evaluate("() => App.state.datasets.length")
    for index, (by, kind) in enumerate([
        ("landform", "polygons"), ("slope_class", "points"),
        ("elevation_bands", "polygons"), ("wetness", "polygons"),
    ], start=1):
        page.select_option("#terrain-zones-by", by)
        page.select_option("#terrain-zones-kind", kind)
        page.click("#btn-terrain-zones")
        page.wait_for_function(
            "n => App.state.datasets.length === n", arg=before + index, timeout=60000)
        assert page.evaluate("() => App.state.terrain.zones.by") == by
        assert page.locator("#terrain-zones-result .note").inner_text().strip()
    # Zones start with no role: they are an output of the analysis, not a
    # layer of the field, and one filed as the yield map would displace it.
    assert page.evaluate(
        "() => App.state.datasets.filter(d => d.origin === 'terrain_zones')"
        ".every(d => !d.role)")
    # And the way onward is one button: the new layer, on the Export tab.
    page.click('#terrain-zones-result button[data-cta^="export-file"]')
    page.wait_for_selector("#exp-source")
    assert page.evaluate("() => App.state.selected.origin") == "terrain_zones"


def test_the_terrain_export_offers_the_zip_to_download(page):
    _analyse_the_demo(page)
    page.click("#btn-terrain-export")
    page.wait_for_selector("#terrain-export-report a", timeout=120000)
    link = page.locator("#terrain-export-report a")
    assert link.get_attribute("href").startswith("/api/download/")
    files = page.locator("#terrain-export-report").inner_text()
    for name in ("elevation.tif", "landform.tif", "contours.shp", "README.txt"):
        assert name in files, name


def test_a_file_with_no_altitude_says_what_to_open_instead(page):
    """The harvest demo carries no elevation at all. The tab answers with the
    analyser's own sentence rather than a form that cannot be submitted."""
    page.click("#btn-demo-harvest")
    page.wait_for_function("() => App.state.selected !== null")
    page.click('#steps button[data-tab="terrain"]')
    page.wait_for_function(
        "() => App.state.terrainRefusal[App.state.selectedId] !== undefined", timeout=30000)
    panel = page.locator("#right-panel").inner_text()
    assert "no usable elevation" in panel
    assert "GPS altitude" in panel
    # The one button that fixes it, and the demo for someone with no such file.
    assert page.locator('#right-panel button[data-cta="load"]').count() == 1
    assert page.locator('#right-panel button[data-cta="demo-terrain"]').count() == 1


def test_a_field_with_no_relief_is_reported_as_level(page, tmp_path):
    """A dead-flat field: no contours at any interval, and a finding that
    says why instead of a map of the rounding noise."""
    grid = ElevationGrid(np.full((80, 80), 700.0), 500_000.0, 5_730_000.0, 5.0, "EPSG:32612")
    path = raster.write_geotiff(grid, tmp_path / "level.tif", dtype="float64")
    page.evaluate(
        """async (path) => {
             const dataset = await App.api('/api/import/path', {method: 'POST', body: {path}});
             await App.refreshDatasets();
             await App.selectDataset(dataset.id);
             App.goToTab('terrain');
           }""", str(path))
    page.wait_for_selector("#btn-run-terrain")
    page.click("#btn-run-terrain")
    page.wait_for_selector("#terrain-layer", timeout=60000)

    panel = page.locator("#right-panel").inner_text()
    assert "level within the precision of its elevation" in panel
    assert "nowhere in particular" in panel
    assert page.eval_on_selector("#terrain-contours", "el => el.disabled") is True
    assert "nothing to contour" in panel


def test_an_analysis_pushed_out_of_memory_says_so_on_the_map(page, tmp_path):
    """The app holds the arrays of the four most recent analyses; the fifth
    takes this one's away. The summary and the tables live on — they are
    saved with the dataset — and the map says what to press."""
    _analyse_the_demo(page)
    demo_id = page.evaluate("() => App.state.selectedId")
    rows, cols = np.mgrid[0:60, 0:60]
    for i in range(4):
        z = 700.0 + 0.05 * cols + 0.3 * np.sin((cols + 5 * i) / 7.0) * np.cos(rows / 9.0)
        grid = ElevationGrid(z.astype(float), 500_000.0 + 2000 * i, 5_730_000.0, 5.0,
                             "EPSG:32612")
        path = raster.write_geotiff(grid, tmp_path / f"hills_{i}.tif", dtype="float64")
        page.evaluate(
            """async (path) => {
                 const d = await App.api('/api/import/path', {method: 'POST', body: {path}});
                 await App.api('/api/terrain/analyze',
                               {method: 'POST', body: {dataset_id: d.id}});
               }""", str(path))
    page.evaluate("""async (id) => {
                       await App.refreshDatasets();
                       await App.selectDataset(id);
                     }""", demo_id)
    # Nothing tells the interface the arrays are gone: the image already on
    # screen goes on showing until something asks the server for another one.
    # Switching the layer is what a person does next, and it is the request
    # the server can no longer answer.
    page.select_option("#terrain-layer", "slope_pct")
    page.wait_for_function("() => App.state.terrain.layersGone !== null", timeout=60000)
    panel = page.locator("#right-panel").inner_text()
    assert "no longer in memory" in panel
    assert "gently undulating" in panel.lower()      # the findings are unaffected
    assert page.locator("img.leaflet-image-layer").count() == 0

    page.click("#btn-run-terrain")
    page.wait_for_selector("#terrain-layer", timeout=60000)
    assert page.locator("img.leaflet-image-layer").count() == 1


def test_the_numbers_and_the_legend_follow_the_unit_preset(page):
    _analyse_the_demo(page)
    relief_m = page.evaluate("() => App.state.terrain.summary.elevation.relief_m")

    def reading():
        return {
            "field": page.locator("#right-panel .panel").nth(2).inner_text(),
            "legend": page.locator("#terrain-legend").inner_text(),
        }

    canadian = reading()
    assert f"{relief_m / 0.3048:,.1f}" in canadian["field"]
    assert "ft · highest to lowest" in canadian["field"]
    assert "(ft)" in canadian["legend"]

    page.select_option("#unit-preset", "metric")
    page.wait_for_function("() => Units.label.length() === 'm'")
    metric = reading()
    assert f"{relief_m:,.1f}" in metric["field"]
    assert "(m)" in metric["legend"]
    # Metric is the unit the findings already speak, so the panel stops
    # warning about the difference.
    assert "carry metric numbers" not in page.locator("#right-panel").inner_text()

    page.select_option("#unit-preset", "canada")
    page.wait_for_function("() => Units.label.length() === 'ft'")
    assert reading() == canadian


def test_leaving_the_tab_gives_the_map_back_to_the_points(page):
    """The relief is an image the points would cover, so the tab takes them
    off — and puts them back on the way out, rather than leaving another
    tab looking at a raster it never asked for."""
    _analyse_the_demo(page)
    assert page.evaluate("() => App.state.terrainOnMap") is True
    assert page.locator("img.leaflet-image-layer").count() == 1

    page.click('#steps button[data-tab="dados"]')
    page.wait_for_function("() => App.state.terrainOnMap === false")
    assert page.locator("img.leaflet-image-layer").count() == 0
    page.wait_for_function("() => document.getElementById('map-status').textContent"
                           ".includes('points')")
