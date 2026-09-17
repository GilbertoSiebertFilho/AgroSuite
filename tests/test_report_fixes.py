"""The printed report: what the second review found, kept from coming back.

Three ways the page could disagree with the screen or with itself: a project
named outside WinAnsi printing as a row of ZapfDingbats squares, a clean
copy's page suggesting "Clean the data" right above its own Cleaning
section, and the on-screen Print button posting the id of a clean copy that
was removed elsewhere and answering with a bare 404.

What can be checked without a browser is checked in Python; the recovery
from the 404 — the list refreshed, the panel redrawn, the next print
succeeding — only shows in a browser, and runs in one where Playwright and
Chromium are installed.
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
from reportlab.pdfbase import pdfmetrics

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from agrosuite import report as report_mod  # noqa: E402
from agrosuite.app.session import Entry  # noqa: E402
from agrosuite.clean import pipeline as clean_pipeline  # noqa: E402
from agrosuite.core import preflight  # noqa: E402
from agrosuite.demo import synthetic_harvest  # noqa: E402
from test_report import CANADA, PROJECT, STAMP, _check_is_pdf, _pdf_text  # noqa: E402

APP_JS = (ROOT / "agrosuite" / "app" / "static" / "app.js").read_text(encoding="utf-8")
CYRILLIC_PROJECT = {**PROJECT, "name": "Домашняя четверть 2026"}
CYRILLIC_LABEL = "Урожай (демо) · чистый"


def _block(name: str) -> str:
    """The source of one App method, up to its closing brace at column two."""
    match = re.search(rf"^  (?:async )?{name}\(", APP_JS, flags=re.MULTILINE)
    assert match, f"App.{name} is not defined"
    end = APP_JS.index("\n  },\n", match.start())
    return APP_JS[match.start():end]


def _base_fonts(data: bytes) -> set[bytes]:
    """The fonts a viewer would load: every /BaseFont the file declares."""
    return set(re.findall(rb"/BaseFont\s*/([A-Za-z0-9+\-]+)", data))


@pytest.fixture(scope="module")
def cleaned():
    """One cleaning, run once: the original with its copy of the report, the
    clean copy, and the removed records, each with its own first look."""
    harvest = synthetic_harvest()
    result = clean_pipeline.run(harvest)
    original = Entry(id="orig", dataset=harvest, label="Harvest (demo)", origin="demo",
                     reports={"preflight": preflight.run(harvest), "clean": result.report})
    clean = Entry(id="clean", dataset=result.clean, label="Harvest (demo) · clean",
                  origin="clean", parent_id="orig",
                  reports={"preflight": preflight.run(result.clean), "clean": result.report})
    removed = Entry(id="removed", dataset=result.removed, label="Harvest (demo) · removed",
                    origin="clean_removed", parent_id="orig",
                    reports={"preflight": preflight.run(result.removed)})
    return original, clean, removed


@pytest.fixture
def fresh_fonts():
    """Let a test register its own fonts, and give the next test the real ones.

    The registration is cached for the process; a test that fakes the disk
    has to clear it, and must not leave its fake behind.
    """
    report_mod._fonts = None
    yield
    report_mod._fonts = None


# ==========================================================================
# 2. A name outside WinAnsi
# ==========================================================================

def test_a_cyrillic_name_prints_in_a_unicode_font(cleaned, tmp_path):
    """The heading, subtitle and footer carry the user's words; with a
    TrueType font registered they are embedded as glyphs, and the squares
    a standard font would substitute never appear."""
    _, clean, _ = cleaned
    entry = Entry(id="cyr", dataset=clean.dataset, label=CYRILLIC_LABEL, origin="clean",
                  reports=clean.reports)
    out = tmp_path / "cyrillic.pdf"
    summary = report_mod.build_pdf(entry, out, CANADA, STAMP, CYRILLIC_PROJECT)
    data = _check_is_pdf(out)
    assert summary["sections"] == ["header", "preflight", "clean", "caveat"]

    regular, bold = report_mod.unicode_fonts()
    fonts = _base_fonts(data)
    if regular == report_mod.FONT:
        pytest.skip("no TrueType font on this machine; the fallback has its own test")
    # A TrueType font is embedded as a subset, "AAAAAA+<face name>".
    for name in {regular, bold}:
        face = pdfmetrics.getFont(name).face.name
        assert any(re.fullmatch(rb"[A-Z]{6}\+" + re.escape(face), f) for f in fonts), \
            (name, face, sorted(fonts))
    # The body stays in the standard font: nothing else on the page moved.
    assert b"Helvetica" in fonts
    # A face that covers Cyrillic leaves nothing for ZapfDingbats to stand in for.
    if ord("Д") in pdfmetrics.getFont(regular).face.charToGlyph:
        assert b"ZapfDingbats" not in fonts, sorted(fonts)


def test_fonts_are_tried_in_the_documented_order(fresh_fonts, monkeypatch, tmp_path):
    """DejaVu before Liberation before Arial, and reportlab's own Vera last —
    it ships with the library, so the last resort is always on the disk."""
    assert [name for name, _, _ in report_mod.UNICODE_FONTS] == \
        ["DejaVuSans", "LiberationSans", "Arial", "Vera"]
    real = report_mod._ttf_index()
    assert "vera.ttf" in real and "verabd.ttf" in real, "reportlab's fonts folder is not searched"

    # Only Vera on the disk, without its bold: the regular stands in for it.
    monkeypatch.setattr(report_mod, "_ttf_index", lambda: {"vera.ttf": real["vera.ttf"]})
    assert report_mod.unicode_fonts() == ("Vera", "Vera")

    # With the bold beside it, the heading gets the bold.
    report_mod._fonts = None
    monkeypatch.setattr(report_mod, "_ttf_index",
                        lambda: {k: real[k] for k in ("vera.ttf", "verabd.ttf")})
    assert report_mod.unicode_fonts() == ("Vera", "Vera-Bold")

    # A font file that does not parse is skipped, silently, for the next one.
    broken = tmp_path / "DejaVuSans.ttf"
    broken.write_bytes(b"not a font")
    report_mod._fonts = None
    monkeypatch.setattr(report_mod, "_ttf_index",
                        lambda: {"dejavusans.ttf": broken, "vera.ttf": real["vera.ttf"]})
    assert report_mod.unicode_fonts() == ("Vera", "Vera")


def test_helvetica_stands_in_when_no_font_loads(fresh_fonts, monkeypatch, cleaned, tmp_path):
    """No TrueType font at all is not an error: the page is still built, in
    the standard font, and the caller is not told."""
    monkeypatch.setattr(report_mod, "_ttf_index", lambda: {})
    assert report_mod.unicode_fonts() == (report_mod.FONT, report_mod.FONT_BOLD)

    _, clean, _ = cleaned
    entry = Entry(id="cyr2", dataset=clean.dataset, label=CYRILLIC_LABEL, origin="clean",
                  reports=clean.reports)
    out = tmp_path / "fallback.pdf"
    summary = report_mod.build_pdf(entry, out, CANADA, STAMP, CYRILLIC_PROJECT)
    fonts = _base_fonts(_check_is_pdf(out))
    assert summary["pages"] >= 1
    assert b"Helvetica" in fonts and b"Helvetica-Bold" in fonts
    assert not any(b"+" in f for f in fonts), sorted(fonts)


def test_the_search_index_is_by_lower_cased_name(monkeypatch, tmp_path):
    """Windows keeps arial.ttf and macOS Arial.ttf; the index answers for both,
    and looks a folder down, where every distribution keeps its fonts."""
    (tmp_path / "Sub").mkdir()
    (tmp_path / "Sub" / "Arial.TTF").write_bytes(b"")
    (tmp_path / "top.ttf").write_bytes(b"")
    monkeypatch.setattr(report_mod.rl_config, "TTFSearchPath",
                        [str(tmp_path), str(tmp_path / "does-not-exist")])
    index = report_mod._ttf_index()
    assert index["arial.ttf"] == tmp_path / "Sub" / "Arial.TTF"
    assert index["top.ttf"] == tmp_path / "top.ttf"


# ==========================================================================
# 3. The first look's next step on a dataset that has been cleaned
# ==========================================================================

def test_a_cleaned_datasets_report_does_not_suggest_cleaning(cleaned, tmp_path):
    """The original that was cleaned and the removed records still carry a
    first look that says "Clean the data"; the clean copy carries its own
    mark and points at the economics instead. On paper the step is done
    either way, and the Cleaning section that answers it sits right below."""
    original, clean, removed = cleaned
    assert original.reports["preflight"]["next_step"]["step"] == "clean"
    assert removed.reports["preflight"]["next_step"]["step"] == "clean"
    assert clean.reports["preflight"]["next_step"]["step"] == "analyse"
    for entry in (original, clean, removed):
        out = tmp_path / f"{entry.id}.pdf"
        report_mod.build_pdf(entry, out, CANADA, STAMP, PROJECT)
        text = _pdf_text(_check_is_pdf(out))
        assert b"Suggested next step" not in text, entry.id
        assert b"Clean the data" not in text, entry.id
        # The rest of the first look is still there.
        assert b"First look" in text
        assert entry.reports["preflight"]["summary"].encode() in text


def test_a_raw_file_still_gets_its_next_step(tmp_path):
    """A file that has only had its first look keeps the suggestion — that is
    where the advice belongs."""
    harvest = synthetic_harvest(n_passes=4)
    entry = Entry(id="raw", dataset=harvest, label="Raw", origin="import",
                  reports={"preflight": preflight.run(harvest)})
    assert entry.reports["preflight"]["next_step"]["step"] == "clean"
    report_mod.build_pdf(entry, tmp_path / "raw.pdf", CANADA, STAMP, PROJECT)
    text = _pdf_text(_check_is_pdf(tmp_path / "raw.pdf"))
    assert b"Suggested next step" in text
    assert b"Clean the data" in text


# ==========================================================================
# 1. The Print button after the clean copy was removed elsewhere
# ==========================================================================

def test_print_recovers_from_a_404_by_refreshing_the_list():
    """A 404 from /api/report means the id the panel carried is gone. The
    button must refresh the list (which prunes the cache), redraw the tab
    and say so — and leave every other failure to the generic handler."""
    body = _block("printReport")
    assert "err.status !== 404" in body
    assert "throw err" in body, "a 400 or a 500 must still reach the generic toast"
    recovery = body[body.index("err.status !== 404"):]
    assert recovery.index("await this.refreshDatasets()") < recovery.index("this.renderTab()")
    assert recovery.index("this.renderTab()") < recovery.index("this.toast(")
    assert '"warn"' in recovery
    assert "return null" in recovery, "a handled 404 must not open a download"
    # When the gone dataset is the selected one there is nothing to redraw
    # for: the selection is cleared the way the remove button clears it.
    assert "this.clearSelection()" in recovery
    assert "this.clearSelection()" in _block("tabDados")
    # The redraw fetches the original's stored report only if the selected
    # detail knows it has one — a flag set after the detail was fetched.
    refresh = _block("refreshDatasets")
    assert "Object.assign(this.state.selected, current)" in refresh
    assert refresh.index("this.forgetGoneReports()") < refresh.index("Object.assign(")


# ------------------------------------------------------------- browser ---

sync_api = pytest.importorskip("playwright.sync_api")

CHROMIUM = Path(os.environ.get("AGROSUITE_TEST_CHROMIUM", "/opt/pw-browsers/chromium"))


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    """The app on a free port, with its own AGROSUITE_HOME."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    folder = tmp_path_factory.mktemp("server")
    log = folder / "server.log"
    with log.open("w") as handle:
        proc = subprocess.Popen(
            [sys.executable, "-m", "agrosuite", "--port", str(port), "--no-browser"],
            cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
            env={**os.environ, "AGROSUITE_HOME": str(folder / "home")},
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
    """A fresh page on the app, with uncaught errors collected."""
    page = browser.new_page(viewport={"width": 1400, "height": 900})
    page.errors = []
    page.on("pageerror", lambda err: page.errors.append(str(err)))
    page.goto(served)
    page.wait_for_selector("#btn-demo-harvest")
    yield page
    assert page.errors == [], page.errors
    page.close()


def _post(page, base, path, body):
    return page.request.post(f"{base}{path}", data=json.dumps(body),
                             headers={"content-type": "application/json"}).json()


def _toasts(page) -> list[str]:
    return page.locator("#toasts .toast .t").all_text_contents()


def test_printing_after_the_clean_copy_was_removed_elsewhere(page, served):
    """The harvest demo cleaned from the Cleaning tab, then its clean copy
    removed straight over HTTP — as the MCP server or another tab would —
    so the page still holds the old id. Print must not end in a bare 404:
    the list is refreshed, the panel redrawn, and the next Print, now from
    the original's own copy of the report, writes the file."""
    imported = _post(page, served, "/api/import/demo", {"kind": "harvest"})
    page.evaluate("async () => { await App.refreshDatasets(); }")
    page.evaluate("async (id) => { await App.selectDataset(id); }", imported["id"])
    page.click('button[data-tab="limpeza"]')
    page.click("#btn-run-clean")
    page.wait_for_selector("#btn-print-clean")
    cached = page.evaluate("(id) => App.state.reports[`${id}:clean`]", imported["id"])
    clean_id = cached["clean"]["id"]

    # Removed behind the page's back: the list on screen still shows it.
    assert page.request.delete(f"{served}/api/datasets/{clean_id}").ok
    assert clean_id in page.evaluate("() => App.state.datasets.map((d) => d.id)")

    # No download must open on the failed attempt, and the recovered one
    # must not need a real browser download to count as written.
    page.evaluate("() => { window.__opened = []; window.open = (url) => { window.__opened.push(url); return true; }; }")
    page.click("#btn-print-clean")
    page.wait_for_selector('#toasts .toast:has-text("The clean copy is gone")')
    titles = _toasts(page)
    assert "That did not work" not in titles, titles
    assert page.evaluate("() => window.__opened") == []

    # The list is the truth again, and the panel was redrawn from the
    # original's copy of the report: the button is back, without the old id.
    assert clean_id not in page.evaluate("() => App.state.datasets.map((d) => d.id)")
    page.wait_for_selector("#btn-print-clean")
    redrawn = page.evaluate("(id) => App.state.reports[`${id}:clean`]", imported["id"])
    assert redrawn["clean"] is None
    assert redrawn["report"]["totals"] == cached["report"]["totals"]

    page.click("#btn-print-clean")
    page.wait_for_selector('#toasts .toast:has-text("Report written")')
    opened = page.evaluate("() => window.__opened")
    assert len(opened) == 1 and opened[0].startswith("/api/download/")
    assert page.request.get(f"{served}{opened[0]}").ok


def test_printing_after_the_selected_dataset_was_removed_elsewhere(page, served):
    """The first look's Print button posts the selected dataset itself. Gone
    from the session, there is nothing to redraw for: the selection is
    cleared as the remove button clears it, the tab says what it needs and
    offers the button that fixes it, and no error escapes."""
    imported = _post(page, served, "/api/import/demo", {"kind": "harvest"})
    page.evaluate("async () => { await App.refreshDatasets(); }")
    page.evaluate("async (id) => { await App.selectDataset(id); }", imported["id"])
    page.click('button[data-tab="dados"]')
    page.wait_for_selector("#btn-print-preflight")

    assert page.request.delete(f"{served}/api/datasets/{imported['id']}").ok
    page.evaluate("() => { window.__opened = []; window.open = (url) => { window.__opened.push(url); return true; }; }")
    page.click("#btn-print-preflight")
    page.wait_for_selector('#toasts .toast:has-text("That dataset is gone")')
    assert "That did not work" not in _toasts(page)
    assert page.evaluate("() => window.__opened") == []

    assert page.evaluate("() => App.state.selectedId") is None
    assert imported["id"] not in page.evaluate("() => App.state.datasets.map((d) => d.id)")
    # The tab with nothing selected is not silent: it says what is missing —
    # a file to open, or one of the open ones to pick — and carries the one
    # button that fixes it.
    panel = page.locator("#right-panel")
    assert panel.locator("button[data-cta]").count() == 1
    assert re.search(r"Nothing is open yet|none is picked", panel.inner_text())
    assert page.locator("#legend").is_hidden()
