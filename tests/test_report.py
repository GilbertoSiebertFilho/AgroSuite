"""The printable report.

What matters in practice: the file is a PDF any viewer opens, it fits on one
page (two at the outside), it carries only the sections the dataset actually
has, and its numbers are in the unit set the user chose — converted with the
same factors as the screen, so the printed optimum is the one they saw.
"""

from __future__ import annotations

import base64
import re
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agrosuite import report as report_mod
from agrosuite.app.session import Entry
from agrosuite.clean import pipeline as clean_pipeline
from agrosuite.core import preflight
from agrosuite.core import units as units_mod
from agrosuite.demo import synthetic_harvest, synthetic_trial
from agrosuite.difm import analysis as difm_analysis

UNIT_KEYS = ("yield_unit", "input_rate_unit", "area_unit", "length_unit",
             "speed_unit", "currency", "crop")
CANADA = {k: units_mod.UNIT_PRESETS["canada"][k] for k in UNIT_KEYS}
METRIC = {k: units_mod.UNIT_PRESETS["metric"][k] for k in UNIT_KEYS}
PROJECT = {"name": "Home quarter 2026", "goal": "difm", "roles": {}, "prices": {},
           "reviewed": set(), "exported": False}
STAMP = "2026-09-17 10:42"


# ==========================================================================
# Reading the PDF back
# ==========================================================================

def _pdf_text(data: bytes) -> bytes:
    """The decoded content streams — the operators a viewer actually draws.

    The report uses the PDF standard fonts, so reportlab writes every string
    literally, ``(Yield \\(bu/ac\\)) Tj``; but it wraps each content stream in
    ASCII85 and, by default, Flate, so the raw bytes contain no readable text.
    Rather than depend on compression being off (a setting the library could
    change without anyone noticing), this undoes whatever filters each stream
    declares and searches the result. It needs no PDF library.
    """
    chunks = []
    for head, body in re.findall(rb"<<([^>]*?)>>\s*stream\r?\n(.*?)endstream", data, re.S):
        raw = body
        if b"ASCII85Decode" in head:
            raw = base64.a85decode(raw.rstrip().removesuffix(b"~>"), ignorechars=b" \t\n\r")
        if b"FlateDecode" in head:
            raw = zlib.decompress(raw)
        # Inside a literal string, parentheses are escaped: "thing\(s\)".
        chunks.append(raw.replace(rb"\(", b"(").replace(rb"\)", b")"))
    return b"\n".join(chunks)


def _page_count(data: bytes) -> int:
    # Page objects carry ``/Type /Page``; the tree carries ``/Type /Pages``.
    return len(re.findall(rb"/Type\s*/Page(?![A-Za-z])", data))


def _check_is_pdf(path: Path) -> bytes:
    data = path.read_bytes()
    assert data.startswith(b"%PDF")
    assert len(data) > 2048
    assert 1 <= _page_count(data) <= 2
    return data


def _money_text(value: float) -> bytes:
    return f"{value:,.2f}".encode()


def _phrase(text: bytes, words: str) -> bool:
    """Whether the words appear in order, wherever the line breaks fall.

    A paragraph is wrapped into one string operator per line, so a phrase
    that runs over a line break reads ``...Data) Tj T* 0 Tw (tab...`` in the
    content stream. Anything that is not a letter, plus the text operators,
    may sit between two words.
    """
    gap = rb"(?:[^A-Za-z]|T[jJd*])*"
    pattern = gap.join(re.escape(word.encode()) for word in words.split())
    return re.search(pattern, text) is not None


# ==========================================================================
# Fixtures: one cleaning and one analysis, run once
# ==========================================================================

@pytest.fixture(scope="module")
def cleaned():
    harvest = synthetic_harvest()
    result = clean_pipeline.run(harvest)
    entry = Entry(
        id="clean01", dataset=result.clean, label="Harvest (demo) · clean", origin="clean",
        reports={"preflight": preflight.run(result.clean), "clean": result.report},
    )
    return entry, result


@pytest.fixture(scope="module")
def analysed():
    trial = synthetic_trial()
    report = difm_analysis.analyze(
        trial, crop_price=0.55, input_cost=1.3, cell_m=20, edge_margin_m=6, zone_column="zone",
    )
    return trial, report


# ==========================================================================
# (a) a cleaned harvest
# ==========================================================================

def test_cleaned_harvest_report_speaks_the_chosen_units(cleaned, tmp_path):
    entry, result = cleaned
    out = tmp_path / "clean.pdf"
    summary = report_mod.build_pdf(entry, out, CANADA, STAMP, PROJECT)

    data = _check_is_pdf(out)
    assert summary["sections"] == ["header", "preflight", "clean", "caveat"]
    assert summary["pages"] == _page_count(data)
    assert summary["path"] == str(out)

    text = _pdf_text(data)
    assert b"bu/ac" in text
    assert STAMP.encode() in text
    assert b"Home quarter 2026" in text

    # The mean after cleaning, converted with canola's test weight the way the
    # screen converts it — not the internal kg/ha figure.
    mean_kg_ha = result.report["statistics"]["after"]["mean"]
    mean_bu_ac = mean_kg_ha / units_mod.unit_factor("rate_mass", "bu/ac", "canola")
    assert f"{mean_bu_ac:,.1f}".encode() in text
    assert f"{mean_kg_ha:,.0f}".encode() not in text

    # Every filter that removed something is named with its count.
    for step in result.report["steps"]:
        if step["removed"]:
            assert step["label"].encode() in text
            assert f"{step['removed']:,}".encode() in text


# ==========================================================================
# (b) a strip trial analysed with zones, through the server's session
# ==========================================================================

@pytest.fixture
def api():
    from agrosuite.app import server as server_mod
    from agrosuite.app.routes import report as routes

    return server_mod, routes


def test_router_is_included_in_the_app(api):
    server_mod, _ = api
    paths = server_mod.app.openapi()["paths"]
    assert set(paths["/api/report"]) == {"post"}


def test_trial_report_through_the_api(api, analysed):
    server_mod, routes = api
    trial, report = analysed
    summary = server_mod._register(trial.copy(), "Trial (demo)", "demo")
    entry = server_mod.state.get(summary["id"])
    entry.reports["difm"] = report

    response = routes.create_report(routes.ReportRequest(
        dataset_id=entry.id, units=CANADA, title="N trial, home quarter",
    ))

    path = Path(response["path"])
    assert path.parent == server_mod.state.exports
    assert path.name == response["filename"]
    assert re.fullmatch(r"report_\d+\.pdf", path.name)
    assert response["sections"] == ["header", "preflight", "difm", "caveat"]
    assert response["dataset_id"] == entry.id
    assert response["units"]["yield_unit"] == "bu/ac"
    assert response["generated_at"]

    # The download link resolves to the file just written.
    token = response["download_url"].rsplit("/", 1)[-1]
    assert response["download_url"].startswith("/api/download/")
    assert server_mod.state.file_for(token) == path

    data = _check_is_pdf(path)
    assert response["pages"] == _page_count(data)
    text = _pdf_text(data)
    assert b"N trial, home quarter" in text
    assert b"lb/ac" in text and b"bu/ac" in text

    # Rates are input rates: the optimum goes out in lb/ac, not kg/ha.
    optimum_kg_ha = report["economics"]["optimum_rate"]
    optimum_lb_ac = optimum_kg_ha / units_mod.unit_factor("rate_mass", "lb/ac")
    assert f"{optimum_lb_ac:,.1f}".encode() in text

    # Money is per hectare inside; on paper it is per acre, in dollars.
    margin_per_ac = report["economics"]["profit_at_optimum"] * units_mod.AREA_TO_HA["ac"]
    assert b"C$ " + _money_text(margin_per_ac) in text

    # The zones, and the decision they support.
    assert b"By zone" in text
    for zone in report["zones"]["by_zone"]:
        assert zone["zone"].encode() in text
    assert b"Variable rate gain" in text
    gain_per_ac = report["zones"]["comparison"]["gain_per_ha"] * units_mod.AREA_TO_HA["ac"]
    assert _money_text(gain_per_ac) in text


def test_reports_are_numbered_and_never_overwritten(api, analysed):
    server_mod, routes = api
    trial, report = analysed
    summary = server_mod._register(trial.copy(), "Trial (demo)", "demo")
    server_mod.state.get(summary["id"]).reports["difm"] = report

    first = routes.create_report(routes.ReportRequest(dataset_id=summary["id"], units=CANADA))
    second = routes.create_report(routes.ReportRequest(dataset_id=summary["id"], units=CANADA))
    assert first["path"] != second["path"]
    assert Path(first["path"]).exists() and Path(second["path"]).exists()
    assert first["download_url"] != second["download_url"]


def test_simultaneous_reports_never_share_a_file(api, analysed):
    """The server runs each request on its own thread, and a PDF reaches the
    disk only once it is built; requests that arrive inside that window must
    still come out as separate files, each holding its own report."""
    server_mod, routes = api
    trial, report = analysed
    summary = server_mod._register(trial.copy(), "Trial (demo)", "demo")
    server_mod.state.get(summary["id"]).reports["difm"] = report

    def print_one(_):
        return routes.create_report(routes.ReportRequest(dataset_id=summary["id"], units=CANADA))

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(print_one, range(8)))

    paths = [Path(r["path"]) for r in results]
    assert len(set(paths)) == len(paths), sorted(p.name for p in paths)
    assert len({r["download_url"] for r in results}) == len(results)
    for response, path in zip(results, paths):
        # Each file is the one its caller was told about: complete, and of
        # the size the response reported.
        _check_is_pdf(path)
        assert path.stat().st_size == response["size_bytes"]
        token = response["download_url"].rsplit("/", 1)[-1]
        assert server_mod.state.file_for(token) == path


def test_a_refused_report_gives_its_number_back(api, analysed):
    """The name is taken the moment it is chosen; a refusal must release it,
    or the folder would collect empty report_<n>.pdf files that no viewer
    opens and every later report would skip past."""
    server_mod, routes = api
    trial, report = analysed
    summary = server_mod._register(trial.copy(), "Trial (demo)", "demo")
    server_mod.state.get(summary["id"]).reports["difm"] = report
    exports = server_mod.state.exports

    before = {p.name for p in exports.glob("report_*.pdf")}
    with pytest.raises(HTTPException):
        routes.create_report(routes.ReportRequest(
            dataset_id=summary["id"], units={**CANADA, "yield_unit": "bushels"},
        ))
    assert {p.name for p in exports.glob("report_*.pdf")} == before
    assert all(p.stat().st_size > 0 for p in exports.glob("report_*.pdf"))

    # The next report takes the number the refusal would otherwise have used.
    number = 1
    while f"report_{number}.pdf" in before:
        number += 1
    written = routes.create_report(routes.ReportRequest(dataset_id=summary["id"], units=CANADA))
    assert written["filename"] == f"report_{number}.pdf"


# ==========================================================================
# (c) a file that has only had its first look
# ==========================================================================

def test_preflight_only_report_has_no_empty_sections(tmp_path):
    small = synthetic_harvest(n_passes=4)
    entry = Entry(id="pre01", dataset=small, label="Small file", origin="import",
                  reports={"preflight": preflight.run(small)})
    out = tmp_path / "preflight.pdf"
    summary = report_mod.build_pdf(entry, out, METRIC, STAMP, PROJECT)

    data = _check_is_pdf(out)
    assert summary["sections"] == ["header", "preflight", "caveat"]
    text = _pdf_text(data)
    assert b"First look" in text
    assert b"Cleaning" not in text
    assert b"Economic report" not in text
    assert b"Small file" in text
    # Area in hectares, records, and the verdict the first look gave.
    assert b" ha" in text or b"(ha)" in text
    assert entry.reports["preflight"]["summary"].encode() in text


def test_printed_first_look_points_at_the_page_not_the_screen(tmp_path):
    """The units finding tells the screen reader that "the button below" applies
    the proposed set; on paper there is no button, so the page must say where
    it is. The proposed unit still has to be named — it is the answer."""
    harvest = synthetic_harvest()
    # A map logged in bu/ac but declared as kg/ha: the mistake the first look
    # exists to catch, and one the demo itself no longer makes.
    harvest.df["value"] /= units_mod.unit_factor("rate_mass", "bu/ac", harvest.meta.crop)
    entry = Entry(id="pre03", dataset=harvest, label="Harvest (demo)", origin="demo",
                  reports={"preflight": preflight.run(harvest)})
    finding = next(f for f in entry.reports["preflight"]["findings"] if f["title"] == "Units look wrong")
    assert "button below" in finding["action"], "the on-screen wording moved; update the test"
    proposed = entry.reports["preflight"]["proposed_units"]["rate"]

    report_mod.build_pdf(entry, tmp_path / "paper.pdf", METRIC, STAMP, PROJECT)
    text = _pdf_text(_check_is_pdf(tmp_path / "paper.pdf"))
    assert b"Units look wrong" in text
    assert b"button" not in text
    assert _phrase(text, f"Apply {proposed}")
    assert _phrase(text, "the first look on the Data tab")


def test_missing_unit_keys_fall_back_to_the_default_preset(tmp_path):
    small = synthetic_harvest(n_passes=4)
    entry = Entry(id="pre02", dataset=small, label="Small file", origin="import",
                  reports={"preflight": preflight.run(small)})
    summary = report_mod.build_pdf(
        entry, tmp_path / "partial.pdf", {"area_unit": "ha", "currency": "EUR"}, STAMP, PROJECT,
    )
    default = units_mod.UNIT_PRESETS[units_mod.DEFAULT_PRESET]
    assert summary["units"]["area_unit"] == "ha"
    assert summary["units"]["currency"] == "EUR"
    assert summary["units"]["yield_unit"] == default["yield_unit"]
    # The dataset knows its crop; that beats the preset's when none was given.
    assert small.meta.crop != default["crop"], "the two must differ for this to prove anything"
    assert summary["units"]["crop"] == small.meta.crop


def test_everything_at_once_still_fits(cleaned, analysed, tmp_path):
    """A clean copy that was then analysed carries all three reports."""
    entry, result = cleaned
    _, report = analysed
    full = Entry(id="all01", dataset=entry.dataset, label=entry.label, origin="clean",
                 reports={**entry.reports, "difm": report})
    summary = report_mod.build_pdf(full, tmp_path / "all.pdf", CANADA, STAMP, PROJECT)
    data = _check_is_pdf(tmp_path / "all.pdf")
    assert summary["sections"] == ["header", "preflight", "clean", "difm", "caveat"]
    assert summary["pages"] <= 2


# ==========================================================================
# Refusals say what to do
# ==========================================================================

def test_unknown_unit_is_refused_with_the_valid_ones(cleaned, tmp_path):
    entry, _ = cleaned
    with pytest.raises(ValueError) as caught:
        report_mod.build_pdf(entry, tmp_path / "bad.pdf", {**CANADA, "yield_unit": "bushels"},
                             STAMP, PROJECT)
    assert "bushels" in str(caught.value)
    assert "bu/ac" in str(caught.value)
    assert not (tmp_path / "bad.pdf").exists()


def test_api_refuses_an_unknown_dataset(api):
    _, routes = api
    with pytest.raises(HTTPException) as caught:
        routes.create_report(routes.ReportRequest(dataset_id="nope", units=CANADA))
    assert caught.value.status_code == 404


def test_api_refuses_a_bad_unit_with_a_message(api, analysed):
    server_mod, routes = api
    trial, report = analysed
    summary = server_mod._register(trial.copy(), "Trial (demo)", "demo")
    server_mod.state.get(summary["id"]).reports["difm"] = report
    with pytest.raises(HTTPException) as caught:
        routes.create_report(routes.ReportRequest(
            dataset_id=summary["id"], units={**CANADA, "area_unit": "acres"},
        ))
    assert caught.value.status_code == 400
    assert "acres" in caught.value.detail and "ac" in caught.value.detail


def test_api_refuses_a_dataset_with_nothing_to_report(api):
    server_mod, routes = api
    # Added straight to the session, so no first look ran.
    entry = server_mod.state.add(synthetic_harvest(n_passes=4), label="Bare", origin="import")
    with pytest.raises(HTTPException) as caught:
        routes.create_report(routes.ReportRequest(dataset_id=entry.id, units=CANADA))
    assert caught.value.status_code == 400
    assert "Bare" in caught.value.detail
