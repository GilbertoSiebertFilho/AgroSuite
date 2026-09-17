"""The "Print report (PDF)" buttons in the interface, checked against the contract.

The interface is vanilla JavaScript and has no unit tests of its own; what
can be checked here without a browser is the contract it relies on. Every
field the page POSTs must be one the route accepts, every unit key it sends
must be one the report reads, each report must be printed from the dataset
that carries it, and the heading style the buttons sit in must exist. A
rename on either side would otherwise only show up in a browser.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agrosuite import report as report_mod
from agrosuite.app.routes import report as report_routes

STATIC = Path(__file__).resolve().parents[1] / "agrosuite" / "app" / "static"
APP_JS = (STATIC / "app.js").read_text(encoding="utf-8")
STYLE = (STATIC / "style.css").read_text(encoding="utf-8")


def _block(name: str) -> str:
    """The source of one App method, up to its closing brace at column two."""
    match = re.search(rf"^  (?:async )?{name}\(", APP_JS, flags=re.MULTILINE)
    assert match, f"App.{name} is not defined"
    end = APP_JS.index("\n  },\n", match.start())
    return APP_JS[match.start():end]


def test_each_report_panel_carries_a_print_button():
    """One button per report, on the panel that shows it — and each is bound
    where the panel is drawn, or it would be a button that does nothing."""
    heads = {
        "preflightPanel": "btn-print-preflight",
        "renderCleanReport": "btn-print-clean",
        "renderDifmReport": "btn-print-difm",
    }
    for method, button_id in heads.items():
        assert f'"{button_id}")' in _block(method), f"{method} draws no #{button_id}"
    # The first-look panel is drawn by preflightPanel but bound by the Data tab.
    assert '"btn-print-preflight")' in _block("tabDados")
    assert "this.printReport(" in _block("tabDados")
    assert "this.printReport(" in _block("renderCleanReport")
    assert "this.printReport(" in _block("renderDifmReport")
    assert "Print report (PDF)" in _block("printableHead")


def test_the_page_posts_only_fields_the_route_accepts():
    """A key the route ignores would be a silently dropped setting."""
    body = _block("printReport")
    body = body[body.index("const body = {"):body.index("};", body.index("const body = {"))]
    posted = set(re.findall(r"^\s+(\w+):", body, flags=re.MULTILINE))
    accepted = set(report_routes.ReportRequest.model_fields)
    assert posted == accepted, f"posted {sorted(posted)} vs accepted {sorted(accepted)}"
    assert 'this.api("/api/report", { method: "POST", body })' in _block("printReport")
    assert "/api/report" in {route.path for route in report_routes.router.routes}


def test_the_unit_keys_sent_are_the_ones_the_report_reads():
    """The report resolves exactly these keys; one missing would silently
    fall back to the default preset and print bu/ac on a metric screen."""
    body = _block("printReport")
    sent = set(re.findall(r'"(\w+_unit|currency|crop)"', body))
    assert sent == set(report_mod._Units.KEYS), sorted(sent)
    # They are read off the live preference set, not a stale copy.
    assert "Units.get()" in body
    assert "this.state.project?.name" in body, "the project name is the heading"


def test_each_report_is_printed_from_the_dataset_that_carries_it():
    """The cleaning report hangs off the clean copy; the economic report and
    the first look belong to the selected dataset."""
    clean = _block("renderCleanReport")
    assert "this.printReport(result.clean?.id || this.state.selectedId" in clean
    assert "this.printReport(this.state.selectedId" in _block("renderDifmReport")
    assert "this.printReport(this.state.selectedId" in _block("tabDados")


def test_a_deleted_clean_copy_drops_out_of_the_report_cache():
    """The cleaning panel is drawn from a cached result that names the clean
    copy it printed from. Once that copy is removed the cache must go with it,
    or the button would post an id the server no longer has; the dataset list
    is the truth, and every removal ends in refreshing it."""
    assert "this.forgetGoneReports()" in _block("refreshDatasets")
    forget = _block("forgetGoneReports")
    for reference in ("clean?.id", "removed?.id", "delete this.state.reports[key]"):
        assert reference in forget, f"forgetGoneReports does not use {reference}"
    # The remove button on the Data tab refreshes the list, which prunes.
    data_tab = _block("tabDados")
    remove = data_tab[data_tab.index('"btn-remove-dataset"'):]
    assert remove.index('method: "DELETE"') < remove.index("await this.refreshDatasets()")


def test_the_file_opens_in_a_new_tab_and_the_path_is_shown():
    body = _block("printReport")
    assert 'window.open(result.download_url, "_blank")' in body
    # A blocked pop-up must not lose the file: an anchor click needs no permission.
    assert "link.download = result.filename" in body
    assert "result.path" in body and 'this.toast("Report written"' in body


def test_the_heading_row_style_exists():
    """printableHead draws `.panel > .head > h3`; without the rule the heading
    would lose the panel style and the button would drop below it."""
    assert ".panel > .head > h3" in STYLE
    assert ".panel > .head {" in STYLE
    assert '<div class="head"><h3>' in _block("printableHead")
