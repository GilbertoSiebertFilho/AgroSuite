"""The compare panes and the hover tooltip, checked against their contract.

The interface is vanilla JavaScript; what can be checked here without a
browser is what the page relies on and what regresses silently: the count
the map status writes when the server sampled the points, the one colour
scale the two panes share, the moments the comparison has to end, what a
pane releases when it goes, and the overlay that must not outlive the
dataset it was drawn for. The browser-driven checks are in test_mapux_ui.py.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agrosuite.app.session import MAP_POINT_LIMIT, map_payload
from agrosuite.demo import synthetic_harvest

STATIC = Path(__file__).resolve().parents[1] / "agrosuite" / "app" / "static"
APP_JS = (STATIC / "app.js").read_text(encoding="utf-8")
MAPVIEW_JS = (STATIC / "mapview.js").read_text(encoding="utf-8")
STYLE = (STATIC / "style.css").read_text(encoding="utf-8")


def _block(name: str) -> str:
    """The source of one App method, up to its closing brace at column two."""
    match = re.search(rf"^  (?:async )?{name}\(", APP_JS, flags=re.MULTILINE)
    assert match, f"App.{name} is not defined"
    end = APP_JS.index("\n  },\n", match.start())
    return APP_JS[match.start():end]


def _fn(source: str, name: str) -> str:
    """One function inside createMapView, up to its closing brace at column two."""
    match = re.search(rf"^  function {name}\(", source, flags=re.MULTILINE)
    assert match, f"{name} is not defined"
    end = source.index("\n  }\n", match.start())
    return source[match.start():end]


# ---------------------------------------------------------------- server ---

def test_map_payload_says_when_it_sampled_and_how_many_there_were():
    """`count` is the size of what was sent; `total` the size of the
    dataset. A status line that shows `count` alone reads, past the limit,
    as if cleaning had added points."""
    dataset = synthetic_harvest()
    whole = map_payload(dataset)
    assert whole["sampled"] is False
    assert whole["count"] == whole["total"] == len(whole["lon"])

    sampled = map_payload(dataset, limit=100)
    assert sampled["sampled"] is True
    assert sampled["total"] == whole["total"]
    assert sampled["count"] == len(sampled["lon"]) <= 100
    assert sampled["count"] < sampled["total"]
    assert MAP_POINT_LIMIT > 0


# ------------------------------------------------------------- the page ---

def test_both_map_modes_count_points_the_same_way():
    """The single map says 'N of M points (sampled)'; the compare status
    must say it too, for each side, or the two numbers are not comparable."""
    helper = _block("pointsStatus")
    assert "payload.sampled" in helper and "payload.total" in helper
    assert "(sampled)" in helper
    assert "this.pointsStatus(payload)" in _block("loadMap")
    compare = _block("paintCompare")
    assert "this.pointsStatus(before)" in compare
    assert "this.pointsStatus(after)" in compare
    assert "before.count" not in compare and "after.count" not in compare


def test_the_second_pane_is_painted_on_the_first_pane_scale():
    """The scale comes from the original; the clean copy is pinned to it."""
    compare = _block("paintCompare")
    assert "const scale = MapView.setPoints(before, conv, { unit })" in compare
    assert "c.view.setPoints(after, conv, { unit, scale })" in compare
    assert "this.renderLegend(c.column, unit, scale)" in compare
    # The point layer honours the pinned scale rather than deriving its own.
    set_points = _fn(MAPVIEW_JS, "setPoints")
    assert "options.scale" in set_points
    assert "colorScale = { low: fixed.low, high: fixed.high" in set_points


def test_the_comparison_is_left_when_it_cannot_be_painted():
    """Both panes exist before either map arrives. When one does not come,
    the single map has to come back, or the user is left with an empty
    'After' pane, a disabled picker and a stale status line."""
    paint = _block("paintCompare")
    assert "if (this.state.compare !== c) return;" in paint
    failure = paint[paint.index("if (!before || !after)"):]
    assert "this.exitCompare();" in failure.split("\n    }")[0]
    # And a clean copy the client already knows is gone is refused up front,
    # with the next step named.
    enter = _block("enterCompare")
    assert "this.state.datasets.some((d) => d.id === result.clean.id)" in enter
    assert "Run the cleaning again." in enter
    assert enter.index("this.state.datasets.some") < enter.index("panes.appendChild(paneB)")


def test_the_comparison_ends_when_its_context_changes():
    """Another tab, another dataset, another cleaning run or another session:
    the comparison belongs to one report on one dataset."""
    assert 'this.state.tab !== "limpeza") this.exitCompare()' in _block("renderTab")
    assert "this.exitCompare({ reload: false })" in _block("selectDataset")
    assert "this.exitCompare()" in _block("runClean")
    assert "this.exitCompare({ reload: false })" in _block("resetSessionState")


def test_the_removed_overlay_belongs_to_one_selection():
    """Drawn for one cleaning, the overlay must not stay on the next dataset,
    nor on the 'Before' pane, where the hover would call a kept record
    removed. The overlay and its legend go together."""
    helper = _block("clearRemovedOverlay")
    assert "MapView.clearOverlay()" in helper
    assert 'getElementById("removed-legend")' in helper
    select = _block("selectDataset")
    assert "this.clearRemovedOverlay()" in select
    assert select.index("this.clearRemovedOverlay()") < select.index("this.state.selectedId = id")
    enter = _block("enterCompare")
    assert "this.clearRemovedOverlay()" in enter
    assert enter.index("this.clearRemovedOverlay()") < enter.index("panes.appendChild(paneB)")
    assert "this.clearRemovedOverlay()" in _block("resetSessionState")
    # Drawing the overlay is a single-map act: it leaves the comparison first.
    show = _block("showRemoved")
    assert "if (this.state.compare) this.exitCompare();" in show
    assert show.index("this.exitCompare()") < show.index("this.api(")
    # The clear button uses the same helper: one place to keep in step.
    assert "this.clearRemovedOverlay()" in _block("renderCleanReport")
    assert "MapView.clearOverlay()" not in _block("renderCleanReport")


def test_a_pane_releases_what_it_took():
    """A pane is created and destroyed as often as the button is pressed;
    each one registers outside its container and must unregister."""
    destroy = _fn(MAPVIEW_JS, "destroy")
    assert 'window.removeEventListener("resize", onWindowResize)' in destroy
    assert "resizeObserver?.disconnect()" in destroy
    assert "cancelAnimationFrame(hoverFrame)" in destroy
    assert "map.remove()" in destroy
    # The link between the panes is undone by the function it returned.
    link = MAPVIEW_JS[MAPVIEW_JS.index("function linkMapViews"):]
    assert 'a.instance()?.off("move", ab)' in link and 'b.instance()?.off("move", ba)' in link
    exit_ = _block("exitCompare")
    for step in ("c.unlink()", "c.view.destroy()", 'getElementById("map-pane-b")?.remove()',
                 '"#map-pane > .map-caption")?.remove()', 'classList.remove("compare")',
                 'getElementById("color-column").disabled = false'):
        assert step in exit_, step


def test_the_tooltip_speaks_the_displayed_unit():
    """The tooltip converts with the same function the colours use and
    writes the unit label the legend shows; a removed point names its
    reason instead. Hover looks the point up in a grid, not by scanning."""
    tip = _fn(MAPVIEW_JS, "updateTooltip")
    assert 'value = "Removed"' in tip
    assert "Units.num((data._convert || ((v) => v))(raw))" in tip
    assert "unit = raw == null ? \"\" : unitLabel" in tip
    assert "buildIndex" in _fn(MAPVIEW_JS, "nearest")
    assert "unit: Units.forColumn" not in _block("loadMap")
    assert "MapView.setPoints(payload, conv, { unit })" in _block("loadMap")
    for selector in (".map-tooltip", ".map-panes.compare", ".map-caption"):
        assert selector in STYLE, f"{selector} is not styled"
