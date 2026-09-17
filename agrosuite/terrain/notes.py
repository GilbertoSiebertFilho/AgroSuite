"""What was done to a file while it was read, kept as facts until someone reads it.

Gridding a file leaves remarks behind — a swath width that had to be
assumed, a cell widened to fit the budget, the smoothing that took the GPS
noise out, readings that turned out to be two fields in one export. They
are provenance: the reader has to have them beside the numbers or the
numbers claim more than they can.

The trouble is that they are *sentences*, and a sentence carries its
numbers inside it. Written once, as the file is gridded, "the surface was
smoothed over 32.8 ft" is stuck in feet: the unit picker moves, every
finding around it is written again from the stored metric numbers, and the
note sits there in the set the file happened to be read in. That is the
one inconsistency the rest of the app does not have.

So a note is not stored as its sentence. It is stored as **the fact it
states** — a key naming the note and the metric quantities it needs — and
the sentence is written at read time, through
:class:`~agrosuite.core.units.Phrase`, in whatever unit set the reader is
working in. :func:`render` is what a report calls to say them again;
nothing is re-read and no stored number moves, exactly as
:func:`agrosuite.terrain.analysis.restate` does for the findings.

A note whose numbers are **counts, shares, CRS names or the name of a unit
the file declared** has nothing in it that moves with the reader, so it is
kept as the finished sentence through :func:`plain` — "668 of the 33 400
records carry an altitude that is not a number" says the same thing to
every reader, and rewriting it would only risk changing it.
"""

from __future__ import annotations

from typing import Any, Callable

from ..core.units import Phrase

#: Writer per note key: ``(fact, Phrase) -> sentence``.
_WRITERS: dict[str, Callable[[dict[str, Any], Phrase], str]] = {}


def _writes(key: str):
    """Register the writer of one note, so a fact and its sentence cannot
    drift apart: the key is declared in exactly one place."""

    def register(writer: Callable[[dict[str, Any], Phrase], str]):
        _WRITERS[key] = writer
        return writer

    return register


def fact(key: str, **quantities: Any) -> dict[str, Any]:
    """One remark as the fact it states: its key and its metric quantities.

    Plain JSON, because it travels in the analysis summary and is saved
    with the project, and a project saved in feet must open in hectares.
    """
    return {"key": key, **quantities}


def plain(text: str) -> dict[str, Any]:
    """A remark with nothing in it to convert, kept as it was written."""
    return {"key": "plain", "text": str(text)}


def render(facts, units: dict[str, Any] | None = None) -> list[str]:
    """The sentences, written in ``units`` — one :class:`Phrase` for all."""
    say = Phrase(units)
    return [_text(dict(item), say) for item in (facts or [])]


def text(item: dict[str, Any], units: dict[str, Any] | None = None) -> str:
    """One note's sentence, written in ``units``."""
    return _text(dict(item), Phrase(units))


def _text(item: dict[str, Any], say: Phrase) -> str:
    key = str(item.get("key") or "")
    writer = _WRITERS.get(key)
    if writer is None:
        # A note this version does not know how to say — a project written
        # by a later one. The caller falls back to the stored sentence
        # rather than dropping a remark about how the file was read.
        raise ValueError(f"Unknown note {key!r}; it cannot be written again.")
    return writer(item, say)


# ==========================================================================
# The notes the grid builder writes
# ==========================================================================

@_writes("plain")
def _plain(item: dict[str, Any], say: Phrase) -> str:
    return str(item.get("text") or "")


@_writes("swath_assumed")
def _swath_assumed(item: dict[str, Any], say: Phrase) -> str:
    return (
        f"The file carries no swath width; {say.length(item['swath_m'], None)} was "
        "assumed for the cell size and the gap tolerance."
    )


@_writes("swath_capped")
def _swath_capped(item: dict[str, Any], say: Phrase) -> str:
    """The declared width is a width of ground like any other — the app is
    holding it in metres, whatever the column said — so it is quoted in the
    reader's unit and the sentence asks them to check the column."""
    return (
        f"The file's swath width reads {say.length(item['declared_m'], None)}, wider "
        f"than any implement; {say.length(item['used_m'], None)} was used for the cell "
        "size and the gap tolerance. Check the swath column's unit."
    )


@_writes("fill_values")
def _fill_values(item: dict[str, Any], say: Phrase) -> str:
    """Zero is zero in every unit, but it is still a height, and a sentence
    that names the reader's unit for it reads beside the ones that do."""
    return (
        f"{int(item['readings'])} readings carried an altitude of exactly "
        f"{say.length(0.0, 0)} or a no-data value while the field lies far from it "
        "(the receiver had no fix there); they were left out."
    )


@_writes("stray_passes")
def _stray_passes(item: dict[str, Any], say: Phrase) -> str:
    count = int(item["passes"])
    shown = ", ".join(
        f"{int(pass_id)} ({say.length(median_m, 0)})" for pass_id, median_m in item["shown"]
    )
    more = f" and {int(item['more'])} more" if item.get("more") else ""
    return (
        f"{count} of the {int(item['of'])} passes {'was' if count == 1 else 'were'} "
        f"logged at an altitude far from the rest of the field (pass {shown}{more}, "
        f"against about {say.length(item['elsewhere_m'], 0)} elsewhere): the receiver "
        f"had lost its fix there. {'Its' if count == 1 else 'Their'} "
        f"{int(item['readings'])} readings were left out rather than blended into the "
        "relief."
    )


@_writes("cell_capped")
def _cell_capped(item: dict[str, Any], say: Phrase) -> str:
    return (
        f"A {say.length(item['requested_m'], None)} cell would make more than "
        f"{say.number(item['max_cells'])} cells; {say.length(item['cell_m'], None)} was "
        "used instead."
    )


@_writes("cell_widened")
def _cell_widened(item: dict[str, Any], say: Phrase) -> str:
    return (
        f"The readings span {say.length(item['span_x_m'], 0)} by "
        f"{say.length(item['span_y_m'], 0)}, so the cell was widened from "
        f"{say.length(item['auto_cell_m'], None)} to {say.length(item['cell_m'], None)} "
        f"to stay within {say.number(item['max_cells'])} cells. Small features are "
        "averaged out at that size."
    )


@_writes("quantised")
def _quantised(item: dict[str, Any], say: Phrase) -> str:
    step = say.length(item["step_m"], None)
    return (
        f"The altitude is recorded in steps of {step} rather than continuously, which "
        f"cuts a gentle slope into terraces; the surface was smoothed over "
        f"{say.length(item['smooth_m'], None)} before reading slopes, and features "
        f"smaller than {step} cannot be told from a single step."
    )


@_writes("smoothing")
def _smoothing(item: dict[str, Any], say: Phrase) -> str:
    smooth_m = float(item["smooth_m"])
    return (
        f"The surface was smoothed over {say.length(smooth_m)} to take out the "
        f"{say.length(item['noise_m'], 2)} of altitude noise in the readings. Hollows "
        f"and bumps narrower than about {say.length(4 * smooth_m)} read shallower than "
        "they are, so small potholes hold more water than the figures say; set the "
        "smoothing to 0 to read the raw surface."
    )


@_writes("spacing")
def _spacing(item: dict[str, Any], say: Phrase) -> str:
    return (
        f"The readings are about {say.length(item['spacing_m'], 0)} apart, farther than "
        f"the {say.length(item['cell_m'], 0)} cell ({say.per_area(item['per_ha'])} of "
        "extent): the relief between them is interpolated, not measured."
    )


@_writes("strip")
def _strip(item: dict[str, Any], say: Phrase) -> str:
    return (
        f"The readings lie along a strip only about {say.length(item['width_m'], 0)} "
        "wide, one or two passes rather than a field. The relief across the strip is "
        "not measured, so only the profile along it means much."
    )


@_writes("blocks")
def _blocks(item: dict[str, Any], say: Phrase) -> str:
    return (
        f"The readings form {int(item['blocks'])} separate blocks of ground "
        f"({say.area(item['largest_ha'])} and {say.area(item['second_ha'])}, about "
        f"{say.length(item['apart_m'], 0)} apart): this file seems to hold more than "
        "one field. The relief, fall and features below mix them; open each field on "
        "its own for a clean answer."
    )


# ==========================================================================
# The notes the raster reader writes
# ==========================================================================

@_writes("dem_coarsened")
def _dem_coarsened(item: dict[str, Any], say: Phrase) -> str:
    factor = int(item["factor"])
    return (
        f"The DEM's {say.length(item['cell_in_m'])} cells were averaged {factor} x "
        f"{factor} into {say.length(item['cell_m'])} cells to keep the analysis under "
        f"{say.number(item['max_cells'])} cells."
    )


@_writes("raster_square_cells")
def _raster_square_cells(item: dict[str, Any], say: Phrase) -> str:
    return (
        "The raster is rotated or has rectangular cells; it was resampled onto "
        f"{say.length(item['cell_m'])} square cells."
    )


@_writes("raster_coarsened")
def _raster_coarsened(item: dict[str, Any], say: Phrase) -> str:
    return (
        f"The raster has {say.number(item['native_cells'])} cells, more than a relief "
        f"question needs; it was coarsened to {say.length(item['cell_m'])} cells (about "
        f"{say.number(item['out_cells'])}) so the analysis stays responsive."
    )


@_writes("raster_resampled")
def _raster_resampled(item: dict[str, Any], say: Phrase) -> str:
    return (
        f"Resampled from {say.length(item['cell_in_m'], 2)} to "
        f"{say.length(item['cell_m'])} cells."
    )


@_writes("dem_points")
def _dem_points(item: dict[str, Any], say: Phrase) -> str:
    """How much of a raster reached the map, and why that is not a loss.

    Nothing here converts — the quantities are counts of cells — but the
    note is kept as a fact like every other so a layer's remarks are one
    kind of thing, written in one place, rather than a list of sentences
    with one stranger in it.
    """
    stride = int(item.get("stride") or 1)
    shown = say.number(item["shown"])
    total = say.number(item["cells"])
    if stride <= 1:
        return f"One point per raster cell ({total} cells)."
    return (
        f"Every {_ordinal(stride)} cell in each direction is shown as a point "
        f"({shown} of {total} cells); the terrain analyser reads the raster "
        f"itself at full resolution."
    )


def _ordinal(n: int) -> str:
    """'2nd', '3rd', '11th' — the word the note needs for a stride."""
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"
