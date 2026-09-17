"""Yield against the relief: a value layer read on the terrain grid.

The relief answers where the water sits and which way the field falls. The
question it raises immediately — *and did it cost me anything?* — needs a
second layer: the yield map. This module takes the points of a value layer,
samples the analysed terrain under each of them, and reports the value by
elevation band, by slope class, by landform and by wet or well-drained
ground, with the correlations that say how much of the season's variation
the relief explains at all.

The name says yield because that is the layer this is asked of nine times
in ten, and because the endpoint, the tab section and the printed section
are all called "Yield against the relief" — a reader looking for that finds
this file. Nothing here is yield-specific: any numeric column of any layer
works, and an as-applied rate read against the relief answers a real
question too (did the machine put more on the hilltops?).

Two layers, or one. A yield map whose own GPS altitude was analysed is the
common case, and then the points being sampled are the points the grid was
built from. A DEM analysed on its own, with the yield map loaded beside it,
is the other, and the two files need not share a metric CRS: the points are
reprojected to the grid's CRS before they are sampled. Nothing else changes
between the two cases, which is why there is one code path.

Everything **stored and returned** here is metric, as everywhere else in
the app — kg/ha, metres, hectares — and the interface converts the
structured numbers at display time. The findings are the exception, and
have to be: a sentence with its numbers written into it cannot be restated
in another unit at display time without re-deciding what it says. So the
reader's unit set travels in, through
:class:`~agrosuite.core.units.Phrase`, and the sentences come out in the
units they will be read in. No unit set means metric.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from ..core import schema as sch
from ..core.dataset import Dataset
from ..core.units import Phrase
from . import landforms as lf
from .analysis import TerrainResult, _codes, _jsonable
from .grid import ElevationGrid

#: Elevation bands the comparison is cut into by default. Eight is enough
#: for a shape to show over a 10 m relief and few enough that each band
#: still holds a tenth of the field: a band with fifty points in it is a
#: claim nobody should make.
DEFAULT_BANDS = 8

#: Bands are equal-count, so the number of them is bounded by the points
#: available. Below two there is nothing to compare; above this a band of a
#: 30 000-point yield map holds a few hundred readings and the chart turns
#: into noise.
MIN_BANDS = 2
MAX_BANDS = 20

#: The fewest matched points a comparison is made from. Below this the
#: means of eight bands are drawn from a handful of readings each, and the
#: usual reason for it is two files that do not cover the same field.
MIN_MATCHED_POINTS = 30

#: An r² under this is quoted as "little": the relief is not the story.
WEAK_R2 = 0.15

#: An r² over this is quoted as most of the variation.
STRONG_R2 = 0.50

#: A band or class mean within this of the field average is quoted as
#: "about the field average" rather than given a direction. Half a percent
#: on a yield map is well inside what the monitor itself can resolve.
SAME_PCT = 2.0

#: Wet ground has to differ from the rest by more than this before the
#: finding says the water cost or gained anything.
WET_GAP_PCT = 3.0

#: The names the wetness rows carry.
WET_KEYS = (
    ("wet", "Likely wet ground"),
    ("dry", "Well-drained ground"),
)


# ==========================================================================
# The comparison
# ==========================================================================

def analyse(
    result: TerrainResult,
    values: Dataset,
    *,
    terrain_id: str = "",
    terrain_label: str = "",
    values_id: str = "",
    values_label: str = "",
    bands: int = DEFAULT_BANDS,
    value_column: str | None = None,
    cleaned: bool = False,
    units: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Read ``values`` against the relief ``result`` describes.

    The elevation bands are **equal-count** rather than equal-width: a
    field's heights are not spread evenly, and equal-width bands put half
    the field in the middle two and thirty readings in the top one, whose
    mean then swings on a single combine pass. Equal counts give every band
    the same weight of evidence; each band carries its own elevation range
    (``from_m`` / ``to_m``) so the reader sees that the top band may be a
    narrow sliver of height and the middle one a wide one, and its
    ``area_ha`` so they see how much ground it actually is.

    ``delta_pct`` is the band's mean against the field mean, signed, in
    percent — the number the reader is after, because "2 740 kg/ha" means
    nothing without "and the field did 2 620".

    ``units`` is the reader's unit set, and it reaches only the findings:
    every number in the structured result stays metric, so nothing stored
    or charted depends on a screen preference.

    Raises ``ValueError`` with the sentence to show when the layer carries
    no value to compare, when it is the zone dataset this analyser itself
    produced, or when too few of its points land on the analysed field.
    """
    grid = result.grid
    bands = _band_count(bands)
    column = _value_column(values, value_column, values_label)
    _refuse_zones(values, values_label)

    raw = pd.to_numeric(values.df[column], errors="coerce").to_numpy(dtype="float64")
    x, y = _grid_coordinates(values, grid, values_label)

    elev = grid.sample(x, y)
    on_grid = np.isfinite(elev) & np.isfinite(raw)
    total = int(raw.size)
    matched = int(np.count_nonzero(on_grid))
    if matched < MIN_MATCHED_POINTS:
        raise ValueError(
            f"Only {matched} of {total} points fall on the analysed field; check that "
            "the two files cover the same field."
        )

    value = raw[on_grid]
    elev = elev[on_grid]
    px, py = x[on_grid], y[on_grid]
    slope = grid.sample(px, py, values=result.layers["slope_pct"])
    twi = grid.sample(px, py, values=result.layers["twi"])
    # Class codes take the nearest cell, never an interpolation: half way
    # between a hilltop and a mid slope is not "upper slope", it is one of
    # the two cells the point sits between.
    code = grid.sample(px, py, values=_codes(result.layers["landform"]).astype("float64"), order=0)
    wet = _wet_at(grid, result, px, py)

    mean = float(np.mean(value))
    overall = {
        "mean": mean,
        "median": float(np.median(value)),
        "sd": float(np.std(value, ddof=1)) if value.size > 1 else 0.0,
        "cv_pct": (100.0 * float(np.std(value, ddof=1)) / mean) if value.size > 1 and mean else 0.0,
        "points": int(value.size),
    }

    summary = {
        "terrain": {"dataset_id": terrain_id, "label": terrain_label or result.source_name},
        "values": {
            "dataset_id": values_id,
            "label": values_label or values.meta.name,
            "column": column,
            "value_label": values.meta.value_label,
            "value_unit": values.meta.value_unit,
            "operation": values.meta.operation,
            "crop": values.meta.crop,
            "cleaned": bool(cleaned),
        },
        "points": {"total": total, "matched": matched, "off_grid": total - matched},
        "overall": overall,
        "elevation_bands": _elevation_bands(grid, elev, value, mean, bands),
        "slope_classes": _slope_classes(slope, value, mean),
        "landforms": _landform_classes(code, value, mean),
        "wetness": _wetness(wet, value, mean),
        "relations": _relations(value, elev, slope, twi),
    }
    summary = _jsonable(summary)
    summary["findings"] = findings(summary, units)
    return summary


# ==========================================================================
# Inputs
# ==========================================================================

def _band_count(bands: int) -> int:
    bands = int(bands)
    if not MIN_BANDS <= bands <= MAX_BANDS:
        raise ValueError(
            f"Cut the field into between {MIN_BANDS} and {MAX_BANDS} elevation bands; "
            f"{bands} is outside that. Eight is the default: enough for a shape to "
            "show, few enough that every band still holds a tenth of the field."
        )
    return bands


def _value_column(values: Dataset, asked: str | None, label: str) -> str:
    """The column to compare, or the sentence saying there is none."""
    name = label or values.meta.name
    numeric = values.numeric_columns()
    if asked:
        if asked not in values.df.columns:
            raise ValueError(
                f"'{name}' has no column called '{asked}'. The columns carrying a "
                f"number are: {', '.join(numeric) or 'none'}."
            )
        column = str(asked)
    elif sch.VALUE in values.df.columns:
        column = sch.VALUE
    else:
        raise ValueError(
            f"'{name}' carries no value column to read against the relief. Open the "
            "yield map (or the as-applied log) of this field and pick it here, or "
            "name the column to use."
        )
    finite = pd.to_numeric(values.df[column], errors="coerce").to_numpy(dtype="float64")
    if not np.isfinite(finite).any():
        raise ValueError(
            f"Every value in '{column}' of '{name}' is empty or not a number, so there "
            "is nothing to compare with the relief."
        )
    return column


def _refuse_zones(values: Dataset, label: str) -> None:
    """The zone layer this analyser cut is not a measurement of the field.

    Its ``value`` column holds the zone code — 1, 2, 3 — so a comparison
    would report that the hilltops average class 1, which is true and says
    nothing. The relief cannot be read against a map of itself.
    """
    extra = values.meta.extra or {}
    if not extra.get("zones_by"):
        return
    source = extra.get("terrain_source") or "the relief"
    raise ValueError(
        f"'{label or values.meta.name}' is the terrain zone layer cut from {source!r}: "
        "its values are zone codes, not a measurement, so reading it against the relief "
        "would only report the relief back. Pick the yield map of this field instead."
    )


def _grid_coordinates(values: Dataset, grid: ElevationGrid, label: str) -> tuple[np.ndarray, np.ndarray]:
    """The layer's points in the grid's metric CRS.

    Two files of the same field can sit in different UTM zones — a field on
    a zone boundary, a DEM delivered in one zone and a monitor file
    projected into another — and sampling one grid with the other's
    eastings would place every point tens of kilometres away and report
    that none of them falls on the field. So the coordinates are
    reprojected whenever the two CRSs differ, rather than assumed to match.
    """
    df = values.df
    if sch.X in df.columns and sch.Y in df.columns and values.metric_crs:
        x = df[sch.X].to_numpy(dtype="float64")
        y = df[sch.Y].to_numpy(dtype="float64")
        if str(values.metric_crs) == str(grid.crs):
            return x, y
        from .grid import _transformer

        out_x, out_y = _transformer(str(values.metric_crs), str(grid.crs)).transform(x, y)
        return np.asarray(out_x, dtype="float64"), np.asarray(out_y, dtype="float64")
    if sch.LON in df.columns and sch.LAT in df.columns:
        return grid.from_lonlat(
            df[sch.LON].to_numpy(dtype="float64"), df[sch.LAT].to_numpy(dtype="float64")
        )
    raise ValueError(
        f"'{label or values.meta.name}' carries no positions, so its values cannot be "
        "placed on the field's relief."
    )


def _wet_at(grid: ElevationGrid, result: TerrainResult, x, y) -> np.ndarray | None:
    """True where the point sits on ground the analysis called likely wet.

    ``None`` when the analysis made no wet ranking at all — a level field,
    or one whose slopes are within the reading noise. A column of False
    would read as "none of it is wet", which is a different claim.
    """
    if result.wet is None:
        return None
    sampled = grid.sample(x, y, values=result.wet.astype("float64"), order=0)
    return np.isfinite(sampled) & (sampled > 0.5)


# ==========================================================================
# The groups
# ==========================================================================

def _delta_pct(group_mean: float, field_mean: float) -> float | None:
    if not field_mean or not math.isfinite(field_mean) or not math.isfinite(group_mean):
        return None
    return 100.0 * (group_mean - field_mean) / field_mean


def _stats(value: np.ndarray, mean: float) -> dict[str, Any]:
    """Mean, median, quartiles, spread and the gap to the field average."""
    if value.size == 0:
        return {"points": 0, "mean": None, "median": None, "p25": None, "p75": None,
                "sd": None, "delta_pct": None}
    group_mean = float(np.mean(value))
    return {
        "points": int(value.size),
        "mean": group_mean,
        "median": float(np.median(value)),
        "p25": float(np.percentile(value, 25)),
        "p75": float(np.percentile(value, 75)),
        "sd": float(np.std(value, ddof=1)) if value.size > 1 else 0.0,
        "delta_pct": _delta_pct(group_mean, mean),
    }


def _band_edges(elev: np.ndarray, bands: int) -> np.ndarray:
    """Equal-count edges, with the duplicates a flat field produces removed.

    A field where a third of the readings sit at exactly one height — a
    quantised altitude, a levelled basin — gives two identical quantiles,
    and a band from 700.0 m to 700.0 m would hold every one of them or
    none, depending on which side of the comparison it fell. The duplicate
    edges are dropped and the comparison comes back with fewer bands than
    asked for, which is the honest answer: the field has fewer distinct
    heights than that.
    """
    edges = np.quantile(elev, np.linspace(0.0, 1.0, bands + 1))
    edges = np.unique(edges)
    if edges.size < 2:
        # Every reading at one height: one band covering it, so the rest of
        # the comparison still has something to stand on.
        return np.array([edges[0], np.nextafter(edges[0], np.inf)])
    return edges


def _elevation_bands(
    grid: ElevationGrid, elev: np.ndarray, value: np.ndarray, mean: float, bands: int
) -> list[dict[str, Any]]:
    """The value by height, in equal-count bands, with the ground each holds."""
    edges = _band_edges(elev, bands)
    cells = grid.z[grid.mask]
    cell_ha = grid.cell ** 2 / 10_000.0
    # The quantiles come from the points, and the grid runs a little wider
    # than they do — the edge ring of cells is extrapolated past the last
    # pass. The outer two edges are opened to the grid's own extremes so
    # that the lowest band really is the lowest ground and the band areas
    # add up to the whole field rather than to the swept part of it.
    if cells.size:
        edges[0] = min(edges[0], float(cells.min()))
        edges[-1] = max(edges[-1], float(cells.max()))
    out = []
    for i in range(edges.size - 1):
        lo, hi = float(edges[i]), float(edges[i + 1])
        last = i == edges.size - 2
        in_band = (elev >= lo) & ((elev <= hi) if last else (elev < hi))
        # The area is the field's own ground between the two heights, not
        # the points': a band the combine drove twice would otherwise read
        # as twice the hectares.
        ground = (cells >= lo) & ((cells <= hi) if last else (cells < hi))
        band = {
            "index": i,
            "from_m": lo,
            "to_m": hi,
            "mean_elev_m": float(np.mean(elev[in_band])) if np.any(in_band) else 0.5 * (lo + hi),
            "area_ha": float(np.count_nonzero(ground)) * cell_ha,
        }
        band.update(_stats(value[in_band], mean))
        out.append(band)
    return out


def _slope_classes(slope: np.ndarray, value: np.ndarray, mean: float) -> list[dict[str, Any]]:
    """The value by agronomic slope class; empty classes are listed as empty,
    because "nothing of this field is steeper than 10 %" is a finding."""
    out = []
    for cls in lf.SLOPE_CLASSES:
        lo, hi = cls["from_pct"], cls["to_pct"]
        with np.errstate(invalid="ignore"):
            sel = np.isfinite(slope) & (slope >= lo)
            if hi is not None:
                sel &= slope < hi
        row = {"key": cls["key"], "label": cls["label"],
               "from_pct": cls["from_pct"], "to_pct": cls["to_pct"]}
        stats = _stats(value[sel], mean)
        row.update({"points": stats["points"], "mean": stats["mean"],
                    "delta_pct": stats["delta_pct"]})
        out.append(row)
    return out


def _landform_classes(code: np.ndarray, value: np.ndarray, mean: float) -> list[dict[str, Any]]:
    """The value by landform class, in code order, with the map's colours."""
    codes = np.where(np.isfinite(code), code, 0.0)
    codes = np.rint(codes).astype("int64")
    out = []
    for cls in lf.CLASSES:
        sel = codes == int(cls["code"])
        row = {"code": cls["code"], "key": cls["key"], "label": cls["label"],
               "color": cls["color"]}
        stats = _stats(value[sel], mean)
        row.update({"points": stats["points"], "mean": stats["mean"],
                    "delta_pct": stats["delta_pct"]})
        out.append(row)
    return out


def _wetness(wet: np.ndarray | None, value: np.ndarray, mean: float) -> list[dict[str, Any]]:
    """The wet ground against the rest — two rows, or none at all.

    When the analysis could not rank wet ground (a level field, or slopes
    within the reading noise) there is nothing to split on, and an empty
    list says so; two rows of which one is empty would read as a field with
    no wet ground, which is a claim the analysis did not make.
    """
    if wet is None:
        return []
    out = []
    for key, label in WET_KEYS:
        sel = wet if key == "wet" else ~wet
        row = {"key": key, "label": label}
        stats = _stats(value[sel], mean)
        row.update({"points": stats["points"], "mean": stats["mean"],
                    "delta_pct": stats["delta_pct"]})
        out.append(row)
    return out


# ==========================================================================
# How much the relief explains
# ==========================================================================

def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    """Pearson r over the pairs where both are finite, or None.

    None rather than 0 when one of the two has no spread: a field of one
    constant slope has no correlation with anything, and reporting r = 0
    would claim it was measured and found to be nothing.
    """
    ok = np.isfinite(a) & np.isfinite(b)
    if int(np.count_nonzero(ok)) < 3:
        return None
    a, b = a[ok], b[ok]
    if not (np.std(a) > 0 and np.std(b) > 0):
        return None
    r = float(np.corrcoef(a, b)[0, 1])
    return r if math.isfinite(r) else None


def _relations(value, elev, slope, twi) -> dict[str, Any]:
    """The three correlations, and which of them explains most.

    r² is the share of the season's variation the layer accounts for, and
    it is the number the findings quote: r = 0.4 sounds like a great deal
    and is a sixth of the variation.
    """
    relations = {
        "elevation_r": _pearson(value, elev),
        "slope_r": _pearson(value, slope),
        "twi_r": _pearson(value, twi),
    }
    relations["elevation_r2"] = (
        None if relations["elevation_r"] is None else relations["elevation_r"] ** 2
    )
    candidates = [
        ("elevation", "the elevation", relations["elevation_r"]),
        ("slope", "the slope", relations["slope_r"]),
        ("twi", "the wetness index", relations["twi_r"]),
    ]
    best = max(
        (c for c in candidates if c[2] is not None),
        key=lambda c: abs(c[2]), default=None,
    )
    relations["strongest"] = None if best is None else {
        "key": best[0], "label": best[1], "r": best[2], "r2": best[2] ** 2,
    }
    return relations


# ==========================================================================
# Findings
# ==========================================================================

def findings(summary: dict[str, Any], units: dict[str, Any] | None = None) -> list[dict[str, str]]:
    """The sentences a farmer reads about yield against the relief.

    Same voice and same contract as :func:`analysis.findings`: each entry
    is ``{'level': 'ok' | 'info' | 'warning', 'text': ...}``. Every quantity
    goes through a :class:`~agrosuite.core.units.Phrase` built on ``units``
    — the unit set the reader chose, the shape of
    ``UNIT_PRESETS['canada']`` — so a sentence says "46.4 bu/ac over 149
    ac" to someone working in acres and "2 598 kg/ha over 60.5 ha" to
    someone working in hectares. ``None`` is the metric store, which is
    what a caller that names no units gets.

    The unit set has to reach here rather than the interface converting
    afterwards, because the sentence is the finding: restating it in
    another unit at display time would mean re-deciding what it says.
    """
    out: list[dict[str, str]] = []
    values = summary["values"]
    overall = summary["overall"]
    mean = overall.get("mean")
    operation = values.get("operation")
    say = Phrase(units, crop=values.get("crop"))
    noun = (values.get("value_label") or "value").lower()
    if mean is None:
        return out

    # (1) what the field did, so every percentage below has a base
    out.append(_info(
        f"{values['label']} averages {say.rate(mean, operation)} over the "
        f"{say.number(summary['points']['matched'])} readings that fall on the "
        "analysed field"
        + (f", and varies by {say.percent(overall['cv_pct'])} around that."
           if overall.get("cv_pct") else ".")
    ))

    # (2) the two ends of the relief, which is the question the chart asks
    bands = [b for b in summary["elevation_bands"] if b.get("mean") is not None]
    if len(bands) >= 2:
        low, high = bands[0], bands[-1]
        out.append(_info(
            f"The lowest ground ({say.length(low['from_m'])} to {say.length(low['to_m'])}) "
            f"averaged {say.rate(low['mean'], operation)}, {_against(low['delta_pct'])}; "
            f"the highest ({say.length(high['from_m'])} to {say.length(high['to_m'])}) "
            f"averaged {say.rate(high['mean'], operation)}, {_against(high['delta_pct'])}. "
            f"The two bands cover {say.area(low['area_ha'])} and "
            f"{say.area(high['area_ha'])} of the field."
        ))

    # (3) the landform classes: the best and the worst, and the gap
    ranked = sorted(
        (c for c in summary["landforms"] if c.get("mean") is not None and c["points"] >= 10),
        key=lambda c: c["mean"],
    )
    if len(ranked) >= 2:
        worst, best = ranked[0], ranked[-1]
        gap = best["mean"] - worst["mean"]
        share = 100.0 * gap / mean if mean else 0.0
        out.append(_info(
            f"By landform, {best['label'].lower()} did best at "
            f"{say.rate(best['mean'], operation)} and {worst['label'].lower()} worst at "
            f"{say.rate(worst['mean'], operation)} — a gap of "
            f"{say.rate(gap, operation)}, {say.percent(share)} of the field average."
        ))

    # (4) wet ground: it can cost yield or gain it, and which one matters
    wet = next((w for w in summary["wetness"] if w["key"] == "wet"), None)
    if wet and wet.get("mean") is not None and wet["points"] >= 10:
        delta = wet.get("delta_pct")
        if delta is None or abs(delta) <= WET_GAP_PCT:
            out.append(_info(
                f"The ground the analysis called likely wet yielded "
                f"{say.rate(wet['mean'], operation)}, within a few percent of the rest of "
                "the field: the water neither cost nor gained anything measurable this "
                "season."
            ))
        elif delta < 0:
            out.append(_warning(
                f"The likely wet ground gave up {say.percent(abs(delta))} — "
                f"{say.rate(wet['mean'], operation)} against {say.rate(mean, operation)} "
                "over the field. Water standing there is costing yield, which is what "
                "makes drainage worth pricing."
            ))
        else:
            out.append(_info(
                f"The likely wet ground yielded {say.percent(delta)} ABOVE the field at "
                f"{say.rate(wet['mean'], operation)}. In a dry season the low ground holds "
                "the water the rest of the field wanted; the same ground in a wet year "
                "usually reads the other way."
            ))

    # (5) how much of it the relief explains at all — said honestly
    strongest = summary["relations"].get("strongest")
    if strongest is None:
        out.append(_info(
            "Neither the elevation, the slope nor the wetness index varies enough here "
            f"to be compared with the {noun}: the relief explains nothing that can be "
            "measured."
        ))
    else:
        r2_pct = 100.0 * float(strongest["r2"])
        direction = "lower" if float(strongest["r"]) < 0 else "higher"
        if strongest["r2"] < WEAK_R2:
            out.append(_info(
                f"The relief explains little of this season's variation — "
                f"{say.percent(r2_pct)} of it. {strongest['label'].capitalize()} is the "
                f"strongest of the three ({direction} {noun} where it is higher), and even "
                f"that leaves {say.percent(100.0 - r2_pct)} to soil, weather, management "
                "and the monitor itself."
            ))
        elif strongest["r2"] >= STRONG_R2:
            out.append(_info(
                f"{strongest['label'].capitalize()} accounts for {say.percent(r2_pct)} of "
                f"this season's variation — {direction} {noun} where it is higher. On this "
                "field the relief is most of the story, which is what makes a zone map "
                "cut from it worth applying."
            ))
        else:
            out.append(_info(
                f"{strongest['label'].capitalize()} accounts for {say.percent(r2_pct)} of "
                f"this season's variation — {direction} {noun} where it is higher. Enough "
                "to act on, not enough to explain the field on its own."
            ))

    # (6) an uncleaned layer still carries the overlap and the turns
    if not values.get("cleaned"):
        out.append(_warning(
            "This layer has not been cleaned, so swath overlap and the headland turns are "
            "still in these numbers — and both fall where the machine turns, which is the "
            "edge of the field rather than its hilltops. Clean it first and the comparison "
            "sharpens."
        ))

    # (7) the caveat that applies whatever the numbers said
    out.append(_info(
        "This is one season on one field. A relation between the relief and the "
        f"{noun} is not a cause: the same low ground that holds water also holds the "
        "clay, the organic matter and whatever the last twenty years of management put "
        "there. Two or three seasons agreeing is what turns it into a zone worth "
        "managing apart."
    ))
    return out


# ==========================================================================
# The profile corridor
# ==========================================================================

def profile_values(
    grid: ElevationGrid,
    profile: dict[str, Any],
    values: Dataset,
    value_column: str | None = None,
    values_label: str = "",
    units: dict[str, Any] | None = None,
) -> tuple[list[float | None], dict[str, Any]]:
    """The value along a profile line, averaged in a corridor per station.

    The elevation of a station is read off the grid, which has a value
    everywhere inside the field. The yield has not: it exists only where
    the machine drove, so a station is given the mean of the readings
    within a corridor around it — ``max(2 x cell, half the median swath)``,
    wide enough to catch the pass the line runs beside and narrow enough
    not to average two passes' worth of ground into one station. A station
    with nothing in its corridor comes back ``None``, so the chart breaks
    there rather than drawing a line across ground that was never
    harvested.

    The series itself is metric, like every other number the app stores.
    ``units`` reaches only ``note``, the one sentence in the answer, which
    states the corridor in the unit the reader works in — the panel shows
    it as it comes.
    """
    from scipy.spatial import cKDTree

    column = _value_column(values, value_column, values_label)
    _refuse_zones(values, values_label)
    raw = pd.to_numeric(values.df[column], errors="coerce").to_numpy(dtype="float64")
    x, y = _grid_coordinates(values, grid, values_label)
    ok = np.isfinite(raw) & np.isfinite(x) & np.isfinite(y)

    swath = pd.to_numeric(values.df.get(sch.SWATH), errors="coerce") if sch.SWATH in values.df.columns else None
    median_swath = float(swath[swath > 0].median()) if swath is not None and (swath > 0).any() else 0.0
    corridor = max(2.0 * grid.cell, 0.5 * median_swath)

    stations = _station_coordinates(grid, profile)
    out: list[float | None] = [None] * len(profile.get("distance_m", []))
    used = 0
    if np.any(ok) and stations is not None:
        tree = cKDTree(np.column_stack([x[ok], y[ok]]))
        near = tree.query_ball_point(stations, r=corridor)
        picked = raw[ok]
        seen: set[int] = set()
        for i, idx in enumerate(near):
            if not idx:
                continue
            out[i] = float(np.mean(picked[idx]))
            seen.update(idx)
        used = len(seen)

    say = Phrase(units, crop=values.meta.crop)
    meta = {
        "column": column,
        "value_label": values.meta.value_label,
        "value_unit": values.meta.value_unit,
        "operation": values.meta.operation,
        "crop": values.meta.crop,
        "corridor_m": float(corridor),
        "points_used": int(used),
        "note": (
            f"Each station is the mean of the {values.meta.value_label.lower()} readings "
            f"within {say.length(corridor)} of the line; {say.number(used)} of them were "
            "used. Where none was near, the line breaks."
        ),
    }
    return _jsonable(out), _jsonable(meta)


def _station_coordinates(grid: ElevationGrid, profile: dict[str, Any]) -> np.ndarray | None:
    """The metric position of every station the profile reports.

    The profile answers in distances along the line and lon/lat vertices;
    the stations themselves are re-walked here from those two, the same way
    :func:`contours.profile` walked them, so the corridor is centred on the
    station the chart draws rather than on an interpolation of it.
    """
    distance = np.asarray(profile.get("distance_m") or [], dtype="float64")
    points = np.asarray(profile.get("points") or [], dtype="float64")
    if distance.size == 0 or points.ndim != 2 or points.shape[0] < 2:
        return None
    x, y = grid.from_lonlat(points[:, 0], points[:, 1])
    step = np.hypot(np.diff(x), np.diff(y))
    keep = np.concatenate(([True], step > 0.0))
    x, y = x[keep], y[keep]
    cumulative = np.concatenate(([0.0], np.cumsum(step[step > 0.0])))
    if cumulative[-1] <= 0:
        return None
    return np.column_stack([
        np.interp(distance, cumulative, x),
        np.interp(distance, cumulative, y),
    ])


# ==========================================================================
# Sentence helpers
# ==========================================================================
#
# Every quantity is rendered by :class:`~agrosuite.core.units.Phrase`, which
# carries the reader's unit set; what is left here is the one shape that has
# no unit of its own.

def _info(text: str) -> dict[str, str]:
    return {"level": "info", "text": text}


def _warning(text: str) -> dict[str, str]:
    return {"level": "warning", "text": text}


def _against(delta_pct: float | None) -> str:
    """"12 % above the field average", or "about the field average"."""
    if delta_pct is None:
        return "with no field average to compare against"
    if abs(delta_pct) < SAME_PCT:
        return "about the field average"
    return f"{abs(delta_pct):.0f} % {'above' if delta_pct > 0 else 'below'} the field average"
