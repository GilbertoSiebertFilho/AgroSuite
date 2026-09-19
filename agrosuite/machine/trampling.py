"""The crop under the tyres: how much ground a pass flattened, and what it cost.

A self-propelled sprayer drives through a standing crop on two tyres a side
apart. Where the rear wheels follow the front ones, each pass leaves two
strips a tyre wide, one either side of the machine's centreline, at half the
track width (the tread, centre to centre of the wheels). Where they do not,
each side is two tyres wide.

The strips are drawn from where the machine actually went — the logger's GPS
inside the field — rather than from distance × width: a sprayer that drives
its own tracks again (tramlines, a second product on the same day, a water
test) crushes that ground once, and the drawing counts it once. Distance ×
width, every metre counted as fresh crop, is kept beside it as the upper
bound; the gap between the two is the ground driven more than once.

What the drawing cannot know is how much of the crop under a tyre is lost.
Late in the season — a pre-harvest pass — nearly all of it; early, much of it
stands back up. That share is the user's to set, and it is the one
assumption in the loss; the area is measured.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

#: A tyre narrower or wider than this is a typing error, not a tyre.
MIN_TYRE_M = 0.10
MAX_TYRE_M = 1.20

#: Samples either side averaged into the path before the strips are drawn:
#: an uncorrected GPS wanders by a metre from one half-second to the next,
#: and a strip drawn through every wobble is wider than any tyre.
SMOOTH_SAMPLES = 5

_METRIC = re.compile(
    r"(?<![\d.])(?:[A-Za-z]{1,3}\s*)?(\d{3})\s*/\s*(\d{2,3})\s*(?:[RB-]|\s)\s*-?\s*(\d{2}(?:\.\d)?)")
_IMPERIAL = re.compile(
    r"(?<![\d.])(\d{1,2}(?:\.\d{1,2})?)\s*(?:/\s*\d{2,3}\s*)?(?:[RB]|-)\s*(\d{2})(?![\d.])")
_EXPLICIT = re.compile(r"^\s*(\d+(?:[.,]\d+)?)\s*(mm|cm|m|in|inch|inches|\"|'')\s*$", re.I)

_TO_M = {"mm": 0.001, "cm": 0.01, "m": 1.0, "in": 0.0254, "inch": 0.0254,
         "inches": 0.0254, '"': 0.0254, "''": 0.0254}


def parse_tyre(text: str) -> dict[str, Any]:
    """The section width of a tyre, from what is written on its sidewall.

    Accepts a metric size (``380/90R46``, ``VF 380/90 R46 173D``,
    ``320/85-38``), an imperial one (``18.4R38``, ``13.6-38``,
    ``15.5/80-24``) or a width with its unit (``380 mm``, ``15 in``,
    ``0.38 m``). The width is the first number of a size: millimetres in a
    metric one, inches in an imperial one.

    Returns ``{"width_m", "designation", "system"}``; raises ``ValueError``
    with the forms it accepts when it cannot find a width.
    """
    raw = str(text or "").strip()
    if not raw:
        raise ValueError("Give the tyre size as written on the sidewall, e.g. 380/90R46.")
    found = _EXPLICIT.match(raw)
    if found:
        width = float(found.group(1).replace(",", ".")) * _TO_M[found.group(2).lower()]
        system, designation = "width", raw
    else:
        found = _METRIC.search(raw)
        if found:
            width = int(found.group(1)) / 1000.0
            system = "metric"
            designation = f"{found.group(1)}/{found.group(2)}R{found.group(3)}"
        else:
            found = _IMPERIAL.search(raw)
            if not found:
                raise ValueError(
                    f"'{raw}' does not read as a tyre size. Write it as on the sidewall "
                    "(380/90R46, 18.4R38) or as a width with its unit (380 mm, 15 in)."
                )
            width = float(found.group(1)) * 0.0254
            system = "imperial"
            designation = f"{found.group(1)}R{found.group(2)}"
    if not MIN_TYRE_M <= width <= MAX_TYRE_M:
        raise ValueError(
            f"'{raw}' reads as a tyre {width * 1000:.0f} mm wide, which no sprayer runs on. "
            "Check the size."
        )
    return {"width_m": width, "designation": designation, "system": system}


@dataclass
class TyreSetup:
    """What the strips are drawn from. All metric."""

    tyre_width_m: float
    track_width_m: float
    rear_follows_front: bool = True

    def problems(self) -> list[str]:
        out = []
        if not (MIN_TYRE_M <= self.tyre_width_m <= MAX_TYRE_M):
            out.append("The tyre width is missing or not a tyre's width.")
        if not self.track_width_m or self.track_width_m <= 0:
            out.append("The track width — centre to centre of the left and right wheels — "
                       "is missing.")
        elif self.track_width_m <= self.side_width_m:
            out.append("The track width is narrower than the tyres themselves; it is measured "
                       "centre to centre of the left and right wheels.")
        return out

    @property
    def side_width_m(self) -> float:
        """Crushed width per side of the machine."""
        return self.tyre_width_m * (1 if self.rear_follows_front else 2)

    @property
    def strips(self) -> int:
        return 2 if self.rear_follows_front else 4


def smooth_runs(runs: list[np.ndarray], samples: int = SMOOTH_SAMPLES) -> list[np.ndarray]:
    """Each run's x/y, averaged over ``samples`` either side. The ends keep
    the average of what is there, so a run neither shrinks nor overshoots."""
    out = []
    for xy in runs:
        if len(xy) < 3 or samples < 1:
            out.append(xy)
            continue
        window = 2 * samples + 1
        kernel = np.ones(window)
        pad = np.pad(xy, ((samples, samples), (0, 0)), mode="edge")
        count = np.convolve(np.ones(len(pad)), kernel, mode="valid")
        smoothed = np.column_stack([np.convolve(pad[:, i], kernel, mode="valid") / count
                                    for i in range(2)])
        out.append(smoothed)
    return out


def _lines(runs: list[np.ndarray]):
    from shapely.geometry import LineString, MultiLineString

    parts = [LineString(xy) for xy in runs if len(xy) >= 2]
    parts = [p for p in parts if p.length > 0]
    return MultiLineString(parts) if parts else None


#: Samples per piece when the strips are drawn. A piece is short enough to
#: be nearly straight, so the gap between its own wheels never reaches back
#: over ground it crushed itself.
PIECE_SAMPLES = 40


def _pieces(runs: list[np.ndarray], size: int = PIECE_SAMPLES) -> list[np.ndarray]:
    """Each run cut into short pieces that share their end segments, so the
    strips meet with no gap on the outside of a bend."""
    out = []
    for xy in runs:
        if len(xy) < 2:
            continue
        start = 0
        while start < len(xy) - 1:
            stop = min(start + size, len(xy) - 1)
            out.append(xy[max(start - 1, 0): stop + 1])
            start = stop
    return out


def footprint(runs: list[np.ndarray], setup: TyreSetup, clip=None):
    """The ground the tyres crossed, as one (multi)polygon in the runs' CRS.

    Each piece of path gets its own two strips — the band between a buffer
    out to half the track width plus half the crushed width and one in to
    half the track width minus it — and the ground is the union of every
    piece's strips. The order matters: the gap between one pass's wheels
    must not erase ground another pass crushed, which is what taking the
    band of the union of the paths would do. Flat ends, so a run ends where
    the machine was; ground crossed twice is one area.
    """
    from shapely import unary_union
    from shapely.geometry import LineString

    half = setup.track_width_m / 2.0
    reach = setup.side_width_m / 2.0
    inner_distance = half - reach
    bands = []
    for xy in _pieces(runs):
        line = LineString(xy)
        if line.length <= 0:
            continue
        outer = line.buffer(half + reach, cap_style="flat", join_style="round")
        if inner_distance > 0:
            outer = outer.difference(line.buffer(inner_distance, cap_style="flat",
                                                 join_style="round"))
        bands.append(outer)
    if not bands:
        return None
    strips = unary_union(bands)
    if clip is not None:
        strips = strips.intersection(clip)
    return strips


def swath(runs: list[np.ndarray], width_m: float, clip=None):
    """The ground under the boom along the same path, overlaps counted once."""
    lines = _lines(runs)
    if lines is None or not width_m or width_m <= 0:
        return None
    covered = lines.buffer(width_m / 2.0, cap_style="flat", join_style="round")
    return covered.intersection(clip) if clip is not None else covered


def assess(runs: list[np.ndarray], setup: TyreSetup, *, field_area_ha: float | None,
           clip=None, yield_kg_ha: float | None = None, loss_fraction: float = 1.0,
           crop_price_per_kg: float | None = None) -> dict[str, Any]:
    """Area crushed, share of the field, and — given a yield — the crop lost.

    Everything metric; ``None`` where an input needed for it is missing.
    """
    problems = setup.problems()
    if problems:
        return {"available": False, "problems": problems}
    smoothed = smooth_runs(runs)
    shape = footprint(smoothed, setup, clip=clip)
    area_ha = float(shape.area) / 10_000.0 if shape is not None else 0.0
    travelled = float(sum(np.hypot(*np.diff(xy, axis=0).T).sum() for xy in smoothed if len(xy) > 1))
    linear_ha = travelled * setup.side_width_m * 2 / 10_000.0
    share = area_ha / field_area_ha if field_area_ha else None
    fraction = min(max(float(loss_fraction), 0.0), 1.0)
    lost_kg = area_ha * yield_kg_ha * fraction if yield_kg_ha else None
    value = lost_kg * crop_price_per_kg if lost_kg is not None and crop_price_per_kg else None
    return {
        "available": True,
        "tyre_width_m": setup.tyre_width_m,
        "track_width_m": setup.track_width_m,
        "rear_follows_front": setup.rear_follows_front,
        "strips": setup.strips,
        "distance_m": travelled,
        "area_ha": area_ha,
        "linear_area_ha": linear_ha,
        "driven_again_share": (1.0 - area_ha / linear_ha) if linear_ha > 0 else 0.0,
        "field_area_ha": field_area_ha,
        "share_of_field": share,
        "yield_kg_ha": yield_kg_ha,
        "loss_fraction": fraction,
        "lost_kg": lost_kg,
        "lost_kg_per_field_ha": (lost_kg / field_area_ha) if lost_kg is not None and field_area_ha else None,
        "crop_price_per_kg": crop_price_per_kg,
        "lost_value": value,
        "geometry": shape,
    }


def is_finite_positive(value: Any) -> bool:
    try:
        return math.isfinite(float(value)) and float(value) > 0
    except (TypeError, ValueError):
        return False
