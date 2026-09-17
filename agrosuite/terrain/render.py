"""Map layers for the interface: colour ramps, web-mercator resampling, PNG.

The interface draws a terrain layer as a Leaflet image overlay: one picture
stretched between two lat/lng corners over the satellite basemap. Leaflet's
display projection is web mercator (EPSG:3857), so a picture is placed
exactly only if it *is* a web-mercator raster. The elevation grid is not: it
lives in the field's UTM zone, whose axes are rotated against web mercator
by the grid convergence — a degree or two, which over an 800 m field puts
the corners twenty or thirty metres off the field edge in the photo. So
every layer is resampled onto a regular web-mercator pixel grid first
(:func:`to_web_mercator`), and the overlay bounds are that pixel grid's
edges converted to lat/lng. Nothing is approximated in that step: web
mercator's x depends on longitude alone and y on latitude alone, so the
bounds round-trip exactly.

Colours are applied after resampling, never before, because interpolating
colours blends a hilltop's brown with its neighbour's yellow into something
that is neither; a categorical layer (landforms) is resampled with nearest
neighbour for the same reason. Cells outside the field are transparent, so
the basemap shows through the notch of an L-shaped field rather than a
rectangle of colour covering the neighbour's land.

The backend returns metric numbers: the legend carries ``vmin`` / ``vmax``
in the layer's own units and the interface formats them.
"""

from __future__ import annotations

import io
import math
from typing import Any, Mapping, Sequence

import numpy as np

from ..core import crs as crs_mod
from .grid import ElevationGrid, _transformer

WEB_MERCATOR = "EPSG:3857"

#: A colour ramp: ``(position in 0..1, '#rrggbb')`` stops in increasing
#: position, starting at 0 and ending at 1.
Stops = Sequence[tuple[float, str]]

#: Named ramps. Each is chosen for what the layer means: elevation reads
#: like a relief map (green lowland to white summit), slope goes white to
#: dark red so flat ground vanishes and the steep bits shout, wetness runs
#: dry yellow to deep blue.
PALETTES: dict[str, list[tuple[float, str]]] = {
    "elevation": [
        (0.00, "#1b5e20"),  # dark green
        (0.14, "#43a047"),  # green
        (0.29, "#9ccc65"),  # yellow-green
        (0.43, "#fdd835"),  # yellow
        (0.57, "#d2b48c"),  # tan
        (0.71, "#8d6e63"),  # brown
        (0.86, "#cfd8dc"),  # light grey
        (1.00, "#ffffff"),  # white
    ],
    "slope": [
        (0.00, "#ffffff"),
        (0.25, "#ffeb3b"),
        (0.50, "#ff9800"),
        (0.75, "#e53935"),
        (1.00, "#7f0000"),
    ],
    "diverging": [
        (0.00, "#2166ac"),
        (0.50, "#f7f7f7"),
        (1.00, "#b2182b"),
    ],
    "wetness": [
        (0.00, "#ffffcc"),
        (0.33, "#9ecae1"),
        (0.67, "#3182bd"),
        (1.00, "#08306b"),
    ],
    "hillshade": [
        (0.00, "#000000"),
        (1.00, "#ffffff"),
    ],
    # Aspect is a compass bearing: the ramp must come back to its first hue
    # at 360 so north-facing ground is one colour, not two.
    "aspect": [
        (0.000, "#d62728"),  # N
        (0.125, "#ff7f0e"),  # NE
        (0.250, "#ffdd57"),  # E
        (0.375, "#2ca02c"),  # SE
        (0.500, "#17becf"),  # S
        (0.625, "#1f77b4"),  # SW
        (0.750, "#9467bd"),  # W
        (0.875, "#e377c2"),  # NW
        (1.000, "#d62728"),  # back to N
    ],
    "accumulation": [
        (0.00, "#ffffff"),
        (0.25, "#c6dbef"),
        (0.50, "#6baed6"),
        (0.75, "#2171b5"),
        (1.00, "#08306b"),
    ],
}

#: Palettes whose ramp runs over ``log1p(value)`` rather than the value.
#: Flow accumulation spans four or five orders of magnitude; on a linear
#: ramp the whole field is white except the last few cells before the
#: outlet.
LOG_PALETTES: frozenset[str] = frozenset({"accumulation"})

#: Landform class codes to colours: warm and light on the high ground,
#: cool and dark in the hollows, so the map reads at a glance.
LANDFORM_COLORS: dict[int, str] = {
    1: "#a0522d",  # hilltop
    2: "#e9967a",  # upper slope
    3: "#f4d35e",  # mid slope
    4: "#c8e6a0",  # flat
    5: "#9ecae1",  # lower slope
    6: "#3182bd",  # valley
}

#: Number of stops in a continuous legend.
_LEGEND_STOPS = 7

#: Percentiles of the default colour stretch. A GPS-derived grid has a few
#: cells of leftover noise at either end; stretching to the true min and
#: max would spend most of the ramp on them.
_STRETCH_PERCENTILES = (2.0, 98.0)


# ==========================================================================
# Colours
# ==========================================================================

def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    text = str(color).strip()
    if text.startswith("#"):
        text = text[1:]
    if len(text) != 6:
        raise ValueError(
            f"Colour {color!r} is not a '#rrggbb' hex colour; palette stops and "
            "categorical maps take colours in that form."
        )
    try:
        return int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)
    except ValueError as exc:
        raise ValueError(
            f"Colour {color!r} is not a '#rrggbb' hex colour; palette stops and "
            "categorical maps take colours in that form."
        ) from exc


def _rgb_to_hex(rgb: Sequence[int]) -> str:
    r, g, b = (int(np.clip(round(float(v)), 0, 255)) for v in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def palette_stops(palette: str | Stops) -> list[tuple[float, str]]:
    """Resolve a palette name or an explicit list of stops, validated.

    Stops must be in increasing position, start at 0 and end at 1; anything
    else would leave part of the value range without a colour.
    """
    if isinstance(palette, str):
        if palette not in PALETTES:
            names = ", ".join(sorted(PALETTES))
            raise ValueError(
                f"Unknown palette {palette!r}. Use one of {names}, or pass a "
                "list of (position, '#rrggbb') stops."
            )
        return list(PALETTES[palette])
    stops = [(float(p), str(c)) for p, c in palette]
    if len(stops) < 2:
        raise ValueError("A colour ramp needs at least two stops.")
    positions = [p for p, _ in stops]
    if positions != sorted(positions) or positions[0] != 0.0 or positions[-1] != 1.0:
        raise ValueError(
            "Colour ramp stops must run from position 0 to position 1 in "
            "increasing order."
        )
    for _, color in stops:
        _hex_to_rgb(color)
    return stops


def _is_log(palette: str | Stops) -> bool:
    return isinstance(palette, str) and palette in LOG_PALETTES


def _forward(values: np.ndarray, log: bool) -> np.ndarray:
    """Values in the space the ramp is linear in."""
    if not log:
        return values
    return np.log1p(np.maximum(values, 0.0))


def _inverse(scaled: np.ndarray, log: bool) -> np.ndarray:
    return np.expm1(scaled) if log else scaled


def ramp(palette: str | Stops, t: np.ndarray) -> np.ndarray:
    """RGB (uint8, shape ``t.shape + (3,)``) of ramp positions ``t`` in 0..1.

    NaN positions come out black; the caller decides their alpha.
    """
    stops = palette_stops(palette)
    positions = np.array([p for p, _ in stops], dtype="float64")
    colours = np.array([_hex_to_rgb(c) for _, c in stops], dtype="float64")
    t = np.asarray(t, dtype="float64")
    tt = np.where(np.isfinite(t), np.clip(t, 0.0, 1.0), 0.0)
    out = np.empty(t.shape + (3,), dtype="uint8")
    for channel in range(3):
        out[..., channel] = np.rint(np.interp(tt, positions, colours[:, channel])).astype("uint8")
    return out


def _positions(values: np.ndarray, palette: str | Stops, vmin: float, vmax: float) -> np.ndarray:
    """Ramp position of every value, NaN kept as NaN, ends clipped."""
    log = _is_log(palette)
    lo, hi = float(_forward(np.array(vmin), log)), float(_forward(np.array(vmax), log))
    v = _forward(np.asarray(values, dtype="float64"), log)
    finite = np.isfinite(v)
    t = np.full(v.shape, np.nan)
    if hi > lo:
        t[finite] = np.clip((v[finite] - lo) / (hi - lo), 0.0, 1.0)
    else:
        # A constant layer has no stretch to speak of; the middle of the
        # ramp is the honest colour for "the one value there is".
        t[finite] = 0.5
    return t


def colormap(
    values: np.ndarray,
    palette: str | Stops = "elevation",
    vmin: float | None = None,
    vmax: float | None = None,
    categorical: Mapping[int, str] | None = None,
) -> np.ndarray:
    """Colour an array: uint8 RGBA shaped ``values.shape + (4,)``.

    Continuous: ``vmin`` maps to the first stop, ``vmax`` to the last, values
    beyond either end take the end colour, and a log palette maps
    ``log1p(value)`` between ``log1p(vmin)`` and ``log1p(vmax)``.
    Categorical: ``categorical`` is a ``{code: '#rrggbb'}`` map applied
    exactly; the palette and range are ignored, and a code the map does not
    name is transparent, like NaN.
    """
    values = np.asarray(values, dtype="float64")
    rgba = np.zeros(values.shape + (4,), dtype="uint8")
    if categorical is not None:
        finite = np.isfinite(values)
        codes = np.where(finite, np.rint(values), np.nan)
        for code, color in categorical.items():
            hit = finite & (codes == float(code))
            if hit.any():
                rgba[hit, :3] = _hex_to_rgb(color)
                rgba[hit, 3] = 255
        return rgba
    if vmin is None or vmax is None:
        vmin, vmax = default_range(values, palette, vmin, vmax)
    t = _positions(values, palette, float(vmin), float(vmax))
    finite = np.isfinite(t)
    rgba[..., :3] = ramp(palette, t)
    rgba[..., 3] = np.where(finite, 255, 0).astype("uint8")
    # Transparent pixels carry no colour: cleaner for anything that reads
    # the PNG back, and they compress to nothing.
    rgba[~finite] = 0
    return rgba


def default_range(
    values: np.ndarray,
    palette: str | Stops = "elevation",
    vmin: float | None = None,
    vmax: float | None = None,
) -> tuple[float, float]:
    """The colour stretch for a layer: ``(vmin, vmax)`` in the layer's units.

    Ends the caller fixed are kept; the others come from the 2nd and 98th
    percentiles. A diverging palette is centred on zero, since TPI and
    curvature mean "above / below the surroundings" and the two colours
    must meet at the neutral value; aspect is a bearing and always 0..360.
    """
    values = np.asarray(values, dtype="float64")
    finite = values[np.isfinite(values)]
    if palette == "aspect":
        return (0.0 if vmin is None else float(vmin), 360.0 if vmax is None else float(vmax))
    if finite.size == 0:
        raise ValueError(
            "This layer has no values inside the field, so there is nothing to "
            "draw and no colour range to choose."
        )
    p_lo, p_hi = np.percentile(finite, _STRETCH_PERCENTILES)
    if palette == "diverging":
        if vmin is None and vmax is None:
            half = float(max(abs(p_lo), abs(p_hi)))
            if not half > 0:
                half = 1.0
            return (-half, half)
    lo = float(p_lo) if vmin is None else float(vmin)
    hi = float(p_hi) if vmax is None else float(vmax)
    if not hi > lo:
        # A flat layer: give it a little room so the map is one colour
        # rather than a division by zero.
        lo, hi = lo - 0.5, hi + 0.5
        if _is_log(palette):
            lo = max(lo, 0.0)
    return (lo, hi)


def legend_for(
    palette: str | Stops = "elevation",
    vmin: float | None = None,
    vmax: float | None = None,
    categorical: Mapping[int, str] | None = None,
) -> dict[str, Any]:
    """The legend the interface draws next to a layer, JSON-safe.

    Continuous: ``{kind: 'continuous', vmin, vmax, stops: [{position,
    value, color}]}`` with seven stops evenly spaced along the ramp;
    ``position`` (0..1) is where the stop sits on the bar and ``value`` the
    number to label it with — on a log palette the values are not evenly
    spaced, which is the point. Categorical: ``{kind: 'categorical',
    classes: [{code, color}]}`` in code order.
    """
    if categorical is not None:
        classes = [
            {"code": int(code), "color": _rgb_to_hex(_hex_to_rgb(color))}
            for code, color in sorted(categorical.items(), key=lambda kv: float(kv[0]))
        ]
        return {"kind": "categorical", "classes": classes}
    if vmin is None or vmax is None:
        raise ValueError("A continuous legend needs both vmin and vmax.")
    log = _is_log(palette)
    lo, hi = float(_forward(np.array(vmin), log)), float(_forward(np.array(vmax), log))
    t = np.linspace(0.0, 1.0, _LEGEND_STOPS)
    scaled = lo + t * (hi - lo)
    values = _inverse(scaled, log)
    colours = ramp(palette, t)
    stops = [
        {"position": float(ti), "value": float(vi), "color": _rgb_to_hex(ci)}
        for ti, vi, ci in zip(t, values, colours)
    ]
    return {"kind": "continuous", "vmin": float(vmin), "vmax": float(vmax), "stops": stops}


# ==========================================================================
# Web mercator
# ==========================================================================

def mercator_bounds(grid: ElevationGrid) -> tuple[float, float, float, float]:
    """``(xmin, ymin, xmax, ymax)`` in EPSG:3857 of the grid's outer edges.

    The four corners and the four edge midpoints are transformed and the
    extremes taken, as :meth:`ElevationGrid.bounds_lonlat` does: a UTM
    rectangle is a slightly rotated, slightly bowed quadrilateral in web
    mercator and the box must hold all of it.
    """
    xmin, ymin, xmax, ymax = grid.bounds_metric()
    xmid, ymid = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    xs = np.array([xmin, xmax, xmax, xmin, xmid, xmax, xmid, xmin])
    ys = np.array([ymax, ymax, ymin, ymin, ymax, ymid, ymin, ymid])
    mx, my = _transformer(grid.crs, WEB_MERCATOR).transform(xs, ys)
    mx, my = np.asarray(mx), np.asarray(my)
    if not (np.all(np.isfinite(mx)) and np.all(np.isfinite(my))):
        raise ValueError(
            f"The grid's CRS {grid.crs!r} could not be placed on a web map; "
            "check that the file's projection is right."
        )
    return (float(mx.min()), float(my.min()), float(mx.max()), float(my.max()))


def to_web_mercator(
    grid: ElevationGrid,
    values: np.ndarray | None = None,
    max_px: int = 1400,
    order: int = 1,
) -> tuple[np.ndarray, tuple[float, float, float, float], tuple[int, int]]:
    """Resample a layer onto a regular EPSG:3857 pixel grid.

    Returns ``(values_3857, bounds_lonlat, size)``: the resampled 2-D array
    (row 0 north, NaN outside the field), the ``(west, south, east, north)``
    corners for Leaflet's ``imageOverlay([[south, west], [north, east]])``,
    and ``size = (width_px, height_px)``.

    The pixel is roughly the native cell — finer would invent detail, coarser
    would lose it — but the longer side never exceeds ``max_px``, which
    keeps a 400 000-cell grid's PNG at a size a browser draws without
    stalling. Pixels are not forced square: the bounds are the true mercator
    box and the pixel count is rounded up, so the image is stretched by a
    fraction of a pixel rather than the bounds being padded.

    ``order`` is the interpolation order handed to :meth:`ElevationGrid.sample`:
    0 (nearest) for class codes, 1 (bilinear) for anything continuous.
    """
    values = grid.z if values is None else np.asarray(values, dtype="float64")
    if values.shape != grid.shape:
        raise ValueError(
            f"A layer shaped {values.shape} cannot be drawn on a grid of "
            f"{grid.shape}; every layer must be computed on the same grid."
        )
    max_px = int(max_px)
    if max_px < 1:
        raise ValueError("max_px must be at least 1 pixel.")

    xmin, ymin, xmax, ymax = mercator_bounds(grid)
    width, height = xmax - xmin, ymax - ymin
    native = max(width / grid.cols, height / grid.rows)
    px = max(native, max(width, height) / max_px)
    nx = min(max(int(math.ceil(width / px - 1e-9)), 1), max_px)
    ny = min(max(int(math.ceil(height / px - 1e-9)), 1), max_px)
    dx, dy = width / nx, height / ny

    xs = xmin + (np.arange(nx) + 0.5) * dx
    ys = ymax - (np.arange(ny) + 0.5) * dy
    X, Y = np.meshgrid(xs, ys)
    gx, gy = _transformer(WEB_MERCATOR, grid.crs).transform(X.ravel(), Y.ravel())
    out = grid.sample(np.asarray(gx), np.asarray(gy), values, order=order).reshape(ny, nx)

    to_ll = _transformer(WEB_MERCATOR, crs_mod.WGS84)
    west, south = to_ll.transform(xmin, ymin)
    east, north = to_ll.transform(xmax, ymax)
    return out, (float(west), float(south), float(east), float(north)), (nx, ny)


# ==========================================================================
# PNG
# ==========================================================================

def _encode_png(rgba: np.ndarray) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgba), "RGBA").save(buf, format="PNG", compress_level=6)
    return buf.getvalue()


def render_png(
    grid: ElevationGrid,
    values: np.ndarray | None = None,
    palette: str | Stops = "elevation",
    vmin: float | None = None,
    vmax: float | None = None,
    categorical: Mapping[int, str] | None = None,
    hillshade: np.ndarray | None = None,
    hillshade_strength: float = 0.6,
    max_px: int = 1400,
) -> tuple[bytes, tuple[float, float, float, float], dict[str, Any]]:
    """Draw a layer as a web-mercator PNG for a Leaflet image overlay.

    Returns ``(png_bytes, bounds_lonlat, legend)``; ``bounds_lonlat`` is
    ``(west, south, east, north)`` and ``legend`` what :func:`legend_for`
    builds for the stretch actually used.

    ``values`` defaults to the grid's elevation. ``vmin`` / ``vmax`` default
    as in :func:`default_range`. With ``categorical`` the layer is drawn
    through that ``{code: colour}`` map with nearest-neighbour resampling.
    A ``hillshade`` array (0..1, computed on the same grid) is blended in by
    scaling the colours by ``1 - strength + strength * shade``, so the relief
    shows through the elevation or wetness colours instead of needing its
    own layer to be toggled on top.
    """
    values = grid.z if values is None else np.asarray(values, dtype="float64")
    if values.shape != grid.shape:
        raise ValueError(
            f"A layer shaped {values.shape} cannot be drawn on a grid of "
            f"{grid.shape}; every layer must be computed on the same grid."
        )
    strength = float(hillshade_strength)
    if not 0.0 <= strength <= 1.0:
        raise ValueError(
            "hillshade_strength is a fraction from 0 (colours only) to 1 "
            "(fully shaded)."
        )
    if hillshade is not None:
        hillshade = np.asarray(hillshade, dtype="float64")
        if hillshade.shape != grid.shape:
            raise ValueError(
                "The hillshade must be computed on the same grid as the layer "
                "it shades."
            )

    if categorical is not None:
        lo = hi = None
        order = 0
    else:
        lo, hi = default_range(values, palette, vmin, vmax)
        order = 1

    layer, bounds, size = to_web_mercator(grid, values, max_px=max_px, order=order)
    rgba = colormap(layer, palette, lo, hi, categorical=categorical)

    if hillshade is not None and strength > 0.0:
        shade, _, _ = to_web_mercator(grid, hillshade, max_px=max_px, order=1)
        shade = np.clip(shade, 0.0, 1.0)
        missing = ~np.isfinite(shade)
        if missing.any():
            # A shade hole inside the field is neither lit nor dark: give it
            # the field's average so it does not print as a bright patch.
            neutral = float(np.nanmean(shade)) if not missing.all() else 1.0
            shade = np.where(missing, neutral, shade)
        factor = (1.0 - strength) + strength * shade
        rgb = rgba[..., :3].astype("float64") * factor[..., None]
        rgba[..., :3] = np.clip(np.rint(rgb), 0, 255).astype("uint8")

    legend = legend_for(palette, lo, hi, categorical=categorical)
    return _encode_png(rgba), bounds, legend
