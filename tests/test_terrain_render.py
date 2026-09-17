"""Tests for the terrain map rendering.

The interface trusts three things about a rendered layer: that the colours
mean what the legend says (vmin is the first stop, vmax the last, NaN is
transparent), that the picture lands on the basemap where the field is (the
web-mercator bounds hold every valid cell and a known summit sits at the
right pixel), and that what is outside the field stays see-through. Each is
checked by decoding the PNG that would be sent, not by inspecting
intermediate arrays.
"""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agrosuite.terrain.grid import ElevationGrid
from agrosuite.terrain.render import (
    LANDFORM_COLORS,
    PALETTES,
    colormap,
    default_range,
    legend_for,
    mercator_bounds,
    palette_stops,
    render_png,
    to_web_mercator,
)
from agrosuite.terrain.synthetic import synthetic_dem

_TO_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


def _decode(png: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(png))
    assert img.mode == "RGBA"
    return np.asarray(img)


def _plane(rows: int = 50, cols: int = 70, cell: float = 5.0, fall: str = "east") -> ElevationGrid:
    """A plane on a UTM grid near Calgary, rising towards ``fall``'s opposite."""
    grid = ElevationGrid(np.zeros((rows, cols)), 323_000.0, 5_736_000.0, cell, "EPSG:32612")
    X, Y = grid.xy()
    if fall == "east":
        z = 700.0 - 0.02 * (X - grid.x0)
    else:
        z = 700.0 - 0.02 * (grid.y0 - Y)
    return grid.with_values(z)


@pytest.fixture(scope="module")
def dem():
    return synthetic_dem()


# ==========================================================================
# Colours
# ==========================================================================

def test_palettes_are_well_formed():
    for name, stops in PALETTES.items():
        positions = [p for p, _ in stops]
        assert positions[0] == 0.0 and positions[-1] == 1.0, name
        assert positions == sorted(positions), name
        assert all(len(c) == 7 and c.startswith("#") for _, c in stops), name
    # Aspect is a bearing: 360 must be the same colour as 0.
    assert PALETTES["aspect"][0][1] == PALETTES["aspect"][-1][1]
    assert set(LANDFORM_COLORS) == {1, 2, 3, 4, 5, 6}


def test_colormap_endpoints_and_nan():
    for name, stops in PALETTES.items():
        rgba = colormap(np.array([10.0, 12.5, 15.0, np.nan, 5.0, 20.0]), name, 10.0, 15.0)
        first = tuple(int(stops[0][1][i:i + 2], 16) for i in (1, 3, 5))
        last = tuple(int(stops[-1][1][i:i + 2], 16) for i in (1, 3, 5))
        assert tuple(rgba[0, :3]) == first, name
        assert tuple(rgba[2, :3]) == last, name
        assert rgba[3, 3] == 0 and rgba[3, :3].sum() == 0, name
        # Beyond the range takes the end colour, still opaque.
        assert tuple(rgba[4, :3]) == first and rgba[4, 3] == 255, name
        assert tuple(rgba[5, :3]) == last and rgba[5, 3] == 255, name
        assert rgba[[0, 1, 2], 3].tolist() == [255, 255, 255], name
    assert colormap(np.ones((3, 4)), "slope", 0.0, 2.0).shape == (3, 4, 4)


def test_colormap_log_palette_is_log1p():
    # log1p(99) is exactly half way between log1p(0) and log1p(9999), so 99
    # must land on the middle stop of the accumulation ramp.
    rgba = colormap(np.array([0.0, 99.0, 9999.0]), "accumulation", 0.0, 9999.0)
    mid = PALETTES["accumulation"][2][1]
    assert tuple(rgba[1, :3]) == tuple(int(mid[i:i + 2], 16) for i in (1, 3, 5))
    assert tuple(rgba[0, :3]) == (255, 255, 255)
    assert tuple(rgba[2, :3]) == (0x08, 0x30, 0x6b)


def test_colormap_categorical_uses_the_map_exactly():
    codes = np.array([[1, 2, 3], [4, 5, 6], [7, np.nan, 3.0]])
    rgba = colormap(codes, "elevation", 0, 1, categorical=LANDFORM_COLORS)
    for r in range(2):
        for c in range(3):
            code = int(codes[r, c])
            hexcol = LANDFORM_COLORS[code]
            assert tuple(rgba[r, c, :3]) == tuple(int(hexcol[i:i + 2], 16) for i in (1, 3, 5))
            assert rgba[r, c, 3] == 255
    # A code the map does not name and NaN are both transparent.
    assert rgba[2, 0, 3] == 0 and rgba[2, 1, 3] == 0
    assert rgba[2, 2, 3] == 255


def test_custom_stops_and_unknown_palette():
    stops = [(0.0, "#000000"), (1.0, "#ff0000")]
    rgba = colormap(np.array([0.0, 0.5, 1.0]), stops, 0.0, 1.0)
    assert tuple(rgba[1, :3]) == (128, 0, 0)
    assert palette_stops("slope") == PALETTES["slope"]
    with pytest.raises(ValueError, match="Unknown palette"):
        colormap(np.zeros(3), "rainbow", 0.0, 1.0)
    with pytest.raises(ValueError, match="position 0 to position 1"):
        palette_stops([(0.2, "#000000"), (1.0, "#ffffff")])
    with pytest.raises(ValueError, match="rrggbb"):
        palette_stops([(0.0, "black"), (1.0, "#ffffff")])


def test_default_range_rules():
    values = np.array([-3.0, -1.0, 0.0, 1.0, np.nan, 2.0])
    lo, hi = default_range(values, "diverging")
    assert lo == -hi and hi > 0
    assert default_range(values, "aspect") == (0.0, 360.0)
    lo, hi = default_range(values, "elevation")
    assert -3.0 <= lo < hi <= 2.0
    assert default_range(values, "elevation", vmin=-10.0)[0] == -10.0
    lo, hi = default_range(np.full(20, 5.0), "elevation")
    assert lo < 5.0 < hi
    with pytest.raises(ValueError, match="nothing to draw"):
        default_range(np.full(4, np.nan), "elevation")


def test_legend_shapes_are_json_safe():
    legend = legend_for("elevation", 100.0, 200.0)
    assert legend["kind"] == "continuous"
    assert legend["vmin"] == 100.0 and legend["vmax"] == 200.0
    assert len(legend["stops"]) == 7
    assert legend["stops"][0] == {"position": 0.0, "value": 100.0, "color": "#1b5e20"}
    assert legend["stops"][-1]["color"] == "#ffffff" and legend["stops"][-1]["value"] == 200.0
    values = [s["value"] for s in legend["stops"]]
    assert values == sorted(values)
    json.dumps(legend)

    log_legend = legend_for("accumulation", 0.0, 9999.0)
    assert log_legend["stops"][3]["value"] == pytest.approx(99.0)

    cat = legend_for(categorical=LANDFORM_COLORS)
    assert cat == {
        "kind": "categorical",
        "classes": [{"code": k, "color": v} for k, v in sorted(LANDFORM_COLORS.items())],
    }
    json.dumps(cat)


# ==========================================================================
# Web mercator
# ==========================================================================

def test_web_mercator_bounds_hold_every_valid_cell(dem):
    grid, _truth = dem
    arr, (west, south, east, north), (nx, ny) = to_web_mercator(grid, max_px=1400)
    assert west < east and south < north
    assert arr.shape == (ny, nx)
    X, Y = grid.xy()
    lon, lat = grid.to_lonlat(X[grid.mask], Y[grid.mask])
    assert west <= lon.min() and lon.max() <= east
    assert south <= lat.min() and lat.max() <= north
    # Roughly the native cell: a 160-cell grid gives about 160 pixels.
    assert 150 <= nx <= 190 and 150 <= ny <= 190
    # The sampled values are the grid's, not something stretched.
    assert np.nanmin(arr) >= np.nanmin(grid.z) - 1e-6
    assert np.nanmax(arr) <= np.nanmax(grid.z) + 1e-6
    # The 3857 box converts back to lon/lat exactly.
    xmin, ymin, xmax, ymax = mercator_bounds(grid)
    assert _TO_3857.transform(west, south) == pytest.approx((xmin, ymin), abs=1e-6)
    assert _TO_3857.transform(east, north) == pytest.approx((xmax, ymax), abs=1e-6)


def test_web_mercator_size_respects_max_px(dem):
    grid, _truth = dem
    arr, _bounds, (nx, ny) = to_web_mercator(grid, max_px=100)
    assert max(nx, ny) <= 100 and min(nx, ny) >= 1
    assert arr.shape == (ny, nx)
    with pytest.raises(ValueError, match="same grid"):
        to_web_mercator(grid, np.zeros((3, 3)))


def test_web_mercator_puts_the_summit_at_the_right_pixel(dem):
    grid, truth = dem
    arr, (west, south, east, north), (nx, ny) = to_web_mercator(grid)
    x0, y0 = _TO_3857.transform(west, south)
    x1, y1 = _TO_3857.transform(east, north)
    hill = truth["hills"][0]
    mx, my = _TO_3857.transform(*hill["summit_lonlat"])
    col = int((mx - x0) / (x1 - x0) * nx)
    row = int((y1 - my) / (y1 - y0) * ny)
    assert abs(arr[row, col] - hill["summit_z_m"]) < 0.1
    # The hill is a local maximum: nothing within 20 m of it is higher.
    window = arr[row - 4:row + 5, col - 4:col + 5]
    assert abs(np.nanmax(window) - hill["summit_z_m"]) < 0.05
    # And the notch (north-west corner) is empty in the resampled array.
    assert np.isnan(arr[int(0.1 * ny), int(0.1 * nx)])


def test_categorical_resampling_keeps_codes(dem):
    grid, _truth = dem
    codes = np.where(grid.mask, 1.0 + (np.arange(grid.cols) % 6)[None, :], np.nan)
    arr, _b, _s = to_web_mercator(grid, codes, order=0)
    finite = arr[np.isfinite(arr)]
    assert set(np.unique(finite).tolist()) <= {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}


# ==========================================================================
# PNG
# ==========================================================================

def test_plane_renders_monotonic_along_the_fall():
    for fall, axis in (("east", 1), ("south", 0)):
        grid = _plane(fall=fall)
        png, (west, south, east, north), legend = render_png(grid, palette="hillshade")
        assert west < east and south < north
        a = _decode(png)
        grey = a[..., 0].astype(int)
        opaque = a[..., 3] == 255
        assert opaque.mean() > 0.9
        lines = range(a.shape[0]) if axis == 1 else range(a.shape[1])
        spread = []
        for k in lines:
            values = grey[k, :] if axis == 1 else grey[:, k]
            keep = opaque[k, :] if axis == 1 else opaque[:, k]
            line = values[keep]
            if line.size < 2:
                continue
            # Falling towards the east/south means darker with the index.
            assert np.all(np.diff(line) <= 0), (fall, k)
            spread.append(line[0] - line[-1])
        assert max(spread) > 200
        assert legend["kind"] == "continuous" and legend["vmin"] < legend["vmax"]


def test_masked_notch_is_transparent(dem):
    grid, _truth = dem
    png, _bounds, _legend = render_png(grid)
    a = _decode(png)
    h, w = a.shape[:2]
    assert a[int(0.1 * h), int(0.1 * w), 3] == 0
    assert a[h // 2, w // 2, 3] == 255
    # The opaque pixels cover the field's area and nothing more. The
    # mercator box is larger than the field (the UTM rectangle sits in it
    # slightly rotated, leaving transparent corner triangles), so compare
    # areas on the ground rather than shares of the array.
    xmin, ymin, xmax, ymax = mercator_bounds(grid)
    _w, south, _e, north = grid.bounds_lonlat()
    scale = np.cos(np.radians(0.5 * (south + north)))
    box_ground_m2 = (xmax - xmin) * (ymax - ymin) * scale ** 2
    opaque_m2 = (a[..., 3] == 255).mean() * box_ground_m2
    assert abs(opaque_m2 / (grid.area_ha() * 1e4) - 1.0) < 0.02


def test_hillshade_blending_darkens_the_shaded_side():
    grid = ElevationGrid(np.full((40, 60), 700.0), 323_000.0, 5_736_000.0, 5.0, "EPSG:32612")
    shade = np.ones(grid.shape)
    shade[:, 30:] = 0.2
    plain, _b, _l = render_png(grid, palette="slope", vmin=699.0, vmax=701.0, hillshade=shade,
                               hillshade_strength=0.0)
    shaded, _b, _l = render_png(grid, palette="slope", vmin=699.0, vmax=701.0, hillshade=shade,
                                hillshade_strength=0.6)
    p, s = _decode(plain).astype(int), _decode(shaded).astype(int)
    h, w = p.shape[:2]
    west_px = (h // 2, w // 4)
    east_px = (h // 2, 3 * w // 4)
    # strength 0 leaves the colours alone; the lit side is untouched at 0.6
    # too, since shade 1 gives a factor of exactly 1.
    assert tuple(p[west_px][:3]) == tuple(s[west_px][:3])
    assert tuple(p[east_px][:3]) == tuple(p[west_px][:3])
    assert s[east_px][:3].sum() < s[west_px][:3].sum() * 0.6
    assert s[east_px][3] == 255
    with pytest.raises(ValueError, match="same grid"):
        render_png(grid, hillshade=np.ones((3, 3)))
    with pytest.raises(ValueError, match="fraction"):
        render_png(grid, hillshade=shade, hillshade_strength=1.5)


def test_categorical_png_and_legend(dem):
    grid, _truth = dem
    codes = np.where(grid.mask, 1.0 + (np.arange(grid.rows) % 6)[:, None], np.nan)
    png, _bounds, legend = render_png(grid, codes, categorical=LANDFORM_COLORS)
    a = _decode(png)
    assert legend["kind"] == "categorical" and len(legend["classes"]) == 6
    opaque = a[a[..., 3] == 255]
    seen = {tuple(px[:3]) for px in opaque[:: max(1, opaque.shape[0] // 2000)]}
    expected = {tuple(int(c[i:i + 2], 16) for i in (1, 3, 5)) for c in LANDFORM_COLORS.values()}
    assert seen <= expected and len(seen) >= 5


def test_png_is_a_valid_rgba_image_with_the_reported_size(dem):
    grid, _truth = dem
    png, bounds, legend = render_png(grid, max_px=90)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    img = Image.open(io.BytesIO(png))
    assert img.mode == "RGBA"
    _arr, _b, size = to_web_mercator(grid, max_px=90)
    assert img.size == size and max(size) <= 90
    json.dumps({"bounds": bounds, "legend": legend})
    with pytest.raises(ValueError, match="nothing to draw"):
        render_png(grid, np.full(grid.shape, np.nan))


def test_render_speed_300x300():
    r, c = np.indices((300, 300))
    z = 700.0 + 0.01 * r + 3.0 * np.sin(c / 30.0)
    grid = ElevationGrid(z, 400_000.0, 5_730_000.0, 5.0, "EPSG:32612")
    shade = np.full(grid.shape, 0.7)
    t0 = time.perf_counter()
    png, _bounds, _legend = render_png(grid, hillshade=shade)
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.5, elapsed
    assert len(png) > 1000
