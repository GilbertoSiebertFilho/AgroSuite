"""Terrain (relief) analysis.

Grids the GPS elevation of a yield or as-applied file — or a DEM — and
describes the relief of the field: hills, valleys, depressions that pond
water, the general fall and how steep it is. :func:`analyze` runs the whole
chain and its :class:`TerrainResult` carries the layers, the summary, the
findings and the zone datasets; the modules below are its building blocks.
"""

from .analysis import (
    TerrainOptions,
    TerrainResult,
    analyze,
    findings,
    zones_points,
    zones_polygons,
)
# The function is re-exported as ``contour_lines``: named ``contours`` here
# it would shadow the submodule, and ``import agrosuite.terrain.contours as m``
# would hand back the function. ``from agrosuite.terrain.contours import
# contours`` is the function's own name and keeps working.
from .contours import choose_interval, contours_to_geodataframe, profile
from .contours import contours as contour_lines
from .derivatives import (
    SECTOR_LABELS,
    SECTOR_NAMES,
    aspect,
    aspect_sectors,
    curvature,
    hillshade,
    plane_trend,
    roughness,
    slope,
    tpi,
    tpi_standardized,
    tri,
)
from .grid import (
    ElevationGrid,
    connected_components,
    disc_kernel,
    fill_nearest,
    gaussian_kernel,
    gaussian_smooth,
    grid_from_points,
    nan_convolve,
)
from .hydrology import (
    Hydrology,
    analyse_hydrology,
    default_twi_threshold,
    depression_depth,
    depression_labels,
    depressions,
    drainage_lines,
    fill_sinks,
    flow_accumulation,
    flow_direction_d8,
    twi,
    wet_cells,
)
from .landforms import (
    CLASSES,
    SLOPE_CLASSES,
    aspect_distribution,
    character,
    classify,
    foot_level,
    hills,
    lows,
    position_label,
    saddle_level,
    slope_classes,
)
from .render import (
    LANDFORM_COLORS,
    PALETTES,
    colormap,
    default_range,
    legend_for,
    mercator_bounds,
    palette_stops,
    ramp,
    render_png,
    to_web_mercator,
)
from .synthetic import synthetic_dem, synthetic_terrain

__all__ = [
    # analysis
    "TerrainOptions", "TerrainResult", "analyze", "findings", "zones_points", "zones_polygons",
    # contours
    "choose_interval", "contour_lines", "contours_to_geodataframe", "profile",
    # derivatives
    "SECTOR_LABELS", "SECTOR_NAMES", "aspect", "aspect_sectors", "curvature", "hillshade",
    "plane_trend", "roughness", "slope", "tpi", "tpi_standardized", "tri",
    # grid
    "ElevationGrid", "connected_components", "disc_kernel", "fill_nearest", "gaussian_kernel",
    "gaussian_smooth", "grid_from_points", "nan_convolve",
    # hydrology
    "Hydrology", "analyse_hydrology", "default_twi_threshold", "depression_depth",
    "depression_labels", "depressions", "drainage_lines", "fill_sinks", "flow_accumulation",
    "flow_direction_d8", "twi", "wet_cells",
    # landforms
    "CLASSES", "SLOPE_CLASSES", "aspect_distribution", "character", "classify", "foot_level",
    "hills", "lows", "position_label", "saddle_level", "slope_classes",
    # render
    "LANDFORM_COLORS", "PALETTES", "colormap", "default_range", "legend_for", "mercator_bounds",
    "palette_stops", "ramp", "render_png", "to_web_mercator",
    # synthetic
    "synthetic_dem", "synthetic_terrain",
]
