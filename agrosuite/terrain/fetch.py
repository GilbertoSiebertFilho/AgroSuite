"""A public elevation model for a field whose file carries no heights.

A boundary, a trial layout or a prescription says where the field is but
not how it lies. The relief is then read from a public digital elevation
model of the field's area, fetched on request:

* **HRDEM** (Natural Resources Canada) — a 1 m terrain model from LiDAR, bare
  earth, where Canada has flown it. Found through the NRCan STAC catalogue;
  the tile covering a field may still be empty over it, so its coverage is
  measured and it is used only when it covers the field.
* **Copernicus GLO-30** (ESA) — 30 m, worldwide. A *surface* model: it
  includes trees and buildings, which on open farmland is the ground, and
  30 m is coarse for a single field — it shows how the field lies, not the
  swale across a headland.

Both are cloud-optimised GeoTIFFs read by HTTP range requests: only the
field's window crosses the wire, and what leaves the computer is the
request for that window — the area's coordinates, nothing else.

The raster is cut to the field's outline plus a margin, so the analysis
describes the field rather than the neighbours, and written as a GeoTIFF in
``$AGROSUITE_HOME/dem`` under a name that depends only on the area and the
source: the same field asked for twice is read from disk the second time,
and a saved project that points at it still finds it tomorrow.
"""

from __future__ import annotations

import hashlib
import json
import math
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

HRDEM_STAC = "https://datacube.services.geo.ca/stac/api/search"
HRDEM_COLLECTION = "hrdem-mosaic-1m"
COPERNICUS_URL = "https://copernicus-dem-30m.s3.amazonaws.com/{tile}/{tile}.tif"

#: Share of the field a source must cover to be used.
MIN_COVERAGE = 0.90
#: Metres of ground kept around the field's outline: the slope at the edge
#: of the field needs the cells beyond it.
MARGIN_M = 30.0
#: Seconds to wait for the catalogue or a tile before giving up on it.
TIMEOUT_S = 25

NODATA = -9999.0

ATTRIBUTION = {
    "hrdem": ("High Resolution Digital Elevation Model (HRDEM), Natural Resources Canada. "
              "Contains information licensed under the Open Government Licence – Canada."),
    "copernicus": ("Copernicus DEM GLO-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space "
                   "GmbH 2014-2018, provided under COPERNICUS by the European Union and ESA."),
}

LABEL = {"hrdem": "HRDEM 1 m (NRCan LiDAR)", "copernicus": "Copernicus 30 m"}


class FetchError(ValueError):
    """Raised with a sentence the interface can show as it is."""


@dataclass
class Fetched:
    path: Path
    source: str
    resolution_m: float
    coverage: float
    cached: bool
    notes: list[str]


# ==========================================================================
# The field's outline
# ==========================================================================

def field_outline(dataset, margin_m: float = MARGIN_M):
    """The field's outline in lon/lat, grown by ``margin_m``.

    The polygons a file declares, merged — or, for points, their hull. A
    trial layout's plots leave alleys between them; the hull and the margin
    close those, which is the field the relief is wanted for.
    """
    from pyproj import Transformer
    from shapely.geometry import MultiPoint
    from shapely.ops import transform, unary_union

    geometries = [g for g in (dataset.geometry or []) if g is not None and not g.is_empty]
    if geometries:
        shape = unary_union(geometries).convex_hull
    else:
        df = dataset.df
        if "lon" not in df or "lat" not in df:
            raise FetchError(f"'{dataset.meta.name}' has no position, so there is no field to fetch "
                             "the elevation for.")
        points = df[["lon", "lat"]].dropna().to_numpy()
        if len(points) < 3:
            raise FetchError(f"'{dataset.meta.name}' has too few positions to outline a field.")
        shape = MultiPoint([tuple(p) for p in points]).convex_hull
    if shape.is_empty or shape.geom_type not in ("Polygon", "MultiPolygon"):
        raise FetchError(f"'{dataset.meta.name}' does not outline an area.")
    centre = shape.centroid
    zone = int((centre.x + 180) // 6) + 1
    utm = f"EPSG:{32600 + zone if centre.y >= 0 else 32700 + zone}"
    to_m = Transformer.from_crs("EPSG:4326", utm, always_xy=True).transform
    to_ll = Transformer.from_crs(utm, "EPSG:4326", always_xy=True).transform
    grown = transform(to_m, shape).buffer(margin_m)
    area_ha = grown.area / 10_000.0
    if area_ha > 20_000:
        raise FetchError(
            f"'{dataset.meta.name}' spans {area_ha:,.0f} ha — more than a field. Fetch the "
            "elevation for one field's boundary at a time.")
    return transform(to_ll, grown), area_ha


# ==========================================================================
# Sources
# ==========================================================================

def copernicus_tiles(west: float, south: float, east: float, north: float) -> list[str]:
    """The 1° tiles of Copernicus GLO-30 covering a lon/lat box."""
    tiles = []
    for lat in range(math.floor(south), math.floor(north) + 1):
        for lon in range(math.floor(west), math.floor(east) + 1):
            ns = f"N{lat:02d}" if lat >= 0 else f"S{-lat:02d}"
            ew = f"E{lon:03d}" if lon >= 0 else f"W{-lon:03d}"
            tiles.append(f"Copernicus_DSM_COG_10_{ns}_00_{ew}_00_DEM")
    return tiles


def _get_json(url: str, timeout: float = TIMEOUT_S) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "AgroSuite"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def hrdem_assets(west, south, east, north, get_json: Callable = _get_json) -> list[str]:
    """URLs of the HRDEM terrain (DTM) tiles covering the box."""
    query = urllib.parse.urlencode({"collections": HRDEM_COLLECTION,
                                    "bbox": f"{west},{south},{east},{north}", "limit": 10})
    found = get_json(f"{HRDEM_STAC}?{query}")
    return [f["assets"]["dtm"]["href"] for f in found.get("features", [])
            if "dtm" in (f.get("assets") or {})]


def _gdal_env():
    import rasterio

    return rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                        GDAL_HTTP_TIMEOUT=str(TIMEOUT_S), GDAL_HTTP_MAX_RETRY="2",
                        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif")


def _read_window(url: str, outline_ll):
    """The part of a remote (or local) raster under the outline, masked to it.

    Returns ``(array, transform, crs, resolution_m, coverage)`` — coverage
    being the share of the outline's cells with a height.
    """
    import rasterio
    from rasterio.features import geometry_mask
    from rasterio.warp import transform_bounds, transform_geom
    from rasterio.windows import from_bounds

    path = url if "://" not in url else "/vsicurl/" + url
    with rasterio.open(path) as src:
        box = transform_bounds("EPSG:4326", src.crs, *outline_ll.bounds, densify_pts=21)
        window = from_bounds(*box, src.transform).round_offsets().round_lengths()
        window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
        z = src.read(1, window=window, masked=True).astype("float32")
        transform = src.window_transform(window)
        outline = transform_geom("EPSG:4326", src.crs, outline_ll.__geo_interface__)
        outside = geometry_mask([outline], out_shape=z.shape, transform=transform)
        valid = ~np.ma.getmaskarray(z) & ~outside
        coverage = float(valid.sum() / max(1, (~outside).sum()))
        data = np.where(valid, z.filled(NODATA), NODATA).astype("float32")
        res = abs(src.res[0])
        if src.crs.is_geographic:
            res *= 111_320.0 * math.cos(math.radians(outline_ll.centroid.y))
        return data, transform, src.crs, float(res), coverage


def _read_merged(urls: list[str], outline_ll):
    """Several tiles of one model joined over the outline, then read as one."""
    import tempfile

    import rasterio
    from rasterio.merge import merge

    sources = [rasterio.open(u if "://" not in u else "/vsicurl/" + u) for u in urls]
    try:
        crs = sources[0].crs
        from rasterio.warp import transform_bounds

        box = transform_bounds("EPSG:4326", crs, *outline_ll.bounds, densify_pts=21)
        mosaic, transform = merge(sources, bounds=box, nodata=NODATA)
    finally:
        for src in sources:
            src.close()
    with tempfile.TemporaryDirectory() as folder:
        joined = Path(folder) / "joined.tif"
        _write(joined, mosaic[0].astype("float32"), transform, crs)
        return _read_window(str(joined), outline_ll)


def _write(path: Path, data, transform, crs) -> None:
    import rasterio

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", width=data.shape[1], height=data.shape[0], count=1,
                       dtype="float32", crs=crs, transform=transform, nodata=NODATA,
                       compress="deflate") as dst:
        dst.write(data, 1)


def dem_cache_dir() -> Path:
    from ..app import persist as persist_mod

    return persist_mod.home_dir() / "dem"


def fetch_dem(outline_ll, *, source: str = "auto", cache_dir: Path | None = None,
              readers: dict[str, Callable] | None = None) -> Fetched:
    """Fetch the elevation under a field outline and write it as a GeoTIFF.

    ``source`` is ``auto`` (HRDEM where it covers the field, else
    Copernicus), ``hrdem`` or ``copernicus``. ``readers`` lets a test stand
    in for the network: ``{"hrdem": callable(outline) -> [urls],
    "copernicus": callable(outline) -> [urls]}``.
    """
    if source not in ("auto", "hrdem", "copernicus"):
        raise FetchError("The elevation source is auto, hrdem or copernicus.")
    cache_dir = Path(cache_dir) if cache_dir else dem_cache_dir()
    west, south, east, north = outline_ll.bounds
    key = hashlib.sha1(outline_ll.wkb).hexdigest()[:12]
    readers = readers or {}
    order = ["hrdem", "copernicus"] if source == "auto" else [source]
    tried: list[str] = []

    for name in order:
        target = cache_dir / f"field_{key}_{name}.tif"
        if target.exists():
            data, transform, crs, res, coverage = _read_window(str(target), outline_ll)
            if coverage >= MIN_COVERAGE:
                return Fetched(target, name, res, coverage, True, _notes(name, res, coverage))
        try:
            if name == "hrdem":
                urls = (readers.get("hrdem") or (lambda o: hrdem_assets(*o.bounds)))(outline_ll)
            else:
                urls = (readers.get("copernicus") or (
                    lambda o: [COPERNICUS_URL.format(tile=t) for t in copernicus_tiles(*o.bounds)]
                ))(outline_ll)
        except (OSError, ValueError) as exc:
            tried.append(f"{LABEL[name]}: its catalogue did not answer ({exc})")
            continue
        best = None
        with _gdal_env():
            for url in urls:
                try:
                    piece = _read_window(url, outline_ll)
                except Exception as exc:  # a missing tile over the sea, a timeout
                    tried.append(f"{LABEL[name]}: {type(exc).__name__}")
                    continue
                if best is None or piece[4] > best[4]:
                    best = piece
                if best[4] >= MIN_COVERAGE:
                    break
            if (best is None or best[4] < MIN_COVERAGE) and len(urls) > 1:
                # A field across the edge of two tiles: neither covers it alone.
                try:
                    merged = _read_merged(urls, outline_ll)
                    if best is None or merged[4] > best[4]:
                        best = merged
                except Exception as exc:
                    tried.append(f"{LABEL[name]}: tiles could not be joined ({type(exc).__name__})")
        if best is not None and best[4] >= MIN_COVERAGE:
            data, transform, crs, res, coverage = best
            _write(target, data, transform, crs)
            return Fetched(target, name, res, coverage, False, _notes(name, res, coverage))
        if best is not None:
            tried.append(f"{LABEL[name]} covers {best[4]:.0%} of the field")
        elif not any(t.startswith(LABEL[name]) for t in tried):
            tried.append(f"{LABEL[name]} has no tile here")

    raise FetchError(
        "No public elevation model could be read for this field: " + "; ".join(tried) + ". "
        "Check the internet connection, or open a DEM GeoTIFF of the field instead.")


def _notes(source: str, res: float, coverage: float) -> list[str]:
    notes = [f"Elevation fetched from the {LABEL[source]} model, {res:.0f} m cells, "
             f"covering {coverage:.0%} of the field."]
    if source == "copernicus":
        notes.append(
            "A 30 m surface model: it shows how the field lies, but it is coarse for a single "
            "field, and trees and buildings are part of its surface.")
    notes.append(ATTRIBUTION[source])
    return notes
