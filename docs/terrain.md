# Terrain

The relief of a field, read from the altitude a monitor already logged.

Every yield and as-applied file carries a GPS height on every point. Nobody
looks at it, because scattered heights along a track say nothing on their
own. Put them on a grid and they answer the questions a farmer actually
asks: where does the water sit, which way does the field fall, which end
washes, where is the ground too steep to work across. That is what the
terrain analyser does, and it works on a file you already have — no survey,
no drone, no LiDAR tile.

A DEM GeoTIFF is the better source when there is one, and it goes through
the same chain: open it like any other file and the analysis reads the
raster itself, at full resolution, rather than the sample of it the map
shows.

This document is the reference for the API and the handoff for the
interface. Everything here was read out of the code, and every response
below came from driving the endpoints against `POST /api/terrain/demo`.

---

## 1 · What it claims, and what it does not

GPS altitude is the least reliable number in a monitor file. It is not a
survey: it is a consumer-grade receiver reporting height above an ellipsoid
while the machine bounces through a field, and everything the analyser says
has to survive that. Five corrections come before the first slope is read
(`agrosuite/terrain/grid.py`, `grid_from_points`):

**A reading that is not a height at all is dropped.** An altitude of exactly
0 m, −9999 or −32768 is a lost fix or a no-data marker, not ground. These
are caught by value, not by neighbourhood, because every neighbour agrees
with them.

**A whole pass sitting away from the rest is discarded.** No field falls
30 m between two adjacent passes, so a pass whose median altitude is more
than 30 m — and more than ten median-absolute-deviations — from the other
passes was not on this ground. The test is relative to the spread of the
pass medians, so a long field genuinely climbing a hillside is left alone.

**Spikes are dropped.** Each reading is compared with its twelve nearest
neighbours; one that disagrees with all of them is the receiver losing
satellites for a second, not a bump.

**The constant offset between passes is solved and removed.** A receiver
re-converges after every headland turn and comes back a few centimetres
higher or lower. Left alone, that prints the pass pattern straight into the
relief — corduroy across the field. One offset per pass is fitted from the
overlapping readings of neighbouring passes and subtracted. How much had to
be taken out is reported as `source.pass_offset_sd_m`: above 0.3 m the
receiver was not RTK-corrected, and a finding says the shape of the relief
is sound but the absolute heights are approximate.

**An altitude quantised to whole metres is recognised.** Some monitors log
the altitude as an integer, and every integer raster does the same. Whole
metres cut a gentle slope into terraces with 1 m risers, which a slope
operator reads as 20 % ground. The step is detected (1 m, 0.5 m or 0.1 m,
declared only after 50 readings land on it) and the surface is smoothed over
ten times the step, which brings a 1 m riser under 4 % — the slope of the
ground it is cut into — while a real hill loses at most a tenth of its
height. A finding says features smaller than one step cannot be told from a
single step.

What is left after all that is measured, not assumed: `vertical_noise_m` is
the scatter of the readings around the fitted surface, and
`surface_noise_m` what remains on the surface itself. **Every detection
threshold is tied to that number.** A hill is reported only when it stands
at least `min_feature_height_m` above the foot of the ground around it, and
that height defaults to `max(0.5 m, 2 × vertical noise, the value step)`.
The same threshold floors the landform classes, the closed depressions and
the wetness ranking. A bump smaller than the noise is the noise.

### A field with no relief is reported as level

Standardized indices always have tails. Rank the cells of a dead-flat field
by wetness and a tenth of them still come out in the top decile; contour a
constant altitude column and marching squares traces 953 loops around
differences of 1e-13 m. So there is a floor. A field whose residual relief
(the 5th to 95th percentile of the surface after its general fall is
removed) **and** whose plane fall are both under
`max(0.05 m, 2 × vertical noise, the value step)` is *level*: it gets no
contour lines, no drainage lines, no wet ground, no hills, lows or
depressions, and one finding says why. The plane fall is part of the test
because a tilted field with no relief beyond its tilt still has real,
parallel contours.

### The outer ring of cells is left out of every share

Slope, aspect and curvature are read off the 3×3 neighbourhood of a cell.
On the outermost ring of valid cells part of that neighbourhood is outside
the field, and the operators fill it with the nearest real value rather than
returning NaN — otherwise the field would lose a ring of cells at every edge
and around every hole. The cost is that the ring's derivatives lean on
copied values: on a straight edge the slope component perpendicular to the
edge is halved, and on a jagged edge a copied diagonal neighbour can push it
either way (a 5 % plane reads 1.5–5.7 % on that ring).

So **the layers keep the ring and the shares do not**. Every percentage in
the summary — slope classes, aspect sectors, landform classes, wet ground —
is counted on the interior only, and the areas are those shares of the whole
field, so they still add up to the field. Without that, a uniform 3 % plane
reports 2 % of itself as flat. The summary carries the arithmetic in
`grid.interior_ha`, `grid.edge_ring_ha` and a ready-made sentence in
`grid.shares_note`.

### What it does not claim

- **Not a survey.** Absolute heights are whatever the receiver's datum said.
  Only differences within one field mean anything, and only after the pass
  offsets were removed.
- **Not a drainage design.** A closed depression's volume is what the
  surface would hold to its spill point. It says nothing about the soil
  under it, the tile already in the ground, or where the water goes next.
- **Smoothing costs small hollows.** Every metre of smoothing shaves the
  depth of a narrow hollow; a finding names the width below which features
  read shallower than they are, and `smooth_m: 0` reads the raw surface.
- **It cannot tell metres from feet.** Nothing in an altitude column
  declares its unit. If the swath width in the same file reads as feet the
  analysis opens with a warning, because a monitor writes every length in
  the one system it was set to — but it will not convert anything behind
  your back. Declare the file's units and analyse the converted copy.

---

## 2 · The API

Prefix `/api/terrain`. Everything is metric in and metric out; the
interface converts for display.

A `TerrainResult` holds a dozen grid-sized arrays — a 400 000-cell grid is
some 40 MB — so only the **four most recent** analyses stay in memory. The
JSON summary is small and outlives them in the session's saved report, which
is why `GET /api/terrain/{id}` can still answer when `/layers` cannot.

All responses below are real: they came out of
`fastapi.testclient.TestClient` driving these endpoints against the demo
field. Long arrays are cut with an ellipsis and a note says what was cut,
and the floats are shown to a few decimals — the API returns full float64.

### POST /api/terrain/demo

Loads the synthetic field of known relief, so the analyser can be tried
without a file. Takes no body. Returns the ordinary dataset summary that
every import returns; the truth about what was put in the field travels in
`meta.extra.terrain_truth` for anyone who wants to compare.

```json
{
  "id": "cf267138d452",
  "label": "Terrain demo",
  "rows": 33400,
  "columns": ["lon", "lat", "timestamp", "value", "swath_m", "speed_kmh",
              "pass_id", "elev_m", "x", "y", "distance_m", "heading_deg"],
  "metric_crs": "EPSG:32612",
  "area_ha": 60.82,
  "role": "yield",
  "preflight": { "verdict": "ok", "summary": "The file came in clean and complete." }
}
```

*Trimmed: `meta`, `stats`, `numeric_columns`, `bounds` and the full
`preflight` block are left out.*

### POST /api/terrain/analyze

The one call that does the work. Body:

```json
{ "dataset_id": "cf267138d452" }
```

| Field | Default | What it is |
|---|---|---|
| `dataset_id` | required | The dataset to analyse |
| `cell_m` | half the swath, at least 5 m, capped at 25 m | Grid cell, metres |
| `smooth_m` | enough to leave ~3 cm of noise | Gaussian scale, metres; `0` reads the raw surface |
| `tpi_small_m` | `max(3 × cell, 30)` | Small position radius, metres |
| `tpi_large_m` | `max(10 × cell, 150)`, capped at a quarter of the field | Large position radius, metres |
| `contour_interval_m` | chosen from the relief | Contour step, metres |
| `detrend_passes` | `true` | Remove the pass-to-pass offsets |
| `remove_outliers` | `true` | Drop altitude spikes |
| `min_feature_height_m` | `max(0.5, 2 × noise, value step)` | How far a feature must stand out, metres |
| `hillshade_azimuth` | `315` | Sun bearing, degrees clockwise from north |
| `hillshade_altitude` | `45` | Sun height, degrees |
| `min_upstream_ha` | `1.0` | Catchment a drainage line needs |

Unknown keys are refused with a 422, so a typo cannot silently fall back to
a default. The resolved values come back in `summary.options`.

Response — `{"dataset_id": ..., "summary": {...}}` — where the summary has
fourteen top-level keys:

```json
{
  "dataset_id": "cf267138d452",
  "summary": {
    "source": {
      "kind": "points",
      "points_total": 33400, "points_used": 33386, "outliers_removed": 14,
      "passes": 89, "passes_dropped": 0,
      "pass_offset_sd_m": 0.3404, "vertical_noise_m": 0.29605,
      "surface_noise_m": 0.02941, "detrended": true,
      "smooth_m": 10.0, "value_step_m": 0.0,
      "missing_elevations": 0, "non_numeric_elevations": 0,
      "dem_path": null, "unit_doubt": null,
      "notes": ["The surface was smoothed over 10 m to take out the 30 cm of altitude noise in the readings. ..."],
      "note_facts": [{"key": "smoothing", "smooth_m": 10.0, "noise_m": 0.296}]
    },
    "grid": {
      "rows": 162, "cols": 161, "cell_m": 5.0, "crs": "EPSG:32612",
      "bounds_lonlat": [-113.55601066, 51.74623566, -113.54395105, 51.75376432],
      "area_ha": 60.4825, "interior_ha": 58.48, "edge_ring_ha": 2.0025,
      "shares_note": "The slope, aspect, landform and wet shares are read on the 58.5 ha inside the field's outermost ring of cells; ..."
    },
    "elevation": {
      "min_m": 693.688, "max_m": 704.224, "mean_m": 699.795, "median_m": 699.873,
      "relief_m": 10.5365, "p05_m": 696.373, "p95_m": 702.779,
      "residual_relief_m": 3.2102, "relief_floor_m": 0.5921, "level": false,
      "histogram": { "edges": [693.688, 694.215, "…21 edges…"], "counts": [95, 132, "…20 counts…"] }
    },
    "slope": {
      "mean_pct": 1.525, "median_pct": 1.153, "p95_pct": 3.598, "max_pct": 5.989,
      "histogram": { "edges": ["…21…"], "counts": ["…20…"] },
      "classes": [
        {"key": "flat",     "label": "Flat (< 2 %)",     "from_pct": 0.0, "to_pct": 2.0,  "area_ha": 43.99, "pct": 72.73},
        {"key": "gentle",   "label": "Gentle (2-5 %)",   "from_pct": 2.0, "to_pct": 5.0,  "area_ha": 15.91, "pct": 26.31},
        {"key": "moderate", "label": "Moderate (5-10 %)","from_pct": 5.0, "to_pct": 10.0, "area_ha": 0.58,  "pct": 0.95},
        {"key": "strong",   "label": "Strong (10-15 %)", "from_pct": 10.0,"to_pct": 15.0, "area_ha": 0.0,   "pct": 0.0},
        {"key": "steep",    "label": "Steep (> 15 %)",   "from_pct": 15.0,"to_pct": null, "area_ha": 0.0,   "pct": 0.0}
      ]
    },
    "aspect": {
      "sectors": [
        {"key": "N",  "label": "Facing north",      "area_ha": 0.80,  "pct": 1.32},
        {"key": "SW", "label": "Facing south-west", "area_ha": 21.15, "pct": 34.96},
        {"key": "flat","label": "Flat (no aspect)", "area_ha": 4.47,  "pct": 7.40}
      ]
    },
    "trend": {
      "gradient_pct": 0.7153, "direction_deg": 222.83,
      "direction_label": "south-west", "drop_m": 8.1149, "r2": 0.7877
    },
    "character": {
      "key": "gently_undulating", "label": "gently undulating",
      "why": "The slope averages 1.5 % (95 % of the field is under 3.6 %) over 10.5 m of relief: gentle undulations rather than distinct hills."
    },
    "landforms": {
      "tpi_small_m": 30.0, "tpi_large_m": 150.0,
      "classes": [
        {"key": "hilltop", "code": 1, "label": "Hilltop / ridge", "area_ha": 3.30, "pct": 5.46, "color": "#a0522d"}
      ]
    },
    "features": {
      "hills": [
        {"id": 1, "label": "Hill 1", "lon": -113.552805, "lat": 51.748702,
         "area_ha": 2.635, "summit_m": 701.519, "height_m": 3.497,
         "mean_slope_pct": 3.177, "position": "south-west part", "elongation": 1.512}
      ],
      "lows": [
        {"id": 1, "label": "Low 1", "lon": -113.553252, "lat": 51.746354,
         "area_ha": 5.3925, "bottom_m": 693.688, "depth_m": 2.279,
         "closed": true, "position": "southern part", "elongation": 6.367}
      ],
      "depressions": [],
      "depressions_unlisted": 3,
      "depression_floor_m": 0.5921
    },
    "wetness": {
      "twi_mean": 8.131, "twi_p90": 10.010, "twi_threshold": 9.692,
      "noise_slope_pct": 0.588, "ranked_share_pct": 87.55,
      "wet_area_ha": 5.391, "wet_pct": 8.913, "drainage_length_m": 3595.02
    },
    "contours": { "interval_m": 1.0, "count": 17 },
    "layers": [
      {"key": "elevation", "label": "Elevation", "kind": "continuous",
       "unit": "m", "min": 693.688, "max": 704.224, "palette": "elevation"}
    ],
    "findings": [
      {"level": "info", "text": "Total relief is 10.5 m over 60.5 ha: a gently undulating field. …"},
      {"level": "warning", "text": "The altitude shifted by about 0.3 m from pass to pass, which means the receiver had no RTK correction. …"}
    ],
    "options": {
      "cell_m": 5.0, "smooth_m": 10.0, "tpi_small_m": 30.0, "tpi_large_m": 150.0,
      "contour_interval_m": 1.0, "detrend_passes": true, "remove_outliers": true,
      "min_feature_height_m": 0.5921, "max_cells": 400000,
      "hillshade_azimuth": 315.0, "hillshade_altitude": 45.0, "min_upstream_ha": 1.0
    }
  }
}
```

*Trimmed: the two histograms (21 edges and 20 counts each), six of the nine
aspect sectors, five of the six landform classes, the second hill and low,
all twelve `layers` entries but the first, and ten of the thirteen findings.
The demo lists no depression at the default threshold — the hollow the
synthetic field carries is 0.19 m deep, under the 0.59 m floor twice the
GPS noise sets, so it is one of the three counted in
`depressions_unlisted`. Analyse with `min_feature_height_m: 0.2` and it is
listed:*

```json
{"id": 1, "label": "Depression 1", "lon": -113.547576, "lat": 51.752323,
 "area_ha": 0.82, "max_depth_m": 0.5395, "volume_m3": 1825.09,
 "spill_m": 700.613, "position": "north-east part"}
```

Refusals carry a sentence, not a code:

| Request | Status | `detail` |
|---|---|---|
| unknown dataset | 404 | `Dataset 'nope' is not loaded in this session.` |
| a zones dataset | 400 | `These are terrain zones derived from 'Terrain demo'; analyse 'Terrain demo' instead (dataset fc8bbcb17f55: post it to /api/terrain/analyze).` |
| no altitude in the file | 400 | `This file carries no usable elevation, so its relief cannot be analysed. Open a yield or as-applied export that includes the GPS altitude, or a DEM GeoTIFF of the field.` |
| a misspelled option | 422 | `[{"type": "extra_forbidden", "loc": ["body", "cel_m"], …}]` |

### GET /api/terrain/{dataset_id}

The summary of the last analysis, same shape as above. It answers from the
cached result when there is one and from the saved report when the arrays
have been evicted, so a report panel keeps working after four other fields
have been analysed. 404 with `Run the terrain analysis first: '…' has not
been analysed. Post its id to /api/terrain/analyze.` when there is neither.

### GET /api/terrain/{dataset_id}/layers

Every map layer with its image URL, corners and legend. This one needs the
arrays, so it 404s once the result has been evicted, with `The terrain
analysis of '…' is no longer in memory: the app keeps the 4 most recent to
hold the map layers. Run the analysis again to bring it back.`

```json
{
  "layers": [
    {
      "key": "elevation", "label": "Elevation", "kind": "continuous",
      "unit": "m", "min": 693.688, "max": 704.224, "palette": "elevation",
      "url": "/api/terrain/cf267138d452/layer/elevation.png",
      "bounds": [[51.74623566, -113.55601066], [51.75376432, -113.54395105]],
      "legend": {
        "kind": "continuous", "vmin": 695.410, "vmax": 703.266,
        "stops": [
          {"position": 0.0, "value": 695.410, "color": "#1b5e20"},
          {"position": 0.1667, "value": 696.719, "color": "#53a84c"},
          {"position": 1.0, "value": 703.266, "color": "#ffffff"}
        ]
      }
    },
    {
      "key": "landform", "label": "Landform", "kind": "categorical",
      "unit": "", "min": 1.0, "max": 6.0, "palette": "landform",
      "url": "/api/terrain/cf267138d452/layer/landform.png",
      "bounds": [[51.74623566, -113.55601066], [51.75376432, -113.54395105]],
      "legend": {
        "kind": "categorical",
        "classes": [
          {"code": 1, "color": "#a0522d", "key": "hilltop", "label": "Hilltop / ridge"},
          {"code": 6, "color": "#3182bd", "key": "valley",  "label": "Valley / hollow"}
        ]
      }
    }
  ]
}
```

*Trimmed: ten of the twelve layers, four of the seven continuous stops, four
of the six landform classes.*

`legend` is `null` when a layer has nothing finite inside the field — it
draws transparent and has no range to label.

### GET /api/terrain/{dataset_id}/layer/{key}.png

The layer as a web-mercator PNG with an alpha channel, `Cache-Control:
no-store` (the image changes whenever the analysis is re-run under the same
URL). Query parameters:

- `hillshade=1` — blend the relief into the colours. Applies to every layer
  but the hillshade itself, at 0.6 strength for continuous layers and 0.35
  for the landform classes, whose flat colours are the legend.
- `vmin`, `vmax` — override the stretch the `/layers` legend was built with.
  Either may be blank (the interface sends `vmin=&vmax=` when the boxes are
  empty), and the other still applies.

An unknown key 404s and lists the real ones.

### GET /api/terrain/{dataset_id}/contours

GeoJSON in WGS84. `interval_m` redraws at another step; on a level field it
is ignored, because 0.1 m there would trace the rounding noise the analysis
refused to draw.

```json
{
  "type": "FeatureCollection",
  "interval_m": 1.0,
  "count": 17,
  "features": [
    {
      "type": "Feature",
      "properties": {"level_m": 694.0, "index": 0, "major": false},
      "geometry": {"type": "LineString",
                   "coordinates": [[-113.5529615, 51.7463386], [-113.5529974, 51.7463597]]}
    }
  ]
}
```

*Trimmed: 16 of 17 features, and 7 of the first feature's 9 vertices.*
`major` is true on every fifth level — the line to draw thicker and label.

### GET /api/terrain/{dataset_id}/features

Hills, lows and closed depressions as polygons, plus the drainage lines.
Each feature's `properties` are the summary's numbers for that feature plus
`kind` (`hill`, `low` or `depression`), so a popup needs no second call.

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {"type": "Polygon", "coordinates": ["…"]},
      "properties": {
        "id": 1, "label": "Hill 1", "lon": -113.552805, "lat": 51.748702,
        "area_ha": 2.635, "summit_m": 701.519, "height_m": 3.497,
        "mean_slope_pct": 3.177, "position": "south-west part",
        "elongation": 1.512, "kind": "hill"
      }
    }
  ],
  "drainage": {
    "type": "FeatureCollection",
    "features": [
      {"type": "Feature",
       "geometry": {"type": "LineString", "coordinates": ["…"]},
       "properties": {"index": 0, "length_m": 926.75}}
    ]
  }
}
```

*Trimmed: 3 of 4 features, 10 of 11 drainage lines, and every coordinate
array.*

### POST /api/terrain/{dataset_id}/profile

Elevation along a line drawn on the map. Body:

```json
{ "points": [[-113.5535987, 51.7477414], [-113.546363, 51.7522586]], "n": 5 }
```

`points` is any number of `(lon, lat)` vertices from two up; `n` is the
number of stations, at most 5000 — a chart a screen wide has a few hundred
pixels, and one station per half cell already reads every cell the line
crosses.

```json
{
  "distance_m": [0.0, 177.18, 354.36, 531.54, 708.72],
  "elev_m": [698.213, 699.513, 697.660, 701.836, 701.861],
  "slope_pct": [null, -0.734, 1.046, -2.357, -0.014],
  "points": [[-113.5535987, 51.7477414], [-113.546363, 51.7522586]],
  "length_m": 708.72
}
```

`slope_pct[i]` is the gradient of the step from station `i-1` to station
`i`, signed so that **walking downhill is positive**; entry 0 is `null`
because nothing precedes the start. Stations off the field are `null` in
both lists. A line that crosses the field at fewer than two stations is
refused: a chart of nothing looks like a flat field rather than a missed
click.

Add `values_dataset_id` (and, optionally, `value_column` and `units`) and
the line carries a second series: the yield, the applied rate — whatever
that layer measures — along the same stations.

```json
{
  "distance_m": [0.0, 177.18, 354.36, 531.54, 708.72],
  "elev_m": [698.213, 699.513, 697.660, 701.836, 701.861],
  "slope_pct": [null, -0.734, 1.046, -2.357, -0.014],
  "values": [2828.19, 2507.91, null, 2621.67, 2789.74],
  "values_meta": {
    "column": "value", "value_label": "Yield", "value_unit": "kg/ha",
    "operation": "harvest", "crop": "canola",
    "corridor_m": 10.0, "points_used": 208,
    "note": "Each station is the mean of the yield readings within 10.0 m of the line; 208 of them were used. Where none was near, the line breaks."
  },
  "points": [[-113.5535987, 51.7477414], [-113.546363, 51.7522586]],
  "length_m": 708.72
}
```

The elevation is read off the grid, which has a value everywhere inside
the field; the yield has not, because it exists only where the machine
drove. So each station takes the mean of the readings within a corridor of
`max(2 × cell, half the median swath)` around it, and a station with
nothing in its corridor is `null` — **break the series there**, as with the
elevation, rather than drawing across ground that was never harvested.

Without `values_dataset_id` the response is exactly the five keys above it,
so every existing caller reads the same answer.

### POST /api/terrain/{dataset_id}/yield

A value layer read against the relief of `dataset_id`. The relief must be
in memory — the comparison samples the grid itself under every point — so
it 404s with `Run the terrain analysis first: …` when it is not.

```json
{ "yield_dataset_id": "cf267138d452", "bands": 8, "value_column": null,
  "units": {"yield_unit": "bu/ac", "area_unit": "ac", "length_unit": "ft", "crop": "canola"} }
```

| Field | Default | What it is |
|---|---|---|
| `yield_dataset_id` | required | The layer carrying the values |
| `bands` | 8 | Elevation bands, 2 to 20 |
| `value_column` | the `value` column | Another numeric column of that layer |
| `units` | metric | The reader's unit set; it reaches **the findings only** |

`yield_dataset_id` may be **the analysed dataset itself** — a yield map
whose own GPS altitude was gridded, which is the common case — or a
**separate layer**, a DEM having been analysed with the yield map loaded
beside it. The two need not share a metric CRS: the points are reprojected
to the grid's before they are sampled.

```json
{
  "terrain": {"dataset_id": "cf267138d452", "label": "Terrain demo"},
  "values": {"dataset_id": "cf267138d452", "label": "Terrain demo",
             "column": "value", "value_label": "Yield", "value_unit": "kg/ha",
             "operation": "harvest", "crop": "canola", "cleaned": false},
  "points": {"total": 33400, "matched": 33400, "off_grid": 0},
  "overall": {"mean": 2598.003, "median": 2593.235, "sd": 307.576,
              "cv_pct": 11.839, "points": 33400},
  "elevation_bands": [
    {"index": 0, "from_m": 693.688, "to_m": 698.554, "mean_elev_m": 697.284,
     "points": 8350, "area_ha": 15.2575, "mean": 2875.090, "median": 2871.719,
     "p25": 2709.056, "p75": 3045.927, "sd": 262.203, "delta_pct": 10.665}
  ],
  "slope_classes": [
    {"key": "gentle", "label": "Gentle (2-5 %)", "from_pct": 2.0, "to_pct": 5.0,
     "points": 8671, "mean": 2646.760, "delta_pct": 1.877}
  ],
  "landforms": [
    {"code": 1, "key": "hilltop", "label": "Hilltop / ridge", "color": "#a0522d",
     "points": 1813, "mean": 2401.903, "delta_pct": -7.548}
  ],
  "wetness": [
    {"key": "wet", "label": "Likely wet ground", "points": 2965,
     "mean": 2651.284, "delta_pct": 2.051}
  ],
  "relations": {"elevation_r": -0.6873, "slope_r": 0.1144, "twi_r": 0.1108,
                "elevation_r2": 0.4724,
                "strongest": {"key": "elevation", "label": "the elevation",
                              "r": -0.6873, "r2": 0.4724}},
  "findings": [
    {"level": "info", "text": "The lowest ground (693.7 m to 698.6 m) averaged 2 875 kg/ha, 11 % above the field average; …"}
  ]
}
```

*Trimmed: three of the four bands shown at `bands: 4`, four of the five
slope classes, five of the six landform classes, the dry half of the
wetness split, and six of the seven findings.*

The bands are **equal-count** (quantile), not equal-width: a field's
heights are not spread evenly, and equal-width bands put half the field in
the middle two and thirty readings in the top one, whose mean then swings
on a single combine pass. So each band carries its own range (`from_m` /
`to_m`, the outer two opened to the grid's extremes so the areas add up to
the field) and its `area_ha`, which is the field's ground at that height —
not the points', or a strip driven twice would read as twice the hectares.
`delta_pct` is the group's mean against the field mean, signed, in percent.

`relations` are Pearson correlations over the matched points; `strongest`
names which of the three explains most and how much, as r². Every class
list is complete — an empty class is listed with `points: 0` and a `null`
mean — and the class lists together account for every matched point.
`wetness` is `[]`, not two rows, when the analysis made no wet ranking at
all (a level field): two rows of which one is empty would read as a field
with no wet ground, which is a claim the analysis did not make.

Refusals carry a sentence:

| Request | Status | `detail` |
|---|---|---|
| relief not analysed | 404 | `Run the terrain analysis first: 'Terrain demo' has not been analysed. …` |
| unknown value layer | 404 | `Dataset 'nope' is not loaded in this session.` |
| a layer with no value | 400 | `'Boundary' carries no value column to read against the relief. …` |
| a zones dataset | 400 | `'Terrain zones (landform) — Terrain demo' is the terrain zone layer cut from 'Terrain demo': its values are zone codes, not a measurement …` |
| another field | 400 | `Only 0 of 6800 points fall on the analysed field; check that the two files cover the same field.` |
| `bands` outside 2–20 | 400 | `Cut the field into between 2 and 20 elevation bands; 50 is outside that. …` |
| a unit that does not exist | 400 | `'furlong' is not a valid length unit. Use one of: m, ft, in, cm, yd.` |
| a misspelled key | 422 | `[{"type": "extra_forbidden", "loc": ["body", "band"], …}]` |

The result is kept on the **analysed** dataset's entry, under
`reports['terrain_yield']`, so `GET /api/datasets/{id}/report/terrain_yield`
answers after a reload and the printed page carries the section.

### POST /api/terrain/{dataset_id}/zones

Turn the relief into a zone dataset the rest of the app already knows how to
export and join. Body: `{"by": "landform", "kind": "points", "bands": 4}` —
`by` is one of `landform`, `slope_class`, `elevation_bands`, `wetness`;
`kind` is `points` (one per grid cell) or `polygons` (one per zone);
`bands` applies to `elevation_bands` only.

```json
{
  "dataset": {
    "id": "f2380ee1e03a",
    "label": "Terrain zones (landform) — Terrain demo",
    "rows": 24193,
    "columns": ["lon", "lat", "value", "zone", "zone_label", "elev_m",
                "slope_pct", "twi", "landform", "aspect_deg", "x", "y"],
    "area_ha": 60.48,
    "origin": "terrain_zones",
    "parent_id": null,
    "role": null,
    "meta": {
      "geometry_type": "point",
      "value_label": "Terrain zone",
      "operation_label": "Elevation (DEM)",
      "notes": ["Zones derived from the relief (landform) of 'Terrain demo'."],
      "extra": {
        "zones_by": "landform", "zones_cell_m": 5.0,
        "terrain_source": "Terrain demo", "terrain_source_id": "cf267138d452",
        "zone_labels": {"1": "Hilltop / ridge", "…": "…", "6": "Valley / hollow"}
      }
    }
  },
  "zone_labels": {"1": "Hilltop / ridge", "2": "Upper slope", "3": "Mid slope",
                  "4": "Flat", "5": "Lower slope", "6": "Valley / hollow"},
  "by": "landform",
  "kind": "points",
  "bands": null,
  "source_id": "cf267138d452"
}
```

*Trimmed: `stats`, `numeric_columns`, `bounds` and `preflight` of the
dataset summary, and four of the six labels inside `meta.extra`.*

`bands` is `null` unless `by` is `elevation_bands`. Wrong values are refused
by name: `Zones can be made by landform, slope_class, elevation_bands,
wetness; 'colour' is not one of them.`

### POST /api/terrain/{dataset_id}/export

GeoTIFFs and vector files for QGIS, zipped, with a README naming every file.
Body: `{"include": ["geotiff", "contours", "features", "drainage"],
"vector_format": "shapefile"}` (`geojson` is the other format; `include`
defaults to all four).

```json
{
  "path": "/tmp/agrosuite_s7d55yz5/exports/terrain_demo_terrain",
  "zip_path": "/tmp/agrosuite_s7d55yz5/exports/terrain_demo_terrain.zip",
  "filename": "terrain_demo_terrain.zip",
  "download_url": "/api/download/ce3139e79639476b",
  "files": ["elevation.tif", "slope_pct.tif", "aspect_deg.tif", "twi.tif",
            "hillshade.tif", "landform.tif", "depression_depth.tif",
            "contours.geojson", "features.geojson", "drainage.geojson", "README.txt"],
  "entries": ["terrain_demo_terrain/README.txt", "…10 more…"],
  "notes": []
}
```

*Trimmed: 10 of the 11 zip entries.*

Every GeoTIFF is on the analysis grid, one band, float32 with nodata −9999
outside the field — except `landform.tif`, which is uint8 with nodata 255
and carries each class name as a `class_<code>` tag, which is how the names
survive the trip to QGIS and back. Vector files are WGS84.
`notes` carries what was left out and why: a level field has no contours
file, a field with no features has no features file.

---

## 3 · The layers

Twelve keys, in the order the interface should list them. `unit` is what the
number means; `palette` names the ramp; `kind` says whether the legend is a
bar or a class list.

| Key | Label | Unit | Palette | Legend |
|---|---|---|---|---|
| `elevation` | Elevation | m | `elevation` | green lowland → yellow → tan → brown → white summit |
| `slope_pct` | Slope | % (100 × rise/run) | `slope` | white → yellow → orange → red → dark red |
| `aspect_deg` | Aspect | degrees clockwise from north | `aspect` | a colour wheel that returns to its first hue at 360 |
| `hillshade` | Hillshade | — (0…1) | `hillshade` | black → white, fixed range |
| `tpi_small` | Position (small scale) | m above/below a 30 m disc | `diverging` | blue → white → red, centred on 0 |
| `tpi_large` | Position (large scale) | m above/below a 150 m disc | `diverging` | blue → white → red, centred on 0 |
| `twi` | Wetness index | index (dimensionless) | `wetness` | pale yellow → blue → dark blue |
| `landform` | Landform | — (class code) | `landform` | the six classes of §4 |
| `curvature_profile` | Profile curvature | index | `diverging` | positive = convex, water accelerates |
| `curvature_plan` | Plan curvature | index | `diverging` | positive = flow spreads, negative = flow gathers |
| `depression_depth` | Ponding depth | m | `wetness` | 0 on most of a field, stretched to the 98th percentile of the positive cells |
| `flow_acc` | Flow accumulation | — (cells draining through, itself included) | `accumulation` | white → dark blue on a **log** ramp |

Three stretches are not the plain 2nd–98th percentile, and the interface
should not try to recompute them: `hillshade` is fixed at 0…1 because it is
a fraction with a fixed meaning; `depression_depth` runs 0 to the 98th
percentile of its *positive* cells, because it is 0 on most of a field and a
percentile stretch would colour the whole map mid-ramp; `flow_acc` runs 1 to
its maximum on a log ramp, because it spans four or five orders of magnitude
and the full range is the point. A `diverging` palette is always centred on
zero — TPI and curvature mean "above / below the surroundings" and the two
colours have to meet at the neutral value. `aspect` is always 0…360.

Take the legend from `/layers` and label the bar with the stops' `value`; do
not assume they are evenly spaced, because on the log ramp they are not
(that is what makes the bar honest).

### Placing the image on the map

Every PNG is resampled onto a regular **web-mercator** pixel grid before it
is coloured, and the `bounds` in `/layers` are that pixel grid's edges in
lat/lng. So:

```js
L.imageOverlay(layer.url, layer.bounds, { opacity: 0.75 }).addTo(map);
```

`bounds` is already Leaflet's `[[south, west], [north, east]]`, in that
order, ready to pass through. It is the same box for every layer of one
analysis, so switching layers never moves the image.

The resampling is not cosmetic. The elevation grid lives in the field's UTM
zone, whose axes are rotated against web mercator by the grid convergence —
a degree or two, which over an 800 m field puts the corners twenty or thirty
metres off the field edge in the photo. Nothing is approximated in the
conversion: web mercator's x depends on longitude alone and y on latitude
alone, so the bounds round-trip exactly.

Colours are applied *after* resampling, and the landform layer is resampled
nearest-neighbour, so a hilltop's brown is never blended with its
neighbour's yellow into a third class that does not exist. Cells outside the
field are transparent, so the basemap shows through the notch of an L-shaped
field.

---

## 4 · Landform codes

Six classes, from the two-scale topographic position index of Weiss (2001)
collapsed from ten: a farmer's decisions — headland, drainage, erosion —
split a field six ways at most, and the finer classes only add noise. Warm
and light on the high ground, cool and dark in the hollows, so the map reads
at a glance.

| Code | Key | Label | Colour |
|---|---|---|---|
| 1 | `hilltop` | Hilltop / ridge | `#a0522d` |
| 2 | `upper_slope` | Upper slope | `#e9967a` |
| 3 | `mid_slope` | Mid slope | `#f4d35e` |
| 4 | `flat` | Flat | `#c8e6a0` |
| 5 | `lower_slope` | Lower slope | `#9ecae1` |
| 6 | `valley` | Valley / hollow | `#3182bd` |

`landform.tif` writes these as uint8 with nodata 255. Read them from
`summary.landforms.classes` or the categorical legend rather than hard-coding
the table, so a change stays in one place.

The slope classes have their own table, in percent, on the usual agronomic
breaks — under 2 % drains slowly, 2–5 % is where sheet erosion starts on
bare soil, above 10 % machinery and erosion both become a concern:

| Key | Label | From | To |
|---|---|---|---|
| `flat` | Flat (< 2 %) | 0 | 2 |
| `gentle` | Gentle (2-5 %) | 2 | 5 |
| `moderate` | Moderate (5-10 %) | 5 | 10 |
| `strong` | Strong (10-15 %) | 10 | 15 |
| `steep` | Steep (> 15 %) | 15 | — |

---

## 5 · The zones datasets

A zone layer is an ordinary dataset in the session. It is registered like an
import, it appears in `/api/datasets`, and every route that takes a
`dataset_id` takes it.

**`kind: "points"`** — one row per valid grid cell:

| Column | What it is |
|---|---|
| `lon`, `lat`, `x`, `y` | the cell centre, WGS84 and the grid's metric CRS |
| `value` | the zone code as a float — the app's standard value column, so maps, statistics and exports work with no special case |
| `zone` | the zone code as an integer |
| `zone_label` | the code's text, e.g. `Hilltop / ridge` |
| `elev_m`, `slope_pct`, `twi`, `landform`, `aspect_deg` | the terrain layers at that cell, carried so a join or an export can use them without a second call |

**`kind: "polygons"`** — one row per zone, dissolved:

| Column | What it is |
|---|---|
| `lon`, `lat`, `x`, `y` | a representative point inside the polygon |
| `zone`, `zone_label`, `value` | as above |
| `area_ha` | the zone's area, hectares |
| `mean_elev_m`, `mean_slope_pct` | its means |

Both carry `meta.extra.zones_by`, `zone_labels`, `zones_cell_m`,
`terrain_source` and `terrain_source_id`.

### How they flow onward

**They start with no role.** Zones are an output of the analysis, not a
layer of the field: the preliminary check files them under elevation because
they carry heights, and the route then clears the role so the yield map is
not displaced. The user gives them one — a plan, say — when they are to go
to the monitor. `parent_id` is `null` too: they sit beside the source, not
as a version of it.

**They cannot be analysed again.** Posting a zones dataset to
`/api/terrain/analyze` is refused by name, with the source's id to post
instead. Their points are cell centres and their polygons carry one row
each; re-gridded as a GPS track they would hand back a coarser copy of the
relief they were cut from, under the label of a new analysis.

**Export.** `POST /api/export` with `{"dataset_id": ..., "formats":
["shapefile"]}` writes the layer as it stands; the formats that branch takes
are `shapefile`, `geojson`, `gpkg` and `csv`, and `columns` picks which
columns travel. `POST /api/qgis/export` takes the same `dataset_id` and
writes the GeoPackage plus the `.qgs` project.

A **prescription** is the other branch of the same route, and it takes
`features` — a GeoJSON FeatureCollection with a rate property — not a
`dataset_id`; that is what reaches `isoxml` and the branded prescription
shapefile, and what `POST /api/export/package` puts in the monitor folder.
So the zone polygons have to be given a rate per zone on the way: a file
whose rate column holds `1, 2, 3` is a map of class codes, not of product,
and the monitor will apply it as written.

**Join.** `POST /api/project/join` joins on `state.project["roles"]`, and
only `yield`, `as_applied`, `plan` and `vigor` are joined — a zone layer
given one of those roles would be read as a rate, which it is not. The route
that works today is the other one: `carry` brings extra columns **from the
yield layer's own dataframe** into the joined table, and `analyse_economics`
then takes that column as `zone_column`. So to fit one response curve per
landform, put the zone column on the yield dataset first (QGIS round trip,
or a join outside the app) rather than expecting the zones dataset to be
joined by role. Worth saying plainly in the interface: a terrain zone layer
is a map to export, and the path from it to a per-zone economic fit goes
through the yield layer's columns.

---

## 6 · Building the Terrain tab

### The map

A **layer picker** over the twelve keys of §3, in that order, driven by
`/layers` — never a hard-coded list, because a new layer should appear
without a front-end change. One layer at a time; the legend beside it comes
from the same response and needs no second call.

An **opacity slider** on the image overlay (start at 0.75). The basemap is
the field the farmer recognises; a solid colour sheet hides the very
landmarks that let them judge whether the analysis is right.

A **hillshade toggle**, which appends `?hillshade=1` to the layer URL. The
blend is done server-side at the right strength per layer kind, so the
toggle is one query parameter, not a second overlay at guessed opacity. Grey
it out on the hillshade layer itself.

**Contour lines** from `/contours` as a GeoJSON layer, thin and dark, with
`major` lines twice the weight and labelled with `level_m` in the user's
length unit. A contour interval box that re-fetches with `interval_m`; show
the interval actually used, since a level field ignores the request. The
lines are what a farmer checks the colours against, so they should sit above
every raster layer.

**Markers** for the hills, the low ground and the depressions, from
`/features` or from `summary.features` — the polygons and the summary carry
the same numbers, so the outline can be drawn and the popup filled from one
fetch. A hill's popup: label, `position`, `height_m` above its surroundings,
`area_ha`, `summit_m`. A low's: `depth_m`, `area_ha`, and whether it is
`closed`. A depression's: `area_ha`, `max_depth_m`, `volume_m3` and the
`spill_m` it holds to. Mark closed depressions differently from open lows —
that is the distinction that decides whether water sits or runs.

A **two-click profile tool**: first click sets the start, second the end,
`POST /profile` with `n` around 200, and a chart of `elev_m` against
`distance_m` under the map with the line drawn on it. Let a third click
extend the polyline; the endpoint takes any number of vertices. Show
`length_m` and the total fall. Stations off the field come back `null` —
leave a gap in the chart rather than joining across it, or the chart invents
ground that was never measured.

### The panel

**The findings list**, verbatim, at the top, styled by `level` (`ok`,
`info`, `warning`). They are already the sentences a farmer acts on, and
they carry their numbers with metric units inside the text — see the unit
note below.

**A slope histogram** from `summary.slope.histogram` (20 bins,
`edges`/`counts`), with the class breaks of §4 marked, and the class table
beside it as area and share.

**An aspect rose** from `summary.aspect.sectors`: eight compass sectors plus
a `flat` entry for ground with no aspect. The `flat` share belongs in the
legend, not as a ninth petal — it has no direction.

The elevation histogram (`summary.elevation.histogram`) and the trend
sentence round out the panel: `drop_m` in the length unit,
`direction_label` and `gradient_pct`.

### Units: what to convert and what is ready

Internally everything is metric, as everywhere else in the app. The
front-end has `Units.convert.length()`, `Units.convert.area()` and
`Units.label.*` in `units.js`; the conversions below are the terrain ones.

**Convert — length (`Units.convert.length`, internal m):**
`elevation.min_m` / `max_m` / `mean_m` / `median_m` / `relief_m` / `p05_m` /
`p95_m`, the elevation histogram `edges`, `trend.drop_m`,
`contours.interval_m` and every contour `level_m`, `grid.cell_m`, all the
`*_m` fields of the features (`summit_m`, `height_m`, `bottom_m`, `depth_m`,
`max_depth_m`, `spill_m`), the profile's `distance_m`, `elev_m` and
`length_m`, `wetness.drainage_length_m`, and the legend `vmin`/`vmax` of the
`elevation`, `tpi_small`, `tpi_large` and `depression_depth` layers.

**Convert — area (`Units.convert.area`, internal ha):** `grid.area_ha`,
`interior_ha`, `edge_ring_ha`, every `area_ha` in the slope classes, aspect
sectors, landform classes and features, and `wetness.wet_area_ha`.

**Convert — volume:** `volume_m3`. There is **no volume group in
`units.py`** — the length factor has to be cubed. With the length unit in
feet, multiply by 1 / 0.3048³ = 35.3147 and label it ft³; in metres, show
m³ unchanged. Say which one is on screen, because a pond volume is the one
number nobody can sanity-check by eye.

**Ready to display, do not convert:** every `pct` and `*_pct` (slope means
and classes, aspect shares, landform shares, `trend.gradient_pct`,
`wet_pct`, `ranked_share_pct`, the profile's `slope_pct`) — percent is
percent everywhere. Also `aspect_deg` and `trend.direction_deg` (degrees),
`twi` and the curvatures (dimensionless indices), `hillshade` (0…1),
`flow_acc` (a count of cells; multiply by the cell area for a catchment), the landform and zone codes, and every colour.

**Slope in degrees is optional and local.** The app stores and returns
percent; if the tab offers a degrees toggle it converts on the spot with
`degrees = atan(pct / 100) × 180 / π` and never sends degrees back.

**Sending values the other way:** `cell_m`, `smooth_m`,
`contour_interval_m`, `min_feature_height_m` and the TPI radii all go to the
server in metres, so run the boxes through `Units.toInternal.length()` and
label them with `Units.label.length()`, exactly as the Cleaning and Trial
tabs do with their own lengths.

**One thing the interface must not convert: the findings.** Their numbers
are written into the sentence, because the sentence *is* the finding and
rewriting it would mean re-deciding what it says. Show them as they come.
Every number in them also exists in the structured summary for a panel that
converts.

The same goes for `source.notes` — the remarks the gridding and the raster
reader left behind, which reach the panel as `info` findings. They come in
the unit set the request was made in, because the analysis keeps what each
one *states* in `source.note_facts` (`{"key": "smoothing", "smooth_m":
10.0, …}`) and writes the sentence again on every read. Show them as they
come too; if something outside the app needs to say them itself,
`agrosuite.terrain.notes.render(facts, units)` is the one place that
turns a fact into its sentence.

Which is why the unit set travels the other way instead. `POST
/api/terrain/{id}/yield` and the profile's `values_meta.note` take a
`units` body field — the shape of `UNIT_PRESETS['canada']` — and write
their sentences in it, so what comes back is already in the units it will
be read in. Send `Units.get()` with the request and post again when the
reader switches preset; the grid is already in memory, so it costs a
round trip and nothing else. `agrosuite.core.units.Phrase` is what renders
them, and it is the piece to reuse anywhere else sentences carry numbers.

---

## 7 · The MCP tools

Two tools on the terrain surface, in `agrosuite/mcp_server.py`. Both call
the endpoints above, so an assistant can do no more than a person clicking.

**`analyse_terrain`** — `dataset_id` (required), and `cell_m`, `smooth_m`,
`contour_interval_m`, `min_feature_height_m`, all in metres. Calls `POST
/api/terrain/analyze` and answers in sentences rather than JSON: the
character of the field and its total relief, which way it falls and how
steeply, the hills, the low ground and the closed depressions with their
positions, sizes, heights or depths and the volume a depression holds, the
share above 10 % slope, the likely wet area, and the findings verbatim.
Every number is written with its unit, because a height whose unit was left
for the assistant to guess is worse than no height — **and the unit is the
one the app is showing**, not the metric store. The tool reads the set from
`GET /api/units`, names it in its first line, and writes its own lines
through the same `Phrase` the findings it quotes were written with: an
answer that said "60.5 ha" above a finding saying "34.6 ft of relief" would
be describing one field in two systems, to someone who has one of them on
screen. The options still go **in** in metres, like every other endpoint
here. Each option defaults to unset rather than 0, since
`smooth_m: 0` is a real request to read the raw surface. A file with no
usable altitude comes back as the analyser's own sentence, which already
names what to open instead.

**`terrain_zones`** — `dataset_id` and `by` (required), `kind` (default
`points`). Calls `POST /api/terrain/{id}/zones` and reports the new
dataset's id, label, row count, geometry and zone labels, so the
conversation goes straight on to giving it a role and exporting it. Run
`analyse_terrain` on the dataset first; asking before that is refused with
the sentence that says so.

See [claude-integration.md](claude-integration.md) for the setup and the
rest of the tool surface.
