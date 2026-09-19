# Machine

A machine's day, read from the logger that rode on it.

A yield or as-applied file says what the machine put down. A telemetry
logger says what the day cost: where the time went, the diesel, the DEF, how
long the engine ran with the machine standing still, and where the tyres
crossed the crop. The Machine tab reads the log of an HBM SoMat eDAQ logger
(`.sie`, SIE 1.0) — the engine's J1939 CAN bus beside a GPS of its own, at
2 Hz, usually for a whole day — and answers those questions.

This document is the reference for the reader, the analysis and the API.
Every number below was checked against a made-up day with known answers
(`agrosuite/machine/synthetic.py`) and against a real day's log from a
self-propelled sprayer.

---

## 1 · Reading the file

`agrosuite/formats/somat.py`. Open a `.sie` like any other file; the
registry recognises it by its extension and its first block.

**The stream.** Blocks of `size | group | 0x51EDA7A0 | payload | checksum |
size`, big-endian. Group 0 carries XML metadata; every other group carries
one channel's samples, decoded as an `int64` count followed by that many
`float64`, little-endian (decoder 2). A last block cut short — the logger
lost power — is reported and the rest read; a broken block anywhere else is
refused, with the byte it broke at.

**The metadata** is written as the log grows, so its root is never closed;
the reader closes it and parses it as XML. Per channel it declares the
group, sample rate, units, data mode, decoder, and — what matters most — the
value the logger writes when it has no reading
(`somat:invalid_data_output_value`) and the channel's physical range.

**Data modes.** A test may write the same channels twice (the first logger
read has `__dm` and `Omni_2022`); `__dm` is read and the other named in the notes.

| Channel | Column | Stored as |
|---|---|---|
| `latitude`, `longitude` | `lat`, `lon` | degrees WGS84 |
| `altitude` | `elev_m` | m |
| `ground_heading` | `heading_deg` | degrees |
| `ground_speed_mph` | `speed_kmh` | km/h |
| `DeltaD` | `distance_m` | m per sample |
| `EngFuelRate` | `fuel_rate_lh` (also `value`) | L/h |
| `DeltaF` | `fuel_l` | L per sample |
| `TotalFuelUsed` | `fuel_counter_l` | L, unwrapped |
| `DistTraveled` | `distance_counter_m` | m, unwrapped |
| `Def` | `def_level_pct` | % of the tank |
| `EngSpeed` | `engine_rpm` | rpm |
| `EngPercentLoadAtCurrentSpeed` | `engine_load_pct` | % |
| `EngCoolantTemp` | `coolant_c` | °C |
| `ElectricalPotential` / `input_voltage` | `battery_v` / `logger_v` | V |
| `EngTotalHoursOfOperation` | `engine_hours` | h |
| `number_of_satellites_in_use` | `satellites` | count |
| anything else | `ch_<name>` | as logged |

The `utc_*` channels set the clock and are dropped; `RunTrigger` is always 1
and dropped too.

**Three things the file does not do for you.**

* *Missing readings are numbers*: −1 L/h, 1000 °C, −500 V, 181° of
  longitude, 255 satellites. Each channel declares its own; they become NaN,
  and so does anything outside the channel's declared range.
* *Counters wrap* at the top of their declared range. `DistTraveled` on the
  first sprayer read goes back to 0 after 100 000 ft — more than once in a
  day, so read at face value most of the distance is lost. The totals are
  rebuilt from the
  increments (`DeltaD`, `DeltaF`), which never wrap, and checked against the
  unwrapped counter, the fuel rate integrated over time and the GPS. On a
  real day the three fuel figures agree to a hundredth of a litre, and the
  distance by increments and by counter to within half a percent of the GPS.
* *The logger's clock is local and drifts.* The GPS UTC channels give the
  offset — on the first log read, six hours and 23 seconds: the time
  zone (UTC−6) and a 23 s lag. Times are set to GPS time in the logger's zone; an offset that is
  not a whole time zone within five minutes is left alone with a note.

`meta.extra["somat"]` records what was checked: the channel inventory with
invalid counts, the three fuel and distance totals, the counter wraps, the
clock drift and zone, the logger's module and serial.

## 2 · The day

`agrosuite/machine/analysis.py`, `analyse(dataset, settings, boundary=...)`.

**Activities.** Every sample is one of:

| Key | Label | Rule |
|---|---|---|
| `off` | Engine off | engine speed missing or under 400 rpm |
| `idle` | Stopped, engine running | speed (5-sample rolling median) under 1 km/h |
| `field` | Working in the field | moving, inside the boundary |
| `road` | On the road | moving, outside the boundary |

Without a boundary, `field` is moving at up to `road_speed_kmh` (25 km/h)
and `road` faster — the yard counts as field, and the findings say so. A
stop shorter than 10 s between two stretches of the same activity is part
of it: a pause at a headland is still field work.

**Per activity**: time, share of the time with the engine running,
distance, diesel (sum of `fuel_l`), diesel per hour. **Fuel**: total, per
hectare the boom covered (field diesel, and all-in), per km on the road,
cost at the price given. The area covered is the union of the boom's swath
along the field path, clipped to the boundary — overlaps once.

**DEF.** The level is read through a one-minute rolling median (it sloshes
with every turn and reads in steps — 0.4 % on the first logger read); a sustained rise of
more than 5 points is a refill, and the drops either side are added. With
the tank's capacity the drop is litres and a share of the diesel. A drop of
fewer than three gauge steps is flagged as rough.

**Engine and speed**: median, 95th percentile and maximum of engine speed,
load, coolant and battery with the engine running; time at 90 % load or
more; the hour meter's advance beside the logged running time; working
speed (median, 10th–90th percentile) and top road speed; GPS satellites.

**Channel statistics**, the way InField reports them, for the whole log and
for each activity: n, duration, min and max with their times, peak to peak,
mean, median, RMS, standard deviation (n−1), variance, skewness, kurtosis
(the plain fourth-moment ratio, 3 for a Gaussian) and crest factor.

## 3 · The crop under the tyres

`agrosuite/machine/trampling.py`.

**The tyre** is read off what is written on it: a metric size
(`380/90R46`, `VF 380/90 R46 173D`, `320/85-38`) gives its section width in
millimetres, an imperial one (`18.4R38`, `13.6-38`, `15.5/80-24`) in inches,
and a width with its unit (`380 mm`, `15 in`, `0.38 m`) is taken as it is.
Anything else, or a width outside 0.10–1.20 m, is refused in words.

**The strips.** The field path — every stretch of `field`, split where the
GPS drops out or jumps more than 50 m, and averaged over five samples either
side to take the uncorrected GPS's wander out — is cut into pieces of 40
samples. Each piece gets the band between a buffer out to half the track
width plus half the crushed width and one in to half the track width minus
it, with flat ends; the ground is the union of every piece's band. The order
matters: taking the band of the union of the paths would let the gap between
one pass's wheels erase ground a neighbouring pass crushed. The crushed width
per side is one tyre when the rear wheels run in the front ones' tracks, two
when they do not.

Checked by hand: a straight kilometre with 380 mm tyres on a 120 in track is
760 m²; the same line driven again 5 cm off is 860 m²; driven again in its
own tracks, 760; with the rear wheels off the front tracks, 1520.

**The numbers**: area crushed inside the field, its share of the field (the
boundary's area, else the area the boom covered), the distance × width
figure — every metre counted as fresh crop — as the upper bound, and the
share of the path that fell on ground already crossed. With a yield and the
share of the crop under a tyre that is lost, the crop lost in kg, per
hectare of the whole field and in money.

**What it cannot know** is how much of the crop under a tyre is lost: nearly
all of it late in the season, much less early. That share is the user's
(default 100 %). And the logger's GPS has no RTK correction: a tramline
driven again a metre off counts as fresh crop, so where tramlines were
reused the area is an upper bound. Both are said beside the result.

## 4 · API

All numbers metric; findings in the units on screen.

| Method | Path | |
|---|---|---|
| `POST` | `/api/machine/demo` | Load the made-up day and its boundary. Returns `{log, boundary, suggested}` |
| `POST` | `/api/machine/analyze` | Analyse a log. Returns `{dataset_id, summary, layers}` |
| `GET` | `/api/machine/{id}` | The stored summary, restated; `layers` while in memory |
| `GET` | `/api/machine/{id}/layers` | The map layers, while in memory |
| `GET` | `/api/machine/{id}/statistics` | The channel statistics as rows |
| `GET` | `/api/machine/{id}/statistics.csv` | The same, as a download |
| `POST` | `/api/machine/{id}/export` | `{vector_format: "shapefile" \| "geojson"}` → the zip |

`analyze` takes `dataset_id` and, all optional: `boundary_id`, `yield_id`,
`boom_width_m`, `tyre_size` or `tyre_width_m`, `track_width_m`,
`rear_follows_front`, `def_tank_l`, `fuel_price` and `def_price` (per
litre), `crop_price_per_kg` (else the project's price), `loss_fraction`
(0–1, or a percentage), `yield_kg_ha` (else the yield map's mean inside the
boundary), `idle_speed_kmh`, `road_speed_kmh`. Unknown keys are refused.

**Layers**, GeoJSON in WGS84: `track` — one line per stretch of field or
road, with `activity`, `label`, `start`, `end`, `duration_s`,
`distance_m`, `fuel_l`; `stops` — one point per stop (`idle` or `off`)
with the same properties; `strips` — the crushed ground, one
(multi)polygon, with `area_ha` and `lost_kg`. Drawn to 1 m for the path and
0.10 m for the strips on screen, 0.01 m in the export; areas are always
measured on the full geometry.

The summary is kept with the project (`entry.reports["machine"]`); the
layers and statistics for the four most recent analyses are kept in memory.

**MCP**: `analyse_machine` takes the same settings and answers with the
findings and the split by activity.

## 5 · The export

`<log>_machine/`, zipped: `activity_track`, `stops` and `tyre_strips` as
shapefiles (or GeoJSON), `samples.csv` (every sample with its activity and
every channel, metric), `statistics.csv`, `summary.json` and a `README.txt`
naming each file and listing the findings.
