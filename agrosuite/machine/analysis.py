"""A machine's day, read from its telemetry.

Every sample of the log is given one of four activities — working in the
field, stopped with the engine running, on the road, engine off — and the
time, distance, fuel and DEF of the day are split between them. That split
is the finding most logs are worth reading for: a sprayer that spent two of
its six engine hours standing still is a different day from one that spent
two on the road, and the fuel total alone says neither.

Where the field is is decided by a boundary when one is loaded — inside is
field work, outside is road — and by speed when there is none: the road is
what is faster than the machine ever sprays. The boundary is the firmer of
the two, and the findings say which was used.

The summary is metric and JSON-safe, like every analysis in the app; the
sentences are written from it in whatever unit set the reader is in, so the
unit picker rewrites them without re-running anything.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy as np
import pandas as pd

from ..core import schema as sch
from ..core.units import Phrase
from . import trampling as tr

#: The four activities, in the order they are shown.
ACTIVITIES = ("field", "idle", "road", "off")

ACTIVITY_LABELS = {
    "field": "Working in the field",
    "idle": "Stopped, engine running",
    "road": "On the road",
    "off": "Engine off",
}

#: Samples, either side, in the rolling median that takes GPS speed noise out
#: before deciding whether the machine is moving.
SPEED_SMOOTH_SAMPLES = 2

#: A rise in the DEF level of more than this many points, sustained over the
#: smoothing window, is a refill rather than slosh.
DEF_REFILL_POINTS = 5.0

#: Seconds in the rolling median the DEF level is read through: the gauge
#: sloshes with every turn, and a minute of it is still.
DEF_SMOOTH_S = 60.0

#: A jump between consecutive positions longer than this, in metres, is a
#: gap in the GPS rather than travel; the path is split there.
MAX_STEP_M = 50.0

#: Channels the statistics table covers, with their units as stored.
STAT_CHANNELS: tuple[tuple[str, str], ...] = (
    (sch.SPEED, "km/h"),
    ("engine_rpm", "rpm"),
    ("engine_load_pct", "%"),
    ("fuel_rate_lh", "L/h"),
    ("coolant_c", "°C"),
    ("battery_v", "V"),
    ("logger_v", "V"),
    ("def_level_pct", "%"),
    (sch.ELEVATION, "m"),
    ("satellites", "sats"),
    ("fuel_l", "L/sample"),
    (sch.DISTANCE, "m/sample"),
    ("engine_hours", "h"),
)


# ==========================================================================
# Settings
# ==========================================================================

@dataclass
class MachineSettings:
    """What the analysis needs to know that the log does not say. Metric."""

    #: Boom width: the ground worked per metre travelled in the field.
    boom_width_m: float = 0.0
    #: Tyre section width, or the size as on the sidewall (parsed when the
    #: width is not given).
    tyre_width_m: float = 0.0
    tyre_size: str = ""
    #: Centre to centre of the left and right wheels.
    track_width_m: float = 0.0
    rear_follows_front: bool = True
    #: DEF tank capacity: turns the level's drop into litres.
    def_tank_l: float = 0.0
    #: Prices per litre and per kg of crop, in the project's currency.
    fuel_price: float | None = None
    def_price: float | None = None
    crop_price_per_kg: float | None = None
    #: Share of the crop under a tyre that is lost (0–1).
    loss_fraction: float = 1.0
    #: Yield of the crop the tyres crossed; overrides a yield map.
    yield_kg_ha: float | None = None
    #: Below this the machine is stopped.
    idle_speed_kmh: float = 1.0
    #: Without a boundary, faster than this is the road.
    road_speed_kmh: float = 25.0
    #: Engine speed at and above which the engine is running.
    engine_on_rpm: float = 400.0
    #: A stop shorter than this, between two stretches of the same activity,
    #: is part of that activity — a pause at a headland is still field work.
    min_stop_s: float = 10.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "MachineSettings":
        data = dict(data or {})
        known = {f.name for f in fields(cls)}
        kwargs: dict[str, Any] = {}
        for key, value in data.items():
            if key not in known or value is None or value == "":
                continue
            if key in ("tyre_size",):
                kwargs[key] = str(value)
            elif key == "rear_follows_front":
                kwargs[key] = bool(value) if not isinstance(value, str) else value.lower() not in ("0", "false", "no")
            else:
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    raise ValueError(f"'{key}' must be a number, not {value!r}.")
                if not math.isfinite(number):
                    raise ValueError(f"'{key}' must be a finite number.")
                if number < 0:
                    raise ValueError(f"'{key}' cannot be negative.")
                kwargs[key] = number
        settings = cls(**kwargs)
        if settings.loss_fraction > 1.0:
            # Typed as a percentage.
            settings.loss_fraction = settings.loss_fraction / 100.0
        if settings.loss_fraction > 1.0:
            raise ValueError("The share of crop lost under a tyre is at most 100 %.")
        if not settings.tyre_width_m and settings.tyre_size:
            settings.tyre_width_m = tr.parse_tyre(settings.tyre_size)["width_m"]
        return settings

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ==========================================================================
# Activities
# ==========================================================================

def sample_interval(df: pd.DataFrame, rate_hz: float | None = None) -> float:
    """Seconds between samples: the declared rate, else the log's own clock."""
    if rate_hz and rate_hz > 0:
        return 1.0 / float(rate_hz)
    if sch.ELAPSED in df and df[sch.ELAPSED].notna().sum() > 2:
        step = float(np.nanmedian(np.diff(df[sch.ELAPSED].to_numpy())))
        if step > 0:
            return step
    return 0.5


def _runs(values: np.ndarray) -> list[tuple[int, int, Any]]:
    """``(start, stop, value)`` for every run of equal values."""
    if not len(values):
        return []
    change = np.flatnonzero(values[1:] != values[:-1]) + 1
    starts = np.concatenate([[0], change])
    stops = np.concatenate([change, [len(values)]])
    return [(int(a), int(b), values[a]) for a, b in zip(starts, stops)]


def _absorb_short_stops(activity: np.ndarray, samples: int) -> np.ndarray:
    out = activity.copy()
    runs = _runs(out)
    for i, (a, b, value) in enumerate(runs):
        if value != "idle" or b - a >= samples or i == 0 or i == len(runs) - 1:
            continue
        before, after = runs[i - 1][2], runs[i + 1][2]
        if before == after and before in ("field", "road"):
            out[a:b] = before
    return out


def classify(df: pd.DataFrame, settings: MachineSettings, *, dt: float,
             boundary=None) -> tuple[np.ndarray, str]:
    """One activity per sample, and what field and road were told apart by."""
    n = len(df)
    rpm = df["engine_rpm"] if "engine_rpm" in df else None
    if rpm is not None and rpm.notna().any():
        engine_on = (rpm.fillna(0.0) >= settings.engine_on_rpm).to_numpy()
    else:
        engine_on = np.ones(n, dtype=bool)

    if sch.SPEED in df and df[sch.SPEED].notna().any():
        speed = df[sch.SPEED].astype(float)
    elif sch.DISTANCE in df:
        speed = df[sch.DISTANCE].astype(float) / dt * 3.6
    else:
        speed = pd.Series(np.zeros(n))
    window = 2 * SPEED_SMOOTH_SAMPLES + 1
    smooth = speed.rolling(window, center=True, min_periods=1).median().fillna(0.0).to_numpy()
    moving = smooth >= settings.idle_speed_kmh

    if boundary is not None and {sch.X, sch.Y}.issubset(df.columns):
        import shapely

        x = df[sch.X].to_numpy(dtype=float)
        y = df[sch.Y].to_numpy(dtype=float)
        located = np.isfinite(x) & np.isfinite(y)
        inside = np.zeros(n, dtype=bool)
        inside[located] = shapely.contains_xy(boundary, x[located], y[located])
        in_field, basis = inside, "boundary"
    else:
        in_field, basis = smooth <= settings.road_speed_kmh, "speed"

    activity = np.where(~engine_on, "off",
                        np.where(~moving, "idle",
                                 np.where(in_field, "field", "road"))).astype(object)
    activity = _absorb_short_stops(activity, max(1, int(round(settings.min_stop_s / dt))))
    return activity, basis


def field_runs(df: pd.DataFrame, activity: np.ndarray, which: str = "field") -> list[np.ndarray]:
    """The path, in metres, of every stretch of one activity — split where
    the GPS drops out or jumps, so no line is drawn across a gap."""
    if not {sch.X, sch.Y}.issubset(df.columns):
        return []
    xy_all = df[[sch.X, sch.Y]].to_numpy(dtype=float)
    out = []
    for a, b, value in _runs(activity):
        if value != which:
            continue
        xy = xy_all[a:b]
        ok = np.isfinite(xy).all(axis=1)
        if not ok.any():
            continue
        # Split at missing positions and at jumps.
        cut = np.zeros(len(xy), dtype=bool)
        cut[~ok] = True
        step = np.full(len(xy), 0.0)
        step[1:] = np.hypot(*np.diff(xy, axis=0).T)
        cut |= ~np.isfinite(step) | (step > MAX_STEP_M)
        piece_start = 0
        for i in list(np.flatnonzero(cut)) + [len(xy)]:
            piece = xy[piece_start:i]
            piece = piece[np.isfinite(piece).all(axis=1)]
            if len(piece) >= 2:
                out.append(piece)
            piece_start = i if i < len(xy) and ok[i] else i + 1
    return out


# ==========================================================================
# DEF
# ==========================================================================

def def_usage(level: pd.Series, engine_on: np.ndarray, dt: float,
              tank_l: float | None = None) -> dict[str, Any]:
    """How far the DEF level fell over the log, across any refills.

    The gauge sloshes with every turn and moves in steps, so the level is
    read through a one-minute rolling median; a sustained rise of more than
    :data:`DEF_REFILL_POINTS` is a refill, and what fell before it is added
    to what fell after.
    """
    valid = level.notna().to_numpy() & engine_on
    if valid.sum() < max(10, int(120 / dt)):
        return {"available": False}
    raw = level.to_numpy(dtype=float)[valid]
    window = max(3, int(round(DEF_SMOOTH_S / dt)))
    smooth = pd.Series(raw).rolling(window, center=True, min_periods=1).median().to_numpy()
    used, refills = 0.0, 0
    start = low = float(smooth[0])
    for value in smooth[1:]:
        value = float(value)
        if value > start and low == start:
            # Still rising — the first minute, or a refill the median is
            # climbing through in steps: the segment starts at its top.
            start = low = value
        elif value > low + DEF_REFILL_POINTS:
            used += start - low
            refills += 1
            start = low = value
        elif value < low:
            low = value
    end = float(smooth[-1])
    used += max(start - end, 0.0)
    steps = np.diff(np.unique(np.round(raw, 3)))
    step = float(pd.Series(steps[steps > 0]).mode().iloc[0]) if (steps > 0).any() else None
    litres = used / 100.0 * tank_l if tank_l and tank_l > 0 else None
    return {
        "available": True,
        "start_pct": float(smooth[0]),
        "end_pct": end,
        "drop_pct": used,
        "refills": refills,
        "step_pct": step,
        "tank_l": tank_l or None,
        "litres": litres,
    }


# ==========================================================================
# Channel statistics, the way InField reports them
# ==========================================================================

def _clock(ts) -> str:
    return pd.Timestamp(ts).strftime("%H:%M:%S.%f")[:-5]


def channel_stats(values: np.ndarray, times: np.ndarray, dt: float) -> dict[str, Any]:
    """n, duration, min/max with their times, peak-to-peak, mean, median,
    RMS, sample standard deviation, variance, skewness, kurtosis (the plain
    fourth-moment ratio — 3 for a Gaussian, as InField has it) and crest
    factor."""
    x = np.asarray(values, dtype=float)
    ok = np.isfinite(x)
    x, t = x[ok], np.asarray(times)[ok]
    n = len(x)
    if n == 0:
        return {"n": 0}
    mean = float(x.mean())
    sd = float(x.std(ddof=1)) if n > 1 else 0.0
    rms = float(np.sqrt(np.mean(x ** 2)))
    m2 = float(np.mean((x - mean) ** 2))
    m3 = float(np.mean((x - mean) ** 3))
    m4 = float(np.mean((x - mean) ** 4))
    i_min, i_max = int(np.argmin(x)), int(np.argmax(x))
    return {
        "n": n, "duration_s": n * dt,
        "min": float(x[i_min]), "t_min": _clock(t[i_min]),
        "max": float(x[i_max]), "t_max": _clock(t[i_max]),
        "peak_to_peak": float(x[i_max] - x[i_min]),
        "mean": mean, "median": float(np.median(x)), "rms": rms,
        "std": sd, "variance": sd ** 2,
        "skewness": m3 / m2 ** 1.5 if m2 > 0 else 0.0,
        "kurtosis": m4 / m2 ** 2 if m2 > 0 else 0.0,
        "crest_factor": float(np.max(np.abs(x)) / rms) if rms > 0 else 0.0,
    }


def statistics_table(df: pd.DataFrame, activity: np.ndarray, dt: float) -> list[dict[str, Any]]:
    """One row per channel for the whole log and for each activity."""
    times = df[sch.TIMESTAMP].to_numpy() if sch.TIMESTAMP in df else np.arange(len(df)) * dt
    selections = [("all", np.ones(len(df), dtype=bool))]
    selections += [(a, activity == a) for a in ACTIVITIES if (activity == a).any()]
    rows = []
    for name, mask in selections:
        for column, unit in STAT_CHANNELS:
            if column not in df:
                continue
            stats = channel_stats(df[column].to_numpy()[mask], times[mask], dt)
            if stats.get("n"):
                rows.append({"selection": name, "channel": column, "unit": unit, **stats})
    return rows


# ==========================================================================
# The day
# ==========================================================================

def _q(values, q) -> float | None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.quantile(values, q)) if len(values) else None


def _sum(values) -> float:
    return float(np.nansum(np.asarray(values, dtype=float)))


@dataclass
class MachineResult:
    summary: dict[str, Any]
    activity: np.ndarray
    footprint: Any = None
    runs: list[np.ndarray] | None = None
    statistics: list[dict[str, Any]] | None = None


def analyse(dataset, settings: MachineSettings | dict | None = None, *,
            boundary=None, boundary_area_ha: float | None = None,
            yield_kg_ha: float | None = None, yield_source: str | None = None) -> MachineResult:
    """Read one telemetry dataset.

    ``boundary`` is a shapely polygon in the dataset's metric CRS (see
    :func:`agrosuite.machine.boundary_polygon`); ``yield_kg_ha`` the yield of
    the crop the tyres crossed, from a yield map, when the settings do not
    give one.
    """
    if not isinstance(settings, MachineSettings):
        settings = MachineSettings.from_dict(settings)
    df = dataset.df
    somat = (dataset.meta.extra or {}).get("somat", {})
    dt = sample_interval(df, somat.get("rate_hz"))
    activity, basis = classify(df, settings, dt=dt, boundary=boundary)
    engine_on = activity != "off"

    # -- time, distance, fuel per activity ----------------------------------
    distance = df[sch.DISTANCE].to_numpy(dtype=float) if sch.DISTANCE in df else None
    if distance is None and {sch.X, sch.Y}.issubset(df.columns):
        steps = np.hypot(*np.diff(df[[sch.X, sch.Y]].to_numpy(dtype=float), axis=0).T)
        distance = np.concatenate([[0.0], np.where(steps > MAX_STEP_M, 0.0, steps)])
    fuel = df["fuel_l"].to_numpy(dtype=float) if "fuel_l" in df else None
    if fuel is None and "fuel_rate_lh" in df:
        fuel = df["fuel_rate_lh"].to_numpy(dtype=float) / 3600.0 * dt
    speed = df[sch.SPEED].to_numpy(dtype=float) if sch.SPEED in df else None

    rows = []
    for key in ACTIVITIES:
        mask = activity == key
        if not mask.any():
            continue
        time_s = float(mask.sum() * dt)
        litres = _sum(fuel[mask]) if fuel is not None else None
        metres = _sum(distance[mask]) if distance is not None else None
        rows.append({
            "key": key, "label": ACTIVITY_LABELS[key], "time_s": time_s,
            "distance_m": metres, "fuel_l": litres,
            "fuel_lh": litres / (time_s / 3600.0) if litres is not None and time_s > 0 else None,
        })
    total_time = float(len(df) * dt)
    engine_time = float(engine_on.sum() * dt)
    for row in rows:
        row["share_time"] = row["time_s"] / total_time if total_time else None
        row["share_engine_time"] = (row["time_s"] / engine_time
                                    if engine_time and row["key"] != "off" else None)
    by = {row["key"]: row for row in rows}
    total_fuel = _sum(fuel) if fuel is not None else None
    for row in rows:
        row["share_fuel"] = (row["fuel_l"] / total_fuel
                             if total_fuel and row["fuel_l"] is not None else None)

    # -- the ground worked -------------------------------------------------------
    runs = field_runs(df, activity)
    covered_ha = None
    if settings.boom_width_m > 0 and runs:
        shape = tr.swath(tr.smooth_runs(runs), settings.boom_width_m, clip=boundary)
        covered_ha = float(shape.area) / 10_000.0 if shape is not None else None
    field_area_ha = boundary_area_ha if boundary_area_ha else covered_ha

    field = by.get("field", {})
    field_fuel = field.get("fuel_l")
    fuel_summary = {
        "total_l": total_fuel,
        "per_ha_field": (field_fuel / covered_ha) if field_fuel and covered_ha else None,
        "per_ha_all": (total_fuel / covered_ha) if total_fuel and covered_ha else None,
        "road_l_per_km": ((by["road"]["fuel_l"] / (by["road"]["distance_m"] / 1000.0))
                          if by.get("road", {}).get("distance_m") and by["road"].get("fuel_l") is not None
                          else None),
        "cost": total_fuel * settings.fuel_price if total_fuel and settings.fuel_price else None,
        "sources": {k: somat.get(f"fuel_from_{k}") for k in ("increments", "counter", "rate")},
    }

    # -- DEF ---------------------------------------------------------------------
    def_summary = ({"available": False} if "def_level_pct" not in df else
                   def_usage(df["def_level_pct"], engine_on, dt, settings.def_tank_l))
    if def_summary.get("litres") is not None:
        def_summary["share_of_fuel"] = (def_summary["litres"] / total_fuel) if total_fuel else None
        def_summary["cost"] = (def_summary["litres"] * settings.def_price
                               if settings.def_price else None)

    # -- engine, speed, GPS ---------------------------------------------------------
    def spread(column, mask=None):
        if column not in df:
            return None
        values = df[column].to_numpy(dtype=float)
        values = values[mask] if mask is not None else values
        if not np.isfinite(values).any():
            return None
        return {"median": _q(values, 0.5), "p95": _q(values, 0.95),
                "max": float(np.nanmax(values)), "min": float(np.nanmin(values))}

    load = df["engine_load_pct"].to_numpy(dtype=float) if "engine_load_pct" in df else None
    engine = {
        "rpm": spread("engine_rpm", engine_on),
        "load": spread("engine_load_pct", engine_on),
        "coolant": spread("coolant_c", engine_on),
        "battery": spread("battery_v", engine_on),
        "time_over_90_load_s": float(((load >= 90) & engine_on).sum() * dt) if load is not None else None,
    }
    hours = df["engine_hours"].dropna() if "engine_hours" in df else pd.Series(dtype=float)
    engine["hour_meter_h"] = float(hours.iloc[-1] - hours.iloc[0]) if len(hours) > 1 else None

    speed_summary = {}
    if speed is not None:
        for key in ("field", "road"):
            mask = activity == key
            if mask.any():
                speed_summary[key] = {"median": _q(speed[mask], 0.5), "p10": _q(speed[mask], 0.1),
                                      "p90": _q(speed[mask], 0.9),
                                      "max": float(np.nanmax(speed[mask]))}

    gps = {"fix_share": somat.get("gps_fix_share")}
    if "satellites" in df:
        sats = df["satellites"].to_numpy(dtype=float)[engine_on]
        gps.update(satellites_median=_q(sats, 0.5),
                   satellites_min=float(np.nanmin(sats)) if np.isfinite(sats).any() else None)

    # -- trampling -------------------------------------------------------------------
    setup = tr.TyreSetup(settings.tyre_width_m, settings.track_width_m, settings.rear_follows_front)
    use_yield = settings.yield_kg_ha or yield_kg_ha
    trampled = tr.assess(runs, setup, field_area_ha=field_area_ha, clip=boundary,
                         yield_kg_ha=use_yield, loss_fraction=settings.loss_fraction,
                         crop_price_per_kg=settings.crop_price_per_kg)
    footprint = trampled.pop("geometry", None)
    if trampled.get("available"):
        trampled["yield_source"] = ("typed" if settings.yield_kg_ha else yield_source) if use_yield else None
        if settings.tyre_size:
            trampled["tyre_size"] = settings.tyre_size

    ts = df[sch.TIMESTAMP] if sch.TIMESTAMP in df else None
    summary = {
        "name": dataset.meta.name,
        "span": {
            "start": str(ts.iloc[0]) if ts is not None and len(ts) else None,
            "end": str(ts.iloc[-1]) if ts is not None and len(ts) else None,
            "duration_s": total_time,
            "engine_on_s": engine_time,
            "sample_s": dt,
            "samples": int(len(df)),
        },
        "basis": basis,
        "activities": rows,
        "distance_m": _sum(distance) if distance is not None else None,
        "fuel": fuel_summary,
        "def": def_summary,
        "engine": engine,
        "speed": speed_summary,
        "gps": gps,
        "area": {"covered_ha": covered_ha, "boundary_ha": boundary_area_ha,
                 "field_ha": field_area_ha},
        "trampling": trampled,
        "settings": settings.to_dict(),
        "source": {
            "clock_drift_s": somat.get("clock_drift_s"),
            "utc_offset_h": somat.get("utc_offset_h"),
            "logger": somat.get("logger"),
            "invalid": somat.get("invalid"),
            "notes": list(dataset.meta.notes or []),
        },
    }
    summary["findings"] = findings(summary)
    result = MachineResult(summary=summary, activity=activity, footprint=footprint, runs=runs)
    result.statistics = statistics_table(df, activity, dt)
    return result


# ==========================================================================
# What it says
# ==========================================================================

def _row(summary, key) -> dict[str, Any]:
    return next((r for r in summary.get("activities", []) if r["key"] == key), {})


def findings(summary: dict[str, Any], units: dict[str, Any] | None = None) -> list[dict[str, str]]:
    """The sentences, written from the summary alone in ``units``.

    Each is ``{'level': 'ok' | 'info' | 'warning', 'text': ...}``, the same
    shape as the relief's, so the interface lists them the same way.
    """
    say = Phrase(units)
    out: list[dict[str, str]] = []
    span = summary.get("span", {})
    field, idle, road = (_row(summary, k) for k in ("field", "idle", "road"))
    fuel = summary.get("fuel", {})

    if span.get("duration_s"):
        start = str(span.get("start") or "")[11:16]
        end = str(span.get("end") or "")[11:16]
        out.append({"level": "info", "text": (
            f"The log covers {say.duration(span['duration_s'])}"
            + (f", {start} to {end}" if start and end else "")
            + f", with the engine running for {say.duration(span.get('engine_on_s'))}.")})

    if fuel.get("total_l"):
        parts = [f"{say.liquid(r['fuel_l'])} {name}" for r, name in
                 ((field, "working"), (road, "on the road"), (idle, "standing"))
                 if r.get("fuel_l")]
        out.append({"level": "info", "text": (
            f"{say.liquid(fuel['total_l'])} of diesel"
            + (": " + ", ".join(parts) if len(parts) > 1 else "")
            + (f" — {say.money(fuel['cost'])} at the price given" if fuel.get("cost") else "")
            + ".")})

    if idle.get("time_s") and idle.get("share_engine_time"):
        share = idle["share_engine_time"]
        out.append({"level": "warning" if share >= 0.2 else "info", "text": (
            f"The machine stood still with the engine running for {say.duration(idle['time_s'])} "
            f"— {say.percent(share * 100)} of engine time, filling, mixing or waiting"
            + (f", burning {say.liquid(idle['fuel_l'])} at {say.liquid_per_hour(idle['fuel_lh'])}"
               if idle.get("fuel_l") else "")
            + ".")})

    if field.get("time_s"):
        speed = summary.get("speed", {}).get("field", {})
        text = f"Field work: {say.duration(field['time_s'])} over {say.distance(field.get('distance_m'))}"
        if speed.get("median") is not None:
            text += f", at {say.speed(speed['median'])}"
            if say.speed(speed["p10"]) != say.speed(speed["p90"]):
                text += f" (most of it between {say.speed(speed['p10'])} and {say.speed(speed['p90'])})"
        if field.get("fuel_lh"):
            text += f", using {say.liquid_per_hour(field['fuel_lh'])}"
        out.append({"level": "info", "text": text + "."})

    if fuel.get("per_ha_field"):
        area = summary.get("area", {})
        out.append({"level": "info", "text": (
            f"Over the {say.area(area.get('covered_ha'))} the boom covered, working used "
            f"{say.liquid_per_area(fuel['per_ha_field'])}"
            + (f"; with the road and the stops, {say.liquid_per_area(fuel['per_ha_all'])}"
               if fuel.get("per_ha_all") else "")
            + ".")})

    if road.get("distance_m"):
        speed = summary.get("speed", {}).get("road", {})
        out.append({"level": "info", "text": (
            f"On the road: {say.distance(road['distance_m'])} in {say.duration(road['time_s'])}"
            + (f", up to {say.speed(speed['max'], 0)}" if speed.get("max") else "")
            + (f", {say.liquid_per_distance(fuel['road_l_per_km'])}" if fuel.get("road_l_per_km") else "")
            + ".")})

    d = summary.get("def", {})
    if d.get("available"):
        fell = (f"DEF fell from {say.percent(d['start_pct'])} to {say.percent(d['end_pct'])} of the tank"
                if not d.get("refills") else
                f"DEF fell by {d['drop_pct']:.0f} points of the tank across {d['refills']} refill"
                f"{'s' if d['refills'] > 1 else ''}")
        if d.get("litres") is not None:
            fell += f" — {say.liquid(d['litres'])}"
            if d.get("share_of_fuel"):
                fell += f", {d['share_of_fuel'] * 100:.1f} % of the diesel"
            out.append({"level": "info", "text": fell + "."})
        else:
            out.append({"level": "info", "text": (
                fell + ". Give the DEF tank's capacity in the machine profile to have it "
                "in litres and against the diesel.")})
        if d.get("step_pct") and d["drop_pct"] < 3 * d["step_pct"]:
            out.append({"level": "warning", "text": (
                f"The DEF gauge reads in steps of {d['step_pct']:.1f} points, and the level moved "
                "only a few of them: the DEF figure is rough.")})

    load = summary.get("engine", {}).get("time_over_90_load_s")
    if load and load >= 60:
        out.append({"level": "info", "text": (
            f"The engine ran at 90 % load or more for {say.duration(load)}.")})

    t = summary.get("trampling", {})
    if t.get("available") and t.get("area_ha") is not None:
        text = (f"The tyres crossed {say.area(t['area_ha'], 2)} of crop"
                + (f" — {say.percent(t['share_of_field'] * 100, 1)} of the field"
                   if t.get("share_of_field") else ""))
        if t.get("lost_kg") is not None:
            text += (f". At {say.rate(t['yield_kg_ha'])} and {t['loss_fraction'] * 100:.0f} % "
                     f"lost under the tyre, that is {say.rate(t['lost_kg_per_field_ha'])} over the "
                     f"whole field" if t.get("lost_kg_per_field_ha") else "")
            if t.get("lost_value"):
                text += f", {say.money(t['lost_value'])} of crop"
        else:
            text += ". Give a yield — or load the yield map — to put a crop loss on it"
        out.append({"level": "warning" if (t.get("share_of_field") or 0) >= 0.02 else "info",
                    "text": text + "."})
        again = t.get("driven_again_share") or 0.0
        out.append({"level": "info", "text": (
            "The strips are drawn from the logger's own GPS, which has no RTK correction: a "
            "track driven again a metre off its first line counts as fresh crop, so where "
            "tramlines were reused the area is an upper bound"
            + (f". {say.percent(again * 100)} of the path did fall on ground already crossed "
               "in this log, and was counted once" if again >= 0.05 else "")
            + ".")})
    elif t.get("problems"):
        out.append({"level": "info", "text": (
            "No trampling estimate: " + " ".join(t["problems"])
            + " Both go in the machine profile.")})

    if summary.get("basis") == "speed":
        out.append({"level": "warning", "text": (
            f"No field boundary, so field and road were told apart by speed: anything up to "
            f"{say.speed(summary.get('settings', {}).get('road_speed_kmh', 25), 0)} counted as "
            "field work, the yard included. Load the field's boundary for a firmer split.")})

    drift = summary.get("source", {}).get("clock_drift_s")
    if drift is not None and abs(drift) >= 5:
        out.append({"level": "info", "text": (
            f"The logger's clock was {abs(drift):.0f} s {'behind' if drift > 0 else 'ahead of'} "
            "its GPS; every time here is GPS time.")})
    return out


def restate(summary: dict[str, Any], units: dict[str, Any] | None = None) -> dict[str, Any]:
    """The same analysis, its sentences written in another unit set."""
    if not summary:
        return summary
    out = dict(summary)
    out["findings"] = findings(summary, units)
    return out
