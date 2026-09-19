"""A SoMat day made up from scratch, written as a real ``.sie`` file.

Tests need a log whose answers are known, and the demo needs one nobody has
to find: this module lays out a sprayer's day — filling in the yard, the
road out, a field sprayed in passes with a stop in the middle, the road
back — and writes it with every awkward habit of the real logger: the XML
split over many blocks with its root never closed, a distance counter that
wraps several times, the CAN channels at their "missing" values while the
engine is off, a clock 23 s behind the GPS, and a second data mode that
repeats the first.

:func:`synthetic_day` returns the samples and the truth; :func:`write_sie`
writes any set of channels; :func:`demo_log` does both into a file.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SYNC = 0x51EDA7A0

#: Where the made-up field lies: near Olds, Alberta, in UTM 12N.
ORIGIN_UTM = (330_000.0, 5_740_000.0)
UTM_CRS = "EPSG:32612"

RATE_HZ = 2.0
#: Hours between the GPS clock (UTC) and the logger's (MDT), and how far the
#: logger's clock lags.
UTC_OFFSET_H = -6
CLOCK_LAG_S = 23.0


# ==========================================================================
# Writing an SIE stream
# ==========================================================================

@dataclass
class Channel:
    """One channel to write: its samples and what its XML declares."""

    name: str
    values: np.ndarray
    units: str = ""
    invalid: float | None = None
    range_min: float | None = None
    range_max: float | None = None
    rate_hz: float = RATE_HZ
    datamode: str = "__dm"
    description: str = ""


def _block(group: int, payload: bytes) -> bytes:
    size = 20 + len(payload)
    return struct.pack(">III", size, group, SYNC) + payload + struct.pack(">II", 0, size)


def _xml(channels: list[tuple[int, Channel]], start_time: str) -> str:
    head = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<sie version="1.0" xmlns="http://www.somat.com/SIE">\n'
        ' <decoder id="0"><loop>'
        '<read var="size" bits="32" type="uint" endian="big"/>'
        '<read var="group" bits="32" type="uint" endian="big"/>'
        '<read var="syncword" bits="32" type="uint" endian="big" value="0x51EDA7A0"/>'
        '<read var="payload" octets="{$size - 20}" type="raw"/>'
        '<read var="checksum" bits="32" type="uint" endian="big"/>'
        '<read var="size2" bits="32" type="uint" endian="big" value="{$size}"/>'
        '</loop></decoder>\n'
        ' <decoder id="2"><read var="v0" bits="64" type="int" endian="little"/>'
        '<loop var="v0"><read var="v1" bits="64" type="float" endian="little"/><sample/></loop>'
        '</decoder>\n'
        f' <test id="0"><tag id="core:start_time">{start_time}</tag></test>\n'
    )
    body = []
    for cid, (group, ch) in enumerate(channels):
        tags = {
            "core:description": ch.description or ch.name,
            "core:sample_rate": f"{ch.rate_hz:g}",
            "somat:datamode_name": ch.datamode,
            "somat:datamode_type": "time_history",
            "somat:output_units": ch.units,
            "somat:module_type": "edaqxr_lite",
            "somat:connector": f"{ch.name}@can1.edaqxr_lite-560999",
        }
        if ch.invalid is not None:
            tags["somat:invalid_data_output_value"] = f"{ch.invalid:g}"
        if ch.range_min is not None:
            tags["somat:physical_range_min"] = f"{ch.range_min:g}"
        if ch.range_max is not None:
            tags["somat:physical_range_max"] = f"{ch.range_max:g}"
        tag_xml = "".join(f'<tag id="{k}">{v}</tag>' for k, v in tags.items())
        body.append(
            f' <ch test="0" id="{cid}" name="{ch.name}" group="{group}">{tag_xml}'
            f'<dim index="0"><tag id="core:units">s</tag>'
            f'<xform scale="{1.0 / ch.rate_hz:g}" offset="0"/></dim>'
            f'<dim index="1"><tag id="core:units">{ch.units}</tag>'
            f'<data decoder="2" v="1"/></dim></ch>\n'
        )
    # No closing </sie>: the logger never writes one.
    return head + "".join(body)


def write_sie(path: str | Path, channels: list[Channel], start_time: str,
              chunk: int = 500, xml_blocks: int = 5, truncate_bytes: int = 0) -> Path:
    """Write ``channels`` as an SIE stream at ``path``.

    The XML goes out in ``xml_blocks`` pieces and every channel's samples in
    blocks of ``chunk``, interleaved the way a logger writes them. A
    ``truncate_bytes`` above zero leaves an unfinished block at the end.
    """
    path = Path(path)
    numbered = [(100 + i, ch) for i, ch in enumerate(channels)]
    xml = _xml(numbered, start_time).encode("utf-8")
    step = max(1, math.ceil(len(xml) / max(1, xml_blocks)))
    out = bytearray()
    for i in range(0, len(xml), step):
        out += _block(0, xml[i:i + step])
    longest = max(len(ch.values) for _, ch in numbered)
    for first in range(0, longest, chunk):
        for group, ch in numbered:
            part = np.asarray(ch.values[first:first + chunk], dtype="<f8")
            if len(part):
                out += _block(group, struct.pack("<q", len(part)) + part.tobytes())
    if truncate_bytes:
        size = 20 + 8 + 8 * 50
        out += struct.pack(">III", size, 101, SYNC) + b"\0" * max(0, truncate_bytes - 12)
    path.write_bytes(bytes(out))
    return path


# ==========================================================================
# A day
# ==========================================================================

@dataclass
class Day:
    """The samples of a made-up day, in metric, and what is true about it."""

    frame: pd.DataFrame
    boundary_utm: list[tuple[float, float]]
    truth: dict[str, Any] = field(default_factory=dict)

    def boundary_lonlat(self) -> list[tuple[float, float]]:
        from pyproj import Transformer

        to_ll = Transformer.from_crs(UTM_CRS, "EPSG:4326", always_xy=True)
        return [tuple(to_ll.transform(x, y)) for x, y in self.boundary_utm]


def _segment(kind: str, xy: np.ndarray, speed_kmh: float, **engine) -> dict[str, Any]:
    return {"kind": kind, "xy": xy, "speed_kmh": speed_kmh, **engine}


def _still(x: float, y: float, seconds: float) -> np.ndarray:
    return np.tile([x, y], (int(round(seconds * RATE_HZ)), 1)).astype(float)


def _travel(points: np.ndarray, speed_kmh: float) -> np.ndarray:
    """Positions along a polyline at a steady speed, one per sample."""
    seg = np.hypot(*np.diff(points, axis=0).T)
    along = np.concatenate([[0.0], np.cumsum(seg)])
    step = speed_kmh / 3.6 / RATE_HZ
    s = np.arange(0.0, along[-1], step)
    return np.column_stack([np.interp(s, along, points[:, 0]), np.interp(s, along, points[:, 1])])


def _field_path(x0: float, y0: float, passes: int, length: float, spacing: float) -> np.ndarray:
    """Back-and-forth passes joined by half-circle turns of the boom's radius."""
    pts = []
    radius = spacing / 2.0
    for i in range(passes):
        x = x0 + i * spacing
        ys = np.array([y0, y0 + length]) if i % 2 == 0 else np.array([y0 + length, y0])
        pts.extend([(x, ys[0]), (x, ys[1])])
        if i < passes - 1:
            top = i % 2 == 0
            angles = np.linspace(math.pi, 0, 24) if top else np.linspace(-math.pi, 0, 24)
            cy = y0 + length if top else y0
            pts.extend((x + radius + radius * math.cos(a), cy + radius * math.sin(a)) for a in angles[1:-1])
    return np.array(pts)


def synthetic_day(seed: int = 7, *, def_refill: bool = False) -> Day:
    """A sprayer's day: yard, road, field, road, yard. Samples at 2 Hz.

    Field: 10 passes of 300 m, 36.6 m apart (a 120 ft boom), at 14 km/h,
    with a 2 min stop half way; the boundary sits 25 m outside the passes
    so the turns are inside it. Road: 4 km each way at 45 km/h.
    """
    rng = np.random.default_rng(seed)
    ox, oy = ORIGIN_UTM
    boom = 36.576
    fx, fy = ox + 4000.0, oy
    passes, length = 10, 300.0

    # The boundary sits 25 m outside the passes; the road stays west of it
    # and the machine crosses the headland slowly to the first pass and back
    # out from the last, so every "road" sample is outside and every
    # "field" one inside.
    west = fx - 25.0
    yard = (ox, oy + 150.0)
    road_out = _travel(np.array([yard, (west - 40.0, oy + 150.0), (west - 40.0, fy),
                                 (west - 1.0, fy)]), 45.0)
    passes_xy = _field_path(fx, fy, passes, length, boom)
    field = _travel(np.vstack([[(west + 1.0, fy)], passes_xy,
                               [(passes_xy[-1][0], fy - 12.0), (west + 1.0, fy - 12.0)]]), 14.0)
    half = len(field) // 2
    road_back = _travel(np.array([(west - 1.0, fy - 12.0), (west - 40.0, fy - 12.0),
                                  (west - 40.0, oy + 150.0), yard]), 45.0)

    segments = [
        _segment("off", _still(*yard, 30), 0.0),
        _segment("idle", _still(*yard, 15 * 60), 0.0, rpm=1500, load=22, fuel=8.0),
        _segment("road", road_out, 45.0, rpm=2100, load=55, fuel=30.0),
        _segment("field", field[:half], 14.0, rpm=2100, load=60, fuel=28.0),
        _segment("idle", _still(*field[half - 1], 120), 0.0, rpm=900, load=12, fuel=4.0),
        _segment("field", field[half:], 14.0, rpm=2100, load=60, fuel=28.0),
        _segment("road", road_back, 45.0, rpm=2100, load=50, fuel=28.0),
        _segment("idle", _still(*yard, 5 * 60), 0.0, rpm=900, load=10, fuel=4.0),
        _segment("off", _still(*yard, 30), 0.0),
    ]

    rows = []
    for seg in segments:
        n = len(seg["xy"])
        on = seg["kind"] != "off"
        rows.append(pd.DataFrame({
            "activity": seg["kind"],
            "x": seg["xy"][:, 0], "y": seg["xy"][:, 1],
            "speed_kmh": np.full(n, seg["speed_kmh"]),
            "rpm": np.full(n, seg.get("rpm", np.nan) if on else np.nan),
            "load": np.full(n, seg.get("load", np.nan) if on else np.nan),
            "fuel_lh": np.full(n, seg.get("fuel", 0.0) if on else np.nan),
        }))
    frame = pd.concat(rows, ignore_index=True)
    n = len(frame)
    dt = 1.0 / RATE_HZ
    on = frame["activity"] != "off"

    step = np.concatenate([[0.0], np.hypot(*np.diff(frame[["x", "y"]].to_numpy(), axis=0).T)])
    frame["step_m"] = step
    frame["fuel_l"] = np.where(on, frame["fuel_lh"].fillna(0.0) / 3600.0 * dt, 0.0)
    headings = np.degrees(np.arctan2(np.diff(frame.x, prepend=frame.x[0]),
                                     np.diff(frame.y, prepend=frame.y[0]))) % 360
    frame["heading"] = headings

    # DEF: 3 % of the diesel from a 60 L tank, read in 0.4 % steps with slosh.
    def_l = np.cumsum(frame["fuel_l"]) * 0.03
    level = 80.0 - def_l / 60.0 * 100.0
    if def_refill:
        level = np.where(np.arange(n) > n // 2, level + 15.0, level)
    moving = frame["speed_kmh"] > 0
    level = level + np.where(moving, rng.normal(0, 0.6, n), 0.0)
    frame["def_pct"] = np.round(level / 0.4) * 0.4

    east = fx + (passes - 1) * boom + 25.0
    boundary = [(west, fy - 25.0), (east, fy - 25.0),
                (east, fy + length + 25.0), (west, fy + length + 25.0)]

    truth: dict[str, Any] = {"sample_s": dt, "samples": n}
    for kind in ("field", "idle", "road", "off"):
        mask = frame["activity"] == kind
        truth[kind] = {"time_s": float(mask.sum() * dt), "distance_m": float(step[mask].sum()),
                       "fuel_l": float(frame.loc[mask, "fuel_l"].sum())}
    truth["fuel_l"] = float(frame["fuel_l"].sum())
    truth["distance_m"] = float(step.sum())
    truth["def_used_l"] = float(def_l.iloc[-1] - def_l.iloc[0])
    truth["def_used_pct"] = truth["def_used_l"] / 60.0 * 100.0
    field_mask = frame["activity"] == "field"
    truth["field_path_m"] = float(step[field_mask].sum())
    truth["boom_m"] = boom
    return Day(frame=frame, boundary_utm=boundary, truth=truth)


def day_channels(day: Day, *, start: pd.Timestamp, gps_noise_m: float = 0.3,
                 distance_top_ft: float = 10_000.0, second_datamode: bool = True,
                 seed: int = 11) -> list[Channel]:
    """The day as the logger would record it, one :class:`Channel` each."""
    from pyproj import Transformer

    rng = np.random.default_rng(seed)
    f = day.frame
    n = len(f)
    on = (f["activity"] != "off").to_numpy()
    x = f.x.to_numpy() + rng.normal(0, gps_noise_m, n)
    y = f.y.to_numpy() + rng.normal(0, gps_noise_m, n)
    lon, lat = Transformer.from_crs(UTM_CRS, "EPSG:4326", always_xy=True).transform(x, y)

    t = np.arange(n) / RATE_HZ
    gps = pd.DatetimeIndex(start + pd.to_timedelta(t + CLOCK_LAG_S, unit="s")
                           - pd.Timedelta(hours=UTC_OFFSET_H))
    feet = f["step_m"].to_numpy() / 0.3048
    dist_ft = np.cumsum(feet) % distance_top_ft
    fuel = f["fuel_l"].to_numpy()

    def can(values, invalid):
        v = np.asarray(values, dtype=float).copy()
        v[~on | ~np.isfinite(v)] = invalid
        return v

    delta_f = np.where(on, fuel, -1e-4)
    channels = [
        Channel("EngSpeed", can(f.rpm, -1), "RPM", -1, 0, 8031.875),
        Channel("EngPercentLoadAtCurrentSpeed", can(f.load, -1), "%", -1, 0, 250),
        Channel("EngFuelRate", can(f.fuel_lh, -1), "L/hr", -1, 0, 3212.75),
        Channel("EngCoolantTemp", can(np.full(n, 82.0), 1000), "C", 1000, -40, 210),
        Channel("ElectricalPotential", can(np.full(n, 14.1), -500), "V", -500, 0, 3212.75),
        Channel("EngTotalHoursOfOperation", can(150.0 + np.cumsum(on) / RATE_HZ / 3600, -1), "H", -1, 0, 9999),
        Channel("Def", can(f.def_pct, -1), "%", -1, 0, 250),
        Channel("DeltaF", delta_f, "L", None, 0, 100),
        Channel("TotalFuelUsed", np.cumsum(fuel), "L", None, 0, 5000),
        Channel("DeltaD", feet, "ft", None, 0, 1000),
        Channel("DistTraveled", dist_ft, "Ft", None, 0, distance_top_ft),
        Channel("latitude", lat, "degrees", 91, -90, 90),
        Channel("longitude", lon, "degrees", 181, -180, 180),
        Channel("altitude", np.full(n, 1010.0), "m", 100001, -10000, 100000),
        Channel("ground_heading", f.heading.to_numpy(), "degrees", 361, 0, 360),
        Channel("ground_speed_mph", f.speed_kmh.to_numpy() / 1.609344, "mph", 1153, 0, 1152),
        Channel("number_of_satellites_in_use", np.full(n, 10.0), "satellites", 255, 0, 24),
        Channel("input_voltage", np.full(n, 13.5), "V", 10000, 0, 40),
        Channel("utc_day", gps.day.to_numpy(dtype=float), "day", 32, 0, 31),
        Channel("utc_hour", gps.hour.to_numpy(dtype=float), "hour", 25, 0, 24),
        Channel("utc_minute", gps.minute.to_numpy(dtype=float), "minute", 61, 0, 60),
        Channel("utc_seconds", (gps.second + gps.microsecond / 1e6).to_numpy(dtype=float), "seconds", 61, 0, 60),
        Channel("RunTrigger", np.ones(n), "", None, 0, 1),
    ]
    if second_datamode:
        channels += [Channel(c.name, c.values, c.units, c.invalid, c.range_min, c.range_max,
                             datamode="Omni_2022") for c in channels[:3]]
    return channels


DEMO_START = pd.Timestamp("2026-09-02 13:40:04")


def demo_log(path: str | Path, seed: int = 7, **kwargs) -> tuple[Path, Day]:
    """Write the made-up day to ``path`` and return it with its truth."""
    day = synthetic_day(seed)
    channels = day_channels(day, start=DEMO_START, **kwargs)
    write_sie(path, channels, DEMO_START.strftime("%Y-%m-%dT%H:%M:%S.000000000"))
    return Path(path), day
