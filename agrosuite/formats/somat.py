"""HBM SoMat eDAQ logs (``.sie``): what the machine did, twice a second.

A SoMat eDAQ logger rides on the machine and records two things side by
side: the engine's CAN bus (J1939 — fuel rate, load, engine speed, DEF
level, the cumulative fuel and distance counters) and a GPS of its own. One
file usually covers the whole day — the yard, the road, every field — which
is what makes it worth reading: it says where the time and the fuel went.

The format
----------
SIE 1.0 is a stream of blocks, each ``size | group | 0x51EDA7A0 | payload |
checksum | size`` with big-endian integers. Group 0 carries XML metadata;
every other group carries the samples of one channel. The XML declares, per
channel, its group, sample rate, units, the decoder of its payload and —
what matters most here — the value the logger writes when it has none
(``somat:invalid_data_output_value``) and the channel's physical range. The
XML is written as the log grows, so its root element is never closed; it is
closed here before it is parsed.

Three things the file does not do for you
-----------------------------------------
* **Missing readings are numbers.** −1 L/h, 1000 °C, −500 V, 181° of
  longitude. Each channel declares its own, and they become NaN before
  anything is computed.
* **Cumulative counters wrap** at the top of their declared range: the
  distance counter of a New Holland 370F goes back to 0 after 100 000 ft,
  twice in an ordinary day. The per-sample increments (``DeltaD``,
  ``DeltaF``) never wrap, so the totals are rebuilt from them and checked
  against the counters and against the GPS.
* **The logger's clock is local and drifts** — 23 s behind the GPS on the
  first file this was written for. The GPS time the log carries (the
  ``utc_*`` channels) sets it right; the correction is measured on every
  file rather than typed in.
"""

from __future__ import annotations

import math
import re
import struct
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..core import schema as sch
from ..core.dataset import Dataset, DatasetMeta
from . import brands as brands_mod

#: The word every block carries after its size and group.
SYNC = 0x51EDA7A0

#: ``size | group | sync`` before the payload, ``checksum | size`` after it.
HEADER_BYTES = 12
FOOTER_BYTES = 8

_NS = "{http://www.somat.com/SIE}"

#: The data mode a SoMat test writes its time histories to. Others (a second
#: ``Omni_2022`` mode on the 370F) repeat the same channels.
MAIN_DATAMODE = "__dm"

#: Raw channel -> column in the dataset. Anything not listed is kept too,
#: under ``ch_<name>``, so a logger wired to other signals loses nothing.
CHANNEL_COLUMNS: dict[str, str] = {
    "latitude": sch.LAT,
    "longitude": sch.LON,
    "altitude": sch.ELEVATION,
    "ground_heading": sch.HEADING,
    "ground_speed_mph": sch.SPEED,
    "ground_speed": sch.SPEED,
    "DeltaD": sch.DISTANCE,
    "EngFuelRate": "fuel_rate_lh",
    "DeltaF": "fuel_l",
    "TotalFuelUsed": "fuel_counter_l",
    "DistTraveled": "distance_counter_m",
    "Def": "def_level_pct",
    "EngSpeed": "engine_rpm",
    "EngPercentLoadAtCurrentSpeed": "engine_load_pct",
    "EngCoolantTemp": "coolant_c",
    "ElectricalPotential": "battery_v",
    "input_voltage": "logger_v",
    "EngTotalHoursOfOperation": "engine_hours",
    "number_of_satellites_in_use": "satellites",
}

#: The GPS clock, read to set the logger's clock right; not kept as columns.
UTC_CHANNELS = ("utc_day", "utc_hour", "utc_minute", "utc_seconds")

#: Channels that carry nothing: ``RunTrigger`` is 1 for as long as it logs.
DROPPED_CHANNELS = {"RunTrigger", *UTC_CHANNELS}

#: Counters whose wraps are undone, with the increment that rebuilds them.
COUNTERS = {
    "distance_counter_m": sch.DISTANCE,
    "fuel_counter_l": "fuel_l",
}

#: Units as the logger writes them -> (unit family, factor to the column's
#: metric unit). A channel in a unit not listed keeps its raw values and says
#: so in the notes rather than being converted by a guess.
UNIT_FACTORS: dict[str, tuple[str, float]] = {
    "ft": ("length", 0.3048),
    "m": ("length", 1.0),
    "km": ("length", 1000.0),
    "mi": ("length", 1609.344),
    "mph": ("speed", 1.609344),
    "km/h": ("speed", 1.0),
    "kph": ("speed", 1.0),
    "m/s": ("speed", 3.6),
    "l": ("volume", 1.0),
    "gal": ("volume", 3.785411784),
    "l/hr": ("volume_rate", 1.0),
    "l/h": ("volume_rate", 1.0),
    "gal/hr": ("volume_rate", 3.785411784),
    "gal/h": ("volume_rate", 3.785411784),
}

#: The metric unit each converted column ends up in.
COLUMN_FAMILY = {
    sch.SPEED: "speed",
    sch.DISTANCE: "length",
    "distance_counter_m": "length",
    sch.ELEVATION: "length",
    "fuel_l": "volume",
    "fuel_counter_l": "volume",
    "fuel_rate_lh": "volume_rate",
}

#: A clock more than this far from the GPS, once the time zone is taken out,
#: is not drift but a wrong setting, and is left alone with a note.
MAX_CLOCK_DRIFT_S = 300.0


# ==========================================================================
# Recognising the file
# ==========================================================================

def is_sie(path: str | Path) -> bool:
    """Whether ``path`` starts like a SoMat SIE stream."""
    try:
        with open(path, "rb") as handle:
            head = handle.read(HEADER_BYTES)
    except OSError:
        return False
    if len(head) < HEADER_BYTES:
        return False
    return struct.unpack(">I", head[8:12])[0] == SYNC


# ==========================================================================
# Blocks, metadata, channels
# ==========================================================================

@dataclass
class SieChannel:
    """One channel as the XML declares it."""

    name: str
    group: int
    datamode: str
    rate_hz: float
    units: str
    decoder: str
    time_scale: float
    time_offset: float
    invalid: float | None
    range_min: float | None
    range_max: float | None
    description: str = ""
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class SieLog:
    """A decoded file: its channels, their samples and what reading them found."""

    channels: list[SieChannel]
    samples: dict[tuple[str, str], np.ndarray]
    start_time: pd.Timestamp | None
    logger: dict[str, str]
    truncated_bytes: int = 0


def _read_blocks(data: bytes) -> tuple[dict[int, list[bytes]], int]:
    """Split the stream into payloads by group.

    A last block cut short — the logger lost power mid-write — ends the read
    and is reported as the number of bytes left over; a broken block anywhere
    else means the file is damaged, and that is refused.
    """
    groups: dict[int, list[bytes]] = defaultdict(list)
    pos, size_total = 0, len(data)
    while pos + HEADER_BYTES + FOOTER_BYTES <= size_total:
        size, group, sync = struct.unpack_from(">III", data, pos)
        if sync != SYNC:
            if pos == 0:
                raise ValueError("This is not a SoMat log: it does not start with an SIE block.")
            raise ValueError(
                f"The SoMat log is damaged at byte {pos:,}: the block there does not start "
                "the way every SIE block does. Copy it off the logger again."
            )
        if size < HEADER_BYTES + FOOTER_BYTES or pos + size > size_total:
            break
        (trailer,) = struct.unpack_from(">I", data, pos + size - 4)
        if trailer != size:
            raise ValueError(
                f"The SoMat log is damaged at byte {pos:,}: a block's two size fields "
                "disagree. Copy it off the logger again."
            )
        groups[group].append(data[pos + HEADER_BYTES: pos + size - FOOTER_BYTES])
        pos += size
    return groups, size_total - pos


def _number(text: str | None) -> float | None:
    try:
        value = float(str(text).strip())
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _metadata(groups: dict[int, list[bytes]]) -> ET.Element:
    if 0 not in groups:
        raise ValueError("The SoMat log carries no metadata block, so its channels cannot be named.")
    text = b"".join(groups[0]).decode("utf-8", "replace")
    # The root is closed only when the logger stops writing, which it never
    # gets to do for the XML: close it here.
    if not re.search(r"</sie>\s*$", text):
        text += "\n</sie>"
    try:
        return ET.fromstring(text)
    except ET.ParseError as exc:
        raise ValueError(f"The SoMat log's metadata could not be read: {exc}.") from exc


def _channels(root: ET.Element) -> list[SieChannel]:
    out: list[SieChannel] = []
    for ch in root.iter(f"{_NS}ch"):
        tags = {t.get("id", ""): (t.text or "").strip() for t in ch.findall(f"{_NS}tag")}
        dims = sorted(ch.findall(f"{_NS}dim"), key=lambda d: int(d.get("index", "0")))
        time_scale, time_offset, decoder = math.nan, 0.0, ""
        for dim in dims:
            xform = dim.find(f"{_NS}xform")
            data = dim.find(f"{_NS}data")
            if dim.get("index") == "0" and xform is not None:
                time_scale = _number(xform.get("scale")) or math.nan
                time_offset = _number(xform.get("offset")) or 0.0
            if data is not None and not decoder:
                decoder = data.get("decoder", "")
        rate = _number(tags.get("core:sample_rate")) or math.nan
        if not math.isfinite(time_scale) and rate and math.isfinite(rate):
            time_scale = 1.0 / rate
        try:
            group = int(ch.get("group", "-1"))
        except ValueError:
            continue
        out.append(SieChannel(
            name=ch.get("name", ""),
            group=group,
            datamode=tags.get("somat:datamode_name", ""),
            rate_hz=rate,
            units=tags.get("somat:output_units", ""),
            decoder=decoder,
            time_scale=time_scale,
            time_offset=time_offset,
            invalid=_number(tags.get("somat:invalid_data_output_value")),
            range_min=_number(tags.get("somat:physical_range_min")),
            range_max=_number(tags.get("somat:physical_range_max")),
            description=tags.get("core:description", ""),
            tags=tags,
        ))
    return out


def _decode(payloads: list[bytes]) -> np.ndarray:
    """Decoder 2: an int64 count, then that many float64, both little-endian."""
    parts = []
    for payload in payloads:
        if len(payload) < 8:
            continue
        (count,) = struct.unpack_from("<q", payload, 0)
        count = max(0, min(int(count), (len(payload) - 8) // 8))
        parts.append(np.frombuffer(payload, dtype="<f8", count=count, offset=8))
    return np.concatenate(parts).astype("float64") if parts else np.array([], dtype="float64")


def _start_time(root: ET.Element) -> pd.Timestamp | None:
    for tag in root.iter(f"{_NS}tag"):
        if tag.get("id") == "core:start_time" and tag.text:
            try:
                return pd.Timestamp(tag.text.strip())
            except (ValueError, TypeError):
                return None
    return None


def _logger_identity(channels: list[SieChannel]) -> dict[str, str]:
    """Module type and serial, read off a channel's connector string."""
    for ch in channels:
        module = ch.tags.get("somat:module_type", "")
        connector = ch.tags.get("somat:connector", "") or ch.tags.get("somat:connection", "")
        serial = re.search(r"-(\d{4,})\b", connector)
        if module or serial:
            return {"module": module, "serial": serial.group(1) if serial else ""}
    return {"module": "", "serial": ""}


def decode_sie(path: str | Path) -> SieLog:
    """Read a ``.sie`` file into its declared channels and raw samples."""
    path = Path(path)
    data = path.read_bytes()
    groups, leftover = _read_blocks(data)
    root = _metadata(groups)
    channels = _channels(root)
    if not channels:
        raise ValueError("The SoMat log declares no channels.")
    samples: dict[tuple[str, str], np.ndarray] = {}
    for ch in channels:
        if ch.decoder not in ("2", ""):
            continue
        samples[(ch.datamode, ch.name)] = _decode(groups.get(ch.group, []))
    return SieLog(
        channels=channels,
        samples=samples,
        start_time=_start_time(root),
        logger=_logger_identity(channels),
        truncated_bytes=leftover,
    )


# ==========================================================================
# From samples to a table
# ==========================================================================

def _column_for(name: str) -> str:
    if name in CHANNEL_COLUMNS:
        return CHANNEL_COLUMNS[name]
    return "ch_" + re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower()


def _unwrap(counter: np.ndarray, top: float | None) -> tuple[np.ndarray, int]:
    """Undo the counter going back to 0 at the top of its range.

    A drop of more than half the range between two valid samples is a wrap;
    anything smaller is noise and left alone. Returns the continuous series
    and how many wraps were undone.
    """
    values = counter.astype("float64").copy()
    if top is None or not math.isfinite(top) or top <= 0:
        return values, 0
    valid = np.flatnonzero(np.isfinite(values))
    if len(valid) < 2:
        return values, 0
    steps = np.diff(values[valid])
    wraps = np.flatnonzero(steps < -top / 2)
    if not len(wraps):
        return values, 0
    added = np.zeros(len(values))
    for w in wraps:
        added[valid[w + 1]:] += top
    return values + added, int(len(wraps))


def _gps_clock(frame: pd.DataFrame, start: pd.Timestamp, device: pd.Series) -> dict[str, Any]:
    """Compare the logger's clock with the GPS time the log carries.

    The GPS gives the day of the month and the time of day in UTC; the year
    and month come from the logger's start time, moved a month either way
    when the day says the log crossed a month boundary in UTC.
    """
    utc = frame[list(UTC_CHANNELS)]
    ok = utc.notna().all(axis=1).to_numpy()
    if ok.sum() < 10:
        return {}
    day = utc["utc_day"].to_numpy()[ok].astype(int)
    base = pd.Timestamp(year=start.year, month=start.month, day=1)
    month_shift = np.where(day < start.day - 15, 1, np.where(day > start.day + 15, -1, 0))
    months = np.array([base + pd.DateOffset(months=int(m)) for m in np.unique(month_shift)])
    first_of = dict(zip(np.unique(month_shift), months))
    stamps = np.array([first_of[m] for m in month_shift], dtype="datetime64[ns]")
    gps = (pd.to_datetime(stamps)
           + pd.to_timedelta(day - 1, unit="D")
           + pd.to_timedelta(utc["utc_hour"].to_numpy()[ok], unit="h")
           + pd.to_timedelta(utc["utc_minute"].to_numpy()[ok], unit="m")
           + pd.to_timedelta(utc["utc_seconds"].to_numpy()[ok], unit="s"))
    offset = (gps - pd.DatetimeIndex(device.to_numpy()[ok])).total_seconds().to_numpy()
    offset_s = float(np.median(offset))
    zone_s = round(offset_s / 900.0) * 900.0
    return {"offset_s": offset_s, "zone_s": zone_s, "drift_s": offset_s - zone_s}


@dataclass
class _Reading:
    frame: pd.DataFrame
    facts: dict[str, Any]
    notes: list[str]


def _table(log: SieLog, path: Path) -> _Reading:
    notes: list[str] = []
    datamodes = sorted({c.datamode for c in log.channels if (c.datamode, c.name) in log.samples})
    datamode = MAIN_DATAMODE if MAIN_DATAMODE in datamodes else max(
        datamodes, key=lambda m: sum(1 for c in log.channels if c.datamode == m), default=None)
    if datamode is None:
        raise ValueError("The SoMat log holds no channel this reader can decode.")
    others = [m for m in datamodes if m != datamode]
    if others:
        plural = len(others) > 1
        notes.append(
            f"The log also carries the data mode{'s' if plural else ''} "
            f"{', '.join(repr(m) for m in others)}, which repeat{'' if plural else 's'} "
            f"the same signals; only '{datamode}' was read."
        )

    chans = [c for c in log.channels if c.datamode == datamode and (c.datamode, c.name) in log.samples]
    rates = [c.rate_hz for c in chans if math.isfinite(c.rate_hz) and c.rate_hz > 0]
    if not rates:
        raise ValueError("The SoMat log's channels declare no sample rate.")
    rate = float(pd.Series(rates).mode().iloc[0])
    lengths = [len(log.samples[(c.datamode, c.name)]) for c in chans if c.rate_hz == rate]
    n = min(lengths) if lengths else 0
    if n == 0:
        raise ValueError("The SoMat log holds no samples.")

    t = np.arange(n) / rate
    columns: dict[str, np.ndarray] = {}
    raw_units: dict[str, str] = {}
    invalid_counts: dict[str, int] = {}
    ranges: dict[str, float | None] = {}
    unconverted: list[str] = []
    inventory: list[dict[str, Any]] = []

    for ch in chans:
        raw = log.samples[(ch.datamode, ch.name)]
        if ch.rate_hz == rate:
            values = raw[:n].copy()
        else:
            # Another rate: each sample of the table takes the channel's
            # latest reading at that moment — a CAN value holds until the
            # next one arrives. Counts and levels must not be averaged.
            ch_t = np.arange(len(raw)) * (ch.time_scale if math.isfinite(ch.time_scale) else 1.0 / ch.rate_hz)
            idx = np.clip(np.searchsorted(ch_t, t, side="right") - 1, 0, max(len(raw) - 1, 0))
            values = raw[idx].copy() if len(raw) else np.full(n, np.nan)
        bad = np.zeros(n, dtype=bool)
        if ch.invalid is not None:
            bad |= values == ch.invalid
        # Outside the channel's own physical range is not a reading either.
        if ch.range_min is not None:
            bad |= values < ch.range_min - 1e-9
        if ch.range_max is not None:
            bad |= values > ch.range_max + 1e-9
        values[bad] = np.nan
        column = _column_for(ch.name)
        inventory.append({
            "channel": ch.name, "column": None if ch.name in DROPPED_CHANNELS else column,
            "units": ch.units, "rate_hz": ch.rate_hz, "samples": int(len(raw)),
            "invalid": int(bad.sum()), "description": ch.description,
        })
        if bad.any():
            invalid_counts[ch.name] = int(bad.sum())
        if ch.name in UTC_CHANNELS:
            columns[ch.name] = values
            continue
        if ch.name in DROPPED_CHANNELS:
            continue
        family = COLUMN_FAMILY.get(column)
        unit_key = ch.units.strip().lower()
        if family is not None:
            found = UNIT_FACTORS.get(unit_key)
            if found and found[0] == family:
                values = values * found[1]
                if ch.range_max is not None:
                    ranges[column] = ch.range_max * found[1]
            elif unit_key:
                unconverted.append(f"{ch.name} ({ch.units})")
                ranges[column] = ch.range_max
        else:
            ranges[column] = ch.range_max
        columns[column] = values
        raw_units[column] = ch.units

    frame = pd.DataFrame(columns)
    facts: dict[str, Any] = {"rate_hz": rate, "samples": n, "datamode": datamode}

    if invalid_counts:
        listed = ", ".join(f"{name} {count:,}" for name, count in
                           sorted(invalid_counts.items(), key=lambda kv: -kv[1]))
        notes.append(
            f"{sum(invalid_counts.values()):,} readings the logger marked as missing, or that "
            f"fell outside their channel's range, were set aside: {listed}."
        )
    if unconverted:
        notes.append(
            "Kept in the units the logger wrote, which this reader does not convert: "
            + ", ".join(unconverted) + "."
        )

    # -- counters -------------------------------------------------------
    for counter, increment in COUNTERS.items():
        if counter not in frame:
            continue
        unwrapped, wraps = _unwrap(frame[counter].to_numpy(), ranges.get(counter))
        frame[counter] = unwrapped
        facts[f"{counter}_wraps"] = wraps
        name = "distance" if "distance" in counter else "fuel"
        if wraps:
            times = {1: "once", 2: "twice"}.get(wraps, f"{wraps} times")
            notes.append(
                f"The {name} counter went back to zero {times} at the top of its range; "
                "its total was stitched back together."
            )
        if increment in frame:
            inc = frame[increment].to_numpy()
            total_inc = float(np.nansum(inc))
            valid = unwrapped[np.isfinite(unwrapped)]
            total_counter = float(valid[-1] - valid[0]) if len(valid) > 1 else math.nan
            facts[f"{name}_from_increments"] = total_inc
            facts[f"{name}_from_counter"] = total_counter
            if math.isfinite(total_counter) and total_inc > 0:
                gap = abs(total_counter - total_inc) / total_inc
                facts[f"{name}_counter_gap"] = gap
                if gap > 0.02:
                    notes.append(
                        f"The {name} counter and the sum of its per-sample increments disagree "
                        f"by {gap:.0%}; the increments are what the analysis uses."
                    )

    # -- fuel rate against the increments -----------------------------------
    if "fuel_rate_lh" in frame and "fuel_l" in frame:
        by_rate = float(np.nansum(frame["fuel_rate_lh"].to_numpy()) / 3600.0 / rate)
        facts["fuel_from_rate"] = by_rate
        by_inc = facts.get("fuel_from_increments") or float(np.nansum(frame["fuel_l"]))
        if by_inc > 0 and abs(by_rate - by_inc) / by_inc > 0.05:
            notes.append(
                f"The fuel rate, summed over time, disagrees with the fuel increments by "
                f"{abs(by_rate - by_inc) / by_inc:.0%}. The increments are used; the rate is "
                "kept for its shape."
            )

    # -- distance against the GPS --------------------------------------------
    if {sch.LON, sch.LAT}.issubset(frame.columns):
        lat = np.radians(frame[sch.LAT].to_numpy())
        lon = np.radians(frame[sch.LON].to_numpy())
        a = (np.sin(np.diff(lat) / 2) ** 2
             + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(np.diff(lon) / 2) ** 2)
        gps_m = float(np.nansum(2 * 6_371_000 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))))
        facts["distance_from_gps"] = gps_m
        fixed = int(np.isfinite(frame[sch.LAT]).sum())
        facts["gps_fix_share"] = fixed / n
        if fixed < n:
            notes.append(f"{n - fixed:,} of {n:,} samples have no GPS position; they count for "
                         "time and fuel but are not on the map.")

    # -- clock --------------------------------------------------------------------
    start = log.start_time
    if start is None:
        notes.append("The log does not say when it started; times are counted from its first sample.")
        device = pd.Series(pd.Timestamp("1970-01-01") + pd.to_timedelta(t, unit="s"))
    else:
        device = pd.Series(start + pd.to_timedelta(t, unit="s"))
    clock = _gps_clock(frame, start, device) if start is not None and all(
        c in frame for c in UTC_CHANNELS) else {}
    if clock and abs(clock["drift_s"]) <= MAX_CLOCK_DRIFT_S:
        frame[sch.TIMESTAMP] = device + pd.to_timedelta(clock["drift_s"], unit="s")
        facts.update(clock_drift_s=clock["drift_s"], utc_offset_h=-clock["zone_s"] / 3600.0)
        zone = -clock["zone_s"] / 3600.0
        zone_text = f"UTC{zone:+g}".replace("+-", "-")
        if abs(clock["drift_s"]) >= 1.0:
            ahead = "behind" if clock["drift_s"] > 0 else "ahead of"
            notes.append(
                f"The logger's clock ran {abs(clock['drift_s']):.0f} s {ahead} the GPS; times "
                f"are set to GPS time, in the logger's zone ({zone_text})."
            )
    else:
        frame[sch.TIMESTAMP] = device
        if clock:
            notes.append(
                f"The logger's clock is {clock['offset_s'] / 3600:.2f} h from the GPS, which is "
                "not a time zone; its own times were kept. Check the logger's clock setting."
            )
        elif start is not None:
            notes.append("The log carries no GPS time, so the logger's clock could not be checked.")
    frame[sch.ELAPSED] = t
    frame = frame.drop(columns=[c for c in UTC_CHANNELS if c in frame])

    if log.truncated_bytes:
        notes.append(
            f"The log's last {log.truncated_bytes:,} bytes are an unfinished block — the logger "
            "stopped mid-write. Everything before them was read."
        )

    facts.update(
        channels=inventory,
        invalid=invalid_counts,
        logger=log.logger,
        start_device=str(start) if start is not None else None,
        start=str(frame[sch.TIMESTAMP].iloc[0]),
        end=str(frame[sch.TIMESTAMP].iloc[-1]),
        duration_s=float(t[-1] + 1.0 / rate) if n else 0.0,
        truncated_bytes=log.truncated_bytes,
        raw_units=raw_units,
    )
    return _Reading(frame=frame, facts=facts, notes=notes)


def read_somat(path: str | Path, brand_hint: str | None = None) -> Dataset:
    """The registry's reader: a ``.sie`` log as a telemetry :class:`Dataset`.

    One row per sample. The map colours it by fuel rate; every channel is a
    column. ``meta.extra["somat"]`` records what was checked and fixed on the
    way in — the clock, the counters, the readings set aside.
    """
    path = Path(path)
    log = decode_sie(path)
    reading = _table(log, path)
    frame = reading.frame
    if "fuel_rate_lh" in frame:
        frame[sch.VALUE] = frame["fuel_rate_lh"]
        value_label, value_unit = "Fuel rate", "L/h"
    elif sch.SPEED in frame:
        frame[sch.VALUE] = frame[sch.SPEED]
        value_label, value_unit = "Speed", "km/h"
    else:
        frame[sch.VALUE] = np.nan
        value_label, value_unit = "Value", ""

    brand = brands_mod.get_brand("somat")
    logger = reading.facts.get("logger") or {}
    who = " ".join(x for x in (logger.get("module"), logger.get("serial")) if x)
    meta = DatasetMeta(
        name=path.stem,
        source_path=str(path),
        source_format="somat_sie",
        brand=brand.key,
        brand_label=brand.label,
        operation="telemetry",
        value_label=value_label,
        value_unit=value_unit,
        source_value_unit=value_unit,
        geometry_type="point",
        notes=[f"SoMat log{(' from ' + who) if who else ''}: {reading.facts['samples']:,} samples at "
               f"{reading.facts['rate_hz']:g} Hz."] + reading.notes,
        extra={"somat": reading.facts},
    )
    return Dataset(frame, meta)
