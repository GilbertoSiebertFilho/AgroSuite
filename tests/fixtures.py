"""Generator of sample files in each monitor's format.

Every function writes a file with the **same column names, units and folder
structure** the corresponding platform really produces. AgroSuite's readers
are tested against these files: if a manufacturer changes a column name, the
matching test breaks and the alias is updated in one place.

The tracks are identical across all the files, so differences in reading come
from the format and not from the data.
"""

from __future__ import annotations

import json
import struct
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ORIGIN_LON, ORIGIN_LAT = -105.8340, 50.4520  # southern Saskatchewan
M_PER_DEG_LAT = 111_320.0
M_PER_DEG_LON = M_PER_DEG_LAT * np.cos(np.radians(ORIGIN_LAT))


def _track(
    n_passes: int = 44,
    points_per_pass: int = 200,
    swath_m: float = 18.29,
    step_m: float = 4.0,
):
    """Back-and-forth track over a quarter section.

    The dimensions are not arbitrary: 44 passes at 60 feet cover the roughly
    805 m of a Prairie quarter section, which is the field size strip trials
    and prescriptions are actually dimensioned for. A sample field that was too
    small would make the trial layout fail for lack of room — and the test
    would have shown nothing about the app.
    """
    rng = np.random.default_rng(11)
    rows = []
    stamp = datetime(2025, 9, 8, 13, 30, 0)
    for p in range(n_passes):
        going = p % 2 == 0
        along = np.arange(points_per_pass) * step_m
        if not going:
            along = along[::-1]
        for i in range(points_per_pass):
            rows.append({
                "x": p * swath_m + rng.normal(0, 0.05),
                "y": float(along[i]),
                "t": stamp,
                "speed_kmh": 8.0 + rng.normal(0, 0.3),
                "swath_m": swath_m,
                "heading": 0.0 if going else 180.0,
                "elev_m": 585.0 + rng.normal(0, 0.4),
                "pass": p,
            })
            stamp += timedelta(seconds=2)
        stamp += timedelta(seconds=25)

    df = pd.DataFrame(rows)
    df["lon"] = ORIGIN_LON + df["x"] / M_PER_DEG_LON
    df["lat"] = ORIGIN_LAT + df["y"] / M_PER_DEG_LAT
    # Yield in kg/ha with a smooth spatial pattern, converted per file.
    df["yield_kg_ha"] = (
        3200
        + 600 * np.sin(2 * np.pi * df["x"] / 280)
        + 350 * np.cos(2 * np.pi * df["y"] / 420)
        + rng.normal(0, 140, len(df))
    ).clip(800, 6000)
    df["moisture"] = np.clip(rng.normal(11.5, 0.8, len(df)), 6, 22)
    # Two management zones, as in a field that has already been mapped.
    df["rate_kg_ha"] = np.where(df["x"] < df["x"].median(), 90.0, 130.0)
    return df


BU_KG = {"canola": 22.6796, "wheat": 27.2155, "corn": 25.4012}
KG_HA_TO_LB_AC = 0.40468564224 / 0.45359237
KMH_TO_MPH = 1 / 1.609344
M_TO_FT = 1 / 0.3048


def _to_shapefile(frame: pd.DataFrame, path: Path, columns: dict[str, str]) -> Path:
    """Write a point shapefile with the given field names."""
    import geopandas as gpd

    data = {alias: frame[source] for alias, source in columns.items()}
    gdf = gpd.GeoDataFrame(
        pd.DataFrame(data),
        geometry=gpd.points_from_xy(frame["lon"], frame["lat"]),
        crs="EPSG:4326",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(path, driver="ESRI Shapefile")
    return path


# ==========================================================================
# John Deere
# ==========================================================================

def john_deere_shapefile(out: Path) -> Path:
    """A harvest export from Operations Center, as a shapefile.

    Operations Center delivers bu/ac, mph and feet, and DBF truncates names at
    ten characters — hence names like ``Yld_Vol_Dr`` and ``Swth_Wdth_``.
    """
    df = _track()
    df = df.assign(
        yld=df["yield_kg_ha"] / (BU_KG["canola"] / 0.40468564224),
        spd=df["speed_kmh"] * KMH_TO_MPH,
        swth=df["swath_m"] * M_TO_FT,
        elev=df["elev_m"] * M_TO_FT,
        dist=2.0 * M_TO_FT,
        track=df["heading"],
        time=df["t"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        moist=df["moisture"],
        crop="Canola",
        machine="S780",
        variety="L233P",
        op="Harvest",
        field="NW-14-32-W2",
    )
    # Operations Center names the yield VRYIELDVOL/VRYIELDMASS — that is the
    # signature distinguishing a John Deere export from an SMS one, which uses
    # Yld_Vol_Dr and Swth_Wdth_.
    return _to_shapefile(df, out / "JD_Colheita_Canola.shp", {
        "VRYIELDVOL": "yld", "Moisture": "moist", "Speed": "spd",
        "SwathWidth": "swth", "Elevation": "elev", "Distance": "dist",
        "Heading": "track", "Time": "time", "Crop": "crop",
        "Machine": "machine", "Variety": "variety", "Product": "crop",
        "Operation": "op", "Field": "field",
    })


def john_deere_zip(out: Path) -> Path:
    """The same shapefile zipped, as it comes off the monitor's USB stick."""
    folder = out / "jd_raw"
    shp = john_deere_shapefile(folder)
    archive = out / "JD_OperationsCenter_Export.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
            sidecar = shp.with_suffix(ext)
            if sidecar.exists():
                zf.write(sidecar, f"Documentation/{sidecar.name}")
    return archive


# ==========================================================================
# Ag Leader / SMS
# ==========================================================================

def ag_leader_shapefile(out: Path) -> Path:
    """An SMS Advanced export — the bridge format between platforms."""
    df = _track()
    df = df.assign(
        obj=np.arange(len(df)),
        yld=df["yield_kg_ha"] / (BU_KG["wheat"] / 0.40468564224),
        spd=df["speed_kmh"] * KMH_TO_MPH,
        swth=df["swath_m"] * M_TO_FT,
        dur=1.0,
        dist=2.0 * M_TO_FT,
        elev=df["elev_m"] * M_TO_FT,
        track=df["heading"],
        moist=df["moisture"],
    )
    return _to_shapefile(df, out / "SMS_Trigo_2025.shp", {
        "Obj__Id": "obj", "Yld_Vol_Dr": "yld", "Swth_Wdth_": "swth",
        "Track_deg_": "track", "Distance_f": "dist", "Duration_s": "dur",
        "Elevation_": "elev", "Moisture": "moist", "Speed_mph_": "spd",
    })


# ==========================================================================
# Raven Viper 4
# ==========================================================================

def raven_viper_csv(out: Path) -> Path:
    """A Viper 4 job log, with boom sections and product."""
    df = _track()
    out.mkdir(parents=True, exist_ok=True)
    path = out / "Viper4_JobLog.csv"
    frame = pd.DataFrame({
        "Longitude": df["lon"].round(8),
        "Latitude": df["lat"].round(8),
        "Date/Time": df["t"].dt.strftime("%m/%d/%Y %H:%M:%S"),
        "Speed(mph)": (df["speed_kmh"] * KMH_TO_MPH).round(2),
        "Swath(ft)": (df["swath_m"] * M_TO_FT).round(1),
        "Target Rate": (df["rate_kg_ha"] * KG_HA_TO_LB_AC).round(1),
        "Applied Rate": (df["rate_kg_ha"] * KG_HA_TO_LB_AC * 0.98).round(1),
        "Product": "Urea 46-0-0",
        "Boom Section": 1,
        "Prod Rate": (df["rate_kg_ha"] * KG_HA_TO_LB_AC).round(1),
    })
    frame.to_csv(path, index=False)
    return path


# ==========================================================================
# Trimble
# ==========================================================================

def trimble_csv(out: Path) -> Path:
    """A coverage export from FmX/GFX."""
    df = _track()
    out.mkdir(parents=True, exist_ok=True)
    path = out / "Trimble_GFX_Coverage.csv"
    pd.DataFrame({
        "Longitude": df["lon"].round(8),
        "Latitude": df["lat"].round(8),
        "Time": df["t"].dt.strftime("%Y-%m-%dT%H:%M:%S"),
        "Elevation(ft)": (df["elev_m"] * M_TO_FT).round(1),
        "Speed(mph)": (df["speed_kmh"] * KMH_TO_MPH).round(2),
        "Heading": df["heading"].round(1),
        "Tgt_Rate_O": (df["rate_kg_ha"] * KG_HA_TO_LB_AC).round(1),
        "Applied Rate": (df["rate_kg_ha"] * KG_HA_TO_LB_AC * 1.01).round(1),
        "Material": "MAP 11-52-0",
        "Swath Width": (df["swath_m"] * M_TO_FT).round(1),
        "Vehicle": "Case 4430",
    }).to_csv(path, index=False)
    return path


# ==========================================================================
# Bourgault
# ==========================================================================

def bourgault_csv(out: Path) -> Path:
    """An X35 log, with one tank per product."""
    df = _track()
    out.mkdir(parents=True, exist_ok=True)
    path = out / "Bourgault_X35_Seeding.csv"
    pd.DataFrame({
        "Longitude": df["lon"].round(8),
        "Latitude": df["lat"].round(8),
        "DateTime": df["t"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        "Ground Speed (mph)": (df["speed_kmh"] * KMH_TO_MPH).round(2),
        "Working Width (ft)": (df["swath_m"] * M_TO_FT).round(1),
        "Tank 1 Rate": (df["rate_kg_ha"] * KG_HA_TO_LB_AC).round(1),
        "Tank 2 Rate": (df["rate_kg_ha"] * KG_HA_TO_LB_AC * 0.4).round(1),
        "Shank Count": 76,
        "Blockage": 0,
    }).to_csv(path, index=False)
    return path


# ==========================================================================
# CSV europeu / brasileiro — ponto e vírgula e vírgula decimal
# ==========================================================================

def semicolon_csv(out: Path) -> Path:
    """An export with ';' separator and decimal comma, common outside North America."""
    df = _track()
    out.mkdir(parents=True, exist_ok=True)
    path = out / "Vaderstad_EControl_Semeadura.csv"
    frame = pd.DataFrame({
        "Longitude": df["lon"].round(8),
        "Latitude": df["lat"].round(8),
        "Zeitstempel": df["t"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        "Geschwindigkeit (km/h)": df["speed_kmh"].round(2),
        "Arbeitsbreite (m)": df["swath_m"].round(2),
        "Saatmenge (kg/ha)": df["rate_kg_ha"].round(1),
        "Duengermenge (kg/ha)": (df["rate_kg_ha"] * 0.6).round(1),
    })
    frame.to_csv(path, index=False, sep=";", decimal=",")
    return path


# ==========================================================================
# ISOXML com log binário — Case IH / New Holland / Väderstad / Topcon
# ==========================================================================

ISO_EPOCH = date(1980, 1, 1)


def isoxml_with_log(out: Path, field_name: str = "NW-14-32-W2") -> Path:
    """A complete TASKDATA folder, with registry, boundary and a binary TLG log.

    The binary follows the ISO 11783-10 layout: a header declaring which fields
    are written, and one record per reading carrying time, position and the
    values of the declared DLVs.
    """
    df = _track()
    taskdata = out / "TASKDATA"
    taskdata.mkdir(parents=True, exist_ok=True)

    # --- field boundary (bounding rectangle with some slack)
    pad = 0.0003
    ring = [
        (df["lon"].min() - pad, df["lat"].min() - pad),
        (df["lon"].max() + pad, df["lat"].min() - pad),
        (df["lon"].max() + pad, df["lat"].max() + pad),
        (df["lon"].min() - pad, df["lat"].max() + pad),
    ]

    points = "".join(
        f'<PNT A="2" C="{lat:.9f}" D="{lon:.9f}" I="{i}"/>'
        for i, (lon, lat) in enumerate(ring, start=1)
    )
    taskdata_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<ISO11783_TaskData VersionMajor="4" VersionMinor="0"
    ManagementSoftwareManufacturer="CNH Industrial"
    ManagementSoftwareVersion="9.2" DataTransferOrigin="2">
  <CTR A="CTR1" B="Siebert Farms"/>
  <FRM A="FRM1" B="Home Quarter" I="CTR1"/>
  <PFD A="PFD1" C="{field_name}" D="647500" E="CTR1" F="FRM1">
    <PLN A="1" C="Field boundary"><LSG A="1">{points}</LSG></PLN>
  </PFD>
  <PDT A="PDT1" B="Urea 46-0-0"/>
  <TSK A="TSK1" B="Seeding {field_name}" C="CTR1" D="FRM1" E="PFD1" G="4">
    <TLG A="TLG00001" B="1"/>
  </TSK>
</ISO11783_TaskData>
"""
    (taskdata / "TASKDATA.XML").write_text(taskdata_xml, encoding="utf-8")

    # --- log header: an empty attribute means the field is in the binary
    (taskdata / "TLG00001.XML").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<TIM A="" D="4">'
        '<PTN A="" B="" C="" D="" G=""/>'
        '<DLV A="0007" B="" C="DET-1"/>'   # actual rate, mg/m2
        '<DLV A="0043" B="" C="DET-1"/>'   # actual ground speed, mm/s
        '<DLV A="0049" B="" C="DET-1"/>'   # working width, mm
        "</TIM>\n",
        encoding="utf-8",
    )

    # --- binary
    payload = bytearray()
    for _, row in df.iterrows():
        stamp = row["t"]
        midnight = datetime(stamp.year, stamp.month, stamp.day)
        time_ms = int((stamp - midnight).total_seconds() * 1000)
        days = (stamp.date() - ISO_EPOCH).days

        payload += struct.pack("<I", time_ms)
        payload += struct.pack("<H", days)
        payload += struct.pack("<i", int(round(row["lat"] * 1e7)))
        payload += struct.pack("<i", int(round(row["lon"] * 1e7)))
        payload += struct.pack("<i", int(round(row["elev_m"] * 1000)))
        payload += struct.pack("<B", 4)   # RTK fixed position
        payload += struct.pack("<B", 18)  # satellites

        values = [
            (0, int(round(row["rate_kg_ha"] * 100))),      # kg/ha -> mg/m²
            (1, int(round(row["speed_kmh"] / 0.0036))),    # km/h  -> mm/s
            (2, int(round(row["swath_m"] * 1000))),        # m     -> mm
        ]
        payload += struct.pack("<B", len(values))
        for index, value in values:
            payload += struct.pack("<B", index) + struct.pack("<i", value)

    (taskdata / "TLG00001.BIN").write_bytes(bytes(payload))
    return taskdata


# ==========================================================================
# Augmenta
# ==========================================================================

def augmenta_geojson(out: Path) -> Path:
    """An Augmenta session: vigour per point and the rate the system applied."""
    df = _track()
    out.mkdir(parents=True, exist_ok=True)
    path = out / "augmenta_session_4471.geojson"

    rng = np.random.default_rng(5)
    vigor = np.clip(0.30 + 0.45 * np.sin(2 * np.pi * df["x"] / 60) / 2
                    + rng.normal(0, 0.04, len(df)) + 0.2, 0.05, 0.95)
    # Augmenta applies more where there is more biomass (fungicide, desiccation).
    applied = (60 + 90 * vigor).round(1)

    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [round(lon, 8), round(lat, 8)]},
            "properties": {
                "vigor": round(float(v), 3),
                "ndvi": round(float(v) * 0.9 + 0.05, 3),
                "applied_rate": float(a),
                "crop_coverage": round(float(v) * 100, 1),
                "speed": round(float(s), 2),
                "timestamp": t.strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
        }
        for lon, lat, v, a, s, t in zip(
            df["lon"], df["lat"], vigor, applied, df["speed_kmh"], df["t"]
        )
    ]
    payload = {
        "augmenta_version": "3.4.1",
        "session": {
            "session_id": "4471",
            "field_name": "NW-14-32-W2",
            "crop": "wheat",
            "machine_id": "AUG-0912",
            "start_time": "2025-09-08T13:30:00Z",
        },
        "application": {"product": "Fungicide", "unit": "L/ha"},
        "data": {"type": "FeatureCollection", "features": features},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ==========================================================================
# Binário proprietário — caso de erro informativo
# ==========================================================================

def proprietary_binaries(out: Path) -> list[Path]:
    """Proprietary files the app should refuse with clear guidance."""
    out.mkdir(parents=True, exist_ok=True)
    created = []
    for name, header in [
        ("GS3_Documentation.gsd", b"\x00GSD\x02"),
        ("FieldDoc.fdd", b"\xffFDD\x01"),
        ("Setup.jdp", b"PK\x03\x04JDP"),
        ("Voyager.vy1", b"\x10VY1\x00"),
        ("Coverage.cn1", b"CN1\x00\x00"),
    ]:
        path = out / name
        path.write_bytes(header + bytes(512))
        created.append(path)
    return created


def build_all(out: Path) -> dict[str, Path]:
    """Generate every sample and return the name -> path map."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    return {
        "john_deere_shp": john_deere_shapefile(out / "john_deere"),
        "john_deere_zip": john_deere_zip(out),
        "ag_leader_shp": ag_leader_shapefile(out / "ag_leader"),
        "raven_csv": raven_viper_csv(out / "raven"),
        "trimble_csv": trimble_csv(out / "trimble"),
        "bourgault_csv": bourgault_csv(out / "bourgault"),
        "semicolon_csv": semicolon_csv(out / "vaderstad"),
        "isoxml": isoxml_with_log(out / "cnh"),
        "augmenta": augmenta_geojson(out / "augmenta"),
    }


if __name__ == "__main__":
    import sys

    target = Path(sys.argv[1] if len(sys.argv) > 1 else "sample_data")
    paths = build_all(target)
    proprietary_binaries(target / "proprietarios")
    for key, path in paths.items():
        print(f"{key:20s} {path}")


# ==========================================================================
# A three-file DIFM project
# ==========================================================================

def difm_project(out: Path) -> dict[str, Path]:
    """Write a trial as the three separate files it really arrives in.

    A DIFM trial never comes as one file. The plan leaves the office software,
    the as-applied log comes off the air drill, and the yield map comes off the
    combine months later — three passes over the same ground, logged by three
    machines at three densities. The samples here reproduce that: different
    point counts, positions that do not coincide, and a few percent of
    execution error between the plan and what actually went out.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(23)

    df = _track(n_passes=45, points_per_pass=200, swath_m=18.29, step_m=4.0)

    # --- the trial design: 5 rates, 3 passes per strip, randomized in blocks
    rates = np.array([0.0, 45.0, 90.0, 135.0, 180.0])
    passes_per_strip, n_blocks = 3, 3
    n_strips = len(rates) * n_blocks
    strip = np.clip(
        (df["x"] / (passes_per_strip * 18.29)).astype(int), 0, n_strips - 1
    )
    assignment = np.concatenate([rng.permutation(len(rates)) for _ in range(n_blocks)])
    planned = rates[assignment[np.clip(strip, 0, len(assignment) - 1)]]

    # --- two fertility zones, with different responses to nitrogen
    zone = (df["y"] > df["y"].median()).astype(int)
    base = np.where(zone == 1, 2_900.0, 2_200.0)
    gain = np.where(zone == 1, 9.0, 12.0)
    curve = np.where(zone == 1, -0.021, -0.028)
    yield_kg = base + gain * planned + curve * planned**2 + rng.normal(0, 130, len(df))

    # --- 1. the plan, as a coarse prescription grid
    plan_idx = np.arange(0, len(df), 8)
    pd.DataFrame({
        "Longitude": df["lon"].iloc[plan_idx].round(8),
        "Latitude": df["lat"].iloc[plan_idx].round(8),
        "Target Rate": (planned[plan_idx] * KG_HA_TO_LB_AC).round(1),
        "Product": "Urea 46-0-0",
    }).to_csv(out / "1_plan_N_rates.csv", index=False)

    # --- 2. the as-applied log, denser, with execution error
    applied_idx = np.arange(0, len(df), 2)
    applied = planned[applied_idx] * rng.normal(1.0, 0.025, len(applied_idx))
    pd.DataFrame({
        "Longitude": (df["lon"].iloc[applied_idx] + rng.normal(0, 2e-6, len(applied_idx))).round(8),
        "Latitude": (df["lat"].iloc[applied_idx] + rng.normal(0, 2e-6, len(applied_idx))).round(8),
        "DateTime": df["t"].iloc[applied_idx].dt.strftime("%Y-%m-%d %H:%M:%S"),
        "Ground Speed (mph)": (df["speed_kmh"].iloc[applied_idx] * KMH_TO_MPH).round(2),
        "Working Width (ft)": (df["swath_m"].iloc[applied_idx] * M_TO_FT).round(1),
        "Tank 1 Rate": (applied * KG_HA_TO_LB_AC).round(1),
    }).to_csv(out / "2_as_applied_N.csv", index=False)

    # --- 3. the yield map, densest of the three
    _to_shapefile(
        df.assign(
            yld=yield_kg / (BU_KG["canola"] / 0.40468564224),
            spd=df["speed_kmh"] * KMH_TO_MPH,
            swth=df["swath_m"] * M_TO_FT,
            moist=df["moisture"],
            zone=zone,
            crop="Canola",
            time=df["t"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        ),
        out / "3_yield_canola.shp",
        {
            "VRYIELDVOL": "yld", "Moisture": "moist", "Speed": "spd",
            "SwathWidth": "swth", "Time": "time", "Crop": "crop", "Zone": "zone",
        },
    )

    return {
        "plan": out / "1_plan_N_rates.csv",
        "as_applied": out / "2_as_applied_N.csv",
        "yield": out / "3_yield_canola.shp",
    }
