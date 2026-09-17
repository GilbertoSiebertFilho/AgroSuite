"""Synthetic monitor data generator.

It serves two purposes: letting you try the app without a file in hand and,
above all, providing a dataset with **known defects** to check the cleaning
filters against — if the overlap filter does not find the overlap that was
planted on purpose, the filter is wrong.

The simulated field has:

* a back-and-forth track with real headland turns;
* a smooth spatial yield pattern (the legitimate variability);
* sensor flow delay;
* one pass driven back over already-harvested ground;
* scattered sensor spikes and zeros;
* stretches of speed outside the operating range.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .core.dataset import Dataset, DatasetMeta


def synthetic_harvest(
    seed: int = 7,
    n_passes: int = 18,
    pass_length_m: float = 420.0,
    swath_m: float = 9.0,
    origin_lon: float = -51.2000,
    origin_lat: float = -23.5000,
    with_defects: bool = True,
) -> Dataset:
    """Generate a synthetic harvest map with planted defects.

    Returns
    -------
    Dataset
        Carrying an extra ``truth_kg_ha`` column — the clean value before the
        defects — to measure how much of the original signal cleaning recovers.
    """
    rng = np.random.default_rng(seed)

    # One degree of latitude is about 111.32 km; longitude shrinks with the cosine.
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = m_per_deg_lat * np.cos(np.radians(origin_lat))

    step_m = 1.4  # typical advance between records at ~5 km/h with 1 s logging
    rows: list[dict] = []
    time_s = 0.0

    for p in range(n_passes):
        n_points = int(pass_length_m / step_m)
        offset_x = p * swath_m
        going_north = p % 2 == 0
        along = np.arange(n_points) * step_m
        if not going_north:
            along = along[::-1]

        speed = 5.6 + rng.normal(0, 0.12, n_points)
        # Entering and leaving a pass: the machine speeds up and slows down.
        ramp = min(18, n_points // 4)
        speed[:ramp] *= np.linspace(0.35, 1.0, ramp)
        speed[-ramp:] *= np.linspace(1.0, 0.35, ramp)

        for i in range(n_points):
            x = offset_x + rng.normal(0, 0.05)
            y = along[i]
            rows.append({
                "x_local": x,
                "y_local": y,
                "speed": speed[i],
                "pass": p,
                "t": time_s,
                "swath": swath_m,
                "heading": 0.0 if going_north else 180.0,
            })
            time_s += step_m / max(speed[i] / 3.6, 0.5)

        time_s += 14.0  # headland turn

    df = pd.DataFrame(rows)

    # --- true yield: two harmonics plus a gradient plus fine noise
    xs = df["x_local"].to_numpy()
    ys = df["y_local"].to_numpy()
    field_width = n_passes * swath_m
    truth = (
        10_500.0
        + 1_800.0 * np.sin(2 * np.pi * xs / (field_width / 1.5))
        + 1_100.0 * np.cos(2 * np.pi * ys / (pass_length_m / 2.2))
        - 900.0 * (ys / pass_length_m)
        + rng.normal(0, 260.0, len(df))
    )
    truth = np.clip(truth, 3_000.0, 18_000.0)
    df["truth_kg_ha"] = truth

    observed = truth.copy()

    if with_defects:
        # Flow delay: the sensor reports the reading about 12 s after the cut.
        median_dt = float(np.median(np.diff(df["t"].to_numpy())))
        lag = max(1, int(round(12.0 / max(median_dt, 0.1))))
        observed = np.roll(observed, lag)
        observed[:lag] = observed[lag]

        # The mass in transit makes the value track speed with a lag.
        speed = df["speed"].to_numpy()
        speed_effect = np.clip(speed / np.median(speed), 0.25, 1.8)
        observed *= 0.35 + 0.65 * np.roll(speed_effect, lag)

        # Sensor zeros and isolated spikes.
        n = len(df)
        observed[rng.choice(n, size=int(n * 0.012), replace=False)] = 0.0
        spikes = rng.choice(n, size=int(n * 0.008), replace=False)
        observed[spikes] *= rng.uniform(2.6, 4.5, size=spikes.size)

        # Re-run: an extra pass over strip 5, already harvested, with a low value.
        overlap_mask = df["pass"] == 5
        extra = df.loc[overlap_mask].copy()
        extra["x_local"] += swath_m * 0.45
        extra["t"] += float(df["t"].max()) + 60.0
        extra["pass"] = n_passes + 1
        extra_observed = truth[overlap_mask.to_numpy()] * rng.uniform(0.12, 0.35, int(overlap_mask.sum()))
        df = pd.concat([df, extra], ignore_index=True)
        observed = np.concatenate([observed, extra_observed])

    df["value"] = observed
    df["moisture_pct"] = np.clip(rng.normal(14.5, 1.1, len(df)), 6.0, 32.0)

    out = pd.DataFrame({
        "lon": origin_lon + df["x_local"].to_numpy() / m_per_deg_lon,
        "lat": origin_lat + df["y_local"].to_numpy() / m_per_deg_lat,
        "timestamp": pd.Timestamp("2025-03-12 08:00:00") + pd.to_timedelta(df["t"], unit="s"),
        "value": df["value"].to_numpy(),
        "speed_kmh": df["speed"].to_numpy(),
        "swath_m": df["swath"].to_numpy(),
        "moisture_pct": df["moisture_pct"].to_numpy(),
        "pass_id": df["pass"].to_numpy(),
        "truth_kg_ha": df["truth_kg_ha"].to_numpy(),
    }).sort_values("timestamp").reset_index(drop=True)

    # The magnitudes are corn's: ten tonnes a hectare. Declared as canola, the
    # first look rightly called the demo's units wrong, and the app's own check
    # contradicting its own sample data was the first thing a new user saw.
    meta = DatasetMeta(
        name="Demo field",
        source_path="<synthetic>",
        source_format="demo",
        brand="generic",
        brand_label="Synthetic data",
        operation="harvest",
        crop="corn",
        field_name="Demo field",
        value_label="Yield",
        value_unit="kg/ha",
        notes=["Synthetic dataset with planted defects, for checking the cleaning."],
    )
    ds = Dataset(out, meta)
    ds.ensure_derived(default_swath_m=swath_m)
    return ds


def synthetic_trial(seed: int = 3, rates: tuple[float, ...] = (0, 60, 120, 180, 240)) -> Dataset:
    """Generate a strip trial with a known rate response.

    The response is quadratic with a plateau and differs between two fertility
    zones, which is exactly what the economic analysis should be able to
    separate.

    The crop is canola, the one the app opens on: this demo is the first thing
    a new user loads, with the Canadian defaults in place, and a trial drawn at
    corn's ten tonnes a hectare would read as 200 bu/ac of canola — which the
    first look rightly refuses to believe.
    """
    rng = np.random.default_rng(seed)
    # Each trial strip is 3 passes wide — any narrower and there is no usable
    # area left once the edges are discarded.
    passes_per_strip = 3
    n_blocks = 4
    n_strips = len(rates) * n_blocks
    ds = synthetic_harvest(
        seed=seed, n_passes=n_strips * passes_per_strip, with_defects=False
    )
    df = ds.df

    x = df["x"].to_numpy()
    x_rel = (x - x.min()) / max(float(np.ptp(x)), 1.0)
    strip = np.clip(np.floor(x_rel * n_strips).astype(int), 0, n_strips - 1)

    # The order of the rates is drawn within each block, as in a randomized
    # block experiment.
    assignment = np.zeros(n_strips, dtype=int)
    for block in range(n_blocks):
        assignment[block * len(rates):(block + 1) * len(rates)] = rng.permutation(len(rates))
    rate = np.array(rates)[assignment[strip]]

    y = df["y"].to_numpy()
    y_rel = (y - y.min()) / max(float(np.ptp(y)), 1.0)
    zone = (y_rel > 0.5).astype(int)  # zone 1 = the more fertile one

    # Quadratic response at canola's scale — a bare check strip near 1.5 t/ha,
    # a well-fed one near 3.5 — with a higher plateau and a lower optimum in
    # the fertile zone.
    base = np.where(zone == 1, 2_000.0, 1_400.0)
    gain = np.where(zone == 1, 12.0, 14.0)
    curve = np.where(zone == 1, -0.0207, -0.0217)
    yield_kg = base + gain * rate + curve * rate**2 + rng.normal(0, 110.0, len(df))

    df["value"] = np.clip(yield_kg, 300.0, None)
    df["applied_rate"] = rate.astype(float)
    df["trial_id"] = strip
    df["zone"] = zone
    df = df.drop(columns=["truth_kg_ha"], errors="ignore")
    ds.df = df

    ds.meta.name = "Strip trial (demo)"
    ds.meta.crop = "canola"
    ds.meta.notes = ["Synthetic strip trial with a known quadratic response."]
    ds.meta.extra["rates"] = list(rates)
    return ds
