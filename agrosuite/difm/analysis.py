"""The economic analysis of on-farm strip trials.

The method treats the commercial field as the experiment itself: strips of
different rates are applied with the grower's own machine, the response is
read off the yield monitor, and the price of the crop against the cost of the
input says where the next unit stops paying for itself. The analysis needs
three precautions that set it apart from a small-plot experiment:

1. **Aggregation.** Point by point, GPS error and sensor noise dominate. The
   data is aggregated into cells or strip segments before any curve is fitted.
2. **Strip edges.** Where two rates meet there is mixing and neighbour
   effects. An inward margin discards that zone.
3. **Heterogeneity.** The optimum rate changes within the field. Analysing by
   zone is what economically justifies variable rate in the first place.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..core import schema as sch
from ..core.dataset import Dataset
from ..core.units import Phrase
from . import response as response_mod


def rate_levels(
    rates: np.ndarray,
    max_levels: int = 12,
    valley_fraction: float = 0.02,
) -> tuple[np.ndarray, list[float]]:
    """Reduce the observed rates to the trial's treatment levels.

    A strip trial has a handful of planned rates, but the machine log records
    small wobbles around each one, and an as-applied layer aggregated per cell
    wobbles further. Grouping by the raw value would create dozens of one-point
    "rates"; grouping by level gives back the experiment's real design.

    The levels are found from the **density** of values along the rate axis,
    not from the gaps between neighbouring ones. Nothing was applied between
    60 and 120 kg/ha, so the histogram there is empty even though the extreme
    tails of the two treatments may nearly touch. Looking at counts survives
    that; looking at gaps does not, because a single stray reading in the
    valley closes it. Equal-count binning is worse still: it ignores the
    valleys altogether and puts cuts inside treatments.

    Returns
    -------
    (level_per_point, list_of_levels)
        ``level_per_point`` carries the nominal rate of each record.
    """
    rates = np.asarray(rates, dtype="float64")
    finite = rates[np.isfinite(rates)]
    if finite.size == 0:
        return rates, []

    unique = np.unique(np.round(finite, 3))
    if unique.size == 1:
        return np.where(np.isfinite(rates), unique[0], np.nan), [float(unique[0])]

    if unique.size <= max_levels:
        # Few distinct values: they are the levels, exactly as logged.
        levels = unique
    else:
        levels = _density_levels(finite, max_levels, valley_fraction)
        if levels is None:
            # No treatment structure to find — a genuinely continuous rate, as
            # in a map that varies smoothly. Equal-count bins at least preserve
            # the ordering so a response can still be fitted.
            edges = np.unique(np.quantile(finite, np.linspace(0, 1, max_levels + 1)))
            levels = (edges[:-1] + edges[1:]) / 2.0

    assigned = levels[np.argmin(np.abs(rates[:, None] - levels[None, :]), axis=1)]
    assigned = np.where(np.isfinite(rates), assigned, np.nan)
    return assigned, [float(v) for v in levels]


def _density_levels(
    values: np.ndarray,
    max_levels: int,
    valley_fraction: float,
    min_mass_fraction: float = 0.03,
) -> np.ndarray | None:
    """Find treatment levels as the peaks of the rate histogram.

    ``min_mass_fraction`` is what separates a treatment from a tail. Every
    planned rate in a trial carries a meaningful share of the records — a
    fifth of them in a five-rate design. A group holding half a percent is the
    far tail of a neighbouring treatment that happened to dip below the valley
    floor and come back; folding it into its neighbour is right, and leaving it
    as its own "rate" would invent a treatment nobody applied.

    Returns ``None`` when the values show no separated groups, which is the
    signal to fall back to binning.
    """
    bins = max(60, min(400, values.size // 20))
    counts, edges = np.histogram(values, bins=bins)

    # A bin counts as empty when it holds a negligible share of the busiest
    # bin. An absolute zero is too strict: one stray reading would bridge two
    # treatments that are otherwise cleanly separated.
    floor = max(1.0, counts.max() * valley_fraction)
    occupied = counts > floor
    if not occupied.any():
        return None

    # Split the occupied bins into runs; each run is one treatment.
    boundaries = np.flatnonzero(np.diff(occupied.astype(np.int8)) != 0) + 1
    runs = [r for r in np.split(np.arange(bins), boundaries) if occupied[r[0]]]
    if len(runs) < 2:
        return None

    minimum_mass = values.size * min_mass_fraction
    centres = []
    for run in runs:
        mass = float(counts[run].sum())
        if mass < minimum_mass:
            continue
        low, high = edges[run[0]], edges[run[-1] + 1]
        inside = values[(values >= low) & (values <= high)]
        if inside.size:
            centres.append(float(np.median(inside)))

    if not (2 <= len(centres) <= max_levels):
        return None
    return np.array(sorted(centres))


def aggregate_cells(
    df: pd.DataFrame,
    cell_m: float = 20.0,
    rate_column: str = sch.APPLIED_RATE,
    value_column: str = sch.VALUE,
    group_columns: tuple[str, ...] = (),
    min_points: int = 3,
    units: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Aggregate records into square cells of ``cell_m`` metres.

    The mean within a cell absorbs positioning error and sensor noise. The
    aggregation includes the **rate level** in the grouping key: without it, a
    cell straddling two neighbouring strips would produce an average rate
    nobody applied, and the resulting point would belong to no treatment in
    the trial.
    """
    required = {sch.X, sch.Y, rate_column, value_column}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Columns missing for aggregation: {', '.join(sorted(missing))}.")

    work = df.copy()
    work[rate_column] = pd.to_numeric(work[rate_column], errors="coerce")
    work[value_column] = pd.to_numeric(work[value_column], errors="coerce")
    work = work.dropna(subset=[sch.X, sch.Y, rate_column, value_column])
    if work.empty:
        raise ValueError("No record has a valid rate and a valid yield at the same time.")

    work["_level"], _ = rate_levels(work[rate_column].to_numpy())
    work["_col"] = np.floor(work[sch.X] / cell_m).astype("int64")
    work["_row"] = np.floor(work[sch.Y] / cell_m).astype("int64")

    keys = ["_col", "_row", "_level"] + [c for c in group_columns if c in work.columns]
    aggregation = {
        value_column: ["mean", "std", "count"],
        rate_column: ["mean", "std"],
        sch.X: "mean",
        sch.Y: "mean",
        sch.LON: "mean",
        sch.LAT: "mean",
    }
    aggregation = {k: v for k, v in aggregation.items() if k in work.columns}
    grouped = work.groupby(keys, observed=True).agg(aggregation)
    grouped.columns = ["_".join(c).strip("_") for c in grouped.columns]
    grouped = grouped.reset_index()

    rename = {
        f"{value_column}_mean": "yield",
        f"{value_column}_std": "yield_sd",
        f"{value_column}_count": "n",
        f"{rate_column}_mean": "rate_mean",
        f"{rate_column}_std": "rate_sd",
        f"{sch.X}_mean": sch.X,
        f"{sch.Y}_mean": sch.Y,
        f"{sch.LON}_mean": sch.LON,
        f"{sch.LAT}_mean": sch.LAT,
    }
    grouped = grouped.rename(columns={k: v for k, v in rename.items() if k in grouped.columns})
    grouped = grouped.rename(columns={"_level": "rate"})
    grouped = grouped[grouped["n"] >= min_points]
    if grouped.empty:
        raise ValueError(
            f"No {Phrase(units).length(cell_m, None)} cell gathered at least "
            f"{min_points} records. Reduce the cell size or the minimum required."
        )
    return grouped


def drop_strip_edges(
    df: pd.DataFrame,
    rate_column: str = sch.APPLIED_RATE,
    margin_m: float = 6.0,
) -> tuple[pd.DataFrame, int]:
    """Discard records near the transition between neighbouring rates.

    At the border between two strips the application mixes and the crop feels
    its neighbour. Keeping those points flattens the response curve and pulls
    the optimum toward the middle of the tested range.

    The border does not sit on the points, it sits **between** two passes: if
    the nearest record with a different rate is ``d`` metres away, the
    transition is roughly at ``d/2``. It is that distance, not ``d``, that is
    compared against the margin — otherwise no point would be flagged whenever
    pass spacing exceeds the requested margin.
    """
    if margin_m <= 0 or sch.X not in df.columns:
        return df, 0

    from scipy.spatial import cKDTree

    work = df.dropna(subset=[sch.X, sch.Y, rate_column])
    if len(work) < 10:
        return df, 0

    points = np.column_stack([work[sch.X].to_numpy(), work[sch.Y].to_numpy()])
    levels, level_values = rate_levels(work[rate_column].to_numpy())
    if len(level_values) < 2:
        return df, 0

    boundary_distance = np.full(len(work), np.inf)
    for level in level_values:
        this = levels == level
        other = ~this
        if not this.any() or not other.any():
            continue
        tree = cKDTree(points[other])
        distance, _ = tree.query(points[this], k=1, workers=-1)
        boundary_distance[this] = distance / 2.0

    near_edge = boundary_distance < margin_m
    keep_index = work.index[~near_edge]
    dropped = int(near_edge.sum())
    # A margin that swallows the whole trial is a parameter mistake, not cleaning.
    if dropped >= len(work) * 0.9:
        return df, 0
    return df.loc[keep_index], dropped


def analyze(
    dataset: Dataset,
    rate_column: str = sch.APPLIED_RATE,
    value_column: str = sch.VALUE,
    crop_price: float = 1.0,
    input_cost: float = 0.0,
    cell_m: float = 20.0,
    edge_margin_m: float = 6.0,
    zone_column: str | None = None,
    models: list[str] | None = None,
    rate_max: float | None = None,
    min_points: int | None = None,
    units: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the full economic analysis and return the report.

    The report carries the fitted curve, the economic optimum rate, the
    comparison between uniform and zone-based variable rate, and the summary
    by applied rate — which is the most direct way to check whether the trial
    came out as planned.

    Every number in the report is internal — kg/ha, metres, money per
    hectare — and its notes are deliberately free of any unit, so the
    interface writes them beside figures it has converted itself. ``units``
    reaches the one sentence that does carry a measurement: the refusal
    when no cell gathered enough records, which names a cell size and has
    to name it in the size the reader typed.
    """
    df = dataset.df
    if rate_column not in df.columns:
        raise ValueError(
            f"Rate column '{rate_column}' not found. Numeric columns available: "
            f"{', '.join(dataset.numeric_columns())}."
        )
    if value_column not in df.columns:
        raise ValueError(f"Yield column '{value_column}' not found.")

    notes: list[str] = []
    trimmed, dropped = drop_strip_edges(df, rate_column, edge_margin_m)
    if dropped:
        notes.append(
            f"{dropped} records discarded for falling inside the strip edge margin, "
            "where neighbouring rates mix."
        )

    group_columns = (zone_column,) if zone_column and zone_column in df.columns else ()

    # A dataset produced by joining layers is already one record per cell, so
    # demanding several records per cell would throw all of it away. Anything
    # coming straight off a monitor still needs the usual minimum.
    if min_points is None:
        min_points = 1 if dataset.meta.extra.get("join_report") else 3

    cells = aggregate_cells(
        trimmed, cell_m=cell_m, rate_column=rate_column,
        value_column=value_column, group_columns=group_columns,
        min_points=min_points, units=units,
    )
    notes.append(f"{len(cells)} cells used to fit the curve.")

    rates = cells["rate"].to_numpy(dtype="float64")
    yields = cells["yield"].to_numpy(dtype="float64")
    observed_max = float(np.max(rates))
    search_max = float(rate_max) if rate_max else observed_max

    best, all_fits = response_mod.fit_best(rates, yields, models)

    economics: dict[str, Any] = {}
    curve: list[dict[str, float]] = []
    if crop_price > 0:
        economics = response_mod.optimum_rate(
            best, crop_price, input_cost, rate_min=0.0, rate_max=search_max
        )
        curve = response_mod.profit_curve(
            best, crop_price, input_cost, rate_min=0.0, rate_max=search_max
        )
        if economics["optimum_rate"] > observed_max - 1e-6:
            notes.append(
                "The optimum rate landed at the top of the tested range: the trial "
                "never reached the point of diminishing returns. Treat the value as "
                "'at least this much' and include higher rates next season."
            )

    by_rate = (
        cells.groupby("rate", observed=True)
        .agg(n=("yield", "size"), mean_yield=("yield", "mean"), sd=("yield", "std"))
        .reset_index()
    )
    if crop_price > 0:
        by_rate["mean_profit"] = (
            by_rate["mean_yield"] * crop_price - by_rate["rate"] * input_cost
        )
    rate_table = [
        {k: (round(float(v), 2) if isinstance(v, (int, float, np.floating)) and pd.notna(v) else None)
         for k, v in row.items()}
        for row in by_rate.to_dict("records")
    ]

    zones = _zone_analysis(
        cells, zone_column, crop_price, input_cost, search_max, models
    ) if group_columns else None

    return {
        "rate_column": rate_column,
        "yield_column": value_column,
        "cells": len(cells),
        # Parameters echoed back in internal units, so the interface can show
        # them in whatever unit the user picked.
        "parameters": {
            "cell_m": cell_m,
            "edge_margin_m": edge_margin_m,
            "edge_records_dropped": dropped,
            "min_points_per_cell": min_points,
        },
        "rates_tested": sorted({round(float(r), 1) for r in np.unique(rates)}),
        "chosen_model": best.to_dict(),
        "models_evaluated": [f.to_dict() for f in all_fits],
        "economics": economics,
        "curve": curve,
        "by_rate": rate_table,
        "zones": zones,
        "notes": notes,
        "prices": {"crop_price": crop_price, "input_cost": input_cost},
        "cells_geojson": _cells_geojson(cells),
    }


def _zone_analysis(
    cells: pd.DataFrame,
    zone_column: str,
    crop_price: float,
    input_cost: float,
    rate_max: float,
    models: list[str] | None,
) -> dict[str, Any]:
    """Fit one curve per zone and compare uniform against variable rate.

    The variable rate gain is the difference between summing each zone's
    profit at its own optimum and applying the single best rate everywhere. If
    that difference does not pay for building and loading the map, uniform
    rate is the right call — and the report says so.
    """
    results: list[dict[str, Any]] = []
    weights: list[float] = []
    fits: list[response_mod.ResponseFit] = []

    for zone_value, group in cells.groupby(zone_column, observed=True):
        rates = group["rate"].to_numpy(dtype="float64")
        yields = group["yield"].to_numpy(dtype="float64")
        if len(np.unique(rates)) < 3:
            results.append({
                "zone": str(zone_value),
                "cells": int(len(group)),
                "error": "Fewer than 3 distinct rates in this zone.",
            })
            continue
        try:
            fit, _ = response_mod.fit_best(rates, yields, models)
        except ValueError as exc:
            results.append({"zone": str(zone_value), "cells": int(len(group)), "error": str(exc)})
            continue

        entry: dict[str, Any] = {
            "zone": str(zone_value),
            "cells": int(len(group)),
            "model": fit.label,
            "r2": round(fit.r2, 4),
        }
        if crop_price > 0:
            optimum = response_mod.optimum_rate(fit, crop_price, input_cost, 0.0, rate_max)
            entry.update(optimum)
        results.append(entry)
        weights.append(float(len(group)))
        fits.append(fit)

    comparison: dict[str, Any] = {}
    if crop_price > 0 and len(fits) >= 2:
        total_weight = sum(weights)
        grid = np.linspace(0.0, rate_max, 1001)

        variable_profit = sum(
            w * max(fit.predict(grid) * crop_price - grid * input_cost)
            for fit, w in zip(fits, weights)
        ) / total_weight

        uniform_profit_by_rate = sum(
            w * (fit.predict(grid) * crop_price - grid * input_cost)
            for fit, w in zip(fits, weights)
        ) / total_weight
        best_uniform_index = int(np.argmax(uniform_profit_by_rate))

        gain = float(variable_profit - uniform_profit_by_rate[best_uniform_index])
        comparison = {
            "best_uniform_rate": round(float(grid[best_uniform_index]), 2),
            "uniform_profit": round(float(uniform_profit_by_rate[best_uniform_index]), 2),
            "variable_profit": round(float(variable_profit), 2),
            "gain_per_ha": round(gain, 2),
            # The gain is computed per hectare in the crop's currency; the
            # interface converts it to the chosen area unit and currency symbol,
            # so the wording here stays free of any unit.
            "reading": (
                "Zone-based variable rate beats the best single rate. Weigh the "
                "gain shown here against the cost of building and running the map "
                "before deciding."
            )
            if gain > 0 else
            "On this dataset variable rate did not beat the best single rate: the "
            "zones responded too similarly to justify the map.",
        }

    return {"zone_column": zone_column, "by_zone": results, "comparison": comparison}


def _cells_geojson(cells: pd.DataFrame, limit: int = 8000) -> dict[str, Any]:
    """Aggregated cells as GeoJSON, for drawing on the map."""
    if sch.LON not in cells.columns or sch.LAT not in cells.columns:
        return {"type": "FeatureCollection", "features": []}
    sample = cells if len(cells) <= limit else cells.sample(limit, random_state=0)
    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [float(row[sch.LON]), float(row[sch.LAT])]},
            "properties": {
                "rate": round(float(row["rate"]), 2),
                "yield": round(float(row["yield"]), 1),
                "n": int(row["n"]),
            },
        }
        for _, row in sample.iterrows()
    ]
    return {"type": "FeatureCollection", "features": features}
