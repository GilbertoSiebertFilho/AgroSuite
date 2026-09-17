"""Joining the layers the economic analysis needs.

A trial is almost never one file. The plan comes out of the office software,
the as-applied log comes off the seeder or the spreader, and the yield comes
off the combine months later — three passes over the same ground, logged by
three machines, at three different point densities, on three different days.

They cannot be joined point to point. The GPS positions never coincide, the
densities differ by an order of magnitude, and matching nearest neighbours
would pair a yield reading with whichever rate record happened to be closest,
which is noise. The join happens on a **common grid**: each layer is
aggregated into the same square cells, and the cells are matched.

Cell size is the one parameter that matters. Too small and each cell holds
too few records of the sparsest layer; too large and it spans two treatments
and averages them into a rate nobody applied. Twenty metres is a reasonable
default for strips two machine passes wide.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..core import schema as sch
from ..core.dataset import Dataset
from ..core.units import Phrase

#: How each role contributes to the joined table.
ROLE_COLUMNS = {
    "yield": ("yield", sch.VALUE),
    "as_applied": ("applied_rate", sch.VALUE),
    "plan": ("planned_rate", sch.TARGET_RATE),
    "vigor": ("vigor", "vigor_index"),
}


def _grid_key(df: pd.DataFrame, cell_m: float, origin: tuple[float, float]) -> pd.DataFrame:
    """Add integer cell indices, anchored to a shared origin.

    The origin matters: two layers projected to the same CRS but binned from
    their own minima would land on grids offset from each other, and the join
    would silently match the wrong cells.
    """
    out = df.copy()
    out["_col"] = np.floor((out[sch.X] - origin[0]) / cell_m).astype("int64")
    out["_row"] = np.floor((out[sch.Y] - origin[1]) / cell_m).astype("int64")
    return out


def _aggregate(
    dataset: Dataset,
    column: str,
    out_name: str,
    cell_m: float,
    origin: tuple[float, float],
    target_crs: str,
    min_points: int = 1,
    discrete: bool = False,
) -> pd.DataFrame:
    """Aggregate one layer onto the shared grid.

    ``discrete`` changes the aggregation for a rate layer. A trial applies a
    handful of planned rates, and averaging a cell that straddles two strips
    would invent a rate nobody applied — a cell half at 60 and half at 120
    would come out at 90, which belongs to no treatment. Instead each cell
    takes the rate level most of its records carry, plus the fraction that
    agree (its *purity*). A cell on a strip boundary shows low purity and can
    be dropped, which is the edge margin expressed at cell level.
    """
    ds = dataset
    if ds.metric_crs != target_crs:
        ds = dataset.copy()
        ds.project(target_crs)

    if column not in ds.df.columns:
        raise ValueError(
            f"Layer '{dataset.meta.name}' has no column '{column}'. "
            f"Available: {', '.join(ds.numeric_columns())}."
        )

    frame = ds.df[[sch.X, sch.Y, column]].copy()
    frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna()
    if frame.empty:
        raise ValueError(f"Layer '{dataset.meta.name}' has no valid values in '{column}'.")

    frame = _grid_key(frame, cell_m, origin)

    if discrete:
        from .analysis import rate_levels

        levels, level_values = rate_levels(frame[column].to_numpy())
        frame["_level"] = levels
        if len(level_values) < 2:
            discrete = False  # a single rate: nothing to protect from mixing

    if discrete:
        counts = (
            frame.groupby(["_col", "_row", "_level"], observed=True)
            .agg(n=(column, "size"), mean=(column, "mean"),
                 x=(sch.X, "mean"), y=(sch.Y, "mean"))
            .reset_index()
        )
        totals = counts.groupby(["_col", "_row"], observed=True)["n"].transform("sum")
        counts["purity"] = counts["n"] / totals
        # Keep the dominant level of each cell.
        dominant = counts.sort_values("n", ascending=False).drop_duplicates(["_col", "_row"])
        dominant = dominant[totals.loc[dominant.index] >= min_points]
        grouped = dominant.rename(columns={
            "_level": out_name, "mean": f"{out_name}_measured",
            "n": f"{out_name}_n", "purity": f"{out_name}_purity",
        })
        return grouped[[
            "_col", "_row", out_name, f"{out_name}_measured",
            f"{out_name}_n", f"{out_name}_purity", "x", "y",
        ]]

    grouped = (
        frame.groupby(["_col", "_row"], observed=True)
        .agg(
            value=(column, "mean"),
            sd=(column, "std"),
            n=(column, "size"),
            x=(sch.X, "mean"),
            y=(sch.Y, "mean"),
        )
        .reset_index()
    )
    grouped = grouped[grouped["n"] >= min_points]
    return grouped.rename(columns={
        "value": out_name, "sd": f"{out_name}_sd", "n": f"{out_name}_n",
    })


def join_layers(
    layers: dict[str, Dataset],
    cell_m: float = 20.0,
    min_points_per_cell: int = 1,
    columns: dict[str, str] | None = None,
    min_purity: float = 0.8,
    carry: list[str] | None = None,
    units: dict[str, Any] | None = None,
) -> tuple[Dataset, dict[str, Any]]:
    """Join yield, as-applied and planned-rate layers onto a shared grid.

    Parameters
    ----------
    layers:
        Map of ``role -> Dataset``. ``yield`` is required; ``as_applied`` or
        ``plan`` supplies the rate. ``vigor`` is optional.
    cell_m:
        Side of the shared grid cell, in metres.
    min_points_per_cell:
        Cells below this many records in a layer are dropped from that layer,
        which keeps a single stray reading from speaking for a whole cell.
    columns:
        Optional override of which column to take from each role.
    min_purity:
        Minimum share of a cell's rate records that must agree on one treatment
        level. Cells below it sit on a strip boundary, where two rates mix, and
        are dropped.
    carry:
        Extra columns to bring across from the yield layer — a management zone,
        a soil class, a variety. Without them the joined table loses whatever
        would let the analysis split the field, and the response of two
        different zones gets pooled into one curve that fits neither.
    units:
        The reader's unit set, which reaches the two sentences that quote
        the cell size — the note on the joined dataset and the refusal when
        the layers share no cell. Every number stays metric.

    Returns
    -------
    (Dataset, report)
        A point dataset, one point per joined cell, carrying ``yield``,
        ``applied_rate`` and ``planned_rate`` as available.
    """
    say = Phrase(units)
    if "yield" not in layers:
        raise ValueError(
            "The economic analysis needs a yield layer. Add the harvest file and "
            "mark its role as Yield."
        )
    if "as_applied" not in layers and "plan" not in layers:
        raise ValueError(
            "The economic analysis needs the rate that was applied. Add the "
            "as-applied log, or the plan if that is all you have."
        )

    target_crs = layers["yield"].metric_crs
    if not target_crs:
        raise ValueError("The yield layer has no projected coordinates.")

    # Anchor every grid to the same origin, taken from the yield layer.
    yield_df = layers["yield"].df
    origin = (float(yield_df[sch.X].min()), float(yield_df[sch.Y].min()))

    report: dict[str, Any] = {
        "cell_m": cell_m,
        "metric_crs": target_crs,
        "layers": [],
        "notes": [],
    }

    merged: pd.DataFrame | None = None
    for role, dataset in layers.items():
        mapping = ROLE_COLUMNS.get(role)
        if mapping is None:
            continue
        out_name, default_column = mapping
        column = (columns or {}).get(role, default_column)

        grid = _aggregate(
            dataset, column, out_name, cell_m, origin, target_crs,
            min_points=min_points_per_cell,
            discrete=role in ("as_applied", "plan"),
        )
        report["layers"].append({
            "role": role,
            "dataset": dataset.meta.name,
            "column": column,
            "rows": len(dataset),
            "cells": len(grid),
        })

        if merged is None:
            merged = grid
        else:
            # Inner join: a cell only counts when every layer covers it. A cell
            # with yield but no rate says nothing about the response.
            merged = merged.merge(
                grid.drop(columns=["x", "y"]), on=["_col", "_row"], how="inner"
            )

    overlap_cells = len(merged) if merged is not None else 0

    # ---- drop the cells that straddle a strip boundary
    if merged is not None and not merged.empty:
        for out_name in ("applied_rate", "planned_rate"):
            purity_column = f"{out_name}_purity"
            if purity_column in merged.columns:
                before = len(merged)
                merged = merged[merged[purity_column] >= min_purity]
                dropped = before - len(merged)
                if dropped:
                    report.setdefault("boundary_cells_dropped", 0)
                    report["boundary_cells_dropped"] += dropped

    if report.get("boundary_cells_dropped"):
        report["notes"].append(
            f"{report['boundary_cells_dropped']} cell(s) dropped for straddling a rate "
            f"boundary — fewer than {min_purity:.0%} of their records agreed on one "
            "rate. That is the strip edge being excluded, not missing data."
        )

    if merged is None or merged.empty:
        overlaps = ", ".join(f"{layer['role']}: {layer['cells']}" for layer in report["layers"])
        raise ValueError(
            f"The layers share no {say.length(cell_m, None)} cell. Cells per layer — "
            f"{overlaps}. Either they cover different fields, or the cell is too small "
            "for the sparsest layer."
        )

    # ---- coverage diagnostics
    # Overlap is measured before the purity filter, because cells dropped for
    # straddling a rate boundary are a deliberate choice, not a sign that the
    # layers cover different ground. Conflating the two would have the app
    # warning about a mismatch that does not exist.
    for layer in report["layers"]:
        overlap = overlap_cells / layer["cells"] * 100.0 if layer["cells"] else 0.0
        layer["overlap_pct"] = round(overlap, 1)
        layer["kept_pct"] = round(
            len(merged) / layer["cells"] * 100.0 if layer["cells"] else 0.0, 1
        )
        if overlap < 50:
            report["notes"].append(
                f"Only {overlap:.0f}% of the {layer['role']} layer's cells are covered "
                "by every other layer — they may not be the same field, or the same job."
            )

    if "as_applied" in layers and "plan" in layers:
        deviation = (merged["applied_rate"] - merged["planned_rate"]).abs()
        with np.errstate(divide="ignore", invalid="ignore"):
            relative = float(
                np.nanmedian(deviation / merged["planned_rate"].replace(0, np.nan)) * 100.0
            )
        report["plan_vs_applied"] = {
            "median_abs_deviation": round(float(deviation.median()), 2),
            "median_relative_pct": round(relative, 1) if np.isfinite(relative) else None,
        }
        if np.isfinite(relative) and relative > 15:
            report["notes"].append(
                f"The machine applied a median {relative:.0f}% away from the plan. "
                "Analyse against the as-applied rate, not the plan — the response "
                "follows what actually went out."
            )

    # ---- carry the yield layer's grouping columns across
    if carry:
        yield_ds = layers["yield"]
        if yield_ds.metric_crs != target_crs:
            yield_ds = layers["yield"].copy()
            yield_ds.project(target_crs)
        available = [c for c in carry if c in yield_ds.df.columns]
        if available:
            extra = _grid_key(yield_ds.df[[sch.X, sch.Y, *available]].copy(), cell_m, origin)
            # The cell's majority value, not its mean: a zone label of 0.5
            # would belong to neither zone.
            majority = (
                extra.groupby(["_col", "_row"], observed=True)[available]
                .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else None)
                .reset_index()
            )
            merged = merged.merge(majority, on=["_col", "_row"], how="left")
            report["carried"] = available

    # ---- back to a dataset, with geographic coordinates for the map
    from pyproj import Transformer

    from ..core.crs import WGS84
    from ..core.dataset import DatasetMeta

    transformer = Transformer.from_crs(target_crs, WGS84, always_xy=True)
    lon, lat = transformer.transform(merged["x"].to_numpy(), merged["y"].to_numpy())

    frame = merged.drop(columns=["_col", "_row"]).copy()
    frame[sch.LON] = lon
    frame[sch.LAT] = lat
    if "yield" in frame.columns:
        frame[sch.VALUE] = frame["yield"]
    if "applied_rate" in frame.columns:
        frame[sch.APPLIED_RATE] = frame["applied_rate"]
    elif "planned_rate" in frame.columns:
        frame[sch.APPLIED_RATE] = frame["planned_rate"]
        report["notes"].append(
            "No as-applied log: the analysis uses the planned rate, which assumes "
            "the machine executed the map exactly."
        )
    if "planned_rate" in frame.columns:
        frame[sch.TARGET_RATE] = frame["planned_rate"]

    meta = DatasetMeta(
        name="Joined layers",
        source_format="join",
        brand="generic",
        brand_label="Joined layers",
        operation="harvest",
        crop=layers["yield"].meta.crop,
        field_name=layers["yield"].meta.field_name,
        value_label="Yield",
        notes=[
            f"{len(frame)} cells of {say.length(cell_m, None)} shared by "
            f"{len(report['layers'])} layer(s).",
            *report["notes"],
        ],
        extra={"join_report": report},
    )
    report["cells"] = len(frame)
    return Dataset(frame, meta), report


def suggest_cell_size(layers: dict[str, Dataset]) -> float:
    """Suggest a cell size from the sparsest layer's point density.

    The binding constraint is the layer with the fewest points per hectare: the
    cell has to be big enough that even that layer puts a handful of records in
    each one. The result is rounded to a round number of metres, because a cell
    of 17.3 m helps nobody read the output.
    """
    # The plan layer is left out: a prescription is a map, not a log, and its
    # point spacing says nothing about how densely the ground was measured.
    densities = []
    for role, dataset in layers.items():
        if role == "plan":
            continue
        area = dataset.area_ha()
        if area > 0:
            densities.append(len(dataset) / area)
    if not densities:
        return 20.0

    sparsest = min(densities)
    # Aim for about 8 records of the sparsest layer per cell.
    # points/ha * (cell_m^2 / 10000) = 8  ->  cell_m = sqrt(80000 / density)
    cell = float(np.sqrt(80_000.0 / sparsest)) if sparsest > 0 else 20.0
    for step in (5, 10, 15, 20, 25, 30, 40, 50):
        if cell <= step:
            return float(step)
    return 60.0
