"""Cleaning filters for monitor data.

Every step is a class with the same interface: it receives the context with
the prepared vectors and returns a boolean mask marking the records to
**remove**, plus the statistics that feed the report.

The order follows established practice in the yield-map editing literature
(position -> time -> width -> overlap -> headland -> statistical filters),
because each step depends on the previous one having cleaned up what feeds
its calculation: there is no point hunting local outliers before removing the
turning points, which are exactly the ones that drag the neighbourhood.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..core import schema as sch
from ..core.units import Phrase

#: Sentinel meaning "no earlier point covered this cell".
_NO_COVER = np.iinfo(np.int32).max


@dataclass
class StepResult:
    """Result of one cleaning step."""

    key: str
    label: str
    removed: int
    remaining: int
    detail: str = ""
    skipped: bool = False
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "removed": int(self.removed),
            "remaining": int(self.remaining),
            "detail": self.detail,
            "skipped": self.skipped,
            "params": self.params,
        }


class Context:
    """State shared across the steps of a cleaning run.

    Holds the mask of still-valid records, the removal reason for each
    discarded record, and the expensive caches — coverage grid, spatial
    neighbourhood — that several steps reuse.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        value_column: str = sch.VALUE,
        units: dict[str, Any] | None = None,
        operation: str | None = None,
    ) -> None:
        self.df = df
        self.value_column = value_column
        # What each step says about itself is a sentence with a width, a
        # speed or a rate written into it, so it is written in the unit set
        # the reader works in; the operation decides whether the main
        # variable is a yield or an input rate.
        self.operation = operation
        self.say = Phrase(units)
        self.n = len(df)
        self.alive = np.ones(self.n, dtype=bool)
        self.reason = np.full(self.n, "", dtype=object)
        self._coverage: dict[str, Any] | None = None
        self._kdtree = None
        self.messages: list[str] = []

    # -- column access ---------------------------------------------------
    def column(self, name: str) -> np.ndarray | None:
        """Float vector for the column, or ``None`` if absent."""
        if name not in self.df.columns:
            return None
        return pd.to_numeric(self.df[name], errors="coerce").to_numpy(dtype="float64")

    @property
    def values(self) -> np.ndarray | None:
        return self.column(self.value_column)

    def apply(self, remove: np.ndarray, reason: str) -> int:
        """Discard the marked records, recording the reason."""
        remove = np.asarray(remove, dtype=bool) & self.alive
        count = int(remove.sum())
        if count:
            self.reason[remove] = reason
            self.alive[remove] = False
        return count

    # -- caches ----------------------------------------------------------
    def coverage(self, cell_size: float | None = None) -> dict[str, Any] | None:
        """Coverage grid for the field, built on demand.

        For each cell it stores the index of the **first** record that covered
        it. With that, a record's overlap is simply the fraction of its cells
        already covered by earlier records — which avoids unioning tens of
        thousands of polygons.
        """
        if self._coverage is not None:
            return self._coverage

        x = self.column(sch.X)
        y = self.column(sch.Y)
        swath = self.column(sch.SWATH)
        if x is None or y is None or swath is None:
            return None

        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(swath) & (swath > 0)
        if valid.sum() < 10:
            return None

        median_swath = float(np.median(swath[valid]))
        cell = cell_size or max(0.5, min(2.0, median_swath / 8.0))

        x0, x1 = float(np.min(x[valid])), float(np.max(x[valid]))
        y0, y1 = float(np.min(y[valid])), float(np.max(y[valid]))
        margin = median_swath
        x0, y0 = x0 - margin, y0 - margin
        x1, y1 = x1 + margin, y1 + margin

        # Cap the grid's memory by growing the cell if needed.
        max_cells = 24_000_000
        while ((x1 - x0) / cell + 1) * ((y1 - y0) / cell + 1) > max_cells:
            cell *= 1.5

        ncols = int((x1 - x0) / cell) + 1
        nrows = int((y1 - y0) / cell) + 1

        # Cross-track samples cover the swath.
        heading = self.column(sch.HEADING)
        if heading is None:
            heading = np.zeros(self.n)
        rad = np.radians(np.nan_to_num(heading))
        # Heading 0 = north; the vector perpendicular to it is (cos, -sin).
        perp_x, perp_y = np.cos(rad), -np.sin(rad)

        samples = max(3, int(np.ceil(median_swath / cell)) + 1)
        offsets = np.linspace(-0.5, 0.5, samples)

        half = np.where(valid, swath, median_swath)
        sample_x = x[:, None] + perp_x[:, None] * half[:, None] * offsets[None, :]
        sample_y = y[:, None] + perp_y[:, None] * half[:, None] * offsets[None, :]

        col = np.clip(((sample_x - x0) / cell).astype(np.int64), 0, ncols - 1)
        row = np.clip(((sample_y - y0) / cell).astype(np.int64), 0, nrows - 1)
        flat = (row * ncols + col).ravel()

        point_index = np.repeat(np.arange(self.n, dtype=np.int64), samples)
        invalid = ~np.repeat(valid, samples)
        point_index = point_index.copy()
        point_index[invalid] = _NO_COVER

        first_cover = np.full(nrows * ncols, _NO_COVER, dtype=np.int64)
        np.minimum.at(first_cover, flat, point_index)

        self._coverage = {
            "cell": cell,
            "ncols": ncols,
            "nrows": nrows,
            "x0": x0,
            "y0": y0,
            "flat": flat,
            "samples": samples,
            "first_cover": first_cover,
            "valid": valid,
        }
        return self._coverage

    def neighbors(self, k: int = 12):
        """Indices of the ``k`` nearest neighbours of each surviving record."""
        if self._kdtree is not None:
            return self._kdtree

        from scipy.spatial import cKDTree

        x = self.column(sch.X)
        y = self.column(sch.Y)
        if x is None or y is None:
            return None
        alive_idx = np.flatnonzero(self.alive & np.isfinite(x) & np.isfinite(y))
        if alive_idx.size < k + 1:
            return None
        points = np.column_stack([x[alive_idx], y[alive_idx]])
        tree = cKDTree(points)
        _, idx = tree.query(points, k=min(k + 1, alive_idx.size), workers=-1)
        self._kdtree = (alive_idx, idx[:, 1:])  # drop the point itself
        return self._kdtree


# ==========================================================================
# Steps
# ==========================================================================

class CleaningStep:
    """Common contract for the cleaning steps."""

    key = "step"
    label = "Step"
    description = ""
    defaults: dict[str, Any] = {}

    def __init__(self, **params: Any) -> None:
        self.params = {**self.defaults, **params}

    def run(self, ctx: Context) -> StepResult:  # pragma: no cover - interface
        raise NotImplementedError

    def _result(
        self, removed: int, ctx: Context, detail: str = "", skipped: bool = False,
        **measured: Any,
    ):
        """The step's result, with the sentence that describes its settings.

        ``measured`` is what the step worked out from the data itself — the
        full swath it estimated, the cell it covered the field with — and it
        is kept beside the settings rather than only inside the sentence, so
        that :func:`detail_for` can write the sentence again for a reader in
        another unit set without running the filter a second time. A step
        that did not run says why in ``detail`` instead, and that sentence
        has no numbers in it.
        """
        params = {**self.params, **measured}
        if not skipped and not detail:
            detail = detail_for(
                {"key": self.key, "params": params}, ctx.say, ctx.operation)
        return StepResult(
            key=self.key, label=self.label, removed=removed,
            remaining=int(ctx.alive.sum()), detail=detail, skipped=skipped,
            params=params,
        )


class NullValueFilter(CleaningStep):
    """Discard records with no valid reading of the variable."""

    key = "null_value"
    label = "Null and non-positive values"
    description = (
        "Removes records with no reading or a value at or below zero. Those are "
        "stops, stretches with the implement lifted and sensor faults — never "
        "real production."
    )
    defaults = {"drop_zero": True, "drop_negative": True}

    def run(self, ctx: Context) -> StepResult:
        values = ctx.values
        if values is None:
            return self._result(0, ctx, "Value column missing.", skipped=True)
        remove = ~np.isfinite(values)
        if self.params.get("drop_negative", True):
            remove |= values < 0
        if self.params.get("drop_zero", True):
            remove |= values == 0
        return self._result(ctx.apply(remove, self.label), ctx)


class ValueRangeFilter(CleaningStep):
    """Apply absolute bounds chosen by the agronomist."""

    key = "value_range"
    label = "Absolute value range"
    description = (
        "Cuts values outside the physically plausible range for the crop. Leave "
        "a box empty to skip that bound."
    )
    defaults = {"min": None, "max": None}

    def run(self, ctx: Context) -> StepResult:
        values = ctx.values
        if values is None:
            return self._result(0, ctx, "Value column missing.", skipped=True)
        lo, hi = self.params.get("min"), self.params.get("max")
        if lo is None and hi is None:
            return self._result(0, ctx, "No bounds set.", skipped=True)
        remove = np.zeros(ctx.n, dtype=bool)
        if lo is not None:
            remove |= values < float(lo)
        if hi is not None:
            remove |= values > float(hi)
        remove &= np.isfinite(values)
        return self._result(ctx.apply(remove, self.label), ctx)


class SpeedRangeFilter(CleaningStep):
    """Remove records outside the operating speed range."""

    key = "speed_range"
    label = "Speed range"
    description = (
        "Below the minimum the machine is stopping or turning; above the maximum "
        "it is road travel or a GPS error."
    )
    defaults = {"min": 1.5, "max": 20.0}

    def run(self, ctx: Context) -> StepResult:
        speed = ctx.column(sch.SPEED)
        if speed is None:
            return self._result(0, ctx, "Speed unavailable.", skipped=True)
        lo = float(self.params.get("min") or 0)
        hi = float(self.params.get("max") or 1e9)
        remove = np.isfinite(speed) & ((speed < lo) | (speed > hi))
        remove |= ~np.isfinite(speed)
        return self._result(ctx.apply(remove, self.label), ctx)


class SpeedChangeFilter(CleaningStep):
    """Remove records during sharp acceleration or braking.

    On a combine, the grain in transit inside the machine keeps the sensor
    reporting the flow of the previous speed. The result is artificial peaks
    and troughs every time the speed changes quickly.
    """

    key = "speed_change"
    label = "Sharp speed change"
    description = (
        "Discards records where speed changed more than the limit between "
        "consecutive readings — flow inertia distorts the measured value."
    )
    defaults = {"max_change_pct": 25.0}

    def run(self, ctx: Context) -> StepResult:
        speed = ctx.column(sch.SPEED)
        if speed is None:
            return self._result(0, ctx, "Speed unavailable.", skipped=True)
        limit = float(self.params.get("max_change_pct", 25.0)) / 100.0
        previous = np.roll(speed, 1)
        previous[0] = speed[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            change = np.abs(speed - previous) / np.where(previous > 0, previous, np.nan)
        remove = np.isfinite(change) & (change > limit)
        return self._result(ctx.apply(remove, self.label), ctx)


class SwathWidthFilter(CleaningStep):
    """Remove passes made with a partially filled header."""

    key = "swath_partial"
    label = "Partial swath"
    description = (
        "When the header is not working at full width, the monitor divides the "
        "mass by an area larger than the real one and the value drops "
        "artificially."
    )
    defaults = {"min_fraction": 0.5}

    def run(self, ctx: Context) -> StepResult:
        swath = ctx.column(sch.SWATH)
        if swath is None:
            return self._result(0, ctx, "Swath width unavailable.", skipped=True)
        valid = np.isfinite(swath) & (swath > 0)
        if valid.sum() < 10:
            return self._result(0, ctx, "Not enough width readings.", skipped=True)
        full = float(np.percentile(swath[valid], 90))
        fraction = float(self.params.get("min_fraction", 0.5))
        remove = valid & (swath < full * fraction)
        remove |= ~valid
        return self._result(
            ctx.apply(remove, self.label), ctx, full_swath_m=full,
        )


class OverlapFilter(CleaningStep):
    """Remove records over already-worked ground.

    Driving back over a strip that was already harvested, the header picks up
    little or no crop, but the monitor keeps counting the full area. It is the
    main source of spurious low values in a yield map.
    """

    key = "overlap"
    label = "Swath overlap"
    description = (
        "Compares the area each record covers against what had already been "
        "worked, and discards whatever exceeds the tolerated fraction."
    )
    defaults = {"max_overlap_pct": 40.0}

    def run(self, ctx: Context) -> StepResult:
        coverage = ctx.coverage()
        if coverage is None:
            return self._result(0, ctx, "No swath geometry to work from.", skipped=True)

        first_cover = coverage["first_cover"]
        flat = coverage["flat"]
        samples = coverage["samples"]
        point_index = np.repeat(np.arange(ctx.n, dtype=np.int64), samples)
        prior = first_cover[flat] < point_index
        fraction = prior.reshape(ctx.n, samples).mean(axis=1)

        limit = float(self.params.get("max_overlap_pct", 40.0)) / 100.0
        remove = coverage["valid"] & (fraction > limit)
        removed = ctx.apply(remove, self.label)
        return self._result(removed, ctx, cell_m=float(coverage["cell"]))


class BoundaryFilter(CleaningStep):
    """Remove the headland and field edge.

    Distance to the edge is measured over the coverage grid itself: a distance
    transform gives, for each worked cell, how many metres remain until the
    unworked area. That follows irregularly shaped fields, which a simple
    convex hull would not.
    """

    key = "boundary"
    label = "Field edge"
    description = (
        "Discards records closer than the given distance to the edge of the "
        "worked area — where turning, compaction and overlap happen."
    )
    defaults = {"buffer_m": 0.0}

    def run(self, ctx: Context) -> StepResult:
        buffer_m = float(self.params.get("buffer_m") or 0.0)
        if buffer_m <= 0:
            return self._result(0, ctx, "Disabled.", skipped=True)

        coverage = ctx.coverage()
        if coverage is None:
            return self._result(0, ctx, "No swath geometry to work from.", skipped=True)

        from scipy import ndimage

        nrows, ncols = coverage["nrows"], coverage["ncols"]
        covered = (coverage["first_cover"] != _NO_COVER).reshape(nrows, ncols)
        distance = ndimage.distance_transform_edt(covered) * coverage["cell"]

        x = ctx.column(sch.X)
        y = ctx.column(sch.Y)
        col = np.clip(((x - coverage["x0"]) / coverage["cell"]).astype(np.int64), 0, ncols - 1)
        row = np.clip(((y - coverage["y0"]) / coverage["cell"]).astype(np.int64), 0, nrows - 1)
        point_distance = distance[row, col]

        remove = coverage["valid"] & (point_distance < buffer_m)
        return self._result(ctx.apply(remove, self.label), ctx)


class PassEndsFilter(CleaningStep):
    """Remove the start and end of every pass.

    Entering a pass the flow has not settled yet; leaving it, whatever remains
    inside the machine keeps being weighed against ground that is already
    finished. Both stretches produce values unrelated to where they are logged.
    """

    key = "pass_ends"
    label = "Pass start and end"
    description = (
        "Discards the first and last metres of each pass, where the flow inside "
        "the machine does not yet match the point being logged."
    )
    defaults = {"start_m": 6.0, "end_m": 6.0}

    def run(self, ctx: Context) -> StepResult:
        if sch.PASS not in ctx.df.columns:
            return self._result(0, ctx, "Passes not identified.", skipped=True)
        distance = ctx.column(sch.DISTANCE)
        if distance is None:
            return self._result(0, ctx, "Distance between records unavailable.", skipped=True)

        start_m = float(self.params.get("start_m") or 0.0)
        end_m = float(self.params.get("end_m") or 0.0)
        if start_m <= 0 and end_m <= 0:
            return self._result(0, ctx, "Disabled.", skipped=True)

        step = np.nan_to_num(distance, nan=0.0)
        pass_id = ctx.df[sch.PASS].to_numpy()
        remove = np.zeros(ctx.n, dtype=bool)

        frame = pd.DataFrame({"pass": pass_id, "step": step})
        cumulative = frame.groupby("pass")["step"].cumsum()
        totals = frame.groupby("pass")["step"].transform("sum")

        if start_m > 0:
            remove |= (cumulative <= start_m).to_numpy()
        if end_m > 0:
            remove |= ((totals - cumulative) <= end_m).to_numpy()

        return self._result(ctx.apply(remove, self.label), ctx)


class ShortPassFilter(CleaningStep):
    """Remove passes too short to be trusted."""

    key = "short_pass"
    label = "Short passes"
    description = (
        "Passes with only a few records are usually turns, headland touch-ups "
        "or a wrong entry — they do not represent the crop."
    )
    defaults = {"min_points": 8}

    def run(self, ctx: Context) -> StepResult:
        if sch.PASS not in ctx.df.columns:
            return self._result(0, ctx, "Passes not identified.", skipped=True)
        minimum = int(self.params.get("min_points", 8))
        pass_id = pd.Series(ctx.df[sch.PASS].to_numpy())
        counts = pass_id.map(pass_id[ctx.alive].value_counts()).fillna(0).to_numpy()
        remove = counts < minimum
        return self._result(ctx.apply(remove, self.label), ctx)


class MoistureFilter(CleaningStep):
    """Apply grain moisture bounds."""

    key = "moisture"
    label = "Moisture range"
    description = (
        "Moisture outside the expected range points to an uncalibrated sensor "
        "or a reading taken empty, and it corrupts the dry-mass correction."
    )
    defaults = {"min": 5.0, "max": 40.0}

    def run(self, ctx: Context) -> StepResult:
        moisture = ctx.column(sch.MOISTURE)
        if moisture is None:
            return self._result(0, ctx, "Moisture unavailable.", skipped=True)
        lo = float(self.params.get("min") or 0)
        hi = float(self.params.get("max") or 100)
        remove = np.isfinite(moisture) & ((moisture < lo) | (moisture > hi))
        return self._result(ctx.apply(remove, self.label), ctx)


class PositionFilter(CleaningStep):
    """Remove repeated coordinates or impossible jumps."""

    key = "position"
    label = "Inconsistent position"
    description = (
        "A repeated coordinate means the GPS froze; too large a jump between "
        "consecutive readings means correction was lost."
    )
    defaults = {"max_jump_m": 25.0, "drop_duplicates": True}

    def run(self, ctx: Context) -> StepResult:
        x, y = ctx.column(sch.X), ctx.column(sch.Y)
        if x is None or y is None:
            return self._result(0, ctx, "Projected coordinates unavailable.", skipped=True)

        remove = ~np.isfinite(x) | ~np.isfinite(y)
        if self.params.get("drop_duplicates", True):
            duplicated = pd.DataFrame({"x": np.round(x, 3), "y": np.round(y, 3)}).duplicated()
            remove |= duplicated.to_numpy()

        max_jump = float(self.params.get("max_jump_m") or 0)
        if max_jump > 0:
            dx = np.diff(x, prepend=x[0])
            dy = np.diff(y, prepend=y[0])
            step = np.hypot(dx, dy)
            step[0] = 0.0
            remove |= np.isfinite(step) & (step > max_jump)

        return self._result(ctx.apply(remove, self.label), ctx)


class GlobalOutlierFilter(CleaningStep):
    """Remove extreme values relative to the whole dataset."""

    key = "global_outlier"
    label = "Global outliers"
    description = (
        "Trims the tail of the whole field's distribution. Use standard "
        "deviation when the distribution is symmetric and percentiles when it "
        "is skewed."
    )
    defaults = {"method": "std", "k": 3.0, "lower_pct": 1.0, "upper_pct": 99.0}

    def run(self, ctx: Context) -> StepResult:
        values = ctx.values
        if values is None:
            return self._result(0, ctx, "Value column missing.", skipped=True)
        alive_values = values[ctx.alive]
        alive_values = alive_values[np.isfinite(alive_values)]
        if alive_values.size < 20:
            return self._result(0, ctx, "Not enough records.", skipped=True)

        method = self.params.get("method", "std")
        measured: dict[str, Any] = {}
        if method == "percentile":
            lo = float(np.percentile(alive_values, float(self.params.get("lower_pct", 1.0))))
            hi = float(np.percentile(alive_values, float(self.params.get("upper_pct", 99.0))))
        else:
            mean = float(np.mean(alive_values))
            std = float(np.std(alive_values, ddof=1))
            lo, hi = mean - float(self.params.get("k", 3.0)) * std, mean + float(
                self.params.get("k", 3.0)) * std
            measured = {"mean": mean, "sd": std}

        remove = np.isfinite(values) & ((values < lo) | (values > hi))
        return self._result(
            ctx.apply(remove, self.label), ctx,
            accepted_min=lo, accepted_max=hi, **measured,
        )


class LocalOutlierFilter(CleaningStep):
    """Remove values that disagree with their immediate neighbourhood.

    This is the filter that separates real variability from noise: in a crop,
    nearby points tend to look alike. A record far from the median of its
    neighbours, measured in median absolute deviations, is a sensor error —
    not a high-yielding spot.
    """

    key = "local_outlier"
    label = "Local outliers"
    description = (
        "Compares each record with the median of its nearest neighbours and "
        "discards those further away than the limit, in median absolute "
        "deviations."
    )
    defaults = {"k_neighbors": 12, "threshold": 3.5}

    def run(self, ctx: Context) -> StepResult:
        values = ctx.values
        if values is None:
            return self._result(0, ctx, "Value column missing.", skipped=True)

        neighbors = ctx.neighbors(int(self.params.get("k_neighbors", 12)))
        if neighbors is None:
            return self._result(0, ctx, "Neighbourhood too small.", skipped=True)

        alive_idx, neighbor_idx = neighbors
        local = values[alive_idx]
        neighbor_values = values[alive_idx[neighbor_idx]]

        median = np.nanmedian(neighbor_values, axis=1)
        mad = np.nanmedian(np.abs(neighbor_values - median[:, None]), axis=1)
        # 1.4826 puts the MAD on the scale of a standard deviation under normality.
        scale = mad * 1.4826
        # Where the neighbourhood is nearly constant the MAD collapses and any
        # difference would look like an outlier; a relative floor avoids that.
        floor = np.nanmedian(np.abs(local - np.nanmedian(local))) * 1.4826 * 0.1
        scale = np.where(scale > floor, scale, floor)

        with np.errstate(divide="ignore", invalid="ignore"):
            score = np.abs(local - median) / scale

        threshold = float(self.params.get("threshold", 3.5))
        flagged = np.isfinite(score) & (score > threshold)

        remove = np.zeros(ctx.n, dtype=bool)
        remove[alive_idx[flagged]] = True
        return self._result(ctx.apply(remove, self.label), ctx)


# ==========================================================================
# What each step says about its own settings
# ==========================================================================
# A step's detail line is prose — "a 3 m strip in from the worked edge" —
# with a width, a speed or a rate written into it. It is composed here,
# from the settings and from what the step measured, rather than inside
# ``run``: a cleaning report is saved with the project and printed months
# later, and the page and the screen have to be able to say it again in
# whatever unit set the reader is in without running the filters again.
# ``None`` for the unit set is the metric store, which is what everything
# here is stored in.


DETAIL_WRITERS: dict[str, Any] = {
    # The unit is written once, at the end of the range, the way a range is
    # spoken: "1.5-20 km/h", not "1.5 km/h-20 km/h".
    "speed_range": lambda p, say, op: (
        f"Accepted range: {say.number(say.to_speed(p.get('min') or 0), 1).rstrip('0').rstrip('.')}"
        f"-{say.speed(p.get('max') or 0, None)}."
    ),
    "speed_change": lambda p, say, op: (
        f"Maximum change tolerated: {float(p.get('max_change_pct', 25.0)):g}%."
    ),
    "swath_partial": lambda p, say, op: (
        f"Full width estimated at {say.length(p.get('full_swath_m'), 2)}; minimum "
        f"accepted {say.length((p.get('full_swath_m') or 0) * float(p.get('min_fraction', 0.5)), 2)}."
    ),
    "overlap": lambda p, say, op: (
        f"{say.length(p.get('cell_m'), 2)} cell; tolerating "
        f"{float(p.get('max_overlap_pct', 40.0)):g}% of repeated area."
    ),
    "boundary": lambda p, say, op: (
        f"{say.length(p.get('buffer_m') or 0, None)} strip in from the worked edge."
    ),
    "pass_ends": lambda p, say, op: (
        f"{say.length(p.get('start_m') or 0, None)} at the start and "
        f"{say.length(p.get('end_m') or 0, None)} at the end of each pass."
    ),
    "short_pass": lambda p, say, op: (
        f"At least {int(p.get('min_points', 8))} records per pass."
    ),
    "moisture": lambda p, say, op: (
        f"Accepted range: {float(p.get('min') or 0):g}-{float(p.get('max') or 0):g}%."
    ),
    "position": lambda p, say, op: (
        f"Maximum jump accepted: {say.length(p.get('max_jump_m') or 0, None)}."
    ),
    # The interval is in the main variable's own unit — a yield on a harvest,
    # an input rate on anything else, which is what ``op`` decides.
    "global_outlier": lambda p, say, op: (
        (f"Mean {say.rate(p.get('mean'), op)} +/- {float(p.get('k', 3.0)):g} x "
         f"{say.rate(p.get('sd'), op)}."
         if p.get("method", "std") != "percentile"
         else f"Percentiles {p.get('lower_pct')}-{p.get('upper_pct')}.")
        + f" Accepted interval: {say.rate(p.get('accepted_min'), op)} to "
        + f"{say.rate(p.get('accepted_max'), op)}."
    ),
    "local_outlier": lambda p, say, op: (
        f"{int(p.get('k_neighbors', 12))} neighbours, limit of "
        f"{float(p.get('threshold', 3.5)):g} deviations."
    ),
}


def detail_for(step: dict[str, Any], say: Phrase, operation: str | None = None) -> str:
    """The detail line of one step, in the unit set ``say`` carries.

    ``step`` is the step as the report stores it. A step that was skipped,
    or one whose settings have nothing to say, keeps the sentence it
    already has: those carry no number and mean the same in every unit.
    """
    if step.get("skipped"):
        return str(step.get("detail") or "")
    writer = DETAIL_WRITERS.get(str(step.get("key") or ""))
    if writer is None:
        return str(step.get("detail") or "")
    return writer(step.get("params") or {}, say, operation)


#: Registry of available steps, in the recommended order of execution.
STEP_CLASSES: tuple[type[CleaningStep], ...] = (
    NullValueFilter,
    PositionFilter,
    ValueRangeFilter,
    MoistureFilter,
    SpeedRangeFilter,
    SpeedChangeFilter,
    SwathWidthFilter,
    ShortPassFilter,
    PassEndsFilter,
    OverlapFilter,
    BoundaryFilter,
    GlobalOutlierFilter,
    LocalOutlierFilter,
)

STEPS_BY_KEY = {cls.key: cls for cls in STEP_CLASSES}
