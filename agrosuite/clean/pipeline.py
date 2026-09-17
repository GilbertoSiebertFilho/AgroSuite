"""Running the cleaning and writing its report.

Cleaning happens in two phases. First the **corrections**, which change
values without discarding anything (flow delay, dry-mass conversion). Then
the **filters**, which only mark records for removal. The split matters:
correcting after filtering would apply the correction to a series full of
holes, and the flow's time shift would stop making sense.

Nothing is overwritten. The result carries the clean set, the removed set
with the reason for each discard, and the comparative report — which is the
material for judging whether the cleaning was appropriate or overdone.

The report's numbers are metric, like everything else stored here. Its
prose is not converted afterwards but written with the reader's unit set
in hand: :func:`run` takes one, and :func:`restate` writes a stored
report's sentences again — from the settings and the measurements the
report already carries — for a reader working in another one. That is
what lets the unit picker change a cleaning report that was made an hour
ago, and the printed page carry the sentences in the units it was asked
for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..core import schema as sch
from ..core.dataset import Dataset
from ..core.units import Phrase
from . import steps as steps_mod

#: Starting configuration per operation type. These are a starting point —
#: the interface exposes every parameter for adjustment.
PRESETS: dict[str, dict[str, Any]] = {
    "harvest": {
        "label": "Harvest (yield map)",
        "description": (
            "Full sequence: corrects the flow delay and applies every filter, "
            "including overlap and field edge."
        ),
        "corrections": {"flow_delay_s": 12.0},
        "steps": {
            "null_value": {"enabled": True},
            "position": {"enabled": True, "max_jump_m": 25.0, "drop_duplicates": True},
            "value_range": {"enabled": False, "min": None, "max": None},
            "moisture": {"enabled": True, "min": 5.0, "max": 40.0},
            "speed_range": {"enabled": True, "min": 1.5, "max": 20.0},
            "speed_change": {"enabled": True, "max_change_pct": 25.0},
            "swath_partial": {"enabled": True, "min_fraction": 0.5},
            "short_pass": {"enabled": True, "min_points": 8},
            "pass_ends": {"enabled": True, "start_m": 6.0, "end_m": 6.0},
            "overlap": {"enabled": True, "max_overlap_pct": 40.0},
            "boundary": {"enabled": False, "buffer_m": 0.0},
            "global_outlier": {"enabled": True, "method": "std", "k": 3.0},
            "local_outlier": {"enabled": True, "k_neighbors": 12, "threshold": 3.5},
        },
    },
    "application": {
        "label": "Application (as-applied)",
        "description": (
            "No flow delay and no local outlier filter: in a variable rate "
            "application, an abrupt rate change between zones is the signal, "
            "not the noise."
        ),
        "corrections": {"flow_delay_s": 0.0},
        "steps": {
            "null_value": {"enabled": True, "drop_zero": False},
            "position": {"enabled": True, "max_jump_m": 30.0, "drop_duplicates": True},
            "value_range": {"enabled": False, "min": None, "max": None},
            "moisture": {"enabled": False},
            "speed_range": {"enabled": True, "min": 1.0, "max": 30.0},
            "speed_change": {"enabled": False, "max_change_pct": 40.0},
            "swath_partial": {"enabled": True, "min_fraction": 0.3},
            "short_pass": {"enabled": True, "min_points": 5},
            "pass_ends": {"enabled": True, "start_m": 3.0, "end_m": 3.0},
            "overlap": {"enabled": True, "max_overlap_pct": 50.0},
            "boundary": {"enabled": False, "buffer_m": 0.0},
            "global_outlier": {"enabled": True, "method": "percentile",
                               "lower_pct": 0.5, "upper_pct": 99.5},
            "local_outlier": {"enabled": False, "k_neighbors": 12, "threshold": 4.0},
        },
    },
    "planting": {
        "label": "Seeding / planting",
        "description": "Focused on metering faults and turning stretches.",
        "corrections": {"flow_delay_s": 0.0},
        "steps": {
            "null_value": {"enabled": True},
            "position": {"enabled": True, "max_jump_m": 25.0, "drop_duplicates": True},
            "value_range": {"enabled": False, "min": None, "max": None},
            "moisture": {"enabled": False},
            "speed_range": {"enabled": True, "min": 2.0, "max": 15.0},
            "speed_change": {"enabled": True, "max_change_pct": 30.0},
            "swath_partial": {"enabled": True, "min_fraction": 0.5},
            "short_pass": {"enabled": True, "min_points": 8},
            "pass_ends": {"enabled": True, "start_m": 4.0, "end_m": 4.0},
            "overlap": {"enabled": True, "max_overlap_pct": 40.0},
            "boundary": {"enabled": False, "buffer_m": 0.0},
            "global_outlier": {"enabled": True, "method": "std", "k": 3.5},
            "local_outlier": {"enabled": True, "k_neighbors": 12, "threshold": 4.0},
        },
    },
    "vigor": {
        "label": "Vigour / Augmenta",
        "description": (
            "Light cleaning: a vigour index legitimately varies between "
            "neighbouring plants, so only position and extremes are treated."
        ),
        "corrections": {"flow_delay_s": 0.0},
        "steps": {
            "null_value": {"enabled": True, "drop_zero": False},
            "position": {"enabled": True, "max_jump_m": 30.0, "drop_duplicates": True},
            "value_range": {"enabled": False, "min": None, "max": None},
            "moisture": {"enabled": False},
            "speed_range": {"enabled": True, "min": 0.5, "max": 30.0},
            "speed_change": {"enabled": False},
            "swath_partial": {"enabled": False},
            "short_pass": {"enabled": False},
            "pass_ends": {"enabled": False},
            "overlap": {"enabled": False},
            "boundary": {"enabled": False, "buffer_m": 0.0},
            "global_outlier": {"enabled": True, "method": "percentile",
                               "lower_pct": 0.5, "upper_pct": 99.5},
            "local_outlier": {"enabled": False},
        },
    },
    "minimal": {
        "label": "Minimal (obvious errors only)",
        "description": "Discards only the indefensible: nulls, invalid positions, duplicates.",
        "corrections": {"flow_delay_s": 0.0},
        "steps": {
            "null_value": {"enabled": True},
            "position": {"enabled": True, "max_jump_m": 50.0, "drop_duplicates": True},
            "value_range": {"enabled": False},
            "moisture": {"enabled": False},
            "speed_range": {"enabled": False},
            "speed_change": {"enabled": False},
            "swath_partial": {"enabled": False},
            "short_pass": {"enabled": False},
            "pass_ends": {"enabled": False},
            "overlap": {"enabled": False},
            "boundary": {"enabled": False},
            "global_outlier": {"enabled": False},
            "local_outlier": {"enabled": False},
        },
    },
}


def preset_for(operation: str) -> str:
    """Recommended preset for an operation type."""
    return operation if operation in PRESETS else "minimal"


@dataclass
class CleaningResult:
    """Complete output of a cleaning run."""

    clean: Dataset
    removed: Dataset
    report: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Corrections
# --------------------------------------------------------------------------

def apply_flow_delay(ds: Dataset, delay_s: float, value_column: str = sch.VALUE) -> str | None:
    """Shift the variable in time to compensate for transit inside the machine.

    A few seconds pass between the cut and the flow sensor. Without this
    correction, the measured value is attributed to where the machine is
    **now** rather than where the crop actually came from — which shifts the
    whole map several metres along the direction of travel.
    """
    if not delay_s or delay_s <= 0 or value_column not in ds.df.columns:
        return None

    dt = ds._time_delta_seconds()
    if dt is None or not np.isfinite(dt).any():
        return "Flow delay not applied: interval between records unavailable."

    median_dt = float(np.nanmedian(dt))
    if not np.isfinite(median_dt) or median_dt <= 0:
        return "Flow delay not applied: interval between records is inconsistent."

    shift = int(round(delay_s / median_dt))
    if shift <= 0:
        return None

    ds.df[value_column] = ds.df[value_column].shift(-shift)
    return (
        f"Flow delay of {delay_s:g} s applied "
        f"({shift} records at {median_dt:.2f} s each)."
    )


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------

def run(
    dataset: Dataset,
    config: dict[str, Any] | None = None,
    value_column: str = sch.VALUE,
    units: dict[str, Any] | None = None,
) -> CleaningResult:
    """Run the cleaning and return clean data, removed data and the report.

    Parameters
    ----------
    dataset:
        A dataset already imported and with derived fields computed.
    config:
        Dictionary in the presets' shape. When absent, the preset for the
        detected operation type is used.
    value_column:
        Column to treat as the main variable.
    units:
        The reader's unit set, which reaches the report's sentences only —
        what each filter says about its own settings, and the readings at
        the end. Every number in the report stays metric. ``None`` is the
        metric store.
    """
    if config is None:
        config = PRESETS[preset_for(dataset.meta.operation)]

    working = dataset.copy()
    working.sort_by_time()
    working.ensure_derived()

    before_stats = working.stats(value_column)
    before_values = pd.to_numeric(
        working.df.get(value_column), errors="coerce"
    ) if value_column in working.df.columns else pd.Series(dtype="float64")

    corrections: list[str] = []
    delay = float((config.get("corrections") or {}).get("flow_delay_s") or 0.0)
    message = apply_flow_delay(working, delay, value_column)
    if message:
        corrections.append(message)

    ctx = steps_mod.Context(
        working.df, value_column, units=units, operation=dataset.meta.operation)
    step_config = config.get("steps") or {}
    results: list[steps_mod.StepResult] = []

    for step_class in steps_mod.STEP_CLASSES:
        settings = dict(step_config.get(step_class.key) or {})
        enabled = settings.pop("enabled", False)
        if not enabled:
            continue
        step = step_class(**{k: v for k, v in settings.items() if v is not None or k in ("min", "max")})
        results.append(step.run(ctx))

    clean = working.subset(ctx.alive)
    removed = working.subset(~ctx.alive)
    if len(removed):
        removed.df["removal_reason"] = ctx.reason[~ctx.alive]

    clean.meta.notes = list(clean.meta.notes) + corrections
    clean.meta.name = f"{dataset.meta.name} (clean)"
    removed.meta.name = f"{dataset.meta.name} (removed)"

    # The clean copy says so in its own metadata, not only in the session's
    # report. Saved to a project file and reopened, or handed to the first
    # look on its own, it must still be recognizable as data the filters have
    # already been through — that is what lets the app stop suggesting a
    # second cleaning and offer the analysis instead.
    clean.meta.extra = {
        **(clean.meta.extra or {}),
        "cleaning": {
            "input": int(len(working)),
            "kept": int(len(clean)),
            "removed": int(len(removed)),
        },
    }

    report = build_report(
        dataset, clean, removed, results, corrections,
        before_stats, before_values, value_column,
    )
    return CleaningResult(clean=clean, removed=removed, report=report)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def _histogram(series: pd.Series, bins: int = 30) -> dict[str, list[float]]:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    if values.size < 2:
        return {"edges": [], "counts": []}
    counts, edges = np.histogram(values, bins=bins)
    return {"edges": [float(e) for e in edges], "counts": [int(c) for c in counts]}


def restate(
    report: dict[str, Any],
    units: dict[str, Any] | None = None,
    operation: str | None = None,
) -> dict[str, Any]:
    """The same cleaning report with its sentences in another unit set.

    Only the prose changes: what each filter says about its settings, from
    the settings and measurements the report already holds. Every count,
    percentage, statistic and histogram edge is left exactly as it was
    stored, and the report itself is not edited — the copy is fresh where
    it differs and shared where it does not.

    The readings at the end (``findings``) are percentages and ratios of
    the file to itself, which read the same in every unit set, so they are
    carried over untouched.
    """
    if not report:
        return report
    say = Phrase(units)
    operation = operation or report.get("operation")
    out = dict(report)
    out["steps"] = [
        {**step, "detail": steps_mod.detail_for(step, say, operation)}
        for step in report.get("steps") or []
    ]
    return out


def _assess(removed_pct: float, before: dict, after: dict) -> list[dict[str, str]]:
    """Turn the report's numbers into readings you can act on.

    Saying how many points came out is not enough; what matters is whether the
    cleaning stayed reasonable and what changed in the distribution.
    """
    findings: list[dict[str, str]] = []

    if removed_pct > 45:
        findings.append({
            "level": "alert",
            "text": (
                f"{removed_pct:.1f}% of the records were discarded. Past roughly 40% "
                "the map starts reflecting the filters more than the crop — revisit "
                "the most aggressive parameters before using the result."
            ),
        })
    elif removed_pct > 25:
        findings.append({
            "level": "warning",
            "text": (
                f"{removed_pct:.1f}% of the records discarded. That is defensible on "
                "harvest data with a lot of turning, but check which filters dominated."
            ),
        })
    elif removed_pct < 2:
        findings.append({
            "level": "warning",
            "text": (
                f"Only {removed_pct:.1f}% was discarded. Raw monitor data is rarely "
                "this clean — check that the filters were actually enabled."
            ),
        })
    else:
        findings.append({
            "level": "ok",
            "text": f"{removed_pct:.1f}% of the records discarded — within the usual range.",
        })

    cv_before = before.get("cv")
    cv_after = after.get("cv")
    if cv_before and cv_after:
        delta = cv_after - cv_before
        if delta < -3:
            findings.append({
                "level": "ok",
                "text": (
                    f"Coefficient of variation fell from {cv_before:.1f}% to "
                    f"{cv_after:.1f}% — the noise came out and what is left is likely "
                    "the field's own variation."
                ),
            })
        elif delta > 2:
            findings.append({
                "level": "alert",
                "text": (
                    f"Coefficient of variation rose from {cv_before:.1f}% to "
                    f"{cv_after:.1f}%. A cleaning that increases spread usually means "
                    "a filter cutting from only one side of the distribution."
                ),
            })

    mean_before, mean_after = before.get("mean"), after.get("mean")
    if mean_before and mean_after:
        shift = (mean_after - mean_before) / mean_before * 100.0
        if abs(shift) > 8:
            findings.append({
                "level": "alert",
                "text": (
                    f"The mean moved {shift:+.1f}% with the cleaning. A shift that "
                    "size changes the agronomic conclusion: confirm the removed "
                    "points were errors and not genuinely low-yielding ground."
                ),
            })
        else:
            findings.append({
                "level": "ok",
                "text": f"The mean moved {shift:+.1f}% — the cleaning kept the field's level.",
            })

    return findings


def build_report(
    original: Dataset,
    clean: Dataset,
    removed: Dataset,
    results: list[steps_mod.StepResult],
    corrections: list[str],
    before_stats: dict,
    before_values: pd.Series,
    value_column: str,
) -> dict[str, Any]:
    """Assemble the comparative cleaning report.

    Every number in it is metric; the sentences in ``steps`` and
    ``findings`` were written by the step and by :func:`_assess` in
    whatever unit set the run was given, and :func:`restate` writes them
    again for another reader.
    """
    total = len(original)
    kept = len(clean)
    removed_count = len(removed)
    removed_pct = (removed_count / total * 100.0) if total else 0.0
    after_stats = clean.stats(value_column)

    by_reason: list[dict[str, Any]] = []
    if removed_count and "removal_reason" in removed.df.columns:
        counts = removed.df["removal_reason"].value_counts()
        by_reason = [
            {
                "reason": str(reason),
                "records": int(count),
                "pct_of_total": round(count / total * 100.0, 2) if total else 0.0,
            }
            for reason, count in counts.items()
        ]

    return {
        # What the main variable is — a yield on a harvest, an input rate on
        # anything else — travels with the report so that the sentences can
        # be written again later without the dataset: a saved project's
        # cleaning report is read long after the session that made it.
        "operation": original.meta.operation,
        "totals": {
            "input": total,
            "kept": kept,
            "removed": removed_count,
            "removed_pct": round(removed_pct, 2),
            "area_ha_before": round(original.area_ha(), 2),
            "area_ha_after": round(clean.area_ha(), 2),
        },
        "value_column": value_column,
        "corrections": corrections,
        "steps": [r.to_dict() for r in results],
        "by_reason": by_reason,
        "statistics": {"before": before_stats, "after": after_stats},
        "histogram": {
            "before": _histogram(before_values),
            "after": _histogram(clean.df.get(value_column, pd.Series(dtype="float64"))),
        },
        "findings": _assess(removed_pct, before_stats, after_stats),
    }
