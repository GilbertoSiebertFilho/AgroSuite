"""Rate response functions and economic optimization.

The point of a strip trial is not to find which rate gave the highest yield —
it is to find which rate gives the highest **profit**. The two answers rarely
coincide: the yield curve keeps climbing long after the increment stops
paying for the input.

The economic optimum rate is where the derivative of the yield curve equals
the price ratio::

    dY/dR = input_cost / crop_price

Below it, each extra unit of input returns more than it costs; above it, less.
Every model here exposes that derivative analytically, so the optimum can be
solved without a numeric search wherever possible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass
class ResponseFit:
    """Fit of a response model to a set of (rate, yield) pairs."""

    model: str
    label: str
    params: dict[str, float]
    r2: float
    rmse: float
    n: int
    predict: Callable[[np.ndarray], np.ndarray] = field(repr=False, default=None)
    derivative: Callable[[np.ndarray], np.ndarray] = field(repr=False, default=None)
    plateau_rate: float | None = None
    converged: bool = True
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "label": self.label,
            "params": {k: round(float(v), 6) for k, v in self.params.items()},
            "r2": round(float(self.r2), 4),
            "rmse": round(float(self.rmse), 2),
            "n": int(self.n),
            "plateau_rate": round(float(self.plateau_rate), 2) if self.plateau_rate else None,
            "converged": self.converged,
            "message": self.message,
        }


def _goodness(y: np.ndarray, y_hat: np.ndarray) -> tuple[float, float]:
    residual = y - y_hat
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = float(np.sqrt(ss_res / len(y))) if len(y) else float("nan")
    return r2, rmse


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

def fit_quadratic(rates: np.ndarray, yields: np.ndarray) -> ResponseFit:
    """Quadratic response: ``Y = a + b*R + c*R^2``.

    The reference model in the fertility literature. It needs only three
    distinct rates and has a closed-form optimum, but it forces the curve to
    be symmetric, which overstates the fall-off at high rates.
    """
    coef = np.polyfit(rates, yields, 2)
    c, b, a = float(coef[0]), float(coef[1]), float(coef[2])
    predict = lambda r: a + b * np.asarray(r, dtype="float64") + c * np.asarray(r, dtype="float64") ** 2
    derivative = lambda r: b + 2 * c * np.asarray(r, dtype="float64")
    r2, rmse = _goodness(yields, predict(rates))
    plateau = -b / (2 * c) if c < 0 else None
    return ResponseFit(
        model="quadratic",
        label="Quadratic",
        params={"a": a, "b": b, "c": c},
        r2=r2, rmse=rmse, n=len(rates),
        predict=predict, derivative=derivative,
        plateau_rate=plateau,
        message="" if c < 0 else "Positive curvature: no agronomic maximum within the tested range.",
    )


def fit_quadratic_plateau(rates: np.ndarray, yields: np.ndarray) -> ResponseFit:
    """Quadratic plateau: climbs to a critical rate and levels off.

    Closer to the biology than the pure quadratic, because it does not force
    yield to fall after the maximum.
    """
    from scipy.optimize import curve_fit

    def model(r, a, b, c):
        r = np.asarray(r, dtype="float64")
        critical = -b / (2 * c) if c < 0 else np.inf
        r_eff = np.minimum(r, critical)
        return a + b * r_eff + c * r_eff**2

    start = np.polyfit(rates, yields, 2)
    p0 = [float(start[2]), float(start[1]), float(min(start[0], -1e-6))]
    try:
        popt, _ = curve_fit(model, rates, yields, p0=p0, maxfev=20_000)
        converged, message = True, ""
    except Exception as exc:
        popt, converged, message = p0, False, f"Fit did not converge ({exc}); initial values used."

    a, b, c = (float(v) for v in popt)
    critical = -b / (2 * c) if c < 0 else float("inf")

    def predict(r):
        r = np.asarray(r, dtype="float64")
        return model(r, a, b, c)

    def derivative(r):
        r = np.asarray(r, dtype="float64")
        return np.where(r < critical, b + 2 * c * r, 0.0)

    r2, rmse = _goodness(yields, predict(rates))
    return ResponseFit(
        model="quadratic_plateau",
        label="Quadratic plateau",
        params={"a": a, "b": b, "c": c},
        r2=r2, rmse=rmse, n=len(rates),
        predict=predict, derivative=derivative,
        plateau_rate=critical if np.isfinite(critical) else None,
        converged=converged, message=message,
    )


def fit_linear_plateau(rates: np.ndarray, yields: np.ndarray) -> ResponseFit:
    """Linear plateau: constant response up to a critical rate, then nothing.

    Useful when the input is clearly limiting up to a point and irrelevant
    after it — common when correcting a deficiency.
    """
    from scipy.optimize import curve_fit

    def model(r, a, b, critical):
        r = np.asarray(r, dtype="float64")
        return a + b * np.minimum(r, critical)

    span = float(np.max(rates) - np.min(rates)) or 1.0
    p0 = [float(np.min(yields)), span and float((np.max(yields) - np.min(yields)) / span),
          float(np.median(rates))]
    try:
        popt, _ = curve_fit(model, rates, yields, p0=p0, maxfev=20_000)
        converged, message = True, ""
    except Exception as exc:
        popt, converged, message = p0, False, f"Fit did not converge ({exc}); initial values used."

    a, b, critical = (float(v) for v in popt)
    predict = lambda r: model(r, a, b, critical)
    derivative = lambda r: np.where(np.asarray(r, dtype="float64") < critical, b, 0.0)
    r2, rmse = _goodness(yields, predict(rates))
    return ResponseFit(
        model="linear_plateau",
        label="Linear plateau",
        params={"a": a, "b": b, "critical": critical},
        r2=r2, rmse=rmse, n=len(rates),
        predict=predict, derivative=derivative,
        plateau_rate=critical,
        converged=converged, message=message,
    )


def fit_mitscherlich(rates: np.ndarray, yields: np.ndarray) -> ResponseFit:
    """Mitscherlich: ``Y = A * (1 - exp(-c * (R + b)))``.

    A smooth asymptote with no fall-off at high rates. It is the most
    conservative model for extrapolating above the highest rate tested.
    """
    from scipy.optimize import curve_fit

    def model(r, A, b, c):
        return A * (1.0 - np.exp(-c * (np.asarray(r, dtype="float64") + b)))

    p0 = [float(np.max(yields) * 1.05), 10.0, 0.01]
    try:
        popt, _ = curve_fit(
            model, rates, yields, p0=p0, maxfev=40_000,
            bounds=([0, -np.inf, 1e-6], [np.inf, np.inf, 1.0]),
        )
        converged, message = True, ""
    except Exception as exc:
        popt, converged, message = p0, False, f"Fit did not converge ({exc}); initial values used."

    A, b, c = (float(v) for v in popt)
    predict = lambda r: model(r, A, b, c)
    derivative = lambda r: A * c * np.exp(-c * (np.asarray(r, dtype="float64") + b))
    r2, rmse = _goodness(yields, predict(rates))
    return ResponseFit(
        model="mitscherlich",
        label="Mitscherlich",
        params={"A": A, "b": b, "c": c},
        r2=r2, rmse=rmse, n=len(rates),
        predict=predict, derivative=derivative,
        converged=converged, message=message,
    )


MODELS: dict[str, Callable[[np.ndarray, np.ndarray], ResponseFit]] = {
    "quadratic": fit_quadratic,
    "quadratic_plateau": fit_quadratic_plateau,
    "linear_plateau": fit_linear_plateau,
    "mitscherlich": fit_mitscherlich,
}

MODEL_LABELS = {
    "quadratic": "Quadratic",
    "quadratic_plateau": "Quadratic plateau",
    "linear_plateau": "Linear plateau",
    "mitscherlich": "Mitscherlich",
}


def fit_best(rates: np.ndarray, yields: np.ndarray, candidates=None) -> tuple[ResponseFit, list[ResponseFit]]:
    """Fit the candidate models and return the best by R-squared, plus the rest.

    Models with fewer distinct rates than parameters are dropped: a fit that
    passes exactly through the points tells you nothing.
    """
    rates = np.asarray(rates, dtype="float64")
    yields = np.asarray(yields, dtype="float64")
    mask = np.isfinite(rates) & np.isfinite(yields)
    rates, yields = rates[mask], yields[mask]

    distinct = len(np.unique(rates))
    if distinct < 3:
        raise ValueError(
            f"At least 3 distinct rates are needed to fit a response curve; "
            f"this dataset has {distinct}."
        )

    names = candidates or list(MODELS)
    if distinct < 4:
        names = [n for n in names if n == "quadratic"] or ["quadratic"]

    fits: list[ResponseFit] = []
    for name in names:
        try:
            fits.append(MODELS[name](rates, yields))
        except Exception:
            continue
    if not fits:
        raise ValueError("No response model could be fitted to this data.")

    fits.sort(key=lambda f: (-f.r2, f.rmse))
    return fits[0], fits


# --------------------------------------------------------------------------
# Economics
# --------------------------------------------------------------------------

def optimum_rate(
    fit: ResponseFit,
    crop_price: float,
    input_cost: float,
    rate_min: float = 0.0,
    rate_max: float = 400.0,
) -> dict[str, float]:
    """Economic optimum rate, and what it returns.

    Parameters
    ----------
    crop_price:
        Price of the harvested crop, per unit of the yield variable (for
        example $/kg when yield is in kg/ha).
    input_cost:
        Cost of the input, per unit of rate (for example $/kg of N).
    rate_min, rate_max:
        Range the search is allowed over. Outside the range actually tested in
        the trial the curve is extrapolation, and the result says so.
    """
    if crop_price <= 0:
        raise ValueError("The crop price must be positive.")

    price_ratio = input_cost / crop_price  # yield units per unit of rate
    grid = np.linspace(rate_min, rate_max, 2001)
    profit = fit.predict(grid) * crop_price - grid * input_cost
    best_index = int(np.argmax(profit))
    best_rate = float(grid[best_index])

    # Refine on the marginal condition, where the derivative crosses the price ratio.
    derivative = fit.derivative(grid)
    crossing = np.flatnonzero(np.diff(np.sign(derivative - price_ratio)) < 0)
    if crossing.size:
        candidate = float(grid[crossing[0] + 1])
        if rate_min <= candidate <= rate_max:
            candidate_profit = float(fit.predict(np.array([candidate]))[0] * crop_price
                                     - candidate * input_cost)
            if candidate_profit >= profit[best_index] - 1e-6:
                best_rate = candidate

    yield_at_best = float(fit.predict(np.array([best_rate]))[0])
    agronomic = fit.plateau_rate
    return {
        "optimum_rate": round(best_rate, 2),
        "yield_at_optimum": round(yield_at_best, 1),
        "profit_at_optimum": round(yield_at_best * crop_price - best_rate * input_cost, 2),
        "price_ratio": round(price_ratio, 4),
        "agronomic_maximum": round(float(agronomic), 2) if agronomic and np.isfinite(agronomic) else None,
        "at_range_limit": bool(best_rate >= rate_max - 1e-6 or best_rate <= rate_min + 1e-6),
    }


def profit_curve(
    fit: ResponseFit,
    crop_price: float,
    input_cost: float,
    rate_min: float = 0.0,
    rate_max: float = 400.0,
    points: int = 60,
) -> list[dict[str, float]]:
    """Sample the yield and profit curves for display."""
    grid = np.linspace(rate_min, rate_max, points)
    predicted = fit.predict(grid)
    profit = predicted * crop_price - grid * input_cost
    return [
        {
            "rate": round(float(r), 2),
            "yield": round(float(y), 1),
            "profit": round(float(p), 2),
        }
        for r, y, p in zip(grid, predicted, profit)
    ]
