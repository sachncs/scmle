"""Forecast families over panel data (many series, entity and time).

`panel.Panel` holds the series; `families` fits them: `Seasonal` (the
seasonal-naive baseline), `Smoothing` (simple exponential smoothing) and
`Boosted` (direct multi-step gradient boosting on lag features).
`scmle.evaluation.forecasting` compares them by rolling-origin backtest.
"""

from scmle.models.forecast import families
from scmle.models.forecast import panel
from scmle.models.forecast.families import Boosted
from scmle.models.forecast.families import Forecaster
from scmle.models.forecast.families import Seasonal
from scmle.models.forecast.families import Smoothing
from scmle.models.forecast.panel import Panel

FAMILIES: dict[str, type[Forecaster]] = {
    family.name: family for family in (Seasonal, Smoothing, Boosted)
}

__all__ = [
    "FAMILIES",
    "Boosted",
    "Forecaster",
    "Panel",
    "Seasonal",
    "Smoothing",
    "families",
    "panel",
]
