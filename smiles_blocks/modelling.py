from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit


class FittingModel(ABC):
    """Base class for all fitting models.

    Models have two (alpha, beta) or three (alpha, beta, gamma) parameters;
    ``n_params``, ``initial_guess`` and ``bounds`` let callers fit any of them
    the same way.
    """

    n_params: int = 2

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the model."""
        pass

    @property
    @abstractmethod
    def formula(self) -> str:
        """LaTeX formula representation."""
        pass

    @abstractmethod
    def func(self, x: np.ndarray, *params: float) -> np.ndarray:
        """The actual fitting function."""
        pass

    @property
    def bounds(self) -> Tuple:
        """Parameter bounds passed to ``curve_fit`` (unbounded by default)."""
        return (-np.inf, np.inf)

    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> list:
        """Starting parameters for ``curve_fit``: alpha at the observed maximum."""
        return [float(np.max(y)), 1.0]

    def fit(
        self, x: np.ndarray, y: np.ndarray, p0: Optional[Tuple[float, ...]] = None, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit the model to data.

        ``p0`` defaults to :meth:`initial_guess` and ``bounds`` to :attr:`bounds`.

        Returns:
            popt: Optimal parameters
            pcov: Covariance matrix
        """
        if p0 is None:
            p0 = self.initial_guess(x, y)
        kwargs.setdefault("bounds", self.bounds)
        return curve_fit(self.func, x, y, p0=p0, **kwargs)

    def x_at_fraction(self, q: float, *params: float) -> float:
        """Number of draws at which the curve reaches ``q`` times its asymptote alpha.

        Only defined for saturating models (closed form in the subclasses).
        """
        raise NotImplementedError(f"{self.name} has no finite asymptote")

    def predict(self, x: np.ndarray, *params: float, **named_params: float) -> np.ndarray:
        """Generate predictions given parameters."""
        return self.func(x, *params, **named_params)

    def r2_score(self, x: np.ndarray, y: np.ndarray, *params: float, **named_params: float) -> float:
        """
        Calculate R² (coefficient of determination) for the model.

        Args:
            x: Input data
            y: Observed values
            *params / **named_params: Model parameters

        Returns:
            R² score
        """
        y_pred = self.predict(x, *params, **named_params)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)


def _x_at_half_max(x: np.ndarray, y: np.ndarray) -> float:
    """Smallest x at which y reaches half its maximum (scale guess for 3-parameter models)."""
    order = np.argsort(x)
    xs, ys = np.asarray(x)[order], np.asarray(y)[order]
    return float(max(xs[np.argmax(ys >= ys.max() / 2)], 1.0))


class SquareRootModel(FittingModel):
    name = "Square Root"
    formula = r"$y = \alpha \sqrt{\beta x}$"

    def func(self, x, alpha, beta):
        return alpha * np.sqrt(beta * x)


class LogarithmicModel(FittingModel):
    name = "Logarithmic"
    formula = r"$y = \alpha \log(\beta x)$"

    def func(self, x, alpha, beta):
        return alpha * np.log(beta * x)


class InverseModel(FittingModel):
    name = "Inverse"
    formula = r"$y = \frac{\alpha}{\beta + \frac{1}{x}}$"

    def func(self, x, alpha, beta):
        return alpha / (beta + 1 / x)


class Inverse2Model(FittingModel):
    name = "Inverse 2"
    formula = r"$y = \frac{\alpha}{1 + \frac{\beta}{x}}$"

    def func(self, x, alpha, beta):
        return alpha / (1 + beta / x)


class ExponentialModel(FittingModel):
    name = "Exponential"
    formula = r"$y = \alpha (1 - e^{-\beta x})$"

    def func(self, x, alpha, beta):
        return alpha * (1 - np.exp(-beta * x))

    def x_at_fraction(self, q, alpha, beta):
        return -np.log1p(-q) / beta


class WeibullModel(FittingModel):
    """Stretched exponential; reduces to the exponential model when gamma = 1."""

    name = "Weibull"
    formula = r"$y = \alpha \left(1 - e^{-(\beta x)^{\gamma}}\right)$"
    n_params = 3

    def func(self, x, alpha, beta, gamma):
        return alpha * (1 - np.exp(-((beta * x) ** gamma)))

    def x_at_fraction(self, q, alpha, beta, gamma):
        return (-np.log1p(-q)) ** (1 / gamma) / beta

    @property
    def bounds(self):
        return ([0, 1e-12, 0.05], [np.inf, np.inf, 5])

    def initial_guess(self, x, y):
        return [float(np.max(y)), 1.0 / _x_at_half_max(x, y), 1.0]


class HillModel(FittingModel):
    """Log-logistic (Hill) curve: a power law in x that levels off at alpha."""

    name = "Hill"
    formula = r"$y = \frac{\alpha}{1 + (\beta / x)^{\gamma}}$"
    n_params = 3

    def func(self, x, alpha, beta, gamma):
        return alpha / (1 + (beta / x) ** gamma)

    def x_at_fraction(self, q, alpha, beta, gamma):
        return beta * (q / (1 - q)) ** (1 / gamma)

    @property
    def bounds(self):
        return ([0, 1e-6, 0.05], [np.inf, np.inf, 20])

    def initial_guess(self, x, y):
        return [float(np.max(y)), _x_at_half_max(x, y), 1.0]


class LomaxModel(FittingModel):
    """Lomax (Pareto type II) saturation curve.

    Equals an average of exponential curves whose rates are gamma-distributed, and
    reduces to the inverse model alpha * beta x / (1 + beta x) when gamma = 1.
    """

    name = "Lomax"
    formula = r"$y = \alpha \left(1 - (1 + \beta x)^{-\gamma}\right)$"
    n_params = 3

    def func(self, x, alpha, beta, gamma):
        return alpha * (1 - (1 + beta * x) ** (-gamma))

    def x_at_fraction(self, q, alpha, beta, gamma):
        return np.expm1(-np.log1p(-q) / gamma) / beta

    @property
    def bounds(self):
        return ([0, 1e-12, 0.01], [np.inf, np.inf, 50])

    def initial_guess(self, x, y):
        return [float(np.max(y)), 1.0 / _x_at_half_max(x, y), 1.0]


# Models retained for the manuscript after the hold-out analysis (reviewer response):
# the exponential plus three 3-parameter saturating curves. sqrt and log never level
# off and cannot predict the plateau; inverse/inverse2 are special cases of Hill and
# Lomax (gamma = 1).
FINAL_MODELS = ("exp", "weibull", "hill", "lomax")


@dataclass
class ModelsRegistry:
    """
    Registry for easy access to fitting models.

    Attributes
    ----------
    models : dict[str, FittingModel]
        Dictionary mapping model identifiers to their respective FittingModel instances.
        Available models:
        - 'sqrt': Square root model
        - 'log': Logarithmic model
        - 'inverse': Inverse model
        - 'inverse2': Alternative inverse model
        - 'exp': Exponential model
        - 'weibull': Stretched exponential (3 parameters)
        - 'hill': Log-logistic / Hill model (3 parameters)
        - 'lomax': Lomax / Pareto II model (3 parameters)
    """

    models: dict[str, FittingModel] = field(
        default_factory=lambda: {
            "sqrt": SquareRootModel(),
            "log": LogarithmicModel(),
            "inverse": InverseModel(),
            "inverse2": Inverse2Model(),
            "exp": ExponentialModel(),
            "weibull": WeibullModel(),
            "hill": HillModel(),
            "lomax": LomaxModel(),
        }
    )
