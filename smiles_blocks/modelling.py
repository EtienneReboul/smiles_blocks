from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit


class FittingModel(ABC):
    """Base class for all fitting models."""

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
    def func(self, x: np.ndarray, alpha: float, beta: float) -> np.ndarray:
        """The actual fitting function."""
        pass

    def fit(
        self, x: np.ndarray, y: np.ndarray, p0: Optional[Tuple[float, float]] = None, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit the model to data.

        Returns:
            popt: Optimal parameters [alpha, beta]
            pcov: Covariance matrix
        """
        return curve_fit(self.func, x, y, p0=p0, **kwargs)

    def predict(self, x: np.ndarray, alpha: float, beta: float) -> np.ndarray:
        """Generate predictions given parameters."""
        return self.func(x, alpha, beta)

    def r2_score(self, x: np.ndarray, y: np.ndarray, alpha: float, beta: float) -> float:
        """
        Calculate R² (coefficient of determination) for the model.

        Args:
            x: Input data
            y: Observed values
            alpha: First parameter
            beta: Second parameter

        Returns:
            R² score
        """
        y_pred = self.predict(x, alpha, beta)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)


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
    """

    models: dict[str, FittingModel] = field(
        default_factory=lambda: {
            "sqrt": SquareRootModel(),
            "log": LogarithmicModel(),
            "inverse": InverseModel(),
            "inverse2": Inverse2Model(),
            "exp": ExponentialModel(),
        }
    )
