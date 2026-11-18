import numpy as np
import pytest

from smiles_blocks.modelling import (
    FINAL_MODELS,
    ExponentialModel,
    HillModel,
    Inverse2Model,
    InverseModel,
    LogarithmicModel,
    LomaxModel,
    ModelsRegistry,
    SquareRootModel,
    WeibullModel,
)


class TestSquareRootModel:
    def test_name(self):
        model = SquareRootModel()
        assert model.name == "Square Root"

    def test_formula(self):
        model = SquareRootModel()
        assert model.formula == r"$y = \alpha \sqrt{\beta x}$"

    def test_func(self):
        model = SquareRootModel()
        x = np.array([1, 4, 9])
        result = model.func(x, alpha=2, beta=1)
        expected = np.array([2, 4, 6])
        np.testing.assert_array_almost_equal(result, expected)

    def test_predict(self):
        model = SquareRootModel()
        x = np.array([1, 4, 9])
        result = model.predict(x, alpha=2, beta=1)
        expected = np.array([2, 4, 6])
        np.testing.assert_array_almost_equal(result, expected)

    def test_r2_score_perfect_fit(self):
        model = SquareRootModel()
        x = np.array([1, 4, 9, 16])
        y = 2 * np.sqrt(x)
        r2 = model.r2_score(x, y, alpha=2, beta=1)
        assert r2 == pytest.approx(1.0, rel=1e-10)


class TestLogarithmicModel:
    def test_name(self):
        model = LogarithmicModel()
        assert model.name == "Logarithmic"

    def test_formula(self):
        model = LogarithmicModel()
        assert model.formula == r"$y = \alpha \log(\beta x)$"

    def test_func(self):
        model = LogarithmicModel()
        x = np.array([1, np.e, np.e**2])
        result = model.func(x, alpha=1, beta=1)
        expected = np.array([0, 1, 2])
        np.testing.assert_array_almost_equal(result, expected)

    def test_fit(self):
        model = LogarithmicModel()
        x = np.array([1, 2, 3, 4, 5])
        y = 2 * np.log(x)
        popt, pcov = model.fit(x, y, p0=(2, 1))
        assert len(popt) == 2
        assert popt[0] == pytest.approx(2, rel=1e-2)


class TestInverseModel:
    def test_name(self):
        model = InverseModel()
        assert model.name == "Inverse"

    def test_formula(self):
        model = InverseModel()
        assert model.formula == r"$y = \frac{\alpha}{\beta + \frac{1}{x}}$"

    def test_func(self):
        model = InverseModel()
        x = np.array([1, 2, 4])
        result = model.func(x, alpha=10, beta=0)
        expected = np.array([10, 20, 40])
        np.testing.assert_array_almost_equal(result, expected)

    def test_fit(self):
        model = InverseModel()
        x = np.array([1, 2, 4, 8])
        y = 10 / (1 / x)
        popt, pcov = model.fit(x, y, p0=(10, 0))
        assert len(popt) == 2


class TestInverse2Model:
    def test_name(self):
        model = Inverse2Model()
        assert model.name == "Inverse 2"

    def test_formula(self):
        model = Inverse2Model()
        assert model.formula == r"$y = \frac{\alpha}{1 + \frac{\beta}{x}}$"

    def test_func(self):
        model = Inverse2Model()
        x = np.array([1, 2, 4])
        result = model.func(x, alpha=10, beta=1)
        expected = np.array([5, 10 / 1.5, 8])
        np.testing.assert_array_almost_equal(result, expected)


class TestExponentialModel:
    def test_name(self):
        model = ExponentialModel()
        assert model.name == "Exponential"

    def test_formula(self):
        model = ExponentialModel()
        assert model.formula == r"$y = \alpha (1 - e^{-\beta x})$"

    def test_func(self):
        model = ExponentialModel()
        x = np.array([0, 1, 10])
        result = model.func(x, alpha=5, beta=1)
        assert result[0] == 0
        assert result[1] == pytest.approx(5 * (1 - np.exp(-1)))
        assert result[2] == pytest.approx(5, rel=1e-3)

    def test_fit(self):
        model = ExponentialModel()
        x = np.linspace(0, 10, 50)
        y = 5 * (1 - np.exp(-0.5 * x)) + np.random.normal(0, 0.01, 50)
        popt, pcov = model.fit(x, y, p0=(5, 0.5))
        assert len(popt) == 2
        assert popt[0] == pytest.approx(5, rel=0.1)
        assert popt[1] == pytest.approx(0.5, rel=0.2)


class TestWeibullModel:
    def test_reduces_to_exponential_when_gamma_is_one(self):
        x = np.array([1.0, 10.0, 100.0, 1000.0])
        np.testing.assert_allclose(
            WeibullModel().func(x, 50, 0.01, 1.0), ExponentialModel().func(x, 50, 0.01)
        )

    def test_fit_recovers_parameters(self):
        model = WeibullModel()
        x = np.logspace(0, 5, 60)
        y = model.func(x, 5000, 1e-3, 0.9)
        popt, _ = model.fit(x, y)
        assert len(popt) == 3
        np.testing.assert_allclose(popt, [5000, 1e-3, 0.9], rtol=1e-3)


class TestHillModel:
    def test_half_max_at_beta(self):
        assert HillModel().func(np.array([200.0]), 10, 200, 1.3)[0] == pytest.approx(5)

    def test_fit_recovers_parameters(self):
        model = HillModel()
        x = np.logspace(0, 5, 60)
        y = model.func(x, 5000, 800, 1.25)
        popt, _ = model.fit(x, y)
        assert len(popt) == 3
        np.testing.assert_allclose(popt, [5000, 800, 1.25], rtol=1e-3)


class TestLomaxModel:
    def test_reduces_to_inverse2_when_gamma_is_one(self):
        x = np.array([1.0, 10.0, 100.0, 1000.0])
        beta = 0.01
        np.testing.assert_allclose(
            LomaxModel().func(x, 50, beta, 1.0), Inverse2Model().func(x, 50, 1 / beta)
        )

    def test_fit_recovers_parameters(self):
        model = LomaxModel()
        x = np.logspace(0, 5, 60)
        y = model.func(x, 5000, 5e-4, 4.0)
        popt, _ = model.fit(x, y)
        assert len(popt) == 3
        np.testing.assert_allclose(popt, [5000, 5e-4, 4.0], rtol=1e-3)


class TestModelsRegistry:
    def test_final_models_are_registered(self):
        registry = ModelsRegistry()
        assert all(name in registry.models for name in FINAL_MODELS)

    def test_initial_guess_matches_parameter_count(self):
        x = np.logspace(0, 4, 20)
        y = 100 * (1 - np.exp(-x / 300))
        for model in ModelsRegistry().models.values():
            assert len(model.initial_guess(x, y)) == model.n_params

    def test_registry_contains_all_models(self):
        registry = ModelsRegistry()
        assert "sqrt" in registry.models
        assert "log" in registry.models
        assert "inverse" in registry.models
        assert "inverse2" in registry.models
        assert "exp" in registry.models

    def test_registry_models_are_instances(self):
        registry = ModelsRegistry()
        assert isinstance(registry.models["sqrt"], SquareRootModel)
        assert isinstance(registry.models["log"], LogarithmicModel)
        assert isinstance(registry.models["inverse"], InverseModel)
        assert isinstance(registry.models["inverse2"], Inverse2Model)
        assert isinstance(registry.models["exp"], ExponentialModel)


class TestR2Score:
    def test_r2_score_poor_fit(self):
        model = SquareRootModel()
        x = np.array([1, 4, 9, 16])
        y = np.array([10, 20, 30, 40])  # Linear, not sqrt
        r2 = model.r2_score(x, y, alpha=2, beta=1)
        assert r2 < 0.95

    def test_r2_score_negative(self):
        model = SquareRootModel()
        x = np.array([1, 4, 9, 16])
        y = np.array([100, 200, 300, 400])
        r2 = model.r2_score(x, y, alpha=1, beta=1)
        assert r2 < 0
