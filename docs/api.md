# API Reference

The `boostlss_py` library provides a scikit-learn-like API for distributional regression.

## Core Classes

### `BoostLssModel`

The primary class for training and predicting models.

```python
BoostLssModel(family: PyFamily, mstop: int = 100, step_length: float = 0.1, algorithm: str = "cyclic")
```

**Parameters:**

- `family` (`PyFamily`): The distribution family.
- `mstop` (`int`): The maximum number of boosting iterations.
- `step_length` (`float`): The learning rate (shrinkage parameter).
- `algorithm` (`str`): The boosting algorithm (`"cyclic"`, `"noncyclic"`, `"noncyclic_outer"`).

**Methods:**

- `add_learner(param: str, learner)`: Add a base learner for a specific distribution parameter.
- `fit(X, y)`: Train the model on the design matrix `X` and response vector `y`.
- `predict(X, param: str) -> np.ndarray`: Predict the value of a specific parameter for data `X`.
- `cvrisk(folds: int = 10)`: Run k-fold cross-validation to find the optimal `mstop`.
- `stabsel(b: int = 100, pfer: float = None, pi_thr: float = None)`: Run stability selection to find informative base learners.
- `feature_importance() -> list[float]`: Get the empirical risk reduction (importance) of each base learner.

---

### `PyFamily`

Represents a statistical distribution family.

```python
PyFamily(name: str)
```

**Supported Families:**

- `"GaussianLSS"`: Normal distribution (models `mu`, `sigma`).
- `"StudentTLSS"`: Student's t-distribution (models `mu`, `sigma`, `df`).
- `"GammaLSS"`: Gamma distribution (models `mu`, `sigma`).
- `"BinomialLSS"`: Binomial distribution (models `mu`).
- `"BetaLSS"`: Beta distribution (models `mu`, `phi`).
- `"WeibullLSS"`: Weibull distribution (models `mu`, `sigma`).
- `"LogNormalLSS"`: Log-Normal distribution (models `mu`, `sigma`).
- `"ZIPLss"`: Zero-Inflated Poisson distribution (models `mu`, `sigma`).
- `"GEVLss"`: Generalized Extreme Value distribution (models `mu`, `sigma`, `xi`).

---

## Base Learners

Base learners define how features are modeled for each parameter.

### `PyLinearLearner(feature_idx: int, intercept: bool = True)`

A simple linear effect for a single feature.

### `PyPSplineLearner(feature_idx: int, df: float = 4.0, degree: int = 3, differences: int = 2, n_knots: int = 20)`

A univariate penalized spline for modeling non-linear effects.

### `PyBivariatePSplineLearner(feature1_idx: int, feature2_idx: int, ...)`

A bivariate tensor-product penalized spline for modeling spatial or interaction effects.

### `PyTreeLearner(max_depth: int = 3, min_samples_split: int = 2, ...)`

A decision tree base learner.

### `PyStumpLearner(feature_idx: int)`

A decision stump (tree with depth 1) for a specific feature.

### `PyRandomEffectsLearner(feature_idx: int, df: float = 4.0)`

A random effects base learner for categorical variables.
