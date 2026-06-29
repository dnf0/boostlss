# API Reference

The `boostlss_py` library provides a scikit-learn-like API for distributional regression, built on top of a highly optimized Rust engine.

## Core Classes

### `BoostLssModel`

The primary class for initializing, training, predicting, and evaluating distributional models.

```python
BoostLssModel(family: PyFamily, mstop: int = 100, step_length: float = 0.1, algorithm: str = "cyclic")
```

**Constructor Parameters:**

- `family` (`PyFamily`): The distributional family to fit. This dictates the statistical assumptions of the response and which parameters (e.g., `mu`, `sigma`) need to be modeled.
- `mstop` (`int`, default=100): The maximum number of boosting iterations.
- `step_length` (`float`, default=0.1): The learning rate or shrinkage parameter. Smaller values require higher `mstop` but generally lead to better generalization.
- `algorithm` (`str`, default="cyclic"): The internal optimization strategy.
  - `"cyclic"`: Updates parameters iteratively in a fixed cycle.
  - `"noncyclic"`: Greedily selects and updates the parameter yielding the best loss improvement.
  - `"noncyclic_outer"`: Updates all parameters simultaneously and chooses the best overall improvement.

**Methods:**

- `add_learner(param: str, learner) -> None`
  Adds a base learner to model a specific parameter of the chosen distribution family.

  - `param` (`str`): The name of the parameter to link the learner to (e.g., `"mu"`, `"sigma"`).
  - `learner`: An instance of a base learner (e.g., `PyLinearLearner`, `PyPSplineLearner`).

- `fit(X, y) -> None`
  Fits the model to the training data.

  - `X` (`numpy.ndarray` or `scipy.sparse` matrix): The feature matrix/design matrix.
  - `y` (`numpy.ndarray`): The 1D target/response vector.

- `predict(X, param: str) -> numpy.ndarray`
  Predicts the specific distributional parameter for a given feature matrix.

  - `X` (`numpy.ndarray` or `scipy.sparse` matrix): The feature matrix to predict on.
  - `param` (`str`): The name of the parameter to predict (e.g., `"mu"`).
  - _Returns_: A 1D numpy array of the predicted parameter values.

- `cvrisk(folds: int = 10) -> dict`
  Executes k-fold cross-validation to determine the optimal early-stopping iteration (`mstop`) to prevent overfitting.

  - `folds` (`int`, default=10): The number of cross-validation folds.
  - _Returns_: A dictionary containing `optimal_mstop` (int) and `mean_risk` (list of floats).

- `stabsel(b: int = 100, pfer: float = None, pi_thr: float = None, q: int = None, mode: str = "joint") -> PyStabselResult`
  Performs Stability Selection to find stable, informative base learners while controlling the False Discovery Rate.

  - `b` (`int`, default=100): Number of subsampling iterations.
  - `pfer` (`float`, optional): Upper bound for the per-family error rate.
  - `pi_thr` (`float`, optional): Threshold for selection probability.
  - `q` (`int`, optional): Number of features to select per subsample.
  - `mode` (`str`, default="joint"): Method of selection (`"joint"` or `"independent"`).
  - _Returns_: A `PyStabselResult` object containing `.selected`, a list of the indices of the chosen learners.

- `feature_importance() -> list[float]`
  Calculates the empirical risk reduction achieved by each base learner during training.

  - _Returns_: A list of importance scores matching the order in which learners were added via `add_learner`.

- `partial_dependence(X, learner_idx: int) -> numpy.ndarray`
  Computes the partial dependence (marginal effect) of a specific base learner over the given dataset.

  - `X` (`numpy.ndarray`): The feature matrix to compute dependencies over.
  - `learner_idx` (`int`): The zero-indexed position of the learner (based on the order added via `add_learner`).
  - _Returns_: A 1D numpy array representing the partial dependence values.

- `save(path: str) -> None`
  Serializes the trained model to disk.

  - `path` (`str`): The file path where the model JSON should be written.

- `@staticmethod load(path: str) -> BoostLssModel`
  Loads a trained model from disk.
  - `path` (`str`): The file path of the serialized model.
  - _Returns_: A fully instantiated and fitted `BoostLssModel`.

---

### `PyFamily`

Represents the statistical distribution assumed for the response variable.

```python
PyFamily(name: str)
```

**Constructor Parameters:**

- `name` (`str`): The identifier of the distribution family.

**Supported Families and their Parameters:**

- `"GaussianLSS"`: Normal distribution (models `mu` for mean, `sigma` for variance).
- `"StudentTLSS"`: Student's t-distribution (models `mu`, `sigma`, `df`).
- `"GammaLSS"`: Gamma distribution (models `mu`, `sigma`).
- `"BinomialLSS"`: Binomial distribution (models `mu` for probabilities).
- `"BetaLSS"`: Beta distribution (models `mu`, `phi`).
- `"WeibullLSS"`: Weibull distribution (models `mu`, `sigma`).
- `"LogNormalLSS"`: Log-Normal distribution (models `mu`, `sigma`).
- `"ZIPLss"`: Zero-Inflated Poisson distribution (models `mu` for poisson mean, `sigma` for zero-inflation probability).
- `"GEVLss"`: Generalized Extreme Value distribution (models `mu`, `sigma`, `xi`).

---

### `PyStabselResult`

The result object returned by `BoostLssModel.stabsel()`.

**Properties:**

- `selected` (`list[int]`): The integer indices of the base learners that were selected as stable features by the algorithm.

---

## Base Learners

Base learners define the functional form used to relate a feature (or features) to a distributional parameter.

### `PyLinearLearner`

Models a standard linear effect for a single continuous feature.

```python
PyLinearLearner(feature_idx: int, intercept: bool = True)
```

- `feature_idx` (`int`): The column index in the design matrix `X`.
- `intercept` (`bool`, default=True): Whether to include an intercept term for this feature.

### `PyPSplineLearner`

Models a smooth, non-linear effect using univariate Penalized Splines.

```python
PyPSplineLearner(feature_idx: int, degree: int = 3, knots: int = 20, differences: int = 2, df: float = 4.0, cyclic: bool = False)
```

- `feature_idx` (`int`): The column index in the design matrix `X`.
- `degree` (`int`, default=3): The degree of the B-spline basis.
- `knots` (`int`, default=20): The number of inner knots.
- `differences` (`int`, default=2): The order of the difference penalty.
- `df` (`float`, default=4.0): The target degrees of freedom (controls smoothness/penalty strength).
- `cyclic` (`bool`, default=False): If True, models periodic/cyclic effects (e.g. day of year, hour of day).

### `PyBivariatePSplineLearner`

Models spatial or complex interaction effects using a tensor-product of two Penalized Splines.

```python
PyBivariatePSplineLearner(feature1_idx: int, feature2_idx: int, degree: int = 3, knots: int = 20, differences: int = 2, df: float = 4.0)
```

- `feature1_idx` (`int`): The column index for the first feature.
- `feature2_idx` (`int`): The column index for the second feature.
- `degree`, `knots`, `differences`, `df`: Hyperparameters applied to the tensor product.

### `PyConstrainedPSplineLearner`

Models a smooth, non-linear effect that is strictly constrained (e.g. monotonically increasing, decreasing, or convex). Note that this is initialized via the `constrained_pspline` helper function.

```python
from boostlss_py import constrained_pspline

learner = constrained_pspline(
    feature_idx: int,
    constraint: str,
    knots: int = 20,
    degree: int = 3,
    differences: int = 2,
    df: float = 4.0,
    max_iter: int = 10,
    tolerance: float = 1e-6
)
```

- `constraint` (`str`): Must be `"monotonic_increasing"`, `"monotonic_decreasing"`, `"convex"`, or `"concave"`.

### `PyTreeLearner`

Models interactions and non-linearities using a standard decision tree.

```python
PyTreeLearner(feature_indices: list[int], max_depth: int = 3, min_samples_leaf: int = 1)
```

- `feature_indices` (`list[int]`): A list of column indices this tree is allowed to split on.
- `max_depth` (`int`, default=3): Maximum depth of the tree.
- `min_samples_leaf` (`int`, default=1): Minimum number of samples required to be at a leaf node.

### `PyStumpLearner`

Models a single split using a decision stump (a decision tree with a maximum depth of 1).

```python
PyStumpLearner(feature_idx: int)
```

- `feature_idx` (`int`): The column index in the design matrix `X`.

### `PyRandomEffectsLearner`

Models random intercepts for unobserved heterogeneity in categorical grouping variables.

```python
PyRandomEffectsLearner(feature_idx: int, df: float = 4.0)
```

- `feature_idx` (`int`): The column index in the design matrix `X`. This should point to an integer-encoded categorical variable.
- `df` (`float`, default=4.0): The target degrees of freedom defining the ridge penalty.
