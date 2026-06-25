# boostlss

`boostlss` is an idiomatic Rust library for boosting GAMLSS (Generalized Additive Models for Location, Scale and Shape) with native Python bindings.

This project is deeply inspired by the `gamboostlss` R package (authored by Thomas Kneib, Andreas Mayr, et al.). Our primary goal is to provide a highly performant, thread-safe core algorithmic engine in Rust, complete with native Python bindings to bring robust distributional regression to the modern Python data science ecosystem.

## Features

- **High Performance**: Core engine written in Rust for maximum speed and memory safety.
- **Distributional Regression**: Model the full conditional distribution, not just the mean.
- **Flexible Algorithms**: Choose from cyclic, non-cyclic, and non-cyclic-outer boosting algorithms.
- **Rich Family Support**: Gaussian, Student-T, Gamma, Binomial, Beta, Weibull, LogNormal, ZIP, and GEV.
- **Diverse Base Learners**: Linear, P-Splines, Constrained P-Splines, Random Effects, Stumps, and Trees.
- **Sparse Matrix Support**: Native support for SciPy sparse matrices (CSR/CSC) for memory-efficient training on high-dimensional data.
- **Advanced Tooling**: Built-in cross-validation and stability selection (`stabsel`).

## Installation

You can install `boostlss` directly from PyPI:

```bash
pip install boostlss-py
```

_Note: The Python package name is `boostlss-py` but you import it as `boostlss_py`._

## Quickstart

Here is a quick example of how to fit a Gaussian LSS model using linear base learners.

```python
import numpy as np
from boostlss_py import PyFamily, PyLinearLearner, BoostLssModel

# 1. Generate some synthetic data
np.random.seed(42)
X = np.random.normal(size=(100, 2))
# y = 2 * X_0 + noise
y = X[:, 0] * 2.0 + np.random.normal(size=100) * 0.5

# 2. Specify the distribution family
family = PyFamily("GaussianLSS")

# 3. Initialize the model
model = BoostLssModel(
    family,
    mstop=100,           # Number of boosting iterations
    step_length=0.1,     # Learning rate
    algorithm="cyclic"   # Boosting algorithm ("cyclic", "noncyclic", "noncyclic_outer")
)

# 4. Add base learners for each parameter of the distribution
# For Gaussian, the parameters are "mu" (mean) and "sigma" (scale)
# We add a linear learner on feature 0 for both parameters
model.add_learner("mu", PyLinearLearner(0))
model.add_learner("sigma", PyLinearLearner(0))

# 5. Fit the model
model.fit(X, y)

# 6. Predict on new data
# You must specify which parameter you want to predict
preds_mu = model.predict(X, "mu")
print(f"Predictions for mu: {preds_mu[:5]}")
```

## Validation and Testing

To ensure strict mathematical correctness during the port, `boostlss` is continuously validated against the original R `gamboostlss` implementation. We use "golden tests"—pre-generated data fixtures output by the R package—to verify that our Rust gradient and loss calculations match the original algorithms down to tight floating-point tolerances.
