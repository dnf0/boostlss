# Quickstart

This guide will walk you through the basics of using `boostlss` to fit a distributional regression model.

## 1. Import the Library

The Python package name is `boostlss_py`. Import the `BoostLssModel` and the base learners or families you intend to use.

```python
import numpy as np
from boostlss_py import PyFamily, PyLinearLearner, BoostLssModel
```

## 2. Prepare Your Data

`boostlss` accepts standard NumPy arrays. It also natively supports SciPy sparse matrices (CSR/CSC formats) for efficient memory usage.

```python
np.random.seed(42)
X = np.random.normal(size=(100, 2))

# Let's generate a response y that depends on X
# y = 2 * X_0 + noise
y = X[:, 0] * 2.0 + np.random.normal(size=100) * 0.5
```

## 3. Choose a Distribution Family

A core component of GAMLSS is specifying the assumed distribution of the response variable. Each family has its own set of parameters (e.g., location, scale, shape).

```python
# We use the Gaussian distribution which models "mu" (mean) and "sigma" (scale)
family = PyFamily("GaussianLSS")
```

## 4. Initialize the Model

Create the `BoostLssModel` specifying the family, the number of boosting iterations (`mstop`), the learning rate (`step_length`), and the boosting algorithm.

```python
model = BoostLssModel(
    family,
    mstop=100,           # Number of boosting iterations
    step_length=0.1,     # Learning rate
    algorithm="cyclic",  # Boosting algorithm
)
```

**Algorithms:**

- `cyclic`: Cycles through the parameters sequentially in each iteration.
- `noncyclic`: Updates the parameter that yields the best overall loss reduction in each iteration.
- `noncyclic_outer`: Updates all parameters simultaneously and chooses the best overall improvement.

## 5. Add Base Learners

You must add at least one base learner for _each_ parameter of the distribution family you selected.

```python
# We add a linear base learner on feature index 0 for both "mu" and "sigma".
model.add_learner("mu", PyLinearLearner(0))
model.add_learner("sigma", PyLinearLearner(0))
```

_Note: You can add multiple learners for a single parameter. The algorithm will automatically select the best learner at each step._

## 6. Fit the Model

Train the model on your data matrix `X` and response array `y`.

```python
model.fit(X, y)
```

## 7. Make Predictions

Once trained, you can predict the specific parameters of your distribution for new data.

```python
preds_mu = model.predict(X, "mu")
print(f"Predictions for mu: {preds_mu[:5]}")

preds_sigma = model.predict(X, "sigma")
print(f"Predictions for sigma: {preds_sigma[:5]}")
```
