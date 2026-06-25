# Examples

Here are some comprehensive end-to-end examples demonstrating advanced features of `boostlss`.

## Non-Linear Modeling with P-Splines

This example demonstrates how to capture non-linear relationships using penalized splines (P-Splines).

```python
import numpy as np
from boostlss_py import PyFamily, PyPSplineLearner, BoostLssModel

# Generate non-linear data
np.random.seed(42)
X = np.random.uniform(-3, 3, size=(200, 1))
y = np.sin(X[:, 0]) + np.random.normal(size=200) * 0.2

# Initialize the Gaussian model
family = PyFamily("GaussianLSS")
model = BoostLssModel(family, mstop=150, step_length=0.05)

# Add a P-Spline learner for the mean ("mu").
# We specify the feature index (0). The default parameters for degrees of freedom,
# number of knots, and spline degree will be used.
model.add_learner("mu", PyPSplineLearner(0))

# We can also model the variance ("sigma") with a P-Spline if we expect heteroscedasticity
model.add_learner("sigma", PyPSplineLearner(0))

# Fit the model
model.fit(X, y)

# Predict the mean and standard deviation
mu_pred = model.predict(X, "mu")
sigma_pred = model.predict(X, "sigma")
```

## Modeling Count Data with ZIP (Zero-Inflated Poisson)

When dealing with count data that contains an excess of zeros, the Zero-Inflated Poisson (ZIP) distribution is ideal.

```python
import numpy as np
import scipy.sparse as sp
from boostlss_py import PyFamily, PyLinearLearner, BoostLssModel

# Generate count data with excess zeros
np.random.seed(42)
X = np.random.normal(size=(500, 2))
# Convert to a sparse matrix to demonstrate sparse support
X_sparse = sp.csr_matrix(X)

# True lambdas and zero-inflation probabilities
true_lambda = np.exp(0.5 * X[:, 0])
true_pi = 1 / (1 + np.exp(-1 * (X[:, 1] - 1)))  # Logit link

y = np.random.poisson(true_lambda)
# Inflate with zeros
is_zero = np.random.binomial(1, true_pi)
y[is_zero == 1] = 0

# Initialize ZIP model
# ZIP models "mu" (poisson mean) and "sigma" (zero-inflation probability)
family = PyFamily("ZIPLss")
model = BoostLssModel(family, mstop=200, step_length=0.1)

# Add linear learners
model.add_learner("mu", PyLinearLearner(0))
model.add_learner("sigma", PyLinearLearner(1))

# Fit directly on the sparse matrix!
model.fit(X_sparse, y)

# Predict lambda and pi
mu_pred = model.predict(X_sparse, "mu")
sigma_pred = model.predict(X_sparse, "sigma")
```

## Cross Validation for Mstop Tuning

To prevent overfitting, it's critical to tune the number of boosting iterations (`mstop`). `boostlss` provides a built-in `cvrisk` method to find the optimal `mstop` using k-fold cross-validation.

```python
import numpy as np
from boostlss_py import PyFamily, PyLinearLearner, BoostLssModel

# Data
np.random.seed(42)
X = np.random.normal(size=(150, 3))
y = X[:, 0] * 1.5 - X[:, 1] * 0.5 + np.random.normal(size=150)

# Initialize Model
family = PyFamily("GaussianLSS")
# Set a high initial mstop; cvrisk will find the optimal stopping point up to this maximum
model = BoostLssModel(family, mstop=500, step_length=0.1)

model.add_learner("mu", PyLinearLearner(0))
model.add_learner("mu", PyLinearLearner(1))
model.add_learner("mu", PyLinearLearner(2))
model.add_learner("sigma", PyLinearLearner(0))

# IMPORTANT: You must fit the model first to provide the training data to the CV engine
model.fit(X, y)

# Run 5-fold cross-validation
cv_results = model.cvrisk(folds=5)

# The result is a dictionary containing the optimal mstop and the mean risk across folds
print(f"Optimal mstop: {cv_results['optimal_mstop']}")
print(f"Mean risk curve length: {len(cv_results['mean_risk'])}")

# You can now re-initialize and train a final model using the optimal_mstop
```

## Stability Selection (stabsel)

Stability selection is a powerful technique for feature selection, controlling the false discovery rate.

```python
import numpy as np
from boostlss_py import PyFamily, PyLinearLearner, BoostLssModel

# High dimensional data with few informative features
np.random.seed(42)
n_features = 20
X = np.random.normal(size=(100, n_features))
y = X[:, 0] * 2.0 + X[:, 5] * -1.5 + np.random.normal(size=100)

model = BoostLssModel(PyFamily("GaussianLSS"), mstop=100)

# Add many learners (one for each feature)
for i in range(n_features):
    model.add_learner("mu", PyLinearLearner(i))

# Fit first to register the data
model.fit(X, y)

# Run stability selection
# b: number of subsamples
# pfer: upper bound for the per-family error rate (False Discoveries)
stabsel_result = model.stabsel(b=50, pfer=1.0)

print("Selected base learners:")
print(stabsel_result.selected)
```

## Model Interpretation

`boostlss` provides methods to interpret the trained model, such as computing the empirical risk reduction (feature importance) and the partial dependence of individual base learners.

```python
import numpy as np
from boostlss_py import PyFamily, PyPSplineLearner, BoostLssModel

np.random.seed(42)
X = np.random.uniform(-3, 3, size=(200, 2))
# y depends heavily on X_0, less on X_1
y = np.sin(X[:, 0]) * 2.0 + X[:, 1] * 0.5 + np.random.normal(size=200) * 0.2

model = BoostLssModel(PyFamily("GaussianLSS"), mstop=150)
model.add_learner("mu", PyPSplineLearner(0))
model.add_learner("mu", PyPSplineLearner(1))

model.fit(X, y)

# 1. Feature Importance
# Returns a list of importance scores corresponding to the order learners were added
importances = model.feature_importance()
print(f"Importance for X_0 learner: {importances[0]}")
print(f"Importance for X_1 learner: {importances[1]}")

# 2. Partial Dependence
# Calculate the marginal effect of the first base learner (X_0) across the data X
pd_X0 = model.partial_dependence(X, learner_idx=0)
print(f"Partial dependence shape: {pd_X0.shape}")
```

## Model Serialization (Save/Load)

You can serialize a trained `BoostLssModel` to disk for later use, including its current state and selected base learners.

```python
import numpy as np
from boostlss_py import PyFamily, PyLinearLearner, BoostLssModel

X = np.random.normal(size=(100, 2))
y = X[:, 0] * 2.0 + np.random.normal(size=100)

model = BoostLssModel(PyFamily("GaussianLSS"), mstop=50)
model.add_learner("mu", PyLinearLearner(0))
model.fit(X, y)

# Save the model to a file
model.save("my_boostlss_model.json")

# Later, or in a different process, load the model
# Note that load() is a static method called on the class itself
loaded_model = BoostLssModel.load("my_boostlss_model.json")

# The loaded model can be used immediately for prediction
preds = loaded_model.predict(X, "mu")
```
