# Robust Regression with Student-T

Standard linear regression and basic Machine Learning models usually optimize the Mean Squared Error (MSE), which implicitly assumes the target variable follows a Gaussian distribution.

The problem with the Gaussian assumption is that MSE heavily penalizes large errors. If your dataset contains extreme outliers, the model's line of best fit will be aggressively pulled towards those outliers, ruining predictions for the majority of the normal data.

`boostlss` solves this elegantly by allowing you to swap out the `GaussianLSS` family for the `StudentTLSS` family. The Student's t-distribution has a parameter called Degrees of Freedom (`df`), which controls the thickness of the distribution's tails. Lower `df` means heavier tails, meaning the model expects and tolerates extreme outliers without shifting the mean ($\mu$) to accommodate them.

## 1. Setup and Data Generation

We'll generate a simple linear dataset but corrupt $10\%$ of the observations with massive outliers.

```python
import numpy as np
import matplotlib.pyplot as plt
from boostlss_py import PyFamily, PyLinearLearner, BoostLssModel

np.random.seed(123)
n_samples = 300

# 1. Normal data
X = np.random.uniform(-5, 5, size=(n_samples, 1))
true_slope = 2.0
true_intercept = 1.0

# y = mx + b + normal_noise
y = true_intercept + true_slope * X[:, 0] + np.random.normal(scale=1.5, size=n_samples)

# 2. Corrupt 10% of the data with massive outliers
outlier_indices = np.random.choice(n_samples, size=int(n_samples * 0.1), replace=False)
y[outlier_indices] += np.random.choice([-1, 1], size=len(outlier_indices)) * np.random.uniform(15, 25, size=len(outlier_indices))
```

## 2. Fitting a Standard Gaussian Model

First, let's fit a standard Gaussian model to see how badly the outliers affect it.

```python
# Initialize Gaussian model
gauss_model = BoostLssModel(PyFamily("GaussianLSS"), mstop=100)
gauss_model.add_learner("mu", PyLinearLearner(0))
gauss_model.add_learner("sigma", PyLinearLearner(0))

gauss_model.fit(X, y)

# Predict the mean
gauss_pred_mu = gauss_model.predict(X, "mu")
```

## 3. Fitting a Robust Student-T Model

Now, we'll swap to the `StudentTLSS` family. The engine will simultaneously learn the mean ($\mu$), the scale ($\sigma$), and the heavy-tailedness ($df$) of the distribution.

```python
# Initialize Student-T model
robust_model = BoostLssModel(PyFamily("StudentTLSS"), mstop=150)

# Model the mean
robust_model.add_learner("mu", PyLinearLearner(0))
# Model the scale
robust_model.add_learner("sigma", PyLinearLearner(0))
# Model the degrees of freedom (tail weight)
# We often just use an intercept-only model for df since it's usually a global property
# Feature index doesn't matter much if we only want an intercept, but we must pass one.
robust_model.add_learner("df", PyLinearLearner(0)) 

robust_model.fit(X, y)

# Predict the mean
robust_pred_mu = robust_model.predict(X, "mu")
```

## 4. Comparison and Visualization

Let's plot the predictions of both models against the corrupted data.

```python
# Create line data for plotting
X_line = np.linspace(-5, 5, 100).reshape(-1, 1)

# We can predict on new data
line_gauss = gauss_model.predict(X_line, "mu")
line_robust = robust_model.predict(X_line, "mu")
line_true = true_intercept + true_slope * X_line[:, 0]

plt.figure(figsize=(10, 6))
# Plot the corrupted data points
plt.scatter(X[:, 0], y, alpha=0.5, label="Data (with outliers)", color="grey")

# Plot the fits
plt.plot(X_line, line_true, 'k--', linewidth=2, label="True Relationship")
plt.plot(X_line, line_gauss, 'r-', linewidth=2, label="Gaussian Fit (Pulled by outliers)")
plt.plot(X_line, line_robust, 'b-', linewidth=2, label="Student-T Fit (Robust)")

plt.legend()
plt.title("Robust Regression: GaussianLSS vs StudentTLSS")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
```

### Key Takeaways
- The **GaussianLSS** line is visibly skewed, its slope and intercept dragged toward the massive outliers.
- The **StudentTLSS** line almost perfectly overlaps the True Relationship. Because it was allowed to fit the `df` parameter, it recognized the data had "fat tails" and naturally downweighted the extreme values during optimization.
- Swapping loss functions in `boostlss` requires zero changes to the underlying model architecture or feature engineering, simply pass `PyFamily("StudentTLSS")`.
