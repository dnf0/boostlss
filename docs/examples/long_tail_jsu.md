# Modeling Long-Tail Distributions with JSU

When modeling real-world data like financial returns, housing prices, or insurance claims, the target variable often exhibits significant skewness and heavy tails. A standard Gaussian assumption will drastically underpredict the frequency of extreme values (the "long tail") and will lead to poorly calibrated risk estimates.

The **Johnson's SU (JSU)** distribution is highly flexible: it can model skewness (asymmetry) and kurtosis (heavy or light tails) independently. It is parameterized by four parameters:

- **$\mu$ (mu)**: Location
- **$\sigma$ (sigma)**: Scale
- **$\nu$ (nu)**: Skewness (negative values indicate right-skewness)
- **$\tau$ (tau)**: Tail weight / Kurtosis (smaller values indicate heavier tails)

In this tutorial, we will use `JSULss` to model a synthetic dataset with a highly non-linear location and heavy, right-skewed tails, and compare it to a standard Mean Squared Error (MSE) and Mean Absolute Error (MAE) model.

## 1. Setup and Data Generation

We'll generate a dataset where both the location and the scale vary non-linearly with $x$.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from boostlss_py import PyFamily, PyPSplineLearner, BoostLssModel

# 1. Simulate data with a non-linear relationship and heavy tails
np.random.seed(42)
n_samples = 10000
x = np.random.uniform(-5, 5, n_samples)

# True underlying parameters for the JSU distribution
true_mu = 2.0 * np.sin(x)                 # Non-linear location
true_sigma = 0.5 + 0.2 * np.abs(x)        # Heteroscedastic scale
true_nu = -1.5                            # Right-skewed
true_tau = 0.8                            # Heavy tails

# Generate the JSU response variable y
z = np.random.normal(0, 1, n_samples)
y = true_mu + true_sigma * np.sinh((z - true_nu) / true_tau)

# Reshape X for the learners
X = x.reshape(-1, 1)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test = np.clip(X_test, X_train.min(), X_train.max())
```

## 2. Training the BoostLSS Model

We'll fit a `BoostLssModel` using the `JSULss` family. Since the relationship is non-linear, we'll use P-Splines (`PyPSplineLearner`) for all four parameters.

```python
# Initialize the model with the JSULss family
family = PyFamily("JSULss")
model_jsu = BoostLssModel(family, mstop=150, step_length=0.1)

# Add P-Spline learners for all four parameters of the JSU distribution
model_jsu.add_learner("mu", PyPSplineLearner(feature_idx=0, df=4.0))
model_jsu.add_learner("sigma", PyPSplineLearner(feature_idx=0, df=4.0))
model_jsu.add_learner("nu", PyPSplineLearner(feature_idx=0, df=4.0))
model_jsu.add_learner("tau", PyPSplineLearner(feature_idx=0, df=4.0))

print("Fitting JSULss model...")
model_jsu.fit(X_train, y_train)
```

## 3. Comparison with Standard Baselines

Let's compare our `JSULss` model against a standard gradient boosting regressor (`HistGradientBoostingRegressor`) trained with Mean Squared Error (MSE) and Mean Absolute Error (MAE). Standard regressors only model the conditional mean or median, entirely ignoring the shape of the tail.

```python
# HGBR (MSE)
model_hgbr_mse = HistGradientBoostingRegressor(max_iter=150, max_depth=3, learning_rate=0.1, loss='squared_error')
model_hgbr_mse.fit(X_train, y_train.ravel())
y_pred_hgbr_mse = model_hgbr_mse.predict(X_test)

# HGBR (MAE)
model_hgbr_mae = HistGradientBoostingRegressor(max_iter=150, max_depth=3, learning_rate=0.1, loss='absolute_error')
model_hgbr_mae.fit(X_train, y_train.ravel())
y_pred_hgbr_mae = model_hgbr_mae.predict(X_test)

# BoostLSS JSU (Expected Value)
# The expected value of a JSU distribution can be calculated analytically:
y_pred_jsu_mu = model_jsu.predict(X_test, "mu")
y_pred_jsu_sigma = model_jsu.predict(X_test, "sigma")
y_pred_jsu_nu = model_jsu.predict(X_test, "nu")
y_pred_jsu_tau = model_jsu.predict(X_test, "tau")

expected_y_jsu = y_pred_jsu_mu - y_pred_jsu_sigma * np.exp(1 / (2 * y_pred_jsu_tau**2)) * np.sinh(y_pred_jsu_nu / y_pred_jsu_tau)

print(f"XGBoost (MSE) MAE: {mean_absolute_error(y_test, y_pred_xgb_mse):.4f}")
print(f"XGBoost (MAE) MAE: {mean_absolute_error(y_test, y_pred_xgb_mae):.4f}")
print(f"BoostLSS (JSU) MAE: {mean_absolute_error(y_test, expected_y_jsu):.4f}")
```

### Why does JSU win on MAE?

Even when comparing Mean Absolute Error, the `JSULss` model often outperforms standard models because it explicitly models the heavy tails and skewness. Standard models trained with MSE are heavily penalized by extreme outliers in the long tail, which warps their predictions for the bulk of the data. By explicitly modeling the tail weight (`tau`), the JSU model allows the main location parameter (`mu`) to remain robust against outliers.

## 4. Predicting the Full Distribution

The true power of `boostlss` is not just in point estimates, but in predicting the full conditional distribution. We can predict the exact parameters for any given point to calculate confidence intervals, quantiles, or simulate future scenarios.

```python
# Predict the parameters for x = 2.0
x_point = np.array([[2.0]])
mu_hat = model_jsu.predict(x_point, "mu")[0]
sigma_hat = model_jsu.predict(x_point, "sigma")[0]
nu_hat = model_jsu.predict(x_point, "nu")[0]
tau_hat = model_jsu.predict(x_point, "tau")[0]

print(f"Predicted parameters at x=2.0:")
print(f"mu: {mu_hat:.2f}, sigma: {sigma_hat:.2f}, nu: {nu_hat:.2f}, tau: {tau_hat:.2f}")
```

These parameters define the full probability distribution at $x = 2.0$, allowing you to accurately estimate the probability of an extreme "long tail" event occurring.
