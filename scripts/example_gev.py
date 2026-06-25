import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import genextreme

from boostlss_py import PyFamily, PyLinearLearner, PyTreeLearner, BoostLssModel

# Generate synthetic weather data (e.g. daily maximum wind speed)
np.random.seed(42)
n_samples = 4000

# Predictors
temperature = np.random.uniform(10, 40, n_samples)
pressure_drop = np.random.uniform(0, 50, n_samples) # e.g. hPa drop over 24h

# The location parameter (mu) varies linearly with temperature and pressure drop
mu_true = 20.0 + 0.5 * temperature + 0.3 * pressure_drop

# The scale parameter (sigma) varies non-linearly with pressure drop
sigma_true = 2.0 + 5.0 * np.exp(-((pressure_drop - 30) / 10)**2)

# The shape parameter (nu) is constant but non-zero (Frechet domain vs Weibull)
nu_true = np.full(n_samples, 0.1)

# Generate max wind speed from GEV
# Scipy genextreme uses c = -nu. 
# BoostLSS parameterization: if nu > 0, it's Frechet-like (heavy tail)
max_wind_speed = genextreme.rvs(c=-0.1, loc=mu_true, scale=sigma_true)

# Fit BoostLSS
X = np.column_stack([temperature, pressure_drop])
y = max_wind_speed

model = BoostLssModel(PyFamily("GEVLSS"), mstop=200, step_length=0.1)

# Linear learners for mu
model.add_learner("mu", PyLinearLearner(feature_idx=0, intercept=True))
model.add_learner("mu", PyLinearLearner(feature_idx=1, intercept=False))

# Tree learners for sigma
model.add_learner("sigma", PyTreeLearner(feature_indices=[0, 1], max_depth=3))

# Constant for nu
model.add_learner("nu", PyLinearLearner(feature_idx=0, intercept=True))

model.fit(X, y)

# Predict
mu_pred = model.predict(X, "mu")
sigma_pred = model.predict(X, "sigma")

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot True vs Predicted mu (Location)
idx_temp = np.argsort(temperature)
ax1.plot(temperature[idx_temp], mu_true[idx_temp], 'k--', label="True $\mu$", linewidth=2)
ax1.scatter(temperature[idx_temp], mu_pred[idx_temp], c='blue', alpha=0.1, label="Predicted $\mu$", s=5)
ax1.set_xlabel("Temperature")
ax1.set_ylabel("$\mu$")
ax1.set_title("Location Parameter ($\mu$) vs Temperature")
ax1.legend()

# Plot True vs Predicted sigma (Scale)
idx_pres = np.argsort(pressure_drop)
ax2.plot(pressure_drop[idx_pres], sigma_true[idx_pres], 'k--', label="True $\sigma$", linewidth=2)
ax2.scatter(pressure_drop[idx_pres], sigma_pred[idx_pres], c='red', alpha=0.3, label="Predicted $\sigma$", s=10)
ax2.set_xlabel("Pressure Drop")
ax2.set_ylabel("$\sigma$")
ax2.set_title("Scale Parameter ($\sigma$) vs Pressure Drop")
ax2.legend()

plt.tight_layout()
plt.savefig("docs/assets/gev_extreme_weather.png", dpi=300, bbox_inches='tight')
print("Saved docs/assets/gev_extreme_weather.png")
