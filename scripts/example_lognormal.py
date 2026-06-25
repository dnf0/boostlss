import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import lognorm

from boostlss_py import PyFamily, PyLinearLearner, PyTreeLearner, BoostLssModel

# Generate synthetic claims data
np.random.seed(123)
n_samples = 5000

# Predictors
age = np.random.uniform(18, 80, n_samples)
car_value = np.random.uniform(10_000, 100_000, n_samples)

# The location parameter (mu) varies with age and car_value
mu_true = 7.0 - 0.01 * age + 0.00001 * car_value

# The scale parameter (sigma) varies non-linearly with age (e.g. young drivers have highly variable claims)
sigma_true = 0.5 + 1.5 * np.exp(-((age - 20) / 10)**2)

# Generate claims from LogNormal
claims = lognorm.rvs(s=sigma_true, scale=np.exp(mu_true))

# Fit BoostLSS
X = np.column_stack([age, car_value])
y = claims

model = BoostLssModel(PyFamily("LogNormalLSS"), mstop=300, step_length=0.1)

# Linear learners for mu
model.add_learner("mu", PyLinearLearner(feature_idx=0, intercept=True))
model.add_learner("mu", PyLinearLearner(feature_idx=1, intercept=False))

# Tree learners for sigma because it's non-linear
model.add_learner("sigma", PyTreeLearner(feature_indices=[0, 1], max_depth=3))

model.fit(X, y)

# Predict
mu_pred = model.predict(X, "mu")
sigma_pred = model.predict(X, "sigma")

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot True vs Predicted mu (Location)
# Sort by car_value for a cleaner line plot
idx = np.argsort(car_value)
ax1.plot(car_value[idx], mu_true[idx], 'k--', label="True $\mu$", linewidth=2)
ax1.scatter(car_value[idx], mu_pred[idx], c='blue', alpha=0.1, label="Predicted $\mu$", s=5)
ax1.set_xlabel("Car Value")
ax1.set_ylabel("$\mu$")
ax1.set_title("Location Parameter ($\mu$) vs Car Value")
ax1.legend()

# Plot True vs Predicted sigma (Scale)
idx_age = np.argsort(age)
ax2.plot(age[idx_age], sigma_true[idx_age], 'k--', label="True $\sigma$", linewidth=2)
ax2.scatter(age[idx_age], sigma_pred[idx_age], c='red', alpha=0.3, label="Predicted $\sigma$", s=10)
ax2.set_xlabel("Age")
ax2.set_ylabel("$\sigma$")
ax2.set_title("Scale Parameter ($\sigma$) vs Age")
ax2.legend()

plt.tight_layout()
plt.savefig("docs/assets/lognormal_claims.png", dpi=300, bbox_inches='tight')
print("Saved docs/assets/lognormal_claims.png")
