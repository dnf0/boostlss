import time
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from boostlss_py import PyFamily, PyPSplineLearner, BoostLssModel


def generate_long_tail_data(n=20000):
    np.random.seed(42)
    x = np.random.uniform(-5, 5, n)

    # JSU parameters
    mu = 2.0 * np.sin(x)
    sigma = 0.5 + 0.2 * np.abs(x)
    nu = -1.5  # skewness
    tau = 0.8  # tail weight

    # Generate JSU target y
    z = np.random.normal(0, 1, n)
    y = mu + sigma * np.sinh((z - nu) / tau)

    return x.reshape(-1, 1), y


X, y = generate_long_tail_data()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_test = np.clip(X_test, X_train.min(), X_train.max())

print("--- Data Summary ---")
print(
    f"y_train min: {y_train.min():.2f}, max: {y_train.max():.2f}, mean: {y_train.mean():.2f}, std: {y_train.std():.2f}"
)

# 1. HistGradientBoostingRegressor (MSE)
print("\n--- Training HistGradientBoosting (MSE) ---")
start_time = time.time()
model_hgbr_mse = HistGradientBoostingRegressor(
    max_iter=150, max_depth=3, learning_rate=0.1, loss="squared_error"
)
model_hgbr_mse.fit(X_train, y_train)
hgbr_mse_time = time.time() - start_time
y_pred_hgbr_mse = model_hgbr_mse.predict(X_test)
hgbr_mse_mae = mean_absolute_error(y_test, y_pred_hgbr_mse)
print(f"Time: {hgbr_mse_time:.2f}s | MAE: {hgbr_mse_mae:.4f}")

# 2. HistGradientBoostingRegressor (MAE)
print("\n--- Training HistGradientBoosting (MAE) ---")
start_time = time.time()
model_hgbr_mae = HistGradientBoostingRegressor(
    max_iter=150, max_depth=3, learning_rate=0.1, loss="absolute_error"
)
model_hgbr_mae.fit(X_train, y_train)
hgbr_mae_time = time.time() - start_time
y_pred_hgbr_mae = model_hgbr_mae.predict(X_test)
hgbr_mae_mae = mean_absolute_error(y_test, y_pred_hgbr_mae)
print(f"Time: {hgbr_mae_time:.2f}s | MAE: {hgbr_mae_mae:.4f}")

# 3. BoostLSS (JSULss)
print("\n--- Training BoostLSS (JSULss) ---")
start_time = time.time()
family = PyFamily("JSULss")
model_jsu = BoostLssModel(family, mstop=150, step_length=0.1)

# Add P-Splines for all 4 parameters
model_jsu.add_learner("mu", PyPSplineLearner(feature_idx=0, df=4.0))
model_jsu.add_learner("sigma", PyPSplineLearner(feature_idx=0, df=4.0))
model_jsu.add_learner("nu", PyPSplineLearner(feature_idx=0, df=4.0))
model_jsu.add_learner("tau", PyPSplineLearner(feature_idx=0, df=4.0))

model_jsu.fit(X_train, y_train)
jsu_time = time.time() - start_time

y_pred_jsu_mu = model_jsu.predict(X_test, "mu")
y_pred_jsu_sigma = model_jsu.predict(X_test, "sigma")
y_pred_jsu_nu = model_jsu.predict(X_test, "nu")
y_pred_jsu_tau = model_jsu.predict(X_test, "tau")

# For JSU, mean is complex. mu is the location parameter, not the mean.
# We will evaluate MAE against true mu vs predicted mu as an example,
# or just calculate the MAE of prediction. Actually, the expected value of JSU is analytically known:
# E[Y] = mu - sigma * exp(1 / (2 * tau^2)) * sinh(nu / tau)
expected_y_jsu = y_pred_jsu_mu - y_pred_jsu_sigma * np.exp(
    1 / (2 * y_pred_jsu_tau**2)
) * np.sinh(y_pred_jsu_nu / y_pred_jsu_tau)

jsu_mae = mean_absolute_error(y_test, expected_y_jsu)
print(f"Time: {jsu_time:.2f}s | MAE: {jsu_mae:.4f}")

# Generate plot of true distribution at a single point to show the fit
x_point = np.array([[2.0]])
mu_hat = model_jsu.predict(x_point, "mu")[0]
sigma_hat = model_jsu.predict(x_point, "sigma")[0]
nu_hat = model_jsu.predict(x_point, "nu")[0]
tau_hat = model_jsu.predict(x_point, "tau")[0]

print(
    f"\nAt x=2.0, predicted parameters: mu={mu_hat:.2f}, sigma={sigma_hat:.2f}, nu={nu_hat:.2f}, tau={tau_hat:.2f}"
)
