import time
import numpy as np
from boostlss_py import PyFamily, PyTreeLearner, BoostLssModel
from xgboostlss.model import XGBoostLSS
from xgboostlss.distributions.Gaussian import Gaussian
import xgboost as xgb

# 1. Generate Synthetic Data
n_samples = 10000
n_features = 10
np.random.seed(42)
X = np.random.normal(size=(n_samples, n_features))
y = X[:, 0] * 2.0 + np.random.normal(size=n_samples) * 0.5

print(f"Data: {X.shape}")

# 2. Benchmark xgboostlss
dtrain = xgb.DMatrix(X, label=y)
xgboostlss_model = XGBoostLSS(Gaussian(stabilization="None"))

params = {"eta": 0.1, "max_depth": 3, "booster": "gbtree", "min_child_weight": 1}

start = time.time()
xgboostlss_model.train(params, dtrain, num_boost_round=50)
xgb_time = time.time() - start
print(f"XGBoostLSS Time: {xgb_time:.3f}s")

# 3. Benchmark boostlss
family = PyFamily("GaussianLss")
model = BoostLssModel(family, step_length=0.1, mstop=50, algorithm="noncyclic")

# Add a tree learner for each feature
for i in range(n_features):
    model.add_learner("mu", PyTreeLearner([i], max_depth=3, min_samples_leaf=1))
    model.add_learner("sigma", PyTreeLearner([i], max_depth=3, min_samples_leaf=1))

start = time.time()
model.fit(X, y)
boost_time = time.time() - start
print(f"BoostLSS Time:   {boost_time:.3f}s")
