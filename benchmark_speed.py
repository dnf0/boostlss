import os
import time
import argparse
import numpy as np

# Workaround for OpenMP conflict with XGBoost
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from boostlss_py import PyFamily, PyTreeLearner, PyHistTreeLearner, BoostLssModel
from xgboostlss.model import XGBoostLSS
from xgboostlss.distributions.Gaussian import Gaussian
import xgboost as xgb


def main():
    parser = argparse.ArgumentParser(description="Benchmark BoostLSS vs XGBoostLSS")
    parser.add_argument(
        "--n-samples", type=int, default=10000, help="Number of samples"
    )
    parser.add_argument("--n-features", type=int, default=10, help="Number of features")
    args = parser.parse_args()

    n_samples = args.n_samples
    n_features = args.n_features

    # 1. Generate Synthetic Data
    np.random.seed(42)
    X = np.random.normal(size=(n_samples, n_features))
    y = X[:, 0] * 2.0 + np.random.normal(size=n_samples) * 0.5

    print(f"Data: {X.shape}")

    # 2. Benchmark xgboostlss
    dtrain = xgb.DMatrix(X, label=y)
    xgboostlss_model = XGBoostLSS(Gaussian(stabilization="None"))

    params = {"eta": 0.1, "max_depth": 3, "booster": "gbtree", "min_child_weight": 1}

    start = time.time()
    print("Starting XGBoostLSS...")
    xgboostlss_model.train(params, dtrain, num_boost_round=50)
    xgb_time = time.time() - start
    print(f"XGBoostLSS Time: {xgb_time:.3f}s")

    # 3. Benchmark boostlss
    family = PyFamily("GaussianLss")
    model = BoostLssModel(family, step_length=0.1, mstop=50, algorithm="noncyclic")

    # Add a single multivariate tree learner for each parameter instead of univariate
    features = list(range(n_features))
    model.add_learner("mu", PyTreeLearner(features, max_depth=3, min_samples_leaf=1))
    model.add_learner("sigma", PyTreeLearner(features, max_depth=3, min_samples_leaf=1))

    start = time.time()
    print("Starting BoostLSS (Exact Tree)...")
    model.fit(X, y)
    boost_time = time.time() - start
    print(f"BoostLSS Time (Exact Tree):   {boost_time:.3f}s")

    # 4. Benchmark boostlss (HistTree)
    hist_model = BoostLssModel(family, step_length=0.1, mstop=50, algorithm="noncyclic")

    hist_model.add_learner(
        "mu", PyHistTreeLearner(features, max_depth=3, min_samples_leaf=1, max_bins=256)
    )
    hist_model.add_learner(
        "sigma",
        PyHistTreeLearner(features, max_depth=3, min_samples_leaf=1, max_bins=256),
    )

    start = time.time()
    print("Starting BoostLSS (HistTree)...")
    hist_model.fit(X, y)
    hist_time = time.time() - start
    print(f"BoostLSS Time (HistTree):     {hist_time:.3f}s")


if __name__ == "__main__":
    main()
