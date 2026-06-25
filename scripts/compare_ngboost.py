import time
import numpy as np
from ngboost import NGBRegressor

from boostlss_py import PyFamily, PyTreeLearner, BoostLssModel

def run_comparison():
    print("=== BoostLSS vs NGBoost (Tree Learners) ===")
    
    n_features = 10
    mstop = 50
    
    sizes = [1_000, 2_000]
    
    for n_samples in sizes:
        print(f"\nBenchmarking N={n_samples:,}, P={n_features}, M={mstop}")
        
        # Generate data
        np.random.seed(42)
        X = np.random.randn(n_samples, n_features)
        true_beta = np.random.randn(n_features)
        y = X @ true_beta + np.random.randn(n_samples)
        
        # --- NGBoost ---
        ngb = NGBRegressor(n_estimators=mstop, learning_rate=0.1, verbose=False)
        start_time = time.time()
        ngb.fit(X, y)
        ngb_fit_time = time.time() - start_time
        
        start_time = time.time()
        _ = ngb.predict(X)
        ngb_pred_time = time.time() - start_time
        
        print(f"  [NGBoost]  Fit: {ngb_fit_time:.3f}s | Predict: {ngb_pred_time:.3f}s")
        
        # --- BoostLSS ---
        family = PyFamily("GaussianLSS")
        model = BoostLssModel(family, mstop=mstop, step_length=0.1)
        
        # Add a single tree learner that can split on all features for mu and sigma
        all_features = list(range(n_features))
        model.add_learner("mu", PyTreeLearner(all_features, max_depth=3))
        model.add_learner("sigma", PyTreeLearner(all_features, max_depth=3))
            
        start_time = time.time()
        model.fit(X, y)
        blss_fit_time = time.time() - start_time
        
        start_time = time.time()
        _ = model.predict(X, "mu")
        blss_pred_time = time.time() - start_time
        
        print(f"  [BoostLSS] Fit: {blss_fit_time:.3f}s | Predict: {blss_pred_time:.3f}s")
        
        speedup_fit = ngb_fit_time / blss_fit_time
        speedup_pred = ngb_pred_time / blss_pred_time
        print(f"  --> BoostLSS is {speedup_fit:.1f}x faster at fitting and {speedup_pred:.1f}x faster at predicting!")

if __name__ == "__main__":
    run_comparison()
