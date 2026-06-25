import time
import numpy as np
from boostlss_py import PyFamily, PyLinearLearner, BoostLssModel

def run_benchmark():
    print("=== BoostLSS Performance Benchmarks ===")
    
    n_features = 20
    mstop = 50
    
    sizes = [1_000, 10_000, 50_000]
    
    for n_samples in sizes:
        print(f"\nBenchmarking N={n_samples:,}, P={n_features}, M={mstop}")
        
        # Generate data
        np.random.seed(42)
        X = np.random.randn(n_samples, n_features)
        true_beta = np.random.randn(n_features)
        y = X @ true_beta + np.random.randn(n_samples)
        
        # Initialize model
        family = PyFamily("GaussianLSS")
        model = BoostLssModel(family, mstop=mstop, step_length=0.1)
        
        # Add a linear learner for mu for every feature
        for i in range(n_features):
            model.add_learner("mu", PyLinearLearner(feature_idx=i, intercept=False))
            
        model.add_learner("sigma", PyLinearLearner(feature_idx=0, intercept=True))
        
        # Fit model
        start_time = time.time()
        model.fit(X, y)
        fit_time = time.time() - start_time
        
        print(f"  Fit time: {fit_time:.3f} seconds")
        
        # Predict
        start_time = time.time()
        _ = model.predict(X, "mu")
        pred_time = time.time() - start_time
        
        print(f"  Predict time: {pred_time:.3f} seconds")

if __name__ == "__main__":
    run_benchmark()
