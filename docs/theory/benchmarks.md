# Performance Benchmarks

BoostLSS is designed for speed and scalability, leveraging Rust's performance and memory safety under the hood, while exposing a seamless and idiomatic Python API.

The following benchmarks demonstrate the fitting and prediction times for a `GaussianLSS` model with $P=20$ features, using linear base learners for both $\mu$ and $\sigma$. 

All benchmarks were run locally (single-threaded CPU execution).

## Linear Base Learners (GaussianLSS)

We evaluated performance on simulated datasets of varying sample sizes ($N$), fixing the number of features ($P=20$) and boosting iterations ($M=50$).

| Samples ($N$) | Fit Time (s) | Predict Time (s) |
|---------------|--------------|------------------|
| 1,000         | 0.408        | 0.015            |
| 10,000        | 3.992        | 0.151            |
| 50,000        | 20.106       | 0.776            |

### Analysis
As demonstrated, BoostLSS scales linearly $\mathcal{O}(N)$ with respect to the number of samples.

* **Fitting:** Fitting a model on $50,000$ samples across 20 features for 50 iterations takes approximately 20 seconds. 
* **Prediction:** Inference is heavily optimized. Generating predictions for $50,000$ new observations takes less than a second.

## Memory Footprint
Because the core engine is written in Rust, it benefits from strict memory management. The Python bindings use zero-copy views (`ndarray::ArrayView`) whenever possible, meaning that transferring large datasets from Python (e.g. NumPy arrays or SciPy sparse matrices) into the Rust engine incurs negligible memory overhead.
