# BinomialLss Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the 1-parameter `BinomialLss` family for classification and proportional data modeling and expose it through the Python bindings.

**Architecture:** We will create `crates/boostlss/src/family/binomial.rs` containing the `BinomialLss` struct implementing the `Family` trait, add it to `family/mod.rs`, and update the Python `BoostLssModel` in `crates/boostlss-py/src/model.rs` to support the new family type.

**Tech Stack:** Rust (`ndarray`), Python (`pyo3`, `numpy`).

---

### Task 1: Implement BinomialLss Core

**Files:**
- Create: `crates/boostlss/src/family/binomial.rs`
- Modify: `crates/boostlss/src/family/mod.rs`

- [ ] **Step 1: Write the BinomialLss implementation and tests**

Create `crates/boostlss/src/family/binomial.rs`:

```rust
use ndarray::Array1;
use serde::{Deserialize, Serialize};

use crate::error::BoostlssError;
use crate::family::Family;
use crate::param::{Link, ParamSpec};
use crate::util::weighted_mean;

const PARAMS: [ParamSpec; 1] = [ParamSpec {
    name: "mu",
    link: Link::Logit,
}];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinomialLss {
    #[serde(skip, default = "default_binomial_params")]
    params: Vec<ParamSpec>,
}

impl Default for BinomialLss {
    fn default() -> Self {
        Self::new()
    }
}

fn default_binomial_params() -> Vec<ParamSpec> {
    PARAMS.to_vec()
}

impl BinomialLss {
    pub fn new() -> Self {
        Self {
            params: PARAMS.to_vec(),
        }
    }
}

impl Family for BinomialLss {
    fn params(&self) -> &[ParamSpec] {
        &self.params
    }

    fn check_response(&self, y: &Array1<f64>) -> Result<(), BoostlssError> {
        if y.iter().all(|&v| v.is_finite() && v >= 0.0 && v <= 1.0) {
            Ok(())
        } else {
            Err(BoostlssError::UnsupportedResponse(
                "BinomialLss requires 0.0 <= y <= 1.0".into(),
            ))
        }
    }

    fn offset(&self, k: usize, y: &Array1<f64>, w: Option<&Array1<f64>>) -> f64 {
        match k {
            0 => {
                let mean = weighted_mean(y, w);
                // Clamp to avoid +/- inf
                let clamped = mean.clamp(1e-5, 1.0 - 1e-5);
                Link::Logit.apply(clamped)
            }
            _ => unreachable!("BinomialLss has 1 parameter"),
        }
    }

    fn risk(&self, y: &Array1<f64>, etas: &[Array1<f64>], w: Option<&Array1<f64>>) -> f64 {
        let eta = &etas[0];
        let mut acc = 0.0;
        for i in 0..y.len() {
            let eta_i = eta[i];
            let y_i = y[i];

            // Stable NLL computation: log(1 + exp(eta)) - y * eta
            // For large positive eta, log(1 + exp(eta)) ≈ eta
            let log_1_plus_exp = if eta_i > 0.0 {
                eta_i + (-eta_i).exp().ln_1p()
            } else {
                eta_i.exp().ln_1p()
            };

            let nll = log_1_plus_exp - y_i * eta_i;

            let weight = match w {
                Some(weights) => weights[i],
                None => 1.0,
            };
            acc += weight * nll;
        }
        acc
    }

    fn negative_gradient(&self, k: usize, y: &Array1<f64>, etas: &[Array1<f64>]) -> Array1<f64> {
        let eta = &etas[0];
        match k {
            0 => Array1::from_shape_fn(y.len(), |i| {
                let mu = Link::Logit.inverse(eta[i]);
                y[i] - mu
            }),
            _ => unreachable!("BinomialLss has 1 parameter"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::assert_gradient_matches;
    use ndarray::array;

    #[test]
    fn check_response_bounds() {
        let fam = BinomialLss::new();
        assert!(fam.check_response(&array![0.0, 0.5, 1.0]).is_ok());
        assert!(fam.check_response(&array![-0.1, 0.5, 1.0]).is_err());
        assert!(fam.check_response(&array![0.0, 0.5, 1.1]).is_err());
    }

    #[test]
    fn gradient_matches_finite_difference() {
        let fam = BinomialLss::new();
        let y = array![0.0, 1.0, 0.3, 0.7, 1.0];
        let etas = vec![array![-2.0, 0.0, 1.5, 5.0, -10.0]];
        assert_gradient_matches(&fam, 0, &y, &etas);
    }
}
```

- [ ] **Step 2: Expose BinomialLss in `family/mod.rs`**

Modify `crates/boostlss/src/family/mod.rs` around line 43:

```rust
pub mod binomial;
pub mod gamma;
pub mod gaussian;
pub mod nbinomial;
pub mod student_t;

pub use binomial::BinomialLss;
pub use gamma::GammaLss;
pub use gaussian::GaussianLss;
pub use nbinomial::NBinomialLss;
pub use student_t::StudentTLss;
```

- [ ] **Step 3: Run the tests**

Run: `cargo test -p boostlss -- binomial`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add crates/boostlss/src/family/
git commit -m "feat: implement BinomialLss family for classification"
```

---

### Task 2: Python Bindings Support for BinomialLss

**Files:**
- Modify: `crates/boostlss-py/src/family.rs`
- Modify: `crates/boostlss-py/src/model.rs`

- [ ] **Step 1: Add BinomialLss to PyFamily enum**

Modify `crates/boostlss-py/src/family.rs`:

```rust
use pyo3::prelude::*;

#[pyclass(module = "boostlss_py")]
#[derive(Clone)]
pub enum PyFamily {
    GaussianLss,
    BinomialLss,
}

#[pymethods]
impl PyFamily {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        match name {
            "GaussianLSS" | "GaussianLss" => Ok(PyFamily::GaussianLss),
            "BinomialLSS" | "BinomialLss" => Ok(PyFamily::BinomialLss),
            _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown family: {}",
                name
            ))),
        }
    }

    fn __getnewargs__(&self) -> (&str,) {
        match self {
            PyFamily::GaussianLss => ("GaussianLSS",),
            PyFamily::BinomialLss => ("BinomialLSS",),
        }
    }
}
```

- [ ] **Step 2: Add `fitted_binomial` and support it in BoostLssModel**

Modify `crates/boostlss-py/src/model.rs`. Add `fitted_binomial` to the struct:

```rust
use boostlss::family::{BinomialLss, GaussianLss};
// ... other imports ...

#[pyclass(module = "boostlss_py")]
pub struct BoostLssModel {
    family: PyFamily,
    mstop: usize,
    step_length: f64,
    learners: Vec<(String, BaseLearner)>,
    fitted_gaussian: Option<Fitted<GaussianLss>>,
    fitted_binomial: Option<Fitted<BinomialLss>>,
    train_data: Option<(ndarray::Array2<f64>, ndarray::Array1<f64>)>,
}
```

Update `new`:

```rust
    #[new]
    #[pyo3(signature = (family, mstop=100, step_length=0.1))]
    fn new(family: PyFamily, mstop: usize, step_length: f64) -> Self {
        Self {
            family,
            mstop,
            step_length,
            learners: Vec::new(),
            fitted_gaussian: None,
            fitted_binomial: None,
            train_data: None,
        }
    }
```

In `fit` method:

```rust
        match self.family {
            PyFamily::GaussianLss => {
                let mut model = BoostLss::new(GaussianLss::new())
                    .step_length(self.step_length)
                    .mstop(Mstop::Scalar(self.mstop));

                for (param, learner) in &self.learners {
                    model = model
                        .on(param.as_str(), |p| p.add(learner.clone()))
                        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                }

                let fitted = fit_cyclical(model, &dataset)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

                self.fitted_gaussian = Some(fitted);
            }
            PyFamily::BinomialLss => {
                let mut model = BoostLss::new(BinomialLss::new())
                    .step_length(self.step_length)
                    .mstop(Mstop::Scalar(self.mstop));

                for (param, learner) in &self.learners {
                    model = model
                        .on(param.as_str(), |p| p.add(learner.clone()))
                        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                }

                let fitted = fit_cyclical(model, &dataset)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

                self.fitted_binomial = Some(fitted);
            }
        }
```

In `predict` method:

```rust
        if let Some(fitted) = &mut self.fitted_gaussian {
            let pred = fitted
                .predict(&dataset, param, Scale::Response)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            let pred_vec: Vec<f64> = pred.into_iter().collect();
            Ok(PyArray1::from_vec_bound(py, pred_vec))
        } else if let Some(fitted) = &mut self.fitted_binomial {
            let pred = fitted
                .predict(&dataset, param, Scale::Response)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            let pred_vec: Vec<f64> = pred.into_iter().collect();
            Ok(PyArray1::from_vec_bound(py, pred_vec))
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Model not fitted",
            ))
        }
```

In `cvrisk`:

```rust
            match self.family {
                PyFamily::GaussianLss => {
                    // existing Gaussian cvrisk code
                    // ...
                }
                PyFamily::BinomialLss => {
                    let mut model = BoostLss::new(BinomialLss::new())
                        .step_length(self.step_length)
                        .mstop(Mstop::Scalar(self.mstop));

                    for (param, learner) in &self.learners {
                        model = model
                            .on(param.as_str(), |p| p.add(learner.clone()))
                            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                    }

                    let cv = CvRisk::new(model, Resampling::KFold { k: folds });
                    let result = cv
                        .run(&dataset)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

                    let dict = PyDict::new_bound(py);
                    match result.optimal_mstop {
                        Mstop::Scalar(m) => dict.set_item("optimal_mstop", m)?,
                        Mstop::PerParam(v) => dict.set_item("optimal_mstop", v)?,
                    }
                    dict.set_item("mean_risk", result.mean_risk)?;
                    Ok(dict)
                }
            }
```

Update `feature_importance`:
```rust
    pub fn feature_importance(&self) -> PyResult<Vec<f64>> {
        if let Some(fitted) = &self.fitted_gaussian {
            Ok(fitted.feature_importance())
        } else if let Some(fitted) = &self.fitted_binomial {
            Ok(fitted.feature_importance())
        } else {
            Err(pyo3::exceptions::PyValueError::new_err(
                "Model must be fitted before calling feature_importance",
            ))
        }
    }
```

Update `partial_dependence`:
```rust
    pub fn partial_dependence<'py>(
        &mut self,
        _py: Python<'py>,
        data: PyReadonlyArray2<f64>,
        param: &str,
        feature_idx: usize,
        grid: Vec<f64>,
    ) -> PyResult<Vec<f64>> {
        // ... (keep the dataset creation part)
        let x_view = data.as_array();
        let x_mat = ndarray::Array2::from_shape_vec(
            (x_view.nrows(), x_view.ncols()),
            x_view.to_owned().into_raw_vec(),
        )
        .unwrap();

        let n_samples = x_mat.nrows();
        let dummy_response = ndarray::Array1::<f64>::zeros(n_samples);

        let ds = Dataset::new(x_mat, dummy_response, None)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        if let Some(fitted) = &mut self.fitted_gaussian {
            let pd = fitted
                .partial_dependence(&ds, param, feature_idx, &grid)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            Ok(pd)
        } else if let Some(fitted) = &mut self.fitted_binomial {
            let pd = fitted
                .partial_dependence(&ds, param, feature_idx, &grid)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            Ok(pd)
        } else {
            Err(pyo3::exceptions::PyValueError::new_err(
                "Model must be fitted before calling partial_dependence",
            ))
        }
    }
```

Update `__getstate__`:
```rust
        let family_str = match self.family {
            PyFamily::GaussianLss => "GaussianLss",
            PyFamily::BinomialLss => "BinomialLss",
        };
        // ...
        if let Some(fitted) = &self.fitted_gaussian {
            let bytes = bincode::serialize(fitted)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            dict.set_item("fitted_gaussian", PyBytes::new_bound(py, &bytes))?;
        }
        if let Some(fitted) = &self.fitted_binomial {
            let bytes = bincode::serialize(fitted)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            dict.set_item("fitted_binomial", PyBytes::new_bound(py, &bytes))?;
        }
```

Update `__setstate__`:
```rust
        self.family = match family_str.as_str() {
            "GaussianLss" => PyFamily::GaussianLss,
            "BinomialLss" => PyFamily::BinomialLss,
            _ => return Err(pyo3::exceptions::PyValueError::new_err("Unknown family")),
        };
        // ...
        if let Some(bytes_any) = state.get_item("fitted_gaussian")? {
            let bytes: &[u8] = bytes_any.extract()?;
            let fitted: Fitted<GaussianLss> = bincode::deserialize(bytes)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            self.fitted_gaussian = Some(fitted);
        }
        if let Some(bytes_any) = state.get_item("fitted_binomial")? {
            let bytes: &[u8] = bytes_any.extract()?;
            let fitted: Fitted<BinomialLss> = bincode::deserialize(bytes)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            self.fitted_binomial = Some(fitted);
        }
```

- [ ] **Step 3: Run cargo check**

Run: `cargo check -p boostlss-py`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add crates/boostlss-py/src/
git commit -m "feat: expose BinomialLss to Python bindings"
```

---

### Task 3: Python Integration Tests

**Files:**
- Create: `crates/boostlss-py/tests/test_binomial.py`

- [ ] **Step 1: Write test_binomial.py**

Create `crates/boostlss-py/tests/test_binomial.py`:

```python
import pytest
import numpy as np
from boostlss_py import PyFamily, PyLinearLearner, BoostLssModel

def test_binomial_classification():
    # 1. Generate binary classification data
    np.random.seed(42)
    X = np.random.uniform(-3, 3, (200, 1))

    # logit(p) = 1.5 * x
    logits = 1.5 * X[:, 0]
    p = 1.0 / (1.0 + np.exp(-logits))

    y = np.random.binomial(1, p).astype(float)

    # 2. Build model
    family = PyFamily("BinomialLSS")
    model = BoostLssModel(family, mstop=50, step_length=0.1)

    # 3. Add learners
    model.add_learner("mu", PyLinearLearner("x", intercept=True))

    # 4. Fit
    model.fit(X, y)

    # 5. Predict
    pred_mu = model.predict(X, "mu")

    # Verify predictions are valid probabilities
    assert len(pred_mu) == 200
    assert not np.isnan(pred_mu).any()
    assert np.all(pred_mu >= 0.0)
    assert np.all(pred_mu <= 1.0)

    # Verify accurate predictions (AUC or accuracy proxy)
    pred_classes = (pred_mu > 0.5).astype(float)
    accuracy = np.mean(pred_classes == y)
    assert accuracy > 0.70  # Should learn the positive correlation
```

- [ ] **Step 2: Run Python integration tests**

Run: `maturin develop && pytest crates/boostlss-py/tests/test_binomial.py`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add crates/boostlss-py/tests/test_binomial.py
git commit -m "test: add python integration tests for BinomialLss"
```
