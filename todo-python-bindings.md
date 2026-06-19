# TODO: Python Bindings Implementation

## Task 1: Scaffolding `boostlss-py` Cargo package

**Files:**

- Modify: `Cargo.toml` (root workspace)
- Create: `crates/boostlss-py/Cargo.toml`
- Create: `crates/boostlss-py/src/lib.rs`
- Create: `crates/boostlss-py/pyproject.toml`

- [x] **Step 1: Update workspace Cargo.toml**
      Add the new crate to the workspace `Cargo.toml`.

```toml
[workspace]
members = [
    "crates/boostlss",
    "crates/boostlss-py"
]
```

- [x] **Step 2: Create `boostlss-py` Cargo.toml**
      Create `crates/boostlss-py/Cargo.toml`:

```toml
[package]
name = "boostlss-py"
version = "0.1.0"
edition = "2021"
publish = false

[lib]
name = "boostlss_py"
crate-type = ["cdylib"]

[dependencies]
boostlss = { path = "../boostlss" }
pyo3 = { version = "0.21.2", features = ["extension-module", "abi3-py38"] }
numpy = "0.21.0"
```

- [x] **Step 3: Create `pyproject.toml` for Maturin**
      Create `crates/boostlss-py/pyproject.toml`:

```toml
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[project]
name = "boostlss"
version = "0.1.0"
requires-python = ">=3.8"
dependencies = [
    "numpy>=1.20"
]
```

- [x] **Step 4: Create a placeholder lib file**
      Create `crates/boostlss-py/src/lib.rs`:

```rust
use pyo3::prelude::*;

#[pyfunction]
fn hello() -> PyResult<String> {
    Ok("Hello from boostlss!".to_string())
}

#[pymodule]
fn boostlss_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello, m)?)?;
    Ok(())
}
```

- [x] **Step 5: Test the library compilation**
      Run: `cargo check -p boostlss-py`
      Expected: PASS.

- [x] **Step 6: Commit**

```bash
git add Cargo.toml crates/boostlss-py
git commit -m "feat: setup boostlss-py crate and maturin pyproject"
```

---

## Task 2: Exposing Families and Base Learners

**Files:**

- Create: `crates/boostlss-py/src/family.rs`
- Create: `crates/boostlss-py/src/learner.rs`
- Modify: `crates/boostlss-py/src/lib.rs`

- [x] **Step 1: Create python bindings for Families**
      Create `crates/boostlss-py/src/family.rs`:

```rust
use pyo3::prelude::*;
use boostlss::family::{GaussianLss, Family};

#[pyclass]
#[derive(Clone)]
pub enum PyFamily {
    GaussianLss,
}

#[pymethods]
impl PyFamily {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        match name {
            "GaussianLSS" => Ok(PyFamily::GaussianLss),
            _ => Err(pyo3::exceptions::PyValueError::new_err(format!("Unknown family: {}", name))),
        }
    }
}
```

- [x] **Step 2: Create python bindings for Base Learners**
      Create `crates/boostlss-py/src/learner.rs`:

```rust
use pyo3::prelude::*;
use boostlss::learner::{BaseLearner, Linear};

#[pyclass]
#[derive(Clone)]
pub struct PyLinearLearner {
    pub name: String,
    pub intercept: bool,
}

#[pymethods]
impl PyLinearLearner {
    #[new]
    #[pyo3(signature = (name, intercept=true))]
    fn new(name: String, intercept: bool) -> Self {
        Self { name, intercept }
    }
}

impl Into<BaseLearner> for PyLinearLearner {
    fn into(self) -> BaseLearner {
        BaseLearner::Linear(Linear::new(&self.name).intercept(self.intercept))
    }
}
```

- [x] **Step 3: Register classes in module**
      Modify `crates/boostlss-py/src/lib.rs`:

```rust
mod family;
mod learner;

use pyo3::prelude::*;
use family::PyFamily;
use learner::PyLinearLearner;

#[pymodule]
fn boostlss_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFamily>()?;
    m.add_class::<PyLinearLearner>()?;
    Ok(())
}
```

- [x] **Step 4: Verify build**
      Run: `cargo check -p boostlss-py`
      Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add crates/boostlss-py/src
git commit -m "feat: expose Family and LinearLearner to Python"
```

---

## Task 3: The BoostLssModel Python Interface

**Files:**

- Create: `crates/boostlss-py/src/model.rs`
- Modify: `crates/boostlss-py/src/lib.rs`

- [x] **Step 1: Create BoostLssModel class**
      Create `crates/boostlss-py/src/model.rs`:

```rust
use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyArray1, ToPyArray};
use boostlss::model::{BoostLss, Fitted, Scale};
use boostlss::data::Dataset;
use boostlss::family::GaussianLss;
use boostlss::engine::Mstop;
use crate::family::PyFamily;
use crate::learner::PyLinearLearner;
use boostlss::engine::cyclical::fit_cyclical;
use ndarray::Array1;

#[pyclass]
pub struct BoostLssModel {
    family: PyFamily,
    mstop: usize,
    step_length: f64,
    learners: Vec<(String, PyLinearLearner)>,
    fitted_gaussian: Option<Fitted<GaussianLss>>,
}

#[pymethods]
impl BoostLssModel {
    #[new]
    #[pyo3(signature = (family, mstop=100, step_length=0.1))]
    fn new(family: PyFamily, mstop: usize, step_length: f64) -> Self {
        Self {
            family,
            mstop,
            step_length,
            learners: Vec::new(),
            fitted_gaussian: None,
        }
    }

    fn add_learner(&mut self, param: String, learner: PyLinearLearner) {
        self.learners.push((param, learner));
    }

    fn fit(&mut self, x: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>) -> PyResult<()> {
        let x_mat = x.as_array().to_owned();
        let y_vec = y.as_array().to_owned();
        let dataset = Dataset::new(x_mat, y_vec, None)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        match self.family {
            PyFamily::GaussianLss => {
                let mut model = BoostLss::new(GaussianLss::new())
                    .step_length(self.step_length)
                    .mstop(Mstop::Scalar(self.mstop));

                for (param, learner) in &self.learners {
                    model = model.on(&param, learner.clone().into())
                        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                }

                let fitted = fit_cyclical(model, &dataset)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

                self.fitted_gaussian = Some(fitted);
            }
        }
        Ok(())
    }

    fn predict<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        param: &str,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x_mat = x.as_array().to_owned();
        // create dummy y for Dataset constructor requirements
        let y_dummy = Array1::zeros(x_mat.nrows());
        let dataset = Dataset::new(x_mat, y_dummy, None)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        if let Some(fitted) = &mut self.fitted_gaussian {
            let pred = fitted.predict(&dataset, param, Scale::Response)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            Ok(pred.to_pyarray_bound(py))
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err("Model not fitted"))
        }
    }
}
```

- [x] **Step 2: Register BoostLssModel**
      Modify `crates/boostlss-py/src/lib.rs`:

```rust
mod family;
mod learner;
mod model;

use pyo3::prelude::*;
use family::PyFamily;
use learner::PyLinearLearner;
use model::BoostLssModel;

#[pymodule]
fn boostlss_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFamily>()?;
    m.add_class::<PyLinearLearner>()?;
    m.add_class::<BoostLssModel>()?;
    Ok(())
}
```

- [x] **Step 3: Verify build**
      Run: `cargo check -p boostlss-py`
      Expected: PASS.

- [x] **Step 4: Commit**

```bash
git add crates/boostlss-py/src
git commit -m "feat: expose BoostLssModel binding with fit and predict methods"
```

---

## Task 4: Integration testing the bindings

**Files:**

- Create: `crates/boostlss-py/tests/test_basic.py`

- [x] **Step 1: Create Python test script**
      Create `crates/boostlss-py/tests/test_basic.py`:

```python
import pytest
import numpy as np
from boostlss_py import PyFamily, PyLinearLearner, BoostLssModel

def test_gaussian_fit_predict():
    # 1. Generate data
    np.random.seed(42)
    X = np.random.uniform(-3, 3, (100, 1))
    mu = 2 * X[:, 0]
    sigma = np.exp(0.5 * X[:, 0])
    y = np.random.normal(mu, sigma)

    # 2. Build model
    family = PyFamily("GaussianLSS")
    model = BoostLssModel(family, mstop=10, step_length=0.1)

    # 3. Add learners
    model.add_learner("mu", PyLinearLearner("x", intercept=True))
    model.add_learner("sigma", PyLinearLearner("x", intercept=True))

    # 4. Fit
    model.fit(X, y)

    # 5. Predict
    pred_mu = model.predict(X, "mu")
    pred_sigma = model.predict(X, "sigma")

    assert len(pred_mu) == 100
    assert len(pred_sigma) == 100
    assert not np.isnan(pred_mu).any()
    assert not np.isnan(pred_sigma).any()
```

- [x] **Step 2: Run Python integration tests**
      Run:

```bash
cd crates/boostlss-py
pip install pytest maturin
maturin develop
pytest tests/test_basic.py
```

Expected: Tests pass.

- [x] **Step 3: Commit**

```bash
git add crates/boostlss-py/tests
git commit -m "test: add python integration tests for model bindings"
```
