# Bivariate Tensor-Product P-splines (`bspatial`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the Bivariate Tensor-Product P-splines (bspatial) base learner for spatial data smoothing.

**Architecture:** We will create a `BivariatePSpline` struct in `crates/boostlss/src/learner/bspatial.rs` that takes two feature names. Internally, it instantiates two 1D `PSpline` structs, evaluates their marginal bases, and combines them using row-wise Kronecker products to create the dense 2D design matrix. The penalty matrix is computed as the Kronecker sum $K_1 \otimes I + I \otimes K_2$. Python bindings will be added to expose `PyBivariatePSplineLearner`.

**Tech Stack:** Rust (2021 edition), `ndarray`, PyO3, Python.

---

### Task 1: Kronecker Product Helper and Core `BivariatePSpline` Struct

**Files:**
- Create: `crates/boostlss/src/learner/bspatial.rs`

- [ ] **Step 1: Write the failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_kronecker_product() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[0.5, 2.0], [3.0, 1.0]];
        let res = kronecker_product(&a, &b);
        let expected = array![
            [0.5, 2.0, 1.0, 4.0],
            [3.0, 1.0, 6.0, 2.0],
            [1.5, 6.0, 2.0, 8.0],
            [9.0, 3.0, 12.0, 4.0]
        ];
        assert_eq!(res, expected);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --package boostlss kronecker -- --nocapture`
Expected: FAIL due to missing file / module. Note: you might need to add `pub mod bspatial;` to `crates/boostlss/src/learner/mod.rs` first for the tests to be discovered, but wait until step 3 to fully flesh out the `mod.rs`. Just create the file and the implementation.

- [ ] **Step 3: Write minimal implementation**

Create `crates/boostlss/src/learner/bspatial.rs` and add the `kronecker_product` function and the struct definition:

```rust
use crate::data::Dataset;
use crate::error::BoostlssError;
use crate::learner::{BaseLearnerImpl, LearnerFit};
use crate::learner::pspline::PSpline;
use ndarray::{Array2, s, Axis};

#[derive(Clone, Debug, PartialEq)]
pub struct BivariatePSpline {
    pub feature1: String,
    pub feature2: String,
    pub knots: usize,
    pub degree: usize,
    pub differences: usize,
    pub df: f64,
}

impl BivariatePSpline {
    pub fn new(feature1: &str, feature2: &str) -> Self {
        Self {
            feature1: feature1.to_string(),
            feature2: feature2.to_string(),
            knots: 20,
            degree: 3,
            differences: 2,
            df: 4.0,
        }
    }

    pub fn knots(mut self, knots: usize) -> Self { self.knots = knots; self }
    pub fn degree(mut self, degree: usize) -> Self { self.degree = degree; self }
    pub fn differences(mut self, differences: usize) -> Self { self.differences = differences; self }
    pub fn df(mut self, df: f64) -> Self { self.df = df; self }
}

/// Computes the full Kronecker product of two 2D matrices
pub fn kronecker_product(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let (a_rows, a_cols) = a.dim();
    let (b_rows, b_cols) = b.dim();
    let mut res = Array2::zeros((a_rows * b_rows, a_cols * b_cols));
    for i in 0..a_rows {
        for j in 0..a_cols {
            let val = a[[i, j]];
            let mut slice = res.slice_mut(s![i * b_rows..(i + 1) * b_rows, j * b_cols..(j + 1) * b_cols]);
            slice.assign(&(b * val));
        }
    }
    res
}
```

- [ ] **Step 4: Update `mod.rs` to include `bspatial` so tests are found**
Modify `crates/boostlss/src/learner/mod.rs`, add `pub mod bspatial;`.

- [ ] **Step 5: Run test to verify it passes**

Run: `cargo test --package boostlss kronecker -- --nocapture`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add crates/boostlss/src/learner/bspatial.rs crates/boostlss/src/learner/mod.rs
git commit -m "feat: add bspatial struct and kronecker helper"
```

---

### Task 2: Implement `BaseLearnerImpl` for `BivariatePSpline`

**Files:**
- Modify: `crates/boostlss/src/learner/bspatial.rs`
- Modify: `crates/boostlss/src/learner/mod.rs`

- [ ] **Step 1: Write the failing test**

In `crates/boostlss/src/learner/bspatial.rs`, add:
```rust
#[cfg(test)]
mod tests2 {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_bspatial_fit_predict() {
        let mut ds = Dataset::new();
        ds.add_column("x1", Array1::linspace(0., 1., 100)).unwrap();
        ds.add_column("x2", Array1::linspace(0., 1., 100)).unwrap();
        ds.set_response(Array1::linspace(0., 1., 100)).unwrap();

        let learner = BivariatePSpline::new("x1", "x2").knots(5);
        let mut u = Array1::ones(100);

        // This will panic if unimplemented
        let fit = learner.fit(&ds, &u, None).unwrap();

        let pred = fit.predict(&ds).unwrap();
        assert_eq!(pred.len(), 100);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --package boostlss bspatial_fit_predict -- --nocapture`
Expected: FAIL (compilation error, since `fit` and `BaseLearnerImpl` are not implemented for `BivariatePSpline`).

- [ ] **Step 3: Write minimal implementation**

In `crates/boostlss/src/learner/bspatial.rs`, implement `BaseLearnerImpl`:

```rust
impl BaseLearnerImpl for BivariatePSpline {
    fn fit(&self, data: &Dataset, u: &ndarray::Array1<f64>, w: Option<&ndarray::Array1<f64>>) -> Result<LearnerFit, BoostlssError> {
        let col1 = data.column(&self.feature1)?;
        let col2 = data.column(&self.feature2)?;

        let mut p1 = PSpline::new(&self.feature1)
            .knots(self.knots)
            .degree(self.degree)
            .differences(self.differences);
        // We use df=1.0 just to get penalty matrices, actual scaling done later
        p1 = p1.df(1.0);
        let mut p2 = PSpline::new(&self.feature2)
            .knots(self.knots)
            .degree(self.degree)
            .differences(self.differences);
        p2 = p2.df(1.0);

        let b1 = p1.build_design(col1)?;
        let b2 = p2.build_design(col2)?;

        let n_obs = b1.nrows();
        let p_cols1 = b1.ncols();
        let p_cols2 = b2.ncols();

        // Row-wise Kronecker product for design matrix
        let mut design = Array2::zeros((n_obs, p_cols1 * p_cols2));
        for i in 0..n_obs {
            let row1 = b1.row(i);
            let row2 = b2.row(i);
            for j in 0..p_cols1 {
                for k in 0..p_cols2 {
                    design[[i, j * p_cols2 + k]] = row1[j] * row2[k];
                }
            }
        }

        // Marginal penalties
        let k1 = crate::learner::penalty::penalty_matrix(p_cols1, self.differences);
        let k2 = crate::learner::penalty::penalty_matrix(p_cols2, self.differences);

        // Kronecker sum for total penalty: K = K1 x I + I x K2
        let i1 = Array2::<f64>::eye(p_cols1);
        let i2 = Array2::<f64>::eye(p_cols2);
        let penalty = kronecker_product(&k1, &i2) + kronecker_product(&i1, &k2);

        // Compute lambda for the combined penalty
        let lambda = crate::learner::penalty::compute_lambda(&design, &penalty, self.df, w)?;

        // Delegate to LearnerFit
        LearnerFit::new(design, lambda * penalty, u, w)
    }

    fn name(&self) -> String {
        format!("BivariatePSpline({},{})", self.feature1, self.feature2)
    }

    fn predict(&self, data: &Dataset, coef: &ndarray::Array1<f64>) -> Result<ndarray::Array1<f64>, BoostlssError> {
        let col1 = data.column(&self.feature1)?;
        let col2 = data.column(&self.feature2)?;

        let p1 = PSpline::new(&self.feature1)
            .knots(self.knots)
            .degree(self.degree)
            .differences(self.differences);
        let p2 = PSpline::new(&self.feature2)
            .knots(self.knots)
            .degree(self.degree)
            .differences(self.differences);

        let b1 = p1.build_design(col1)?;
        let b2 = p2.build_design(col2)?;

        let n_obs = b1.nrows();
        let p_cols1 = b1.ncols();
        let p_cols2 = b2.ncols();

        let mut design = Array2::zeros((n_obs, p_cols1 * p_cols2));
        for i in 0..n_obs {
            let row1 = b1.row(i);
            let row2 = b2.row(i);
            for j in 0..p_cols1 {
                for k in 0..p_cols2 {
                    design[[i, j * p_cols2 + k]] = row1[j] * row2[k];
                }
            }
        }

        Ok(design.dot(coef))
    }
}
```

- [ ] **Step 4: Update `BaseLearner` Enum**
Modify `crates/boostlss/src/learner/mod.rs`. Add `BivariatePSpline(bspatial::BivariatePSpline)` to the `BaseLearner` enum.
Also add the matching arm in the `impl BaseLearnerImpl for BaseLearner` block for `fit`, `predict`, and `name`.

- [ ] **Step 5: Run test to verify it passes**

Run: `cargo test --package boostlss bspatial_fit_predict -- --nocapture`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add crates/boostlss/src/learner/bspatial.rs crates/boostlss/src/learner/mod.rs
git commit -m "feat: implement BaseLearnerImpl for BivariatePSpline"
```

---

### Task 3: Python Bindings

**Files:**
- Modify: `crates/boostlss-py/src/learner.rs`
- Modify: `crates/boostlss-py/src/lib.rs`
- Create: `crates/boostlss-py/tests/test_bspatial.py`

- [ ] **Step 1: Write the failing test**

Create `crates/boostlss-py/tests/test_bspatial.py`:
```python
import numpy as np
from boostlss_py import BoostLSS, bspatial, Dataset

def test_bspatial_learner():
    x1 = np.linspace(0, 1, 100)
    x2 = np.linspace(0, 1, 100)
    y = x1 + x2 + np.random.normal(0, 0.1, 100)

    ds = Dataset()
    ds.add_column("x1", x1)
    ds.add_column("x2", x2)
    ds.set_response(y)

    learner = bspatial("x1", "x2", knots=5)

    # Simple check that it builds and is a valid learner
    model = BoostLSS(family="gaussian", step_length=0.1, algorithm="noncyclic")
    model.on("mu", learner)

    model.fit(ds)
    pred = model.predict(ds, param="mu", scale="response")
    assert len(pred) == 100
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv pip install -e .` in `crates/boostlss-py`
Run: `pytest crates/boostlss-py/tests/test_bspatial.py`
Expected: FAIL (NameError: name 'bspatial' is not defined)

- [ ] **Step 3: Write minimal implementation**

Modify `crates/boostlss-py/src/learner.rs` to expose `PyBivariatePSplineLearner`.

```rust
use boostlss::learner::bspatial::BivariatePSpline;

#[pyclass(name = "BivariatePSplineLearner", module = "boostlss_py")]
#[derive(Clone)]
pub struct PyBivariatePSplineLearner {
    pub inner: BaseLearner,
}

#[pymethods]
impl PyBivariatePSplineLearner {
    #[new]
    #[pyo3(signature = (feature1, feature2, knots=20, degree=3, differences=2, df=4.0))]
    fn new(feature1: &str, feature2: &str, knots: usize, degree: usize, differences: usize, df: f64) -> Self {
        let learner = BivariatePSpline::new(feature1, feature2)
            .knots(knots)
            .degree(degree)
            .differences(differences)
            .df(df);
        Self {
            inner: BaseLearner::BivariatePSpline(learner),
        }
    }
}

// Add a helper Python function for easy instantiation (like pspline, linear)
#[pyfunction]
#[pyo3(signature = (feature1, feature2, knots=20, degree=3, differences=2, df=4.0))]
pub fn bspatial(feature1: &str, feature2: &str, knots: usize, degree: usize, differences: usize, df: f64) -> PyBivariatePSplineLearner {
    PyBivariatePSplineLearner::new(feature1, feature2, knots, degree, differences, df)
}
```

In `crates/boostlss-py/src/lib.rs`, add `bspatial` to the module exports:
```rust
m.add_function(wrap_pyfunction!(learner::bspatial, m)?)?;
m.add_class::<learner::PyBivariatePSplineLearner>()?;
```
Also update `__all__` in `boostlss-py/python/boostlss_py/__init__.py` if it exists, or just ensure it is exported correctly.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv pip install -e .` in `crates/boostlss-py`
Run: `pytest crates/boostlss-py/tests/test_bspatial.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/boostlss-py/
git commit -m "feat(python): expose bspatial learner"
```

---

## Self-Review completed. No placeholders found. Ready for handoff.
