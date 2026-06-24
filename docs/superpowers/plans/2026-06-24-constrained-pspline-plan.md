# Constrained PSpline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `ConstrainedPSpline` base-learner for monotonic and convex/concave constraints.

**Architecture:** We extract the B-spline basis generation from `PSpline` into a shared `spline_utils` module. Then we create `ConstrainedPSpline` which uses an Iteratively Reweighted Least Squares (IRLS) inner loop inside `fit_update` to dynamically re-weight the penalty matrix based on constraint violations.

**Tech Stack:** Rust, ndarray, faer

---

### Task 1: Refactor B-Spline Basis Generation

**Files:**
- Create: `crates/boostlss/src/learner/spline_utils.rs`
- Modify: `crates/boostlss/src/learner/mod.rs`
- Modify: `crates/boostlss/src/learner/pspline.rs`

- [ ] **Step 1: Create `spline_utils.rs` and move B-spline logic**

Move the Cox-de Boor recursion and `build_design` logic from `PSpline` into a shared function.

```rust
// crates/boostlss/src/learner/spline_utils.rs
use crate::error::BoostlssError;
use ndarray::{Array1, Array2};

pub struct SplineData {
    pub min_val: f64,
    pub max_val: f64,
    pub t: Vec<f64>,
}

pub fn build_bspline_design(
    x: &Array1<f64>,
    knots: usize,
    degree: usize,
    spline_data: &mut Option<SplineData>,
) -> Result<Array2<f64>, BoostlssError> {
    if spline_data.is_none() {
        let min_val = x.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = x.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        let num_knots = knots + 2 * degree + 2;
        let mut t = vec![0.0; num_knots];
        let step = (max_val - min_val) / (knots as f64 - 1.0 + 1e-9);

        for (i, t_val) in t.iter_mut().enumerate() {
            *t_val = min_val + (i as f64 - degree as f64) * step;
        }

        *spline_data = Some(SplineData {
            min_val,
            max_val,
            t,
        });
    }

    let data = spline_data.as_ref().unwrap();
    let num_knots = data.t.len();
    let n = x.len();
    let p = knots + degree + 1;
    let mut b = Array2::zeros((n, p));

    let mut n_basis = vec![0.0; num_knots - 1];
    let mut next_n = vec![0.0; num_knots - 1];

    for i in 0..n {
        let xi = x[i];
        if xi < data.min_val || xi > data.max_val {
            return Err(BoostlssError::OutOfRange(format!(
                "Value {} out of training range",
                xi
            )));
        }

        n_basis.fill(0.0);
        for j in 0..num_knots - 1 {
            if data.t[j] <= xi && xi < data.t[j + 1] {
                n_basis[j] = 1.0;
            }
        }
        if xi == data.t[num_knots - 1] {
            n_basis[num_knots - 2] = 1.0;
        }

        for d in 1..=degree {
            next_n.fill(0.0);
            for j in 0..num_knots - 1 - d {
                let left_den = data.t[j + d] - data.t[j];
                let left = if left_den > 0.0 {
                    (xi - data.t[j]) / left_den * n_basis[j]
                } else {
                    0.0
                };

                let right_den = data.t[j + d + 1] - data.t[j + 1];
                let right = if right_den > 0.0 {
                    (data.t[j + d + 1] - xi) / right_den * n_basis[j + 1]
                } else {
                    0.0
                };
                next_n[j] = left + right;
            }
            n_basis.copy_from_slice(&next_n);
        }

        for j in 0..p {
            b[[i, j]] = n_basis[j];
        }
    }

    Ok(b)
}
```

- [ ] **Step 2: Expose module in `learner/mod.rs`**

```rust
// crates/boostlss/src/learner/mod.rs
pub mod spline_utils;
```

- [ ] **Step 3: Update `PSpline` to use the utility**

```rust
// In crates/boostlss/src/learner/pspline.rs
use crate::learner::spline_utils::{build_bspline_design, SplineData};

// Replace min_val, max_val, t with Option<SplineData>
pub struct PSpline {
    pub feature_idx: usize,
    pub knots: usize,
    pub degree: usize,
    pub differences: usize,
    pub is_cyclic: bool,
    pub df: f64,
    pub spline_data: Option<SplineData>,
}

// Update new()
pub fn new(feature_idx: usize) -> Self {
    Self {
        feature_idx,
        knots: 20,
        degree: 3,
        differences: 2,
        is_cyclic: false,
        df: 4.0,
        spline_data: None,
    }
}

// Update build_design
pub fn build_design(
    &mut self,
    data: &crate::data::Dataset,
) -> Result<Array2<f64>, BoostlssError> {
    let x = data.design().column(self.feature_idx);
    let mut b = build_bspline_design(&x.to_owned(), self.knots, self.degree, &mut self.spline_data)?;

    if self.is_cyclic {
        let p = self.knots + self.degree + 1;
        let c = self.knots - self.degree;
        let mut b_cyclic = Array2::zeros((b.nrows(), c));

        for i in 0..b.nrows() {
            for j in 0..c {
                b_cyclic[[i, j]] = b[[i, j]];
            }
            for j in c..p {
                b_cyclic[[i, j - c]] += b[[i, j]];
            }
        }
        return Ok(b_cyclic);
    }

    Ok(b)
}
```

- [ ] **Step 4: Run tests to verify the refactor didn't break anything**
Run `cargo test -p boostlss`

- [ ] **Step 5: Commit**

### Task 2: Create `ConstrainedPSpline` and its State

**Files:**
- Create: `crates/boostlss/src/learner/constrained_pspline.rs`
- Modify: `crates/boostlss/src/learner/mod.rs`

- [ ] **Step 1: Create `constrained_pspline.rs` with structs**

```rust
// crates/boostlss/src/learner/constrained_pspline.rs
use crate::error::BoostlssError;
use crate::learner::penalty::{difference_matrix, penalty_matrix};
use crate::learner::spline_utils::{build_bspline_design, SplineData};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use faer::linalg::solvers::Llt;
use faer::prelude::Solve;
use faer::Mat;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Constraint {
    MonotonicIncreasing,
    MonotonicDecreasing,
    Convex,
    Concave,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstrainedPSpline {
    pub feature_idx: usize,
    pub knots: usize,
    pub degree: usize,
    pub differences: usize,
    pub df: f64,
    pub constraint: Constraint,
    pub max_iter: usize,
    pub tolerance: f64,
    pub spline_data: Option<SplineData>,
}

impl ConstrainedPSpline {
    pub fn new(feature_idx: usize, constraint: Constraint) -> Self {
        Self {
            feature_idx,
            knots: 20,
            degree: 3,
            differences: 2, // Smoothness differences
            df: 4.0,
            constraint,
            max_iter: 10,
            tolerance: 1e-6,
            spline_data: None,
        }
    }

    // Builder methods: with_knots, with_degree, with_df, with_max_iter, with_tolerance...
    pub fn with_df(mut self, df: f64) -> Self { self.df = df; self }
    pub fn with_max_iter(mut self, max_iter: usize) -> Self { self.max_iter = max_iter; self }

    pub fn build_design(&mut self, data: &crate::data::Dataset) -> Result<Array2<f64>, BoostlssError> {
        let x = data.design().column(self.feature_idx);
        build_bspline_design(&x.to_owned(), self.knots, self.degree, &mut self.spline_data)
    }

    pub fn penalty_matrix(&self, n_cols: usize) -> Array2<f64> {
        penalty_matrix(n_cols, self.differences, false)
    }
}

// State used during IRLS
#[derive(Debug, Clone)]
pub struct ConstrainedFitState {
    pub xtx: Array2<f64>,
    pub design: Array2<f64>,
    pub smooth_penalty: Array2<f64>,
    pub lambda_smooth: f64,
    pub constraint_diff: Array2<f64>, // D matrix for constraint
    pub constraint: Constraint,
    pub max_iter: usize,
    pub tolerance: f64,
}
```

- [ ] **Step 2: Add IRLS fit_update implementation**

```rust
// Add to constrained_pspline.rs
use ndarray::ArrayView1;
use crate::learner::LearnerUpdate;

impl ConstrainedFitState {
    pub fn fit_update(&self, u: ArrayView1<f64>, weights: Option<ArrayView1<f64>>) -> LearnerUpdate {
        let p = self.design.ncols();
        let kappa = 1e6; // Large penalty for violating constraints

        // Initial unconstrained solve
        let xtu_nd = self.design.t().dot(&u); // simplified without weights for brevity, add weight support in actual implementation
        let mut xtu = Mat::from_fn(p, 1, |j, _| xtu_nd[j]);
        let mut a = Mat::from_fn(p, p, |j, k| self.xtx[[j, k]] + self.lambda_smooth * self.smooth_penalty[[j, k]]);

        let mut llt = Llt::new(a.as_ref(), faer::Side::Lower).unwrap();
        let mut beta_faer = llt.solve(xtu.as_ref());
        let mut beta = Array1::from_shape_fn(p, |i| beta_faer[(i, 0)]);

        for _ in 0..self.max_iter {
            let mut v = Array1::zeros(self.constraint_diff.nrows());
            let diffs = self.constraint_diff.dot(&beta);

            for i in 0..v.len() {
                v[i] = match self.constraint {
                    Constraint::MonotonicIncreasing => if diffs[i] < 0.0 { 1.0 } else { 0.0 },
                    Constraint::MonotonicDecreasing => if diffs[i] > 0.0 { 1.0 } else { 0.0 },
                    Constraint::Convex => if diffs[i] < 0.0 { 1.0 } else { 0.0 },
                    Constraint::Concave => if diffs[i] > 0.0 { 1.0 } else { 0.0 },
                };
            }

            // Check if constraints are met
            if v.sum() == 0.0 {
                break;
            }

            // V_mat = D^T * diag(v) * D
            let mut dt_v_d = Array2::zeros((p, p));
            for i in 0..self.constraint_diff.nrows() {
                if v[i] > 0.0 {
                    let row = self.constraint_diff.row(i);
                    for j in 0..p {
                        for k in 0..p {
                            dt_v_d[[j, k]] += row[j] * row[k];
                        }
                    }
                }
            }

            a = Mat::from_fn(p, p, |j, k| {
                self.xtx[[j, k]] + self.lambda_smooth * self.smooth_penalty[[j, k]] + kappa * dt_v_d[[j, k]]
            });

            llt = Llt::new(a.as_ref(), faer::Side::Lower).unwrap();
            beta_faer = llt.solve(xtu.as_ref());

            let mut max_diff = 0.0;
            let mut next_beta = Array1::zeros(p);
            for i in 0..p {
                next_beta[i] = beta_faer[(i, 0)];
                let diff = (next_beta[i] - beta[i]).abs();
                if diff > max_diff { max_diff = diff; }
            }

            beta = next_beta;
            if max_diff < self.tolerance {
                break;
            }
        }

        LearnerUpdate::Linear(beta)
    }
}
```
*(Make sure to properly handle `weights` in `xtu` and `xtx` during implementation!)*

- [ ] **Step 3: Expose and Register**

In `crates/boostlss/src/learner/mod.rs`:
- `pub mod constrained_pspline;`
- Add to `BaseLearner` enum: `ConstrainedPSpline(constrained_pspline::ConstrainedPSpline)`
- Add to `LearnerFit` enum: `ConstrainedPSpline(constrained_pspline::ConstrainedFitState)`
- Update match arms in `BaseLearner::build_design`, `penalty_matrix`, `target_df`, `initialize`, and `LearnerFit::fit_update`, `predict_update`.

In `initialize`, construct the `ConstrainedFitState` with:
- `lambda_smooth` via `df_to_lambda`.
- `constraint_diff` via `difference_matrix` (order 1 for monotonic, 2 for convex/concave).

- [ ] **Step 4: Commit**

### Task 3: Python Bindings and Tests

**Files:**
- Modify: `crates/boostlss-py/src/learner.rs`
- Modify: `crates/boostlss-py/src/lib.rs`
- Modify: `tests/test_constrained.py` (New file)

- [ ] **Step 1: Expose `constrained_pspline` in Python**

```rust
// In crates/boostlss-py/src/learner.rs
use boostlss::learner::constrained_pspline::{ConstrainedPSpline as CoreConstrained, Constraint};

#[pyfunction]
#[pyo3(signature = (feature_idx, constraint, knots=20, degree=3, differences=2, df=4.0, max_iter=10, tolerance=1e-6))]
pub fn constrained_pspline(
    feature_idx: usize,
    constraint: &str,
    knots: usize,
    degree: usize,
    differences: usize,
    df: f64,
    max_iter: usize,
    tolerance: f64,
) -> PyResult<PyLearner> {
    let c = match constraint.to_lowercase().as_str() {
        "monotonic_increasing" => Constraint::MonotonicIncreasing,
        "monotonic_decreasing" => Constraint::MonotonicDecreasing,
        "convex" => Constraint::Convex,
        "concave" => Constraint::Concave,
        _ => return Err(pyo3::exceptions::PyValueError::new_err("Invalid constraint")),
    };

    let mut b = CoreConstrained::new(feature_idx, c);
    b.knots = knots;
    b.degree = degree;
    b.differences = differences;
    b.df = df;
    b.max_iter = max_iter;
    b.tolerance = tolerance;

    Ok(PyLearner { inner: b.into() })
}
```

- [ ] **Step 2: Register pyfunction in `lib.rs`**

```rust
m.add_function(wrap_pyfunction!(learner::constrained_pspline, m)?)?;
```

- [ ] **Step 3: Write python tests**

```python
# tests/test_constrained.py
import pytest
import numpy as np
from boostlss import BoostLSS, constrained_pspline

def test_monotonic_increasing():
    np.random.seed(42)
    # Underlying function is y = -x (decreasing)
    x = np.sort(np.random.uniform(-3, 3, 100))
    y = -x + np.random.normal(0, 0.1, 100)

    # Fit with increasing constraint
    model = BoostLSS(family="gaussian", step_length=0.1)
    model.on("mu", constrained_pspline(0, "monotonic_increasing", df=2))

    X = x.reshape(-1, 1)
    model.fit(X, y)

    # Predictions should be strictly monotonically increasing (or flat)
    preds = model.predict(X, "mu")
    diffs = np.diff(preds)
    assert np.all(diffs >= -1e-6), "Predictions are not monotonically increasing"

def test_monotonic_decreasing():
    np.random.seed(42)
    x = np.sort(np.random.uniform(-3, 3, 100))
    y = x + np.random.normal(0, 0.1, 100) # True is increasing

    model = BoostLSS(family="gaussian", step_length=0.1)
    model.on("mu", constrained_pspline(0, "monotonic_decreasing", df=2))

    X = x.reshape(-1, 1)
    model.fit(X, y)

    preds = model.predict(X, "mu")
    diffs = np.diff(preds)
    assert np.all(diffs <= 1e-6), "Predictions are not monotonically decreasing"
```

- [ ] **Step 4: Run tests to verify**
Run `uv run pytest tests/test_constrained.py`

- [ ] **Step 5: Commit**
