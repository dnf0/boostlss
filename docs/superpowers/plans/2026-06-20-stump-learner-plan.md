# Stump Learner and Abstraction Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decouple the boosting engine from Cholesky solvers and implement a Decision Stump base learner.

**Architecture:** We will introduce `LearnerFit` and `LearnerUpdate` enums, moving the Cholesky factorization into `LinearFitState`/`PSplineFitState`. The `cyclical.rs` engine will just call `fit_update` on the learner's state. We will then implement a `Stump` learner that finds optimal splits dynamically, and expose it to Python.

**Tech Stack:** Rust (`ndarray`), Python (`pyo3`).

---

### Task 1: Refactor `BaseLearner` and define `LearnerFit` / `LearnerUpdate`

**Files:**
- Modify: `crates/boostlss/src/learner/mod.rs`
- Modify: `crates/boostlss/src/error.rs`
- Modify: `crates/boostlss/src/model.rs`
- Modify: `crates/boostlss/src/engine/cyclical.rs`

- [ ] **Step 1: Write `LearnerUpdate` and `LearnerFit` enums**
In `crates/boostlss/src/learner/mod.rs`, replace the existing `LearnerFit` struct with the new enums and state structs:
```rust
use ndarray::{Array1, Array2, ArrayView1};
use faer::linalg::solvers::Llt;

#[derive(Debug, Clone)]
pub enum LearnerUpdate {
    Linear(Array1<f64>),
    Stump { split_val: f64, left_val: f64, right_val: f64 },
}

#[derive(Debug, Clone)]
pub struct LinearFitState {
    pub coef: Array1<f64>,
    pub llt: Llt<f64>,
    pub design: Array2<f64>,
}

#[derive(Debug, Clone)]
pub enum LearnerFit {
    Linear(LinearFitState),
    // Stump state will be added later
}
```

- [ ] **Step 2: Add `fit_update` and `initialize` signatures**
In `crates/boostlss/src/learner/mod.rs`, add the implementation for `LearnerFit` and modify `BaseLearner`:

```rust
impl LearnerFit {
    pub fn fit_update(&self, u: ArrayView1<f64>, _weights: Option<ArrayView1<f64>>) -> LearnerUpdate {
        match self {
            Self::Linear(state) => {
                let p = state.design.ncols();
                let xtu_nd = state.design.t().dot(&u);
                let mut xtu = faer::Mat::from_fn(p, 1, |j, _| xtu_nd[j]);
                state.llt.solve_in_place(xtu.as_mut());
                LearnerUpdate::Linear(Array1::from_shape_fn(p, |i| xtu[(i, 0)]))
            }
        }
    }
}

// Modify BaseLearner
impl BaseLearner {
    pub fn initialize(&self, x: &Array1<f64>) -> Result<LearnerFit, crate::error::BoostlssError> {
        let design = self.build_design(x)?;
        let penalty = self.penalty_matrix(design.ncols());
        let lambda = match self.target_df() {
            Some(df) => {
                let xtx = design.t().dot(&design);
                crate::learner::penalty::df_to_lambda(&xtx, &penalty, df)
            }
            None => 0.0,
        };

        let p = design.ncols();
        let xtx = design.t().dot(&design);
        let a = faer::Mat::from_fn(p, p, |j, k| xtx[[j, k]] + lambda * penalty[[j, k]]);
        let llt = faer::linalg::solvers::Llt::new(a.as_ref(), faer::Side::Lower).map_err(|_| {
            crate::error::BoostlssError::DataError(
                "Cholesky decomposition failed: matrix not positive definite".to_string(),
            )
        })?;

        Ok(LearnerFit::Linear(LinearFitState {
            coef: Array1::zeros(p),
            llt,
            design,
        }))
    }
}
```
*(Remove the old `LearnerFit::new` and `solve_update` methods).*

- [ ] **Step 3: Fix `UpdateStep` in `model.rs`**
Modify `crates/boostlss/src/model.rs` so `UpdateStep` uses `LearnerUpdate` instead of `Array1<f64>`:
```rust
use crate::learner::LearnerUpdate;

#[derive(Debug, Clone)]
pub struct UpdateStep {
    pub param_idx: usize,
    pub learner_idx: usize,
    pub update: LearnerUpdate, // changed from coef: Array1<f64>
}
```

- [ ] **Step 4: Fix `cyclical.rs` engine**
Modify `crates/boostlss/src/engine/cyclical.rs` to use the new `initialize` and `fit_update`:
```rust
// Replace cached_learners setup:
let mut cached_learners = Vec::new();
for (idx, (param_idx, learner)) in learners.iter_mut().enumerate() {
    let fit_state = learner.initialize(&x_col)?;
    cached_learners.push(CachedLearner {
        param_idx: *param_idx,
        learner_idx: idx,
        fit_state,
    });
}

// Inside the boosting loop, replace `solve_update`:
let update = cached.fit_state.fit_update(gradients.view(), data.weights());

// Reconstruct u_hat based on the update type:
let u_hat = match &update {
    LearnerUpdate::Linear(coef) => {
        if let LearnerFit::Linear(state) = &cached.fit_state {
            state.design.dot(coef)
        } else {
            unreachable!()
        }
    },
    LearnerUpdate::Stump { .. } => unreachable!(), // handle later
};

// ... later when saving the update:
updates.push(UpdateStep {
    param_idx: k,
    learner_idx: l_idx,
    update: match best_update.unwrap() {
        LearnerUpdate::Linear(coef) => LearnerUpdate::Linear(coef * nu),
        LearnerUpdate::Stump { split_val, left_val, right_val } => {
            LearnerUpdate::Stump { split_val, left_val: left_val * nu, right_val: right_val * nu }
        }
    },
});
```

- [ ] **Step 5: Fix prediction logic in `model.rs`**
Modify `crates/boostlss/src/model.rs` `predict` method to handle `LearnerUpdate`:
```rust
for step in &self.updates {
    if step.param_idx == param_idx {
        let learner = &self.learners[step.learner_idx].1;
        match &step.update {
            LearnerUpdate::Linear(coef) => {
                let x_col = data.design().column(0).to_owned();
                let design = learner.build_design(&x_col)?; // Rebuild design for new data
                pred = pred + design.dot(coef);
            },
            LearnerUpdate::Stump { split_val, left_val, right_val } => {
                let x_col = data.design().column(0);
                let u_hat = x_col.mapv(|val| if val <= *split_val { *left_val } else { *right_val });
                pred = pred + u_hat;
            }
        }
    }
}
```

- [ ] **Step 6: Run `cargo check` and fix compilation errors**
Run: `cargo check -p boostlss`
Ensure tests compile and run: `cargo test -p boostlss`
Expected: PASS

- [ ] **Step 7: Commit**
```bash
git add crates/boostlss/src/
git commit -m "refactor: abstract learner interface with LearnerFit and LearnerUpdate"
```

### Task 2: Implement `Stump` Learner

**Files:**
- Create: `crates/boostlss/src/learner/stump.rs`
- Modify: `crates/boostlss/src/learner/mod.rs`
- Modify: `crates/boostlss/src/engine/cyclical.rs`

- [ ] **Step 1: Write failing test for `Stump`**
Create `crates/boostlss/src/learner/stump.rs`:
```rust
use ndarray::{Array1, ArrayView1};
use crate::learner::LearnerUpdate;

#[derive(Debug, Clone)]
pub struct Stump {
    pub feature_name: String,
}

impl Stump {
    pub fn new(feature_name: &str) -> Self {
        Self { feature_name: feature_name.to_string() }
    }
}

#[derive(Debug, Clone)]
pub struct StumpFitState {
    pub sorted_x: Vec<(f64, usize)>,
}

impl StumpFitState {
    pub fn fit_update(&self, u: ArrayView1<f64>, weights: Option<ArrayView1<f64>>) -> LearnerUpdate {
        unimplemented!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_stump_fit() {
        let x = vec![(1.0, 0), (2.0, 1), (3.0, 2), (4.0, 3)];
        let state = StumpFitState { sorted_x: x };
        let u = array![-1.0, -1.0, 1.0, 1.0];

        let update = state.fit_update(u.view(), None);
        if let LearnerUpdate::Stump { split_val, left_val, right_val } = update {
            assert!(split_val >= 2.0 && split_val < 3.0);
            assert_eq!(left_val, -1.0);
            assert_eq!(right_val, 1.0);
        } else {
            panic!("Expected Stump update");
        }
    }
}
```

- [ ] **Step 2: Add to `mod.rs` and run test**
In `crates/boostlss/src/learner/mod.rs`:
```rust
pub mod stump;
pub use stump::Stump;

#[derive(Debug, Clone)]
pub enum BaseLearner {
    Linear(Linear),
    PSpline(PSpline),
    Stump(Stump),
}

#[derive(Debug, Clone)]
pub enum LearnerFit {
    Linear(LinearFitState),
    Stump(stump::StumpFitState),
}
```
Update `BaseLearner::initialize` to handle `Stump`:
```rust
Self::Stump(_) => {
    let mut sorted_x: Vec<(f64, usize)> = x.iter().copied().enumerate().map(|(i, val)| (val, i)).collect();
    // Use partial_cmp.unwrap() carefully, assuming no NaNs
    sorted_x.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    Ok(LearnerFit::Stump(stump::StumpFitState { sorted_x }))
}
```
(Provide dummy implementations for `build_design`, `penalty_matrix`, `target_df` for Stump returning an error or dummy values, since Stump doesn't use them).

Update `LearnerFit::fit_update` to call `stump_state.fit_update(u, weights)` for `LearnerFit::Stump(stump_state)`.

Run: `cargo test -p boostlss -- learner::stump::tests`
Expected: FAIL with `unimplemented!`

- [ ] **Step 3: Implement `StumpFitState::fit_update`**
Implement the split search.
1. Extract `w` (use 1.0 if weights is None).
2. Calculate total sum of $w$, sum of $w \cdot u$.
3. Iterate over `sorted_x`. For each step, update left sums and right sums.
4. If $x_i == x_{i+1}$, continue (only split on unique boundaries).
5. Calculate variance reduction for the split.
6. Track the best split point and return `LearnerUpdate::Stump`.

- [ ] **Step 4: Update `cyclical.rs` to handle Stump `u_hat`**
In `crates/boostlss/src/engine/cyclical.rs`, update the `u_hat` calculation:
```rust
LearnerUpdate::Stump { split_val, left_val, right_val } => {
    let x_col = data.design().column(0);
    x_col.mapv(|val| if val <= *split_val { *left_val } else { *right_val })
}
```

- [ ] **Step 5: Run tests**
Run: `cargo test -p boostlss`
Expected: PASS

- [ ] **Step 6: Commit**
```bash
git add crates/boostlss/src/
git commit -m "feat: implement stump learner"
```

### Task 3: Python Bindings for `Stump`

**Files:**
- Modify: `crates/boostlss-py/src/learner.rs`
- Modify: `crates/boostlss-py/src/lib.rs`
- Modify: `crates/boostlss-py/src/model.rs`
- Modify: `crates/boostlss-py/tests/test_basic.py`

- [ ] **Step 1: Write failing python test**
Add to `crates/boostlss-py/tests/test_basic.py`:
```python
def test_stump_learner():
    import numpy as np
    from boostlss_py import PyFamily, PyStumpLearner, BoostLssModel

    np.random.seed(42)
    X = np.random.uniform(-3, 3, (20, 2))
    y = np.random.normal(0, 1, 20)

    family = PyFamily("GaussianLSS")
    model = BoostLssModel(family, mstop=10, step_length=0.1)
    # Add a stump instead of linear learner
    model.add_learner("mu", PyStumpLearner("x"))
    model.add_learner("sigma", PyStumpLearner("x"))

    model.fit(X, y)

    pred_mu = model.predict(X, "mu")
    assert len(pred_mu) == 20
```

- [ ] **Step 2: Run test to verify it fails**
Run: `maturin develop && pytest crates/boostlss-py/tests/test_basic.py::test_stump_learner`
Expected: FAIL due to missing `PyStumpLearner`.

- [ ] **Step 3: Expose `PyStumpLearner`**
In `crates/boostlss-py/src/learner.rs`, create `PyStumpLearner`:
```rust
use boostlss::learner::{BaseLearner, Stump};

#[pyclass]
#[derive(Clone)]
pub struct PyStumpLearner {
    pub name: String,
}

#[pymethods]
impl PyStumpLearner {
    #[new]
    fn new(name: String) -> Self {
        Self { name }
    }
}

impl From<PyStumpLearner> for BaseLearner {
    fn from(val: PyStumpLearner) -> Self {
        BaseLearner::Stump(Stump::new(&val.name))
    }
}
```
Update `crates/boostlss-py/src/lib.rs` to register the class:
```rust
use learner::PyStumpLearner;
m.add_class::<PyStumpLearner>()?;
```

- [ ] **Step 4: Update `add_learner` to extract `PyStumpLearner`**
In `crates/boostlss-py/src/model.rs`, modify the `add_learner` method so it attempts to extract `PyStumpLearner` if extracting `PyLinearLearner` fails:
```rust
// Replace current extraction logic with:
let base_learner = if let Ok(l) = learner.extract::<PyLinearLearner>() {
    l.into()
} else if let Ok(s) = learner.extract::<PyStumpLearner>() {
    s.into()
} else {
    return Err(pyo3::exceptions::PyValueError::new_err("Invalid learner type"));
};
```

- [ ] **Step 5: Run test to verify it passes**
Run: `maturin develop && pytest crates/boostlss-py/tests/test_basic.py::test_stump_learner`
Expected: PASS

- [ ] **Step 6: Commit**
```bash
git add crates/boostlss-py/
git commit -m "feat: expose stump learner to python"
```
