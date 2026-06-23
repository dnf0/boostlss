# Random Effects Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a Random Effects base-learner (`RandomEffects`) that takes a single column of 0-indexed group integers, dynamically expanding it into an $N \times C$ dummy matrix with an identity ridge penalty.

**Architecture:** Create `random_effects.rs` inside `crates/boostlss/src/learner/`, integrate it into `BaseLearner` and `LearnerUpdate`, and expose a `PyRandomEffectsLearner` in `boostlss-py`.

**Tech Stack:** Rust (`ndarray`, `faer`), Python (`pyo3`).

---

### Task 1: Create the RandomEffects Learner Core

**Files:**
- Create: `crates/boostlss/src/learner/random_effects.rs`
- Modify: `crates/boostlss/src/learner/mod.rs`

- [ ] **Step 1: Write the failing tests**

Create `crates/boostlss/src/learner/random_effects.rs`:

```rust
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use crate::error::BoostlssError;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomEffects {
    pub feature: String,
    pub df: f64,
}

impl RandomEffects {
    pub fn new(feature: &str) -> Self {
        Self {
            feature: feature.to_string(),
            df: 4.0,
        }
    }

    pub fn df(mut self, df: f64) -> Self {
        self.df = df;
        self
    }

    pub fn build_design(&self, x: &Array1<f64>) -> Result<Array2<f64>, BoostlssError> {
        let n_obs = x.len();
        if n_obs == 0 {
            return Ok(Array2::zeros((0, 0)));
        }

        let mut max_idx = 0;
        for &val in x.iter() {
            if val.fract() != 0.0 || val < 0.0 {
                return Err(BoostlssError::DataError(
                    "RandomEffects requires non-negative integer indices".to_string(),
                ));
            }
            let idx = val as usize;
            if idx > max_idx {
                max_idx = idx;
            }
        }

        let n_cols = max_idx + 1;
        let mut design = Array2::zeros((n_obs, n_cols));

        for (i, &val) in x.iter().enumerate() {
            let idx = val as usize;
            design[[i, idx]] = 1.0;
        }

        Ok(design)
    }

    pub fn penalty_matrix(&self, n_cols: usize) -> Array2<f64> {
        Array2::eye(n_cols)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_random_effects_design() {
        let re = RandomEffects::new("group");
        let x = array![0.0, 2.0, 1.0, 0.0];
        let design = re.build_design(&x).unwrap();

        assert_eq!(design.shape(), &[4, 3]);
        assert_eq!(design.row(0), array![1.0, 0.0, 0.0].view());
        assert_eq!(design.row(1), array![0.0, 0.0, 1.0].view());
        assert_eq!(design.row(2), array![0.0, 1.0, 0.0].view());
        assert_eq!(design.row(3), array![1.0, 0.0, 0.0].view());
    }

    #[test]
    fn test_random_effects_invalid_data() {
        let re = RandomEffects::new("group");
        assert!(re.build_design(&array![-1.0, 0.0]).is_err());
        assert!(re.build_design(&array![0.5, 1.0]).is_err());
    }

    #[test]
    fn test_random_effects_penalty() {
        let re = RandomEffects::new("group");
        let pen = re.penalty_matrix(3);
        assert_eq!(pen.shape(), &[3, 3]);
        assert_eq!(pen.diag(), array![1.0, 1.0, 1.0].view());
    }
}
```

Modify `crates/boostlss/src/learner/mod.rs` to expose the module and add the enum variant.

Add to top:
```rust
pub mod random_effects;
pub use random_effects::RandomEffects;
```

Update `BaseLearner` enum:
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BaseLearner {
    Linear(Linear),
    PSpline(PSpline),
    Stump(Stump),
    Tree(Tree),
    RandomEffects(RandomEffects),
}
```

Update `From` implementations:
```rust
impl From<RandomEffects> for BaseLearner {
    fn from(r: RandomEffects) -> Self {
        Self::RandomEffects(r)
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p boostlss -- test_random_effects`
Expected: Fail to compile due to missing enum match arms in `BaseLearner`.

- [ ] **Step 3: Update `BaseLearner` methods to handle `RandomEffects`**

Modify `crates/boostlss/src/learner/mod.rs` to update `build_design`, `penalty_matrix`, and `target_df`:

```rust
    pub fn build_design(
        &mut self,
        x: &Array1<f64>,
    ) -> Result<Array2<f64>, crate::error::BoostlssError> {
        match self {
            Self::Linear(l) => l.build_design(x),
            Self::PSpline(p) => p.build_design(x),
            Self::RandomEffects(r) => r.build_design(x),
            Self::Stump(_) => Err(crate::error::BoostlssError::DataError(
                "Stump does not use build_design".into(),
            )),
            Self::Tree(_) => Err(crate::error::BoostlssError::DataError(
                "Tree does not use build_design".into(),
            )),
        }
    }

    pub fn penalty_matrix(&self, n_cols: usize) -> Array2<f64> {
        match self {
            Self::Linear(l) => l.penalty_matrix(n_cols),
            Self::PSpline(p) => p.penalty_matrix(n_cols),
            Self::RandomEffects(r) => r.penalty_matrix(n_cols),
            Self::Stump(_) => Array2::zeros((0, 0)),
            Self::Tree(_) => Array2::zeros((0, 0)),
        }
    }

    pub fn target_df(&self) -> Option<f64> {
        match self {
            Self::Linear(_) => None,
            Self::PSpline(p) => Some(p.df),
            Self::RandomEffects(r) => Some(r.df),
            Self::Stump(_) => None,
            Self::Tree(_) => None,
        }
    }
```

Add tests to `mod.rs`:
```rust
    #[test]
    fn test_from_impls_random_effects() {
        let r = RandomEffects::new("x");
        let bl: BaseLearner = r.into();
        assert!(matches!(bl, BaseLearner::RandomEffects(_)));
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p boostlss -- test_random_effects`
Run: `cargo check`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/boostlss/src/learner/random_effects.rs crates/boostlss/src/learner/mod.rs
git commit -m "feat: implement RandomEffects base-learner core"
```

---

### Task 2: Prediction Logic for Random Effects

**Files:**
- Modify: `crates/boostlss/src/model.rs`

- [ ] **Step 1: Write the failing tests**

Modify `crates/boostlss/src/model.rs` by adding a test for RandomEffects predict out of bounds behavior:

```rust
    #[test]
    fn test_predict_random_effects_out_of_bounds() {
        use crate::learner::RandomEffects;
        let mut model = BoostLss::new(GaussianLss::new())
            .on("mu", |p| p.add(RandomEffects::new("x")))
            .unwrap()
            .step_length(0.1)
            .mstop(Mstop::Scalar(1));

        let x_train = ndarray::array![0.0, 1.0, 0.0, 1.0];
        let y_train = ndarray::array![10.0, 20.0, 10.0, 20.0];
        let ds_train = Dataset::new(
            ndarray::Array2::from_shape_vec((4, 1), x_train.to_vec()).unwrap(),
            y_train,
            None
        ).unwrap();

        let fitted = model.fit(&ds_train).unwrap();

        // Predict on unseen group index (2.0) and negative index (-1.0)
        let x_test = ndarray::array![0.0, 2.0, -1.0];
        let ds_test = Dataset::new(
            ndarray::Array2::from_shape_vec((3, 1), x_test.to_vec()).unwrap(),
            ndarray::Array1::zeros(3),
            None
        ).unwrap();

        let preds = fitted.predict(&ds_test, "mu", Scale::Link).unwrap();

        // Unseen/invalid groups should get exactly 0.0 addition to the intercept
        // First element is seen, so it has a non-intercept effect.
        // Elements 1 and 2 are unseen, so they should equal the global offset exactly.
        let offset = fitted.offset(0);
        assert!((preds[1] - offset).abs() < 1e-10);
        assert!((preds[2] - offset).abs() < 1e-10);
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p boostlss -- test_predict_random_effects`
Expected: Compile error on missing enum arm `LearnerUpdate::RandomEffects` or execution failure if unhandled.

- [ ] **Step 3: Write minimal implementation**

Wait, `RandomEffects` currently falls under the `LearnerFit::Linear` state during initialization, which produces a `LearnerUpdate::Linear(Array1<f64>)`. We do not need a new update state for it!

However, we need to handle out-of-bounds group indices during the *prediction* phase. The design matrix built during *predict* might not match the size of the coefficient vector if the test data has a higher maximum index than the training data!

Modify `crates/boostlss/src/model.rs` around line 282, where we match on `LearnerUpdate`:

```rust
                LearnerUpdate::Linear(coef) => {
                    let mut design = match learner {
                        BaseLearner::Linear(l) => l.build_design(x)?,
                        BaseLearner::PSpline(p) => p.build_predict_design(x)?,
                        BaseLearner::RandomEffects(r) => {
                            // Custom predict design for RandomEffects:
                            // We need exactly coef.len() columns to match the training groups.
                            let mut d = ndarray::Array2::zeros((x.len(), coef.len()));
                            for (i, &val) in x.iter().enumerate() {
                                // If val is negative or out of bounds (unseen group),
                                // we leave the row as all zeros.
                                if val >= 0.0 && val.fract() == 0.0 {
                                    let idx = val as usize;
                                    if idx < coef.len() {
                                        d[[i, idx]] = 1.0;
                                    }
                                }
                            }
                            d
                        }
                        _ => unreachable!(),
                    };
                    eta = eta + design.dot(coef);
                }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p boostlss -- test_predict_random_effects`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/boostlss/src/model.rs
git commit -m "feat: implement RandomEffects out-of-bounds predict logic"
```

---

### Task 3: Python Bindings for Random Effects

**Files:**
- Modify: `crates/boostlss-py/src/lib.rs`
- Modify: `crates/boostlss-py/src/model.rs`
- Create: `crates/boostlss-py/tests/test_random_effects.py`

- [ ] **Step 1: Expose PyRandomEffectsLearner**

Modify `crates/boostlss-py/src/model.rs`. Add `PyRandomEffectsLearner`:

```rust
use boostlss::learner::RandomEffects;

#[pyclass(module = "boostlss_py")]
#[derive(Clone)]
pub struct PyRandomEffectsLearner {
    feature: String,
    df: f64,
}

#[pymethods]
impl PyRandomEffectsLearner {
    #[new]
    #[pyo3(signature = (feature, df=4.0))]
    fn new(feature: &str, df: f64) -> Self {
        Self {
            feature: feature.to_string(),
            df,
        }
    }
}
```

Inside `BoostLssModel::add_learner` in `model.rs` (near the other `if let Ok` extractions):
```rust
        } else if let Ok(r) = learner.extract::<PyRandomEffectsLearner>() {
            self.learners.push((
                param.to_string(),
                BaseLearner::RandomEffects(RandomEffects::new(&r.feature).df(r.df)),
            ));
```

Modify `crates/boostlss-py/src/lib.rs`. Add to the module init function:
```rust
    m.add_class::<model::PyRandomEffectsLearner>()?;
```

- [ ] **Step 2: Write Python integration test**

Create `crates/boostlss-py/tests/test_random_effects.py`:

```python
import pytest
import numpy as np
from boostlss_py import PyFamily, PyRandomEffectsLearner, BoostLssModel

def test_random_effects():
    # 3 groups, indexed 0, 1, 2
    groups = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])

    # Ground truth means: group 0 -> 10, group 1 -> 20, group 2 -> 30
    y = np.array([10.1, 9.9, 20.1, 19.9, 30.1, 29.9])

    # Needs to be 2D for input
    X = groups.reshape(-1, 1)

    family = PyFamily("GaussianLSS")
    model = BoostLssModel(family, mstop=100, step_length=0.1)

    # Pass df=1 for minimal penalization to quickly fit means
    model.add_learner("mu", PyRandomEffectsLearner("x", df=1.0))
    model.fit(X, y, columns=["x"])

    preds = model.predict(X, "mu")

    assert abs(preds[0] - 10.0) < 1.0
    assert abs(preds[2] - 20.0) < 1.0
    assert abs(preds[4] - 30.0) < 1.0

    # Unseen group (index 3) should predict exactly the global offset (global mean)
    # The global mean of 10, 20, 30 is 20.
    X_unseen = np.array([[3.0]])
    unseen_pred = model.predict(X_unseen, "mu")
    assert abs(unseen_pred[0] - 20.0) < 1.0

def test_random_effects_invalid_index():
    groups = np.array([0.5, 1.2]) # Invalid indices
    X = groups.reshape(-1, 1)
    y = np.array([10.0, 20.0])

    model = BoostLssModel(PyFamily("GaussianLSS"), mstop=1)
    model.add_learner("mu", PyRandomEffectsLearner("x"))

    with pytest.raises(ValueError, match="non-negative integer"):
        model.fit(X, y, columns=["x"])
```

- [ ] **Step 3: Run python tests**

Run: `maturin develop && pytest crates/boostlss-py/tests/test_random_effects.py`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add crates/boostlss-py/src/ crates/boostlss-py/tests/test_random_effects.py
git commit -m "feat: expose RandomEffects learner to Python bindings"
```
