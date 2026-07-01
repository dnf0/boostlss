# Early Stopping & Validation Sets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement early stopping and validation set tracking for boostlss, exposing evaluation history and early stopping controls to Python.

**Architecture:** We will update the Rust `Fitted` struct to contain evaluation history and the best iteration. The core `fit` and engine functions will be updated to accept optional validation data and early stopping rounds. The engines will evaluate training and validation loss at each iteration, halting early if validation loss doesn't improve, and truncating the model updates. The Python API will be updated to accept `eval_set` and expose `evals_result_` and `best_iteration_`.

**Tech Stack:** Rust, PyO3, Python, Pytest.

---

### Task 1: Update Rust Models and Engine Signatures

**Files:**
- Modify: `crates/boostlss/src/model.rs`
- Modify: `crates/boostlss/src/engine/cyclical.rs`
- Modify: `crates/boostlss/src/engine/noncyclical.rs`
- Modify: `crates/boostlss-py/src/model.rs`

- [ ] **Step 1: Update `model.rs` with `EvalResults` and signature changes**

Update `crates/boostlss/src/model.rs`:
```rust
// Add at the bottom of the file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalResults {
    pub train_loss: Vec<f64>,
    pub val_loss: Option<Vec<f64>>,
}

// Update Fitted struct
#[allow(dead_code)]
#[derive(Debug, Serialize, Deserialize)]
pub struct Fitted<F: Family> {
    family: F,
    offsets: Vec<f64>,
    pub updates: Vec<UpdateStep>,
    pub learners: Vec<(String, Box<dyn BaseLearner>)>,
    pub eval_results: EvalResults,
    pub best_iteration: usize,
}

// Update BoostLss::fit signature
    pub fn fit(
        self,
        data: &Dataset,
        eval_data: Option<&Dataset>,
        early_stopping_rounds: Option<usize>,
    ) -> Result<Fitted<F>, BoostlssError> {
        match self.config.algorithm {
            Algorithm::Cyclic => crate::engine::cyclical::fit_cyclical(self, data, eval_data, early_stopping_rounds),
            Algorithm::NonCyclic => {
                if matches!(self.config.mstop, Mstop::PerParam(_)) {
                    return Err(BoostlssError::InvalidConfig(
                        "NonCyclic algorithm requires a Scalar Mstop".into(),
                    ));
                }
                crate::engine::noncyclical::fit_noncyclical(self, data, eval_data, early_stopping_rounds)
            }
            Algorithm::NonCyclicOuter => {
                if matches!(self.config.mstop, Mstop::PerParam(_)) {
                    return Err(BoostlssError::InvalidConfig(
                        "NonCyclic algorithm requires a Scalar Mstop".into(),
                    ));
                }
                crate::engine::noncyclical::fit_noncyclical_outer(self, data, eval_data, early_stopping_rounds)
            }
        }
    }
```

- [ ] **Step 2: Update Engine Signatures and Returns**

In `crates/boostlss/src/engine/cyclical.rs`, update `fit_cyclical` signature and return:
```rust
// Signature
pub fn fit_cyclical<F: Family + Clone>(
    model: BoostLss<F>,
    data: &Dataset,
    eval_data: Option<&Dataset>,
    early_stopping_rounds: Option<usize>,
) -> Result<Fitted<F>, BoostlssError> {
```
```rust
// Return at the bottom of the function
    Ok(Fitted {
        family,
        offsets,
        updates,
        learners: cached_learners
            .into_iter()
            .map(|c| (c.param_name, c.fit_state.learner))
            .collect(),
        eval_results: crate::model::EvalResults { train_loss: vec![], val_loss: None },
        best_iteration: max_mstop,
    })
```
Fix the test mock in `cyclical.rs`:
```rust
        let fitted = model.fit(&data, None, None).unwrap();
```

Do the same for `fit_noncyclical` and `fit_noncyclical_outer` in `crates/boostlss/src/engine/noncyclical.rs`, updating their signatures, returns, and tests to pass `None, None`.

- [ ] **Step 3: Update PyO3 Macros**

In `crates/boostlss-py/src/model.rs`, update the `fit_family!` macro:
```rust
macro_rules! fit_family {
    ($self:expr, $dataset:expr, $eval_data:expr, $early_stopping:expr, $family:expr, $variant:path) => {{
        let mut model = BoostLss::new($family, $self.mstop, $self.step_length, &$self.algorithm)?;
        for learner in &$self.learners {
            model.add_learner(learner.param.clone(), learner.learner.clone())?;
        }
        let fitted = model.fit($dataset, $eval_data, $early_stopping)?;
        $self.fitted = Some($variant(fitted));
        Ok(())
    }};
}
```
Update all usages of `fit_family!` in `PyBoostLssModel::fit` to pass `None, None` for now:
```rust
        match self.family {
            InternalFamily::Gaussian => {
                fit_family!(self, &dataset, None, None, GaussianLss::new(), FittedModel::Gaussian)
            }
// ... do this for ALL variants in the match block ...
```

- [ ] **Step 4: Verify Compilation**

Run `cargo check -p boostlss` and `cargo check -p boostlss-py`. Expect PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/boostlss/src/model.rs crates/boostlss/src/engine/cyclical.rs crates/boostlss/src/engine/noncyclical.rs crates/boostlss-py/src/model.rs
git commit -m "feat: update engine signatures for early stopping"
```

### Task 2: Implement Early Stopping in Cyclical Engine

**Files:**
- Modify: `crates/boostlss/src/engine/cyclical.rs`

- [ ] **Step 1: Write the failing test**

In `crates/boostlss/src/engine/cyclical.rs`:
```rust
    #[test]
    fn test_early_stopping_cyclical() {
        let n = 100;
        let mut rng = StdRng::seed_from_u64(42);
        let x = Array2::random_using((n, 1), Uniform::new(0., 1.), &mut rng);
        let y = Array1::random_using(n, Uniform::new(0., 1.), &mut rng);
        let data = Dataset::new(x.clone(), y.clone(), None).unwrap();

        let family = crate::family::GaussianLss::new();
        // High mstop, should stop early
        let mut model = BoostLss::new(family, 1000, 0.1, "cyclic").unwrap();
        model.add_learner("mu", Box::new(crate::learner::LinearLearner::new(0, true))).unwrap();

        // Use identical data for eval, just to test tracking/stopping mechanics
        let fitted = model.fit(&data, Some(&data), Some(5)).unwrap();

        assert!(fitted.best_iteration < 1000);
        assert_eq!(fitted.updates.len(), fitted.best_iteration * 1); // 1 param
        assert!(fitted.eval_results.val_loss.is_some());
        assert_eq!(fitted.eval_results.train_loss.len(), fitted.best_iteration);
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run `cargo test -p boostlss test_early_stopping_cyclical`
Expected: FAIL (assertion `fitted.best_iteration < 1000` fails, as it's not implemented)

- [ ] **Step 3: Implement early stopping**

In `fit_cyclical`:
```rust
    let mut best_val_nll = f64::INFINITY;
    let mut best_iteration = 0;
    let mut train_losses = Vec::new();
    let mut val_losses = if eval_data.is_some() { Some(Vec::new()) } else { None };

    let mut current_eval_predictions = if let Some(e_data) = eval_data {
        let mut preds = Vec::new();
        for offset in &offsets {
            preds.push(ndarray::Array1::from_elem(e_data.n_obs(), *offset));
        }
        Some(preds)
    } else {
        None
    };

    for m in 1..=max_mstop {
        for k in 0..family.params().len() {
            // ... existing update code ...

            // At the end of the k loop, update eval predictions if needed
            if let Some(ref mut eval_preds) = current_eval_predictions {
                if let Some(e_data) = eval_data {
                    let learner = &cached_learners[best_learner_idx].fit_state.learner;
                    let p = learner.predict(e_data)?;
                    eval_preds[k] = &eval_preds[k] + &(&p * nu);
                }
            }
        }

        // Track losses
        let train_nll = family.nll(data, &current_predictions)?;
        train_losses.push(train_nll);

        if let Some(ref eval_preds) = current_eval_predictions {
            if let Some(e_data) = eval_data {
                let val_nll = family.nll(e_data, eval_preds)?;
                val_losses.as_mut().unwrap().push(val_nll);

                if val_nll < best_val_nll {
                    best_val_nll = val_nll;
                    best_iteration = m;
                }

                if let Some(patience) = early_stopping_rounds {
                    if m - best_iteration >= patience {
                        break;
                    }
                }
            }
        } else {
            best_iteration = m;
        }
    }

    // Truncate updates to best iteration
    if let Some(_patience) = early_stopping_rounds {
        if eval_data.is_some() {
            let params_count = family.params().len();
            updates.truncate(best_iteration * params_count);
        }
    }

    Ok(Fitted {
        family,
        offsets,
        updates,
        learners: cached_learners
            .into_iter()
            .map(|c| (c.param_name, c.fit_state.learner))
            .collect(),
        eval_results: crate::model::EvalResults { train_loss: train_losses, val_loss: val_losses },
        best_iteration,
    })
```

- [ ] **Step 4: Run test to verify it passes**

Run `cargo test -p boostlss test_early_stopping_cyclical`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/boostlss/src/engine/cyclical.rs
git commit -m "feat: implement early stopping in cyclical engine"
```

### Task 3: Implement Early Stopping in NonCyclical Engines

**Files:**
- Modify: `crates/boostlss/src/engine/noncyclical.rs`

- [ ] **Step 1: Write the failing tests**

In `crates/boostlss/src/engine/noncyclical.rs`, add tests similar to `test_early_stopping_cyclical` for both `fit_noncyclical` and `fit_noncyclical_outer`. High `mstop`, same data for eval, assert early stopping truncates.

- [ ] **Step 2: Run tests to verify they fail**

Run `cargo test -p boostlss`

- [ ] **Step 3: Implement early stopping**

Apply the same logic used in Task 2 to `fit_noncyclical` and `fit_noncyclical_outer`.
- Initialize `train_losses`, `val_losses`, `best_val_nll`, `best_iteration`.
- Initialize `current_eval_predictions`.
- After finding the `best_global_learner_idx` and applying the update to `current_predictions`, apply it to `current_eval_predictions`.
- Calculate `train_nll` and `val_nll`.
- Check patience and `break`.
- Truncate updates: `updates.truncate(best_iteration)` (since noncyclic does 1 update per `m`).
- Return correct `eval_results` and `best_iteration`.

- [ ] **Step 4: Run tests to verify they pass**

Run `cargo test -p boostlss`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/boostlss/src/engine/noncyclical.rs
git commit -m "feat: implement early stopping in noncyclical engines"
```

### Task 4: Python API Updates for Early Stopping

**Files:**
- Modify: `crates/boostlss-py/src/model.rs`

- [ ] **Step 1: Update `fit` signature**

Change the `fit` method on `PyBoostLssModel` in `crates/boostlss-py/src/model.rs`:
```rust
    #[pyo3(signature = (data, response, weights=None, eval_set=None, early_stopping_rounds=None))]
    pub fn fit(
        &mut self,
        data: PyReadonlyArray2<f64>,
        response: PyReadonlyArray<f64>,
        weights: Option<PyReadonlyArray1<f64>>,
        eval_set: Option<(PyReadonlyArray2<f64>, PyReadonlyArray<f64>)>,
        early_stopping_rounds: Option<usize>,
    ) -> PyResult<()> {
```

- [ ] **Step 2: Process `eval_set`**

Inside `fit`, construct the validation dataset:
```rust
        // Under let dataset = ...
        let eval_dataset = if let Some((e_x, e_y)) = eval_set {
            let e_x_array = e_x.as_array().to_owned();
            let e_y_array = match e_y.ndim() {
                1 => e_y.as_array().into_dimensionality::<ndarray::Ix1>().map_err(|_| {
                    pyo3::exceptions::PyValueError::new_err("eval_set response must be 1D")
                })?.to_owned(),
                2 => {
                    let y_view = e_y.as_array().into_dimensionality::<ndarray::Ix2>().map_err(|_| {
                        pyo3::exceptions::PyValueError::new_err("eval_set response must be 1D or 2D column vector")
                    })?;
                    if y_view.ncols() != 1 {
                        return Err(pyo3::exceptions::PyValueError::new_err("eval_set response must be a single column"));
                    }
                    y_view.column(0).to_owned()
                },
                _ => return Err(pyo3::exceptions::PyValueError::new_err("eval_set response must be 1D or 2D column vector")),
            };
            Some(Dataset::new(e_x_array, e_y_array, None).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(e.to_string())
            })?)
        } else {
            None
        };
        let eval_data_ref = eval_dataset.as_ref();
```

- [ ] **Step 3: Pass arguments to macros**

Update all `fit_family!` calls in the match block to pass `eval_data_ref` and `early_stopping_rounds`:
```rust
        match self.family {
            InternalFamily::Gaussian => {
                fit_family!(self, &dataset, eval_data_ref, early_stopping_rounds, GaussianLss::new(), FittedModel::Gaussian)
            }
// ... update all variants ...
```

- [ ] **Step 4: Expose Properties**

Add getters to `PyBoostLssModel`:
```rust
    #[getter]
    pub fn evals_result_(&self, py: Python) -> PyResult<PyObject> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Model not fitted"))?;

        let results = match fitted {
            FittedModel::Gaussian(f) => &f.eval_results,
            FittedModel::Binomial(f) => &f.eval_results,
            // ... add all variants ...
        };

        let dict = pyo3::types::PyDict::new_bound(py);
        let train_dict = pyo3::types::PyDict::new_bound(py);
        train_dict.set_item("loss", results.train_loss.clone())?;
        dict.set_item("train", train_dict)?;

        if let Some(val_loss) = &results.val_loss {
            let val_dict = pyo3::types::PyDict::new_bound(py);
            val_dict.set_item("loss", val_loss.clone())?;
            dict.set_item("valid", val_dict)?;
        }

        Ok(dict.into())
    }

    #[getter]
    pub fn best_iteration_(&self) -> PyResult<usize> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Model not fitted"))?;

        match fitted {
            FittedModel::Gaussian(f) => Ok(f.best_iteration),
            FittedModel::Binomial(f) => Ok(f.best_iteration),
            // ... add all variants ...
        }
    }
```

- [ ] **Step 5: Verify Compilation**

Run `cargo check -p boostlss-py`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add crates/boostlss-py/src/model.rs
git commit -m "feat: expose early stopping in Python API"
```

### Task 5: Python Tests

**Files:**
- Create: `crates/boostlss-py/tests/test_early_stopping.py`

- [ ] **Step 1: Write early stopping tests**

Create `crates/boostlss-py/tests/test_early_stopping.py`:
```python
import pytest
import numpy as np
from boostlss_py import BoostLssModel, PyFamily, PyLinearLearner

def test_early_stopping():
    np.random.seed(42)
    X = np.random.randn(200, 5)
    y = 2.0 * X[:, 0] + np.random.randn(200) * 0.1

    X_train, y_train = X[:100], y[:100]
    X_val, y_val = X[100:], y[100:]

    family = PyFamily("GaussianLss")
    model = BoostLssModel(family, mstop=1000, step_length=0.1)
    model.add_learner("mu", PyLinearLearner(0, intercept=True))
    model.add_learner("sigma", PyLinearLearner(0, intercept=True))

    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=10)

    assert model.best_iteration_ < 1000
    evals = model.evals_result_
    assert "train" in evals
    assert "valid" in evals
    assert len(evals["train"]["loss"]) == model.best_iteration_
    assert len(evals["valid"]["loss"]) == model.best_iteration_
```

- [ ] **Step 2: Run test**

Run `uv run pytest tests/test_early_stopping.py` from `crates/boostlss-py`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add crates/boostlss-py/tests/test_early_stopping.py
git commit -m "test: add python tests for early stopping"
```
