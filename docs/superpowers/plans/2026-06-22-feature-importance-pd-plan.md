# Feature Importance and Partial Dependence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement model interpretability tools: Feature Importance (Empirical Risk Reduction) and Partial Dependence (Friedman's method), exposing both to the Rust core and Python bindings.

**Architecture:** We will track the Empirical Risk Reduction in `cyclical.rs` during training and store it in `UpdateStep`. We will add methods to `Fitted` to aggregate these reductions for Feature Importance and to compute Partial Dependence via grid-based dataset manipulation and prediction averaging. Python bindings will wrap these methods.

**Tech Stack:** Rust (boostlss core), PyO3 (boostlss-py bindings)

---

### Task 1: Update Storage for Empirical Risk Reduction

**Files:**
- Modify: `crates/boostlss/src/model.rs`

- [ ] **Step 1: Write the failing tests**

Modify `crates/boostlss/src/model.rs`. Add a test at the end of the file to verify the new field exists in `UpdateStep`.

```rust
    #[test]
    fn test_update_step_has_risk_reduction() {
        let update = UpdateStep {
            param_idx: 0,
            learner_idx: 1,
            update: crate::learner::LearnerUpdate::Linear(ndarray::Array1::zeros(2)),
            risk_reduction: 1.5,
        };
        assert_eq!(update.risk_reduction, 1.5);
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p boostlss -- test_update_step_has_risk_reduction`
Expected: FAIL due to missing `risk_reduction` field.

- [ ] **Step 3: Write minimal implementation**

Modify `crates/boostlss/src/model.rs` near line 124 to add `risk_reduction` to `UpdateStep`.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateStep {
    pub param_idx: usize,
    pub learner_idx: usize,
    pub update: LearnerUpdate,
    pub risk_reduction: f64,
}
```

*Note: This will temporarily break `fit_cyclical` compilation, which we fix in Step 5.*

- [ ] **Step 4: Fix compilation in `cyclical.rs`**

Modify `crates/boostlss/src/engine/cyclical.rs` near line 155 to include a dummy `risk_reduction` for now so tests compile.

```rust
                updates.push(UpdateStep {
                    param_idx: k,
                    learner_idx: l_idx,
                    risk_reduction: 0.0, // Will be implemented in Task 2
                    update: match update {
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test -p boostlss`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add crates/boostlss/src/model.rs crates/boostlss/src/engine/cyclical.rs
git commit -m "feat: add risk_reduction field to UpdateStep"
```

---

### Task 2: Calculate Empirical Risk Reduction in Cyclical Engine

**Files:**
- Modify: `crates/boostlss/src/engine/cyclical.rs`

- [ ] **Step 1: Write the failing test**

Modify `crates/boostlss/src/engine/cyclical.rs` at the bottom to test risk reduction calculation.

```rust
    #[test]
    fn test_risk_reduction_calculation() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![2.0, 4.0, 6.0, 8.0];
        let data = Dataset::new(x, y, None).unwrap();

        let model = BoostLss::new(GaussianLss::new())
            .on("mu", |p| p.add(Linear::new("x")))
            .unwrap()
            .algorithm(crate::engine::Algorithm::Cyclic)
            .mstop(Mstop::Scalar(1));

        let fitted = fit_cyclical(model, &data).unwrap();

        assert_eq!(fitted.updates.len(), 1);
        assert!(fitted.updates[0].risk_reduction > 0.0);
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p boostlss -- test_risk_reduction_calculation`
Expected: FAIL because `risk_reduction` is currently hardcoded to `0.0`.

- [ ] **Step 3: Write minimal implementation**

Modify `crates/boostlss/src/engine/cyclical.rs` inside the `for k in 0..family.params().len()` loop, right after `let mut gradients = ...` and stabilization (around line 61).

```rust
            let base_rss = match data.weights() {
                Some(w) => (&gradients * &gradients * w).sum(),
                None => (&gradients * &gradients).sum(),
            };
```

Then, later when creating the `UpdateStep` (around line 155), replace the `0.0` with the actual reduction:

```rust
            if let (Some(update), Some(u_hat), Some(l_idx)) =
                (best_update, best_u_hat, best_learner_idx)
            {
                current_predictions[k] = &current_predictions[k] + &(&u_hat * nu);
                let risk_reduction = base_rss - best_rss;
                updates.push(UpdateStep {
                    param_idx: k,
                    learner_idx: l_idx,
                    risk_reduction,
                    update: match update {
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p boostlss -- test_risk_reduction_calculation`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/boostlss/src/engine/cyclical.rs
git commit -m "feat: calculate and store empirical risk reduction in cyclical engine"
```

---

### Task 3: Implement Feature Importance API on Fitted

**Files:**
- Modify: `crates/boostlss/src/model.rs`

- [ ] **Step 1: Write the failing test**

Modify `crates/boostlss/src/model.rs` to add a test for `feature_importance`.

```rust
    #[test]
    fn test_feature_importance() {
        let family = GaussianLss::new();
        let learners = vec![(0, BaseLearner::Linear(Linear::new("x"))), (0, BaseLearner::Linear(Linear::new("x2")))];
        let mut fitted = Fitted::new(family, vec![0.0, 0.0], learners);

        fitted.updates.push(UpdateStep {
            param_idx: 0,
            learner_idx: 0,
            update: crate::learner::LearnerUpdate::Linear(ndarray::Array1::zeros(2)),
            risk_reduction: 2.0,
        });
        fitted.updates.push(UpdateStep {
            param_idx: 0,
            learner_idx: 1,
            update: crate::learner::LearnerUpdate::Linear(ndarray::Array1::zeros(2)),
            risk_reduction: 1.5,
        });
        fitted.updates.push(UpdateStep {
            param_idx: 0,
            learner_idx: 0,
            update: crate::learner::LearnerUpdate::Linear(ndarray::Array1::zeros(2)),
            risk_reduction: 0.5,
        });

        let importance = fitted.feature_importance();
        assert_eq!(importance.len(), 2);
        assert_eq!(importance[0], 2.5); // 2.0 + 0.5
        assert_eq!(importance[1], 1.5);
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p boostlss -- test_feature_importance`
Expected: FAIL due to missing method `feature_importance`.

- [ ] **Step 3: Write minimal implementation**

Modify `crates/boostlss/src/model.rs` inside the `impl<F: Family> Fitted<F>` block to add `feature_importance`.

```rust
    pub fn feature_importance(&self) -> Vec<f64> {
        let mut importances = vec![0.0; self.learners.len()];
        for update in &self.updates {
            if update.learner_idx < importances.len() {
                importances[update.learner_idx] += update.risk_reduction;
            }
        }
        importances
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p boostlss -- test_feature_importance`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/boostlss/src/model.rs
git commit -m "feat: implement feature_importance aggregation on Fitted model"
```

---

### Task 4: Implement Partial Dependence API on Fitted

**Files:**
- Modify: `crates/boostlss/src/model.rs`

- [ ] **Step 1: Write the failing test**

Modify `crates/boostlss/src/model.rs` at the bottom.

```rust
    #[test]
    fn test_partial_dependence() {
        use ndarray::{array, Array1, Array2};
        let family = GaussianLss::new();
        let learners = vec![(0, BaseLearner::Linear(Linear::new("x")))];
        let mut fitted = Fitted::new(family, vec![0.0, 0.0], learners);

        // Mock an update where predicted mu = 2.0 * x
        fitted.updates.push(UpdateStep {
            param_idx: 0,
            learner_idx: 0,
            update: crate::learner::LearnerUpdate::Linear(array![0.0, 2.0]),
            risk_reduction: 0.0,
        });

        let data = Dataset::new(Array2::<f64>::zeros((5, 2)), Array1::<f64>::zeros(5), None).unwrap();
        let grid = vec![1.0, 2.0, 3.0];

        // We evaluate feature_idx 1 (the 'x' column in the design matrix, since 0 is intercept for Linear usually, but let's test feature_idx=0 here)
        let pd = fitted.partial_dependence(&data, "mu", 0, &grid).unwrap();

        assert_eq!(pd.len(), 3);
        // pred = offset(0) + 0.0(intercept) + 2.0 * grid_val
        assert_eq!(pd[0], 2.0); // 2.0 * 1.0
        assert_eq!(pd[1], 4.0); // 2.0 * 2.0
        assert_eq!(pd[2], 6.0); // 2.0 * 3.0
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p boostlss -- test_partial_dependence`
Expected: FAIL due to missing method `partial_dependence`.

- [ ] **Step 3: Write minimal implementation**

Modify `crates/boostlss/src/model.rs` inside the `impl<F: Family> Fitted<F>` block to add `partial_dependence`.

```rust
    pub fn partial_dependence(
        &mut self,
        data: &Dataset,
        param: &str,
        feature_idx: usize,
        grid: &[f64],
    ) -> Result<Vec<f64>, BoostlssError> {
        let mut results = Vec::with_capacity(grid.len());

        for &val in grid {
            let mut modified_design = data.design().clone();
            if feature_idx >= modified_design.ncols() {
                return Err(BoostlssError::DataError(format!("Feature index {} out of bounds for design matrix with {} columns", feature_idx, modified_design.ncols())));
            }
            modified_design.column_mut(feature_idx).fill(val);

            let modified_data = Dataset::new(
                modified_design,
                data.response().clone(),
                data.weights().cloned()
            )?;

            let preds = self.predict(&modified_data, param, Scale::Link)?;
            let mean_pred = preds.sum() / (preds.len() as f64);
            results.push(mean_pred);
        }

        Ok(results)
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p boostlss -- test_partial_dependence`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/boostlss/src/model.rs
git commit -m "feat: implement partial_dependence computation on Fitted model"
```

---

### Task 5: Python Bindings for Feature Importance & Partial Dependence

**Files:**
- Modify: `crates/boostlss-py/src/model.rs`
- Modify: `crates/boostlss-py/tests/test_fitted.py` (or create if needed)

- [ ] **Step 1: Write the failing tests**

Modify `crates/boostlss-py/tests/test_fitted.py` (create it if it doesn't exist).

```python
import numpy as np
from boostlss_py import PyDataset, BoostLss, GaussianLss, PyLinearLearner, Mstop, Algorithm

def test_feature_importance_and_partial_dependence():
    x = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([2.0, 4.0, 6.0, 8.0])
    data = PyDataset(x, y)

    model = BoostLss(GaussianLss())
    model.on("mu", PyLinearLearner("x"))
    model.set_algorithm(Algorithm.Cyclic)
    model.set_mstop(Mstop.Scalar(5))

    fitted = model.fit(data)

    # Test feature importance
    importance = fitted.feature_importance()
    assert len(importance) == 1
    assert importance[0] > 0.0

    # Test partial dependence
    grid = [1.0, 2.0, 3.0]
    pd = fitted.partial_dependence(data, "mu", 0, grid)
    assert len(pd) == 3
    assert pd[0] < pd[1] < pd[2]  # Monotonic increase
```

- [ ] **Step 2: Run test to verify it fails**

Run: `maturin develop && pytest crates/boostlss-py/tests/test_fitted.py`
Expected: FAIL due to `PyFitted` missing the methods.

- [ ] **Step 3: Write minimal implementation**

Modify `crates/boostlss-py/src/model.rs` inside the `#[pymethods] impl PyFitted` block. Add `feature_importance` and `partial_dependence`.

```rust
    pub fn feature_importance(&self) -> PyResult<Vec<f64>> {
        let mut inner = self.inner.lock().unwrap();
        let importance = match &mut *inner {
            FittedState::Gaussian(f) => f.feature_importance(),
            FittedState::StudentT(f) => f.feature_importance(),
            FittedState::Gamma(f) => f.feature_importance(),
            FittedState::NegativeBinomial(f) => f.feature_importance(),
        };
        Ok(importance)
    }

    pub fn partial_dependence(
        &self,
        data: &PyDataset,
        param: &str,
        feature_idx: usize,
        grid: Vec<f64>,
    ) -> PyResult<Vec<f64>> {
        let mut inner = self.inner.lock().unwrap();
        let pd = match &mut *inner {
            FittedState::Gaussian(f) => f.partial_dependence(&data.inner, param, feature_idx, &grid),
            FittedState::StudentT(f) => f.partial_dependence(&data.inner, param, feature_idx, &grid),
            FittedState::Gamma(f) => f.partial_dependence(&data.inner, param, feature_idx, &grid),
            FittedState::NegativeBinomial(f) => f.partial_dependence(&data.inner, param, feature_idx, &grid),
        }.map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(pd)
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `maturin develop && pytest crates/boostlss-py/tests/test_fitted.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/boostlss-py/src/model.rs crates/boostlss-py/tests/test_fitted.py
git commit -m "feat: expose feature_importance and partial_dependence to python bindings"
```
