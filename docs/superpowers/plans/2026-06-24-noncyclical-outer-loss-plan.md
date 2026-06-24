# Non-Cyclical Outer-Loss Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the "outer-loss" variant of the non-cyclical boosting algorithm, which evaluates all learners across all parameters against the global NLL simultaneously.

**Architecture:** We add a new `Algorithm::NonCyclicOuter` enum variant, expose it in the Python bindings as `algorithm="noncyclic_outer"`, and implement the evaluation logic inside `engine/noncyclical.rs` by extracting a dedicated `fit_noncyclical_outer` function that measures candidate learner risk reductions via the global NLL instead of using an intermediate RSS ranking step.

**Tech Stack:** Rust, PyO3, Python (pytest)

---

### Task 1: Add `Algorithm::NonCyclicOuter` Variant

**Files:**
- Modify: `crates/boostlss/src/engine/mod.rs`
- Modify: `crates/boostlss/src/engine/noncyclical.rs`

- [ ] **Step 1: Write the failing test**

```rust
// In crates/boostlss/src/engine/noncyclical.rs, inside mod tests
#[test]
fn test_algorithm_variants_exist() {
    use crate::engine::Algorithm;
    let _a1 = Algorithm::Cyclic;
    let _a2 = Algorithm::NonCyclic;
    let _a3 = Algorithm::NonCyclicOuter;
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p boostlss --lib engine::noncyclical::tests::test_algorithm_variants_exist`
Expected: FAIL with compilation error (unresolved variant `NonCyclicOuter`)

- [ ] **Step 3: Write minimal implementation**

```rust
// In crates/boostlss/src/engine/mod.rs
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Algorithm {
    Cyclic,
    NonCyclic,
    NonCyclicOuter,
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p boostlss --lib engine::noncyclical::tests::test_algorithm_variants_exist`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/boostlss/src/engine/mod.rs crates/boostlss/src/engine/noncyclical.rs
git commit -m "feat(engine): add NonCyclicOuter algorithm variant"
```

---

### Task 2: Implement Outer-Loss Evaluation Logic

**Files:**
- Modify: `crates/boostlss/src/engine/noncyclical.rs`

- [ ] **Step 1: Write the failing test**

```rust
// In crates/boostlss/src/engine/noncyclical.rs, inside mod tests
#[test]
fn test_fit_noncyclical_outer() {
    let x = array![[1.0], [2.0], [3.0], [4.0]];
    let y = array![2.0, 4.0, 6.0, 8.0];
    let data = Dataset::new(x, y, None).unwrap();

    let model = BoostLss::new(GaussianLss::new())
        .on("mu", |p| p.add(Linear::new(0))).unwrap()
        .on("sigma", |p| p.add(Linear::new(0))).unwrap()
        .algorithm(Algorithm::NonCyclicOuter)
        .mstop(Mstop::Scalar(1));

    // fit_noncyclical_outer should only generate 1 update and it should reduce NLL
    let fitted = fit_noncyclical_outer(model, &data).unwrap();
    assert_eq!(fitted.updates.len(), 1);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p boostlss --lib engine::noncyclical::tests::test_fit_noncyclical_outer`
Expected: FAIL (cannot find function `fit_noncyclical_outer`)

- [ ] **Step 3: Write minimal implementation**

```rust
// In crates/boostlss/src/engine/noncyclical.rs, add:

pub fn fit_noncyclical_outer<F: Family + Clone>(
    model: BoostLss<F>,
    data: &Dataset,
) -> Result<Fitted<F>, BoostlssError> {
    let mut current_predictions = Vec::new();
    let offsets = model.family().init_offsets(data)?;
    for offset in &offsets {
        current_predictions.push(ndarray::Array1::from_elem(data.n_obs(), *offset));
    }

    let mut cached_learners = Vec::new();
    let (family, config, mut learners) = model.into_parts();
    for (idx, (param_idx, learner)) in learners.iter_mut().enumerate() {
        let fit_state = learner.initialize(data)?;
        cached_learners.push(CachedLearner {
            param_idx: *param_idx,
            learner_idx: idx,
            fit_state,
        });
    }

    let max_mstop = match config.mstop {
        Mstop::Scalar(m) => m,
        Mstop::PerParam(_) => {
            return Err(BoostlssError::InvalidConfig(
                "NonCyclic algorithm requires a Scalar Mstop".into(),
            ));
        }
    };
    let nu = config.step_length;
    let mut updates = Vec::new();

    for _m in 1..=max_mstop {
        let base_nll = family.nll(data, &current_predictions)?;
        let num_params = family.params().len();

        let mut best_nll = f64::INFINITY;
        let mut best_candidate = None;

        for k in 0..num_params {
            let mut gradients = family.ngradient(data, &current_predictions, k)?;
            stabilize(&mut gradients, config.stabilization, data.weights());

            for cached in cached_learners.iter().filter(|c| c.param_idx == k) {
                let update = cached.fit_state.fit_update(gradients.view(), data.weights().map(|w| w.view()));
                let u_hat = cached.fit_state.predict_update(&update, data);
                let step = &u_hat * nu;

                // Temporarily apply update
                current_predictions[k] = &current_predictions[k] + &step;
                let nll = family.nll(data, &current_predictions)?;

                if nll < best_nll {
                    best_nll = nll;
                    best_candidate = Some((k, cached.learner_idx, update.clone(), step.clone()));
                }

                // Revert update
                current_predictions[k] = &current_predictions[k] - &step;
            }
        }

        if let Some((k, l_idx, mut update, step)) = best_candidate {
            current_predictions[k] = &current_predictions[k] + &step;
            let risk_reduction = (base_nll - best_nll).max(0.0);
            update.scale(nu);

            updates.push(UpdateStep {
                param_idx: k,
                learner_idx: l_idx,
                risk_reduction,
                update,
            });
        }
    }

    let mut fitted = Fitted::new(family, offsets, learners);
    fitted.updates = updates;
    Ok(fitted)
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p boostlss --lib engine::noncyclical::tests`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/boostlss/src/engine/noncyclical.rs
git commit -m "feat(engine): implement fit_noncyclical_outer logic"
```

---

### Task 3: Expose Outer-Loss to Python Bindings

**Files:**
- Modify: `crates/boostlss-py/src/model.rs`
- Create: `crates/boostlss-py/tests/test_noncyclic_outer.py`

- [ ] **Step 1: Write the failing test**

```python
// In crates/boostlss-py/tests/test_noncyclic_outer.py
import pytest
import numpy as np
from boostlss_py import BoostLssModel, GaussianLss, Linear

def test_noncyclic_outer():
    X = np.random.normal(size=(100, 2))
    y = X[:, 0] * 2.0 + np.random.normal(size=100) * 0.1

    model = BoostLssModel(GaussianLss(), mstop=10, step_length=0.1, algorithm="noncyclic_outer")
    model.add_learner("mu", Linear(0))
    model.add_learner("sigma", Linear(0))

    model.fit(X, y)
    preds = model.predict(X, "mu")
    assert len(preds) == 100
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run maturin develop -m crates/boostlss-py/Cargo.toml && uv run pytest crates/boostlss-py/tests/test_noncyclic_outer.py`
Expected: FAIL (`ValueError: algorithm must be 'cyclic' or 'noncyclic'`)

- [ ] **Step 3: Write minimal implementation**

In `crates/boostlss-py/src/model.rs`, update `BoostLssModel::new`:

```rust
// Replace algorithm parsing:
        let algorithm_enum = match algorithm {
            "cyclic" => Algorithm::Cyclic,
            "noncyclic" => Algorithm::NonCyclic,
            "noncyclic_outer" => Algorithm::NonCyclicOuter,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "algorithm must be 'cyclic', 'noncyclic', or 'noncyclic_outer'",
                ))
            }
        };
```

Update `BoostLssModel::fit` and `BoostLssModel::cvrisk` and `BoostLssModel::stabsel` for EVERY family (Gaussian, Binomial, Beta, Weibull, LogNormal, Zip, Gev) to match the new enum variant. You must add `Algorithm::NonCyclicOuter` to the `match self.algorithm` block inside `fit`, `cvrisk`, and `stabsel` methods:

```rust
// For fit:
                let fitted = match self.algorithm {
                    Algorithm::Cyclic => fit_cyclical(model, &dataset)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
                    Algorithm::NonCyclic => {
                        let model = model.algorithm(Algorithm::NonCyclic);
                        fit_noncyclical(model, &dataset)
                            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
                    }
                    Algorithm::NonCyclicOuter => {
                        let model = model.algorithm(Algorithm::NonCyclicOuter);
                        boostlss::engine::noncyclical::fit_noncyclical_outer(model, &dataset)
                            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
                    }
                };

// For cvrisk and stabsel:
                let model = match self.algorithm {
                    Algorithm::Cyclic => model,
                    Algorithm::NonCyclic => model.algorithm(Algorithm::NonCyclic),
                    Algorithm::NonCyclicOuter => model.algorithm(Algorithm::NonCyclicOuter),
                };
```
(Apply this change to all 7 family match arms in `fit`, `cvrisk`, and `stabsel`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run maturin develop -m crates/boostlss-py/Cargo.toml && uv run pytest crates/boostlss-py/tests/test_noncyclic_outer.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/boostlss-py/src/model.rs crates/boostlss-py/tests/test_noncyclic_outer.py
git commit -m "feat(python): expose noncyclic_outer algorithm in bindings"
```
