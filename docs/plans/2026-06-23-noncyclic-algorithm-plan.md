# Non-Cyclical (`NonCyclic`) Boosting Algorithm Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the `NonCyclic` boosting algorithm which selects and applies the single best parameter-learner update per iteration to minimize the total Negative Log-Likelihood (NLL).

**Architecture:** We use a Two-Stage Selection architecture. In Stage 1, we find the optimal base-learner for each parameter using RSS against the negative gradient. In Stage 2, we evaluate the full NLL for each of the $K$ candidates and select the one that yields the lowest NLL.

**Tech Stack:** Rust (2021 edition), `ndarray`.

---

## File structure

| File                                        | Responsibility                                                 |
| ------------------------------------------- | -------------------------------------------------------------- |
| `crates/boostlss/src/engine/noncyclical.rs` | Implements the `fit_noncyclical` function.                     |
| `crates/boostlss/src/engine/mod.rs`         | Export `noncyclical` module.                                   |
| `crates/boostlss/src/model.rs`              | Update `BoostLss::fit` and enforce scalar `Mstop` requirement. |

---

### Task 1: Module and Error Routing

**Files:**

- Modify: `crates/boostlss/src/engine/mod.rs`
- Modify: `crates/boostlss/src/model.rs`
- Create: `crates/boostlss/src/engine/noncyclical.rs`

- [ ] **Step 1: Write the routing and failing test**

In `crates/boostlss/src/engine/noncyclical.rs`:

```rust
use crate::data::Dataset;
use crate::error::BoostlssError;
use crate::family::Family;
use crate::model::{BoostLss, Fitted};

pub fn fit_noncyclical<F: Family + Clone>(
    _model: BoostLss<F>,
    _data: &Dataset,
) -> Result<Fitted<F>, BoostlssError> {
    Err(BoostlssError::NotConverged("NonCyclic fit unimplemented".into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Dataset;
    use crate::engine::Mstop;
    use crate::family::GaussianLss;
    use crate::learner::Linear;
    use ndarray::array;

    #[test]
    fn test_fit_noncyclical_unimplemented() {
        let x = array![[1.0], [2.0]];
        let y = array![2.0, 4.0];
        let data = Dataset::new(x, y, None).unwrap();

        let model = BoostLss::new(GaussianLss::new())
            .on("mu", |p| p.add(Linear::new("x")))
            .unwrap()
            .algorithm(crate::engine::Algorithm::NonCyclic)
            .mstop(Mstop::Scalar(1));

        let res = fit_noncyclical(model, &data);
        assert!(res.is_err());
    }
}
```

- [ ] **Step 2: Enable module and route `BoostLss::fit`**

In `crates/boostlss/src/engine/mod.rs`, add:

```rust
pub mod noncyclical;
```

In `crates/boostlss/src/model.rs`, update `fit`:

```rust
    pub fn fit(self, data: &Dataset) -> Result<Fitted<F>, BoostlssError> {
        match self.config.algorithm {
            Algorithm::Cyclic => crate::engine::cyclical::fit_cyclical(self, data),
            Algorithm::NonCyclic => {
                if matches!(self.config.mstop, Mstop::PerParam(_)) {
                    return Err(BoostlssError::InvalidConfig(
                        "NonCyclic algorithm requires a Scalar Mstop".into(),
                    ));
                }
                crate::engine::noncyclical::fit_noncyclical(self, data)
            }
        }
    }
```

- [ ] **Step 3: Add `BoostLss::fit` test for invalid Mstop**

In `crates/boostlss/src/model.rs` `mod tests`:

```rust
    #[test]
    fn test_boostlss_fit_noncyclic_requires_scalar_mstop() {
        use crate::data::Dataset;
        use crate::engine::Mstop;
        use crate::family::GaussianLss;
        use crate::learner::Linear;
        use ndarray::{array, Array2};

        let x = Array2::<f64>::zeros((2, 1));
        let y = array![1.0, 2.0];
        let data = Dataset::new(x, y, None).unwrap();

        let model = BoostLss::new(GaussianLss::new())
            .on("mu", |p| p.add(Linear::new("x")))
            .unwrap()
            .algorithm(crate::engine::Algorithm::NonCyclic)
            .mstop(Mstop::PerParam(vec![10, 10])); // Invalid for NonCyclic

        let result = model.fit(&data);
        assert!(matches!(result, Err(BoostlssError::InvalidConfig(_))));
    }
```

- [ ] **Step 4: Run tests to verify**

Run: `cargo test -p boostlss -- engine::noncyclical::tests`
Run: `cargo test -p boostlss -- model::tests::test_boostlss_fit_noncyclic_requires_scalar_mstop`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/boostlss/src/engine/mod.rs crates/boostlss/src/model.rs crates/boostlss/src/engine/noncyclical.rs
git commit -m "feat: scaffold NonCyclic engine routing and validation"
```

---

### Task 2: Core Algorithm Loop (Stage 1 & 2)

**Files:**

- Modify: `crates/boostlss/src/engine/noncyclical.rs`

- [ ] **Step 1: Write the implementation and test**

Replace the contents of `crates/boostlss/src/engine/noncyclical.rs` with:

```rust
use crate::data::Dataset;
use crate::engine::stabilization::stabilize;
use crate::engine::Mstop;
use crate::error::BoostlssError;
use crate::family::Family;
use crate::learner::{LearnerFit, LearnerUpdate};
use crate::model::{BoostLss, Fitted, UpdateStep};

struct CachedLearner {
    param_idx: usize,
    learner_idx: usize,
    fit_state: LearnerFit,
}

pub fn fit_noncyclical<F: Family + Clone>(
    model: BoostLss<F>,
    data: &Dataset,
) -> Result<Fitted<F>, BoostlssError> {
    let mut current_predictions = Vec::new();

    let offsets = model.family().init_offsets(data)?;

    for offset in &offsets {
        current_predictions.push(ndarray::Array1::from_elem(data.n_obs(), *offset));
    }

    let mut cached_learners = Vec::new();
    let x_col = data.design().column(0).to_owned();

    let (family, config, mut learners) = model.into_parts();
    for (idx, (param_idx, learner)) in learners.iter_mut().enumerate() {
        let fit_state = learner.initialize(&x_col, data)?;
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

        let mut candidate_updates = Vec::new();

        // Stage 1: Find best learner per parameter using RSS on negative gradients
        for k in 0..num_params {
            let mut gradients = family.ngradient(data, &current_predictions, k)?;
            stabilize(&mut gradients, config.stabilization, data.weights());

            let mut best_rss = f64::INFINITY;
            let mut best_update: Option<LearnerUpdate> = None;
            let mut best_u_hat: Option<ndarray::Array1<f64>> = None;
            let mut best_learner_idx = None;

            for cached in cached_learners.iter().filter(|c| c.param_idx == k) {
                let update = cached
                    .fit_state
                    .fit_update(gradients.view(), data.weights().map(|w| w.view()));

                let u_hat = match &update {
                    LearnerUpdate::Linear(coef) => {
                        if let LearnerFit::Linear(state) = &cached.fit_state {
                            state.design.dot(coef)
                        } else {
                            unreachable!()
                        }
                    }
                    LearnerUpdate::Stump {
                        split_val,
                        left_val,
                        right_val,
                    } => {
                        let x_col = data.design().column(0);
                        x_col.mapv(|val| if val <= *split_val { *left_val } else { *right_val })
                    }
                    LearnerUpdate::Tree { node: root, param: _ } => {
                        let mut u_hat = ndarray::Array1::zeros(data.response().len());
                        for i in 0..u_hat.len() {
                            let mut node_ptr = root;
                            loop {
                                match node_ptr {
                                    crate::learner::TreeNode::Leaf { value, .. } => {
                                        u_hat[i] = *value;
                                        break;
                                    }
                                    crate::learner::TreeNode::Split {
                                        feature_idx,
                                        threshold,
                                        left,
                                        right,
                                    } => {
                                        if let LearnerFit::Tree(state) = &cached.fit_state {
                                            let val = state.sorted_features[*feature_idx]
                                                .iter()
                                                .find(|(_, idx)| *idx == i)
                                                .unwrap()
                                                .0;
                                            if val <= *threshold {
                                                node_ptr = left;
                                            } else {
                                                node_ptr = right;
                                            }
                                        } else {
                                            unreachable!()
                                        }
                                    }
                                }
                            }
                        }
                        u_hat
                    }
                };

                let residuals = &gradients - &u_hat;
                let rss = match data.weights() {
                    Some(w) => (&residuals * &residuals * w).sum(),
                    None => (&residuals * &residuals).sum(),
                };

                if rss < best_rss {
                    best_rss = rss;
                    best_update = Some(update);
                    best_u_hat = Some(u_hat);
                    best_learner_idx = Some(cached.learner_idx);
                }
            }

            if let (Some(update), Some(u_hat), Some(l_idx)) =
                (best_update, best_u_hat, best_learner_idx)
            {
                candidate_updates.push((k, l_idx, update, u_hat));
            }
        }

        // Stage 2: Select the candidate that minimizes total NLL
        let mut best_nll = f64::INFINITY;
        let mut selected_candidate = None;

        for (k, l_idx, update, u_hat) in candidate_updates {
            // Apply step length nu
            let step = &u_hat * nu;

            // Temporarily apply update
            current_predictions[k] = &current_predictions[k] + &step;

            let nll = family.nll(data, &current_predictions)?;

            if nll < best_nll {
                best_nll = nll;
                selected_candidate = Some((k, l_idx, update, step));
            }

            // Revert update
            current_predictions[k] = &current_predictions[k] - &(&u_hat * nu);
        }

        if let Some((k, l_idx, update, step)) = selected_candidate {
            current_predictions[k] = &current_predictions[k] + &step;
            let risk_reduction = base_nll - best_nll;

            // Standard boosting applies the update even if empirical NLL increases slightly,
            // but we use max(0.0) for risk_reduction to avoid negative importance.
            let risk_reduction = risk_reduction.max(0.0);

            updates.push(UpdateStep {
                param_idx: k,
                learner_idx: l_idx,
                risk_reduction,
                update: match update {
                    LearnerUpdate::Linear(coef) => LearnerUpdate::Linear(coef * nu),
                    LearnerUpdate::Stump {
                        split_val,
                        left_val,
                        right_val,
                    } => LearnerUpdate::Stump {
                        split_val,
                        left_val: left_val * nu,
                        right_val: right_val * nu,
                    },
                    LearnerUpdate::Tree { mut node, param } => {
                        node.scale(nu);
                        LearnerUpdate::Tree { node, param }
                    }
                },
            });
        }
    }

    let mut fitted = Fitted::new(family, offsets, learners);
    fitted.updates = updates;
    Ok(fitted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Dataset;
    use crate::engine::{Algorithm, Mstop};
    use crate::family::GaussianLss;
    use crate::learner::Linear;
    use ndarray::array;

    #[test]
    fn test_fit_noncyclical_selects_best_param() {
        // x is perfectly correlated with y
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![2.0, 4.0, 6.0, 8.0];
        let data = Dataset::new(x, y, None).unwrap();

        let model = BoostLss::new(GaussianLss::new())
            .on("mu", |p| p.add(Linear::new("x")))
            .unwrap()
            .on("sigma", |p| p.add(Linear::new("x")))
            .unwrap()
            .algorithm(Algorithm::NonCyclic)
            .mstop(Mstop::Scalar(1));

        let fitted = fit_noncyclical(model, &data).unwrap();

        // There should be exactly 1 update since mstop=1
        assert_eq!(fitted.updates.len(), 1);

        // Since mu is far off (mean vs true values), it should be updated to reduce NLL.
        // The single update should target param_idx 0 (mu).
        assert_eq!(fitted.updates[0].param_idx, 0);
    }
}
```

- [ ] **Step 2: Run tests to verify**

Run: `cargo test -p boostlss -- engine::noncyclical::tests`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add crates/boostlss/src/engine/noncyclical.rs
git commit -m "feat: implement NonCyclic boosting loop with two-stage selection"
```
