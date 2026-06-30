# HistTree Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a high-speed Histogram Tree base learner (`HistTree`) to scale the BoostLSS engine to millions of rows.

**Architecture:** We will create a completely new BaseLearner called `HistTree` that holds quantized `u8` data and original float thresholds. Node splitting will populate a `max_bins` sized histogram in $O(N)$ and find splits in $O(\text{num\_bins})$.

**Tech Stack:** Rust (core engine) and PyO3 (Python wrappers).

---

### Task 1: Create the HistTree Structs and Enum Wiring

**Files:**

- Modify: `crates/boostlss/src/learner/mod.rs`
- Create: `crates/boostlss/src/learner/hist_tree.rs`

- [ ] **Step 1: Write the failing test**

```rust
// In crates/boostlss/src/learner/hist_tree.rs
#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Dataset;
    use ndarray::{array, Array2};

    #[test]
    fn test_hist_tree_initialization() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let data = Dataset::new(x, y.clone(), None, None).unwrap();

        let hist_tree = HistTree::new(vec![0]).max_bins(3);
        let state = hist_tree.build_fit_state(&data).unwrap();

        // Assert we have quantized data and thresholds
        assert_eq!(state.quantized_data.shape(), &[5, 1]);
        assert_eq!(state.thresholds.len(), 1);
        assert!(state.thresholds[0].len() <= 3);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run cargo test --lib learner::hist_tree::tests::test_hist_tree_initialization`
Expected: FAIL with "unresolved import" or "cannot find value"

- [ ] **Step 3: Write minimal implementation**

In `crates/boostlss/src/learner/hist_tree.rs`:

```rust
use crate::data::Dataset;
use ndarray::Array2;
use std::collections::BTreeSet;

#[derive(Clone, Debug)]
pub struct HistTree {
    pub feature_indices: Vec<usize>,
    pub max_depth: usize,
    pub min_samples_leaf: usize,
    pub max_bins: usize,
}

impl HistTree {
    pub fn new(feature_indices: Vec<usize>) -> Self {
        Self {
            feature_indices,
            max_depth: 3,
            min_samples_leaf: 1,
            max_bins: 256,
        }
    }

    pub fn max_bins(mut self, max_bins: usize) -> Self {
        self.max_bins = max_bins;
        self
    }

    pub fn build_fit_state(&self, data: &Dataset) -> Result<HistTreeFitState, crate::error::BoostlssError> {
        let n_obs = data.n_obs();
        let mut quantized_data = Array2::<u8>::zeros((n_obs, self.feature_indices.len()));
        let mut thresholds = Vec::with_capacity(self.feature_indices.len());

        for (feat_idx_out, &feat_idx_in) in self.feature_indices.iter().enumerate() {
            let col = data.x().column(feat_idx_in);

            // 1. Get exact unique sorted values
            let mut unique_vals: Vec<f64> = col.iter().copied().collect::<BTreeSet<_>>().into_iter().collect();

            // 2. Sample to get quantiles if there are too many unique values
            let boundaries = if unique_vals.len() <= self.max_bins {
                unique_vals
            } else {
                let step = unique_vals.len() as f64 / self.max_bins as f64;
                (0..self.max_bins).map(|i| {
                    let idx = (i as f64 * step) as usize;
                    unique_vals[idx.min(unique_vals.len() - 1)]
                }).collect::<BTreeSet<_>>().into_iter().collect() // ensure uniqueness and sorted
            };

            thresholds.push(boundaries.clone());

            // 3. Map continuous data to bins (0 to len-1)
            for (i, &val) in col.iter().enumerate() {
                // Find the first boundary that is > val.
                // Since `partition_point` returns the index of the first element not satisfying the predicate (i.e. >= val),
                // we actually want the index of the interval it falls into.
                // A simpler binary search:
                let bin = boundaries.binary_search_by(|v| v.partial_cmp(&val).unwrap())
                                   .unwrap_or_else(|e| e);
                // Clamp bin to max u8
                quantized_data[[i, feat_idx_out]] = bin.min(255) as u8;
            }
        }

        Ok(HistTreeFitState {
            feature_indices: self.feature_indices.clone(),
            quantized_data,
            thresholds,
            max_depth: self.max_depth,
            min_samples_leaf: self.min_samples_leaf,
        })
    }
}

#[derive(Clone, Debug)]
pub struct HistTreeFitState {
    pub feature_indices: Vec<usize>,
    pub quantized_data: Array2<u8>,
    pub thresholds: Vec<Vec<f64>>,
    pub max_depth: usize,
    pub min_samples_leaf: usize,
}
```

In `crates/boostlss/src/learner/mod.rs`: Add `HistTree` to the `BaseLearner` and `LearnerFitState` enums exactly as `Tree` is structured.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run cargo test`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/boostlss/src/learner/
git commit -m "feat(engine): add HistTree base learner and quantization logic"
```

### Task 2: Implement Node Splitting and Histogram Accumulation

**Files:**

- Modify: `crates/boostlss/src/learner/hist_tree.rs`

- [ ] **Step 1: Write the failing test**

```rust
// In crates/boostlss/src/learner/hist_tree.rs
#[cfg(test)]
mod tree_build_tests {
    use super::*;
    use ndarray::{array, Array1};

    #[test]
    fn test_hist_tree_fit() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let data = crate::data::Dataset::new(x, y.clone(), None, None).unwrap();

        let hist_tree = HistTree::new(vec![0]).max_bins(5).max_depth(1);
        let state = hist_tree.build_fit_state(&data).unwrap();

        let u = array![-1.0, -1.0, 0.0, 1.0, 1.0]; // pseudo-residuals
        let weights = Array1::ones(5);

        let update = state.fit_update(&u.view(), &weights.view()).unwrap();

        // Assert we built a tree update
        match update {
            crate::learner::LearnerUpdate::HistTree { node, .. } => {
                assert!(node.left.is_some());
                assert!(node.right.is_some());
            },
            _ => panic!("Wrong update type"),
        }
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run cargo test`
Expected: FAIL because `fit_update` is not implemented on `HistTreeFitState`.

- [ ] **Step 3: Write minimal implementation**

Implement `fit_update` on `HistTreeFitState` and recursive `build_tree` logic using histograms.

```rust
impl HistTreeFitState {
    pub fn fit_update(
        &self,
        u: &ndarray::ArrayView1<f64>,
        weights: &ndarray::ArrayView1<f64>,
    ) -> Result<crate::learner::LearnerUpdate, crate::error::BoostlssError> {
        let active_indices: Vec<usize> = (0..u.len()).filter(|&i| weights[i] > 0.0).collect();
        let root = self.build_tree(active_indices, 0, u, weights)?;
        Ok(crate::learner::LearnerUpdate::HistTree { node: root, feature_idx: self.feature_indices[0] }) // just root feature for now
    }

    fn build_tree(
        &self,
        active_indices: Vec<usize>,
        depth: usize,
        u: &ndarray::ArrayView1<f64>,
        weights: &ndarray::ArrayView1<f64>,
    ) -> Result<crate::learner::tree::TreeNode, crate::error::BoostlssError> {
        let mut total_w = 0.0;
        let mut total_wu = 0.0;
        for &idx in &active_indices {
            total_w += weights[idx];
            total_wu += weights[idx] * u[idx];
        }

        let mut node = crate::learner::tree::TreeNode {
            is_leaf: true,
            value: if total_w > 0.0 { total_wu / total_w } else { 0.0 },
            feature_idx: 0,
            threshold: 0.0,
            left: None,
            right: None,
            indices: active_indices.clone(),
        };

        if depth >= self.max_depth || active_indices.len() < self.min_samples_leaf * 2 {
            return Ok(node);
        }

        let mut best_gain = 0.0;
        let mut best_split: Option<(usize, u8, f64, f64, Vec<usize>, Vec<usize>)> = None;

        for (feat_out, _) in self.feature_indices.iter().enumerate() {
            let max_bin = self.thresholds[feat_out].len();
            // 1. Build Histogram
            let mut hist_w = vec![0.0; max_bin + 1];
            let mut hist_wu = vec![0.0; max_bin + 1];

            for &idx in &active_indices {
                let bin = self.quantized_data[[idx, feat_out]] as usize;
                hist_w[bin] += weights[idx];
                hist_wu[bin] += weights[idx] * u[idx];
            }

            // 2. Scan Histogram for split
            let mut left_w = 0.0;
            let mut left_wu = 0.0;

            for bin in 0..max_bin {
                left_w += hist_w[bin];
                left_wu += hist_wu[bin];

                let right_w = total_w - left_w;
                let right_wu = total_wu - left_wu;

                // Enforce min child weight implicitly here (can add strict min_child_weight later)
                if left_w <= 1e-7 || right_w <= 1e-7 {
                    continue;
                }

                let gain = (left_wu * left_wu / left_w) + (right_wu * right_wu / right_w) - (total_wu * total_wu / total_w);

                if gain > best_gain {
                    best_gain = gain;

                    // Split the indices explicitly for the children
                    let mut left_indices = Vec::with_capacity(active_indices.len() / 2);
                    let mut right_indices = Vec::with_capacity(active_indices.len() / 2);

                    for &idx in &active_indices {
                        if self.quantized_data[[idx, feat_out]] <= bin as u8 {
                            left_indices.push(idx);
                        } else {
                            right_indices.push(idx);
                        }
                    }

                    best_split = Some((
                        self.feature_indices[feat_out],
                        bin as u8,
                        left_w,
                        right_w,
                        left_indices,
                        right_indices
                    ));
                }
            }
        }

        if let Some((best_feat, best_bin, _lw, _rw, left_idx, right_idx)) = best_split {
            if left_idx.len() >= self.min_samples_leaf && right_idx.len() >= self.min_samples_leaf {
                node.is_leaf = false;
                node.feature_idx = best_feat;
                // De-quantize threshold:
                let feat_out_idx = self.feature_indices.iter().position(|&x| x == best_feat).unwrap();
                node.threshold = self.thresholds[feat_out_idx][best_bin as usize];

                node.left = Some(Box::new(self.build_tree(left_idx, depth + 1, u, weights)?));
                node.right = Some(Box::new(self.build_tree(right_idx, depth + 1, u, weights)?));
            }
        }

        Ok(node)
    }
}
```

_(Ensure `LearnerUpdate::HistTree` is also defined in `mod.rs` and has a `predict` mapping identical to `Tree`)._

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run cargo test`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/boostlss/src/learner/
git commit -m "feat(engine): implement histogram accumulation and split finding"
```

### Task 3: Python Bindings and Integration

**Files:**

- Modify: `crates/boostlss-py/src/learner.rs`
- Modify: `crates/boostlss-py/src/model.rs` (ensure `HistTree` is match-handled appropriately if needed)
- Create: `crates/boostlss-py/tests/test_histtree.py`

- [ ] **Step 1: Write the failing test**

```python
# In crates/boostlss-py/tests/test_histtree.py
import pytest
import numpy as np
from boostlss_py import BoostLssModel, GaussianLss, HistTree

def test_histtree_learner():
    X = np.random.normal(size=(1000, 2))
    y = X[:, 0] * 2.0 + np.random.normal(size=1000) * 0.1

    model = BoostLssModel(GaussianLss(), mstop=10, step_length=0.1)
    model.add_learner("mu", HistTree(0, max_bins=64, max_depth=3))
    model.add_learner("sigma", HistTree(0, max_bins=64, max_depth=3))

    model.fit(X, y)
    preds = model.predict(X, "mu")
    assert len(preds) == 1000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest crates/boostlss-py/tests/test_histtree.py`
Expected: FAIL due to `HistTree` not being available in python.

- [ ] **Step 3: Write minimal implementation**

In `crates/boostlss-py/src/learner.rs`:

```rust
#[pyclass(extends=PyBaseLearner, subclass)]
pub struct HistTree {
    pub inner: boostlss::learner::hist_tree::HistTree,
}

#[pymethods]
impl HistTree {
    #[new]
    #[pyo3(signature = (feature_idx, max_depth=3, min_samples_leaf=1, max_bins=256))]
    fn new(feature_idx: usize, max_depth: usize, min_samples_leaf: usize, max_bins: usize) -> (Self, PyBaseLearner) {
        let inner = boostlss::learner::hist_tree::HistTree::new(vec![feature_idx])
            .max_depth(max_depth)
            .min_samples_leaf(min_samples_leaf)
            .max_bins(max_bins);
        (
            Self { inner },
            PyBaseLearner {
                learner_type: "HistTree".to_string(),
                feature_indices: vec![feature_idx],
            },
        )
    }
}

// In the `impl_into_base_learner` macro logic or manually implement:
impl From<HistTree> for boostlss::learner::BaseLearner {
    fn from(l: HistTree) -> Self {
        boostlss::learner::BaseLearner::HistTree(l.inner)
    }
}

// Add the `HistTree` class to the module initialization in lib.rs
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run maturin develop && uv run pytest crates/boostlss-py/tests/test_histtree.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/boostlss-py/
git commit -m "feat(python): expose HistTree base learner to python"
```
