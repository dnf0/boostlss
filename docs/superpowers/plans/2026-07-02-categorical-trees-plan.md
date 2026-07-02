# Categorical Feature Support in Trees Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement LightGBM-style categorical feature support in `Tree` and `HistTree` learners, where categorical splits route specific category sets (identified by sorting categories by their gradient statistics) to the left or right child.

**Architecture:**
- Add `CategoricalSplit` variant to `TreeNode`.
- Add `categorical_features: Vec<usize>` to `Tree` and `HistTree`.
- Update the recursive `build_tree` methods in `TreeFitState` and `HistTreeFitState` to handle the `categorical_features`. If a feature is categorical, aggregate gradients by category, sort categories by `sum_wu / sum_w`, and find the best subset split point.
- Expose `categorical_features` in Python for `PyTreeLearner` and `PyHistTreeLearner`.

**Tech Stack:** Rust, PyO3, Python, Pytest.

---

### Task 1: Update `TreeNode` and Basic Structs

**Files:**
- Modify: `crates/boostlss/src/learner/tree.rs`
- Modify: `crates/boostlss/src/learner/hist_tree.rs`

- [ ] **Step 1: Write the failing test**

In `crates/boostlss/src/learner/tree.rs`:
```rust
    #[test]
    fn test_categorical_treenode_scale() {
        let mut node = TreeNode::CategoricalSplit {
            feature_idx: 0,
            left_categories: vec![1.0, 3.0],
            left: Box::new(TreeNode::Leaf { value: 2.0, samples: 10 }),
            right: Box::new(TreeNode::Leaf { value: 4.0, samples: 10 }),
        };
        node.scale(0.5);
        if let TreeNode::CategoricalSplit { left, right, .. } = node {
            if let TreeNode::Leaf { value, .. } = *left {
                assert_eq!(value, 1.0);
            }
            if let TreeNode::Leaf { value, .. } = *right {
                assert_eq!(value, 2.0);
            }
        } else {
            panic!("Expected CategoricalSplit");
        }
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run `cargo test -p boostlss test_categorical_treenode_scale`
Expected: FAIL (does not compile)

- [ ] **Step 3: Update `TreeNode`**

In `crates/boostlss/src/learner/tree.rs`:
```rust
pub enum TreeNode {
    Leaf {
        value: f64,
        samples: usize,
    },
    Split {
        feature_idx: usize,
        threshold: f64,
        left: Box<TreeNode>,
        right: Box<TreeNode>,
    },
    CategoricalSplit {
        feature_idx: usize,
        left_categories: Vec<f64>,
        left: Box<TreeNode>,
        right: Box<TreeNode>,
    },
}

impl TreeNode {
    pub fn scale(&mut self, factor: f64) {
        match self {
            TreeNode::Leaf { value, .. } => {
                *value *= factor;
            }
            TreeNode::Split { left, right, .. } => {
                left.scale(factor);
                right.scale(factor);
            }
            TreeNode::CategoricalSplit { left, right, .. } => {
                left.scale(factor);
                right.scale(factor);
            }
        }
    }
}
```

- [ ] **Step 4: Update `Tree` and `HistTree` structs**

In `tree.rs`:
```rust
pub struct Tree {
    pub max_depth: usize,
    pub min_samples_leaf: usize,
    pub feature_indices: Vec<usize>,
    pub categorical_features: Vec<usize>,
}

impl Tree {
    pub fn new(feature_indices: Vec<usize>) -> Self {
        Self {
            max_depth: 3,
            min_samples_leaf: 1,
            feature_indices,
            categorical_features: Vec::new(),
        }
    }

    pub fn categorical_features(mut self, cat_features: Vec<usize>) -> Self {
        self.categorical_features = cat_features;
        self
    }
    // ...
```

In `hist_tree.rs`:
```rust
pub struct HistTree {
    pub feature_indices: Vec<usize>,
    pub max_depth: usize,
    pub min_samples_leaf: usize,
    pub max_bins: usize,
    pub categorical_features: Vec<usize>,
}

impl HistTree {
    pub fn new(feature_indices: Vec<usize>) -> Self {
        Self {
            feature_indices,
            max_depth: 3,
            min_samples_leaf: 1,
            max_bins: 256,
            categorical_features: Vec::new(),
        }
    }

    pub fn categorical_features(mut self, cat_features: Vec<usize>) -> Self {
        self.categorical_features = cat_features;
        self
    }
    // ...
```

- [ ] **Step 5: Run tests**

Run `cargo test -p boostlss test_categorical_treenode_scale`
Expected: PASS
Run `cargo check -p boostlss`

- [ ] **Step 6: Commit**

```bash
git add crates/boostlss/src/learner/tree.rs crates/boostlss/src/learner/hist_tree.rs
git commit -m "feat: add CategoricalSplit to TreeNode and update Tree structs"
```

### Task 2: Implement Categorical Prediction and FitState for `Tree`

**Files:**
- Modify: `crates/boostlss/src/learner/tree.rs`

- [ ] **Step 1: Write prediction test**

In `crates/boostlss/src/learner/tree.rs` tests module:
```rust
    #[test]
    fn test_tree_predict_categorical() {
        let root = TreeNode::CategoricalSplit {
            feature_idx: 0,
            left_categories: vec![1.0, 3.0],
            left: Box::new(TreeNode::Leaf { value: 10.0, samples: 1 }),
            right: Box::new(TreeNode::Leaf { value: 20.0, samples: 1 }),
        };
        let tree = Tree::new(vec![0]).categorical_features(vec![0]);
        let data = crate::data::Dataset::new(array![[1.0], [2.0], [3.0], [4.0]], array![0., 0., 0., 0.], None).unwrap();

        let preds = tree.predict(&root, &data).unwrap();
        assert_eq!(preds[0], 10.0);
        assert_eq!(preds[1], 20.0);
        assert_eq!(preds[2], 10.0);
        assert_eq!(preds[3], 20.0);
    }
```

- [ ] **Step 2: Implement prediction**

In `Tree::predict`:
```rust
                    TreeNode::Split {
                        feature_idx,
                        threshold,
                        left,
                        right,
                    } => {
                        let val = features_data[*feature_idx][i];
                        if val <= *threshold {
                            node_ptr = left;
                        } else {
                            node_ptr = right;
                        }
                    }
                    TreeNode::CategoricalSplit {
                        feature_idx,
                        left_categories,
                        left,
                        right,
                    } => {
                        let val = features_data[*feature_idx][i];
                        if left_categories.contains(&val) {
                            node_ptr = left;
                        } else {
                            node_ptr = right;
                        }
                    }
```

- [ ] **Step 3: Update `TreeFitState`**

Add `categorical_features` to `TreeFitState` and populate it in `build_fit_state`.
```rust
pub struct TreeFitState {
    pub max_depth: usize,
    pub min_samples_leaf: usize,
    pub feature_indices: Vec<usize>,
    pub categorical_features: Vec<usize>, // New field
    pub sorted_features: Vec<Vec<(f64, usize)>>,
}

// In Tree::build_fit_state
        Ok(TreeFitState {
            max_depth: self.max_depth,
            min_samples_leaf: self.min_samples_leaf,
            feature_indices: self.feature_indices.clone(),
            categorical_features: self.categorical_features.clone(), // Set here
            sorted_features,
        })
```

Update mock `TreeFitState` in `test_tree_fit_update` test to include `categorical_features: vec![]`.

- [ ] **Step 4: Run test to verify it passes**

Run `cargo test -p boostlss test_tree_predict_categorical` and `cargo check -p boostlss`

- [ ] **Step 5: Commit**

```bash
git add crates/boostlss/src/learner/tree.rs
git commit -m "feat: implement categorical prediction and state for Tree"
```

### Task 3: Implement Categorical Splitting in `Tree`

**Files:**
- Modify: `crates/boostlss/src/learner/tree.rs`

- [ ] **Step 1: Write the failing test**

```rust
    #[test]
    fn test_tree_categorical_split() {
        let sorted_f0 = vec![(0.0, 0), (1.0, 1), (2.0, 2), (3.0, 3)];
        let state = TreeFitState {
            max_depth: 2,
            min_samples_leaf: 1,
            feature_indices: vec![0],
            categorical_features: vec![0],
            sorted_features: vec![sorted_f0],
        };
        // Categories: 0, 1, 2, 3
        // u values:   10, -10, 10, -10
        // Best split should group (0, 2) together and (1, 3) together.
        let u = array![10.0, -10.0, 10.0, -10.0];

        let update = state.fit_update(u.view(), None);
        if let LearnerUpdate::Tree {
            node: TreeNode::CategoricalSplit { left_categories, .. },
            ..
        } = update
        {
            assert!(left_categories.contains(&0.0) && left_categories.contains(&2.0) || left_categories.contains(&1.0) && left_categories.contains(&3.0));
        } else {
            panic!("Expected CategoricalSplit");
        }
    }
```

- [ ] **Step 2: Implement subset evaluation**

Inside `TreeFitState::build_tree` loop over features:
```rust
        for (feat_idx_out, sorted) in self.sorted_features.iter().enumerate() {
            let feat_idx_in = self.feature_indices[feat_idx_out];
            let is_categorical = self.categorical_features.contains(&feat_idx_in);

            let active_sorted: Vec<&(f64, usize)> = sorted
                .iter()
                .filter(|(_, orig_idx)| active_mask[*orig_idx])
                .collect();

            if is_categorical {
                // Categorical Splitting Logic
                use std::collections::HashMap;

                // 1. Aggregate stats per category
                let mut cat_stats: HashMap<u64, (f64, f64, usize)> = HashMap::new(); // key is val.to_bits() to hash f64 exactly
                for (val, orig_idx) in &active_sorted {
                    let weight = w[**orig_idx];
                    let grad = u[**orig_idx];
                    let entry = cat_stats.entry(val.to_bits()).or_insert((0.0, 0.0, 0));
                    entry.0 += weight;
                    entry.1 += weight * grad;
                    entry.2 += 1;
                }

                // 2. Sort categories by mean gradient
                let mut categories: Vec<(f64, f64, f64, usize)> = cat_stats.into_iter().map(|(bits, (w_sum, wu_sum, count))| {
                    let val = f64::from_bits(bits);
                    let mean_grad = if w_sum > 1e-9 { wu_sum / w_sum } else { 0.0 };
                    (val, w_sum, wu_sum, count)
                }).collect();

                categories.sort_by(|a, b| {
                    let mean_a = if a.1 > 1e-9 { a.2 / a.1 } else { 0.0 };
                    let mean_b = if b.1 > 1e-9 { b.2 / b.1 } else { 0.0 };
                    mean_a.partial_cmp(&mean_b).unwrap_or(std::cmp::Ordering::Equal)
                });

                // 3. Evaluate subset splits
                let mut left_w = 0.0;
                let mut left_wu = 0.0;
                let mut left_count = 0;

                for i in 0..categories.len().saturating_sub(1) {
                    let (_, cat_w, cat_wu, cat_count) = categories[i];
                    left_w += cat_w;
                    left_wu += cat_wu;
                    left_count += cat_count;

                    let right_count = active_sorted.len() - left_count;
                    if left_count < self.min_samples_leaf || right_count < self.min_samples_leaf {
                        continue;
                    }

                    let right_w = total_w - left_w;
                    let right_wu = total_wu - left_wu;

                    if left_w > 1e-9 && right_w > 1e-9 {
                        let gain = (left_wu * left_wu) / left_w + (right_wu * right_wu) / right_w;
                        if gain > best_gain {
                            best_gain = gain;
                            // Collect categories up to i
                            let left_cats: Vec<f64> = categories[0..=i].iter().map(|c| c.0).collect();
                            // Pack this into a new enum variant we define for best_split
                            best_split = Some((feat_idx_out, SplitCandidate::Categorical(left_cats)));
                        }
                    }
                }
            } else {
                // Existing continuous logic...
```

You'll need to define a local enum to hold the split point candidate since it could be continuous or categorical:
```rust
        enum SplitCandidate {
            Continuous(f64),
            Categorical(Vec<f64>),
        }
        let mut best_split: Option<(usize, SplitCandidate)> = None;
```
Update the existing continuous logic to assign `best_split = Some((feat_idx_out, SplitCandidate::Continuous((val + next_val) / 2.0)));`.

And when applying the best split:
```rust
        if let Some((feat_idx_out, split_val)) = best_split {
            let mut left_indices = Vec::new();
            let mut right_indices = Vec::new();

            match split_val {
                SplitCandidate::Continuous(threshold) => {
                    for &(val, idx) in &self.sorted_features[feat_idx_out] {
                        if active_mask[idx] {
                            if val <= threshold { left_indices.push(idx); } else { right_indices.push(idx); }
                        }
                    }
                    let left = Box::new(self.build_tree(depth + 1, &left_indices, u, w));
                    let right = Box::new(self.build_tree(depth + 1, &right_indices, u, w));
                    TreeNode::Split { feature_idx: feat_idx_out, threshold, left, right }
                },
                SplitCandidate::Categorical(left_cats) => {
                    for &(val, idx) in &self.sorted_features[feat_idx_out] {
                        if active_mask[idx] {
                            if left_cats.contains(&val) { left_indices.push(idx); } else { right_indices.push(idx); }
                        }
                    }
                    let left = Box::new(self.build_tree(depth + 1, &left_indices, u, w));
                    let right = Box::new(self.build_tree(depth + 1, &right_indices, u, w));
                    TreeNode::CategoricalSplit { feature_idx: feat_idx_out, left_categories: left_cats, left, right }
                }
            }
        } else { // ... leaf
```

- [ ] **Step 3: Run test**

Run `cargo test -p boostlss test_tree_categorical_split`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add crates/boostlss/src/learner/tree.rs
git commit -m "feat: implement categorical subset evaluation in Tree"
```

### Task 4: Implement Categorical Splitting in `HistTree`

**Files:**
- Modify: `crates/boostlss/src/learner/hist_tree.rs`

- [ ] **Step 1: Update State and Predict**

In `HistTreeFitState` and `build_fit_state`, pass `categorical_features` through:
```rust
pub struct HistTreeFitState {
    // ...
    pub categorical_features: Vec<usize>,
}
```
In `build_fit_state`:
For continuous features, use the existing quantile logic.
For categorical features (`self.categorical_features.contains(&feat_idx_in)`), `thresholds` is NOT populated (or holds the exact unique categories, similar to bins). Skip quantization using quantiles. Just find unique exact values, sort them, and `boundaries` = `unique_vals`.
```rust
            let boundaries = if self.categorical_features.contains(&feat_idx_in) {
                // If there are more than 256 categories, panic or truncate. For now, take first 256.
                let mut b = unique_vals;
                if b.len() > self.max_bins { b.truncate(self.max_bins); }
                b
            } else if unique_vals.len() <= self.max_bins { // ...
```

In `HistTree` prediction logic, you do NOT need to change anything because `predict` relies on `TreeNode` which we already updated. However, the splitting logic must produce `TreeNode::CategoricalSplit`. Wait, `HistTree` doesn't have `predict` itself, it returns `TreeNode` which is evaluated by `BoostLss` via the `BaseLearner` trait `predict` which calls `Tree::predict(node, data)`. Oh wait! `HistTree` uses `TreeNode` and provides `predict` by traversing `TreeNode`.
Actually, `HistTree` does not have its own `predict` method for nodes. Wait, the `BaseLearner` implementation for `HistTree` usually delegates. I'll need to check where `predict` is implemented for `HistTree` in `mod.rs`. Yes, it's evaluated exactly like `Tree`.

- [ ] **Step 2: Write Splitting Logic in `build_tree`**

In `HistTreeFitState::build_tree`, loop over features. Apply similar logic to Task 3 using `SplitCandidate`.
```rust
        enum SplitCandidate {
            Continuous(u8),
            Categorical(Vec<u8>),
        }
```
When evaluating a categorical feature in `HistTreeFitState`, instead of checking bins sequentially `0..active_bins`, you aggregate `(bin_w, bin_wu, bin_count)` per bin, sort the bins by `bin_wu / bin_w`, and do a linear scan to find `best_gain`. The `best_split` will be `SplitCandidate::Categorical(left_bins)`.

Then apply the split:
```rust
                SplitCandidate::Categorical(left_bins) => {
                    for &idx in &active_indices {
                        if left_bins.contains(&self.quantized_data[[idx, feat_idx_out]]) { left_indices.push(idx); } else { right_indices.push(idx); }
                    }
                    // Crucial: Map bin IDs back to original category values for the TreeNode!
                    let left_cats: Vec<f64> = left_bins.iter().map(|&b| self.thresholds[feat_idx_out][b as usize]).collect();
                    let left = Box::new(self.build_tree(left_indices, depth + 1, u, weights));
                    let right = Box::new(self.build_tree(right_indices, depth + 1, u, weights));
                    crate::learner::tree::TreeNode::CategoricalSplit { feature_idx: feat_idx_out, left_categories: left_cats, left, right }
                }
```

- [ ] **Step 3: Test and Commit**

Run `cargo check -p boostlss`
```bash
git add crates/boostlss/src/learner/hist_tree.rs
git commit -m "feat: implement categorical splitting in HistTree"
```

### Task 5: Python API Updates

**Files:**
- Modify: `crates/boostlss-py/src/learner.rs`

- [ ] **Step 1: Update constructors**

In `PyTreeLearner`:
```rust
pub struct PyTreeLearner {
    pub feature_indices: Vec<usize>,
    pub max_depth: usize,
    pub min_samples_leaf: usize,
    pub categorical_features: Vec<usize>, // New field
}

    #[pyo3(signature = (feature_indices, max_depth=3, min_samples_leaf=1, categorical_features=None))]
    fn new(feature_indices: Vec<usize>, max_depth: usize, min_samples_leaf: usize, categorical_features: Option<Vec<usize>>) -> Self {
        Self {
            feature_indices,
            max_depth,
            min_samples_leaf,
            categorical_features: categorical_features.unwrap_or_default(),
        }
    }

// In From<PyTreeLearner>
        tree.categorical_features = val.categorical_features;
```

In `PyHistTreeLearner`:
```rust
pub struct PyHistTreeLearner {
    pub feature_indices: Vec<usize>,
    pub max_bins: usize,
    pub max_depth: usize,
    pub min_samples_leaf: usize,
    pub categorical_features: Vec<usize>, // New field
}

    #[pyo3(signature = (feature_indices, max_bins=256, max_depth=3, min_samples_leaf=1, categorical_features=None))]
    fn new(
        feature_indices: Vec<usize>,
        max_bins: usize,
        max_depth: usize,
        min_samples_leaf: usize,
        categorical_features: Option<Vec<usize>>,
    ) -> Self {
        Self {
            feature_indices,
            max_bins,
            max_depth,
            min_samples_leaf,
            categorical_features: categorical_features.unwrap_or_default(),
        }
    }

// In From<PyHistTreeLearner>
        hist_tree.categorical_features = val.categorical_features;
```

- [ ] **Step 2: Run check and commit**

```bash
cargo check -p boostlss-py
git add crates/boostlss-py/src/learner.rs
git commit -m "feat: expose categorical_features in Python API"
```

### Task 6: Python Integration Tests

**Files:**
- Create: `crates/boostlss-py/tests/test_categorical.py`

- [ ] **Step 1: Write integration tests**

```python
import pytest
import numpy as np
from boostlss_py import BoostLssModel, PyFamily, PyTreeLearner, PyHistTreeLearner

def test_categorical_tree():
    np.random.seed(42)
    # 200 samples, 2 continuous, 1 categorical (index 2)
    X = np.random.randn(200, 3)
    X[:, 2] = np.random.choice([0, 1, 2, 3], size=200)

    # Target depends explicitly on categorical value 2
    y = X[:, 0] + (X[:, 2] == 2) * 5.0 + np.random.randn(200) * 0.1

    family = PyFamily("GaussianLss")

    # Test Exact Tree
    model1 = BoostLssModel(family, mstop=10, step_length=0.1)
    learner1 = PyTreeLearner([0, 1, 2], categorical_features=[2])
    model1.add_learner("mu", learner1)
    model1.fit(X, y)
    preds1 = model1.predict(X)["mu"]
    assert len(preds1) == 200

    # Test Hist Tree
    model2 = BoostLssModel(family, mstop=10, step_length=0.1)
    learner2 = PyHistTreeLearner([0, 1, 2], categorical_features=[2])
    model2.add_learner("mu", learner2)
    model2.fit(X, y)
    preds2 = model2.predict(X)["mu"]
    assert len(preds2) == 200
```

- [ ] **Step 2: Run test and commit**

```bash
uv run pytest crates/boostlss-py/tests/test_categorical.py
git add crates/boostlss-py/tests/test_categorical.py
git commit -m "test: add python tests for categorical trees"
```
