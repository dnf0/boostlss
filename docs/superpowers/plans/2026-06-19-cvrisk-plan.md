# cvrisk Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the cross-validation and risk tuning module (`cvrisk`) allowing cyclical and non-cyclical risk evaluation over grid points, and expose it to Python.

**Architecture:** A standalone `cv.rs` module that takes a `BoostLss` model and iterates over folds. For Cyclical algorithms, it will build a multidimensional grid and clone+fit the model per grid point. We will add `rand` for sampling and `rayon` as an optional feature for parallelizing fold evaluation.

**Tech Stack:** Rust (`ndarray`, `rand`, `rayon`), Python (`pyo3`).

---

### Task 1: Scaffolding `cv.rs`, Dependencies, and `Resampling`

**Files:**
- Modify: `crates/boostlss/Cargo.toml`
- Modify: `crates/boostlss/src/lib.rs`
- Create: `crates/boostlss/src/cv.rs`

- [ ] **Step 1: Add `rand` dependency**
Modify `crates/boostlss/Cargo.toml` to add `rand`:
```toml
[dependencies]
faer = "0.24.0"
ndarray = "0.17.2"
statrs = "0.18.0"
thiserror = "2.0.18"
rand = "0.8.5"
```

- [ ] **Step 2: Expose `cv` module**
Modify `crates/boostlss/src/lib.rs` to include `pub mod cv;`.

- [ ] **Step 3: Create `cv.rs` and `Resampling` enum**
Create `crates/boostlss/src/cv.rs` with the resampling variants and a failing test for it.
```rust
use ndarray::Array1;
use rand::Rng;

pub enum Resampling {
    Bootstrap { b: usize },
    KFold { k: usize },
    Subsampling { b: usize, prob: f64 },
}

impl Resampling {
    pub fn generate_weights(&self, n: usize, rng: &mut impl Rng) -> Vec<Array1<f64>> {
        unimplemented!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_kfold_weights() {
        let resampling = Resampling::KFold { k: 2 };
        let mut rng = StdRng::seed_from_u64(42);
        let weights = resampling.generate_weights(10, &mut rng);
        assert_eq!(weights.len(), 2);
        assert_eq!(weights[0].sum(), 5.0);
        assert_eq!(weights[1].sum(), 5.0);
    }
}
```

- [ ] **Step 4: Run the test to verify it fails**
Run: `cargo test -p boostlss -- cv::tests::test_kfold_weights`
Expected: FAIL with `unimplemented!`

- [ ] **Step 5: Implement `generate_weights`**
Implement the minimal code for `KFold`, `Bootstrap`, and `Subsampling` to generate the `Array1<f64>` weights.
For KFold: split `n` into `k` groups, assigning weight 0 to validation indices and 1 to training indices.
For Bootstrap: for `b` iterations, draw `n` items with replacement, tracking integer weight multiplicities.
For Subsampling: for `b` iterations, draw `n * prob` items without replacement, assigning weight 1 to drawn and 0 to others. (You can use `rand::seq::index::sample` for drawing without replacement).

- [ ] **Step 6: Run tests to verify passing**
Run: `cargo test -p boostlss -- cv`
Expected: PASS

- [ ] **Step 7: Commit**
```bash
git add crates/boostlss/Cargo.toml crates/boostlss/src/lib.rs crates/boostlss/src/cv.rs
git commit -m "feat: setup cv.rs and Resampling"
```

### Task 2: Implement Grid Generation for Cyclical

**Files:**
- Modify: `crates/boostlss/src/cv.rs`
- Modify: `crates/boostlss/src/model.rs` (to ensure `BoostLss` implements `Clone`)

- [ ] **Step 1: Ensure `BoostLss` is Cloneable**
In `crates/boostlss/src/model.rs` and `crates/boostlss/src/engine/mod.rs` (`Config`, `Mstop`, `Algorithm`), derive `Clone` for `BoostLss` and all its contained structs. This is necessary so `cvrisk` can clone the model for each grid point. `BaseLearner` and `Family` types already have or need `Clone`.
Run `cargo check -p boostlss` to ensure all bounds are satisfied.

- [ ] **Step 2: Write failing test for grid generation**
In `crates/boostlss/src/cv.rs`, add:
```rust
use crate::engine::Mstop;

pub fn make_grid(params_count: usize, mstop_max: usize, length_out: usize) -> Vec<Mstop> {
    unimplemented!()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_make_grid() {
        let grid = make_grid(2, 10, 3);
        // length_out = 3, min = 1, max = 10. log-spaced rounded: 1, 3, 10.
        // grid size = 3 * 3 = 9
        assert_eq!(grid.len(), 9);
        assert!(matches!(&grid[0], Mstop::PerParam(v) if v == &vec![1, 1]));
    }
}
```

- [ ] **Step 3: Run the test to verify it fails**
Run: `cargo test -p boostlss -- cv::tests::test_make_grid`
Expected: FAIL with `unimplemented!`

- [ ] **Step 4: Implement `make_grid`**
Write `make_grid` to generate `length_out` log-spaced integers between 1 and `mstop_max` for each parameter. Then, compute the Cartesian product of these values across the `params_count` dimensions. Return a `Vec<Mstop>` where each `Mstop` is a `Mstop::PerParam(config)`.

- [ ] **Step 5: Run tests to verify passing**
Run: `cargo test -p boostlss -- cv::tests::test_make_grid`
Expected: PASS

- [ ] **Step 6: Commit**
```bash
git add crates/boostlss/src/cv.rs crates/boostlss/src/model.rs crates/boostlss/src/engine/mod.rs
git commit -m "feat: implement cvrisk grid generation and clone bounds"
```

### Task 3: Implement `CvRisk` runner for Cyclical

**Files:**
- Modify: `crates/boostlss/src/cv.rs`
- Modify: `crates/boostlss/src/model.rs` (ensure `BoostLss` can get number of params)

- [ ] **Step 1: Write failing test for `CvRisk`**
Add to `crates/boostlss/src/cv.rs`:
```rust
use crate::data::Dataset;
use crate::error::BoostlssError;
use crate::family::Family;
use crate::model::BoostLss;

pub struct CvRiskResult {
    pub optimal_mstop: Mstop,
    pub grid: Vec<Mstop>,
    pub mean_risk: Vec<f64>,
    pub risk_path_per_fold: Vec<Vec<f64>>,
}

pub struct CvRisk<F: Family> {
    model: BoostLss<F>,
    resampling: Resampling,
}

impl<F: Family + Clone> CvRisk<F> {
    pub fn new(model: BoostLss<F>) -> Self {
        Self { model, resampling: Resampling::Bootstrap { b: 25 } }
    }

    pub fn resampling(mut self, resampling: Resampling) -> Self {
        self.resampling = resampling;
        self
    }

    pub fn run(&self, dataset: &Dataset) -> Result<CvRiskResult, BoostlssError> {
        unimplemented!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::GaussianLss;
    use ndarray::{Array1, Array2};
    use crate::engine::Algorithm;

    #[test]
    fn test_cvrisk_run() {
        let data = Dataset::new(Array2::zeros((10, 2)), Array1::zeros(10), None).unwrap();
        let model = BoostLss::new(GaussianLss::new()).algorithm(Algorithm::Cyclic);
        let cv = CvRisk::new(model).resampling(Resampling::KFold { k: 2 });
        let result = cv.run(&data).unwrap();
        assert!(!result.grid.is_empty());
    }
}
```

- [ ] **Step 2: Run test to verify it fails**
Run: `cargo test -p boostlss -- cv::tests::test_cvrisk_run`
Expected: FAIL with `unimplemented!`

- [ ] **Step 3: Implement `CvRisk::run`**
Implement the evaluation for `Algorithm::Cyclic`:
1. Use `make_grid` to generate the grid. Assume `mstop_max` is extracted from the model's `config.mstop`.
2. Generate fold weights using `resampling.generate_weights`.
3. Loop over folds, and for each fold loop over the grid points.
4. For each grid point, clone the `model` and `dataset`. Update the model's `mstop` to the grid point, and set the dataset weights to the fold's in-bag weights.
5. Fit the cloned model using `crate::engine::cyclical::fit_cyclical`.
6. Calculate the Out-Of-Bag (OOB) risk by evaluating the fitted model on the original dataset, filtering for observations where the fold weight is 0. (For now, you can just call `family.risk(...)` on the OOB rows).
7. Aggregate the risk into `mean_risk` and find `optimal_mstop`.

- [ ] **Step 4: Run test to verify it passes**
Run: `cargo test -p boostlss -- cv::tests::test_cvrisk_run`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add crates/boostlss/src/cv.rs
git commit -m "feat: implement cvrisk run for cyclical algorithm"
```

### Task 4: Parallelization with `rayon`

**Files:**
- Modify: `crates/boostlss/Cargo.toml`
- Modify: `crates/boostlss/src/cv.rs`

- [ ] **Step 1: Add `rayon` behind a `parallel` feature**
Modify `crates/boostlss/Cargo.toml`:
```toml
[dependencies]
rayon = { version = "1.10.0", optional = true }

[features]
parallel = ["rayon"]
```

- [ ] **Step 2: Use `par_iter` when feature is enabled**
In `crates/boostlss/src/cv.rs`, refactor the fold evaluation loop:
```rust
#[cfg(feature = "parallel")]
use rayon::prelude::*;

// ... inside run()
#[cfg(feature = "parallel")]
let fold_results: Vec<Vec<f64>> = weights_list.par_iter().map(|w| { ... }).collect();

#[cfg(not(feature = "parallel"))]
let fold_results: Vec<Vec<f64>> = weights_list.iter().map(|w| { ... }).collect();
```
(Adjust variable names as appropriate for your implementation).

- [ ] **Step 3: Verify it compiles**
Run: `cargo check -p boostlss --features parallel`
Expected: PASS

- [ ] **Step 4: Commit**
```bash
git add crates/boostlss/Cargo.toml crates/boostlss/src/cv.rs
git commit -m "feat: add optional parallel cvrisk execution using rayon"
```

### Task 5: Python Bindings for `cvrisk`

**Files:**
- Modify: `crates/boostlss-py/src/model.rs`
- Modify: `crates/boostlss-py/tests/test_basic.py`

- [ ] **Step 1: Write failing python test**
Add to `crates/boostlss-py/tests/test_basic.py`:
```python
def test_cvrisk():
    import numpy as np
    from boostlss import BoostLssModel

    X = np.zeros((20, 2))
    y = np.zeros(20)

    model = BoostLssModel(family="GaussianLSS", mstop=10)
    model.fit(X, y)

    res = model.cvrisk(folds=2)
    assert res is not None
    assert "optimal_mstop" in res
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest crates/boostlss-py/tests/test_basic.py::test_cvrisk`
Expected: FAIL because `cvrisk` does not exist on `BoostLssModel`.

- [ ] **Step 3: Expose `cvrisk` in Python**
In `crates/boostlss-py/src/model.rs`, implement `cvrisk` on `BoostLssModel`.
Note that `cvrisk` needs to reconstruct the `Dataset` and `BoostLss` model just like `fit` did, because `CvRisk::run` operates on the unfitted `BoostLss` model, not the `Fitted` one.
Alternatively, save the initialized dataset and model in `BoostLssModel` when `fit` is called, or allow `cvrisk` to be called with `X` and `y`.
The CRAN package `cvrisk` takes a fitted model, so the `BoostLssModel` struct should store the `Dataset` and the unfitted `BoostLss` internally after `fit()` is called, so `cvrisk` can re-use them.
```rust
// Adjust BoostLssModel fields to store the Dataset and BoostLss
// Implement cvrisk:
fn cvrisk(&mut self, py: Python<'_>, folds: usize) -> PyResult<PyObject> {
    // call CvRisk::new(self.model.clone()).resampling(Resampling::KFold{k: folds}).run(&self.dataset)
    // return dict with optimal_mstop
}
```
*Note: Due to Rust ownership, `BoostLssModel` might need a bit of restructuring to keep the unfitted model. Implement whatever is minimal to pass the test.*

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest crates/boostlss-py/tests/test_basic.py::test_cvrisk`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add crates/boostlss-py/src/model.rs crates/boostlss-py/tests/test_basic.py
git commit -m "feat: python bindings for cvrisk"
```
