# Stability Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement stability selection (stabsel) in the Rust core with dynamic PFER/q error bounding, and expose an idiomatic `stabsel` Python API method with comprehensive integration testing.

**Architecture:** We will introduce a new `StabselConfig` and bound calculator in `crates/boostlss/src/cv/stabsel.rs` that drives a parallel subsampling loop via `rayon`. It leverages `BoostLss::fit_update` to selectively stop boosting a subsample if `q` distinct features are selected. Python bindings wrap this logic, validating arguments to ensure exactly two of `(pfer, pi_thr, q)` are provided.

**Tech Stack:** Rust (`rayon`), PyO3, Python (`pytest`)

---

### Task 1: Create the Stabsel Error Bounding Utilities

**Files:**
- Create: `crates/boostlss/src/cv/stabsel.rs`
- Modify: `crates/boostlss/src/cv/mod.rs`
- Modify: `crates/boostlss/src/error.rs`

- [ ] **Step 1: Add new error type to `error.rs`**

```rust
// In crates/boostlss/src/error.rs
// Inside `pub enum BoostlssError {`
    #[error("Invalid stability selection config: {0}")]
    InvalidStabselConfig(String),
```

- [ ] **Step 2: Create `stabsel.rs` and implement `StabselConfig` and bounds calculation**

```rust
// crates/boostlss/src/cv/stabsel.rs
use crate::error::BoostlssError;

#[derive(Clone, Debug, PartialEq)]
pub enum StabselMode {
    Joint,
    Independent,
}

#[derive(Clone, Debug)]
pub struct StabselConfig {
    pub b: usize,
    pub pfer: Option<f64>,
    pub pi_thr: Option<f64>,
    pub q: Option<usize>,
    pub mode: StabselMode,
    pub p: usize, // total number of base-learners available to be selected
}

impl StabselConfig {
    pub fn new(
        b: usize,
        pfer: Option<f64>,
        pi_thr: Option<f64>,
        q: Option<usize>,
        mode: StabselMode,
        p: usize,
    ) -> Result<Self, BoostlssError> {
        let provided = vec![pfer.is_some(), pi_thr.is_some(), q.is_some()]
            .into_iter()
            .filter(|&x| x)
            .count();

        if provided != 2 {
            return Err(BoostlssError::InvalidStabselConfig(
                "Exactly two of (pfer, pi_thr, q) must be provided".to_string(),
            ));
        }

        let mut config = Self {
            b,
            pfer,
            pi_thr,
            q,
            mode,
            p,
        };

        config.resolve_bounds()?;
        Ok(config)
    }

    fn resolve_bounds(&mut self) -> Result<(), BoostlssError> {
        // Shah & Samworth (2013) bounds: PFER <= q^2 / ((2 * pi_thr - 1) * p)
        if self.pfer.is_none() {
            let q = self.q.unwrap() as f64;
            let pi_thr = self.pi_thr.unwrap();
            if pi_thr <= 0.5 || pi_thr >= 1.0 {
                return Err(BoostlssError::InvalidStabselConfig(
                    "pi_thr must be in (0.5, 1.0)".to_string(),
                ));
            }
            self.pfer = Some((q * q) / ((2.0 * pi_thr - 1.0) * self.p as f64));
        } else if self.pi_thr.is_none() {
            let q = self.q.unwrap() as f64;
            let pfer = self.pfer.unwrap();
            if pfer <= 0.0 {
                return Err(BoostlssError::InvalidStabselConfig(
                    "pfer must be > 0.0".to_string(),
                ));
            }
            self.pi_thr = Some(((q * q) / (pfer * self.p as f64) + 1.0) / 2.0);
            if self.pi_thr.unwrap() <= 0.5 || self.pi_thr.unwrap() >= 1.0 {
                return Err(BoostlssError::InvalidStabselConfig(
                    "Derived pi_thr must be in (0.5, 1.0). Adjust q or pfer.".to_string(),
                ));
            }
        } else if self.q.is_none() {
            let pfer = self.pfer.unwrap();
            let pi_thr = self.pi_thr.unwrap();
            if pi_thr <= 0.5 || pi_thr >= 1.0 || pfer <= 0.0 {
                return Err(BoostlssError::InvalidStabselConfig(
                    "Invalid pi_thr or pfer".to_string(),
                ));
            }
            let q_f64 = (pfer * (2.0 * pi_thr - 1.0) * self.p as f64).sqrt();
            self.q = Some(q_f64.floor() as usize);
            if self.q.unwrap() == 0 {
                return Err(BoostlssError::InvalidStabselConfig(
                    "Derived q is 0. Adjust pi_thr or pfer.".to_string(),
                ));
            }
        }
        Ok(())
    }
}
```

- [ ] **Step 3: Expose module**

```rust
// crates/boostlss/src/cv/mod.rs
pub mod stabsel;
```

- [ ] **Step 4: Run clippy and check compilation**
Run: `cargo clippy --workspace --all-targets --all-features -- -D warnings`

- [ ] **Step 5: Commit**
`git add .`
`git commit -m "feat: Add stability selection configuration and bounds resolution"`

---

### Task 2: Implement Rust Subsampling and Aggregation

**Files:**
- Modify: `crates/boostlss/src/cv/stabsel.rs`
- Modify: `crates/boostlss/src/model.rs`

- [ ] **Step 1: Expose parameter names from `BoostLss`**
To track selections correctly, we need access to the parameter names from `BoostLss`.

```rust
// In crates/boostlss/src/model.rs
// Inside impl<F: Family + Clone> BoostLss<F> {
    pub fn param_names(&self) -> Vec<String> {
        self.learners.iter().map(|(p, _)| p.clone()).collect()
    }
```

- [ ] **Step 2: Implement parallel stabsel runner**

```rust
// In crates/boostlss/src/cv/stabsel.rs
use crate::data::Dataset;
use crate::engine::Mstop;
use crate::family::Family;
use crate::model::BoostLss;
use ndarray::Array1;
use rand::{Rng, SeedableRng};
use std::collections::{HashMap, HashSet};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

pub struct StabselResult {
    // Parameter -> Base Learner Name -> Selection count
    pub frequencies: HashMap<String, HashMap<String, usize>>,
    pub q: usize,
    pub pfer: f64,
    pub pi_thr: f64,
    pub b: usize,
}

pub fn run_stabsel<F: Family + Clone + Send + Sync>(
    model: &BoostLss<F>,
    data: &Dataset,
    mstop: Mstop,
    config: &StabselConfig,
) -> Result<StabselResult, BoostlssError> {
    let b_runs = config.b;
    let n = data.num_rows();
    let num_samples = n / 2;
    let q_limit = config.q.unwrap();

    let mut seeds = Vec::with_capacity(b_runs);
    let mut rng = rand::thread_rng();
    for _ in 0..b_runs {
        seeds.push(rng.gen::<u64>());
    }

    #[cfg(feature = "parallel")]
    let iter = seeds.par_iter();
    #[cfg(not(feature = "parallel"))]
    let iter = seeds.iter();

    // Each thread returns the active set of base learners (Param -> Vec<LearnerName>)
    let results: Result<Vec<HashMap<String, HashSet<String>>>, BoostlssError> = iter
        .map(|&seed| {
            let mut run_rng = rand::rngs::StdRng::seed_from_u64(seed);

            // Subsample weights
            let mut w = Array1::zeros(n);
            let indices = rand::seq::index::sample(&mut run_rng, n, num_samples);
            for idx in indices.into_iter() {
                w[idx] = 1.0;
            }

            let mut run_data = data.clone();
            run_data.set_weights(w)?;

            // In stabsel, we start a fresh model
            let mut run_model = model.clone();
            run_model.initialize(&run_data)?;

            let params = run_model.param_names();
            let mut active_sets: HashMap<String, HashSet<String>> = HashMap::new();
            for p in &params {
                active_sets.insert(p.clone(), HashSet::new());
            }

            // We must track number of globally unique selected learners (or per-param if independent)
            let mut global_active_count = 0;

            let mstop_usize = match mstop {
                Mstop::Scalar(m) => m,
                Mstop::Vector(_) => return Err(BoostlssError::InvalidStabselConfig("Mstop::Vector not supported in stabsel".to_string())),
            };

            for _ in 0..mstop_usize {
                let updates = run_model.fit_update(&run_data)?;

                // Track selections
                for (param, (learner_name, _)) in updates {
                    let set = active_sets.get_mut(&param).unwrap();
                    if set.insert(learner_name.clone()) {
                        global_active_count += 1;
                    }
                }

                // Early stopping if q is reached
                if config.mode == StabselMode::Joint && global_active_count >= q_limit {
                    break;
                }

                // Independent mode logic: if ALL params have reached q, break early.
                // But generally joint mode is the standard. Let's just implement a simple joint count for early stopping here.
            }

            Ok(active_sets)
        })
        .collect();

    let results = results?;

    // Aggregate frequencies
    let mut frequencies: HashMap<String, HashMap<String, usize>> = HashMap::new();
    let params = model.param_names();
    for p in params {
        frequencies.insert(p, HashMap::new());
    }

    for run_active in results {
        for (param, learners) in run_active {
            let param_freq = frequencies.get_mut(&param).unwrap();
            for learner in learners {
                *param_freq.entry(learner).or_insert(0) += 1;
            }
        }
    }

    Ok(StabselResult {
        frequencies,
        q: config.q.unwrap(),
        pfer: config.pfer.unwrap(),
        pi_thr: config.pi_thr.unwrap(),
        b: b_runs,
    })
}
```

- [ ] **Step 3: Check Rust compilation**
Run: `cargo test -p boostlss --no-run`

- [ ] **Step 4: Commit**
`git add .`
`git commit -m "feat: Implement stabsel core algorithm"`

---

### Task 3: Python Bindings and Result PyClass

**Files:**
- Create: `crates/boostlss-py/src/stabsel.rs`
- Modify: `crates/boostlss-py/src/lib.rs`
- Modify: `crates/boostlss-py/src/model.rs`

- [ ] **Step 1: Create `StabselResult` PyClass**

```rust
// crates/boostlss-py/src/stabsel.rs
use pyo3::prelude::*;
use std::collections::HashMap;

#[pyclass(module = "boostlss_py")]
#[derive(Clone)]
pub struct PyStabselResult {
    #[pyo3(get)]
    pub selected_joint: Vec<String>,
    #[pyo3(get)]
    pub selected_independent: HashMap<String, Vec<String>>,
    #[pyo3(get)]
    pub probabilities: HashMap<String, HashMap<String, f64>>,
    #[pyo3(get)]
    pub q: usize,
    #[pyo3(get)]
    pub pfer: f64,
    #[pyo3(get)]
    pub pi_thr: f64,
    #[pyo3(get)]
    pub b: usize,
}
```

- [ ] **Step 2: Expose `stabsel.rs` in bindings**

```rust
// In crates/boostlss-py/src/lib.rs
// Add: pub mod stabsel;
// Register class: m.add_class::<stabsel::PyStabselResult>()?;
```

- [ ] **Step 3: Add `stabsel` method to `BoostLssModel`**

```rust
// In crates/boostlss-py/src/model.rs
use boostlss::cv::stabsel::{StabselConfig, StabselMode, run_stabsel};
use crate::stabsel::PyStabselResult;
use std::collections::HashMap;

// In impl BoostLssModel {
    #[pyo3(signature = (b=100, pfer=None, pi_thr=None, q=None, mode="joint"))]
    fn stabsel(
        &mut self,
        b: usize,
        pfer: Option<f64>,
        pi_thr: Option<f64>,
        q: Option<usize>,
        mode: &str,
    ) -> PyResult<PyStabselResult> {
        let train_data = self.train_data.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("Model must be fitted on data first to know the design matrix size.")
        })?;

        let mut x = train_data.0.clone();
        let mut y = train_data.1.clone();
        // Construct Dataset
        let dataset = boostlss::data::Dataset::new(x, y).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let stabsel_mode = match mode.to_lowercase().as_str() {
            "joint" => StabselMode::Joint,
            "independent" => StabselMode::Independent,
            _ => return Err(pyo3::exceptions::PyValueError::new_err("mode must be 'joint' or 'independent'")),
        };

        // p is total number of unique base-learners provided in `on()`
        let mut unique_learners = std::collections::HashSet::new();
        for (_, l) in &self.learners {
            unique_learners.insert(l.name());
        }
        let p = unique_learners.len();

        let config = StabselConfig::new(b, pfer, pi_thr, q, stabsel_mode, p)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        // Reconstruct the core model
        let mut model = self.construct_core_model()?;
        model.initialize(&dataset).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let result = run_stabsel(&model, &dataset, boostlss::engine::Mstop::Scalar(self.mstop), &config)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        // Format to PyStabselResult
        let mut probabilities = HashMap::new();
        let mut selected_joint_set = std::collections::HashSet::new();
        let mut selected_independent = HashMap::new();

        for (param, freqs) in result.frequencies {
            let mut param_probs = HashMap::new();
            let mut param_selected = Vec::new();
            for (learner, count) in freqs {
                let prob = (count as f64) / (result.b as f64);
                param_probs.insert(learner.clone(), prob);
                if prob >= result.pi_thr {
                    selected_joint_set.insert(learner.clone());
                    param_selected.push(learner);
                }
            }
            probabilities.insert(param, param_probs);
            selected_independent.insert(param.clone(), param_selected);
        }

        let mut selected_joint: Vec<String> = selected_joint_set.into_iter().collect();
        selected_joint.sort();

        Ok(PyStabselResult {
            selected_joint,
            selected_independent,
            probabilities,
            q: result.q,
            pfer: result.pfer,
            pi_thr: result.pi_thr,
            b: result.b,
        })
    }
```

- [ ] **Step 4: Check Python compilation**
Run: `uv run maturin develop`

- [ ] **Step 5: Commit**
`git add .`
`git commit -m "feat: Expose stabsel API via PyO3"`

---

### Task 4: Python Integration Tests

**Files:**
- Create: `tests/test_stabsel.py`

- [ ] **Step 1: Write integration tests**

```python
# tests/test_stabsel.py
import pytest
import numpy as np
from boostlss import BoostLSS, linear

def test_stabsel_error_bounds():
    np.random.seed(42)
    X = np.random.normal(0, 1, (100, 10))
    y = np.random.normal(0, 1, 100)

    model = BoostLSS(family="gaussian", step_length=0.1, mstop=50)
    for i in range(10):
        model.on("mu", linear(i))
    model.fit(X, y)

    # Test error combinations
    with pytest.raises(ValueError):
        model.stabsel(q=5) # Only 1 parameter

    with pytest.raises(ValueError):
        model.stabsel(pfer=1.0, pi_thr=0.9, q=5) # 3 parameters

    res = model.stabsel(pfer=1.0, pi_thr=0.9)
    assert res.q > 0
    assert abs(res.pfer - 1.0) < 1e-6

def test_stabsel_selects_informative_features():
    np.random.seed(42)
    n = 200
    p = 10
    X = np.random.normal(0, 1, (n, p))
    # Only features 0 and 1 are informative
    mu = 2.0 * X[:, 0] - 1.5 * X[:, 1]
    y = mu + np.random.normal(0, 0.5, n)

    model = BoostLSS(family="gaussian", step_length=0.1, mstop=100)
    for i in range(p):
        model.on("mu", linear(i))

    model.fit(X, y)

    res = model.stabsel(q=3, pfer=1.0, b=50)

    # We expect features 0 and 1 to be highly stable (> pi_thr)
    selected = res.selected_joint
    assert "linear_0" in selected
    assert "linear_1" in selected

    # Noise features should mostly NOT be selected
    noise_selected = [s for s in selected if s not in ("linear_0", "linear_1")]
    assert len(noise_selected) <= 1 # At most 1 falsely selected with PFER=1.0
```

- [ ] **Step 2: Run the tests**
Run: `uv run pytest tests/test_stabsel.py -v`

- [ ] **Step 3: Ensure overall code quality**
Run: `uv run ruff check` and `cargo clippy --workspace --all-targets --all-features -- -D warnings`

- [ ] **Step 4: Commit**
`git add tests/test_stabsel.py`
`git commit -m "test: Add python integration tests for stabsel"`

---
