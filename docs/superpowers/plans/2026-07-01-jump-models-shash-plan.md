# Jump Models and SHASH Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the Merton Jump-Diffusion and Sinh-Arcsinh (SHASH) distribution families for boostlss.

**Architecture:** Each family will be implemented as a Rust struct implementing the `Family` trait, exposing 5 and 4 parameters respectively. Python bindings will be added using PyO3, and integration tested using pytest.

**Tech Stack:** Rust, PyO3, Python, pytest.

---

### Task 1: Implement Merton Jump-Diffusion Family

**Files:**
- Create: `crates/boostlss/src/family/merton.rs`
- Modify: `crates/boostlss/src/family/mod.rs`
- Modify: `crates/boostlss-py/src/family.rs`
- Modify: `crates/boostlss-py/tests/test_family.py`

- [ ] **Step 1: Write the failing test**

```python
# In crates/boostlss-py/tests/test_family.py
def test_merton():
    from boostlss_py import MertonJumpDiffusionLss, BoostLssModel, Linear
    import numpy as np

    fam = MertonJumpDiffusionLss(max_jumps=10)
    model = BoostLssModel(fam, mstop=2)
    model.add_learner("mu", Linear(0))
    model.add_learner("sigma", Linear(0))
    model.add_learner("lam", Linear(0))
    model.add_learner("mu_j", Linear(0))
    model.add_learner("sigma_j", Linear(0))

    # Generate some jumpy data
    np.random.seed(42)
    y = np.random.normal(loc=0.05, scale=0.1, size=20)
    # add some jumps
    jumps = np.random.poisson(lam=0.5, size=20)
    y += jumps * np.random.normal(loc=-0.1, scale=0.2, size=20)

    X = np.random.normal(size=(20, 2))

    model.fit(X, y)
    assert len(model.predict(X, "mu")) == 20
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest crates/boostlss-py/tests/test_family.py::test_merton`
Expected: FAIL (ImportError for MertonJumpDiffusionLss)

- [ ] **Step 3: Implement MertonJumpDiffusionLss in Rust**

Create `crates/boostlss/src/family/merton.rs`:
```rust
use crate::data::Dataset;
use crate::error::BoostlssError;
use crate::family::Family;
use crate::param::{IdentityLink, LogLink, ParamSpec};
use ndarray::Array1;
use serde::{Deserialize, Serialize};

fn default_merton_params() -> Vec<ParamSpec> {
    vec![
        ParamSpec::new("mu", IdentityLink),
        ParamSpec::new("sigma", LogLink),
        ParamSpec::new("lam", LogLink),
        ParamSpec::new("mu_j", IdentityLink),
        ParamSpec::new("sigma_j", LogLink),
    ]
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MertonJumpDiffusionLss {
    pub max_jumps: usize,
    #[serde(skip, default = "default_merton_params")]
    params: Vec<ParamSpec>,
}

impl MertonJumpDiffusionLss {
    pub fn new(max_jumps: usize) -> Self {
        Self {
            max_jumps,
            params: default_merton_params(),
        }
    }
}

impl Default for MertonJumpDiffusionLss {
    fn default() -> Self {
        Self::new(10)
    }
}

impl Family for MertonJumpDiffusionLss {
    fn params(&self) -> &[ParamSpec] {
        &self.params
    }

    fn nll(&self, data: &Dataset, eta: &[Array1<f64>]) -> Result<f64, BoostlssError> {
        let y = data.response();
        let mu_link = &self.params[0].link;
        let sigma_link = &self.params[1].link;
        let lam_link = &self.params[2].link;
        let mu_j_link = &self.params[3].link;
        let sigma_j_link = &self.params[4].link;
        let w = data.weights();

        let mut total_nll = 0.0;
        let pi = std::f64::consts::PI;

        // Pre-compute factorial log for logsumexp trick
        let mut ln_fact = vec![0.0; self.max_jumps + 1];
        for j in 1..=self.max_jumps {
            ln_fact[j] = ln_fact[j - 1] + (j as f64).ln();
        }

        for i in 0..y.len() {
            let yi = y[i];
            let mu = mu_link.response(eta[0][i]);
            let sigma = sigma_link.response(eta[1][i]).max(1e-10);
            let lam = lam_link.response(eta[2][i]).max(1e-10);
            let mu_j = mu_j_link.response(eta[3][i]);
            let sigma_j = sigma_j_link.response(eta[4][i]).max(1e-10);
            let wi = w.map_or(1.0, |weights| weights[i]);

            let var_diff = sigma * sigma;
            let var_jump = sigma_j * sigma_j;
            let drift = mu - 0.5 * var_diff;

            // Use LogSumExp trick for stability
            let mut log_terms = Vec::with_capacity(self.max_jumps + 1);
            for j in 0..=self.max_jumps {
                let j_f64 = j as f64;
                let mu_total = drift + j_f64 * mu_j;
                let var_total = var_diff + j_f64 * var_jump;
                let std_total = var_total.sqrt();

                // ln_prob_jump = -lam + j*ln(lam) - ln(j!)
                let ln_prob_jump = -lam + j_f64 * lam.ln() - ln_fact[j];

                let diff = yi - mu_total;
                let ln_norm = -0.5 * (2.0 * pi).ln() - std_total.ln() - 0.5 * (diff * diff) / var_total;

                log_terms.push(ln_prob_jump + ln_norm);
            }

            // logsumexp
            let max_log = log_terms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mut sum_exp = 0.0;
            for lt in log_terms {
                sum_exp += (lt - max_log).exp();
            }
            let ln_likelihood = max_log + sum_exp.ln();

            total_nll -= wi * ln_likelihood;
        }
        Ok(total_nll)
    }

    fn init_offsets(&self, data: &Dataset) -> Result<Vec<f64>, BoostlssError> {
        let y = data.response();
        let mu_init = y.mean().unwrap_or(0.0);
        let sigma_init = y.var(1.0).sqrt().max(1e-3);
        let lam_init = 1.0;
        let mu_j_init = 0.0;
        let sigma_j_init = sigma_init; // Initialize jump volatility same as diffusion

        Ok(vec![
            self.params[0].link.link(mu_init),
            self.params[1].link.link(sigma_init),
            self.params[2].link.link(lam_init),
            self.params[3].link.link(mu_j_init),
            self.params[4].link.link(sigma_j_init),
        ])
    }
}
```

- [ ] **Step 4: Update Exports and Bindings**

Update `crates/boostlss/src/family/mod.rs`:
```rust
// Add:
pub mod merton;
pub use merton::MertonJumpDiffusionLss;
```

Update `crates/boostlss-py/src/family.rs`:
```rust
use boostlss::family::MertonJumpDiffusionLss;

// ...

#[pyclass(name = "MertonJumpDiffusionLss")]
#[derive(Clone)]
pub struct PyMertonJumpDiffusionLss {
    pub inner: MertonJumpDiffusionLss,
}

#[pymethods]
impl PyMertonJumpDiffusionLss {
    #[new]
    #[pyo3(signature = (max_jumps=10))]
    fn new(max_jumps: usize) -> Self {
        Self {
            inner: MertonJumpDiffusionLss::new(max_jumps),
        }
    }
}

// In FamilyEnum, add:
Merton(PyMertonJumpDiffusionLss),

// In extract_family macro matches, add:
FamilyEnum::Merton(f) => $body(&f.inner),

// In create_module, add:
m.add_class::<PyMertonJumpDiffusionLss>()?;
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run maturin develop -m crates/boostlss-py/Cargo.toml && uv run pytest crates/boostlss-py/tests/test_family.py::test_merton`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add crates/boostlss/src/family/merton.rs crates/boostlss/src/family/mod.rs crates/boostlss-py/src/family.rs crates/boostlss-py/tests/test_family.py
git commit -m "feat(family): implement Merton Jump-Diffusion distribution"
```

---

### Task 2: Implement Sinh-Arcsinh (SHASH) Family

**Files:**
- Create: `crates/boostlss/src/family/shash.rs`
- Modify: `crates/boostlss/src/family/mod.rs`
- Modify: `crates/boostlss-py/src/family.rs`
- Modify: `crates/boostlss-py/tests/test_family.py`

- [ ] **Step 1: Write the failing test**

```python
# In crates/boostlss-py/tests/test_family.py
def test_shash():
    from boostlss_py import SHASHLss, BoostLssModel, Linear
    import numpy as np

    fam = SHASHLss()
    model = BoostLssModel(fam, mstop=2)
    model.add_learner("mu", Linear(0))
    model.add_learner("sigma", Linear(0))
    model.add_learner("nu", Linear(0))
    model.add_learner("tau", Linear(0))

    y = np.random.normal(loc=5, scale=2, size=10)
    X = np.random.normal(size=(10, 2))

    model.fit(X, y)
    assert len(model.predict(X, "mu")) == 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest crates/boostlss-py/tests/test_family.py::test_shash`
Expected: FAIL

- [ ] **Step 3: Implement SHASHLss in Rust**

Create `crates/boostlss/src/family/shash.rs`:
```rust
use crate::data::Dataset;
use crate::error::BoostlssError;
use crate::family::Family;
use crate::param::{IdentityLink, LogLink, ParamSpec};
use ndarray::Array1;
use serde::{Deserialize, Serialize};

fn default_shash_params() -> Vec<ParamSpec> {
    vec![
        ParamSpec::new("mu", IdentityLink),
        ParamSpec::new("sigma", LogLink),
        ParamSpec::new("nu", IdentityLink),
        ParamSpec::new("tau", LogLink),
    ]
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SHASHLss {
    #[serde(skip, default = "default_shash_params")]
    params: Vec<ParamSpec>,
}

impl SHASHLss {
    pub fn new() -> Self {
        Self {
            params: default_shash_params(),
        }
    }
}

impl Default for SHASHLss {
    fn default() -> Self {
        Self::new()
    }
}

impl Family for SHASHLss {
    fn params(&self) -> &[ParamSpec] {
        &self.params
    }

    fn nll(&self, data: &Dataset, eta: &[Array1<f64>]) -> Result<f64, BoostlssError> {
        let y = data.response();
        let mu_link = &self.params[0].link;
        let sigma_link = &self.params[1].link;
        let nu_link = &self.params[2].link;
        let tau_link = &self.params[3].link;
        let w = data.weights();

        let mut nll = 0.0;
        let pi = std::f64::consts::PI;

        for i in 0..y.len() {
            let yi = y[i];
            let mu = mu_link.response(eta[0][i]);
            let sigma = sigma_link.response(eta[1][i]).max(1e-10);
            let nu = nu_link.response(eta[2][i]);
            let tau = tau_link.response(eta[3][i]).max(1e-10);
            let wi = w.map_or(1.0, |weights| weights[i]);

            let z = (yi - mu) / sigma;
            // The rust inverse hyperbolic sine is .asinh()
            let asinh_z = z.asinh();

            let exp_tau = (tau * asinh_z).exp();
            let exp_nu = (-nu * asinh_z).exp();

            let r = 0.5 * (exp_tau - exp_nu);
            let c = 0.5 * (tau * exp_tau + nu * exp_nu);

            let ln_c = c.max(1e-10).ln();
            let ln_sigma = sigma.ln();
            let ln_z_term = 0.5 * (1.0 + z * z).ln();
            let r_sq_term = 0.5 * r * r;

            let log_pdf = ln_c - 0.5 * (2.0 * pi).ln() - ln_sigma - ln_z_term - r_sq_term;

            nll -= wi * log_pdf;
        }
        Ok(nll)
    }

    fn init_offsets(&self, data: &Dataset) -> Result<Vec<f64>, BoostlssError> {
        let y = data.response();
        let mu_init = y.mean().unwrap_or(0.0);
        let sigma_init = y.var(1.0).sqrt().max(1e-3);
        let nu_init = 0.0;
        let tau_init = 1.0;

        Ok(vec![
            self.params[0].link.link(mu_init),
            self.params[1].link.link(sigma_init),
            self.params[2].link.link(nu_init),
            self.params[3].link.link(tau_init),
        ])
    }
}
```

- [ ] **Step 4: Update Exports and Bindings**

Update `crates/boostlss/src/family/mod.rs`:
```rust
// Add:
pub mod shash;
pub use shash::SHASHLss;
```

Update `crates/boostlss-py/src/family.rs`:
```rust
use boostlss::family::SHASHLss;

// ...

#[pyclass(name = "SHASHLss")]
#[derive(Clone)]
pub struct PySHASHLss {
    pub inner: SHASHLss,
}

#[pymethods]
impl PySHASHLss {
    #[new]
    fn new() -> Self {
        Self {
            inner: SHASHLss::new(),
        }
    }
}

// In FamilyEnum, add:
SHASH(PySHASHLss),

// In extract_family macro matches, add:
FamilyEnum::SHASH(f) => $body(&f.inner),

// In create_module, add:
m.add_class::<PySHASHLss>()?;
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run maturin develop -m crates/boostlss-py/Cargo.toml && uv run pytest crates/boostlss-py/tests/test_family.py::test_shash`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add crates/boostlss/src/family/shash.rs crates/boostlss/src/family/mod.rs crates/boostlss-py/src/family.rs crates/boostlss-py/tests/test_family.py
git commit -m "feat(family): implement Sinh-Arcsinh (SHASH) distribution"
```
