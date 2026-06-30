# Additional Families Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement four new distributional families (Tweedie, ZINB, Logistic, Laplace) for BoostLSS.

**Architecture:** Each family implements the `Family` trait in `crates/boostlss/src/family/`. They use either closed-form gradients or finite difference fallbacks. They are exposed to Python via PyO3 in `crates/boostlss-py/src/family.rs`.

**Tech Stack:** Rust, PyO3, Python, pytest.

---

### Task 1: Tweedie Distribution

**Files:**
- Create: `crates/boostlss/src/family/tweedie.rs`
- Modify: `crates/boostlss/src/family/mod.rs`
- Modify: `crates/boostlss-py/src/family.rs`
- Modify: `crates/boostlss-py/src/model.rs`
- Modify: `crates/boostlss/src/model.rs`
- Modify: `crates/boostlss-py/tests/test_family.py`

- [ ] **Step 1: Write the failing test**

```python
# In crates/boostlss-py/tests/test_family.py (Add this test)
def test_tweedie():
    from boostlss_py import TweedieLss, BoostLssModel, Linear
    import numpy as np

    fam = TweedieLss(p=1.5)
    model = BoostLssModel(fam, mstop=2)
    model.add_learner("mu", Linear(0))
    model.add_learner("phi", Linear(0))

    # Must use positive response for Tweedie
    y = np.random.poisson(lam=5, size=10) + np.random.gamma(shape=2, scale=1, size=10)
    y = np.maximum(y, 0.0)
    X = np.random.normal(size=(10, 2))

    model.fit(X, y)
    assert len(model.predict(X, "mu")) == 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest crates/boostlss-py/tests/test_family.py::test_tweedie`
Expected: FAIL with ImportError/NameError for TweedieLss

- [ ] **Step 3: Implement TweedieLss in Rust**

Create `crates/boostlss/src/family/tweedie.rs`:
```rust
use crate::data::Dataset;
use crate::error::BoostlssError;
use crate::family::Family;
use crate::param::{LogLink, ParamSpec};
use ndarray::Array1;
use serde::{Deserialize, Serialize};

fn default_tweedie_params() -> Vec<ParamSpec> {
    vec![ParamSpec::new("mu", LogLink), ParamSpec::new("phi", LogLink)]
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TweedieLss {
    pub p: f64,
    #[serde(skip, default = "default_tweedie_params")]
    params: Vec<ParamSpec>,
}

impl TweedieLss {
    pub fn new(p: f64) -> Self {
        assert!(p > 1.0 && p < 2.0, "Tweedie p must be in (1, 2)");
        Self {
            p,
            params: default_tweedie_params(),
        }
    }
}

impl Default for TweedieLss {
    fn default() -> Self {
        Self::new(1.5)
    }
}

impl Family for TweedieLss {
    fn params(&self) -> &[ParamSpec] {
        &self.params
    }

    fn nll(&self, data: &Dataset, eta: &[Array1<f64>]) -> Result<f64, BoostlssError> {
        let y = data.response();
        let mu_link = &self.params[0].link;
        let phi_link = &self.params[1].link;
        let p = self.p;
        let w = data.weights();

        let mut nll = 0.0;
        for i in 0..y.len() {
            let yi = y[i].max(0.0);
            let mu = mu_link.response(eta[0][i]).max(1e-10);
            let phi = phi_link.response(eta[1][i]).max(1e-10);
            let wi = w.map_or(1.0, |weights| weights[i]);

            // Using Tweedie Deviance approximation for NLL
            // d(y, mu) = 2 * ( y^(2-p)/((1-p)*(2-p)) - y*mu^(1-p)/(1-p) + mu^(2-p)/(2-p) )
            let p1 = 1.0 - p;
            let p2 = 2.0 - p;

            let term1 = if yi > 0.0 { yi.powf(p2) / (p1 * p2) } else { 0.0 };
            let term2 = yi * mu.powf(p1) / p1;
            let term3 = mu.powf(p2) / p2;

            let deviance = 2.0 * (term1 - term2 + term3);

            // Approximate NLL: 0.5 * deviance / phi + 0.5 * ln(phi)
            nll += wi * (0.5 * deviance / phi + 0.5 * phi.ln());
        }
        Ok(nll)
    }

    fn init_offsets(&self, data: &Dataset) -> Result<Vec<f64>, BoostlssError> {
        let y = data.response();
        let mu_init = y.mean().unwrap_or(1.0).max(1e-3);
        let phi_init = 1.0;
        Ok(vec![
            self.params[0].link.link(mu_init),
            self.params[1].link.link(phi_init),
        ])
    }
}
```

Update `crates/boostlss/src/family/mod.rs` to expose `TweedieLss`:
```rust
// Add:
pub mod tweedie;
pub use tweedie::TweedieLss;
```

Update `crates/boostlss-py/src/family.rs`:
```rust
use boostlss::family::{TweedieLss, Family}; // Update import

#[pyclass(name = "TweedieLss")]
#[derive(Clone)]
pub struct PyTweedieLss {
    pub inner: TweedieLss,
}

#[pymethods]
impl PyTweedieLss {
    #[new]
    #[pyo3(signature = (p=1.5))]
    fn new(p: f64) -> Self {
        Self {
            inner: TweedieLss::new(p),
        }
    }
}

// In FamilyEnum, add:
Tweedie(PyTweedieLss),

// In extract_family macro matches, add:
FamilyEnum::Tweedie(f) => $body(&f.inner),

// In m.add_class
m.add_class::<PyTweedieLss>()?;
```

Update `crates/boostlss-py/src/model.rs` and `crates/boostlss/src/model.rs` macro calls to include Tweedie if they match on the family variants.
*(Wait, looking at the macro `extract_family` in `crates/boostlss-py/src/family.rs`, it handles dispatching automatically if we just add it to `FamilyEnum`. So we don't need to touch `model.rs` directly unless it explicitly matches on families! Let's check `family.rs` macro. Assuming the macro is in `family.rs`, we just update `FamilyEnum` and `m.add_class`)*.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run maturin develop -m crates/boostlss-py/Cargo.toml && uv run pytest crates/boostlss-py/tests/test_family.py::test_tweedie`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/boostlss/src/family/tweedie.rs crates/boostlss/src/family/mod.rs crates/boostlss-py/src/family.rs crates/boostlss-py/tests/test_family.py
git commit -m "feat(family): implement Tweedie distribution"
```

---

### Task 2: Zero-Inflated Negative Binomial (ZINB)

**Files:**
- Create: `crates/boostlss/src/family/zinb.rs`
- Modify: `crates/boostlss/src/family/mod.rs`
- Modify: `crates/boostlss-py/src/family.rs`
- Modify: `crates/boostlss-py/tests/test_family.py`

- [ ] **Step 1: Write the failing test**

```python
# In crates/boostlss-py/tests/test_family.py
def test_zinb():
    from boostlss_py import ZINBLss, BoostLssModel, Linear
    import numpy as np

    fam = ZINBLss()
    model = BoostLssModel(fam, mstop=2)
    model.add_learner("mu", Linear(0))
    model.add_learner("sigma", Linear(0))
    model.add_learner("nu", Linear(0))

    y = np.random.poisson(lam=5, size=10)
    y[0:3] = 0.0 # Force zeros
    X = np.random.normal(size=(10, 2))

    model.fit(X, y)
    assert len(model.predict(X, "mu")) == 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest crates/boostlss-py/tests/test_family.py::test_zinb`

- [ ] **Step 3: Implement ZINBLss in Rust**

Create `crates/boostlss/src/family/zinb.rs`:
```rust
use crate::data::Dataset;
use crate::error::BoostlssError;
use crate::family::Family;
use crate::param::{LogLink, LogitLink, ParamSpec};
use ndarray::Array1;
use serde::{Deserialize, Serialize};

fn default_zinb_params() -> Vec<ParamSpec> {
    vec![
        ParamSpec::new("mu", LogLink),
        ParamSpec::new("sigma", LogLink),
        ParamSpec::new("nu", LogitLink),
    ]
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZINBLss {
    #[serde(skip, default = "default_zinb_params")]
    params: Vec<ParamSpec>,
}

impl ZINBLss {
    pub fn new() -> Self {
        Self {
            params: default_zinb_params(),
        }
    }
}

impl Default for ZINBLss {
    fn default() -> Self {
        Self::new()
    }
}

impl Family for ZINBLss {
    fn params(&self) -> &[ParamSpec] {
        &self.params
    }

    fn nll(&self, data: &Dataset, eta: &[Array1<f64>]) -> Result<f64, BoostlssError> {
        let y = data.response();
        let mu_link = &self.params[0].link;
        let sigma_link = &self.params[1].link;
        let nu_link = &self.params[2].link;
        let w = data.weights();

        let mut nll = 0.0;
        for i in 0..y.len() {
            let yi = y[i];
            let mu = mu_link.response(eta[0][i]).max(1e-10);
            let sigma = sigma_link.response(eta[1][i]).max(1e-10);
            let nu = nu_link.response(eta[2][i]).clamp(1e-10, 1.0 - 1e-10);
            let wi = w.map_or(1.0, |weights| weights[i]);

            let var = mu + sigma * mu * mu;
            let p = mu / var;
            let r = mu * mu / (var - mu).max(1e-10);

            let log_pdf = if yi == 0.0 {
                let nb_zero = p.powf(r);
                (nu + (1.0 - nu) * nb_zero).ln()
            } else {
                let ln_gamma_r_y = statrs::function::gamma::ln_gamma(r + yi);
                let ln_gamma_r = statrs::function::gamma::ln_gamma(r);
                let ln_gamma_y_1 = statrs::function::gamma::ln_gamma(yi + 1.0);

                (1.0 - nu).ln() + ln_gamma_r_y - ln_gamma_r - ln_gamma_y_1
                    + r * p.ln() + yi * (1.0 - p).ln()
            };
            nll -= wi * log_pdf;
        }
        Ok(nll)
    }

    fn init_offsets(&self, data: &Dataset) -> Result<Vec<f64>, BoostlssError> {
        let y = data.response();
        let mu_init = y.mean().unwrap_or(1.0).max(1e-3);
        let sigma_init = 1.0;
        let zeros = y.iter().filter(|&&val| val == 0.0).count();
        let nu_init = (zeros as f64 / y.len() as f64).clamp(0.01, 0.99);

        Ok(vec![
            self.params[0].link.link(mu_init),
            self.params[1].link.link(sigma_init),
            self.params[2].link.link(nu_init),
        ])
    }
}
```

Update `crates/boostlss/src/family/mod.rs` to expose `ZINBLss`.
Update `crates/boostlss-py/src/family.rs` to expose `PyZINBLss` and add it to `FamilyEnum`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run maturin develop -m crates/boostlss-py/Cargo.toml && uv run pytest crates/boostlss-py/tests/test_family.py::test_zinb`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/boostlss/src/family/zinb.rs crates/boostlss/src/family/mod.rs crates/boostlss-py/src/family.rs crates/boostlss-py/tests/test_family.py
git commit -m "feat(family): implement ZINB distribution"
```

---

### Task 3: Logistic Distribution

**Files:**
- Create: `crates/boostlss/src/family/logistic.rs`
- Modify: `crates/boostlss/src/family/mod.rs`
- Modify: `crates/boostlss-py/src/family.rs`
- Modify: `crates/boostlss-py/tests/test_family.py`

- [ ] **Step 1: Write the failing test**

```python
# In crates/boostlss-py/tests/test_family.py
def test_logistic():
    from boostlss_py import LogisticLss, BoostLssModel, Linear
    import numpy as np

    fam = LogisticLss()
    model = BoostLssModel(fam, mstop=2)
    model.add_learner("mu", Linear(0))
    model.add_learner("s", Linear(0))

    y = np.random.logistic(loc=5, scale=2, size=10)
    X = np.random.normal(size=(10, 2))

    model.fit(X, y)
    assert len(model.predict(X, "mu")) == 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest crates/boostlss-py/tests/test_family.py::test_logistic`

- [ ] **Step 3: Implement LogisticLss in Rust**

Create `crates/boostlss/src/family/logistic.rs`:
```rust
use crate::data::Dataset;
use crate::error::BoostlssError;
use crate::family::Family;
use crate::param::{IdentityLink, LogLink, ParamSpec};
use ndarray::Array1;
use serde::{Deserialize, Serialize};

fn default_logistic_params() -> Vec<ParamSpec> {
    vec![
        ParamSpec::new("mu", IdentityLink),
        ParamSpec::new("s", LogLink),
    ]
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogisticLss {
    #[serde(skip, default = "default_logistic_params")]
    params: Vec<ParamSpec>,
}

impl LogisticLss {
    pub fn new() -> Self {
        Self {
            params: default_logistic_params(),
        }
    }
}

impl Default for LogisticLss {
    fn default() -> Self {
        Self::new()
    }
}

impl Family for LogisticLss {
    fn params(&self) -> &[ParamSpec] {
        &self.params
    }

    fn nll(&self, data: &Dataset, eta: &[Array1<f64>]) -> Result<f64, BoostlssError> {
        let y = data.response();
        let mu_link = &self.params[0].link;
        let s_link = &self.params[1].link;
        let w = data.weights();

        let mut nll = 0.0;
        for i in 0..y.len() {
            let yi = y[i];
            let mu = mu_link.response(eta[0][i]);
            let s = s_link.response(eta[1][i]).max(1e-10);
            let wi = w.map_or(1.0, |weights| weights[i]);

            let z = (yi - mu) / s;
            nll += wi * (z + s.ln() + 2.0 * (1.0 + (-z).exp()).ln());
        }
        Ok(nll)
    }

    fn init_offsets(&self, data: &Dataset) -> Result<Vec<f64>, BoostlssError> {
        let y = data.response();
        let mu_init = y.mean().unwrap_or(0.0);
        let var = y.var(1.0);
        let s_init = (var * 3.0 / std::f64::consts::PI.powi(2)).sqrt().max(1e-3);

        Ok(vec![
            self.params[0].link.link(mu_init),
            self.params[1].link.link(s_init),
        ])
    }
}
```

Update `crates/boostlss/src/family/mod.rs` and `crates/boostlss-py/src/family.rs` to expose `LogisticLss`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run maturin develop -m crates/boostlss-py/Cargo.toml && uv run pytest crates/boostlss-py/tests/test_family.py::test_logistic`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/boostlss/src/family/logistic.rs crates/boostlss/src/family/mod.rs crates/boostlss-py/src/family.rs crates/boostlss-py/tests/test_family.py
git commit -m "feat(family): implement Logistic distribution"
```

---

### Task 4: Laplace Distribution

**Files:**
- Create: `crates/boostlss/src/family/laplace.rs`
- Modify: `crates/boostlss/src/family/mod.rs`
- Modify: `crates/boostlss-py/src/family.rs`
- Modify: `crates/boostlss-py/tests/test_family.py`

- [ ] **Step 1: Write the failing test**

```python
# In crates/boostlss-py/tests/test_family.py
def test_laplace():
    from boostlss_py import LaplaceLss, BoostLssModel, Linear
    import numpy as np

    fam = LaplaceLss()
    model = BoostLssModel(fam, mstop=2)
    model.add_learner("mu", Linear(0))
    model.add_learner("b", Linear(0))

    y = np.random.laplace(loc=5, scale=2, size=10)
    X = np.random.normal(size=(10, 2))

    model.fit(X, y)
    assert len(model.predict(X, "mu")) == 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest crates/boostlss-py/tests/test_family.py::test_laplace`

- [ ] **Step 3: Implement LaplaceLss in Rust**

Create `crates/boostlss/src/family/laplace.rs`:
```rust
use crate::data::Dataset;
use crate::error::BoostlssError;
use crate::family::Family;
use crate::param::{IdentityLink, LogLink, ParamSpec};
use ndarray::Array1;
use serde::{Deserialize, Serialize};

fn default_laplace_params() -> Vec<ParamSpec> {
    vec![
        ParamSpec::new("mu", IdentityLink),
        ParamSpec::new("b", LogLink),
    ]
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaplaceLss {
    #[serde(skip, default = "default_laplace_params")]
    params: Vec<ParamSpec>,
}

impl LaplaceLss {
    pub fn new() -> Self {
        Self {
            params: default_laplace_params(),
        }
    }
}

impl Default for LaplaceLss {
    fn default() -> Self {
        Self::new()
    }
}

impl Family for LaplaceLss {
    fn params(&self) -> &[ParamSpec] {
        &self.params
    }

    fn nll(&self, data: &Dataset, eta: &[Array1<f64>]) -> Result<f64, BoostlssError> {
        let y = data.response();
        let mu_link = &self.params[0].link;
        let b_link = &self.params[1].link;
        let w = data.weights();

        let mut nll = 0.0;
        let epsilon = 1e-6; // Pseudo-Huber smoothing for differentiability

        for i in 0..y.len() {
            let yi = y[i];
            let mu = mu_link.response(eta[0][i]);
            let b = b_link.response(eta[1][i]).max(1e-10);
            let wi = w.map_or(1.0, |weights| weights[i]);

            // Smoothed absolute value: sqrt((y-mu)^2 + e^2) - e
            let abs_diff = ((yi - mu).powi(2) + epsilon * epsilon).sqrt() - epsilon;

            nll += wi * ((2.0 * b).ln() + abs_diff / b);
        }
        Ok(nll)
    }

    fn init_offsets(&self, data: &Dataset) -> Result<Vec<f64>, BoostlssError> {
        let y = data.response();

        // Median is better for Laplace mu, but mean is an okay approximation for init
        let mut sorted_y: Vec<f64> = y.iter().cloned().collect();
        sorted_y.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mid = sorted_y.len() / 2;
        let mu_init = sorted_y[mid];

        let mut mad = 0.0;
        for &yi in &sorted_y {
            mad += (yi - mu_init).abs();
        }
        let b_init = (mad / sorted_y.len() as f64).max(1e-3);

        Ok(vec![
            self.params[0].link.link(mu_init),
            self.params[1].link.link(b_init),
        ])
    }
}
```

Update `crates/boostlss/src/family/mod.rs` and `crates/boostlss-py/src/family.rs` to expose `LaplaceLss`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run maturin develop -m crates/boostlss-py/Cargo.toml && uv run pytest crates/boostlss-py/tests/test_family.py::test_laplace`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/boostlss/src/family/laplace.rs crates/boostlss/src/family/mod.rs crates/boostlss-py/src/family.rs crates/boostlss-py/tests/test_family.py
git commit -m "feat(family): implement Laplace distribution"
```

---
