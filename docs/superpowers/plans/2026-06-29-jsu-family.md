# JSU Family Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the Johnson's SU (JSU) 4-parameter family distribution for modeling skewed and heavy-tailed data in BoostLSS.

**Architecture:** Create a new `JSULss` struct that implements the `Family` trait. It will use the finite difference gradient approach and provide parameterization for Location ($\mu$), Scale ($\sigma$), Skewness ($\nu$), and Tail Weight ($\tau$).

**Tech Stack:** Rust, PyO3, BoostLSS

---

### Task 1: Create the JSULss Family Struct

**Files:**
- Create: `crates/boostlss/src/family/jsu.rs`
- Modify: `crates/boostlss/src/family/mod.rs`

- [ ] **Step 1: Write the failing test**

```rust
// In crates/boostlss/src/family/jsu.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jsu_params() {
        let fam = JSULss::new();
        assert_eq!(fam.params().len(), 4);
        assert_eq!(fam.params()[0].name, "mu");
        assert_eq!(fam.params()[1].name, "sigma");
        assert_eq!(fam.params()[2].name, "nu");
        assert_eq!(fam.params()[3].name, "tau");
    }
}
```

- [ ] **Step 2: Add to module to run test to verify it fails**

In `crates/boostlss/src/family/mod.rs`, add:
```rust
pub mod jsu;
pub use jsu::JSULss;
```

Run: `cargo test -p boostlss -- test_jsu_params`
Expected: FAIL (unresolved import / struct not found)

- [ ] **Step 3: Write minimal implementation**

In `crates/boostlss/src/family/jsu.rs`:
```rust
use crate::data::Dataset;
use crate::error::BoostlssError;
use crate::family::Family;
use crate::param::{IdentityLink, LogLink, ParamSpec};
use crate::util::{minimize_1d, weighted_mean, weighted_sd};
use ndarray::Array1;
use serde::{Deserialize, Serialize};

fn default_jsu_params() -> Vec<ParamSpec> {
    JSULss::new().params
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JSULss {
    #[serde(skip, default = "default_jsu_params")]
    pub params: Vec<ParamSpec>,
}

impl JSULss {
    pub fn new() -> Self {
        Self {
            params: vec![
                ParamSpec::new("mu", IdentityLink),
                ParamSpec::new("sigma", LogLink),
                ParamSpec::new("nu", IdentityLink),
                ParamSpec::new("tau", LogLink),
            ],
        }
    }
}

impl Default for JSULss {
    fn default() -> Self {
        Self::new()
    }
}

impl Family for JSULss {
    fn params(&self) -> &[ParamSpec] {
        &self.params
    }

    fn nll(&self, data: &Dataset, eta: &[Array1<f64>]) -> Result<f64, BoostlssError> {
        unimplemented!()
    }

    fn init_offsets(&self, data: &Dataset) -> Result<Vec<f64>, BoostlssError> {
        unimplemented!()
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p boostlss -- test_jsu_params`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/boostlss/src/family/jsu.rs crates/boostlss/src/family/mod.rs
git commit -m "feat: stub JSULss family and parameters"
```

---

### Task 2: Implement JSU Negative Log-Likelihood (NLL)

**Files:**
- Modify: `crates/boostlss/src/family/jsu.rs`

- [ ] **Step 1: Write the failing test**

```rust
// In crates/boostlss/src/family/jsu.rs tests module
#[test]
fn test_jsu_nll() {
    use ndarray::{array, Array2};
    let fam = JSULss::new();
    let y = array![0.0, 1.0, 2.0];
    let ds = Dataset::new(Array2::<f64>::zeros((3, 1)), y, None).unwrap();
    // mu=0, sigma=1 (eta=0), nu=0, tau=1 (eta=0)
    let eta = vec![
        array![0.0, 0.0, 0.0],
        array![0.0, 0.0, 0.0],
        array![0.0, 0.0, 0.0],
        array![0.0, 0.0, 0.0],
    ];
    let nll = fam.nll(&ds, &eta).unwrap();
    assert!(nll > 0.0);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p boostlss -- test_jsu_nll`
Expected: FAIL (unimplemented)

- [ ] **Step 3: Write minimal implementation**

Update `nll` in `crates/boostlss/src/family/jsu.rs`:
```rust
    fn nll(&self, data: &Dataset, eta: &[Array1<f64>]) -> Result<f64, BoostlssError> {
        let mu = &eta[0]; // Identity
        let sigma = eta[1].mapv(|x| x.exp().max(1e-10)); // Log
        let nu = &eta[2]; // Identity
        let tau = eta[3].mapv(|x| x.exp().max(1e-10)); // Log

        let mut total_nll = 0.0;
        let y = data.response();
        let w = data.weights();

        let half_log_2pi = 0.5 * (2.0 * std::f64::consts::PI).ln();

        for i in 0..data.n_obs() {
            let sig = sigma[i];
            let z = (y[i] - mu[i]) / sig;

            // r = -nu + tau * asinh(z)
            let asinh_z = z.asinh();
            let r = -nu[i] + tau[i] * asinh_z;

            // log_pdf = log(tau) - log(sigma) - 0.5*log(z^2 + 1) - 0.5*log(2*pi) - 0.5*r^2
            let log_pdf = tau[i].ln() - sig.ln() - 0.5 * (z * z + 1.0).ln() - half_log_2pi - 0.5 * r * r;

            let weight = w.map(|w_arr| w_arr[i]).unwrap_or(1.0);
            total_nll -= weight * log_pdf;
        }

        Ok(total_nll)
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p boostlss -- test_jsu_nll`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/boostlss/src/family/jsu.rs
git commit -m "feat: implement JSU negative log-likelihood"
```

---

### Task 3: Implement Initialization and Default Gradients Test

**Files:**
- Modify: `crates/boostlss/src/family/jsu.rs`

- [ ] **Step 1: Write the failing tests**

```rust
// In crates/boostlss/src/family/jsu.rs tests module
#[test]
fn test_jsu_init_offsets() {
    use ndarray::{array, Array2};
    let fam = JSULss::new();
    let y = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let ds = Dataset::new(Array2::<f64>::zeros((5, 1)), y, None).unwrap();
    let offsets = fam.init_offsets(&ds).unwrap();
    assert_eq!(offsets.len(), 4);
}

#[test]
fn test_jsu_gradients() {
    use ndarray::{array, Array2};
    let fam = JSULss::new();
    let y = array![0.0, 1.0, 2.0];
    let ds = Dataset::new(Array2::<f64>::zeros((3, 1)), y, None).unwrap();
    let eta = vec![
        array![0.0, 0.0, 0.0],
        array![0.0, 0.0, 0.0],
        array![0.0, 0.0, 0.0],
        array![0.0, 0.0, 0.0],
    ];
    let grad = fam.ngradient(&ds, &eta, 0).unwrap();
    assert_eq!(grad.len(), 3);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p boostlss -- test_jsu_init_offsets`
Expected: FAIL (unimplemented)

- [ ] **Step 3: Write minimal implementation**

Update `init_offsets` in `crates/boostlss/src/family/jsu.rs`:
```rust
    fn init_offsets(&self, data: &Dataset) -> Result<Vec<f64>, BoostlssError> {
        let mean = weighted_mean(data.response(), data.weights());
        let sd = weighted_sd(data.response(), data.weights()).max(1e-10);

        let mut mu_val = mean;
        let mut sigma_val = sd;
        let mut nu_val = 0.0;
        let mut tau_val = 1.0;

        let y_arr = data.response().clone();
        let w_arr = data.weights().cloned();

        let dense_design = match data.design() {
            crate::data::DesignMatrix::Dense(mat) => mat.clone(),
            _ => {
                return Err(BoostlssError::DataError(
                    "jsu init requires dense matrix".to_string(),
                ))
            }
        };
        let ds = Dataset::new(dense_design, y_arr.clone(), w_arr.clone()).unwrap();

        let mut eta = vec![
            Array1::from_elem(y_arr.len(), self.params[0].link.link(mu_val)),
            Array1::from_elem(y_arr.len(), self.params[1].link.link(sigma_val)),
            Array1::from_elem(y_arr.len(), self.params[2].link.link(nu_val)),
            Array1::from_elem(y_arr.len(), self.params[3].link.link(tau_val)),
        ];

        // Refine with minimize_1d for 2 iterations
        for _ in 0..2 {
            let opt_eta_mu = minimize_1d(
                |m| {
                    eta[0].fill(m);
                    self.nll(&ds, &eta).unwrap_or(f64::MAX)
                },
                self.params[0].link.link(mean - 2.0 * sd),
                self.params[0].link.link(mean + 2.0 * sd),
            );
            eta[0].fill(opt_eta_mu);
            mu_val = self.params[0].link.response(opt_eta_mu);

            let opt_eta_sigma = minimize_1d(
                |s| {
                    eta[1].fill(s);
                    self.nll(&ds, &eta).unwrap_or(f64::MAX)
                },
                self.params[1].link.link(sd * 0.1),
                self.params[1].link.link(sd * 10.0),
            );
            eta[1].fill(opt_eta_sigma);
            sigma_val = self.params[1].link.response(opt_eta_sigma);

            let opt_eta_nu = minimize_1d(
                |n| {
                    eta[2].fill(n);
                    self.nll(&ds, &eta).unwrap_or(f64::MAX)
                },
                self.params[2].link.link(-5.0),
                self.params[2].link.link(5.0),
            );
            eta[2].fill(opt_eta_nu);
            nu_val = self.params[2].link.response(opt_eta_nu);

            let opt_eta_tau = minimize_1d(
                |t| {
                    eta[3].fill(t);
                    self.nll(&ds, &eta).unwrap_or(f64::MAX)
                },
                self.params[3].link.link(0.1),
                self.params[3].link.link(10.0),
            );
            eta[3].fill(opt_eta_tau);
            tau_val = self.params[3].link.response(opt_eta_tau);
        }

        Ok(vec![
            self.params[0].link.link(mu_val),
            self.params[1].link.link(sigma_val),
            self.params[2].link.link(nu_val),
            self.params[3].link.link(tau_val),
        ])
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p boostlss`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/boostlss/src/family/jsu.rs
git commit -m "feat: implement JSU initialization"
```

---

### Task 4: Expose JSU to Python

**Files:**
- Modify: `crates/boostlss-py/src/family.rs`
- Modify: `crates/boostlss/src/lib.rs` (ensure jsu is public if needed)

- [ ] **Step 1: Expose enum variant in pyo3 wrapper**

In `crates/boostlss-py/src/family.rs`:
Add `Jsu` to the `PyFamily` enum:
```rust
pub enum PyFamily {
    Gaussian,
    Binomial,
    Beta,
    Weibull,
    LogNormal,
    Zip,
    Gev,
    Jsu,
}
```

Add parsing in `new`:
```rust
            "JSULSS" | "JSULss" | "JsuLss" => Ok(PyFamily::Jsu),
```

Add handling in `__getnewargs__`:
```rust
            PyFamily::Jsu => ("JSULSS",),
```

In `crates/boostlss-py/src/model.rs`, update `_fit` method match:
```rust
            PyFamily::Jsu => {
                let model = BoostLss::<boostlss::family::JSULss>::new(mstop, step_length);
                let trained = model.fit(&ds, &learner_map)?;
                self.fitted = Some(FittedModel::Jsu(trained));
            }
```

In `crates/boostlss-py/src/model.rs`, add `Jsu` variant to `FittedModel` enum:
```rust
enum FittedModel {
    Gaussian(FittedBoostLss<boostlss::family::GaussianLss>),
    Binomial(FittedBoostLss<boostlss::family::BinomialLss>),
    Beta(FittedBoostLss<boostlss::family::BetaLss>),
    Weibull(FittedBoostLss<boostlss::family::WeibullLss>),
    LogNormal(FittedBoostLss<boostlss::family::LogNormalLss>),
    Zip(FittedBoostLss<boostlss::family::ZIPLss>),
    Gev(FittedBoostLss<boostlss::family::GEVLss>),
    Jsu(FittedBoostLss<boostlss::family::JSULss>),
}
```

Update `predict` method match in `crates/boostlss-py/src/model.rs`:
```rust
            FittedModel::Jsu(m) => {
                let p = m.predict(&ds, param_name)?;
                Ok(PyArray1::from_vec_bound(py, p.to_vec()).into_any())
            }
```

- [ ] **Step 2: Run cargo tests to verify everything compiles and passes**

Run: `cargo test --workspace`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add crates/boostlss-py/src/family.rs crates/boostlss-py/src/model.rs
git commit -m "feat: expose JSU distribution to python bindings"
```
