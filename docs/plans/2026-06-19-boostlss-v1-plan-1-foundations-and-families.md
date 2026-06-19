# boostlss v1 — Plan 1: Foundations & Families — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the `boostlss` cargo workspace and implement the data model, link functions, the `Family` trait, and all four v1 distribution families (`GaussianLss`, `StudentTLss`, `GammaLss`, `NBinomialLss`) with analytic gradients verified against finite differences.

**Architecture:** A single library crate `crates/boostlss` inside a cargo workspace. Each distribution is a self-contained module implementing the `Family` trait (links, intercept-only offsets, empirical risk = negative log-likelihood, and per-parameter negative gradient). The negative gradient is the unit of correctness: every `(family, parameter)` pair is tested by comparing the analytic gradient to a central finite difference of `risk`.

**Tech Stack:** Rust (2021 edition), `ndarray` (vectors/arrays), `statrs` (lgamma/digamma special functions), `thiserror` (typed errors), `approx` (float assertions, dev). `faer` is added in Plan 2.

**Source of truth:** the design spec `docs/design/2026-06-19-boostlss-v1-design.md` (§4 data model, §5 families, §12 errors). All formulas below come from §5.2 of that spec, which was verified against gamboostlss `R/families.R`.

> **Spec addendum (record in the spec's dependency list during this plan):** §3.2 of the spec lists `ndarray`, `faer`, `thiserror`. Special functions (`lgamma`, `digamma`) require `statrs` and float-comparison tests require `approx` (dev). Task 11 adds a one-line note to the spec recording these.

---

## File structure (Plan 1)

| File                                      | Responsibility                                                              |
| ----------------------------------------- | --------------------------------------------------------------------------- |
| `Cargo.toml` (root)                       | Workspace manifest (members, resolver)                                      |
| `.gitignore` (modify)                     | Add `/target`                                                               |
| `crates/boostlss/Cargo.toml`              | Core crate manifest + dependencies                                          |
| `crates/boostlss/src/lib.rs`              | Crate root; module declarations + public re-exports                         |
| `crates/boostlss/src/error.rs`            | `BoostlssError` (typed error enum)                                          |
| `crates/boostlss/src/util.rs`             | `weighted_mean`, `weighted_sd`, `minimize_1d` (1-D minimizer)               |
| `crates/boostlss/src/data.rs`             | `Dataset` (named f64 columns + response + weights) with boundary validation |
| `crates/boostlss/src/param.rs`            | `Link` enum, `ParamSpec`                                                    |
| `crates/boostlss/src/family/mod.rs`       | `Family` trait; `#[cfg(test)]` finite-difference harness                    |
| `crates/boostlss/src/family/gaussian.rs`  | `GaussianLss` (μ identity, σ log)                                           |
| `crates/boostlss/src/family/student_t.rs` | `StudentTLss` (μ identity, σ log, df log)                                   |
| `crates/boostlss/src/family/gamma.rs`     | `GammaLss` (μ log, σ=shape log)                                             |
| `crates/boostlss/src/family/nbinomial.rs` | `NBinomialLss` (μ log, σ=size log)                                          |

---

## Task 1: Workspace and crate scaffolding

**Files:**

- Create: `Cargo.toml` (workspace root)
- Modify: `.gitignore`
- Create: `crates/boostlss/Cargo.toml`
- Create: `crates/boostlss/src/lib.rs`

- [ ] **Step 1: Create the workspace manifest**

Create `Cargo.toml` at the repo root:

```toml
[workspace]
resolver = "2"
members = ["crates/boostlss"]
```

- [ ] **Step 2: Ignore the build directory**

Append to `.gitignore` (the file already exists):

```gitignore

# Rust build artifacts
/target
```

- [ ] **Step 3: Create the crate manifest**

Create `crates/boostlss/Cargo.toml`:

```toml
[package]
name = "boostlss"
version = "0.0.0"
edition = "2021"
description = "Boosting GAMLSS (distributional regression) in Rust"
license = "MIT OR Apache-2.0"

[dependencies]

[dev-dependencies]
```

- [ ] **Step 4: Create the crate root**

Create `crates/boostlss/src/lib.rs`:

```rust
//! boostlss — boosting GAMLSS (distributional regression) in Rust.

pub mod error;
pub mod util;
pub mod data;
pub mod param;
pub mod family;

pub use error::BoostlssError;
```

> Note: the `mod`/`pub use` lines above reference files created in later tasks. They will not compile until those files exist, so this step adds only the lines whose targets exist _as of the task that creates them_. To keep each task green, **comment out** the not-yet-created module lines now and uncomment each as its file lands. Start `lib.rs` as:

```rust
//! boostlss — boosting GAMLSS (distributional regression) in Rust.

pub mod error;
// pub mod util;   // Task 2
// pub mod data;   // Task 4
// pub mod param;  // Task 5
// pub mod family; // Task 6

// pub use error::BoostlssError;  // uncomment in Task 3
```

- [ ] **Step 5: Add dependencies via cargo (records current versions in Cargo.toml)**

Run from the repo root:

```bash
cargo add --package boostlss ndarray statrs thiserror
cargo add --package boostlss --dev approx proptest
```

Expected: `Cargo.toml` gains `[dependencies] ndarray`, `statrs`, `thiserror` and `[dev-dependencies] approx`, `proptest` with the currently-published versions.

- [ ] **Step 6: Verify the workspace builds**

Run: `cargo build --workspace`
Expected: PASS (compiles an empty crate with `error` module pending — actually `lib.rs` declares `pub mod error;`; comment it out until Task 3 if needed). If `error` is not yet created, temporarily make `lib.rs` just the doc comment so this step is green:

```rust
//! boostlss — boosting GAMLSS (distributional regression) in Rust.
```

Expected: `cargo build --workspace` PASS.

- [ ] **Step 7: Commit**

```bash
git add Cargo.toml .gitignore crates/boostlss/Cargo.toml crates/boostlss/src/lib.rs Cargo.lock
git commit -m "chore: scaffold boostlss cargo workspace and core crate"
```

---

## Task 2: Numeric utilities (`util.rs`)

**Files:**

- Create: `crates/boostlss/src/util.rs`
- Modify: `crates/boostlss/src/lib.rs` (uncomment `pub mod util;`)

- [ ] **Step 1: Write the failing tests**

Create `crates/boostlss/src/util.rs`:

```rust
//! Small numeric helpers shared across families.

use ndarray::Array1;

/// Weighted mean of `y`. With `w = None`, the ordinary mean.
pub fn weighted_mean(y: &Array1<f64>, w: Option<&Array1<f64>>) -> f64 {
    match w {
        None => y.sum() / y.len() as f64,
        Some(w) => {
            let sw: f64 = w.sum();
            y.iter().zip(w.iter()).map(|(yi, wi)| yi * wi).sum::<f64>() / sw
        }
    }
}

/// Weighted sample standard deviation (denominator = effective n - 1).
/// With `w = None` this is the ordinary sample standard deviation.
pub fn weighted_sd(y: &Array1<f64>, w: Option<&Array1<f64>>) -> f64 {
    let m = weighted_mean(y, w);
    match w {
        None => {
            let n = y.len() as f64;
            let ss: f64 = y.iter().map(|yi| (yi - m).powi(2)).sum();
            (ss / (n - 1.0)).sqrt()
        }
        Some(w) => {
            let sw: f64 = w.sum();
            let ss: f64 = y
                .iter()
                .zip(w.iter())
                .map(|(yi, wi)| wi * (yi - m).powi(2))
                .sum();
            (ss / (sw - 1.0)).sqrt()
        }
    }
}

/// Minimize a unimodal `f` on `[lo, hi]` by golden-section search.
/// Used for intercept-only MLE offsets that have no closed form.
pub fn minimize_1d<F: Fn(f64) -> f64>(f: F, lo: f64, hi: f64) -> f64 {
    const INV_PHI: f64 = 0.618_033_988_749_894_8; // 1/golden ratio
    const ITERS: usize = 100;
    let (mut a, mut b) = (lo, hi);
    let mut c = b - (b - a) * INV_PHI;
    let mut d = a + (b - a) * INV_PHI;
    let (mut fc, mut fd) = (f(c), f(d));
    for _ in 0..ITERS {
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - (b - a) * INV_PHI;
            fc = f(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + (b - a) * INV_PHI;
            fd = f(d);
        }
    }
    (a + b) / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn weighted_mean_unweighted_is_arithmetic_mean() {
        let y = array![1.0, 2.0, 3.0, 4.0];
        assert_relative_eq!(weighted_mean(&y, None), 2.5, epsilon = 1e-12);
    }

    #[test]
    fn weighted_mean_respects_weights() {
        let y = array![1.0, 3.0];
        let w = array![3.0, 1.0];
        assert_relative_eq!(weighted_mean(&y, Some(&w)), 1.5, epsilon = 1e-12);
    }

    #[test]
    fn weighted_sd_unweighted_is_sample_sd() {
        let y = array![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        // sample sd (n-1) of this classic set is exactly sqrt(32/7).
        assert_relative_eq!(weighted_sd(&y, None), (32.0_f64 / 7.0).sqrt(), epsilon = 1e-12);
    }

    #[test]
    fn minimize_1d_finds_parabola_vertex() {
        let x = minimize_1d(|x| (x - 3.0).powi(2) + 1.0, -10.0, 10.0);
        assert_relative_eq!(x, 3.0, epsilon = 1e-6);
    }
}
```

- [ ] **Step 2: Enable the module**

In `crates/boostlss/src/lib.rs`, ensure `pub mod util;` is present (uncomment it).

- [ ] **Step 3: Run tests to verify they pass**

Run: `cargo test --package boostlss util`
Expected: PASS (4 tests).

- [ ] **Step 4: Commit**

```bash
git add crates/boostlss/src/util.rs crates/boostlss/src/lib.rs
git commit -m "feat: add weighted-mean/sd and 1-D minimizer utilities"
```

---

## Task 3: Typed error enum (`error.rs`)

**Files:**

- Create: `crates/boostlss/src/error.rs`
- Modify: `crates/boostlss/src/lib.rs` (uncomment `pub use error::BoostlssError;`)

- [ ] **Step 1: Write the failing test**

Create `crates/boostlss/src/error.rs`:

```rust
//! Typed errors for boostlss.

use thiserror::Error;

/// All fallible boostlss operations return `Result<_, BoostlssError>`.
#[derive(Debug, Error)]
pub enum BoostlssError {
    #[error("dimension mismatch: {0}")]
    DimensionMismatch(String),

    #[error("unknown column: {0}")]
    UnknownColumn(String),

    #[error("non-finite value encountered: {0}")]
    NonFinite(String),

    #[error("response not supported by family: {0}")]
    UnsupportedResponse(String),

    #[error("feature value out of fitted range: {0}")]
    OutOfRange(String),

    #[error("singular system: {0}")]
    Singular(String),

    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("model did not converge: {0}")]
    NotConverged(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_includes_context() {
        let e = BoostlssError::UnknownColumn("vol".into());
        assert_eq!(e.to_string(), "unknown column: vol");
    }
}
```

- [ ] **Step 2: Enable the module and re-export**

In `crates/boostlss/src/lib.rs`: ensure `pub mod error;` is present and uncomment `pub use error::BoostlssError;`.

- [ ] **Step 3: Run test to verify it passes**

Run: `cargo test --package boostlss error`
Expected: PASS (1 test).

- [ ] **Step 4: Commit**

```bash
git add crates/boostlss/src/error.rs crates/boostlss/src/lib.rs
git commit -m "feat: add BoostlssError typed error enum"
```

---

## Task 4: Data model (`data.rs`)

**Files:**

- Create: `crates/boostlss/src/data.rs`
- Modify: `crates/boostlss/src/lib.rs` (uncomment `pub mod data;`)

- [ ] **Step 1: Write the failing tests**

Create `crates/boostlss/src/data.rs`:

```rust
//! `Dataset`: named numeric (f64) covariate columns plus a response vector
//! and optional observation weights. Validation happens at construction
//! (fail fast). v1 is numeric-only: categorical predictors are one-hot
//! encoded upstream by the caller.

use ndarray::Array1;

use crate::error::BoostlssError;

#[derive(Debug, Clone)]
pub struct Dataset {
    columns: Vec<(String, Array1<f64>)>,
    response: Array1<f64>,
    weights: Option<Array1<f64>>,
}

impl Dataset {
    /// Construct and validate a dataset.
    ///
    /// Errors if columns have unequal length, lengths disagree with the
    /// response, any value is non-finite, or weights are negative / wrong-sized.
    pub fn new(
        columns: Vec<(String, Array1<f64>)>,
        response: Array1<f64>,
        weights: Option<Array1<f64>>,
    ) -> Result<Self, BoostlssError> {
        let n = response.len();
        if n == 0 {
            return Err(BoostlssError::DimensionMismatch("empty response".into()));
        }
        if !response.iter().all(|v| v.is_finite()) {
            return Err(BoostlssError::NonFinite("response".into()));
        }
        for (name, col) in &columns {
            if col.len() != n {
                return Err(BoostlssError::DimensionMismatch(format!(
                    "column `{name}` has length {} but response has length {n}",
                    col.len()
                )));
            }
            if !col.iter().all(|v| v.is_finite()) {
                return Err(BoostlssError::NonFinite(format!("column `{name}`")));
            }
        }
        if let Some(w) = &weights {
            if w.len() != n {
                return Err(BoostlssError::DimensionMismatch(format!(
                    "weights have length {} but response has length {n}",
                    w.len()
                )));
            }
            if !w.iter().all(|v| v.is_finite() && *v >= 0.0) {
                return Err(BoostlssError::NonFinite("weights".into()));
            }
        }
        Ok(Self { columns, response, weights })
    }

    pub fn n_obs(&self) -> usize {
        self.response.len()
    }

    pub fn response(&self) -> &Array1<f64> {
        &self.response
    }

    pub fn weights(&self) -> Option<&Array1<f64>> {
        self.weights.as_ref()
    }

    /// Look up a covariate column by name.
    pub fn column(&self, name: &str) -> Result<&Array1<f64>, BoostlssError> {
        self.columns
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, c)| c)
            .ok_or_else(|| BoostlssError::UnknownColumn(name.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn ok_dataset() -> Dataset {
        Dataset::new(
            vec![("x".to_string(), array![1.0, 2.0, 3.0])],
            array![10.0, 20.0, 30.0],
            None,
        )
        .unwrap()
    }

    #[test]
    fn constructs_and_reads_back() {
        let d = ok_dataset();
        assert_eq!(d.n_obs(), 3);
        assert_eq!(d.column("x").unwrap(), &array![1.0, 2.0, 3.0]);
    }

    #[test]
    fn unknown_column_errors() {
        let d = ok_dataset();
        assert!(matches!(d.column("nope"), Err(BoostlssError::UnknownColumn(_))));
    }

    #[test]
    fn length_mismatch_errors() {
        let r = Dataset::new(
            vec![("x".to_string(), array![1.0, 2.0])],
            array![10.0, 20.0, 30.0],
            None,
        );
        assert!(matches!(r, Err(BoostlssError::DimensionMismatch(_))));
    }

    #[test]
    fn non_finite_response_errors() {
        let r = Dataset::new(vec![], array![1.0, f64::NAN], None);
        assert!(matches!(r, Err(BoostlssError::NonFinite(_))));
    }

    #[test]
    fn negative_weight_errors() {
        let r = Dataset::new(
            vec![],
            array![1.0, 2.0],
            Some(array![1.0, -0.5]),
        );
        assert!(matches!(r, Err(BoostlssError::NonFinite(_))));
    }
}
```

- [ ] **Step 2: Enable the module**

In `crates/boostlss/src/lib.rs`, uncomment `pub mod data;`.

- [ ] **Step 3: Run tests to verify they pass**

Run: `cargo test --package boostlss data`
Expected: PASS (5 tests).

- [ ] **Step 4: Commit**

```bash
git add crates/boostlss/src/data.rs crates/boostlss/src/lib.rs
git commit -m "feat: add Dataset with boundary validation"
```

---

## Task 5: Links and parameter specs (`param.rs`)

**Files:**

- Create: `crates/boostlss/src/param.rs`
- Modify: `crates/boostlss/src/lib.rs` (uncomment `pub mod param;`)

- [ ] **Step 1: Write the failing tests**

Create `crates/boostlss/src/param.rs`:

```rust
//! Link functions and per-parameter specifications.

/// Link mapping a distribution parameter (response scale) to/from the
/// additive-predictor (link) scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Link {
    Identity,
    Log,
    Logit,
}

impl Link {
    /// Parameter scale -> link scale.
    pub fn apply(&self, x: f64) -> f64 {
        match self {
            Link::Identity => x,
            Link::Log => x.ln(),
            Link::Logit => (x / (1.0 - x)).ln(),
        }
    }

    /// Link scale -> parameter scale (inverse link).
    pub fn inverse(&self, eta: f64) -> f64 {
        match self {
            Link::Identity => eta,
            Link::Log => eta.exp(),
            Link::Logit => 1.0 / (1.0 + (-eta).exp()),
        }
    }

    /// d(inverse)/d(eta) — provided for diagnostics.
    pub fn inverse_deriv(&self, eta: f64) -> f64 {
        match self {
            Link::Identity => 1.0,
            Link::Log => eta.exp(),
            Link::Logit => {
                let p = self.inverse(eta);
                p * (1.0 - p)
            }
        }
    }
}

/// Name + link for one distribution parameter (e.g. `{ "sigma", Log }`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParamSpec {
    pub name: &'static str,
    pub link: Link,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn identity_roundtrips() {
        assert_relative_eq!(Link::Identity.inverse(Link::Identity.apply(3.5)), 3.5, epsilon = 1e-12);
    }

    #[test]
    fn log_roundtrips() {
        assert_relative_eq!(Link::Log.inverse(Link::Log.apply(2.0)), 2.0, epsilon = 1e-12);
        assert_relative_eq!(Link::Log.inverse(0.0), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn logit_roundtrips() {
        assert_relative_eq!(Link::Logit.inverse(Link::Logit.apply(0.3)), 0.3, epsilon = 1e-12);
        assert_relative_eq!(Link::Logit.inverse(0.0), 0.5, epsilon = 1e-12);
    }
}
```

- [ ] **Step 2: Enable the module**

In `crates/boostlss/src/lib.rs`, uncomment `pub mod param;`.

- [ ] **Step 3: Run tests to verify they pass**

Run: `cargo test --package boostlss param`
Expected: PASS (3 tests).

- [ ] **Step 4: Commit**

```bash
git add crates/boostlss/src/param.rs crates/boostlss/src/lib.rs
git commit -m "feat: add Link functions and ParamSpec"
```

---

## Task 6: `Family` trait and finite-difference harness (`family/mod.rs`)

**Files:**

- Create: `crates/boostlss/src/family/mod.rs`
- Modify: `crates/boostlss/src/lib.rs` (uncomment `pub mod family;`)

- [ ] **Step 1: Write the trait and the test harness**

Create `crates/boostlss/src/family/mod.rs`:

```rust
//! The `Family` trait: one distribution = one self-contained module.

use ndarray::Array1;

use crate::error::BoostlssError;
use crate::param::ParamSpec;

pub trait Family {
    /// Parameter specs in canonical order (index = parameter `k`).
    fn params(&self) -> &[ParamSpec];

    /// Number of distribution parameters.
    fn n_params(&self) -> usize {
        self.params().len()
    }

    /// Reject responses outside the family's support.
    fn check_response(&self, y: &Array1<f64>) -> Result<(), BoostlssError>;

    /// Intercept-only MLE for parameter `k` (the offset / starting value),
    /// returned on the link scale.
    fn offset(&self, k: usize, y: &Array1<f64>, w: Option<&Array1<f64>>) -> f64;

    /// Empirical risk = total (weighted) negative log-likelihood given all
    /// predictors `etas` on the link scale.
    fn risk(&self, y: &Array1<f64>, etas: &[Array1<f64>], w: Option<&Array1<f64>>) -> f64;

    /// Negative gradient of the risk w.r.t. parameter `k`'s predictor
    /// (link scale), per observation. This is the pseudo-response that
    /// base-learners fit. Defined as `-d(NLL_i)/d(eta_k_i)` (unweighted;
    /// the engine applies weights when selecting/fitting learners).
    fn negative_gradient(&self, k: usize, y: &Array1<f64>, etas: &[Array1<f64>]) -> Array1<f64>;
}

pub mod gaussian;
pub mod student_t;
pub mod gamma;
pub mod nbinomial;

pub use gaussian::GaussianLss;
pub use student_t::StudentTLss;
pub use gamma::GammaLss;
pub use nbinomial::NBinomialLss;

/// Test-only harness: assert that a family's analytic `negative_gradient`
/// for parameter `k` matches the central finite difference of `risk`.
///
/// Since `risk` is the sum of per-observation NLLs and observation `i`'s NLL
/// depends only on `eta_k[i]`, the derivative of the *total* risk w.r.t.
/// `eta_k[i]` equals `d(NLL_i)/d(eta_k[i]) = -negative_gradient[i]`.
#[cfg(test)]
pub(crate) fn assert_gradient_matches<F: Family>(
    fam: &F,
    k: usize,
    y: &Array1<f64>,
    etas: &[Array1<f64>],
) {
    use approx::assert_relative_eq;
    const H: f64 = 1e-6;
    let analytic = fam.negative_gradient(k, y, etas);
    for i in 0..y.len() {
        let mut plus: Vec<Array1<f64>> = etas.to_vec();
        let mut minus: Vec<Array1<f64>> = etas.to_vec();
        plus[k][i] += H;
        minus[k][i] -= H;
        let d_risk = (fam.risk(y, &plus, None) - fam.risk(y, &minus, None)) / (2.0 * H);
        let numeric_ngrad = -d_risk;
        assert_relative_eq!(analytic[i], numeric_ngrad, max_relative = 1e-4, epsilon = 1e-6);
    }
}
```

- [ ] **Step 2: Enable the module (families pending)**

In `crates/boostlss/src/lib.rs`, uncomment `pub mod family;`. This will **not compile yet** because `family/mod.rs` declares the four family submodules created in Tasks 7–10. To keep Task 6 green, temporarily comment out the four `pub mod`/`pub use` family lines in `family/mod.rs`, leaving only the trait + harness:

```rust
// pub mod gaussian;     // Task 7
// pub mod student_t;    // Task 8
// pub mod gamma;        // Task 9
// pub mod nbinomial;    // Task 10
// pub use gaussian::GaussianLss;
// pub use student_t::StudentTLss;
// pub use gamma::GammaLss;
// pub use nbinomial::NBinomialLss;
```

- [ ] **Step 3: Verify it compiles (harness is unused until Task 7)**

Run: `cargo build --package boostlss`
Expected: PASS. (`cargo test` may warn that `assert_gradient_matches` is unused — acceptable until Task 7; do not add `#[allow]`, the warning disappears once used.)

- [ ] **Step 4: Commit**

```bash
git add crates/boostlss/src/family/mod.rs crates/boostlss/src/lib.rs
git commit -m "feat: add Family trait and finite-difference gradient harness"
```

---

## Task 7: `GaussianLss` (μ identity, σ log)

**Files:**

- Create: `crates/boostlss/src/family/gaussian.rs`
- Modify: `crates/boostlss/src/family/mod.rs` (uncomment the `gaussian` module + re-export)

- [ ] **Step 1: Write the implementation and tests**

Create `crates/boostlss/src/family/gaussian.rs`:

```rust
//! GaussianLss: Normal with mean μ (identity link) and standard deviation σ
//! (log link). σ is a STANDARD DEVIATION, not a variance.

use ndarray::Array1;
use std::f64::consts::PI;

use crate::error::BoostlssError;
use crate::family::Family;
use crate::param::{Link, ParamSpec};
use crate::util::{weighted_mean, weighted_sd};

const PARAMS: [ParamSpec; 2] = [
    ParamSpec { name: "mu", link: Link::Identity },
    ParamSpec { name: "sigma", link: Link::Log },
];

#[derive(Debug, Clone, Copy, Default)]
pub struct GaussianLss;

impl Family for GaussianLss {
    fn params(&self) -> &[ParamSpec] {
        &PARAMS
    }

    fn check_response(&self, y: &Array1<f64>) -> Result<(), BoostlssError> {
        if y.iter().all(|v| v.is_finite()) {
            Ok(())
        } else {
            Err(BoostlssError::UnsupportedResponse("Gaussian requires finite y".into()))
        }
    }

    fn offset(&self, k: usize, y: &Array1<f64>, w: Option<&Array1<f64>>) -> f64 {
        match k {
            0 => weighted_mean(y, w),          // mu: identity link
            1 => weighted_sd(y, w).ln(),       // sigma: log link
            _ => unreachable!("GaussianLss has 2 parameters"),
        }
    }

    fn risk(&self, y: &Array1<f64>, etas: &[Array1<f64>], w: Option<&Array1<f64>>) -> f64 {
        let (mu, log_sigma) = (&etas[0], &etas[1]);
        let mut acc = 0.0;
        for i in 0..y.len() {
            let sigma = log_sigma[i].exp();
            let z = (y[i] - mu[i]) / sigma;
            // -log dnorm = 0.5*ln(2π) + ln(σ) + 0.5 z²
            let nll = 0.5 * (2.0 * PI).ln() + log_sigma[i] + 0.5 * z * z;
            acc += weight(w, i) * nll;
        }
        acc
    }

    fn negative_gradient(&self, k: usize, y: &Array1<f64>, etas: &[Array1<f64>]) -> Array1<f64> {
        let (mu, log_sigma) = (&etas[0], &etas[1]);
        match k {
            0 => Array1::from_shape_fn(y.len(), |i| {
                let inv_var = (-2.0 * log_sigma[i]).exp(); // 1/σ²
                (y[i] - mu[i]) * inv_var
            }),
            1 => Array1::from_shape_fn(y.len(), |i| {
                let inv_var = (-2.0 * log_sigma[i]).exp();
                -1.0 + inv_var * (y[i] - mu[i]).powi(2)
            }),
            _ => unreachable!("GaussianLss has 2 parameters"),
        }
    }
}

#[inline]
fn weight(w: Option<&Array1<f64>>, i: usize) -> f64 {
    w.map_or(1.0, |w| w[i])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::assert_gradient_matches;
    use approx::assert_relative_eq;
    use ndarray::array;

    fn etas() -> Vec<Array1<f64>> {
        // arbitrary non-trivial predictor values on the link scale
        vec![
            array![0.5, -1.0, 2.0, 0.0],        // mu (identity)
            array![-0.3, 0.2, 0.7, -0.5],       // log sigma
        ]
    }

    fn y() -> Array1<f64> {
        array![0.2, -0.8, 1.5, 0.4]
    }

    #[test]
    fn offsets_are_mean_and_log_sd() {
        let fam = GaussianLss;
        let y = array![1.0, 2.0, 3.0, 4.0];
        assert_relative_eq!(fam.offset(0, &y, None), 2.5, epsilon = 1e-12);
        let expected_log_sd = weighted_sd(&y, None).ln();
        assert_relative_eq!(fam.offset(1, &y, None), expected_log_sd, epsilon = 1e-12);
    }

    #[test]
    fn risk_matches_manual_normal_nll() {
        let fam = GaussianLss;
        // single observation: y=0, mu=0, sigma=1 -> nll = 0.5*ln(2π)
        let r = fam.risk(&array![0.0], &[array![0.0], array![0.0]], None);
        assert_relative_eq!(r, 0.5 * (2.0 * std::f64::consts::PI).ln(), epsilon = 1e-12);
    }

    #[test]
    fn gradient_mu_matches_finite_difference() {
        assert_gradient_matches(&GaussianLss, 0, &y(), &etas());
    }

    #[test]
    fn gradient_sigma_matches_finite_difference() {
        assert_gradient_matches(&GaussianLss, 1, &y(), &etas());
    }
}
```

- [ ] **Step 2: Enable the module**

In `crates/boostlss/src/family/mod.rs`, uncomment `pub mod gaussian;` and `pub use gaussian::GaussianLss;`.

- [ ] **Step 3: Run tests to verify they pass**

Run: `cargo test --package boostlss family::gaussian`
Expected: PASS (4 tests), including both finite-difference gradient checks.

- [ ] **Step 4: Commit**

```bash
git add crates/boostlss/src/family/gaussian.rs crates/boostlss/src/family/mod.rs
git commit -m "feat: add GaussianLss family with gradient checks"
```

---

## Task 8: `StudentTLss` (μ identity, σ log, df log)

**Files:**

- Create: `crates/boostlss/src/family/student_t.rs`
- Modify: `crates/boostlss/src/family/mod.rs` (uncomment the `student_t` module + re-export)

- [ ] **Step 1: Write the implementation and tests**

Create `crates/boostlss/src/family/student_t.rs`:

```rust
//! StudentTLss: Student-t with location μ (identity), scale σ (log), and
//! degrees of freedom df (log). σ is a SCALE, not a variance.

use ndarray::Array1;
use statrs::function::gamma::{digamma, ln_gamma};

use crate::error::BoostlssError;
use crate::family::Family;
use crate::param::{Link, ParamSpec};
use crate::util::{minimize_1d, weighted_mean, weighted_sd};

const PARAMS: [ParamSpec; 3] = [
    ParamSpec { name: "mu", link: Link::Identity },
    ParamSpec { name: "sigma", link: Link::Log },
    ParamSpec { name: "df", link: Link::Log },
];

#[derive(Debug, Clone, Copy, Default)]
pub struct StudentTLss;

/// Per-observation negative log-likelihood given μ, σ, df.
fn nll_one(yi: f64, mu: f64, sigma: f64, df: f64) -> f64 {
    // ln_gamma(0.5) == 0.5*ln(π); it is the normalizing constant of the t density.
    let z2 = (yi - mu).powi(2) / (df * sigma * sigma);
    -(ln_gamma((df + 1.0) / 2.0)
        - sigma.ln()
        - ln_gamma(0.5)
        - ln_gamma(df / 2.0)
        - 0.5 * df.ln()
        - (df + 1.0) / 2.0 * (1.0 + z2).ln())
}

impl Family for StudentTLss {
    fn params(&self) -> &[ParamSpec] {
        &PARAMS
    }

    fn check_response(&self, y: &Array1<f64>) -> Result<(), BoostlssError> {
        if y.iter().all(|v| v.is_finite()) {
            Ok(())
        } else {
            Err(BoostlssError::UnsupportedResponse("Student-t requires finite y".into()))
        }
    }

    fn offset(&self, k: usize, y: &Array1<f64>, w: Option<&Array1<f64>>) -> f64 {
        // mu, sigma have natural plug-ins; df via 1-D MLE holding mu, sigma fixed.
        let mu = weighted_mean(y, w);
        let sigma = weighted_sd(y, w);
        match k {
            0 => mu,
            1 => sigma.ln(),
            2 => {
                // minimize total NLL over log-df in a wide, safe interval.
                let f = |log_df: f64| {
                    let df = log_df.exp();
                    y.iter().map(|&yi| nll_one(yi, mu, sigma, df)).sum::<f64>()
                };
                minimize_1d(f, (1.0_f64).ln(), (100.0_f64).ln())
            }
            _ => unreachable!("StudentTLss has 3 parameters"),
        }
    }

    fn risk(&self, y: &Array1<f64>, etas: &[Array1<f64>], w: Option<&Array1<f64>>) -> f64 {
        let (mu, log_sigma, log_df) = (&etas[0], &etas[1], &etas[2]);
        let mut acc = 0.0;
        for i in 0..y.len() {
            let sigma = log_sigma[i].exp();
            let df = log_df[i].exp();
            acc += w.map_or(1.0, |w| w[i]) * nll_one(y[i], mu[i], sigma, df);
        }
        acc
    }

    fn negative_gradient(&self, k: usize, y: &Array1<f64>, etas: &[Array1<f64>]) -> Array1<f64> {
        let (mu, log_sigma, log_df) = (&etas[0], &etas[1], &etas[2]);
        match k {
            0 => Array1::from_shape_fn(y.len(), |i| {
                let sigma = log_sigma[i].exp();
                let df = log_df[i].exp();
                let r = y[i] - mu[i];
                (df + 1.0) * r / (df * sigma * sigma + r * r)
            }),
            1 => Array1::from_shape_fn(y.len(), |i| {
                let sigma = log_sigma[i].exp();
                let df = log_df[i].exp();
                let r2 = (y[i] - mu[i]).powi(2);
                // safe algebraic form of: -1 + (df+1)/(df·σ²/r² + 1)
                (df + 1.0) * r2 / (df * sigma * sigma + r2) - 1.0
            }),
            2 => Array1::from_shape_fn(y.len(), |i| {
                let f = log_df[i];
                let df = f.exp();
                let sigma = log_sigma[i].exp();
                let r2 = (y[i] - mu[i]).powi(2);
                df / 2.0 * (digamma((df + 1.0) / 2.0) - digamma(df / 2.0))
                    - 0.5
                    - (df / 2.0 * (1.0 + r2 / (df * sigma * sigma)).ln()
                        - r2 / (sigma * sigma + r2 / df) * ((-f).exp() + 1.0) / 2.0)
            }),
            _ => unreachable!("StudentTLss has 3 parameters"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::assert_gradient_matches;
    use ndarray::array;

    fn y() -> Array1<f64> {
        array![0.2, -0.8, 1.5, 0.4, -2.0]
    }

    fn etas() -> Vec<Array1<f64>> {
        vec![
            array![0.5, -1.0, 2.0, 0.0, -0.5],   // mu
            array![-0.3, 0.2, 0.7, -0.5, 0.1],   // log sigma
            array![1.0, 1.5, 0.8, 2.0, 1.2],     // log df
        ]
    }

    #[test]
    fn gradient_mu_matches_finite_difference() {
        assert_gradient_matches(&StudentTLss, 0, &y(), &etas());
    }

    #[test]
    fn gradient_sigma_matches_finite_difference() {
        assert_gradient_matches(&StudentTLss, 1, &y(), &etas());
    }

    #[test]
    fn gradient_df_matches_finite_difference() {
        assert_gradient_matches(&StudentTLss, 2, &y(), &etas());
    }

    #[test]
    fn offset_df_is_in_range() {
        let off = StudentTLss.offset(2, &y(), None);
        assert!(off.exp() >= 1.0 && off.exp() <= 100.0);
    }
}
```

- [ ] **Step 2: Enable the module**

In `crates/boostlss/src/family/mod.rs`, uncomment `pub mod student_t;` and `pub use student_t::StudentTLss;`.

- [ ] **Step 3: Run tests to verify they pass**

Run: `cargo test --package boostlss family::student_t`
Expected: PASS (4 tests). If `ln_gamma`/`digamma` fail to resolve, confirm the path on docs.rs for the installed `statrs` version (module `statrs::function::gamma`).

- [ ] **Step 4: Commit**

```bash
git add crates/boostlss/src/family/student_t.rs crates/boostlss/src/family/mod.rs
git commit -m "feat: add StudentTLss family with gradient checks"
```

---

## Task 9: `GammaLss` (μ log, σ=shape log)

**Files:**

- Create: `crates/boostlss/src/family/gamma.rs`
- Modify: `crates/boostlss/src/family/mod.rs` (uncomment the `gamma` module + re-export)

- [ ] **Step 1: Write the implementation and tests**

Create `crates/boostlss/src/family/gamma.rs`:

```rust
//! GammaLss: mean μ (log link) and shape σ (log link).
//! Implies Y ~ Gamma(shape = σ, rate = σ/μ); mean μ, variance μ²/σ.

use ndarray::Array1;
use statrs::function::gamma::{digamma, ln_gamma};

use crate::error::BoostlssError;
use crate::family::Family;
use crate::param::{Link, ParamSpec};
use crate::util::minimize_1d;

const PARAMS: [ParamSpec; 2] = [
    ParamSpec { name: "mu", link: Link::Log },
    ParamSpec { name: "sigma", link: Link::Log },
];

#[derive(Debug, Clone, Copy, Default)]
pub struct GammaLss;

/// Per-observation NLL given μ (mean) and σ (shape):
/// lgamma(σ) + σ·y/μ − σ·ln y − σ·ln σ + σ·ln μ + ln y.
fn nll_one(yi: f64, mu: f64, sigma: f64) -> f64 {
    ln_gamma(sigma) + sigma * yi / mu - sigma * yi.ln() - sigma * sigma.ln()
        + sigma * mu.ln()
        + yi.ln()
}

impl Family for GammaLss {
    fn params(&self) -> &[ParamSpec] {
        &PARAMS
    }

    fn check_response(&self, y: &Array1<f64>) -> Result<(), BoostlssError> {
        if y.iter().all(|v| v.is_finite() && *v > 0.0) {
            Ok(())
        } else {
            Err(BoostlssError::UnsupportedResponse("Gamma requires y > 0".into()))
        }
    }

    fn offset(&self, k: usize, y: &Array1<f64>, w: Option<&Array1<f64>>) -> f64 {
        // Method-of-moments plug-ins, then a 1-D MLE for the requested parameter.
        let n = y.len() as f64;
        let mean = match w {
            None => y.sum() / n,
            Some(w) => y.iter().zip(w).map(|(a, b)| a * b).sum::<f64>() / w.sum(),
        };
        let var = y.iter().map(|yi| (yi - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let shape_mom = (mean * mean / var).max(1e-3);
        match k {
            0 => {
                let f = |log_mu: f64| {
                    let mu = log_mu.exp();
                    y.iter().map(|&yi| nll_one(yi, mu, shape_mom)).sum::<f64>()
                };
                minimize_1d(f, mean.ln() - 5.0, mean.ln() + 5.0)
            }
            1 => {
                let f = |log_sigma: f64| {
                    let sigma = log_sigma.exp();
                    y.iter().map(|&yi| nll_one(yi, mean, sigma)).sum::<f64>()
                };
                minimize_1d(f, shape_mom.ln() - 5.0, shape_mom.ln() + 5.0)
            }
            _ => unreachable!("GammaLss has 2 parameters"),
        }
    }

    fn risk(&self, y: &Array1<f64>, etas: &[Array1<f64>], w: Option<&Array1<f64>>) -> f64 {
        let (log_mu, log_sigma) = (&etas[0], &etas[1]);
        let mut acc = 0.0;
        for i in 0..y.len() {
            let mu = log_mu[i].exp();
            let sigma = log_sigma[i].exp();
            acc += w.map_or(1.0, |w| w[i]) * nll_one(y[i], mu, sigma);
        }
        acc
    }

    fn negative_gradient(&self, k: usize, y: &Array1<f64>, etas: &[Array1<f64>]) -> Array1<f64> {
        let (log_mu, log_sigma) = (&etas[0], &etas[1]);
        match k {
            0 => Array1::from_shape_fn(y.len(), |i| {
                let mu = log_mu[i].exp();
                let sigma = log_sigma[i].exp();
                sigma * (y[i] / mu - 1.0)
            }),
            1 => Array1::from_shape_fn(y.len(), |i| {
                let mu = log_mu[i].exp();
                let sigma = log_sigma[i].exp();
                sigma * (-digamma(sigma) + sigma.ln() + 1.0 - mu.ln() + y[i].ln() - y[i] / mu)
            }),
            _ => unreachable!("GammaLss has 2 parameters"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::assert_gradient_matches;
    use ndarray::array;

    fn y() -> Array1<f64> {
        array![0.5, 1.2, 2.0, 3.5, 0.9]
    }

    fn etas() -> Vec<Array1<f64>> {
        vec![
            array![0.1, 0.5, 0.8, 1.2, 0.0],   // log mu
            array![0.3, -0.2, 0.6, 0.1, 0.4],  // log sigma (shape)
        ]
    }

    #[test]
    fn rejects_nonpositive_response() {
        assert!(GammaLss.check_response(&array![1.0, 0.0]).is_err());
        assert!(GammaLss.check_response(&array![1.0, 2.0]).is_ok());
    }

    #[test]
    fn gradient_mu_matches_finite_difference() {
        assert_gradient_matches(&GammaLss, 0, &y(), &etas());
    }

    #[test]
    fn gradient_sigma_matches_finite_difference() {
        assert_gradient_matches(&GammaLss, 1, &y(), &etas());
    }
}
```

- [ ] **Step 2: Enable the module**

In `crates/boostlss/src/family/mod.rs`, uncomment `pub mod gamma;` and `pub use gamma::GammaLss;`.

- [ ] **Step 3: Run tests to verify they pass**

Run: `cargo test --package boostlss family::gamma`
Expected: PASS (3 tests).

- [ ] **Step 4: Commit**

```bash
git add crates/boostlss/src/family/gamma.rs crates/boostlss/src/family/mod.rs
git commit -m "feat: add GammaLss family with gradient checks"
```

---

## Task 10: `NBinomialLss` (μ log, σ=size log)

**Files:**

- Create: `crates/boostlss/src/family/nbinomial.rs`
- Modify: `crates/boostlss/src/family/mod.rs` (uncomment the `nbinomial` module + re-export)

- [ ] **Step 1: Write the implementation and tests**

Create `crates/boostlss/src/family/nbinomial.rs`:

```rust
//! NBinomialLss: mean μ (log link) and size σ=θ (log link).
//! Implies Y ~ NB(mean = μ, size = σ); variance μ + μ²/σ.

use ndarray::Array1;
use statrs::function::gamma::{digamma, ln_gamma};

use crate::error::BoostlssError;
use crate::family::Family;
use crate::param::{Link, ParamSpec};
use crate::util::minimize_1d;

const PARAMS: [ParamSpec; 2] = [
    ParamSpec { name: "mu", link: Link::Log },
    ParamSpec { name: "sigma", link: Link::Log },
];

#[derive(Debug, Clone, Copy, Default)]
pub struct NBinomialLss;

/// Per-observation NLL given μ (mean, = e^{eta_mu}) and σ (size):
/// −[lgamma(y+σ) − lgamma(σ) − lgamma(y+1) + σ·ln σ + y·ln μ − (y+σ)·ln(μ+σ)].
fn nll_one(yi: f64, mu: f64, sigma: f64) -> f64 {
    -(ln_gamma(yi + sigma) - ln_gamma(sigma) - ln_gamma(yi + 1.0)
        + sigma * sigma.ln()
        + yi * mu.ln()
        - (yi + sigma) * (mu + sigma).ln())
}

impl Family for NBinomialLss {
    fn params(&self) -> &[ParamSpec] {
        &PARAMS
    }

    fn check_response(&self, y: &Array1<f64>) -> Result<(), BoostlssError> {
        let ok = y
            .iter()
            .all(|v| v.is_finite() && *v >= 0.0 && (*v - v.round()).abs() < 1e-9);
        if ok {
            Ok(())
        } else {
            Err(BoostlssError::UnsupportedResponse(
                "Negative Binomial requires non-negative integer y".into(),
            ))
        }
    }

    fn offset(&self, k: usize, y: &Array1<f64>, w: Option<&Array1<f64>>) -> f64 {
        let n = y.len() as f64;
        let mean = match w {
            None => y.sum() / n,
            Some(w) => y.iter().zip(w).map(|(a, b)| a * b).sum::<f64>() / w.sum(),
        };
        let var = (y.iter().map(|yi| (yi - mean).powi(2)).sum::<f64>() / (n - 1.0)).max(mean + 1e-6);
        // var = mu + mu²/size  =>  size = mu² / (var - mu)
        let size_mom = (mean * mean / (var - mean)).max(1e-3);
        match k {
            0 => {
                let f = |log_mu: f64| {
                    let mu = log_mu.exp();
                    y.iter().map(|&yi| nll_one(yi, mu, size_mom)).sum::<f64>()
                };
                minimize_1d(f, mean.max(1e-6).ln() - 5.0, mean.max(1e-6).ln() + 5.0)
            }
            1 => {
                let f = |log_sigma: f64| {
                    let sigma = log_sigma.exp();
                    y.iter().map(|&yi| nll_one(yi, mean, sigma)).sum::<f64>()
                };
                minimize_1d(f, size_mom.ln() - 5.0, size_mom.ln() + 5.0)
            }
            _ => unreachable!("NBinomialLss has 2 parameters"),
        }
    }

    fn risk(&self, y: &Array1<f64>, etas: &[Array1<f64>], w: Option<&Array1<f64>>) -> f64 {
        let (log_mu, log_sigma) = (&etas[0], &etas[1]);
        let mut acc = 0.0;
        for i in 0..y.len() {
            let mu = log_mu[i].exp();
            let sigma = log_sigma[i].exp();
            acc += w.map_or(1.0, |w| w[i]) * nll_one(y[i], mu, sigma);
        }
        acc
    }

    fn negative_gradient(&self, k: usize, y: &Array1<f64>, etas: &[Array1<f64>]) -> Array1<f64> {
        let (log_mu, log_sigma) = (&etas[0], &etas[1]);
        match k {
            0 => Array1::from_shape_fn(y.len(), |i| {
                let mu = log_mu[i].exp();
                let sigma = log_sigma[i].exp();
                y[i] - (y[i] + sigma) * mu / (mu + sigma)
            }),
            1 => Array1::from_shape_fn(y.len(), |i| {
                let mu = log_mu[i].exp();
                let sigma = log_sigma[i].exp();
                sigma
                    * (digamma(y[i] + sigma) - digamma(sigma) + sigma.ln() + 1.0
                        - (mu + sigma).ln()
                        - (sigma + y[i]) / (mu + sigma))
            }),
            _ => unreachable!("NBinomialLss has 2 parameters"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::assert_gradient_matches;
    use ndarray::array;

    fn y() -> Array1<f64> {
        array![0.0, 1.0, 3.0, 2.0, 5.0]
    }

    fn etas() -> Vec<Array1<f64>> {
        vec![
            array![0.3, 0.8, 1.1, 0.5, 1.4],   // log mu
            array![0.5, 0.2, 0.9, 0.4, 0.7],   // log sigma (size)
        ]
    }

    #[test]
    fn rejects_noninteger_or_negative_response() {
        assert!(NBinomialLss.check_response(&array![0.0, 1.5]).is_err());
        assert!(NBinomialLss.check_response(&array![0.0, -1.0]).is_err());
        assert!(NBinomialLss.check_response(&array![0.0, 1.0, 7.0]).is_ok());
    }

    #[test]
    fn gradient_mu_matches_finite_difference() {
        assert_gradient_matches(&NBinomialLss, 0, &y(), &etas());
    }

    #[test]
    fn gradient_sigma_matches_finite_difference() {
        assert_gradient_matches(&NBinomialLss, 1, &y(), &etas());
    }
}
```

- [ ] **Step 2: Enable the module**

In `crates/boostlss/src/family/mod.rs`, uncomment `pub mod nbinomial;` and `pub use nbinomial::NBinomialLss;`.

- [ ] **Step 3: Run tests to verify they pass**

Run: `cargo test --package boostlss family::nbinomial`
Expected: PASS (3 tests).

- [ ] **Step 4: Commit**

```bash
git add crates/boostlss/src/family/nbinomial.rs crates/boostlss/src/family/mod.rs
git commit -m "feat: add NBinomialLss family with gradient checks"
```

---

## Task 11: Final wiring, lint gate, and spec dependency note

**Files:**

- Modify: `crates/boostlss/src/lib.rs` (public re-exports)
- Modify: `docs/design/2026-06-19-boostlss-v1-design.md` (§3.2 dependency note)

- [ ] **Step 1: Finalize public API re-exports**

Set `crates/boostlss/src/lib.rs` to:

```rust
//! boostlss — boosting GAMLSS (distributional regression) in Rust.

pub mod data;
pub mod error;
pub mod family;
pub mod param;
pub mod util;

pub use data::Dataset;
pub use error::BoostlssError;
pub use family::{Family, GammaLss, GaussianLss, NBinomialLss, StudentTLss};
pub use param::{Link, ParamSpec};
```

- [ ] **Step 2: Add an integration smoke test**

Create `crates/boostlss/tests/families_smoke.rs`:

```rust
use boostlss::family::Family;
use boostlss::{GaussianLss, NBinomialLss};
use ndarray::array;

#[test]
fn families_expose_expected_parameter_counts() {
    assert_eq!(GaussianLss.n_params(), 2);
    assert_eq!(GaussianLss.params()[1].name, "sigma");
}

#[test]
fn offset_initializes_finite_predictors() {
    let y = array![2.0, 3.0, 5.0, 7.0];
    let off_mu = GaussianLss.offset(0, &y, None);
    let off_sigma = GaussianLss.offset(1, &y, None);
    assert!(off_mu.is_finite() && off_sigma.is_finite());

    let counts = array![0.0, 1.0, 2.0, 4.0];
    assert!(NBinomialLss.offset(0, &counts, None).is_finite());
}
```

- [ ] **Step 3: Record the new dependencies in the spec**

In `docs/design/2026-06-19-boostlss-v1-design.md`, §3.2, add two bullets after the `thiserror` line:

```markdown
- `statrs` — special functions (`lgamma`, `digamma`) used by family log-likelihoods/gradients.
- `approx` (dev) — float-tolerance assertions in gradient/recovery tests.
```

- [ ] **Step 4: Run the full quality gate (from AGENTS.md)**

Run each and confirm:

```bash
cargo fmt -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
```

Expected: `fmt` clean; `clippy` no warnings; all tests PASS (utilities, error, data, links, and the four families' gradient checks + smoke tests). If `clippy` flags anything, fix the code (do not silence with `#[allow]` unless justified in a comment).

- [ ] **Step 5: Commit**

```bash
git add crates/boostlss/src/lib.rs crates/boostlss/tests/families_smoke.rs docs/design/2026-06-19-boostlss-v1-design.md
git commit -m "feat: finalize Plan 1 public API and record family dependencies"
```

---

## Done criteria (Plan 1)

- The workspace builds; `cargo fmt`/`clippy -D warnings`/`test` all pass.
- `Dataset` validates inputs at construction.
- `Link` (Identity/Log/Logit) round-trips.
- All four families implement `Family`, with **every `(family, parameter)` gradient verified against finite differences** and response-support checks for Gamma/NBinomial.
- Public API re-exports `Dataset`, `Family`, the four families, `Link`, `ParamSpec`, `BoostlssError`.

**Next:** Plan 2 — Base-learners (`Linear`, `PSpline`): B-spline basis, difference penalty, df↔λ via Demmler–Reinsch, faer Cholesky solve with cached factorization, and out-of-range linear extrapolation (spec §6).
