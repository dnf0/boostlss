# boostlss — Design Spec (v1)

- Status: Draft for review
- Date: 2026-06-19
- Author: Daniel Fisher (with Claude Code)
- Scope: First version (v1) of a Rust reimplementation of the R package
  [`gamboostlss`](https://github.com/boost-r/gamboostlss) and the parts of
  [`mboost`](https://github.com/boost-R/mboost) it depends on.

---

## 1. Goal and non-goals

### Goal

Build an **idiomatic, performant Rust library** for _boosting GAMLSS_ —
Generalized Additive Models for Location, Scale and Shape — i.e. distributional
regression via component-wise gradient boosting, where **every** parameter of a
distribution (location, scale, shape) gets its own additive predictor built from
base-learners.

The library is inspired by `gamboostlss`/`mboost` and reproduces their
algorithms and numerical conventions faithfully, but it is free to present a
clean, type-safe Rust API rather than mirror the R object model.

### Non-goals (v1)

- Bit-for-bit numerical parity with R (we validate _closeness_, see §11), not
  identical floating-point results.
- A formula-string DSL (the typed builder is the v1 API; a formula/macro layer
  is on the roadmap, §13).
- Categorical/factor expansion (users one-hot encode upstream in v1, §4).
- The base-learners, families, and features listed in §13 (Roadmap).

### Primary objective alignment

Per `AGENTS.md`: clear, maintainable implementations over clever shortcuts;
`Result`-centric APIs with typed errors; narrow, composable modules; no
`unwrap`/`expect` on production paths; dependencies declared deliberately.

---

## 2. Background (verified against primary sources)

GAMLSS models a response `y` with a distribution whose `K` parameters
`θ₁…θ_K` (e.g. μ, σ, ν) are each modeled by an additive predictor `η_{θk}` on a
**link scale**. Component-wise gradient boosting fits these predictors by
repeatedly:

1. computing the negative gradient of the empirical risk (negative
   log-likelihood, NLL) w.r.t. a parameter's predictor,
2. fitting every candidate base-learner to that pseudo-response by (penalized)
   least squares,
3. selecting the best base-learner by residual sum of squares (RSS), and
4. updating the predictor by a small step (`step_length · fit`).

Two fitting schemes exist:

- **Cyclical** (Mayr et al. 2012): in each iteration, update each parameter in
  turn (one base-learner each), conditioning on the others' current fits.
  `mstop` is a **vector** (one per parameter).
- **Non-cyclical** (Thomas et al. 2018): in each iteration, find the best update
  for every parameter, then commit only the single parameter giving the largest
  risk reduction. `mstop` is a **scalar**. The currently-shipped package variant
  selects each parameter's base-learner by RSS (the "inner-loss" variant), then
  chooses the parameter to update by NLL reduction.

All numeric constants, formulas, links, and defaults in this spec are taken from
a primary-source review of the `gamboostlss`/`mboost` source, the 2012 and 2018
papers, and the CRAN manuals. Key references are listed in §14.

---

## 3. Architecture overview

Chosen architecture: **generic family + enum base-learners** (static dispatch on
the hot path; each family is a self-contained, independently testable unit).

- `Family` is a **trait**; the fitted model is **generic over `F: Family`** so
  family calls monomorphize — no dynamic dispatch in the boosting loop.
- Base-learners are a **`BaseLearner` enum** (`Linear`, `PSpline`) — a small,
  bounded, known set dispatched by `match`.
- Fitting algorithm (`Cyclic` / `NonCyclic`) is selected by config and operates
  over a shared engine.

### 3.1 Workspace layout

```
boostlss/
  Cargo.toml                  # cargo workspace
  crates/
    boostlss/                 # core Rust library
      src/
        lib.rs
        data.rs               # Dataset: named f64 columns + response
        param.rs              # Param, Link (Identity/Log/Logit), ParamSpec
        error.rs              # BoostlssError (thiserror)
        family/
          mod.rs              # Family trait, ParamSpec
          gaussian.rs         # GaussianLss  (mu, sigma)
          student_t.rs        # StudentTLss  (mu, sigma, df)
          gamma.rs            # GammaLss     (mu, sigma)
          nbinomial.rs        # NBinomialLss (mu, sigma)
        learner/
          mod.rs              # BaseLearner enum + LearnerFit (cached state)
          linear.rs           # bols-style
          pspline.rs          # bbs-style (B-spline basis, difference penalty)
          penalty.rs          # df<->lambda via Demmler-Reinsch
        engine/
          mod.rs              # shared loop primitives, offsets, stabilization
          cyclical.rs
          noncyclical.rs
        model.rs              # BoostLss builder + Fitted (predict/coef)
        cv.rs                 # cvrisk: folds, grid, risk paths
      examples/
      benches/                # criterion
      tests/
        fixtures/             # committed R golden outputs + generate.R
    boostlss-py/              # pyo3 + maturin bindings (thin)
  docs/
    adr/                      # architecture decision records
    superpowers/specs/        # this spec
```

### 3.2 Dependencies (declared deliberately, per AGENTS.md)

- `ndarray` — data, gradients, design matrices (`Array1<f64>`, `Array2<f64>`).
- `faer` — Cholesky/QR factorizations for penalized least-squares solves.
- `thiserror` — typed error enum.
- `rayon` (optional feature `parallel`) — parallelize cvrisk folds.
- `proptest` (dev) — property-based tests.
- `criterion` (dev) — benchmarks.
- `boostlss-py`: `pyo3`, `numpy`, `maturin` (build).

The ndarray↔faer bridge (converting design matrices/vectors between the two) is
isolated in one module and documented in an ADR.

---

## 4. Data model

`Dataset` holds **numeric (f64) columns by name** plus a response vector.

```rust
pub struct Dataset {
    columns: Vec<(String, Array1<f64>)>,  // covariates, by name
    response: Array1<f64>,                 // y
    weights: Option<Array1<f64>>,          // observation weights (default all 1)
}
```

- v1 is **numeric-only**: categorical predictors are one-hot encoded by the user
  before constructing a `Dataset`.
- Construction validates: equal column lengths, finite values, response length
  matches, weights (if present) non-negative and correctly sized. Fail fast at
  the boundary (`Result<Dataset, BoostlssError>`).
- Each base-learner names the column(s) it consumes; the builder resolves names
  to columns at `build()` time and errors on unknown names.

For the Python layer, numpy arrays + a list of column names map onto this struct
(see §10).

---

## 5. Families

### 5.1 The `Family` trait

```rust
pub struct ParamSpec { pub name: &'static str, pub link: Link }

pub enum Link { Identity, Log, Logit }
impl Link {
    fn apply(&self, x: f64) -> f64;     // parameter scale -> link scale
    fn inverse(&self, eta: f64) -> f64; // link scale -> parameter scale
    fn deriv(&self, eta: f64) -> f64;   // d(inverse)/d(eta), for diagnostics
}

pub trait Family {
    fn params(&self) -> &[ParamSpec];
    fn check_response(&self, y: &Array1<f64>) -> Result<(), BoostlssError>;

    /// Intercept-only MLE for parameter k (offset / starting value), link scale.
    fn offset(&self, k: usize, y: &Array1<f64>, w: Option<&Array1<f64>>) -> f64;

    /// Empirical risk = total negative log-likelihood given all predictors
    /// (link scale), summed (weighted) over observations.
    fn risk(&self, y: &Array1<f64>, etas: &[Array1<f64>], w: Option<&Array1<f64>>) -> f64;

    /// Negative gradient of the risk w.r.t. parameter k's predictor (link scale),
    /// per observation. This is the pseudo-response base-learners fit.
    fn negative_gradient(&self, k: usize, y: &Array1<f64>, etas: &[Array1<f64>]) -> Array1<f64>;
}
```

`etas[k]` is the current additive predictor for parameter `k` on the link scale.
`response(k, eta)` is provided as `Link::inverse` applied elementwise.

Each family is one module with one struct, so the finite-difference gradient
test (§11) targets each `(family, parameter)` pair in isolation.

### 5.2 The four v1 families

Conventions: `f` = predictor on the link scale for the parameter in question;
`ngradient` = `−d(NLL)/d(f)`; other parameters held fixed at current fits. These
match the package's `R/families.R` implementation exactly (reported as
implemented, not re-derived).

> **Parameterization warning** (differs from `gamlss.dist`): Gaussian/Student-t
> `σ` is a **standard deviation / scale** (not a variance); Gamma `σ` is the
> **shape**; NBinomial `σ` is the **size** (θ). We adopt the gamboostlss
> parameterization.

#### GaussianLss — μ (identity), σ (log)

- `μ` offset: `weighted_mean(y, w)`; `σ` offset: `log(weighted_sd(y, w))`.
- `ngradient_μ = (y − μ) / σ²` where `μ = f_μ`, `σ = exp(f_σ)`.
- `ngradient_σ = −1 + exp(−2 f_σ) · (y − μ)²`.
- NLL per obs: `−log dnorm(y; μ, σ)`.

#### StudentTLss — μ (identity), σ (log), ν=df (log)

- `ngradient_μ = (df+1)(y−μ) / (df·σ² + (y−μ)²)`.
- `ngradient_σ = −1 + (df+1) / (df·e^{2 f_σ}/(y−μ)² + 1)`.
- `ngradient_df` (with `df = exp(f)`): `exp(f)/2·(ψ((exp(f)+1)/2) − ψ(exp(f)/2)) − 0.5 − ( exp(f)/2·log(1 + (y−μ)²/(exp(f)·σ²)) − (y−μ)²/(σ² + (y−μ)²/exp(f))·(exp(−f)+1)/2 )`.
- NLL per obs: `−[lgamma((df+1)/2) − log σ − lgamma(1/2) − lgamma(df/2) − 0.5·log df − (df+1)/2·log(1 + (y−μ)²/(df·σ²))]`.
- μ, σ offsets closed-form analogous to Gaussian; df offset via 1-D MLE.

#### GammaLss — μ=mean (log), σ=shape (log)

Implied `Y ~ Gamma(shape = σ, rate = σ/μ)`; mean μ, variance μ²/σ.

- `ngradient_μ = σ·(y/μ − 1)` where `μ = exp(f_μ)`.
- `ngradient_σ = σ·(−ψ(σ) + log σ + 1 − log μ + log y − y/μ)` where `σ = exp(f_σ)`.
- Offsets: 1-D numerical MLE (`optimize`) with the complementary parameter set by
  method of moments.

#### NBinomialLss — μ=mean (log), σ=size (log)

Implied `Y ~ NB(mean = μ, size = σ)`; variance μ + μ²/σ.

- `ngradient_μ = y − (y+σ)·μ/(μ+σ)` where `μ = exp(f_μ)`.
- `ngradient_σ = σ·(ψ(y+σ) − ψ(σ) + log σ + 1 − log(μ+σ) − (σ+y)/(μ+σ))`.
- Offsets: 1-D numerical MLE.

`ψ` = digamma. `check_response`: Gamma requires `y > 0`; NBinomial requires
non-negative integers; Gaussian/Student-t require finite `y`.

---

## 6. Base-learners

```rust
pub enum BaseLearner { Linear(Linear), PSpline(PSpline) }
```

Every base-learner is a **linear smoother** `û = X (XᵀX + λK)⁻¹ Xᵀ u = S·u`
(λ = 0 ⇒ OLS).

### 6.1 Performance: precompute once, solve each iteration

A base-learner's design `X` and penalty `K` are **fixed for the whole fit**. At
fit start we factor `(XᵀX + λK)` **once** (faer Cholesky) and cache the factor in
`LearnerFit`. Each boosting iteration is then a cheap solve against the current
negative gradient `u`: compute `Xᵀu`, back/forward-substitute, and `X · β̂`.
`LearnerFit` also stores accumulated coefficients and a selection counter (for
`selection_frequencies`). This is the main speed lever; it is recorded in an ADR.

### 6.2 Linear (`bols`-style)

- `intercept = true` (default) prepends a column of ones; `lambda = 0` (default)
  ⇒ unpenalized OLS.
- `intercept = false` ⇒ no intercept column (user mean-centers continuous
  covariates upstream).
- If a multi-column matrix is supplied directly, it is used as the design as-is
  (no intercept added) — matches mboost.

### 6.3 P-spline (`bbs`-style)

Defaults (from mboost `bbs`):

| Param         | Default           | Meaning                                        |
| ------------- | ----------------- | ---------------------------------------------- |
| `knots`       | 20                | equidistant **inner** knots                    |
| `degree`      | 3                 | cubic B-splines                                |
| `differences` | 2                 | 2nd-order difference penalty                   |
| `df`          | 4                 | target degrees of freedom                      |
| `lambda`      | derived from `df` | smoothing parameter                            |
| `center`      | false             | reparameterize away the unpenalized null space |

- Boundary knots default to the data range.
- Penalty `K = DᵀD`, `D = diff(I, differences)`.
- With `center = false` and a 2nd-order penalty, the linear part is unpenalized,
  so `df` must exceed `differences` (df > 2). `center = true` removes the null
  space (any positive df admissible).

### 6.4 `df` ↔ `λ` (Demmler–Reinsch)

- Default df definition: `df(λ) = trace(2S − SᵀS)` (the mboost default, chosen so
  RSS comparison across base-learners is unbiased). The alternate `trace(S)` is a
  config option.
- Computation (Ruppert et al. 2003, App. B.1.1): form weighted Gram
  `XtX = Xᵀ W X` and penalty `K`; stabilize `A = XtX + ε·K`; Cholesky
  `Rm = chol(A)⁻¹`; eigenvalues `d` of `Rmᵀ K Rm` (generalized eigenvalues of
  `(K, XtX)`). Then `trace(S) = Σ 1/(1+λ dⱼ)`,
  `trace(SᵀS) = Σ 1/(1+λ dⱼ)²`. Solve `df(λ) = target_df` for λ by 1-D
  root-finding. Edge cases: `df ≥ rank(X) ⇒ λ = 0`; `df` and `λ` are mutually
  exclusive.

---

## 7. Boosting engine

### 7.1 Shared config

```rust
pub struct Config {
    pub algorithm: Algorithm,         // Cyclic | NonCyclic
    pub step_length: f64,             // ν, default 0.1
    pub stabilization: Stabilization, // None | Mad | L2, default None
}
pub enum Mstop { Scalar(usize), PerParam(Vec<usize>) } // default 100/param
```

Initialization: each `η_{θk}^[0]` is set to `family.offset(k, …)` (broadcast to a
constant vector).

### 7.2 Cyclical

For `m = 1..max(mstopₖ)`, for each parameter `k`:

1. if `m > mstopₖ`, freeze and skip;
2. `u = stabilize(family.negative_gradient(k, y, etas))`;
3. fit every base-learner of `k` to `u` (cached-factor solve);
4. select `j*` minimizing RSS `Σ(uᵢ − û_{ij})²`;
5. `etas[k] += step_length · û_{j*}` (immediately visible to later parameters).

`Mstop` must be `PerParam` (or `Scalar` broadcast) in cyclic mode.

### 7.3 Non-cyclical (inner-loss variant, as shipped)

`Mstop` is `Scalar`. For `m = 1..mstop`:

1. for each parameter `k`: compute `u`, fit base-learners, select `j*` by **RSS**
   (inner loss), and compute the **risk after** the tentative update
   `Δρₖ = risk(η with etas[k] += ν·û_{j*})`;
2. choose `k* = argminₖ Δρₖ`;
3. commit only `k*`: `etas[k*] += step_length · û_{k* j*}`; all other parameters
   unchanged.

(The outer-loss variant — selecting each parameter's base-learner by NLL rather
than RSS — is deferred to the roadmap, §13.)

### 7.4 Gradient stabilization

Applied to the negative gradient `u` before fitting (matches
`stabilize_ngradient`):

- `None` (default): unchanged.
- `Mad`: divide by the weighted median absolute deviation
  `median_i(|uᵢ − median_j(uⱼ)|)`; divisor floored at `1e-4` (no ceiling).
- `L2`: divide by the weighted RMS `sqrt(weighted_mean(u²))`; divisor clamped to
  `[1e-4, 1e4]`.

`weighted_median` follows mboost's tie-breaking (drop zero weights; sort; first
index where cumulative weight fraction > 0.5; if total weight is an even integer,
mean of the two bracketing values). Named constants (`MAD_FLOOR = 1e-4`,
`L2_CLAMP = (1e-4, 1e4)`) per AGENTS.md (no magic numbers).

---

## 8. cvrisk (cross-validation tuning)

### 8.1 Resampling

```rust
pub enum Resampling { Bootstrap { b: usize }, KFold { k: usize }, Subsampling { b: usize, prob: f64 } }
```

Defaults: Bootstrap `b = 25`, KFold `k = 10`, Subsampling `prob = 0.5`. Bootstrap
draws integer in-bag multiplicities (`rmultinom`-equivalent); a fold's
out-of-bag observations have weight 0. Optional `strata` for stratified folds.
Folds parallelize over the `parallel` feature (rayon).

### 8.2 Risk

For each fold: fit on the in-bag sample to the maximum iteration, and at every
step record the **mean out-of-bag NLL** (total OOB risk normalized by OOB weight
sum). The chosen `mstop` minimizes the mean OOB risk across folds.

### 8.3 Grid (the cyclic/noncyclic asymmetry)

- **Non-cyclical → 1-D path**: evaluate `1..sum(mstop_max)` (one global counter,
  one risk curve). Cheap — the primary motivation for non-cyclical.
- **Cyclical → multi-dimensional grid**: a `make_grid`-style helper produces a
  matrix with one column per parameter (default `length_out = 10` points per
  parameter, log-spaced and rounded, `min = 1`), with the μ axis optionally
  densified (step 1) when μ's mstop dominates. Documented as the expensive mode.

Both modes are in v1.

---

## 9. Fitted model API (predict / coef)

```rust
let model = BoostLss::builder(GaussianLss::default())
    .on(Param::named("mu"),    PSpline::new("x1"))
    .on(Param::named("mu"),    Linear::new("x2"))
    .on(Param::named("sigma"), PSpline::new("x1"))
    .algorithm(Algorithm::NonCyclic)
    .mstop(Mstop::Scalar(500))
    .step_length(0.1)
    .build()?;

let fitted: Fitted<GaussianLss> = model.fit(&data)?;
```

- `fitted.predict(&new_data, param, Scale::Link | Scale::Response)` → `Array1`.
  Default scale is `Link` (matches `predict.mboostLSS`); `Response` applies the
  inverse link.
- `fitted.predict_all(&new_data)` → per-parameter predictions on the response
  scale.
- `fitted.coef(param)` → coefficients of the **selected** base-learners for that
  parameter.
- `fitted.risk_path()`, `fitted.selection_frequencies()`, `fitted.mstop()`.

`Fitted<F>` is generic over the family `F` (monomorphized).

---

## 10. Python bindings (`boostlss-py`)

Thin pyo3 wrapper that lowers onto the Rust builder; numpy arrays bridge to
ndarray via the `numpy` crate (zero-copy where possible).

```python
m = BoostLSS(family="gaussian", step_length=0.1, algorithm="noncyclic")
m.on("mu", pspline("x1")).on("mu", linear("x2")).on("sigma", pspline("x1"))
m.fit(X, y, columns=["x1", "x2"])
m.cvrisk(folds=10)
pred = m.predict(Xnew, param="mu", scale="response")
```

- Family selected by string (`"gaussian" | "student_t" | "gamma" | "nbinomial"`).
- `BoostlssError` maps to Python exceptions.
- Packaging via `maturin` + `pyproject.toml`, abi3 wheels. **v1 guarantees**
  `maturin develop` + a Python test suite; wheel-building CI is a stretch goal.

---

## 11. Testing strategy

All four methods are committed in v1.

1. **Finite-difference gradient checks** (primary correctness guard): for each
   `(family, parameter)`, assert the analytic `negative_gradient` matches the
   numerical derivative of `risk` (central differences) within tolerance, over
   randomized predictor values. Co-located unit tests per family module.
2. **Synthetic recovery**: simulate from each family with known covariate effects
   on multiple parameters; fit; assert recovered effects / risk within tolerance.
3. **R golden-value**: `tests/fixtures/generate.R` is run **once, offline**, and
   commits datasets plus gamboostlss `coef`/`risk`/`predict` outputs as fixture
   files. Rust tests read the committed fixtures and assert closeness within a
   documented tolerance (which absorbs algorithmic/FP divergence). **CI requires
   no R.** The regeneration procedure is documented in the fixtures README.
4. **Property-based** (`proptest`): risk is non-increasing across boosting
   iterations; predictions are finite; cvrisk-selected `mstop ∈ [1, max]`;
   stabilization never produces NaN/Inf.

Plus `criterion` benchmarks for the boosting hot path (cached-solve per
iteration) and for `cvrisk`.

Tests are co-located with the code they validate where structure allows
(per AGENTS.md); integration/golden/property tests live under `tests/`.

---

## 12. Error handling and quality gates

- One `BoostlssError` enum (`thiserror`): `DimensionMismatch`, `UnknownColumn`,
  `NonFinite`, `UnsupportedResponse`, `Singular`, `InvalidConfig`, `NotConverged`,
  … All fallible APIs return `Result<_, BoostlssError>`. No `unwrap`/`expect` on
  production paths; panics only in tests.
- Inputs validated at the boundary (fail fast): finite checks, per-family
  response-support checks, dimension checks, config validity (e.g. P-spline
  `df > differences` when `center = false`).
- CI gates (from `AGENTS.md`): `cargo fmt -- --check`,
  `cargo clippy --workspace --all-targets --all-features -- -D warnings`,
  `cargo test --workspace --all-features`. Feature branch only (never `main`);
  Conventional Commits; warnings-as-errors.
- ADRs (`docs/adr/`) for significant decisions: the generic-family/enum-learner
  architecture, the precompute-factorization choice, and the ndarray↔faer bridge.

---

## 13. Roadmap (explicitly deferred from v1)

- **Families**: Beta, Weibull, log-normal, ZIP/zero-inflated, GEV, and more.
- **Base-learners**: random effects, spatial, trees; cyclic P-splines;
  monotone/constrained effects.
- **Non-cyclical outer-loss** variant (base-learner selection by NLL).
- **Stability selection** (`stabsel`).
- **Formula DSL / macro** layer lowering onto the typed builder.
- **Sparse design matrices** for high-dimensional / many-knot settings.
- **Wheel-publishing CI** for the Python bindings.

---

## 14. References (primary sources)

- gamboostlss source: `R/mboostLSS.R`, `R/families.R`, `R/helpers.R`,
  `R/methods.R`, `R/cvrisk.R`, `R/cvrisk.nc_mboostLSS.R` —
  https://github.com/boost-r/gamboostlss
- mboost source: `R/control.R`, `R/crossvalidation.R`, base-learners `R/bl.R` —
  https://github.com/boost-R/mboost
- Mayr, Fenske, Hofner, Kneib, Schmid (2012), "GAMLSS for high-dimensional data —
  a flexible approach based on boosting", JRSS-C —
  https://doi.org/10.1111/j.1467-9876.2011.01033.x
- Thomas, Mayr, Bischl, Schmid, Fenske, Hofner (2018), "Gradient boosting for
  distributional regression — faster tuning and improved variable selection" —
  https://arxiv.org/abs/1611.10171
- Hofner, Mayr, Fenske, Schmid (2016), gamboostlss JSS paper —
  https://www.jstatsoft.org/article/view/v074i01
- CRAN manuals: gamboostlss
  https://cran.r-project.org/web/packages/gamboostLSS/gamboostLSS.pdf , mboost
  https://cran.r-project.org/web/packages/mboost/mboost.pdf , tutorial
  https://cran.r-project.org/web/packages/mboost/vignettes/mboost_tutorial.pdf
- Ruppert, Wand, Carroll (2003), _Semiparametric Regression_ (App. B.1.1) —
  df↔λ computation.
