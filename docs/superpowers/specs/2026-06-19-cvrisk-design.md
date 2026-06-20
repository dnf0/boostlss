# cvrisk (Cross-Validation Tuning) — Design Spec

- Status: Approved
- Date: 2026-06-19
- Scope: Implementation of `cvrisk` cross-validation tuning for the `boostlss` engine.

## 1. Architecture & API

We will isolate the cross-validation and tuning logic from the core boosting engine by introducing a dedicated `cv.rs` module.

### Core Structures
- **`Resampling` Enum**: Defines the strategy for creating folds.
  - `Bootstrap { b: usize }`: Random sampling with replacement.
  - `KFold { k: usize }`: Standard k-fold cross-validation.
  - `Subsampling { b: usize, prob: f64 }`: Random sampling without replacement.
  - Includes a helper method `generate_weights(&self, n: usize, rng: &mut impl Rng) -> Vec<Array1<f64>>` to generate the in-bag weights for each fold.

- **`CvRisk` Struct**: The main execution runner.
  ```rust
  pub struct CvRisk {
      model: BoostLss,
      resampling: Resampling,
  }

  impl CvRisk {
      pub fn new(model: BoostLss) -> Self;
      pub fn resampling(mut self, r: Resampling) -> Self;
      pub fn run(&self, dataset: &Dataset) -> Result<CvRiskResult, BoostlssError>;
  }
  ```

- **`CvRiskResult` Struct**: A rich diagnostic object returned upon completion.
  ```rust
  pub struct CvRiskResult {
      pub optimal_mstop: Mstop,
      pub grid: Vec<Mstop>,
      pub mean_risk: Vec<f64>,
      pub risk_path_per_fold: Vec<Vec<f64>>, // [fold][grid_point]
  }
  ```

## 2. Grid & Risk Evaluation Strategy

The evaluation path differs drastically depending on the fitting algorithm selected by the model.

### Non-Cyclical Mode
- The model operates with a single scalar `mstop`.
- The evaluation "grid" is simply the 1-D sequence `1..=mstop`.
- **Optimization**: For each fold, we do not refit the model from scratch for each point. Instead, we incrementally fit the model step-by-step up to `max(mstop)`. At each iteration step `m`, we pause and calculate the Out-Of-Bag (OOB) risk (where fold weights == 0). This means the cost of evaluating the entire path is roughly equal to fitting the model once.

### Cyclical Mode
- The model uses a parameter-specific `mstop` vector.
- We implement a `make_grid` helper that constructs a multi-dimensional matrix of parameter configurations (e.g., if there are 2 parameters, a default grid might evaluate 10 log-spaced points for each parameter, resulting in a 100-point grid).
- **Evaluation**: For each fold, and for *each specific point* in the grid, we must fit the model up to that configuration and evaluate the OOB risk. This is computationally expensive but necessary for cyclical evaluation.

## 3. Execution & Parallelism

- Fold evaluation is highly parallelizable since each fold is completely independent.
- We will add the `rayon` crate as an optional dependency behind a `parallel` feature flag in `Cargo.toml`.
- If `parallel` is enabled, `CvRisk::run` will use `rayon::prelude::par_iter` over the generated fold weights to execute the folds concurrently on available CPU cores.
- Once all folds complete their evaluation, the runner aggregates the risk arrays across the folds to calculate the `mean_risk` for each grid point and selects the `optimal_mstop` that minimizes this mean risk.
