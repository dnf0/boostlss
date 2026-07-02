# Sparse Design Matrices Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable sparse design matrices across the boosting hot-path using `sprs` and `sprs-ldl` to prevent OOM errors on high-dimensional data (e.g., `RandomEffects`).

**Architecture:** We will modify `BaseLearner`'s `build_design` to return the existing `DesignMatrix` enum instead of a dense `Array2<f64>`. We will upgrade `RandomEffects` to return a `DesignMatrix::Csc` and refactor `LearnerFit::initialize` to use `sprs-ldl` for LDLt Cholesky factorizations whenever the design matrix is sparse.

**Tech Stack:** Rust, `ndarray`, `faer` (dense solver), `sprs` (sparse matrices), `sprs-ldl` (sparse LDLt solver)

---

### Task 1: Add Dependencies

**Files:**
- Modify: `crates/boostlss/Cargo.toml`

- [ ] **Step 1: Add sprs dependencies**

```toml
# In crates/boostlss/Cargo.toml, under [dependencies]
sprs = "0.11.4"
sprs-ldl = "0.10.0"
```

- [ ] **Step 2: Run cargo check**

Run: `uv run cargo check -p boostlss`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add crates/boostlss/Cargo.toml crates/boostlss/Cargo.lock
git commit -m "build: add sprs and sprs-ldl dependencies"
```

### Task 2: Extend DesignMatrix API

**Files:**
- Modify: `crates/boostlss/src/data.rs`

- [ ] **Step 1: Add to_csc methods**

Add the following methods to convert our internal structs into `sprs::CsMat<f64>` representations.

```rust
// Inside `crates/boostlss/src/data.rs`
impl SparseMatrix {
    pub fn to_csc(&self) -> Result<sprs::CsMat<f64>, BoostlssError> {
        let expected_csr = self.shape.0 + 1;
        let expected_csc = self.shape.1 + 1;
        if self.indptr.len() == expected_csc {
            Ok(sprs::CsMat::new_csc(self.shape, self.indptr.to_vec(), self.indices.to_vec(), self.data.to_vec()))
        } else if self.indptr.len() == expected_csr {
            let csr = sprs::CsMat::new(self.shape, self.indptr.to_vec(), self.indices.to_vec(), self.data.to_vec());
            Ok(csr.to_csc())
        } else {
            Err(BoostlssError::DataError("Invalid sparse shape".into()))
        }
    }
}

impl DesignMatrix {
    pub fn to_csc(&self) -> Result<sprs::CsMat<f64>, BoostlssError> {
        match self {
            Self::Csc(sparse) | Self::Csr(sparse) => sparse.to_csc(),
            Self::Dense(_) => Err(BoostlssError::DataError("Cannot convert dense to csc trivially".into())),
        }
    }
}
```

- [ ] **Step 2: Verify it builds**

Run: `uv run cargo check -p boostlss`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add crates/boostlss/src/data.rs
git commit -m "feat(data): add sprs export methods for DesignMatrix"
```

### Task 3: Refactor BaseLearner trait

**Files:**
- Modify: `crates/boostlss/src/learner/mod.rs`
- Modify: `crates/boostlss/src/learner/linear.rs`
- Modify: `crates/boostlss/src/learner/pspline.rs`
- Modify: `crates/boostlss/src/learner/constrained_pspline.rs`
- Modify: `crates/boostlss/src/learner/bspatial.rs`
- Modify: `crates/boostlss/src/learner/random_effects.rs`

- [ ] **Step 1: Change trait signatures**

In `crates/boostlss/src/learner/mod.rs`, change `BaseLearner::build_design` to return `Result<crate::data::DesignMatrix, BoostlssError>`.

Change `BaseLearner::penalty_matrix` to return `crate::data::DesignMatrix`.

- [ ] **Step 2: Update all learners to wrap existing output in Dense**

In `linear.rs`, `pspline.rs`, `constrained_pspline.rs`, `bspatial.rs`, `random_effects.rs`, change their `build_design` methods to return `DesignMatrix::Dense(...)` and their `penalty_matrix` methods to return `DesignMatrix::Dense(...)`.

*(We are doing this intermediate step just to get things compiling before doing the big RandomEffects and LearnerFit rewrites).*

- [ ] **Step 3: Update `LearnerFit::initialize` to extract Dense**

In `crates/boostlss/src/learner/mod.rs` `initialize`, change the assignment to unwrap dense:

```rust
        let (design, penalty) = match self {
            Self::BivariatePSpline(bp) => {
                let design = bp.build_design(data)?;
                let penalty = bp.penalty_matrix(bp.knots + bp.degree + 1, bp.knots + bp.degree + 1);
                (design, penalty)
            }
            _ => {
                let d = self.build_design(data)?;
                let p = self.penalty_matrix(d.ncols());
                (d, p)
            }
        };

        // For now, extract dense (to be rewritten in Task 5)
        let design_dense = match design {
            crate::data::DesignMatrix::Dense(mat) => mat,
            _ => panic!("Expected dense matrix for initialization in this step"),
        };
        let penalty_dense = match penalty {
            crate::data::DesignMatrix::Dense(mat) => mat,
            _ => panic!("Expected dense penalty"),
        };

        // Replace `design` with `design_dense` and `penalty` with `penalty_dense` throughout the rest of `initialize` where they are used.
```

- [ ] **Step 4: Verify it builds**

Run: `uv run cargo check -p boostlss`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/boostlss/src/learner/
git commit -m "refactor: update BaseLearner to return DesignMatrix"
```

### Task 4: Upgrade RandomEffects to Sparse

**Files:**
- Modify: `crates/boostlss/src/learner/random_effects.rs`

- [ ] **Step 1: Rewrite RandomEffects to output `DesignMatrix::Csc`**

```rust
// In crates/boostlss/src/learner/random_effects.rs
use crate::data::{DesignMatrix, SparseMatrix};
use ndarray::Array1;

impl RandomEffects {
    pub fn build_design(&self, data: &crate::data::Dataset) -> Result<DesignMatrix, BoostlssError> {
        let col = data.design().get_column(self.feature_idx)?;
        let n_obs = col.len();
        if n_obs == 0 {
            return Ok(DesignMatrix::Dense(ndarray::Array2::zeros((0, 0))));
        }

        let mut max_idx = 0;
        for &val in col.iter() {
            let idx = val as usize;
            if idx > max_idx {
                max_idx = idx;
            }
        }
        let n_cols = max_idx + 1;

        // Construct CSC matrix directly.
        let mut row_counts = vec![0; n_cols];
        for &val in col.iter() {
            row_counts[val as usize] += 1;
        }

        let mut indptr = Vec::with_capacity(n_cols + 1);
        indptr.push(0);
        let mut current = 0;
        for &count in row_counts.iter() {
            current += count;
            indptr.push(current);
        }

        let mut indices = vec![0; n_obs];
        let data_vals = vec![1.0; n_obs];

        let mut offsets = indptr.clone();
        for (row_idx, &val) in col.iter().enumerate() {
            let col_idx = val as usize;
            let offset = offsets[col_idx];
            indices[offset] = row_idx;
            offsets[col_idx] += 1;
        }

        let sparse = SparseMatrix::new(
            Array1::from_vec(data_vals),
            Array1::from_vec(indices),
            Array1::from_vec(indptr),
            (n_obs, n_cols),
        )?;

        Ok(DesignMatrix::Csc(sparse))
    }

    pub fn penalty_matrix(&self, n_cols: usize) -> DesignMatrix {
        // Identity matrix as CSC
        let indptr: Vec<usize> = (0..=n_cols).collect();
        let indices: Vec<usize> = (0..n_cols).collect();
        let data_vals: Vec<f64> = vec![1.0; n_cols];

        let sparse = SparseMatrix::new(
            Array1::from_vec(data_vals),
            Array1::from_vec(indices),
            Array1::from_vec(indptr),
            (n_cols, n_cols),
        ).unwrap();

        DesignMatrix::Csc(sparse)
    }
}
```
*Note: Also remove the `MAX_SAFE_COLS` and OOM checks since OOM is no longer a concern!*

- [ ] **Step 2: Update tests in random_effects.rs to handle DesignMatrix**

Change test assertions to extract CSC and verify structure.

- [ ] **Step 3: Run tests**

Run: `uv run cargo test learner::random_effects`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add crates/boostlss/src/learner/random_effects.rs
git commit -m "feat: upgrade RandomEffects to generate sparse design matrices"
```

### Task 5: Upgrade LearnerFit with Sparse Solver

**Files:**
- Modify: `crates/boostlss/src/learner/mod.rs`

- [ ] **Step 1: Create `SolverState` enum**

Replace `LinearFitState` struct with:
```rust
#[derive(Debug, Clone)]
pub enum SolverState {
    Dense {
        coef: Array1<f64>,
        llt: faer::linalg::solvers::Llt<f64>,
        design: ndarray::Array2<f64>,
    },
    Sparse {
        coef: Array1<f64>,
        ldl: std::sync::Arc<sprs_ldl::LdlNumeric<f64, usize>>,
        design: sprs::CsMat<f64>,
    }
}
```
*(Wrapped `LdlNumeric` in Arc to satisfy Clone).*
Update `LearnerFit` to use `Linear(SolverState)` instead of `Linear(LinearFitState)`.

- [ ] **Step 2: Refactor `initialize`**

Inside `LearnerFit::initialize`, handle both dense and sparse paths:
```rust
        if let (Ok(design_csc), Ok(penalty_csc)) = (design.to_csc(), penalty.to_csc()) {
            // SPARSE PATH
            let p = design_csc.cols();

            let design_csr = design_csc.transpose_view().to_csr();
            let mut xtw = design_csr.clone();

            // Weighting
            if let Some(w) = data.weights() {
                for (row_idx, mut row_vec) in xtw.outer_iterator_mut().enumerate() {
                    let w_val = w[row_idx];
                    for val in row_vec.iter_mut() {
                        *val *= w_val;
                    }
                }
            }

            // X^T W X
            let mut xtwx = &xtw * &design_csc;

            // + lambda K
            // Construct a from xtwx and penalty. Ensure they have the same sparsity pattern or are added properly.
            // Using sprs addition:
            let scaled_penalty = penalty_csc.map(|x| x * lambda);
            let a = (&xtwx + &scaled_penalty);

            let ldl = sprs_ldl::LdlNumeric::new(a.view()).unwrap();

            return Ok(LearnerFit::Linear(SolverState::Sparse {
                coef: Array1::zeros(p),
                ldl: std::sync::Arc::new(ldl),
                design: design_csc,
            }));
        } else {
            // DENSE PATH (restore the original dense logic from Task 3)
            // ...
        }
```

- [ ] **Step 3: Refactor `fit_update` and `predict_update`**

In `fit_update`:
```rust
            Self::Linear(SolverState::Sparse { design, ldl, .. }) => {
                let p = design.cols();
                let mut xtu = vec![0.0; p];
                for (col_idx, vec) in design.outer_iterator().enumerate() {
                    for (row_idx, &val) in vec.iter() {
                        xtu[col_idx] += val * u[row_idx];
                    }
                }

                let beta = ldl.solve(&xtu);
                LearnerUpdate::Linear(Array1::from_vec(beta))
            }
```

In `predict_update`:
```rust
            (Self::Linear(SolverState::Sparse { design, .. }), LearnerUpdate::Linear(coef)) => {
                let mut pred = vec![0.0; design.rows()];
                for (col_idx, vec) in design.outer_iterator().enumerate() {
                    let beta_j = coef[col_idx];
                    for (row_idx, &val) in vec.iter() {
                        pred[row_idx] += val * beta_j;
                    }
                }
                Array1::from_vec(pred)
            }
```

- [ ] **Step 4: Run Tests**

Run: `uv run pytest crates/boostlss-py/tests/test_model.py` and `uv run cargo test`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/boostlss/src/learner/
git commit -m "feat: implement sparse solver in LearnerFit"
```
