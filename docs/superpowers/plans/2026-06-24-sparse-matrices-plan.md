# Sparse Matrices Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement zero-copy sparse matrix support (CSR and CSC formats) across the full stack from Python to the Rust core engine.

**Architecture:** We will introduce a `DesignMatrix` enum in the Rust core to encapsulate dense and sparse representations, enabling memory-efficient storage. Base learners will extract single features using a fast `get_column` method. The Python bindings will extract `data`, `indices`, and `indptr` numpy arrays from `scipy.sparse` matrices and pass them directly to Rust to avoid costly copying and conversions.

**Tech Stack:** Rust (`ndarray`), Python (`scipy.sparse`, `numpy`), PyO3

---

### Task 1: Create the `DesignMatrix` Enum and `SparseMatrix` Struct

**Files:**
- Modify: `crates/boostlss/src/data.rs`

- [ ] **Step 1: Write the failing tests**

```rust
// At the bottom of crates/boostlss/src/data.rs (in tests module)
    #[test]
    fn test_design_matrix_dense() {
        let dense = Array2::from_elem((3, 2), 1.0);
        let dm = DesignMatrix::Dense(dense);
        let col = dm.get_column(1).unwrap();
        assert_eq!(col, Array1::from_elem(3, 1.0));
    }

    #[test]
    fn test_design_matrix_csc() {
        // [[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]]
        let data = array![1.0, 3.0, 2.0, 4.0];
        let indices = array![0, 2, 1, 2];
        let indptr = array![0, 2, 4];
        let sparse = SparseMatrix { data, indices, indptr, shape: (3, 2) };
        let dm = DesignMatrix::Csc(sparse);

        let col0 = dm.get_column(0).unwrap();
        assert_eq!(col0, array![1.0, 0.0, 3.0]);

        let col1 = dm.get_column(1).unwrap();
        assert_eq!(col1, array![0.0, 2.0, 4.0]);
    }

    #[test]
    fn test_design_matrix_csr() {
        // [[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]]
        let data = array![1.0, 2.0, 3.0, 4.0];
        let indices = array![0, 1, 0, 1];
        let indptr = array![0, 1, 2, 4];
        let sparse = SparseMatrix { data, indices, indptr, shape: (3, 2) };
        let dm = DesignMatrix::Csr(sparse);

        let col0 = dm.get_column(0).unwrap();
        assert_eq!(col0, array![1.0, 0.0, 3.0]);

        let col1 = dm.get_column(1).unwrap();
        assert_eq!(col1, array![0.0, 2.0, 4.0]);
    }
```

- [ ] **Step 2: Run test to verify it fails**
Run: `cargo test -p boostlss -- data::tests::test_design_matrix`
Expected: Compilation failure because types are missing.

- [ ] **Step 3: Implement minimal code**

```rust
// In crates/boostlss/src/data.rs (top of file)
use crate::error::BoostlssError;

#[derive(Clone, Debug, PartialEq)]
pub struct SparseMatrix {
    pub data: Array1<f64>,
    pub indices: Array1<usize>,
    pub indptr: Array1<usize>,
    pub shape: (usize, usize),
}

#[derive(Clone, Debug, PartialEq)]
pub enum DesignMatrix {
    Dense(Array2<f64>),
    Csr(SparseMatrix),
    Csc(SparseMatrix),
}

impl DesignMatrix {
    pub fn get_column(&self, col_idx: usize) -> Result<Array1<f64>, BoostlssError> {
        match self {
            Self::Dense(mat) => {
                if col_idx >= mat.ncols() {
                    return Err(BoostlssError::DataError("Column index out of bounds".to_string()));
                }
                Ok(mat.column(col_idx).to_owned())
            }
            Self::Csc(sparse) => {
                if col_idx >= sparse.shape.1 {
                    return Err(BoostlssError::DataError("Column index out of bounds".to_string()));
                }
                let mut col = Array1::zeros(sparse.shape.0);
                let start = sparse.indptr[col_idx];
                let end = sparse.indptr[col_idx + 1];
                for i in start..end {
                    let row_idx = sparse.indices[i];
                    col[row_idx] = sparse.data[i];
                }
                Ok(col)
            }
            Self::Csr(sparse) => {
                if col_idx >= sparse.shape.1 {
                    return Err(BoostlssError::DataError("Column index out of bounds".to_string()));
                }
                let mut col = Array1::zeros(sparse.shape.0);
                for row_idx in 0..sparse.shape.0 {
                    let start = sparse.indptr[row_idx];
                    let end = sparse.indptr[row_idx + 1];
                    for i in start..end {
                        if sparse.indices[i] == col_idx {
                            col[row_idx] = sparse.data[i];
                            break;
                        }
                    }
                }
                Ok(col)
            }
        }
    }

    pub fn nrows(&self) -> usize {
        match self {
            Self::Dense(mat) => mat.nrows(),
            Self::Csr(sparse) => sparse.shape.0,
            Self::Csc(sparse) => sparse.shape.0,
        }
    }

    pub fn ncols(&self) -> usize {
        match self {
            Self::Dense(mat) => mat.ncols(),
            Self::Csr(sparse) => sparse.shape.1,
            Self::Csc(sparse) => sparse.shape.1,
        }
    }
}
```

- [ ] **Step 4: Run test to verify it passes**
Run: `cargo test -p boostlss -- data::tests`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/boostlss/src/data.rs
git commit -m "feat: Add DesignMatrix enum and SparseMatrix structures"
```

---

### Task 2: Integrate `DesignMatrix` into `Dataset`

**Files:**
- Modify: `crates/boostlss/src/data.rs`

- [ ] **Step 1: Modify `Dataset` struct and initialization**

Change the `Dataset` struct to use `DesignMatrix`:
```rust
// In crates/boostlss/src/data.rs
#[derive(Clone, Debug)]
pub struct Dataset {
    design: DesignMatrix,
    response: Array1<f64>,
    weights: Option<Array1<f64>>,
}
```

Add constructors for sparse matrices and update the old constructor:
```rust
// In crates/boostlss/src/data.rs
impl Dataset {
    pub fn new(
        design: Array2<f64>,
        response: Array1<f64>,
        weights: Option<Array1<f64>>,
    ) -> Result<Self, BoostlssError> {
        let n = design.nrows();
        if response.len() != n {
            return Err(BoostlssError::DataError(format!(
                "Design has {} rows, but response has length {}",
                n,
                response.len()
            )));
        }
        if let Some(w) = &weights {
            if w.len() != n {
                return Err(BoostlssError::DataError(format!(
                    "Design has {} rows, but weights have length {}",
                    n,
                    w.len()
                )));
            }
        }
        Ok(Self {
            design: DesignMatrix::Dense(design),
            response,
            weights,
        })
    }

    pub fn new_csr(
        sparse: SparseMatrix,
        response: Array1<f64>,
        weights: Option<Array1<f64>>,
    ) -> Result<Self, BoostlssError> {
        let n = sparse.shape.0;
        if response.len() != n {
            return Err(BoostlssError::DataError("Row mismatch".into()));
        }
        Ok(Self {
            design: DesignMatrix::Csr(sparse),
            response,
            weights,
        })
    }

    pub fn new_csc(
        sparse: SparseMatrix,
        response: Array1<f64>,
        weights: Option<Array1<f64>>,
    ) -> Result<Self, BoostlssError> {
        let n = sparse.shape.0;
        if response.len() != n {
            return Err(BoostlssError::DataError("Row mismatch".into()));
        }
        Ok(Self {
            design: DesignMatrix::Csc(sparse),
            response,
            weights,
        })
    }
}
```

- [ ] **Step 2: Fix `Dataset` methods**

Update the methods that previously relied on `Array2`:
```rust
// In crates/boostlss/src/data.rs
    pub fn design(&self) -> &DesignMatrix {
        &self.design
    }

    pub fn n_obs(&self) -> usize {
        self.design.nrows()
    }

    pub fn n_features(&self) -> usize {
        self.design.ncols()
    }
```

- [ ] **Step 3: Fix internal `data.rs` tests**
Fix any broken tests in `data.rs` that passed `.design()` to expectations of `Array2` by pattern matching or unwrapping the `Dense` variant in test setups.

- [ ] **Step 4: Check compilation locally**
Run: `cargo check -p boostlss`
Expected: Fails due to `Dataset` changes cascading to learners. That's expected for this step.

- [ ] **Step 5: Commit**

```bash
git add crates/boostlss/src/data.rs
git commit -m "refactor: Update Dataset to use DesignMatrix"
```

---

### Task 3: Refactor Base-Learners for `DesignMatrix`

**Files:**
- Modify: `crates/boostlss/src/learner/linear.rs`
- Modify: `crates/boostlss/src/learner/pspline.rs`
- Modify: `crates/boostlss/src/learner/constrained_pspline.rs`
- Modify: `crates/boostlss/src/learner/random_effects.rs`
- Modify: `crates/boostlss/src/learner/stump.rs`
- Modify: `crates/boostlss/src/learner/tree.rs`

- [ ] **Step 1: Replace `.column(idx)` with `get_column(idx)`**

In all learner implementations of `build_design`, replace the manual extraction from `data.design()`:

For `linear.rs`:
```rust
        let mut design = Array2::zeros((n_obs, n_cols));
        let col = data.design().get_column(self.feature_idx)?;
        let mut offset = 0;
        if self.intercept {
            design.column_mut(0).fill(1.0);
            offset = 1;
        }
        design.column_mut(offset).assign(&col);
```

For `pspline.rs`, `constrained_pspline.rs`, `stump.rs`:
```rust
        let col = data.design().get_column(self.feature_idx)?;
        // use col instead of data.design().column(self.feature_idx)
```

For `bspatial.rs`:
```rust
        let col1 = data.design().get_column(self.feature1_idx)?;
        let col2 = data.design().get_column(self.feature2_idx)?;
```

For `tree.rs`:
Instead of expecting the entire dense matrix, we might need a dense matrix locally to run the tree algorithm. If tree learner needs the whole matrix, we can extract all columns.
```rust
        // Inside tree.rs build_design or fit
        let n_cols = data.n_features();
        let mut dense_mat = Array2::zeros((data.n_obs(), n_cols));
        for i in 0..n_cols {
            let col = data.design().get_column(i)?;
            dense_mat.column_mut(i).assign(&col);
        }
```

- [ ] **Step 2: Run tests to verify logic holds**
Run: `cargo test -p boostlss`
Ensure you fix any compilation issues that arise in family testing or internal learner testing.

- [ ] **Step 3: Commit**

```bash
git add crates/boostlss/src/learner/
git commit -m "refactor: Update all learners to use DesignMatrix::get_column"
```

---

### Task 4: Python Bindings for Sparse Data

**Files:**
- Modify: `crates/boostlss-py/src/model.rs`
- Modify: `crates/boostlss-py/src/lib.rs`

- [ ] **Step 1: Add new `fit_sparse` method to `BoostLssModel`**

```rust
// In crates/boostlss-py/src/model.rs
use numpy::PyReadonlyArray1;
use boostlss::data::SparseMatrix;

    // Inside impl BoostLssModel {
    #[pyo3(signature = (data, indices, indptr, shape, y, format))]
    fn fit_sparse(
        &mut self,
        data: PyReadonlyArray1<f64>,
        indices: PyReadonlyArray1<usize>,
        indptr: PyReadonlyArray1<usize>,
        shape: (usize, usize),
        y: PyReadonlyArray1<f64>,
        format: &str,
    ) -> PyResult<()> {
        let data_arr = data.as_array().to_owned();
        let indices_arr = indices.as_array().to_owned();
        let indptr_arr = indptr.as_array().to_owned();
        let y_vec = y.as_array().to_owned();

        let sparse = SparseMatrix {
            data: data_arr,
            indices: indices_arr,
            indptr: indptr_arr,
            shape,
        };

        let dataset = match format.to_lowercase().as_str() {
            "csr" => Dataset::new_csr(sparse, y_vec, None),
            "csc" => Dataset::new_csc(sparse, y_vec, None),
            _ => return Err(pyo3::exceptions::PyValueError::new_err("Format must be 'csr' or 'csc'")),
        }.map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        self.fit_internal(dataset)?;
        Ok(())
    }
```
*Note: You will need to extract the core `fit` logic out of the current `fit(x, y)` method into a private `fit_internal(&mut self, dataset: Dataset)` method so both `fit` and `fit_sparse` can share it.*

- [ ] **Step 2: Add new `predict_sparse` method**

```rust
// In crates/boostlss-py/src/model.rs
    #[pyo3(signature = (data, indices, indptr, shape, param, format))]
    fn predict_sparse<'py>(
        &mut self,
        py: Python<'py>,
        data: PyReadonlyArray1<f64>,
        indices: PyReadonlyArray1<usize>,
        indptr: PyReadonlyArray1<usize>,
        shape: (usize, usize),
        param: &str,
        format: &str,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let data_arr = data.as_array().to_owned();
        let indices_arr = indices.as_array().to_owned();
        let indptr_arr = indptr.as_array().to_owned();
        let y_dummy = ndarray::Array1::zeros(shape.0);

        let sparse = SparseMatrix {
            data: data_arr,
            indices: indices_arr,
            indptr: indptr_arr,
            shape,
        };

        let dataset = match format.to_lowercase().as_str() {
            "csr" => Dataset::new_csr(sparse, y_dummy, None),
            "csc" => Dataset::new_csc(sparse, y_dummy, None),
            _ => return Err(pyo3::exceptions::PyValueError::new_err("Format must be 'csr' or 'csc'")),
        }.map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        self.predict_internal(py, dataset, param)
    }
```
*Note: Similarly, extract core prediction logic to `predict_internal`.*

- [ ] **Step 3: Update `BoostLssModel` tracking**
The current code stores `self.train_data: Option<(Array2<f64>, Array1<f64>)>` to support `stabsel`. You will need to change this to store a `Dataset` instead, so that sparse data can be correctly tracked during stabsel reconstruction.

- [ ] **Step 4: Check Python compilation**
Run: `uv run maturin develop`

- [ ] **Step 5: Commit**

```bash
git add crates/boostlss-py/src/model.rs
git commit -m "feat: Add sparse bindings to PyO3 model"
```

---

### Task 5: Python Wrapper Integration and Tests

**Files:**
- Create: `tests/test_sparse.py`

- [ ] **Step 1: Write integration tests comparing Dense vs Sparse**

```python
# tests/test_sparse.py
import pytest
import numpy as np
from scipy import sparse
from boostlss_py import BoostLssModel, PyFamily, PyLinearLearner

def test_sparse_csr_csc():
    np.random.seed(42)
    # Generate some sparse data
    dense_X = np.random.binomial(1, 0.1, (100, 5)).astype(np.float64)
    dense_y = 2.0 * dense_X[:, 0] - 1.5 * dense_X[:, 1] + np.random.normal(0, 0.1, 100)

    # 1. Fit Dense Model
    model_dense = BoostLssModel(PyFamily("GaussianLSS"), mstop=50, step_length=0.1)
    for i in range(5):
        model_dense.add_learner("mu", PyLinearLearner(i, True))
        model_dense.add_learner("sigma", PyLinearLearner(i, True))
    model_dense.fit(dense_X, dense_y)
    dense_preds = model_dense.predict(dense_X, "mu")

    # 2. Fit CSR Model
    X_csr = sparse.csr_matrix(dense_X)
    model_csr = BoostLssModel(PyFamily("GaussianLSS"), mstop=50, step_length=0.1)
    for i in range(5):
        model_csr.add_learner("mu", PyLinearLearner(i, True))
        model_csr.add_learner("sigma", PyLinearLearner(i, True))

    # Send CSR matrices
    model_csr.fit_sparse(X_csr.data, X_csr.indices.astype(np.uint64), X_csr.indptr.astype(np.uint64), X_csr.shape, dense_y, "csr")
    csr_preds = model_csr.predict_sparse(X_csr.data, X_csr.indices.astype(np.uint64), X_csr.indptr.astype(np.uint64), X_csr.shape, "mu", "csr")

    # 3. Fit CSC Model
    X_csc = sparse.csc_matrix(dense_X)
    model_csc = BoostLssModel(PyFamily("GaussianLSS"), mstop=50, step_length=0.1)
    for i in range(5):
        model_csc.add_learner("mu", PyLinearLearner(i, True))
        model_csc.add_learner("sigma", PyLinearLearner(i, True))

    model_csc.fit_sparse(X_csc.data, X_csc.indices.astype(np.uint64), X_csc.indptr.astype(np.uint64), X_csc.shape, dense_y, "csc")
    csc_preds = model_csc.predict_sparse(X_csc.data, X_csc.indices.astype(np.uint64), X_csc.indptr.astype(np.uint64), X_csc.shape, "mu", "csc")

    # All predictions should be perfectly identical
    assert np.allclose(dense_preds, csr_preds)
    assert np.allclose(dense_preds, csc_preds)
```

- [ ] **Step 2: Verify tests pass**
Run: `uv run pytest tests/test_sparse.py -v`

- [ ] **Step 3: Commit**

```bash
git add tests/test_sparse.py
git commit -m "test: Add comprehensive dense vs sparse equality tests"
```
