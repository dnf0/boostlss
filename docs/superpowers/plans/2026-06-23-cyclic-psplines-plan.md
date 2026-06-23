# Cyclic P-Splines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing PSpline learner to support Cyclic P-splines for modeling periodic effects like time of day or seasonality.

**Architecture:** We will add a `cyclic` boolean flag to the `PSpline` learner. When true, the basis matrix evaluation will wrap the tail of the basis into the head to enforce periodic matching at boundaries. The penalty matrix will similarly wrap around to enforce smoothness across the boundaries.

**Tech Stack:** Rust (ndarray), Python (PyO3)

---

### Task 1: Update Rust Struct & Penalty Matrix

**Files:**
- Modify: `crates/boostlss/src/learner/pspline.rs`
- Modify: `crates/boostlss/src/learner/penalty.rs`

- [ ] **Step 1: Write the failing tests**

In `crates/boostlss/src/learner/penalty.rs`, add the failing test for `penalty_matrix` wrapping:

```rust
    #[test]
    fn test_cyclic_difference_matrix_d1() {
        // d=1 difference matrix for 3 columns:
        // [ -1   1   0 ]
        // [  0  -1   1 ]
        // [  1   0  -1 ]  <- The last row wraps around!
        let mut expected = Array2::zeros((3, 3));
        expected[[0, 0]] = -1.0; expected[[0, 1]] = 1.0;
        expected[[1, 1]] = -1.0; expected[[1, 2]] = 1.0;
        expected[[2, 0]] = 1.0; expected[[2, 2]] = -1.0;

        let mat = difference_matrix(3, 1, true);
        assert_eq!(mat, expected);
    }
```

In `crates/boostlss/src/learner/pspline.rs`, add the builder test:

```rust
    #[test]
    fn test_pspline_cyclic_builder() {
        let ps = PSpline::new("x1").cyclic(true);
        assert!(ps.is_cyclic);
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p boostlss -- penalty` and `cargo test -p boostlss -- pspline_cyclic`
Expected: FAIL (either `difference_matrix` doesn't take 3 arguments or tests fail).

- [ ] **Step 3: Write minimal implementation**

In `crates/boostlss/src/learner/pspline.rs`:
Add `pub is_cyclic: bool` to `pub struct PSpline`.
Update `PSpline::new`: `is_cyclic: false`.
Add builder method:
```rust
    pub fn cyclic(mut self, cyclic: bool) -> Self {
        self.is_cyclic = cyclic;
        self
    }
```

In `crates/boostlss/src/learner/penalty.rs`:
Update `difference_matrix` signature:
```rust
fn difference_matrix(n: usize, d: usize, cyclic: bool) -> Array2<f64> {
    if d == 0 {
        return Array2::eye(n);
    }

    let mut mat = Array2::zeros((n, n));
    if cyclic {
        for i in 0..n {
            mat[[i, i]] = -1.0;
            mat[[i, (i + 1) % n]] = 1.0;
        }
    } else {
        for i in 0..(n - 1) {
            mat[[i, i]] = -1.0;
            mat[[i, i + 1]] = 1.0;
        }
    }

    if d > 1 {
        let prev = difference_matrix(n, d - 1, cyclic);
        mat.dot(&prev)
    } else {
        mat
    }
}
```

Update `penalty_matrix` to take `cyclic: bool` and call `difference_matrix` with it:
```rust
pub fn penalty_matrix(n_cols: usize, differences: usize, cyclic: bool) -> Array2<f64> {
    let d_mat = difference_matrix(n_cols, differences, cyclic);
    d_mat.t().dot(&d_mat)
}
```

Update `penalty_matrix` calls everywhere it's used. In `pspline.rs`:
```rust
    pub fn penalty_matrix(&self, n_cols: usize) -> Array2<f64> {
        penalty_matrix(n_cols, self.differences, self.is_cyclic)
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p boostlss`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/boostlss/src/learner/
git commit -m "feat(pspline): add cyclic state and cyclic penalty matrix"
```

---

### Task 2: Implement Basis Wrapping for Cyclic P-splines

**Files:**
- Modify: `crates/boostlss/src/learner/pspline.rs`

- [ ] **Step 1: Write the failing test**

In `crates/boostlss/src/learner/pspline.rs`:
```rust
    #[test]
    fn test_pspline_build_design_cyclic() {
        let mut ps = PSpline::new("x1").with_knots(5).with_degree(3).cyclic(true);
        let x = array![0.0, 0.5, 1.0];
        let design = ps.build_design(&x).unwrap();

        // Standard dimension is knots + degree + 1 (5 + 3 + 1 = 9)
        // Cyclic dimension drops the rightmost degree columns: knots + 1 = 6
        assert_eq!(design.shape(), &[3, 6]);

        // Ensure values sum to 1 row-wise (partition of unity)
        for i in 0..3 {
            let sum: f64 = design.row(i).sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p boostlss -- test_pspline_build_design_cyclic`
Expected: FAIL (because dimension returned is `knots + degree + 1`, not `knots + 1`).

- [ ] **Step 3: Write minimal implementation**

In `crates/boostlss/src/learner/pspline.rs` in `build_design`, replace the final part of the function:

```rust
        let final_b = if self.is_cyclic {
            // For cyclic, wrap the rightmost degree columns into the first degree columns
            let out_cols = self.knots + 1;
            let mut cyclic_b = Array2::zeros((n, out_cols));

            for i in 0..n {
                for j in 0..p {
                    let wrapped_j = j % out_cols;
                    cyclic_b[[i, wrapped_j]] += b[[i, j]];
                }
            }
            cyclic_b
        } else {
            b
        };

        Ok(final_b)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p boostlss -- test_pspline_build_design_cyclic`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/boostlss/src/learner/pspline.rs
git commit -m "feat(pspline): implement cyclic basis matrix wrapping"
```

---

### Task 3: Python Bindings & Integration Test

**Files:**
- Modify: `crates/boostlss-py/src/model.rs`
- Modify/Create: `crates/boostlss-py/tests/test_pspline.py`

- [ ] **Step 1: Write the failing tests**

In `crates/boostlss-py/tests/test_pspline.py`:
```python
import numpy as np
from boostlss import PyBoostLss, PyFamily, PyPSplineLearner

def test_cyclic_pspline():
    X = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
    y = np.sin(X).flatten() + np.random.normal(0, 0.1, 100)

    # Model with cyclic spline
    model = PyBoostLss(PyFamily.GaussianLSS(), 10, 0.1)
    model.add_learner("x0", PyPSplineLearner(cyclic=True))
    model.fit(X, y)

    preds = model.predict(X)

    # Test boundary continuity: predict at 0 and 2*pi
    pred_0 = model.predict(np.array([[0.0]]))[0]
    pred_2pi = model.predict(np.array([[2 * np.pi]]))[0]

    assert np.abs(pred_0 - pred_2pi) < 0.2, "Cyclic predictions should match at boundaries"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd crates/boostlss-py && maturin develop && pytest tests/test_pspline.py`
Expected: FAIL (because `PyPSplineLearner` does not accept `cyclic` argument).

- [ ] **Step 3: Write minimal implementation**

In `crates/boostlss-py/src/model.rs`, update `PyPSplineLearner`:

```rust
#[pyclass]
#[derive(Clone)]
pub struct PyPSplineLearner {
    pub feature: Option<usize>,
    pub knots: usize,
    pub degree: usize,
    pub differences: usize,
    pub df: f64,
    pub cyclic: bool,
}

#[pymethods]
impl PyPSplineLearner {
    #[new]
    #[pyo3(signature = (feature=None, knots=20, degree=3, differences=2, df=4.0, cyclic=false))]
    pub fn new(
        feature: Option<usize>,
        knots: usize,
        degree: usize,
        differences: usize,
        df: f64,
        cyclic: bool,
    ) -> Self {
        Self {
            feature,
            knots,
            degree,
            differences,
            df,
            cyclic,
        }
    }
}
```

In `crates/boostlss-py/src/model.rs` inside the `impl PyBoostLss` `add_learner` method matching `PyLearner::PSpline(p)`:
```rust
                    let mut ps = PSpline::new(&col_name)
                        .with_knots(p.knots)
                        .with_degree(p.degree)
                        .with_differences(p.differences)
                        .with_df(p.df)
                        .cyclic(p.cyclic); // Add this!
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd crates/boostlss-py && maturin develop && pytest tests/test_pspline.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/boostlss-py/
git commit -m "feat(python): expose cyclic p-splines to python bindings"
```
