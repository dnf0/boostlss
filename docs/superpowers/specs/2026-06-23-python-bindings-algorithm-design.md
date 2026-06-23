# Python Bindings Algorithm Parameter Spec

## Context
The core Rust `BoostLss` engine supports both `Algorithm::Cyclic` and `Algorithm::NonCyclic`. However, the Python bindings currently hardcode `fit_cyclical` in `crates/boostlss-py/src/model.rs`. We need to expose the algorithm selection to Python users.

## Approach
We will keep the existing `add_learner` pattern as it perfectly aligns with our Rust builder API. We will add an `algorithm` parameter to the `BoostLssModel` constructor in Python.

## Design

### 1. Update BoostLssModel struct
Add an `algorithm: String` field to `BoostLssModel` in `crates/boostlss-py/src/model.rs`.

### 2. Update Python Constructor
Modify the `#[new]` method for `BoostLssModel` to accept `algorithm` as a kwarg with a default of `"cyclic"`.

```rust
#[new]
#[pyo3(signature = (family, mstop=100, step_length=0.1, algorithm="cyclic"))]
fn new(family: PyFamily, mstop: usize, step_length: f64, algorithm: &str) -> Self {
    // ... validate algorithm is "cyclic" or "noncyclic" ...
}
```

### 3. Update `fit()` and `cvrisk()` Methods
In both methods, instead of hardcoding `fit_cyclical`, we will check the `algorithm` field:
- If `"cyclic"`, call `fit_cyclical(model, &dataset)`.
- If `"noncyclic"`, configure the model with `.algorithm(boostlss::engine::Algorithm::NonCyclic)` and call `fit_noncyclic(model, &dataset)`.

### 4. Tests
Update Python tests in `crates/boostlss-py/tests/test_basic.py` to assert that models can be fitted with `algorithm="noncyclic"`.
