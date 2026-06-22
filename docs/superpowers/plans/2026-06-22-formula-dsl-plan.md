# Formula DSL / Builder API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve the ergonomics of the `BoostLss` builder API by allowing users to elegantly chain multiple base-learners to a single distribution parameter without repetitive `.on()` calls or explicit `BaseLearner::Variant(...)` wrapping.

**Architecture:** We will introduce a `ParamBuilder` struct that accumulates base-learners. We will implement `From` traits for all learners into `BaseLearner`. We will change `BoostLss::on` to accept a closure taking and returning a `ParamBuilder`. We will update the Python bindings and existing tests to use the new API.

**Tech Stack:** Rust, PyO3

---

### Task 1: Add `From` Implementations for Base Learners

**Files:**
- Modify: `crates/boostlss/src/learner/mod.rs`

- [ ] **Step 1: Write the failing test**

Modify `crates/boostlss/src/learner/mod.rs` to add a test module at the bottom.

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    // ... existing test ...

    #[test]
    fn test_from_impls() {
        let l = Linear::new("x");
        let bl: BaseLearner = l.into();
        assert!(matches!(bl, BaseLearner::Linear(_)));

        let p = PSpline::new("x");
        let bl: BaseLearner = p.into();
        assert!(matches!(bl, BaseLearner::PSpline(_)));

        let s = Stump::new("x");
        let bl: BaseLearner = s.into();
        assert!(matches!(bl, BaseLearner::Stump(_)));

        let t = Tree::new(vec!["x".to_string()]);
        let bl: BaseLearner = t.into();
        assert!(matches!(bl, BaseLearner::Tree(_)));
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p boostlss -- test_from_impls`
Expected: FAIL due to missing `From` implementations.

- [ ] **Step 3: Write minimal implementation**

Modify `crates/boostlss/src/learner/mod.rs` below the `BaseLearner` enum definition to add the `From` implementations:

```rust
impl From<Linear> for BaseLearner {
    fn from(l: Linear) -> Self {
        Self::Linear(l)
    }
}

impl From<PSpline> for BaseLearner {
    fn from(p: PSpline) -> Self {
        Self::PSpline(p)
    }
}

impl From<Stump> for BaseLearner {
    fn from(s: Stump) -> Self {
        Self::Stump(s)
    }
}

impl From<Tree> for BaseLearner {
    fn from(t: Tree) -> Self {
        Self::Tree(t)
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p boostlss -- test_from_impls`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/boostlss/src/learner/mod.rs
git commit -m "feat: add From impls for base learners to BaseLearner"
```

---

### Task 2: Implement `ParamBuilder` and Update `BoostLss::on`

**Files:**
- Modify: `crates/boostlss/src/model.rs`
- Modify: `crates/boostlss/src/cv.rs`
- Modify: `crates/boostlss/src/engine/cyclical.rs`

- [ ] **Step 1: Write the failing tests and update existing ones**

Modify `crates/boostlss/src/model.rs` to replace the `on` test and add a new test for `ParamBuilder`.

```rust
    #[test]
    fn test_boostlss_on_valid_param() {
        let model = BoostLss::new(GaussianLss::new())
            .on("mu", |p| p.add(Linear::new("x")))
            .unwrap();

        assert_eq!(model.learners().len(), 1);
        assert_eq!(model.learners()[0].0, 0);
    }

    #[test]
    fn test_boostlss_on_invalid_param() {
        let result = BoostLss::new(GaussianLss::new())
            .on("invalid_param", |p| p.add(Linear::new("x")));

        assert!(matches!(result, Err(BoostlssError::InvalidConfig(_))));
    }

    #[test]
    fn test_boostlss_on_multiple_learners() {
        let model = BoostLss::new(GaussianLss::new())
            .on("mu", |p| p.add(Linear::new("x")).add(crate::learner::PSpline::new("y")))
            .unwrap();

        assert_eq!(model.learners().len(), 2);
        assert_eq!(model.learners()[0].0, 0);
        assert_eq!(model.learners()[1].0, 0);
    }
```

Update `crates/boostlss/src/cv.rs` (lines ~292-295):
```rust
        let model = BoostLss::new(GaussianLss::new())
            .on("mu", |p| p.add(Linear::new("x").intercept(true)))
            .unwrap()
            .on("sigma", |p| p.add(Linear::new("x").intercept(true)))
            .unwrap()
            .mstop(Mstop::Scalar(10));
```

Update `crates/boostlss/src/engine/cyclical.rs` (lines ~203-206):
```rust
        let mut model = BoostLss::new(GaussianLss::new())
            .on("mu", |p| p.add(Linear::new("x").intercept(true)))
            .unwrap()
            .on("sigma", |p| p.add(Linear::new("x").intercept(true)))
            .unwrap()
            .algorithm(Algorithm::Cyclic)
            .mstop(Mstop::PerParam(vec![2, 2]));
```

- [ ] **Step 2: Run test to verify compilation fails**

Run: `cargo check`
Expected: FAIL due to API changes (BoostLss::on signature change).

- [ ] **Step 3: Write minimal implementation**

Modify `crates/boostlss/src/model.rs`. Add `ParamBuilder` struct before `BoostLss` methods:

```rust
pub struct ParamBuilder {
    pub(crate) learners: Vec<BaseLearner>,
}

impl ParamBuilder {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self { learners: Vec::new() }
    }

    pub fn add<L: Into<BaseLearner>>(mut self, learner: L) -> Self {
        self.learners.push(learner.into());
        self
    }
}
```

Modify the `BoostLss::on` method in `crates/boostlss/src/model.rs`:

```rust
    /// Registers base learners for a specific parameter using a builder closure.
    ///
    /// Example:
    /// ```
    /// # use boostlss::model::BoostLss;
    /// # use boostlss::family::GaussianLss;
    /// # use boostlss::learner::{Linear, PSpline};
    /// let model = BoostLss::new(GaussianLss::new())
    ///     .on("mu", |p| p.add(Linear::new("x1")).add(PSpline::new("x2")));
    /// ```
    pub fn on(
        mut self,
        param_name: &str,
        build_fn: impl FnOnce(ParamBuilder) -> ParamBuilder,
    ) -> Result<Self, BoostlssError> {
        let params = self.family.params();
        let k = params
            .iter()
            .position(|p| p.name == param_name)
            .ok_or_else(|| {
                BoostlssError::InvalidConfig(format!("Unknown parameter {}", param_name))
            })?;

        let builder = build_fn(ParamBuilder::new());
        for learner in builder.learners {
            self.learners.push((k, learner));
        }
        Ok(self)
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p boostlss`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/boostlss/src/model.rs crates/boostlss/src/cv.rs crates/boostlss/src/engine/cyclical.rs
git commit -m "feat: implement ParamBuilder and update BoostLss::on API"
```

---

### Task 3: Update Python Bindings to New API

**Files:**
- Modify: `crates/boostlss-py/src/model.rs`

- [ ] **Step 1: Write the failing tests**

We just need to check if the workspace compiles.
Run: `cargo check --workspace`
Expected: FAIL in `crates/boostlss-py/src/model.rs` due to the updated `on` signature.

- [ ] **Step 2: Write minimal implementation**

Modify `crates/boostlss-py/src/model.rs`.
Around line 79, replace:
```rust
                    let mut b = b.clone();
                    b = b
                        .on(param.as_str(), learner.clone())
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;
```
with:
```rust
                    let mut b = b.clone();
                    b = b
                        .on(param.as_str(), |p| p.add(learner.clone()))
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;
```

Around line 136, replace:
```rust
                        let mut b = b.clone();
                        b = b
                            .on(param.as_str(), learner.clone())
                            .map_err(|e| PyValueError::new_err(e.to_string()))?;
```
with:
```rust
                        let mut b = b.clone();
                        b = b
                            .on(param.as_str(), |p| p.add(learner.clone()))
                            .map_err(|e| PyValueError::new_err(e.to_string()))?;
```

- [ ] **Step 3: Run test to verify it passes**

Run: `cargo check --workspace`
Expected: PASS

Run: `maturin develop && pytest crates/boostlss-py/tests`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add crates/boostlss-py/src/model.rs
git commit -m "chore: update python bindings for new ParamBuilder API"
```
