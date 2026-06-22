# boostlss — Formula DSL / Builder API Design

## Goal
Improve the ergonomics of the `BoostLss` builder API by allowing users to elegantly chain multiple base-learners to a single distribution parameter without repetitive `.on()` calls or explicit `BaseLearner::Variant(...)` wrapping.

## Current API (Before)
```rust
let model = BoostLss::new(GaussianLss::new())
    .on("mu", BaseLearner::Linear(Linear::new("x1").intercept(false)))?
    .on("mu", BaseLearner::PSpline(PSpline::new("x2")))?
    .on("sigma", BaseLearner::Linear(Linear::new("x1")))?;
```

## Proposed API (After)
```rust
let model = BoostLss::new(GaussianLss::new())
    .on("mu", |p| p
        .add(Linear::new("x1").intercept(false))
        .add(PSpline::new("x2"))
    )?
    .on("sigma", |p| p
        .add(Linear::new("x1"))
    )?;
```

## Implementation Details

1. **`ParamBuilder` Struct:**
   Create a new struct `ParamBuilder` in `crates/boostlss/src/model.rs` that accumulates base learners.
   ```rust
   pub struct ParamBuilder {
       pub(crate) learners: Vec<BaseLearner>,
   }

   impl ParamBuilder {
       pub fn new() -> Self {
           Self { learners: Vec::new() }
       }

       pub fn add<L: Into<BaseLearner>>(mut self, learner: L) -> Self {
           self.learners.push(learner.into());
           self
       }
   }
   ```

2. **`From` Implementations for `BaseLearner`:**
   Implement `From<Linear> for BaseLearner`, `From<PSpline> for BaseLearner`, `From<Stump> for BaseLearner`, and `From<Tree> for BaseLearner` in `crates/boostlss/src/learner/mod.rs` to support the `Into<BaseLearner>` bound in `ParamBuilder::add`.

3. **Modify `BoostLss::on`:**
   Change the signature of `BoostLss::on` to accept a closure that takes a `ParamBuilder` and returns a `ParamBuilder`.
   ```rust
   pub fn on(mut self, param_name: &str, build_fn: impl FnOnce(ParamBuilder) -> ParamBuilder) -> Result<Self, BoostlssError> {
       let params = self.family.params();
       let k = params.iter().position(|p| p.name == param_name)
           .ok_or_else(|| BoostlssError::InvalidConfig(...))?;

       let builder = build_fn(ParamBuilder::new());
       for learner in builder.learners {
           self.learners.push((k, learner));
       }
       Ok(self)
   }
   ```
   *Note: This replaces the previous `on` method. Existing tests will need to be updated to use the new closure-based syntax.*

4. **Python API Impact:**
   The Python API currently uses `.add_learner("mu", PyLinearLearner("x1"))`. It will remain largely unaffected by this Rust-side change, though we will need to update `crates/boostlss-py/src/model.rs` internally to use the new `.on` signature (e.g. `model.on(param.as_str(), |p| p.add(learner.clone()))`).
