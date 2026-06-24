# Stability Selection (stabsel) Design

## Overview
Implement stability selection with error control for the `boostlss` library. This feature provides a robust method to select influential base-learners across subsamples, with theoretical error guarantees (Per-Family Error Rate).

## Goals
- Add stability selection capabilities natively in the Rust core for performance.
- Expose an idiomatic method `model.stabsel()` on the Python `BoostLssModel` interface.
- Provide a hybrid configuration mechanism requiring exactly two of `(q, PFER, pi_thr)`, deriving the third.
- Support early stopping of subsample boosting runs once `q` variables are selected.
- Provide configurable selection tracking: either joint across all distribution parameters or independent per parameter.

## Components and Data Flow

### 1. Rust Core (`crates/boostlss/src/cv/stabsel.rs`)
- **`StabselMode` Enum:** `Joint` (select if base-learner is active in ANY parameter) vs `Independent` (track selection per parameter).
- **`StabselConfig` Struct:** Encapsulates the runtime configuration (`B` subsamples, `mode`, `q`, `PFER`, `pi_thr`).
- **Error Bounding:** Implement Shah & Samworth (2013) bounds calculation to dynamically solve for the missing constraint out of `(q, pfer, pi_thr)`.
- **Parallel Subsampling:** Utilize `rayon` to spawn `B` independent runs. Each run:
  1. Draws a random sample of size `N/2` without replacement.
  2. Iteratively fits the model on the subsample.
  3. Evaluates the number of active base-learners.
  4. Triggers early stopping immediately when `q` unique base-learners have been selected (if `q` is acting as the stopping constraint).
- **Aggregation:** Tally selection frequencies globally across the `B` runs.

### 2. Python Bindings (`crates/boostlss-py/src/`)
- Expose a new method on `BoostLssModel`:
  ```python
  def stabsel(self, B: int = 100, pfer: float = None, pi_thr: float = None, q: int = None, mode: str = "joint") -> StabselResult:
  ```
- Validate inputs in the wrapper to guarantee exactly two of `(pfer, pi_thr, q)` are non-null before invoking the Rust core.

### 3. Output Format
- Define a Python-accessible `StabselResult` PyClass containing:
  - `selected`: A list of selected base-learner names (for `joint` mode) or a dict mapping parameter names to lists of selected base-learners (for `independent` mode).
  - `probabilities`: A dict mapping base-learners to their selection probabilities (or nested dict by parameter).
  - `cutoff`: The explicit or derived `pi_thr`.
  - `q`: The explicit or derived `q`.

## Error Handling
- Return `BoostlssError::InvalidConfig` if fewer or more than two parameters out of `(pfer, pi_thr, q)` are provided.
- Ensure mathematically valid ranges: `pi_thr` in `(0.5, 1.0)`, `pfer` `> 0`, `q` `> 0`.
- Log a warning if a user provides an explicit `mstop` override that completes before `q` variables are ever selected.

## Testing Strategy
- **Rust Unit Tests:** Validate Shah & Samworth bounds logic against known constant values. Test subsampling array generation.
- **Python Integration Tests:** End-to-end execution on a synthetic dataset with known informative and noise features. Assert informative features are captured in `selected` and noise features fall below `pi_thr`.
