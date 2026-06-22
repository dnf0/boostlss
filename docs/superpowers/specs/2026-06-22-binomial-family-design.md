# BinomialLss Family Design Spec

## 1. Overview
Implement the `BinomialLss` distribution family to support binary classification and proportional data modeling in the `boostlss` engine. While typical GAMLSS models involve multiple parameters (Location, Scale, Shape), the standard binomial formulation is a 1-parameter family modeling the probability of success `mu`.

## 2. Architecture & Components
- **File Location:** `crates/boostlss/src/family/binomial.rs`
- **Family Struct:** `BinomialLss` implementing the `Family` trait.
- **Parameters:**
  - `mu`: Represents the probability of success.
  - Link function: `Link::Logit` (maps `[0, 1]` to `(-inf, inf)`).
- **Validation:**
  - `check_response` will strictly enforce `0.0 <= y <= 1.0`. Any response outside this range will return `BoostlssError::UnsupportedResponse`.

## 3. Mathematical Formulation & Data Flow

Let `y` be the response vector (`0 <= y <= 1`), and let `eta` (or `f`) be the additive predictor on the logit link scale. Then `mu = 1 / (1 + exp(-eta))`.

### 3.1 Offset (Intercept-only MLE)
The optimal constant starting value `eta_0` minimizes the risk over the training data.
- **Formula:** `logit(weighted_mean(y, w))`
- **Numerical Stability:** If `mean(y)` is exactly `0` or `1`, the logit function evaluates to `-inf` or `inf`. To prevent this, the mean will be clamped to `[1e-5, 1.0 - 1e-5]` before applying the logit function.

### 3.2 Risk (Empirical Loss)
The Empirical Risk is the total (weighted) negative log-likelihood (NLL).
- **NLL per observation:** `- [y * log(mu) + (1-y) * log(1-mu)]`
- **Link-scale computation:** To avoid numerical underflow with `log(mu)`, we compute the NLL directly from `eta`:
  - `NLL = log(1 + exp(eta)) - y * eta`
  - *Stability Note:* If `eta` is large and positive, `exp(eta)` overflows. We will use a stable log-add-exp formulation:
    - If `eta > 0`, `NLL = eta + log(1 + exp(-eta)) - y * eta`
    - If `eta <= 0`, `NLL = log(1 + exp(eta)) - y * eta`

### 3.3 Negative Gradient
The negative gradient of the risk with respect to the predictor `eta` is the pseudo-response fitted by base learners.
- **Formula:** `- d(NLL)/d(eta) = y - mu`
- **Implementation:** We will use `mu = Link::Logit.inverse(eta)` for simplicity, as `Link::Logit.inverse` already contains overflow protections if implemented correctly, or we'll compute `mu` stably.

## 4. Testing Strategy
- **Finite-Difference Check:** Use the existing `assert_gradient_matches` test harness to ensure the analytic negative gradient exactly matches the central finite difference of the risk function for random `y` and `eta` vectors.
- **Validation Constraints:** Verify that out-of-bounds `y` values (e.g., `-0.1`, `1.1`) correctly return `BoostlssError::UnsupportedResponse`.
- **Offset Tests:** Verify the `offset` correctly computes the logit of the mean, and handles clamping for edge cases (all zeros or all ones).
