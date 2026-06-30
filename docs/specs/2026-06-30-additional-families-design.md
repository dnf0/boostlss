# Additional Families Design

## Overview

We will implement four new distributional families for `BoostLSS` to expand its modeling capabilities: **Tweedie**, **ZINB** (Zero-Inflated Negative Binomial), **Logistic**, and **Laplace**. Each family will implement the `Family` trait in Rust, be exported via `boostlss-py`, and integrated into the overall `BoostLss` engine.

## 1. Tweedie Distribution (`TweedieLss`)

Tweedie is a compound Poisson-Gamma distribution widely used for modeling positive continuous data with an exact mass at zero.

- **Parameters:**
  - $\mu$ (mean): Boosted via `LogLink`. Constraint: $\mu > 0$.
  - $\phi$ (dispersion): Boosted via `LogLink`. Constraint: $\phi > 0$.
- **Hyperparameter:**
  - $p$ (variance power): Fixed configuration value (default $1.5$). Valid range: $1 < p < 2$.
- **Implementation Strategy:**
  - Since the Tweedie PDF lacks a closed-form solution and requires an infinite series approximation, calculating the exact NLL and its gradients is extremely complex and slow.
  - XGBoost natively uses the Tweedie deviance as the loss function. We will implement the exact Tweedie negative log-likelihood approximation (using finite differences for the gradient if necessary, or closed-form gradients for the deviance representation).
  - Actually, minimizing the Tweedie Deviance is mathematically equivalent to maximizing the likelihood for $\mu$. For $\phi$, we need the full likelihood. The `tweedie` crate or custom approximations will be evaluated. We'll use finite difference gradients by default for complex NLL functions.

## 2. Zero-Inflated Negative Binomial (`ZINBLss`)

Combines overdispersed count data with excess zeros.

- **Parameters:**
  - $\mu$ (mean of NB component): `LogLink`.
  - $\sigma$ (dispersion/shape of NB component): `LogLink`. (Using $\sigma$ rather than $\phi$ to align with `NBinomialLss`).
  - $\nu$ (zero-inflation probability): `LogitLink`.
- **Implementation Strategy:**
  - Will combine the logic from our `NBinomialLss` and `ZIPLss`.
  - PDF: $P(Y=0) = \nu + (1-\nu) \cdot NB(0; \mu, \sigma)$
  - PDF: $P(Y=y) = (1-\nu) \cdot NB(y; \mu, \sigma)$ for $y > 0$.

## 3. Logistic Distribution (`LogisticLss`)

Heavy-tailed alternative to Gaussian for robust regression.

- **Parameters:**
  - $\mu$ (location): `IdentityLink`.
  - $s$ (scale): `LogLink`.
- **Implementation Strategy:**
  - NLL: $- \ln(PDF) = \frac{y-\mu}{s} + \ln(s) + 2 \ln(1 + \exp(-\frac{y-\mu}{s}))$
  - Closed-form or finite difference gradients are straightforward.

## 4. Laplace Distribution (`LaplaceLss`)

Used for robust median regression (L1 loss equivalent).

- **Parameters:**
  - $\mu$ (location): `IdentityLink`.
  - $b$ (scale): `LogLink`.
- **Implementation Strategy:**
  - NLL: $\ln(2b) + \frac{|y-\mu|}{b}$
  - Note: The absolute value is non-differentiable at $y=\mu$. We will use a smoothed approximation (Pseudo-Huber or small epsilon addition) to ensure stable finite-difference or closed-form gradients. $\sqrt{(y-\mu)^2 + \epsilon}$.

## Integration

- Add the struct definitions to `crates/boostlss/src/family/`.
- Export them in `crates/boostlss/src/family/mod.rs`.
- Wrap them in Python in `crates/boostlss-py/src/family.rs`.
- Register them in the `BoostLss` internal enums (`Algorithm`, `model.rs`).
- Update `test_family.py` to ensure they can be instantiated and fit.
