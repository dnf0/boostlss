# Jump Models and SHASH Distributions Design Spec

## Overview
This specification details the implementation of two new flexible distributional families in `boostlss`: the **Merton Jump-Diffusion** model and the **Sinh-Arcsinh (SHASH)** distribution. These families are useful for modeling data with structural breaks, heavy tails, and complex skewness/kurtosis.

## 1. Merton Jump-Diffusion (`MertonJumpDiffusionLss`)
The Merton Jump-Diffusion model combines a continuous Brownian motion diffusion process with discrete Poisson jumps. For cross-sectional data, we assume $\Delta t = 1$.

### Parameters (5)
1. `mu` (Diffusion drift) -> `IdentityLink`
2. `sigma` (Diffusion volatility) -> `LogLink`
3. `lam` (Jump intensity $\lambda$) -> `LogLink`
4. `mu_j` (Jump mean) -> `IdentityLink`
5. `sigma_j` (Jump volatility $\delta$) -> `LogLink`

### Configuration
The distribution accepts a single hyperparameter at initialization:
*   `max_jumps` (default: 10): The truncation limit for the infinite sum of Poisson probabilities.

### Negative Log-Likelihood Formulation
For each observation $x_i$ with weight $w_i$, the likelihood is an infinite sum over the number of jumps $j$. We truncate this sum at `max_jumps`:

$$ L_i = \sum_{j=0}^{max\_jumps} \frac{e^{-\lambda} \lambda^j}{j!} \phi(x_i; \mu_j, \sigma_j^2) $$

Where:
*   $\mu_j = \mu - \frac{1}{2}\sigma^2 + j\mu_j$
*   $\sigma_j^2 = \sigma^2 + j\sigma_j^2$
*   $\phi(x; m, v^2)$ is the normal PDF with mean $m$ and variance $v^2$.

To maintain numerical stability, we will use the `logsumexp` trick or safe probability accumulation to compute the NLL:
$$ \text{NLL} = -\sum_{i=1}^{n} w_i \ln(L_i) $$

---

## 2. Sinh-Arcsinh (`SHASHLss`)
The standard Sinh-Arcsinh distribution is a 4-parameter distribution capable of modeling varying degrees of skewness and kurtosis.

### Parameters (4)
1. `mu` (Location) -> `IdentityLink`
2. `sigma` (Scale) -> `LogLink`
3. `nu` (Skewness) -> `IdentityLink`
4. `tau` (Kurtosis) -> `LogLink`

### Negative Log-Likelihood Formulation
For each observation $y_i$ with weight $w_i$:
Let $z = (y_i - \mu) / \sigma$.
Let $r = \frac{1}{2} \left[ \exp(\tau \sinh^{-1}(z)) - \exp(-\nu \sinh^{-1}(z)) \right]$.
Let $c = \frac{1}{2} \left[ \tau \exp(\tau \sinh^{-1}(z)) + \nu \exp(-\nu \sinh^{-1}(z)) \right]$.

The PDF is:
$$ f(y_i) = \frac{c}{\sqrt{2\pi}\sigma(1+z^2)^{1/2}} \exp(-r^2/2) $$

The NLL is:
$$ \text{NLL} = -\sum_{i=1}^{n} w_i \left[ \ln(c) - \frac{1}{2}\ln(2\pi) - \ln(\sigma) - \frac{1}{2}\ln(1+z^2) - \frac{r^2}{2} \right] $$

## Architecture & Integration
Both models will follow the standard integration path for `boostlss` families:
1.  **Rust implementation:** Added to `crates/boostlss/src/family/`.
2.  **Trait Implementation:** Implement the `Family` trait, overriding `nll` and `init_offsets`. `ngradient` will use the default finite-difference implementation.
3.  **Python bindings:** Exposed in `crates/boostlss-py/src/family.rs` using PyO3.
4.  **Macros:** Added to `FamilyEnum` and macro dispatchers.
5.  **Tests:** Tested for pickling, instantiation, and fitting in `crates/boostlss-py/tests/`.
