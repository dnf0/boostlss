# Distribution Families

In standard regression (e.g., OLS), we only model the conditional mean ($\mu = E[Y|X]$) and assume the variance ($\sigma^2$) is constant for all observations.

**Distributional Regression (GAMLSS)** expands this by allowing *all* parameters of a target distribution to depend on covariates $X$.

$$ Y \sim \mathcal{D}(\theta_1(X), \theta_2(X), \dots, \theta_k(X)) $$

Where $\theta_k(X)$ are the parameters of the distribution, which usually map to:
- Location ($\mu$): Mean, median, or general central tendency.
- Scale ($\sigma$): Variance or dispersion.
- Shape ($\nu, \tau, \dots$): Skewness, kurtosis, or tail weight.

## Link Functions

Base learners always output predictions on the real line $(-\infty, \infty)$. However, distribution parameters often have strict bounds. For example, a variance ($\sigma$) must be strictly positive.

To solve this, `boostlss` uses **Link Functions**.
$$ \theta_k = h^{-1}(\eta_k) $$

Where:
- $\eta_k$ is the unbounded additive predictor formed by the base learners.
- $h^{-1}$ is the inverse link function that maps the unbounded prediction into the valid parameter space.

Example: For $\sigma > 0$, `boostlss` uses the log-link. The base learners predict $\log(\sigma)$, and the engine applies $\exp()$ to get the final parameter.

## Supported Families

### Continuous Data

- **`GaussianLSS`**: Normal distribution. 
  - Models: $\mu$ (Identity link), $\sigma$ (Log link).
  - Use case: Standard continuous regression with heteroscedasticity.
- **`StudentTLSS`**: Student's t-distribution. 
  - Models: $\mu$ (Identity), $\sigma$ (Log), $df$ (Log shift).
  - Use case: Robust regression. The degrees of freedom ($df$) parameter allows modeling heavy tails to handle outliers gracefully.
- **`LogNormalLSS`**: Log-Normal distribution.
  - Models: $\mu$ (Identity), $\sigma$ (Log).
  - Use case: Strictly positive, right-skewed data like income or time-to-event.

### Fractional / Bounded Data

- **`BetaLSS`**: Beta distribution.
  - Models: $\mu$ (Logit link, bounded $(0,1)$), $\phi$ (Log link for precision).
  - Use case: Proportions, rates, or percentages strictly between 0 and 1.

### Count Data

- **`ZIPLss`**: Zero-Inflated Poisson distribution.
  - Models: $\mu$ (Log link for Poisson mean), $\sigma$ (Logit link for zero-inflation probability).
  - Use case: Count data with an excess of structural zeros (e.g., number of insurance claims).

### Extreme Values

- **`GEVLss`**: Generalized Extreme Value distribution.
  - Models: $\mu$ (Identity), $\sigma$ (Log), $\xi$ (Identity for shape).
  - Use case: Modeling block maxima, rare events, or extreme weather phenomena.
- **`WeibullLSS`**: Weibull distribution.
  - Models: $\mu$ (Log), $\sigma$ (Log).
  - Use case: Reliability analysis and survival modeling.

### Binary Data

- **`BinomialLSS`**: Binomial distribution.
  - Models: $\mu$ (Logit link for probability).
  - Use case: Binary classification or N-trial success rates.
