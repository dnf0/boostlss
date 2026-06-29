# JSU (Johnson's SU) Distribution Family

## Objective
Implement the JSU (Johnson's SU) distribution as a 4-parameter family in BoostLSS to support heavy-tailed and skewed data modeling.

## Architecture & Components
We will add a new family `JSULss` in `crates/boostlss/src/family/jsu.rs`.
Register `JSULss` in `crates/boostlss/src/family/mod.rs` and expose it to the Python bindings in `crates/boostlss-py/src/family.rs`.

### Parameterization
The standard 4-parameter JSU distribution (as per `gamlss.dist`) uses:
1. `mu` (Location): Identity link
2. `sigma` (Scale): Log link (strictly positive)
3. `nu` (Skewness): Identity link
4. `tau` (Tail weight): Log link (strictly positive)

### Implementation Details
1. **NLL (Negative Log-Likelihood)**:
   Evaluate the JSU NLL for each observation $y$. Let $z = \frac{y - \mu}{\sigma}$. 
   $r = -\nu + \tau \sinh^{-1}(z)$
   $\log(PDF) = \log(\tau) - \log(\sigma) - 0.5 \log(z^2 + 1) - 0.5 \log(2\pi) - 0.5 r^2$
   NLL is the negative sum of weighted log PDFs over the dataset.

2. **Gradients**:
   Use the default finite difference implementation for gradients to avoid complex error-prone analytical derivatives.

3. **Initialization**:
   - `mu`: Weighted mean of response.
   - `sigma`: Weighted standard deviation of response.
   - `nu`: 0.0
   - `tau`: 1.0
   We will refine these using `minimize_1d` iteratively over the parameters to find optimal constant offsets, similar to the GEV initialization.

4. **Tests**:
   - Verify `init_offsets` converges reasonably using standard dummy data.
   - We do not need a specific analytical gradient finite-difference test since we are using finite-differences natively, but we will add basic sanity tests for bounds.

## Data Flow
- `JSULss::new()` initializes the 4 parameters with their corresponding links.
- During boosting, `nll` is called to evaluate risk, and `ngradient` calculates pseudo-residuals via finite difference.

## Python Bindings
Add `Jsu` / `JSULss` to `PyFamily` in `crates/boostlss-py/src/family.rs`.
