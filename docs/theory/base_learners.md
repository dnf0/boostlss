# Base Learners

Base learners are the functional building blocks of gradient boosting. In `boostlss`, they act as weak models that are sequentially fitted to the negative gradients (pseudo-residuals) of the loss function.

## Regularization via Degrees of Freedom (`df`)

Unlike standard Machine Learning where base learners are solely regularized by learning rate and iteration count, `boostlss` (following the GAMLSS framework) relies heavily on penalized regression.

Base learners such as P-Splines, Spatial Splines, and Random Effects are regularized by a penalty matrix $\mathbf{K}$ and a smoothing parameter $\lambda$. To make this interpretable, `boostlss` parameterizes this via the **Degrees of Freedom (`df`)**.

The target `df` implicitly dictates how smooth or flexible the fit should be:
- **Low `df` (e.g., 1 or 2):** Strong penalty, highly smoothed, approaches a simple linear fit.
- **High `df` (e.g., 10 or 15):** Weak penalty, highly flexible, tracks data closely (can overfit).

During the fitting step, `boostlss` automatically computes the exact $\lambda$ via Demmler-Reinsch orthogonalization to match your requested `df`.

---

## Linear Learner

```python
PyLinearLearner(feature_idx: int, intercept: bool = True)
```

Fits an unpenalized simple linear regression $y = \beta_0 + \beta_1 x$. 
- If `intercept=True` (default), the design matrix is expanded to include a column of ones.
- Linear learners have $df = 1$ (or $2$ with an intercept).

---

## Penalized Splines (P-Splines)

```python
PyPSplineLearner(feature_idx: int, degree: int = 3, knots: int = 20, differences: int = 2, df: float = 4.0, cyclic: bool = False)
```

Models non-linear continuous features. It constructs a basis matrix using B-splines and applies a discrete difference penalty to adjacent basis coefficients.

- **`degree`**: The degree of the polynomial pieces. `degree=3` yields cubic splines (smooth curves).
- **`knots`**: The number of inner breakpoints. More knots allow for more local wiggles, but the penalty (`df`) keeps it smooth. Usually, 20 is sufficient for most data.
- **`differences`**: The order of the discrete penalty. `differences=2` penalizes the second derivative (encouraging linearity), while `1` penalizes the first derivative (encouraging flat constants).
- **`cyclic`**: If `True`, the spline wraps around at the boundaries. This is crucial for periodic features like "day of year" or "hour of day" to ensure the prediction at 23:59 matches the prediction at 00:00.

---

## Bivariate Tensor-Product Splines

```python
PyBivariatePSplineLearner(feature1_idx: int, feature2_idx: int, df: float = 4.0)
```

Used to model spatial data (e.g., Latitude and Longitude) or complex continuous interactions. It computes the Kronecker tensor-product of two marginal B-spline bases and applies an anisotropic penalty matrix.

---

## Constrained P-Splines

```python
constrained_pspline(feature_idx: int, constraint: str, df: float = 4.0)
```

When domain knowledge dictates the shape of the effect, you can enforce shape constraints on the P-Spline using an asymmetric L2 penalty. 
- `"monotonic_increasing"`: Enforces $f(x) \geq f(x - \epsilon)$.
- `"monotonic_decreasing"`: Enforces $f(x) \leq f(x - \epsilon)$.
- `"convex"`: Enforces a positive second derivative (U-shape).
- `"concave"`: Enforces a negative second derivative (Cap-shape).

---

## Random Effects

```python
PyRandomEffectsLearner(feature_idx: int, df: float = 4.0)
```

Designed for categorical variables with high cardinality (e.g., user IDs, city codes, product SKUs). It constructs an indicator design matrix and applies an $L_2$ Ridge penalty ($\mathbf{K} = \mathbf{I}$).
This shrinks the group-specific intercepts towards the global mean, acting as a Bayesian Random Intercept model.

---

## Trees and Stumps

```python
PyTreeLearner(feature_indices: list[int], max_depth: int = 3, min_samples_leaf: int = 1)
PyStumpLearner(feature_idx: int)
```

While traditional GAMLSS relies on additive component models (splines/linear), `boostlss` also supports regression trees.
- A **Stump** is a tree with exactly one split (depth 1), looking at a single feature.
- A **Tree** can model deep interactions across multiple features.

*Note: Trees do not use the `df` penalty system. They are regularized by `max_depth` and `min_samples_leaf`.*
