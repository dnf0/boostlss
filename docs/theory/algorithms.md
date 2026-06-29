# Boosting Algorithms

`boostlss` provides three distinct algorithms for optimizing the negative log-likelihood of distributional models. This flexibility allows practitioners to trade off computational efficiency and statistical performance based on their specific dataset and model requirements.

## 1. Cyclic Algorithm (`"cyclic"`)

The **Cyclic** algorithm updates the parameters of the selected distribution (e.g., location $\mu$, scale $\sigma$) in a fixed, round-robin sequence during each boosting iteration.

**How it works:**

For iteration $m = 1 \dots M$:

1.  For each parameter $k \in \{\mu, \sigma, \dots\}$:
    1.  Compute the negative gradient (pseudo-residuals) of the loss function with respect to parameter $k$, evaluated at the current predictions of all parameters.
    2.  Fit all available base learners for parameter $k$ to this negative gradient.
    3.  Select the base learner that minimizes the sum of squared errors against the negative gradient.
    4.  Update the parameter $k$ prediction by adding the predictions of the selected base learner, scaled by the learning rate (`step_length`).

**When to use:**

- This is the traditional approach used in `gamboostLSS`.
- Best when parameters are relatively independent or when interpretability of the exact iteration process is required.
- **Drawback:** It rigidly forces an update to _every_ parameter in every iteration, which can lead to overfitting if some parameters require fewer updates than others.

---

## 2. Non-Cyclic Algorithm (`"noncyclic"`)

The **Non-Cyclic** algorithm takes a greedy approach. Instead of updating every parameter in a cycle, it evaluates the best possible update across _all_ parameters and applies only the single best update.

**How it works:**

For iteration $m = 1 \dots M$:

1.  For each parameter $k \in \{\mu, \sigma, \dots\}$:
    1.  Compute the negative gradient.
    2.  Fit all base learners for parameter $k$ and select the best one based on residual error.
    3.  Compute the _overall empirical risk (loss)_ if this base learner were to be added.
2.  Select the **single parameter** $k^*$ and its corresponding base learner that results in the lowest overall loss.
3.  Update _only_ parameter $k^*$ by adding the selected base learner's predictions scaled by the learning rate.

**When to use:**

- Highly recommended for modern predictive tasks.
- Inherently provides variable selection across parameters. If the variance ($\sigma$) does not need modeling, the algorithm will automatically spend more iterations updating the mean ($\mu$).
- Generally provides better out-of-sample prediction metrics than Cyclic, as it avoids forced, unnecessary updates.

---

## 3. Non-Cyclic Outer Algorithm (`"noncyclic_outer"`)

The **Non-Cyclic Outer** algorithm is an aggressive variant of the non-cyclic approach. In an iteration, it computes the optimal base learner for _every_ parameter, tentatively updates _all_ of them simultaneously, and evaluates whether the joint update is better than the individual updates.

**How it works:**

For iteration $m = 1 \dots M$:

1. Evaluate the single best update across all parameters (identical to the standard Non-Cyclic approach). Let the best parameter be $k^*$.
2. Compute the loss if we were to apply the best update for _every_ parameter simultaneously (an "outer" update).
3. If the simultaneous "outer" update yields a lower loss than the single best update for $k^*$, apply the simultaneous update to all parameters. Otherwise, apply only the single update to $k^*$.

**When to use:**

- Useful when parameters are highly coupled (e.g., changes in mean drastically affect the variance scale).
- Can converge in fewer iterations ($M$) than standard non-cyclic, but each iteration requires more computation.
