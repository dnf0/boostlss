import numpy as np
from boostlss_py import BoostLssModel, PyFamily, constrained_pspline


def test_monotonic_increasing():
    np.random.seed(42)
    # Underlying function is y = -x (decreasing)
    x = np.sort(np.random.uniform(-3, 3, 100))
    y = -x + np.random.normal(0, 0.1, 100)

    # Fit with increasing constraint
    model = BoostLssModel(PyFamily("GaussianLSS"), mstop=100, step_length=0.1)
    model.add_learner(
        "mu",
        constrained_pspline(
            0, "monotonic_increasing", df=2.0, max_iter=100, tolerance=1e-8
        ),
    )

    X = x.reshape(-1, 1)
    model.fit(X, y)

    # Predictions should be strictly monotonically increasing (or flat)
    preds = model.predict(X, "mu")
    diffs = np.diff(preds)
    assert np.all(diffs >= -1e-3), "Predictions are not monotonically increasing"


def test_monotonic_decreasing():
    np.random.seed(42)
    x = np.sort(np.random.uniform(-3, 3, 100))
    y = x + np.random.normal(0, 0.1, 100)  # True is increasing

    model = BoostLssModel(PyFamily("GaussianLSS"), mstop=100, step_length=0.1)
    model.add_learner(
        "mu",
        constrained_pspline(
            0, "monotonic_decreasing", df=2.0, max_iter=100, tolerance=1e-8
        ),
    )

    X = x.reshape(-1, 1)
    model.fit(X, y)

    preds = model.predict(X, "mu")
    diffs = np.diff(preds)
    assert np.all(diffs <= 1e-3), "Predictions are not monotonically decreasing"
