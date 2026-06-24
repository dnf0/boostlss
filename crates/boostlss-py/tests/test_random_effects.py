import pytest  # type: ignore
import numpy as np  # type: ignore
from boostlss_py import PyFamily, PyRandomEffectsLearner, BoostLssModel  # type: ignore


def test_random_effects():
    # 3 groups, indexed 0, 1, 2
    groups = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])

    # Ground truth means: group 0 -> 10, group 1 -> 20, group 2 -> 30
    y = np.array([10.1, 9.9, 20.1, 19.9, 30.1, 29.9])

    # Needs to be 2D for input
    X = groups.reshape(-1, 1)

    family = PyFamily("GaussianLSS")
    model = BoostLssModel(family, mstop=1000, step_length=1.0)

    # Pass df=3 for minimal penalization to quickly fit means
    model.add_learner("mu", PyRandomEffectsLearner(0, df=3.0))
    model.fit(X, y)

    preds = model.predict(X, "mu")

    assert abs(preds[0] - 10.0) < 1.0
    assert abs(preds[2] - 20.0) < 1.0
    assert abs(preds[4] - 30.0) < 1.0

    # Unseen group (index 3) should predict exactly the global offset (global mean)
    # The global mean of 10, 20, 30 is 20.
    X_unseen = np.array([[3.0]])
    unseen_pred = model.predict(X_unseen, "mu")
    assert abs(unseen_pred[0] - 20.0) < 1.0


def test_random_effects_invalid_index():
    groups = np.array([0.5, 1.2])  # Invalid indices
    X = groups.reshape(-1, 1)
    y = np.array([10.0, 20.0])

    model = BoostLssModel(PyFamily("GaussianLSS"), mstop=1)
    model.add_learner("mu", PyRandomEffectsLearner(0))

    with pytest.raises(RuntimeError, match="non-negative integer"):
        model.fit(X, y)
