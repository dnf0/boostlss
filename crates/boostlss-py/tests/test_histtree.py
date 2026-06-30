import numpy as np
from boostlss_py import PyFamily, PyHistTreeLearner, BoostLssModel  # type: ignore


def test_histtree_learner():
    np.random.seed(42)
    X = np.random.uniform(-3, 3, (100, 2))
    # Create an arbitrary non-linear target
    mu = 2.0 * np.sin(X[:, 0]) + X[:, 1]
    y = np.random.normal(mu, 0.5)

    family = PyFamily("GaussianLSS")
    model = BoostLssModel(family, mstop=20, step_length=0.1)

    model.add_learner(
        "mu", PyHistTreeLearner([0, 1], max_bins=256, max_depth=3, min_samples_leaf=5)
    )
    model.add_learner(
        "sigma",
        PyHistTreeLearner([0, 1], max_bins=256, max_depth=2, min_samples_leaf=10),
    )

    model.fit(X, y)

    pred_mu = model.predict(X, "mu")
    pred_sigma = model.predict(X, "sigma")

    assert len(pred_mu) == 100
    assert len(pred_sigma) == 100
    assert not np.isnan(pred_mu).any()
    assert not np.isnan(pred_sigma).any()
