import numpy as np
from boostlss_py import PyFamily, PyLinearLearner, BoostLssModel


def test_noncyclic_outer():
    X = np.random.normal(size=(100, 2))
    y = X[:, 0] * 2.0 + np.random.normal(size=100) * 0.1

    family = PyFamily("GaussianLSS")
    model = BoostLssModel(
        family, mstop=10, step_length=0.1, algorithm="noncyclic_outer"
    )
    model.add_learner("mu", PyLinearLearner(0))
    model.add_learner("sigma", PyLinearLearner(0))

    model.fit(X, y)
    preds = model.predict(X, "mu")
    assert len(preds) == 100
