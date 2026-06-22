import numpy as np
from boostlss_py import PyFamily, PyLinearLearner, BoostLssModel  # type: ignore


def test_binomial_fit_predict():
    np.random.seed(42)
    # increase sample size to 200
    X = np.random.uniform(-3, 3, (200, 1))
    # p = exp(x) / (1 + exp(x))
    p = np.exp(X[:, 0]) / (1 + np.exp(X[:, 0]))
    y = np.random.binomial(1, p).astype(float)

    family = PyFamily("BinomialLSS")
    # bump mstop to 50
    model = BoostLssModel(family, mstop=50, step_length=0.1)

    model.add_learner("mu", PyLinearLearner("x", intercept=True))

    model.fit(X, y)

    pred_mu = model.predict(X, "mu")

    assert len(pred_mu) == 200
    assert not np.isnan(pred_mu).any()

    # bounds checks
    assert (pred_mu >= 0.0).all() and (pred_mu <= 1.0).all()

    # accuracy checks
    pred_classes = (pred_mu > 0.5).astype(float)
    accuracy = np.mean(pred_classes == y)
    assert accuracy > 0.70
