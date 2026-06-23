import numpy as np
from boostlss_py import PyFamily, PyLinearLearner, BoostLssModel


def test_lognormal_fit_predict():
    np.random.seed(42)
    X = np.random.uniform(-3, 3, (100, 1))
    # Dummy positive data for LogNormal
    y = np.exp(np.random.normal(0, 1, 100))

    family = PyFamily("LogNormalLSS")
    model = BoostLssModel(family, mstop=10, step_length=0.1)

    model.add_learner("mu", PyLinearLearner("x", intercept=True))
    model.add_learner("sigma", PyLinearLearner("x", intercept=True))

    model.fit(X, y)

    pred_mu = model.predict(X, "mu")
    pred_sigma = model.predict(X, "sigma")

    assert len(pred_sigma) == 100
    assert not np.isnan(pred_mu).any()
    assert not np.isnan(pred_sigma).any()
