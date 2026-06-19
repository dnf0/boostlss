import numpy as np
from boostlss_py import PyFamily, PyLinearLearner, BoostLssModel  # type: ignore


def test_gaussian_fit_predict():
    # 1. Generate data
    np.random.seed(42)
    X = np.random.uniform(-3, 3, (100, 1))
    mu = 2 * X[:, 0]
    sigma = np.exp(0.5 * X[:, 0])
    y = np.random.normal(mu, sigma)

    # 2. Build model
    family = PyFamily("GaussianLSS")
    model = BoostLssModel(family, mstop=10, step_length=0.1)

    # 3. Add learners
    model.add_learner("mu", PyLinearLearner("x", intercept=True))
    model.add_learner("sigma", PyLinearLearner("x", intercept=True))

    # 4. Fit
    model.fit(X, y)

    # 5. Predict
    pred_mu = model.predict(X, "mu")
    pred_sigma = model.predict(X, "sigma")

    assert len(pred_mu) == 100
    assert len(pred_sigma) == 100
    assert not np.isnan(pred_mu).any()
    assert not np.isnan(pred_sigma).any()
