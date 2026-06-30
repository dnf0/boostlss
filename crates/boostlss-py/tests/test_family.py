def test_tweedie():
    from boostlss_py import TweedieLss, BoostLssModel, PyLinearLearner
    import numpy as np

    fam = TweedieLss(p=1.5)
    model = BoostLssModel(fam, mstop=2)
    model.add_learner("mu", PyLinearLearner(0))
    model.add_learner("phi", PyLinearLearner(0))

    # Must use positive response for Tweedie
    y = np.random.poisson(lam=5, size=10) + np.random.gamma(shape=2, scale=1, size=10)
    y = np.maximum(y, 0.0)
    X = np.random.normal(size=(10, 2))

    model.fit(X, y)
    assert len(model.predict(X, "mu")) == 10


def test_zinb():
    from boostlss_py import ZINBLss, BoostLssModel, PyLinearLearner
    import numpy as np

    fam = ZINBLss()
    model = BoostLssModel(fam, mstop=2)
    model.add_learner("mu", PyLinearLearner(0))
    model.add_learner("sigma", PyLinearLearner(0))
    model.add_learner("nu", PyLinearLearner(0))

    y = np.random.poisson(lam=5, size=10)
    y[0:3] = 0.0  # Force zeros
    X = np.random.normal(size=(10, 2))

    model.fit(X, y.astype(float))
    assert len(model.predict(X, "mu")) == 10
