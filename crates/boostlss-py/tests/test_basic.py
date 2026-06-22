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

    assert len(pred_sigma) == 100
    assert not np.isnan(pred_mu).any()
    assert not np.isnan(pred_sigma).any()


def test_cvrisk():
    import numpy as np
    from boostlss_py import PyFamily, PyLinearLearner, BoostLssModel

    np.random.seed(42)
    X = np.random.uniform(-3, 3, (20, 2))
    y = np.random.normal(0, 1, 20)

    family = PyFamily("GaussianLSS")
    model = BoostLssModel(family, mstop=10, step_length=0.1)
    model.add_learner("mu", PyLinearLearner("x", intercept=True))
    model.add_learner("sigma", PyLinearLearner("x", intercept=True))

    model.fit(X, y)

    res = model.cvrisk(folds=2)
    assert res is not None
    assert "optimal_mstop" in res


def test_stump_learner():
    import numpy as np
    from boostlss_py import PyFamily, PyStumpLearner, BoostLssModel

    np.random.seed(42)
    X = np.random.uniform(-3, 3, (20, 2))
    y = np.random.normal(0, 1, 20)

    family = PyFamily("GaussianLSS")
    model = BoostLssModel(family, mstop=10, step_length=0.1)
    # Add a stump instead of linear learner
    model.add_learner("mu", PyStumpLearner("x"))
    model.add_learner("sigma", PyStumpLearner("x"))

    model.fit(X, y)

    pred_mu = model.predict(X, "mu")
    assert len(pred_mu) == 20


def test_tree_learner():
    import numpy as np
    from boostlss_py import PyFamily, PyTreeLearner, BoostLssModel

    np.random.seed(42)
    X = np.random.uniform(-3, 3, (20, 2))
    y = np.random.normal(0, 1, 20)

    family = PyFamily("GaussianLSS")
    model = BoostLssModel(family, mstop=10, step_length=0.1)
    model.add_learner("mu", PyTreeLearner([0, 1], max_depth=2, min_samples_leaf=1))
    model.add_learner("sigma", PyTreeLearner([0, 1]))

    model.fit(X, y)

    pred_mu = model.predict(X, "mu")
    assert len(pred_mu) == 20


def test_feature_importance():
    import numpy as np
    from boostlss_py import PyFamily, PyLinearLearner, BoostLssModel

    np.random.seed(42)
    X = np.random.uniform(-3, 3, (20, 1))
    y = np.random.normal(0, 1, 20)

    family = PyFamily("GaussianLSS")
    model = BoostLssModel(family, mstop=10, step_length=0.1)
    model.add_learner("mu", PyLinearLearner("x", intercept=True))
    model.add_learner("sigma", PyLinearLearner("x", intercept=True))

    model.fit(X, y)

    fi = model.feature_importance()
    assert len(fi) == 2


def test_partial_dependence():
    import numpy as np
    from boostlss_py import PyFamily, PyLinearLearner, BoostLssModel

    np.random.seed(42)
    X = np.random.uniform(-3, 3, (20, 2))
    y = np.random.normal(0, 1, 20)

    family = PyFamily("GaussianLSS")
    model = BoostLssModel(family, mstop=10, step_length=0.1)
    model.add_learner("mu", PyLinearLearner("x", intercept=True))

    model.fit(X, y)

    grid = np.linspace(-3, 3, 10).tolist()
    pd = model.partial_dependence(X, "mu", 0, grid)
    assert len(pd) == 10
