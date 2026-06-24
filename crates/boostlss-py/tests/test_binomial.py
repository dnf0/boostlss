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

    model.add_learner("mu", PyLinearLearner(0, intercept=True))

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


def test_binomial_feature_importance():
    np.random.seed(42)
    X = np.random.uniform(-3, 3, (100, 1))
    y = np.random.binomial(1, 0.5, 100).astype(float)

    family = PyFamily("BinomialLSS")
    model = BoostLssModel(family, mstop=10, step_length=0.1)
    model.add_learner("mu", PyLinearLearner(0, intercept=True))
    model.fit(X, y)

    fi = model.feature_importance()
    assert len(fi) == 1
    assert fi[0] >= 0.0


def test_binomial_partial_dependence():
    np.random.seed(42)
    X = np.random.uniform(-3, 3, (100, 1))
    y = np.random.binomial(1, 0.5, 100).astype(float)

    family = PyFamily("BinomialLSS")
    model = BoostLssModel(family, mstop=10, step_length=0.1)
    model.add_learner("mu", PyLinearLearner(0, intercept=True))
    model.fit(X, y)

    grid = np.linspace(-3, 3, 10).tolist()
    pd = model.partial_dependence(X, "mu", 0, grid)
    assert len(pd) == 10


def test_binomial_cvrisk():
    np.random.seed(42)
    X = np.random.uniform(-3, 3, (100, 1))
    y = np.random.binomial(1, 0.5, 100).astype(float)

    family = PyFamily("BinomialLSS")
    model = BoostLssModel(family, mstop=10, step_length=0.1)
    model.add_learner("mu", PyLinearLearner(0, intercept=True))
    model.fit(X, y)

    cv_result = model.cvrisk(folds=2)
    assert "optimal_mstop" in cv_result
    assert "mean_risk" in cv_result
