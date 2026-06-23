import numpy as np
import pytest
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


@pytest.fixture(scope="module")
def fitted_model_and_data():
    np.random.seed(42)
    # 2D features: X[:, 0] is signal, X[:, 1] is noise
    X = np.random.uniform(-3, 3, (100, 2))
    # y depends only on X[:, 0] with a clear linear relationship
    y = 2.0 * X[:, 0]

    family = PyFamily("GaussianLSS")
    model = BoostLssModel(family, mstop=50, step_length=0.1)

    # We add two learners: one for the signal, one for the noise
    # Both are assigned to 'mu'
    model.add_learner("mu", PyLinearLearner("x0", intercept=True))
    model.add_learner("mu", PyLinearLearner("x1", intercept=True))

    model.fit(X, y)
    return model, X, y


def test_feature_importance(fitted_model_and_data):
    model, _, _ = fitted_model_and_data
    # 1. Feature Importance
    fi = model.feature_importance()
    assert len(fi) == 2
    # The first learner (signal) should have much higher importance than the second (noise)
    assert fi[0] > 0
    assert fi[0] > fi[1] * 10


def test_partial_dependence(fitted_model_and_data):
    model, X, _ = fitted_model_and_data
    # 2. Partial Dependence
    grid = np.linspace(-3, 3, 10).tolist()
    # PD for the first feature (feature_idx=0), which is the signal
    pd = model.partial_dependence(X, "mu", 0, grid)
    assert len(pd) == 10

    # Check that PD is monotonically increasing, since y = 2.0 * X[:, 0]
    for i in range(1, len(pd)):
        assert pd[i] > pd[i - 1]


def test_algorithm_param():
    from boostlss_py import PyFamily, BoostLssModel

    family = PyFamily("GaussianLSS")

    # default should be cyclic
    model = BoostLssModel(family, mstop=10, step_length=0.1)
    assert model is not None

    # explicit valid options
    model_cyclic = BoostLssModel(family, mstop=10, step_length=0.1, algorithm="cyclic")
    assert model_cyclic is not None
    model_noncyclic = BoostLssModel(
        family, mstop=10, step_length=0.1, algorithm="noncyclic"
    )
    assert model_noncyclic is not None

    # invalid option
    with pytest.raises(ValueError, match="algorithm must be 'cyclic' or 'noncyclic'"):
        BoostLssModel(family, mstop=10, step_length=0.1, algorithm="invalid_algo")


def test_noncyclic_fit():
    import numpy as np
    from boostlss_py import PyFamily, PyLinearLearner, BoostLssModel

    np.random.seed(42)
    X = np.random.uniform(-3, 3, (50, 1))
    mu = 2 * X[:, 0]
    sigma = np.exp(0.5 * X[:, 0])
    y = np.random.normal(mu, sigma)

    family = PyFamily("GaussianLSS")
    model = BoostLssModel(family, mstop=10, step_length=0.1, algorithm="noncyclic")
    model.add_learner("mu", PyLinearLearner("x", intercept=True))
    model.add_learner("sigma", PyLinearLearner("x", intercept=True))

    # This should not raise an error
    model.fit(X, y)

    pred_mu = model.predict(X, "mu")
    pred_sigma = model.predict(X, "sigma")

    assert len(pred_sigma) == 50
    assert not np.isnan(pred_mu).any()
    assert not np.isnan(pred_sigma).any()


def test_cvrisk_noncyclic():
    import numpy as np
    from boostlss_py import PyFamily, PyLinearLearner, BoostLssModel

    np.random.seed(42)
    X = np.random.uniform(-3, 3, (20, 2))
    y = np.random.normal(0, 1, 20)

    family = PyFamily("GaussianLSS")
    model = BoostLssModel(family, mstop=10, step_length=0.1, algorithm="noncyclic")
    model.add_learner("mu", PyLinearLearner("x", intercept=True))
    model.add_learner("sigma", PyLinearLearner("x", intercept=True))

    model.fit(X, y)

    res = model.cvrisk(folds=2)
    assert res is not None
    assert "optimal_mstop" in res
