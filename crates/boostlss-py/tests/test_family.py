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


def test_logistic():
    from boostlss_py import LogisticLss, BoostLssModel, PyLinearLearner
    import numpy as np

    fam = LogisticLss()
    model = BoostLssModel(fam, mstop=2)
    model.add_learner("mu", PyLinearLearner(0))
    model.add_learner("s", PyLinearLearner(0))

    y = np.random.logistic(loc=5, scale=2, size=10)
    X = np.random.normal(size=(10, 2))

    model.fit(X, y)
    assert len(model.predict(X, "mu")) == 10


def test_laplace():
    from boostlss_py import LaplaceLss, BoostLssModel, PyLinearLearner
    import numpy as np

    fam = LaplaceLss()
    model = BoostLssModel(fam, mstop=2)
    model.add_learner("mu", PyLinearLearner(0))
    model.add_learner("b", PyLinearLearner(0))

    y = np.random.laplace(loc=5, scale=2, size=10)
    X = np.random.normal(size=(10, 2))

    model.fit(X, y)
    assert len(model.predict(X, "mu")) == 10


def test_merton():
    from boostlss_py import MertonJumpDiffusionLss, BoostLssModel, PyLinearLearner
    import numpy as np

    fam = MertonJumpDiffusionLss(max_jumps=10)
    model = BoostLssModel(fam, mstop=2)
    model.add_learner("mu", PyLinearLearner(0))
    model.add_learner("sigma", PyLinearLearner(0))
    model.add_learner("lam", PyLinearLearner(0))
    model.add_learner("mu_j", PyLinearLearner(0))
    model.add_learner("sigma_j", PyLinearLearner(0))

    # Generate some jumpy data
    np.random.seed(42)
    y = np.random.normal(loc=0.05, scale=0.1, size=20)
    # add some jumps
    jumps = np.random.poisson(lam=0.5, size=20)
    y += jumps * np.random.normal(loc=-0.1, scale=0.2, size=20)

    X = np.random.normal(size=(20, 2))

    model.fit(X, y)
    assert len(model.predict(X, "mu")) == 20


def test_shash():
    from boostlss_py import SHASHLss, BoostLssModel, PyLinearLearner
    import numpy as np

    fam = SHASHLss()
    model = BoostLssModel(fam, mstop=2)
    model.add_learner("mu", PyLinearLearner(0))
    model.add_learner("sigma", PyLinearLearner(0))
    model.add_learner("nu", PyLinearLearner(0))
    model.add_learner("tau", PyLinearLearner(0))

    np.random.seed(42)
    y = np.random.normal(size=10)
    X = np.random.normal(size=(10, 2))

    model.fit(X, y)
    assert len(model.predict(X, "mu")) == 10
