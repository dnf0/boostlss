import numpy as np
from boostlss_py import BoostLssModel, PyFamily, PyLinearLearner


def test_early_stopping():
    np.random.seed(42)
    X = np.random.randn(200, 5)
    y = 2.0 * X[:, 0] + np.random.randn(200) * 0.1

    X_train, y_train = X[:100], y[:100]
    X_val, y_val = X[100:], y[100:]

    family = PyFamily("GaussianLss")
    model = BoostLssModel(family, mstop=1000, step_length=0.1)
    model.add_learner("mu", PyLinearLearner(0, intercept=True))
    model.add_learner("sigma", PyLinearLearner(0, intercept=True))

    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=10)

    assert model.best_iteration_ < 1000
    evals = model.evals_result_
    assert "train" in evals
    assert "valid" in evals
    assert len(evals["train"]["loss"]) == model.best_iteration_
    assert len(evals["valid"]["loss"]) == model.best_iteration_
