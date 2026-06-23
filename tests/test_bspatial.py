import numpy as np
from boostlss_py import BoostLssModel, PyFamily, PyBivariatePSplineLearner


def test_bspatial_init():
    learner = PyBivariatePSplineLearner(0, 1, knots=15, degree=4, differences=1, df=5.0)
    assert learner.feature1_idx == 0
    assert learner.feature2_idx == 1
    assert learner.knots == 15
    assert learner.degree == 4
    assert learner.differences == 1
    assert learner.df == 5.0


def test_bspatial_fit_predict():
    np.random.seed(42)
    x1 = np.linspace(0, 1, 100)
    x2 = np.linspace(0, 1, 100)
    # create grid
    xx1, xx2 = np.meshgrid(x1, x2)
    X = np.column_stack([xx1.ravel(), xx2.ravel()])
    # response
    y = np.sin(2 * np.pi * X[:, 0]) * np.cos(2 * np.pi * X[:, 1]) + np.random.normal(
        0, 0.1, size=10000
    )

    model = BoostLssModel(PyFamily.GaussianLss, mstop=5, step_length=0.1)
    learner = PyBivariatePSplineLearner(0, 1, knots=10)
    model.add_learner("mu", learner)

    # Train for a few iterations
    model.fit(X, y)

    # Predict
    preds = model.predict(X, "mu")
    assert preds.shape == (10000,)
    assert not np.any(np.isnan(preds))
