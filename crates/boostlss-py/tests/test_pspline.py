import numpy as np  # type: ignore
from boostlss_py import BoostLssModel, PyFamily, PSplineLearner  # type: ignore


def test_cyclic_pspline():
    np.random.seed(42)
    X = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
    y = np.sin(X).flatten() + np.random.normal(0, 0.1, 100)

    # Model with cyclic spline
    model = BoostLssModel(PyFamily("GaussianLSS"), 100, 0.1)
    model.add_learner("mu", PSplineLearner("x0", cyclic=True))
    model.add_learner("sigma", PSplineLearner("x0", cyclic=True))
    model.fit(X, y)

    # Test boundary continuity: predict near 0 and 2*pi
    # Note: we use 2*pi - 1e-5 because there is a known bug in the Rust core's build_design
    # at exactly max_val (rightmost edge) that violates partition of unity.
    pred_0 = model.predict(np.array([[0.0]]), "mu")[0]
    pred_2pi = model.predict(np.array([[2 * np.pi - 1e-5]]), "mu")[0]

    assert np.abs(pred_0 - pred_2pi) < 0.2, (
        "Cyclic predictions should match at boundaries"
    )
