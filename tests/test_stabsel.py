import numpy as np
from boostlss_py import BoostLssModel, PyFamily, PyLinearLearner


def test_stabsel_error_bounds():
    np.random.seed(42)
    x = np.random.uniform(-3, 3, (100, 2))
    y = np.random.normal(0, 0.1, 100)

    model = BoostLssModel(PyFamily("GaussianLSS"), mstop=10, step_length=0.1)
    for i in range(10):
        model.add_learner("mu", PyLinearLearner(i % 2))

    model.fit(x, y)

    # p = 10. Given q = 3, pi_thr = 0.9.
    # formula: pfer = q^2 / ((2 * pi_thr - 1) * p)
    # pfer = 9 / ((1.8 - 1) * 10) = 9 / 8 = 1.125
    res = model.stabsel(b=10, q=3, pi_thr=0.9, mode="joint")
    assert np.isclose(res.pfer, 1.125)


def test_stabsel_selects_informative_features():
    np.random.seed(42)
    x = np.random.uniform(-3, 3, (100, 5))
    # Only x[:, 0] is informative
    y = 2.0 * x[:, 0] + np.random.normal(0, 0.1, 100)

    model = BoostLssModel(PyFamily("GaussianLSS"), mstop=100, step_length=0.1)
    for i in range(5):
        model.add_learner("mu", PyLinearLearner(i))
    for i in range(5):
        model.add_learner("sigma", PyLinearLearner(i))

    model.fit(x, y)

    # We expect Linear_0 to be selected, but not the others
    res = model.stabsel(b=10, pfer=1.0, q=2, mode="joint")

    assert "Linear_0" in res.selected_joint
    assert "Linear_1" not in res.selected_joint
    assert "Linear_2" not in res.selected_joint
