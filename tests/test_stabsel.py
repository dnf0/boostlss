import numpy as np
from boostlss_py import BoostLssModel, PyFamily, PyLinearLearner


def test_stabsel():
    np.random.seed(42)
    x = np.random.uniform(-3, 3, (100, 2))
    y = 2.0 * x[:, 0] + np.random.normal(0, 0.1, 100)

    model = BoostLssModel(PyFamily("GaussianLSS"), mstop=100, step_length=0.1)
    model.add_learner("mu", PyLinearLearner(0))
    model.add_learner("mu", PyLinearLearner(1))

    model.fit(x, y)

    # Expected to fail since stabsel is not yet exposed via PyO3
    res = model.stabsel(b=10, pfer=1.0, q=1, mode="joint")

    assert res.q == 1
    assert res.b == 10
    assert "Linear_0" in res.selected_joint
    assert len(res.selected_joint) > 0
