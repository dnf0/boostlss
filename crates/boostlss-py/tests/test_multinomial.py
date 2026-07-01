import pytest
import numpy as np
from boostlss_py import BoostLssModel, PyFamily


def test_multinomial_fit():
    np.random.seed(42)
    X = np.random.randn(100, 5)
    # 3 classes
    y = np.random.randint(0, 3, size=100).astype(float)

    family = PyFamily("Multinomial")
    model = BoostLssModel(family)
    model.fit(X, y)

    preds_0 = model.predict(X, "pi_0")
    preds_1 = model.predict(X, "pi_1")
    preds_2 = model.predict(X, "pi_2")
    assert preds_0.shape == (100,)
    assert preds_1.shape == (100,)
    assert preds_2.shape == (100,)


def test_multinomial_invalid_y():
    np.random.seed(42)
    X = np.random.randn(100, 5)
    # float values should fail check_response
    y = np.random.rand(100)

    family = PyFamily("Multinomial")
    model = BoostLssModel(family)
    with pytest.raises(Exception):
        model.fit(X, y)
