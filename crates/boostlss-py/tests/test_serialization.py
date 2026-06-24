import pytest
import numpy as np
import pickle
import os
from boostlss_py import BoostLssModel, PyFamily as Family, PyLinearLearner as Linear  # type: ignore


def test_pickle_roundtrip():
    X = np.random.rand(100, 2)
    y = 2.0 * X[:, 0] + np.random.randn(100) * 0.1

    model = BoostLssModel(Family.Gaussian, mstop=10, step_length=0.1)
    model.add_learner("mu", Linear(1))
    model.fit(X, y)

    preds_before = model.predict(X, "mu")

    dumped = pickle.dumps(model)
    loaded_model = pickle.loads(dumped)

    preds_after = loaded_model.predict(X, "mu")
    np.testing.assert_allclose(preds_before, preds_after, rtol=1e-10)

    # Verify cvrisk fails gracefully because train_data is dropped
    with pytest.raises(RuntimeError):
        loaded_model.cvrisk(5)


def test_save_load(tmp_path):
    X = np.random.rand(100, 2)
    y = 2.0 * X[:, 0] + np.random.randn(100) * 0.1

    model = BoostLssModel(Family.Gaussian, mstop=10, step_length=0.1)
    model.add_learner("mu", Linear(1))
    model.fit(X, y)

    preds_before = model.predict(X, "mu")

    save_path = str(tmp_path / "model.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(model, f)

    assert os.path.exists(save_path)

    with open(save_path, "rb") as f:
        loaded_model = pickle.load(f)
    preds_after = loaded_model.predict(X, "mu")
    np.testing.assert_allclose(preds_before, preds_after, rtol=1e-10)
