import numpy as np
from boostlss_py import BoostLssModel, PyFamily, PyTreeLearner, PyHistTreeLearner


def test_categorical_tree():
    np.random.seed(42)
    # 200 samples, 2 continuous, 1 categorical (index 2)
    X = np.random.randn(200, 3)
    X[:, 2] = np.random.choice([0, 1, 2, 3], size=200)

    # Target depends explicitly on categorical value 2
    y = X[:, 0] + (X[:, 2] == 2) * 5.0 + np.random.randn(200) * 0.1

    family = PyFamily("GaussianLss")

    # Test Exact Tree
    model1 = BoostLssModel(family, mstop=10, step_length=0.1)
    learner1 = PyTreeLearner([0, 1, 2], categorical_features=[2])
    model1.add_learner("mu", learner1)
    model1.fit(X, y)
    preds1 = model1.predict(X, "mu")
    assert len(preds1) == 200

    # Test Hist Tree
    model2 = BoostLssModel(family, mstop=10, step_length=0.1)
    learner2 = PyHistTreeLearner([0, 1, 2], categorical_features=[2])
    model2.add_learner("mu", learner2)
    model2.fit(X, y)
    preds2 = model2.predict(X, "mu")
    assert len(preds2) == 200
