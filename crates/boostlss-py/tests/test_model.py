import numpy as np
import scipy.sparse as sp
from boostlss_py import PyFamily, PyLinearLearner, BoostLssModel
import pytest


@pytest.mark.parametrize("matrix_format", ["dense", "csr", "csc"])
def test_sparse_matrix_support(matrix_format):
    np.random.seed(42)

    # Generate some dummy data
    n_samples = 100
    n_features = 5
    X_dense = np.random.normal(size=(n_samples, n_features))

    # Make it sparse
    X_dense[X_dense < 0] = 0

    # Create variant
    if matrix_format == "dense":
        X = X_dense
    elif matrix_format == "csr":
        X = sp.csr_matrix(X_dense)
    elif matrix_format == "csc":
        X = sp.csc_matrix(X_dense)

    # Target
    y = np.random.normal(size=n_samples)

    family = PyFamily("GaussianLSS")

    # Train dense baseline model for comparison
    model_baseline = BoostLssModel(family, mstop=10, step_length=0.1)
    for i in range(n_features):
        model_baseline.add_learner("mu", PyLinearLearner(i, intercept=True))
        model_baseline.add_learner("sigma", PyLinearLearner(i, intercept=True))
    model_baseline.fit(X_dense, y)
    preds_baseline_mu = model_baseline.predict(X_dense, "mu")
    preds_baseline_sigma = model_baseline.predict(X_dense, "sigma")

    # Train test model
    model = BoostLssModel(family, mstop=10, step_length=0.1)
    for i in range(n_features):
        model.add_learner("mu", PyLinearLearner(i, intercept=True))
        model.add_learner("sigma", PyLinearLearner(i, intercept=True))
    model.fit(X, y)
    preds_mu = model.predict(X, "mu")
    preds_sigma = model.predict(X, "sigma")

    # Assert predictions are identical to baseline across formats
    np.testing.assert_allclose(preds_baseline_mu, preds_mu, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(preds_baseline_sigma, preds_sigma, rtol=1e-5, atol=1e-5)
