import numpy as np
import scipy.sparse as sp
from boostlss_py import PyFamily, PyLinearLearner, BoostLssModel


def test_sparse_matrix_support():
    np.random.seed(42)

    # Generate some dummy data
    n_samples = 100
    n_features = 5
    X_dense = np.random.normal(size=(n_samples, n_features))

    # Make it sparse
    X_dense[X_dense < 0] = 0

    # Create CSR and CSC variants
    X_csr = sp.csr_matrix(X_dense)
    X_csc = sp.csc_matrix(X_dense)

    # Target
    y = np.random.normal(size=n_samples)

    family = PyFamily("GaussianLSS")

    # Train dense model
    model_dense = BoostLssModel(family, mstop=10, step_length=0.1)
    for i in range(n_features):
        model_dense.add_learner("mu", PyLinearLearner(i, intercept=True))
        model_dense.add_learner("sigma", PyLinearLearner(i, intercept=True))
    model_dense.fit(X_dense, y)
    preds_dense_mu = model_dense.predict(X_dense, "mu")
    preds_dense_sigma = model_dense.predict(X_dense, "sigma")

    # Train CSR model
    model_csr = BoostLssModel(family, mstop=10, step_length=0.1)
    for i in range(n_features):
        model_csr.add_learner("mu", PyLinearLearner(i, intercept=True))
        model_csr.add_learner("sigma", PyLinearLearner(i, intercept=True))
    model_csr.fit(X_csr, y)
    preds_csr_mu = model_csr.predict(X_csr, "mu")
    preds_csr_sigma = model_csr.predict(X_csr, "sigma")

    # Train CSC model
    model_csc = BoostLssModel(family, mstop=10, step_length=0.1)
    for i in range(n_features):
        model_csc.add_learner("mu", PyLinearLearner(i, intercept=True))
        model_csc.add_learner("sigma", PyLinearLearner(i, intercept=True))
    model_csc.fit(X_csc, y)
    preds_csc_mu = model_csc.predict(X_csc, "mu")
    preds_csc_sigma = model_csc.predict(X_csc, "sigma")

    # Assert predictions are identical across formats
    np.testing.assert_allclose(preds_dense_mu, preds_csr_mu, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(preds_dense_mu, preds_csc_mu, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(preds_dense_sigma, preds_csr_sigma, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(preds_dense_sigma, preds_csc_sigma, rtol=1e-5, atol=1e-5)
