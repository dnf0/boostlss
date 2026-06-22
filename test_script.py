import numpy as np
from boostlss_py import PyFamily, PyLinearLearner, BoostLssModel  # type: ignore

np.random.seed(42)
X = np.random.uniform(-3, 3, (100, 2))
y = 2.0 * X[:, 0]

family = PyFamily("GaussianLSS")
model = BoostLssModel(family, mstop=50, step_length=0.1)
model.add_learner("mu", PyLinearLearner("x", intercept=True))
model.add_learner("sigma", PyLinearLearner("x", intercept=True))

model.fit(X, y)
fi = model.feature_importance()
print(f"fi: {fi}")

grid = np.linspace(-3, 3, 10).tolist()
pd = model.partial_dependence(X, "mu", 0, grid)
print(f"pd: {pd}")
