import numpy as np
from boostlss_py import BoostLssModel, PyFamily, constrained_pspline

np.random.seed(42)
x = np.sort(np.random.uniform(-3, 3, 100))
y = -x + np.random.normal(0, 0.1, 100)

model = BoostLssModel(PyFamily("GaussianLSS"), mstop=100, step_length=0.1)
model.add_learner("mu", constrained_pspline(0, "monotonic_increasing", df=2.0))

X = x.reshape(-1, 1)
model.fit(X, y)

preds = model.predict(X, "mu")
diffs = np.diff(preds)
print("min diff:", np.min(diffs))
