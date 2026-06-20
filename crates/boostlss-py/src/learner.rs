use boostlss::learner::{BaseLearner, Linear, Stump};
use pyo3::prelude::*;

#[pyclass]
#[derive(Clone)]
pub struct PyLinearLearner {
    pub name: String,
    pub intercept: bool,
}

#[pymethods]
impl PyLinearLearner {
    #[new]
    #[pyo3(signature = (name, intercept=true))]
    fn new(name: String, intercept: bool) -> Self {
        Self { name, intercept }
    }
}

impl From<PyLinearLearner> for BaseLearner {
    fn from(val: PyLinearLearner) -> Self {
        BaseLearner::Linear(Linear::new(&val.name).intercept(val.intercept))
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyStumpLearner {
    pub name: String,
}

#[pymethods]
impl PyStumpLearner {
    #[new]
    fn new(name: String) -> Self {
        Self { name }
    }
}

impl From<PyStumpLearner> for BaseLearner {
    fn from(val: PyStumpLearner) -> Self {
        BaseLearner::Stump(Stump::new(&val.name))
    }
}
