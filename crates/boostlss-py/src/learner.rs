use boostlss::learner::{BaseLearner, Linear, Stump, Tree};
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

#[pyclass]
#[derive(Clone)]
pub struct PyTreeLearner {
    pub feature_indices: Vec<usize>,
    pub max_depth: usize,
    pub min_samples_leaf: usize,
}

#[pymethods]
impl PyTreeLearner {
    #[new]
    #[pyo3(signature = (feature_indices, max_depth=3, min_samples_leaf=1))]
    fn new(feature_indices: Vec<usize>, max_depth: usize, min_samples_leaf: usize) -> Self {
        Self {
            feature_indices,
            max_depth,
            min_samples_leaf,
        }
    }
}

impl From<PyTreeLearner> for BaseLearner {
    fn from(val: PyTreeLearner) -> Self {
        let mut tree = Tree::new(val.feature_indices);
        tree.max_depth = val.max_depth;
        tree.min_samples_leaf = val.min_samples_leaf;
        BaseLearner::Tree(tree)
    }
}

#[pyclass(name = "PSplineLearner")]
#[derive(Clone)]
pub struct PyPSplineLearner {
    pub feature: String,
    pub degree: usize,
    pub knots: usize,
    pub differences: usize,
    pub df: f64,
    pub cyclic: bool,
}

#[pymethods]
impl PyPSplineLearner {
    #[new]
    #[pyo3(signature = (feature, degree=3, knots=20, differences=2, df=4.0, cyclic=false))]
    fn new(
        feature: String,
        degree: usize,
        knots: usize,
        differences: usize,
        df: f64,
        cyclic: bool,
    ) -> Self {
        Self {
            feature,
            degree,
            knots,
            differences,
            df,
            cyclic,
        }
    }

    #[getter]
    fn get_cyclic(&self) -> bool {
        self.cyclic
    }

    #[setter]
    fn set_cyclic(&mut self, cyclic: bool) {
        self.cyclic = cyclic;
    }
}

impl From<PyPSplineLearner> for BaseLearner {
    fn from(p: PyPSplineLearner) -> Self {
        BaseLearner::PSpline(
            boostlss::learner::PSpline::new(&p.feature)
                .with_degree(p.degree)
                .with_knots(p.knots)
                .with_differences(p.differences)
                .with_df(p.df)
                .cyclic(p.cyclic),
        )
    }
}
