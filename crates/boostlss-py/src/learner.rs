use boostlss::learner::{BaseLearner, HistTree, Linear, Stump, Tree};
use pyo3::prelude::*;

#[pyclass(module = "boostlss_py")]
#[derive(Clone)]
pub struct PyLinearLearner {
    pub feature_idx: usize,
    pub intercept: bool,
}

#[pymethods]
impl PyLinearLearner {
    #[new]
    #[pyo3(signature = (feature_idx, intercept=true))]
    fn new(feature_idx: usize, intercept: bool) -> Self {
        Self {
            feature_idx,
            intercept,
        }
    }
}

impl From<PyLinearLearner> for BaseLearner {
    fn from(val: PyLinearLearner) -> Self {
        BaseLearner::Linear(Linear::new(val.feature_idx).intercept(val.intercept))
    }
}

#[pyclass(module = "boostlss_py")]
#[derive(Clone)]
pub struct PyStumpLearner {
    pub feature_idx: usize,
}

#[pymethods]
impl PyStumpLearner {
    #[new]
    fn new(feature_idx: usize) -> Self {
        Self { feature_idx }
    }
}

impl From<PyStumpLearner> for BaseLearner {
    fn from(val: PyStumpLearner) -> Self {
        BaseLearner::Stump(Stump::new(val.feature_idx))
    }
}

#[pyclass(module = "boostlss_py")]
#[derive(Clone)]
pub struct PyTreeLearner {
    pub feature_indices: Vec<usize>,
    pub max_depth: usize,
    pub min_samples_leaf: usize,
    pub categorical_features: Vec<usize>,
}

#[pymethods]
impl PyTreeLearner {
    #[new]
    #[pyo3(signature = (feature_indices, max_depth=3, min_samples_leaf=1, categorical_features=None))]
    fn new(
        feature_indices: Vec<usize>,
        max_depth: usize,
        min_samples_leaf: usize,
        categorical_features: Option<Vec<usize>>,
    ) -> Self {
        Self {
            feature_indices,
            max_depth,
            min_samples_leaf,
            categorical_features: categorical_features.unwrap_or_default(),
        }
    }
}

impl From<PyTreeLearner> for BaseLearner {
    fn from(val: PyTreeLearner) -> Self {
        let mut tree = Tree::new(val.feature_indices);
        tree.max_depth = val.max_depth;
        tree.min_samples_leaf = val.min_samples_leaf;
        tree.categorical_features = val.categorical_features;
        BaseLearner::Tree(tree)
    }
}

#[pyclass(module = "boostlss_py")]
#[derive(Clone)]
pub struct PyHistTreeLearner {
    pub feature_indices: Vec<usize>,
    pub max_bins: usize,
    pub max_depth: usize,
    pub min_samples_leaf: usize,
    pub categorical_features: Vec<usize>,
}

#[pymethods]
impl PyHistTreeLearner {
    #[new]
    #[pyo3(signature = (feature_indices, max_bins=256, max_depth=3, min_samples_leaf=1, categorical_features=None))]
    fn new(
        feature_indices: Vec<usize>,
        max_bins: usize,
        max_depth: usize,
        min_samples_leaf: usize,
        categorical_features: Option<Vec<usize>>,
    ) -> Self {
        Self {
            feature_indices,
            max_bins,
            max_depth,
            min_samples_leaf,
            categorical_features: categorical_features.unwrap_or_default(),
        }
    }
}

impl From<PyHistTreeLearner> for BaseLearner {
    fn from(val: PyHistTreeLearner) -> Self {
        let mut hist_tree = HistTree::new(val.feature_indices)
            .max_bins(val.max_bins)
            .max_depth(val.max_depth)
            .min_samples_leaf(val.min_samples_leaf);
        hist_tree.categorical_features = val.categorical_features;
        BaseLearner::HistTree(hist_tree)
    }
}

#[pyclass(module = "boostlss_py")]
#[derive(Clone)]
pub struct PyPSplineLearner {
    pub feature_idx: usize,
    pub degree: usize,
    pub knots: usize,
    pub differences: usize,
    pub df: f64,
    pub cyclic: bool,
}

#[pymethods]
impl PyPSplineLearner {
    #[new]
    #[pyo3(signature = (feature_idx, degree=3, knots=20, differences=2, df=4.0, cyclic=false))]
    fn new(
        feature_idx: usize,
        degree: usize,
        knots: usize,
        differences: usize,
        df: f64,
        cyclic: bool,
    ) -> Self {
        Self {
            feature_idx,
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
            boostlss::learner::PSpline::new(p.feature_idx)
                .with_degree(p.degree)
                .with_knots(p.knots)
                .with_differences(p.differences)
                .with_df(p.df)
                .cyclic(p.cyclic),
        )
    }
}

#[pyclass(module = "boostlss_py")]
#[derive(Clone)]
pub struct PyBivariatePSplineLearner {
    #[pyo3(get)]
    pub feature1_idx: usize,
    #[pyo3(get)]
    pub feature2_idx: usize,
    #[pyo3(get)]
    pub degree: usize,
    #[pyo3(get)]
    pub knots: usize,
    #[pyo3(get)]
    pub differences: usize,
    #[pyo3(get)]
    pub df: f64,
}

#[pymethods]
impl PyBivariatePSplineLearner {
    #[new]
    #[pyo3(signature = (feature1_idx, feature2_idx, degree=3, knots=20, differences=2, df=4.0))]
    fn new(
        feature1_idx: usize,
        feature2_idx: usize,
        degree: usize,
        knots: usize,
        differences: usize,
        df: f64,
    ) -> Self {
        Self {
            feature1_idx,
            feature2_idx,
            degree,
            knots,
            differences,
            df,
        }
    }
}

impl From<PyBivariatePSplineLearner> for BaseLearner {
    fn from(b: PyBivariatePSplineLearner) -> Self {
        BaseLearner::BivariatePSpline(
            boostlss::learner::bspatial::BivariatePSpline::new(b.feature1_idx, b.feature2_idx)
                .degree(b.degree)
                .knots(b.knots)
                .differences(b.differences)
                .df(b.df),
        )
    }
}

#[pyclass(module = "boostlss_py")]
#[derive(Clone)]
pub struct PyConstrainedPSplineLearner {
    pub inner: boostlss::learner::constrained_pspline::ConstrainedPSpline,
}

impl From<PyConstrainedPSplineLearner> for BaseLearner {
    fn from(val: PyConstrainedPSplineLearner) -> Self {
        BaseLearner::ConstrainedPSpline(val.inner)
    }
}

#[pyfunction]
#[pyo3(signature = (feature_idx, constraint, knots=20, degree=3, differences=2, df=4.0, max_iter=10, tolerance=1e-6))]
#[allow(clippy::too_many_arguments)]
pub fn constrained_pspline(
    feature_idx: usize,
    constraint: &str,
    knots: usize,
    degree: usize,
    differences: usize,
    df: f64,
    max_iter: usize,
    tolerance: f64,
) -> PyResult<PyConstrainedPSplineLearner> {
    let c = match constraint.to_lowercase().as_str() {
        "monotonic_increasing" => {
            boostlss::learner::constrained_pspline::Constraint::MonotonicIncreasing
        }
        "monotonic_decreasing" => {
            boostlss::learner::constrained_pspline::Constraint::MonotonicDecreasing
        }
        "convex" => boostlss::learner::constrained_pspline::Constraint::Convex,
        "concave" => boostlss::learner::constrained_pspline::Constraint::Concave,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Invalid constraint",
            ))
        }
    };

    let mut b = boostlss::learner::constrained_pspline::ConstrainedPSpline::new(feature_idx, c);
    b.knots = knots;
    b.degree = degree;
    b.differences = differences;
    b.df = df;
    b.max_iter = max_iter;
    b.tolerance = tolerance;

    Ok(PyConstrainedPSplineLearner { inner: b })
}
