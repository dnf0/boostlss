use crate::family::PyFamily;
use boostlss::cv::{CvRisk, Resampling};
use boostlss::data::Dataset;
use boostlss::engine::cyclical::fit_cyclical;
use boostlss::engine::Mstop;
use boostlss::family::{
    BetaLss, BinomialLss, GEVLss, GaussianLss, LogNormalLss, WeibullLss, ZIPLss,
};
use boostlss::learner::{BaseLearner, RandomEffects};
use boostlss::model::{BoostLss, Fitted, Scale};
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyType};

enum FittedModel {
    Gaussian(Fitted<GaussianLss>),
    Binomial(Fitted<BinomialLss>),
    Beta(Fitted<BetaLss>),
    Weibull(Fitted<WeibullLss>),
    LogNormal(Fitted<LogNormalLss>),
    ZIP(Fitted<ZIPLss>),
    GEV(Fitted<GEVLss>),
}

impl FittedModel {
    fn predict(
        &mut self,
        dataset: &Dataset,
        param: &str,
        scale: Scale,
    ) -> pyo3::PyResult<ndarray::Array1<f64>> {
        match self {
            Self::Gaussian(fitted) => fitted
                .predict(dataset, param, scale)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string())),
            Self::Binomial(fitted) => fitted
                .predict(dataset, param, scale)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string())),
            Self::Beta(fitted) => fitted
                .predict(dataset, param, scale)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string())),
            Self::Weibull(fitted) => fitted
                .predict(dataset, param, scale)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string())),
            Self::LogNormal(fitted) => fitted
                .predict(dataset, param, scale)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string())),
            Self::ZIP(fitted) => fitted
                .predict(dataset, param, scale)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string())),
            Self::GEV(fitted) => fitted
                .predict(dataset, param, scale)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string())),
        }
    }

    fn feature_importance(&self) -> Vec<f64> {
        match self {
            Self::Gaussian(fitted) => fitted.feature_importance(),
            Self::Binomial(fitted) => fitted.feature_importance(),
            Self::Beta(fitted) => fitted.feature_importance(),
            Self::Weibull(fitted) => fitted.feature_importance(),
            Self::LogNormal(fitted) => fitted.feature_importance(),
            Self::ZIP(fitted) => fitted.feature_importance(),
            Self::GEV(fitted) => fitted.feature_importance(),
        }
    }

    fn partial_dependence(
        &mut self,
        dataset: &Dataset,
        param: &str,
        feature_idx: usize,
        grid: &[f64],
    ) -> pyo3::PyResult<Vec<f64>> {
        match self {
            Self::Gaussian(fitted) => fitted
                .partial_dependence(dataset, param, feature_idx, grid)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string())),
            Self::Binomial(fitted) => fitted
                .partial_dependence(dataset, param, feature_idx, grid)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string())),
            Self::Beta(fitted) => fitted
                .partial_dependence(dataset, param, feature_idx, grid)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string())),
            Self::Weibull(fitted) => fitted
                .partial_dependence(dataset, param, feature_idx, grid)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string())),
            Self::LogNormal(fitted) => fitted
                .partial_dependence(dataset, param, feature_idx, grid)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string())),
            Self::ZIP(fitted) => fitted
                .partial_dependence(dataset, param, feature_idx, grid)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string())),
            Self::GEV(fitted) => fitted
                .partial_dependence(dataset, param, feature_idx, grid)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string())),
        }
    }
}

#[pyclass(module = "boostlss_py")]
#[derive(Clone)]
pub struct PyRandomEffectsLearner {
    feature: String,
    df: f64,
}

#[pymethods]
impl PyRandomEffectsLearner {
    #[new]
    #[pyo3(signature = (feature, df=4.0))]
    fn new(feature: &str, df: f64) -> Self {
        Self {
            feature: feature.to_string(),
            df,
        }
    }
}

#[pyclass(module = "boostlss_py")]
pub struct BoostLssModel {
    family: PyFamily,
    mstop: usize,
    step_length: f64,
    algorithm: String,
    learners: Vec<(String, BaseLearner)>,
    fitted: Option<FittedModel>,
    train_data: Option<(ndarray::Array2<f64>, ndarray::Array1<f64>)>,
}

#[pymethods]
impl BoostLssModel {
    #[new]
    #[pyo3(signature = (family, mstop=100, step_length=0.1, algorithm="cyclic"))]
    fn new(family: PyFamily, mstop: usize, step_length: f64, algorithm: &str) -> PyResult<Self> {
        if algorithm != "cyclic" && algorithm != "noncyclic" {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "algorithm must be 'cyclic' or 'noncyclic'",
            ));
        }

        Ok(Self {
            family,
            mstop,
            step_length,
            algorithm: algorithm.to_string(),
            learners: Vec::new(),
            fitted: None,
            train_data: None,
        })
    }

    fn add_learner(&mut self, param: String, learner: &Bound<'_, PyAny>) -> PyResult<()> {
        let base_learner = if let Ok(l) = learner.extract::<crate::learner::PyLinearLearner>() {
            l.into()
        } else if let Ok(s) = learner.extract::<crate::learner::PyStumpLearner>() {
            s.into()
        } else if let Ok(t) = learner.extract::<crate::learner::PyTreeLearner>() {
            t.into()
        } else if let Ok(p) = learner.extract::<crate::learner::PyPSplineLearner>() {
            p.into()
        } else if let Ok(b) = learner.extract::<crate::learner::PyBivariatePSplineLearner>() {
            b.into()
        } else if let Ok(r) = learner.extract::<PyRandomEffectsLearner>() {
            BaseLearner::RandomEffects(RandomEffects::new(&r.feature).df(r.df))
        } else {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Invalid learner type",
            ));
        };
        self.learners.push((param, base_learner));
        Ok(())
    }

    fn fit(&mut self, x: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>) -> PyResult<()> {
        let x_view = x.as_array();
        let y_view = y.as_array();
        let x_mat = ndarray::Array2::from_shape_vec(
            (x_view.nrows(), x_view.ncols()),
            x_view.to_owned().into_raw_vec(),
        )
        .unwrap();

        let y_vec =
            ndarray::Array1::from_shape_vec((y_view.len(),), y_view.to_owned().into_raw_vec())
                .unwrap();

        let dataset = Dataset::new(x_mat.clone(), y_vec.clone(), None)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        self.train_data = Some((x_mat, y_vec));

        match self.family {
            PyFamily::GaussianLss => {
                let mut model = BoostLss::new(GaussianLss::new())
                    .step_length(self.step_length)
                    .mstop(Mstop::Scalar(self.mstop));

                for (param, learner) in &self.learners {
                    model = model
                        .on(param.as_str(), |p| p.add(learner.clone()))
                        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                }

                let fitted = fit_cyclical(model, &dataset)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

                self.fitted = Some(FittedModel::Gaussian(fitted));
            }
            PyFamily::BinomialLss => {
                let mut model = BoostLss::new(BinomialLss::new())
                    .step_length(self.step_length)
                    .mstop(Mstop::Scalar(self.mstop));

                for (param, learner) in &self.learners {
                    model = model
                        .on(param.as_str(), |p| p.add(learner.clone()))
                        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                }

                let fitted = fit_cyclical(model, &dataset)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

                self.fitted = Some(FittedModel::Binomial(fitted));
            }
            PyFamily::BetaLss => {
                let mut model = BoostLss::new(BetaLss::new())
                    .step_length(self.step_length)
                    .mstop(Mstop::Scalar(self.mstop));

                for (param, learner) in &self.learners {
                    model = model
                        .on(param.as_str(), |p| p.add(learner.clone()))
                        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                }

                let fitted = fit_cyclical(model, &dataset)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

                self.fitted = Some(FittedModel::Beta(fitted));
            }
            PyFamily::WeibullLss => {
                let mut model = BoostLss::new(WeibullLss::new())
                    .step_length(self.step_length)
                    .mstop(Mstop::Scalar(self.mstop));

                for (param, learner) in &self.learners {
                    model = model
                        .on(param.as_str(), |p| p.add(learner.clone()))
                        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                }

                let fitted = fit_cyclical(model, &dataset)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

                self.fitted = Some(FittedModel::Weibull(fitted));
            }
            PyFamily::LogNormalLss => {
                let mut model = BoostLss::new(LogNormalLss::new())
                    .step_length(self.step_length)
                    .mstop(Mstop::Scalar(self.mstop));

                for (param, learner) in &self.learners {
                    model = model
                        .on(param.as_str(), |p| p.add(learner.clone()))
                        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                }

                let fitted = fit_cyclical(model, &dataset)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

                self.fitted = Some(FittedModel::LogNormal(fitted));
            }
            PyFamily::ZIPLss => {
                let mut model = BoostLss::new(ZIPLss::new())
                    .step_length(self.step_length)
                    .mstop(Mstop::Scalar(self.mstop));

                for (param, learner) in &self.learners {
                    model = model
                        .on(param.as_str(), |p| p.add(learner.clone()))
                        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                }

                let fitted = fit_cyclical(model, &dataset)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

                self.fitted = Some(FittedModel::ZIP(fitted));
            }
            PyFamily::GEVLss => {
                let mut model = BoostLss::new(GEVLss::new())
                    .step_length(self.step_length)
                    .mstop(Mstop::Scalar(self.mstop));

                for (param, learner) in &self.learners {
                    model = model
                        .on(param.as_str(), |p| p.add(learner.clone()))
                        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                }

                let fitted = fit_cyclical(model, &dataset)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

                self.fitted = Some(FittedModel::GEV(fitted));
            }
        }
        Ok(())
    }

    fn predict<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        param: &str,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x_view = x.as_array();
        let x_mat = ndarray::Array2::from_shape_vec(
            (x_view.nrows(), x_view.ncols()),
            x_view.to_owned().into_raw_vec(),
        )
        .unwrap();
        // create dummy y for Dataset constructor requirements
        let y_dummy = ndarray::Array1::zeros(x_mat.nrows());
        let dataset = Dataset::new(x_mat, y_dummy, None)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        if let Some(fitted) = &mut self.fitted {
            let pred = fitted.predict(&dataset, param, Scale::Response)?;
            let pred_vec: Vec<f64> = pred.into_iter().collect();
            Ok(PyArray1::from_vec_bound(py, pred_vec))
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Model not fitted",
            ))
        }
    }

    #[pyo3(signature = (folds=10))]
    fn cvrisk<'py>(&mut self, py: Python<'py>, folds: usize) -> PyResult<Bound<'py, PyDict>> {
        if let Some((x_mat, y_vec)) = &self.train_data {
            let dataset = Dataset::new(x_mat.clone(), y_vec.clone(), None)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

            match self.family {
                PyFamily::GaussianLss => {
                    let mut model = BoostLss::new(GaussianLss::new())
                        .step_length(self.step_length)
                        .mstop(Mstop::Scalar(self.mstop));

                    for (param, learner) in &self.learners {
                        model = model
                            .on(param.as_str(), |p| p.add(learner.clone()))
                            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                    }

                    let cv = CvRisk::new(model, Resampling::KFold { k: folds });
                    let result = cv
                        .run(&dataset)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

                    let dict = PyDict::new_bound(py);
                    match result.optimal_mstop {
                        Mstop::Scalar(m) => dict.set_item("optimal_mstop", m)?,
                        Mstop::PerParam(v) => dict.set_item("optimal_mstop", v)?,
                    }
                    dict.set_item("mean_risk", result.mean_risk)?;
                    Ok(dict)
                }
                PyFamily::BinomialLss => {
                    let mut model = BoostLss::new(BinomialLss::new())
                        .step_length(self.step_length)
                        .mstop(Mstop::Scalar(self.mstop));

                    for (param, learner) in &self.learners {
                        model = model
                            .on(param.as_str(), |p| p.add(learner.clone()))
                            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                    }

                    let cv = CvRisk::new(model, Resampling::KFold { k: folds });
                    let result = cv
                        .run(&dataset)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

                    let dict = PyDict::new_bound(py);
                    match result.optimal_mstop {
                        Mstop::Scalar(m) => dict.set_item("optimal_mstop", m)?,
                        Mstop::PerParam(v) => dict.set_item("optimal_mstop", v)?,
                    }
                    dict.set_item("mean_risk", result.mean_risk)?;
                    Ok(dict)
                }
                PyFamily::BetaLss => {
                    let mut model = BoostLss::new(BetaLss::new())
                        .step_length(self.step_length)
                        .mstop(Mstop::Scalar(self.mstop));

                    for (param, learner) in &self.learners {
                        model = model
                            .on(param.as_str(), |p| p.add(learner.clone()))
                            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                    }

                    let cv = CvRisk::new(model, Resampling::KFold { k: folds });
                    let result = cv
                        .run(&dataset)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

                    let dict = PyDict::new_bound(py);
                    match result.optimal_mstop {
                        Mstop::Scalar(m) => dict.set_item("optimal_mstop", m)?,
                        Mstop::PerParam(v) => dict.set_item("optimal_mstop", v)?,
                    }
                    dict.set_item("mean_risk", result.mean_risk)?;
                    Ok(dict)
                }
                PyFamily::WeibullLss => {
                    let mut model = BoostLss::new(WeibullLss::new())
                        .step_length(self.step_length)
                        .mstop(Mstop::Scalar(self.mstop));

                    for (param, learner) in &self.learners {
                        model = model
                            .on(param.as_str(), |p| p.add(learner.clone()))
                            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                    }

                    let cv = CvRisk::new(model, Resampling::KFold { k: folds });
                    let result = cv
                        .run(&dataset)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

                    let dict = PyDict::new_bound(py);
                    match result.optimal_mstop {
                        Mstop::Scalar(m) => dict.set_item("optimal_mstop", m)?,
                        Mstop::PerParam(v) => dict.set_item("optimal_mstop", v)?,
                    }
                    dict.set_item("mean_risk", result.mean_risk)?;
                    Ok(dict)
                }
                PyFamily::LogNormalLss => {
                    let mut model = BoostLss::new(LogNormalLss::new())
                        .step_length(self.step_length)
                        .mstop(Mstop::Scalar(self.mstop));

                    for (param, learner) in &self.learners {
                        model = model
                            .on(param.as_str(), |p| p.add(learner.clone()))
                            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                    }

                    let cv = CvRisk::new(model, Resampling::KFold { k: folds });
                    let result = cv
                        .run(&dataset)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

                    let dict = PyDict::new_bound(py);
                    match result.optimal_mstop {
                        Mstop::Scalar(m) => dict.set_item("optimal_mstop", m)?,
                        Mstop::PerParam(v) => dict.set_item("optimal_mstop", v)?,
                    }
                    dict.set_item("mean_risk", result.mean_risk)?;
                    Ok(dict)
                }
                PyFamily::ZIPLss => {
                    let mut model = BoostLss::new(ZIPLss::new())
                        .step_length(self.step_length)
                        .mstop(Mstop::Scalar(self.mstop));

                    for (param, learner) in &self.learners {
                        model = model
                            .on(param.as_str(), |p| p.add(learner.clone()))
                            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                    }

                    let cv = CvRisk::new(model, Resampling::KFold { k: folds });
                    let result = cv
                        .run(&dataset)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

                    let dict = PyDict::new_bound(py);
                    match result.optimal_mstop {
                        Mstop::Scalar(m) => dict.set_item("optimal_mstop", m)?,
                        Mstop::PerParam(v) => dict.set_item("optimal_mstop", v)?,
                    }
                    dict.set_item("mean_risk", result.mean_risk)?;
                    Ok(dict)
                }
                PyFamily::GEVLss => {
                    let mut model = BoostLss::new(GEVLss::new())
                        .step_length(self.step_length)
                        .mstop(Mstop::Scalar(self.mstop));

                    for (param, learner) in &self.learners {
                        model = model
                            .on(param.as_str(), |p| p.add(learner.clone()))
                            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                    }

                    let cv = CvRisk::new(model, Resampling::KFold { k: folds });
                    let result = cv
                        .run(&dataset)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

                    let dict = PyDict::new_bound(py);
                    match result.optimal_mstop {
                        Mstop::Scalar(m) => dict.set_item("optimal_mstop", m)?,
                        Mstop::PerParam(v) => dict.set_item("optimal_mstop", v)?,
                    }
                    dict.set_item("mean_risk", result.mean_risk)?;
                    Ok(dict)
                }
            }
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Model not fitted, cannot run cvrisk without data",
            ))
        }
    }

    /// Returns the feature importance (empirical risk reduction) for each base learner.
    pub fn feature_importance(&self) -> PyResult<Vec<f64>> {
        if let Some(fitted) = &self.fitted {
            Ok(fitted.feature_importance())
        } else {
            Err(pyo3::exceptions::PyValueError::new_err(
                "Model must be fitted before calling feature_importance",
            ))
        }
    }

    /// Computes Friedman's partial dependence for a specific feature across a grid of values.
    pub fn partial_dependence<'py>(
        &mut self,
        _py: Python<'py>,
        data: PyReadonlyArray2<f64>,
        param: &str,
        feature_idx: usize,
        grid: Vec<f64>,
    ) -> PyResult<Vec<f64>> {
        let x_view = data.as_array();
        let x_mat = ndarray::Array2::from_shape_vec(
            (x_view.nrows(), x_view.ncols()),
            x_view.to_owned().into_raw_vec(),
        )
        .unwrap();

        let n_samples = x_mat.nrows();
        let dummy_response = ndarray::Array1::<f64>::zeros(n_samples);

        let ds = Dataset::new(x_mat, dummy_response, None)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        if let Some(fitted) = &mut self.fitted {
            fitted.partial_dependence(&ds, param, feature_idx, &grid)
        } else {
            Err(pyo3::exceptions::PyValueError::new_err(
                "Model must be fitted before calling partial_dependence",
            ))
        }
    }

    fn __getnewargs__(&self) -> (PyFamily, usize, f64, String) {
        (self.family.clone(), self.mstop, self.step_length, self.algorithm.clone())
    }

    fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new_bound(py);

        let family_str = match self.family {
            PyFamily::GaussianLss => "GaussianLss",
            PyFamily::BinomialLss => "BinomialLss",
            PyFamily::BetaLss => "BetaLss",
            PyFamily::WeibullLss => "WeibullLss",
            PyFamily::LogNormalLss => "LogNormalLss",
            PyFamily::ZIPLss => "ZIPLss",
            PyFamily::GEVLss => "GEVLss",
        };
        dict.set_item("family", family_str)?;
        dict.set_item("mstop", self.mstop)?;
        dict.set_item("step_length", self.step_length)?;
        dict.set_item("algorithm", self.algorithm.clone())?;
        // Skip train_data entirely!

        if let Some(fitted) = &self.fitted {
            let bytes = match fitted {
                FittedModel::Gaussian(f) => bincode::serialize(f),
                FittedModel::Binomial(f) => bincode::serialize(f),
                FittedModel::Beta(f) => bincode::serialize(f),
                FittedModel::Weibull(f) => bincode::serialize(f),
                FittedModel::LogNormal(f) => bincode::serialize(f),
                FittedModel::ZIP(f) => bincode::serialize(f),
                FittedModel::GEV(f) => bincode::serialize(f),
            }
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            dict.set_item("fitted", PyBytes::new_bound(py, &bytes))?;
        }

        // Serialize learners
        let learners_bytes = bincode::serialize(&self.learners)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        dict.set_item("learners", PyBytes::new_bound(py, &learners_bytes))?;

        Ok(dict)
    }

    fn __setstate__(&mut self, state: &Bound<'_, PyDict>) -> PyResult<()> {
        let family_str: String = state
            .get_item("family")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("Missing key 'family'"))?
            .extract()?;
        self.family = match family_str.as_str() {
            "GaussianLss" => PyFamily::GaussianLss,
            "BinomialLss" => PyFamily::BinomialLss,
            "BetaLss" => PyFamily::BetaLss,
            "WeibullLss" => PyFamily::WeibullLss,
            "LogNormalLss" => PyFamily::LogNormalLss,
            "ZIPLss" => PyFamily::ZIPLss,
            "GEVLss" => PyFamily::GEVLss,
            _ => return Err(pyo3::exceptions::PyValueError::new_err("Unknown family")),
        };
        self.mstop = state
            .get_item("mstop")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("Missing key 'mstop'"))?
            .extract()?;
        self.step_length = state
            .get_item("step_length")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("Missing key 'step_length'"))?
            .extract()?;
        self.algorithm = if let Some(algo_any) = state.get_item("algorithm")? {
            algo_any.extract()?
        } else {
            "cyclic".to_string()
        };

        if let Some(bytes_any) = state.get_item("fitted")? {
            let bytes: &[u8] = bytes_any.extract()?;
            self.fitted = Some(match self.family {
                PyFamily::GaussianLss => FittedModel::Gaussian(
                    bincode::deserialize(bytes)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
                ),
                PyFamily::BinomialLss => FittedModel::Binomial(
                    bincode::deserialize(bytes)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
                ),
                PyFamily::BetaLss => FittedModel::Beta(
                    bincode::deserialize(bytes)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
                ),
                PyFamily::WeibullLss => FittedModel::Weibull(
                    bincode::deserialize(bytes)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
                ),
                PyFamily::LogNormalLss => FittedModel::LogNormal(
                    bincode::deserialize(bytes)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
                ),
                PyFamily::ZIPLss => FittedModel::ZIP(
                    bincode::deserialize(bytes)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
                ),
                PyFamily::GEVLss => FittedModel::GEV(
                    bincode::deserialize(bytes)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
                ),
            });
        } else if let Some(bytes_any) = state.get_item("fitted_gaussian")? {
            let bytes: &[u8] = bytes_any.extract()?;
            self.fitted = Some(FittedModel::Gaussian(
                bincode::deserialize(bytes)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
            ));
        } else if let Some(bytes_any) = state.get_item("fitted_binomial")? {
            let bytes: &[u8] = bytes_any.extract()?;
            self.fitted = Some(FittedModel::Binomial(
                bincode::deserialize(bytes)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
            ));
        } else {
            self.fitted = None;
        }

        if let Some(learners_any) = state.get_item("learners")? {
            let bytes: &[u8] = learners_any.extract()?;
            let learners: Vec<(String, BaseLearner)> = bincode::deserialize(bytes)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            self.learners = learners;
        }

        self.train_data = None; // Reset train_data
        Ok(())
    }

    fn save(&self, py: Python<'_>, path: &str) -> PyResult<()> {
        let state = self.__getstate__(py)?;
        let json_str =
            serde_json::to_string(&state.to_string()) // Simplified, normally wouldn't just be to_string
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        std::fs::write(path, json_str)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(())
    }

    #[classmethod]
    fn load(_cls: &Bound<'_, PyType>, _py: Python<'_>, _path: &str) -> PyResult<Self> {
        // Need to pass py to getstate
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Load unimplemented. Use pickle instead.",
        ))
    }
}
