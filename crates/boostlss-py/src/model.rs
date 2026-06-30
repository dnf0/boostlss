use crate::family::{PyFamily, PyTweedieLss, PyZINBLss};
use boostlss::cv::{CvRisk, Resampling};
use boostlss::data::Dataset;
use boostlss::engine::cyclical::fit_cyclical;
use boostlss::engine::noncyclical::{fit_noncyclical, fit_noncyclical_outer};
use boostlss::engine::{Algorithm, Mstop};
use boostlss::family::{
    BetaLss, BinomialLss, GEVLss, GaussianLss, JSULss, LogNormalLss, WeibullLss, ZIPLss,
};
use boostlss::family::{TweedieLss, ZINBLss};
use boostlss::learner::{BaseLearner, RandomEffects};
use boostlss::model::{BoostLss, Fitted, Scale};
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyType};

#[derive(Clone)]
pub enum InternalFamily {
    Gaussian,
    Binomial,
    Beta,
    Weibull,
    LogNormal,
    Zip,
    Gev,
    Jsu,
    Tweedie(TweedieLss),
    Zinb(ZINBLss),
}

enum FittedModel {
    Gaussian(Fitted<GaussianLss>),
    Binomial(Fitted<BinomialLss>),
    Beta(Fitted<BetaLss>),
    Weibull(Fitted<WeibullLss>),
    LogNormal(Fitted<LogNormalLss>),
    Zip(Fitted<ZIPLss>),
    Gev(Fitted<GEVLss>),
    Jsu(Fitted<JSULss>),
    Tweedie(Fitted<TweedieLss>),
    Zinb(Fitted<ZINBLss>),
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
            Self::Zip(fitted) => fitted
                .predict(dataset, param, scale)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string())),
            Self::Gev(fitted) => fitted
                .predict(dataset, param, scale)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string())),
            Self::Jsu(fitted) => fitted
                .predict(dataset, param, scale)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string())),
            Self::Tweedie(fitted) => fitted
                .predict(dataset, param, scale)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string())),
            Self::Zinb(fitted) => fitted
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
            Self::Zip(fitted) => fitted.feature_importance(),
            Self::Gev(fitted) => fitted.feature_importance(),
            Self::Jsu(fitted) => fitted.feature_importance(),
            Self::Tweedie(fitted) => fitted.feature_importance(),
            Self::Zinb(fitted) => fitted.feature_importance(),
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
            Self::Zip(fitted) => fitted
                .partial_dependence(dataset, param, feature_idx, grid)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string())),
            Self::Gev(fitted) => fitted
                .partial_dependence(dataset, param, feature_idx, grid)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string())),
            Self::Jsu(fitted) => fitted
                .partial_dependence(dataset, param, feature_idx, grid)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string())),
            Self::Tweedie(fitted) => fitted
                .partial_dependence(dataset, param, feature_idx, grid)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string())),
            Self::Zinb(fitted) => fitted
                .partial_dependence(dataset, param, feature_idx, grid)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string())),
        }
    }
}

#[pyclass(module = "boostlss_py")]
#[derive(Clone)]
pub struct PyRandomEffectsLearner {
    feature_idx: usize,
    df: f64,
}

#[pymethods]
impl PyRandomEffectsLearner {
    #[new]
    #[pyo3(signature = (feature_idx, df=4.0))]
    fn new(feature_idx: usize, df: f64) -> Self {
        Self { feature_idx, df }
    }
}

#[pyclass(module = "boostlss_py")]
pub struct BoostLssModel {
    family: InternalFamily,
    mstop: usize,
    step_length: f64,
    algorithm: Algorithm,
    learners: Vec<(String, BaseLearner)>,
    fitted: Option<FittedModel>,
    train_data: Option<Dataset>,
}

#[pymethods]
impl BoostLssModel {
    #[new]
    #[pyo3(signature = (family, mstop=100, step_length=0.1, algorithm="cyclic"))]
    fn new(
        family: &Bound<'_, PyAny>,
        mstop: usize,
        step_length: f64,
        algorithm: &str,
    ) -> PyResult<Self> {
        let algorithm_enum = match algorithm {
            "cyclic" => Algorithm::Cyclic,
            "noncyclic" => Algorithm::NonCyclic,
            "noncyclic_outer" => Algorithm::NonCyclicOuter,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "algorithm must be 'cyclic', 'noncyclic', or 'noncyclic_outer'",
                ))
            }
        };

        let family_enum = if let Ok(f) = family.extract::<PyFamily>() {
            match f {
                PyFamily::Gaussian => InternalFamily::Gaussian,
                PyFamily::Binomial => InternalFamily::Binomial,
                PyFamily::Beta => InternalFamily::Beta,
                PyFamily::Weibull => InternalFamily::Weibull,
                PyFamily::LogNormal => InternalFamily::LogNormal,
                PyFamily::Zip => InternalFamily::Zip,
                PyFamily::Gev => InternalFamily::Gev,
                PyFamily::Jsu => InternalFamily::Jsu,
            }
        } else if let Ok(t) = family.extract::<PyTweedieLss>() {
            InternalFamily::Tweedie(t.inner.clone())
        } else if let Ok(z) = family.extract::<PyZINBLss>() {
            InternalFamily::Zinb(z.inner.clone())
        } else {
            return Err(pyo3::exceptions::PyValueError::new_err("Invalid family"));
        };

        Ok(Self {
            family: family_enum,
            mstop,
            step_length,
            algorithm: algorithm_enum,
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
        } else if let Ok(ht) = learner.extract::<crate::learner::PyHistTreeLearner>() {
            ht.into()
        } else if let Ok(p) = learner.extract::<crate::learner::PyPSplineLearner>() {
            p.into()
        } else if let Ok(b) = learner.extract::<crate::learner::PyBivariatePSplineLearner>() {
            b.into()
        } else if let Ok(c) = learner.extract::<crate::learner::PyConstrainedPSplineLearner>() {
            c.into()
        } else if let Ok(r) = learner.extract::<PyRandomEffectsLearner>() {
            BaseLearner::RandomEffects(RandomEffects::new(r.feature_idx).df(r.df))
        } else {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Invalid learner type",
            ));
        };
        self.learners.push((param, base_learner));
        Ok(())
    }

    #[pyo3(signature = (x, y))]
    fn fit<'py>(
        &mut self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<()> {
        let y_view = y.as_array();
        let y_vec =
            ndarray::Array1::from_shape_vec((y_view.len(),), y_view.to_owned().into_raw_vec())
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let dataset = crate::data::extract_dataset(py, x, Some(y_vec))?;

        self.train_data = Some(dataset.clone());

        match &self.family {
            InternalFamily::Gaussian => {
                let mut model = BoostLss::new(GaussianLss::new())
                    .step_length(self.step_length)
                    .mstop(Mstop::Scalar(self.mstop));

                for (param, learner) in &self.learners {
                    model = model
                        .on(param.as_str(), |p| p.add(learner.clone()))
                        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                }

                let fitted = match self.algorithm {
                    Algorithm::NonCyclicOuter => {
                        let model = model.algorithm(Algorithm::NonCyclicOuter);
                        fit_noncyclical_outer(model, &dataset)
                            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
                    }
                    Algorithm::Cyclic => fit_cyclical(model, &dataset)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,

                    Algorithm::NonCyclic => {
                        let model = model.algorithm(Algorithm::NonCyclic);
                        fit_noncyclical(model, &dataset)
                            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
                    }
                };

                self.fitted = Some(FittedModel::Gaussian(fitted));
            }
            InternalFamily::Binomial => {
                let mut model = BoostLss::new(BinomialLss::new())
                    .step_length(self.step_length)
                    .mstop(Mstop::Scalar(self.mstop));

                for (param, learner) in &self.learners {
                    model = model
                        .on(param.as_str(), |p| p.add(learner.clone()))
                        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                }

                let fitted = match self.algorithm {
                    Algorithm::NonCyclicOuter => {
                        let model = model.algorithm(Algorithm::NonCyclicOuter);
                        fit_noncyclical_outer(model, &dataset)
                            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
                    }
                    Algorithm::Cyclic => fit_cyclical(model, &dataset)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,

                    Algorithm::NonCyclic => {
                        let model = model.algorithm(Algorithm::NonCyclic);
                        fit_noncyclical(model, &dataset)
                            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
                    }
                };

                self.fitted = Some(FittedModel::Binomial(fitted));
            }
            InternalFamily::Beta => {
                let mut model = BoostLss::new(BetaLss::new())
                    .step_length(self.step_length)
                    .mstop(Mstop::Scalar(self.mstop));

                for (param, learner) in &self.learners {
                    model = model
                        .on(param.as_str(), |p| p.add(learner.clone()))
                        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                }

                let fitted = match self.algorithm {
                    Algorithm::NonCyclicOuter => {
                        let model = model.algorithm(Algorithm::NonCyclicOuter);
                        fit_noncyclical_outer(model, &dataset)
                            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
                    }
                    Algorithm::Cyclic => fit_cyclical(model, &dataset)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,

                    Algorithm::NonCyclic => {
                        let model = model.algorithm(Algorithm::NonCyclic);
                        fit_noncyclical(model, &dataset)
                            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
                    }
                };

                self.fitted = Some(FittedModel::Beta(fitted));
            }
            InternalFamily::Weibull => {
                let mut model = BoostLss::new(WeibullLss::new())
                    .step_length(self.step_length)
                    .mstop(Mstop::Scalar(self.mstop));

                for (param, learner) in &self.learners {
                    model = model
                        .on(param.as_str(), |p| p.add(learner.clone()))
                        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                }

                let fitted = match self.algorithm {
                    Algorithm::NonCyclicOuter => {
                        let model = model.algorithm(Algorithm::NonCyclicOuter);
                        fit_noncyclical_outer(model, &dataset)
                            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
                    }
                    Algorithm::Cyclic => fit_cyclical(model, &dataset)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,

                    Algorithm::NonCyclic => {
                        let model = model.algorithm(Algorithm::NonCyclic);
                        fit_noncyclical(model, &dataset)
                            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
                    }
                };

                self.fitted = Some(FittedModel::Weibull(fitted));
            }
            InternalFamily::LogNormal => {
                let mut model = BoostLss::new(LogNormalLss::new())
                    .step_length(self.step_length)
                    .mstop(Mstop::Scalar(self.mstop));

                for (param, learner) in &self.learners {
                    model = model
                        .on(param.as_str(), |p| p.add(learner.clone()))
                        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                }

                let fitted = match self.algorithm {
                    Algorithm::NonCyclicOuter => {
                        let model = model.algorithm(Algorithm::NonCyclicOuter);
                        fit_noncyclical_outer(model, &dataset)
                            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
                    }
                    Algorithm::Cyclic => fit_cyclical(model, &dataset)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,

                    Algorithm::NonCyclic => {
                        let model = model.algorithm(Algorithm::NonCyclic);
                        fit_noncyclical(model, &dataset)
                            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
                    }
                };

                self.fitted = Some(FittedModel::LogNormal(fitted));
            }
            InternalFamily::Zip => {
                let mut model = BoostLss::new(ZIPLss::new())
                    .step_length(self.step_length)
                    .mstop(Mstop::Scalar(self.mstop));

                for (param, learner) in &self.learners {
                    model = model
                        .on(param.as_str(), |p| p.add(learner.clone()))
                        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                }

                let fitted = match self.algorithm {
                    Algorithm::NonCyclicOuter => {
                        let model = model.algorithm(Algorithm::NonCyclicOuter);
                        fit_noncyclical_outer(model, &dataset)
                            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
                    }
                    Algorithm::Cyclic => fit_cyclical(model, &dataset)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,

                    Algorithm::NonCyclic => {
                        let model = model.algorithm(Algorithm::NonCyclic);
                        fit_noncyclical(model, &dataset)
                            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
                    }
                };

                self.fitted = Some(FittedModel::Zip(fitted));
            }
            InternalFamily::Gev => {
                let mut model = BoostLss::new(GEVLss::new())
                    .step_length(self.step_length)
                    .mstop(Mstop::Scalar(self.mstop));

                for (param, learner) in &self.learners {
                    model = model
                        .on(param.as_str(), |p| p.add(learner.clone()))
                        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                }

                let fitted = match self.algorithm {
                    Algorithm::NonCyclicOuter => {
                        let model = model.algorithm(Algorithm::NonCyclicOuter);
                        fit_noncyclical_outer(model, &dataset)
                            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
                    }
                    Algorithm::Cyclic => fit_cyclical(model, &dataset)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,

                    Algorithm::NonCyclic => {
                        let model = model.algorithm(Algorithm::NonCyclic);
                        fit_noncyclical(model, &dataset)
                            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
                    }
                };

                self.fitted = Some(FittedModel::Gev(fitted));
            }
            InternalFamily::Jsu => {
                let mut model = BoostLss::new(JSULss::new())
                    .step_length(self.step_length)
                    .mstop(Mstop::Scalar(self.mstop));

                for (param, learner) in &self.learners {
                    model = model
                        .on(param.as_str(), |p| p.add(learner.clone()))
                        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                }

                let fitted = match self.algorithm {
                    Algorithm::NonCyclicOuter => {
                        let model = model.algorithm(Algorithm::NonCyclicOuter);
                        fit_noncyclical_outer(model, &dataset)
                            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
                    }
                    Algorithm::Cyclic => fit_cyclical(model, &dataset)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,

                    Algorithm::NonCyclic => {
                        let model = model.algorithm(Algorithm::NonCyclic);
                        fit_noncyclical(model, &dataset)
                            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
                    }
                };

                self.fitted = Some(FittedModel::Jsu(fitted));
            }
            InternalFamily::Tweedie(ref t_fam) => {
                let mut model = BoostLss::new(t_fam.clone())
                    .step_length(self.step_length)
                    .mstop(Mstop::Scalar(self.mstop));

                for (param, learner) in &self.learners {
                    model = model
                        .on(param.as_str(), |p| p.add(learner.clone()))
                        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                }

                let fitted = match self.algorithm {
                    Algorithm::NonCyclicOuter => {
                        let model = model.algorithm(Algorithm::NonCyclicOuter);
                        fit_noncyclical_outer(model, &dataset)
                            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
                    }
                    Algorithm::Cyclic => fit_cyclical(model, &dataset)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,

                    Algorithm::NonCyclic => {
                        let model = model.algorithm(Algorithm::NonCyclic);
                        fit_noncyclical(model, &dataset)
                            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
                    }
                };

                self.fitted = Some(FittedModel::Tweedie(fitted));
            }
            InternalFamily::Zinb(ref z_fam) => {
                let mut model = BoostLss::new(z_fam.clone())
                    .step_length(self.step_length)
                    .mstop(Mstop::Scalar(self.mstop));

                for (param, learner) in &self.learners {
                    model = model
                        .on(param.as_str(), |p| p.add(learner.clone()))
                        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                }

                let fitted = match self.algorithm {
                    Algorithm::NonCyclicOuter => {
                        let model = model.algorithm(Algorithm::NonCyclicOuter);
                        fit_noncyclical_outer(model, &dataset)
                            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
                    }
                    Algorithm::Cyclic => fit_cyclical(model, &dataset)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,

                    Algorithm::NonCyclic => {
                        let model = model.algorithm(Algorithm::NonCyclic);
                        fit_noncyclical(model, &dataset)
                            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
                    }
                };

                self.fitted = Some(FittedModel::Zinb(fitted));
            }
        }
        Ok(())
    }

    #[pyo3(signature = (x, param))]
    fn predict<'py>(
        &mut self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
        param: &str,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let dataset = crate::data::extract_dataset(py, x, None)?;

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
        if let Some(dataset) = &self.train_data {
            let dataset = dataset.clone();

            match &self.family {
                InternalFamily::Gaussian => {
                    let mut model = BoostLss::new(GaussianLss::new())
                        .step_length(self.step_length)
                        .mstop(Mstop::Scalar(self.mstop));

                    for (param, learner) in &self.learners {
                        model = model
                            .on(param.as_str(), |p| p.add(learner.clone()))
                            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                    }

                    let model = match self.algorithm {
                        Algorithm::NonCyclicOuter => model.algorithm(Algorithm::NonCyclicOuter),
                        Algorithm::Cyclic => model,
                        Algorithm::NonCyclic => model.algorithm(Algorithm::NonCyclic),
                    };

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
                InternalFamily::Tweedie(ref t_fam) => {
                    let mut model = BoostLss::new(t_fam.clone())
                        .step_length(self.step_length)
                        .mstop(Mstop::Scalar(self.mstop));

                    for (param, learner) in &self.learners {
                        model = model
                            .on(param.as_str(), |p| p.add(learner.clone()))
                            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                    }

                    let model = match self.algorithm {
                        Algorithm::NonCyclicOuter => model.algorithm(Algorithm::NonCyclicOuter),
                        Algorithm::Cyclic => model,
                        Algorithm::NonCyclic => model.algorithm(Algorithm::NonCyclic),
                    };

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
                InternalFamily::Zinb(ref z_fam) => {
                    let mut model = BoostLss::new(z_fam.clone())
                        .step_length(self.step_length)
                        .mstop(Mstop::Scalar(self.mstop));

                    for (param, learner) in &self.learners {
                        model = model
                            .on(param.as_str(), |p| p.add(learner.clone()))
                            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                    }

                    let model = match self.algorithm {
                        Algorithm::NonCyclicOuter => model.algorithm(Algorithm::NonCyclicOuter),
                        Algorithm::Cyclic => model,
                        Algorithm::NonCyclic => model.algorithm(Algorithm::NonCyclic),
                    };

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
                InternalFamily::Binomial => {
                    let mut model = BoostLss::new(BinomialLss::new())
                        .step_length(self.step_length)
                        .mstop(Mstop::Scalar(self.mstop));

                    for (param, learner) in &self.learners {
                        model = model
                            .on(param.as_str(), |p| p.add(learner.clone()))
                            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                    }

                    let model = match self.algorithm {
                        Algorithm::NonCyclicOuter => model.algorithm(Algorithm::NonCyclicOuter),
                        Algorithm::Cyclic => model,
                        Algorithm::NonCyclic => model.algorithm(Algorithm::NonCyclic),
                    };

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
                InternalFamily::Beta => {
                    let mut model = BoostLss::new(BetaLss::new())
                        .step_length(self.step_length)
                        .mstop(Mstop::Scalar(self.mstop));

                    for (param, learner) in &self.learners {
                        model = model
                            .on(param.as_str(), |p| p.add(learner.clone()))
                            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                    }

                    let model = match self.algorithm {
                        Algorithm::NonCyclicOuter => model.algorithm(Algorithm::NonCyclicOuter),
                        Algorithm::Cyclic => model,
                        Algorithm::NonCyclic => model.algorithm(Algorithm::NonCyclic),
                    };

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
                InternalFamily::Weibull => {
                    let mut model = BoostLss::new(WeibullLss::new())
                        .step_length(self.step_length)
                        .mstop(Mstop::Scalar(self.mstop));

                    for (param, learner) in &self.learners {
                        model = model
                            .on(param.as_str(), |p| p.add(learner.clone()))
                            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                    }

                    let model = match self.algorithm {
                        Algorithm::NonCyclicOuter => model.algorithm(Algorithm::NonCyclicOuter),
                        Algorithm::Cyclic => model,
                        Algorithm::NonCyclic => model.algorithm(Algorithm::NonCyclic),
                    };

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
                InternalFamily::LogNormal => {
                    let mut model = BoostLss::new(LogNormalLss::new())
                        .step_length(self.step_length)
                        .mstop(Mstop::Scalar(self.mstop));

                    for (param, learner) in &self.learners {
                        model = model
                            .on(param.as_str(), |p| p.add(learner.clone()))
                            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                    }

                    let model = match self.algorithm {
                        Algorithm::NonCyclicOuter => model.algorithm(Algorithm::NonCyclicOuter),
                        Algorithm::Cyclic => model,
                        Algorithm::NonCyclic => model.algorithm(Algorithm::NonCyclic),
                    };

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
                InternalFamily::Zip => {
                    let mut model = BoostLss::new(ZIPLss::new())
                        .step_length(self.step_length)
                        .mstop(Mstop::Scalar(self.mstop));

                    for (param, learner) in &self.learners {
                        model = model
                            .on(param.as_str(), |p| p.add(learner.clone()))
                            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                    }

                    let model = match self.algorithm {
                        Algorithm::NonCyclicOuter => model.algorithm(Algorithm::NonCyclicOuter),
                        Algorithm::Cyclic => model,
                        Algorithm::NonCyclic => model.algorithm(Algorithm::NonCyclic),
                    };

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
                InternalFamily::Gev => {
                    let mut model = BoostLss::new(GEVLss::new())
                        .step_length(self.step_length)
                        .mstop(Mstop::Scalar(self.mstop));

                    for (param, learner) in &self.learners {
                        model = model
                            .on(param.as_str(), |p| p.add(learner.clone()))
                            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                    }

                    let model = match self.algorithm {
                        Algorithm::NonCyclicOuter => model.algorithm(Algorithm::NonCyclicOuter),
                        Algorithm::Cyclic => model,
                        Algorithm::NonCyclic => model.algorithm(Algorithm::NonCyclic),
                    };

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
                InternalFamily::Jsu => {
                    let mut model = BoostLss::new(JSULss::new())
                        .step_length(self.step_length)
                        .mstop(Mstop::Scalar(self.mstop));

                    for (param, learner) in &self.learners {
                        model = model
                            .on(param.as_str(), |p| p.add(learner.clone()))
                            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                    }

                    let model = match self.algorithm {
                        Algorithm::NonCyclicOuter => model.algorithm(Algorithm::NonCyclicOuter),
                        Algorithm::Cyclic => model,
                        Algorithm::NonCyclic => model.algorithm(Algorithm::NonCyclic),
                    };

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

    #[pyo3(signature = (b=100, pfer=None, pi_thr=None, q=None, mode="joint"))]
    fn stabsel(
        &mut self,
        b: usize,
        pfer: Option<f64>,
        pi_thr: Option<f64>,
        q: Option<usize>,
        mode: &str,
    ) -> PyResult<crate::stabsel::PyStabselResult> {
        let train_data = self.train_data.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "Model must be fitted on data first to know the design matrix size.",
            )
        })?;

        let dataset = train_data.clone();

        let stabsel_mode = match mode.to_lowercase().as_str() {
            "joint" => boostlss::cv::stabsel::StabselMode::Joint,
            "independent" => boostlss::cv::stabsel::StabselMode::Independent,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "mode must be 'joint' or 'independent'",
                ))
            }
        };

        // We track unique (parameter, base_learner) pairs in run_stabsel.
        // Therefore, the total number of available features `p` MUST be the
        // total number of parameter-learner pairs.
        let p = self.learners.len();

        let config = boostlss::cv::stabsel::StabselConfig::new(b, pfer, pi_thr, q, stabsel_mode, p)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let result = match &self.family {
            InternalFamily::Gaussian => {
                let mut model = BoostLss::new(GaussianLss::new())
                    .step_length(self.step_length)
                    .mstop(Mstop::Scalar(self.mstop));
                for (param, learner) in &self.learners {
                    model = model
                        .on(param.as_str(), |p| p.add(learner.clone()))
                        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                }
                let model = match self.algorithm {
                    Algorithm::NonCyclicOuter => model.algorithm(Algorithm::NonCyclicOuter),
                    Algorithm::Cyclic => model,
                    Algorithm::NonCyclic => model.algorithm(Algorithm::NonCyclic),
                };
                boostlss::cv::stabsel::run_stabsel(
                    &model,
                    &dataset,
                    Mstop::Scalar(self.mstop),
                    &config,
                )
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
            }
            InternalFamily::Binomial => {
                let mut model = BoostLss::new(BinomialLss::new())
                    .step_length(self.step_length)
                    .mstop(Mstop::Scalar(self.mstop));
                for (param, learner) in &self.learners {
                    model = model
                        .on(param.as_str(), |p| p.add(learner.clone()))
                        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                }
                let model = match self.algorithm {
                    Algorithm::NonCyclicOuter => model.algorithm(Algorithm::NonCyclicOuter),
                    Algorithm::Cyclic => model,
                    Algorithm::NonCyclic => model.algorithm(Algorithm::NonCyclic),
                };
                boostlss::cv::stabsel::run_stabsel(
                    &model,
                    &dataset,
                    Mstop::Scalar(self.mstop),
                    &config,
                )
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
            }
            InternalFamily::Beta => {
                let mut model = BoostLss::new(BetaLss::new())
                    .step_length(self.step_length)
                    .mstop(Mstop::Scalar(self.mstop));
                for (param, learner) in &self.learners {
                    model = model
                        .on(param.as_str(), |p| p.add(learner.clone()))
                        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                }
                let model = match self.algorithm {
                    Algorithm::NonCyclicOuter => model.algorithm(Algorithm::NonCyclicOuter),
                    Algorithm::Cyclic => model,
                    Algorithm::NonCyclic => model.algorithm(Algorithm::NonCyclic),
                };
                boostlss::cv::stabsel::run_stabsel(
                    &model,
                    &dataset,
                    Mstop::Scalar(self.mstop),
                    &config,
                )
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
            }
            InternalFamily::Weibull => {
                let mut model = BoostLss::new(WeibullLss::new())
                    .step_length(self.step_length)
                    .mstop(Mstop::Scalar(self.mstop));
                for (param, learner) in &self.learners {
                    model = model
                        .on(param.as_str(), |p| p.add(learner.clone()))
                        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                }
                let model = match self.algorithm {
                    Algorithm::NonCyclicOuter => model.algorithm(Algorithm::NonCyclicOuter),
                    Algorithm::Cyclic => model,
                    Algorithm::NonCyclic => model.algorithm(Algorithm::NonCyclic),
                };
                boostlss::cv::stabsel::run_stabsel(
                    &model,
                    &dataset,
                    Mstop::Scalar(self.mstop),
                    &config,
                )
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
            }
            InternalFamily::LogNormal => {
                let mut model = BoostLss::new(LogNormalLss::new())
                    .step_length(self.step_length)
                    .mstop(Mstop::Scalar(self.mstop));
                for (param, learner) in &self.learners {
                    model = model
                        .on(param.as_str(), |p| p.add(learner.clone()))
                        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                }
                let model = match self.algorithm {
                    Algorithm::NonCyclicOuter => model.algorithm(Algorithm::NonCyclicOuter),
                    Algorithm::Cyclic => model,
                    Algorithm::NonCyclic => model.algorithm(Algorithm::NonCyclic),
                };
                boostlss::cv::stabsel::run_stabsel(
                    &model,
                    &dataset,
                    Mstop::Scalar(self.mstop),
                    &config,
                )
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
            }
            InternalFamily::Zip => {
                let mut model = BoostLss::new(ZIPLss::new())
                    .step_length(self.step_length)
                    .mstop(Mstop::Scalar(self.mstop));
                for (param, learner) in &self.learners {
                    model = model
                        .on(param.as_str(), |p| p.add(learner.clone()))
                        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                }
                let model = match self.algorithm {
                    Algorithm::NonCyclicOuter => model.algorithm(Algorithm::NonCyclicOuter),
                    Algorithm::Cyclic => model,
                    Algorithm::NonCyclic => model.algorithm(Algorithm::NonCyclic),
                };
                boostlss::cv::stabsel::run_stabsel(
                    &model,
                    &dataset,
                    Mstop::Scalar(self.mstop),
                    &config,
                )
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
            }
            InternalFamily::Gev => {
                let mut model = BoostLss::new(GEVLss::new())
                    .step_length(self.step_length)
                    .mstop(Mstop::Scalar(self.mstop));
                for (param, learner) in &self.learners {
                    model = model
                        .on(param.as_str(), |p| p.add(learner.clone()))
                        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                }
                let model = match self.algorithm {
                    Algorithm::NonCyclicOuter => model.algorithm(Algorithm::NonCyclicOuter),
                    Algorithm::Cyclic => model,
                    Algorithm::NonCyclic => model.algorithm(Algorithm::NonCyclic),
                };
                boostlss::cv::stabsel::run_stabsel(
                    &model,
                    &dataset,
                    Mstop::Scalar(self.mstop),
                    &config,
                )
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
            }
            InternalFamily::Jsu => {
                let mut model = BoostLss::new(JSULss::new())
                    .step_length(self.step_length)
                    .mstop(Mstop::Scalar(self.mstop));
                for (param, learner) in &self.learners {
                    model = model
                        .on(param.as_str(), |p| p.add(learner.clone()))
                        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                }
                let model = match self.algorithm {
                    Algorithm::NonCyclicOuter => model.algorithm(Algorithm::NonCyclicOuter),
                    Algorithm::Cyclic => model,
                    Algorithm::NonCyclic => model.algorithm(Algorithm::NonCyclic),
                };
                boostlss::cv::stabsel::run_stabsel(
                    &model,
                    &dataset,
                    Mstop::Scalar(self.mstop),
                    &config,
                )
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
            }
            InternalFamily::Tweedie(ref t_fam) => {
                let mut model = BoostLss::new(t_fam.clone())
                    .step_length(self.step_length)
                    .mstop(Mstop::Scalar(self.mstop));
                for (param, learner) in &self.learners {
                    model = model
                        .on(param.as_str(), |p| p.add(learner.clone()))
                        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                }
                let model = match self.algorithm {
                    Algorithm::NonCyclicOuter => model.algorithm(Algorithm::NonCyclicOuter),
                    Algorithm::Cyclic => model,
                    Algorithm::NonCyclic => model.algorithm(Algorithm::NonCyclic),
                };
                boostlss::cv::stabsel::run_stabsel(
                    &model,
                    &dataset,
                    Mstop::Scalar(self.mstop),
                    &config,
                )
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
            }
            InternalFamily::Zinb(ref z_fam) => {
                let mut model = BoostLss::new(z_fam.clone())
                    .step_length(self.step_length)
                    .mstop(Mstop::Scalar(self.mstop));
                for (param, learner) in &self.learners {
                    model = model
                        .on(param.as_str(), |p| p.add(learner.clone()))
                        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                }
                let model = match self.algorithm {
                    Algorithm::NonCyclicOuter => model.algorithm(Algorithm::NonCyclicOuter),
                    Algorithm::Cyclic => model,
                    Algorithm::NonCyclic => model.algorithm(Algorithm::NonCyclic),
                };
                boostlss::cv::stabsel::run_stabsel(
                    &model,
                    &dataset,
                    Mstop::Scalar(self.mstop),
                    &config,
                )
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
            }
        };

        let mut probabilities = std::collections::HashMap::new();
        let mut selected_joint_set = std::collections::HashSet::new();
        let mut selected_independent = std::collections::HashMap::new();

        for (param, freqs) in result.frequencies {
            let mut param_probs = std::collections::HashMap::new();
            let mut param_selected = Vec::new();
            for (learner, count) in freqs {
                let prob = (count as f64) / (result.b as f64);
                param_probs.insert(learner.clone(), prob);
                if prob >= result.pi_thr {
                    selected_joint_set.insert(learner.clone());
                    param_selected.push(learner);
                }
            }
            probabilities.insert(param.clone(), param_probs);
            selected_independent.insert(param, param_selected);
        }

        let mut selected_joint: Vec<String> = selected_joint_set.into_iter().collect();
        selected_joint.sort();

        Ok(crate::stabsel::PyStabselResult {
            selected_joint,
            selected_independent,
            probabilities,
            q: result.q,
            pfer: result.pfer,
            pi_thr: result.pi_thr,
            b: result.b,
        })
    }

    fn __getnewargs__<'py>(&self, py: Python<'py>) -> (pyo3::PyObject, usize, f64, String) {
        let algo_str = match self.algorithm {
            Algorithm::NonCyclicOuter => "noncyclic_outer",
            Algorithm::Cyclic => "cyclic",
            Algorithm::NonCyclic => "noncyclic",
        }
        .to_string();
        {
            let fam_obj = match &self.family {
                InternalFamily::Tweedie(t) => {
                    crate::family::PyTweedieLss { inner: t.clone() }.into_py(py)
                }
                InternalFamily::Zinb(z) => {
                    crate::family::PyZINBLss { inner: z.clone() }.into_py(py)
                }
                f => {
                    let py_fam = match f {
                        InternalFamily::Gaussian => PyFamily::Gaussian,
                        InternalFamily::Binomial => PyFamily::Binomial,
                        InternalFamily::Beta => PyFamily::Beta,
                        InternalFamily::Weibull => PyFamily::Weibull,
                        InternalFamily::LogNormal => PyFamily::LogNormal,
                        InternalFamily::Zip => PyFamily::Zip,
                        InternalFamily::Gev => PyFamily::Gev,
                        InternalFamily::Jsu => PyFamily::Jsu,
                        InternalFamily::Tweedie(_) => unreachable!(),
                        InternalFamily::Zinb(_) => unreachable!(),
                    };
                    py_fam.into_py(py)
                }
            };
            (fam_obj, self.mstop, self.step_length, algo_str)
        }
    }

    fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new_bound(py);

        let family_str = match &self.family {
            InternalFamily::Gaussian => "GaussianLss",
            InternalFamily::Binomial => "BinomialLss",
            InternalFamily::Beta => "BetaLss",
            InternalFamily::Weibull => "WeibullLss",
            InternalFamily::LogNormal => "LogNormalLss",
            InternalFamily::Zip => "ZIPLss",
            InternalFamily::Gev => "GEVLss",
            InternalFamily::Jsu => "JSULss",
            InternalFamily::Tweedie(t) => {
                dict.set_item("tweedie_p", t.p)?;
                "TweedieLss"
            }
            InternalFamily::Zinb(_) => "ZINBLss",
        };
        dict.set_item("family", family_str)?;
        dict.set_item("mstop", self.mstop)?;
        dict.set_item("step_length", self.step_length)?;
        let algo_str = match self.algorithm {
            Algorithm::NonCyclicOuter => "noncyclic_outer",
            Algorithm::Cyclic => "cyclic",
            Algorithm::NonCyclic => "noncyclic",
        };
        dict.set_item("algorithm", algo_str)?;
        // Skip train_data entirely!

        if let Some(fitted) = &self.fitted {
            let bytes = match fitted {
                FittedModel::Gaussian(f) => bincode::serialize(f),
                FittedModel::Binomial(f) => bincode::serialize(f),
                FittedModel::Beta(f) => bincode::serialize(f),
                FittedModel::Weibull(f) => bincode::serialize(f),
                FittedModel::LogNormal(f) => bincode::serialize(f),
                FittedModel::Zip(f) => bincode::serialize(f),
                FittedModel::Gev(f) => bincode::serialize(f),
                FittedModel::Jsu(f) => bincode::serialize(f),
                FittedModel::Tweedie(f) => bincode::serialize(f),
                FittedModel::Zinb(f) => bincode::serialize(f),
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
            "GaussianLss" => InternalFamily::Gaussian,
            "BinomialLss" => InternalFamily::Binomial,
            "BetaLss" => InternalFamily::Beta,
            "WeibullLss" => InternalFamily::Weibull,
            "LogNormalLss" => InternalFamily::LogNormal,
            "ZIPLss" => InternalFamily::Zip,
            "GEVLss" => InternalFamily::Gev,
            "JSULss" => InternalFamily::Jsu,
            "TweedieLss" => {
                let p: f64 = state
                    .get_item("tweedie_p")?
                    .ok_or_else(|| {
                        pyo3::exceptions::PyKeyError::new_err("Missing key 'tweedie_p'")
                    })?
                    .extract()?;
                InternalFamily::Tweedie(boostlss::family::TweedieLss::new(p))
            }
            "ZINBLss" => InternalFamily::Zinb(boostlss::family::ZINBLss::new()),
            _ => return Err(pyo3::exceptions::PyRuntimeError::new_err("Unknown family")),
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
            let algo_str: String = algo_any.extract()?;
            match algo_str.as_str() {
                "cyclic" => Algorithm::Cyclic,
                "noncyclic" => Algorithm::NonCyclic,
                "noncyclic_outer" => Algorithm::NonCyclicOuter,
                _ => return Err(pyo3::exceptions::PyValueError::new_err("Unknown algorithm")),
            }
        } else {
            Algorithm::Cyclic
        };

        if let Some(bytes_any) = state.get_item("fitted")? {
            let bytes: &[u8] = bytes_any.extract()?;
            self.fitted = Some(match &self.family {
                InternalFamily::Gaussian => FittedModel::Gaussian(
                    bincode::deserialize(bytes)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
                ),
                InternalFamily::Binomial => FittedModel::Binomial(
                    bincode::deserialize(bytes)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
                ),
                InternalFamily::Beta => FittedModel::Beta(
                    bincode::deserialize(bytes)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
                ),
                InternalFamily::Weibull => FittedModel::Weibull(
                    bincode::deserialize(bytes)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
                ),
                InternalFamily::LogNormal => FittedModel::LogNormal(
                    bincode::deserialize(bytes)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
                ),
                InternalFamily::Zip => FittedModel::Zip(
                    bincode::deserialize(bytes)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
                ),
                InternalFamily::Gev => FittedModel::Gev(
                    bincode::deserialize(bytes)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
                ),
                InternalFamily::Jsu => FittedModel::Jsu(
                    bincode::deserialize(bytes)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
                ),
                InternalFamily::Tweedie(_) => FittedModel::Tweedie(
                    bincode::deserialize(bytes)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
                ),
                InternalFamily::Zinb(_) => FittedModel::Zinb(
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
