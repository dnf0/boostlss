use crate::family::{
    PyFamily, PyLaplaceLss, PyMertonJumpDiffusionLss, PySHASHLss, PyTweedieLss, PyZINBLss,
};
use boostlss::cv::{CvRisk, Resampling};
use boostlss::data::Dataset;
use boostlss::engine::cyclical::fit_cyclical;
use boostlss::engine::noncyclical::{fit_noncyclical, fit_noncyclical_outer};
use boostlss::engine::{Algorithm, Mstop};
use boostlss::family::{
    BetaLss, BinomialLss, Burr12Lss, GEVLss, GammaLss, GaussianLss, GedLss, GpdLss,
    InverseGaussianLss, JSULss, LaplaceLss, LogLogisticLss, LogNormalLss, MultinomialLss,
    NBinomialLss, NigLss, PoissonLss, StudentTLss, WeibullLss, ZIPLss,
};
use boostlss::family::{MertonJumpDiffusionLss, SHASHLss, TweedieLss, ZINBLss};
use boostlss::learner::{BaseLearner, RandomEffects};
use boostlss::model::{BoostLss, Fitted, Scale};
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyType};

macro_rules! fit_family {
    ($self:expr, $dataset:expr, $eval_data:expr, $early_stopping:expr, $family:expr, $fitted_variant:path) => {{
        let mut model = BoostLss::new($family)
            .step_length($self.step_length)
            .mstop(Mstop::Scalar($self.mstop))
            .algorithm($self.algorithm.clone());

        for (param, learner) in &$self.learners {
            model = model
                .on(param.as_str(), |p| p.add(learner.clone()))
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        }

        let fitted = model
            .fit($dataset, $eval_data, $early_stopping)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        $self.fitted = Some($fitted_variant(fitted));
    }};
}

macro_rules! cvrisk_family {
    ($self:expr, $dataset:expr, $family:expr, $folds:expr, $py:expr) => {{
        let mut model = BoostLss::new($family)
            .step_length($self.step_length)
            .mstop(Mstop::Scalar($self.mstop));

        for (param, learner) in &$self.learners {
            model = model
                .on(param.as_str(), |p| p.add(learner.clone()))
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        }

        let model = match $self.algorithm {
            Algorithm::NonCyclicOuter => model.algorithm(Algorithm::NonCyclicOuter),
            Algorithm::Cyclic => model,
            Algorithm::NonCyclic => model.algorithm(Algorithm::NonCyclic),
        };

        let cv = CvRisk::new(model, Resampling::KFold { k: $folds });
        let result = cv
            .run($dataset)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let dict = PyDict::new_bound($py);
        match result.optimal_mstop {
            Mstop::Scalar(m) => dict.set_item("optimal_mstop", m)?,
            Mstop::PerParam(v) => dict.set_item("optimal_mstop", v)?,
        }
        dict.set_item("mean_risk", result.mean_risk)?;
        Ok(dict)
    }};
}

macro_rules! stabsel_family {
    ($self:expr, $dataset:expr, $family:expr, $config:expr) => {{
        let mut model = BoostLss::new($family)
            .step_length($self.step_length)
            .mstop(Mstop::Scalar($self.mstop));

        for (param, learner) in &$self.learners {
            model = model
                .on(param.as_str(), |p| p.add(learner.clone()))
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        }

        let model = match $self.algorithm {
            Algorithm::NonCyclicOuter => model.algorithm(Algorithm::NonCyclicOuter),
            Algorithm::Cyclic => model,
            Algorithm::NonCyclic => model.algorithm(Algorithm::NonCyclic),
        };

        boostlss::cv::stabsel::run_stabsel(&model, $dataset, Mstop::Scalar($self.mstop), $config)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
    }};
}

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
    Burr12,
    Gamma,
    Ged,
    Gpd,
    InverseGaussian,
    LogLogistic,
    NBinomial,
    Nig,
    Poisson,
    StudentT,
    Multinomial,
    Tweedie(TweedieLss),
    Logistic,
    Zinb(ZINBLss),
    Laplace(LaplaceLss),
    Merton(MertonJumpDiffusionLss),
    Shash(SHASHLss),
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
    Burr12(Fitted<Burr12Lss>),
    Gamma(Fitted<GammaLss>),
    Ged(Fitted<GedLss>),
    Gpd(Fitted<GpdLss>),
    InverseGaussian(Fitted<InverseGaussianLss>),
    LogLogistic(Fitted<LogLogisticLss>),
    NBinomial(Fitted<NBinomialLss>),
    Nig(Fitted<NigLss>),
    Poisson(Fitted<PoissonLss>),
    StudentT(Fitted<StudentTLss>),
    Multinomial(Fitted<MultinomialLss>),
    Tweedie(Fitted<TweedieLss>),
    Logistic(Fitted<boostlss::family::LogisticLss>),
    Zinb(Fitted<ZINBLss>),
    Laplace(Fitted<LaplaceLss>),
    Merton(Fitted<MertonJumpDiffusionLss>),
    Shash(Fitted<SHASHLss>),
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
            Self::Burr12(fitted) => fitted
                .predict(dataset, param, scale)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string())),
            Self::Gamma(fitted) => fitted
                .predict(dataset, param, scale)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string())),
            Self::Ged(fitted) => fitted
                .predict(dataset, param, scale)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string())),
            Self::Gpd(fitted) => fitted
                .predict(dataset, param, scale)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string())),
            Self::InverseGaussian(fitted) => fitted
                .predict(dataset, param, scale)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string())),
            Self::LogLogistic(fitted) => fitted
                .predict(dataset, param, scale)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string())),
            Self::NBinomial(fitted) => fitted
                .predict(dataset, param, scale)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string())),
            Self::Nig(fitted) => fitted
                .predict(dataset, param, scale)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string())),
            Self::Poisson(fitted) => fitted
                .predict(dataset, param, scale)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string())),
            Self::StudentT(fitted) => fitted
                .predict(dataset, param, scale)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string())),
            Self::Multinomial(fitted) => fitted
                .predict(dataset, param, scale)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string())),
            Self::Logistic(fitted) => fitted
                .predict(dataset, param, scale)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string())),
            Self::Tweedie(fitted) => fitted
                .predict(dataset, param, scale)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string())),
            Self::Zinb(fitted) => fitted
                .predict(dataset, param, scale)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string())),
            Self::Laplace(fitted) => fitted
                .predict(dataset, param, scale)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string())),
            Self::Merton(fitted) => fitted
                .predict(dataset, param, scale)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string())),
            Self::Shash(fitted) => fitted
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
            Self::Burr12(fitted) => fitted.feature_importance(),
            Self::Gamma(fitted) => fitted.feature_importance(),
            Self::Ged(fitted) => fitted.feature_importance(),
            Self::Gpd(fitted) => fitted.feature_importance(),
            Self::InverseGaussian(fitted) => fitted.feature_importance(),
            Self::LogLogistic(fitted) => fitted.feature_importance(),
            Self::NBinomial(fitted) => fitted.feature_importance(),
            Self::Nig(fitted) => fitted.feature_importance(),
            Self::Poisson(fitted) => fitted.feature_importance(),
            Self::StudentT(fitted) => fitted.feature_importance(),
            Self::Multinomial(fitted) => fitted.feature_importance(),
            Self::Logistic(fitted) => fitted.feature_importance(),
            Self::Tweedie(fitted) => fitted.feature_importance(),
            Self::Zinb(fitted) => fitted.feature_importance(),
            Self::Laplace(fitted) => fitted.feature_importance(),
            Self::Merton(fitted) => fitted.feature_importance(),
            Self::Shash(fitted) => fitted.feature_importance(),
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
            Self::Burr12(fitted) => fitted
                .partial_dependence(dataset, param, feature_idx, grid)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string())),
            Self::Gamma(fitted) => fitted
                .partial_dependence(dataset, param, feature_idx, grid)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string())),
            Self::Ged(fitted) => fitted
                .partial_dependence(dataset, param, feature_idx, grid)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string())),
            Self::Gpd(fitted) => fitted
                .partial_dependence(dataset, param, feature_idx, grid)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string())),
            Self::InverseGaussian(fitted) => fitted
                .partial_dependence(dataset, param, feature_idx, grid)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string())),
            Self::LogLogistic(fitted) => fitted
                .partial_dependence(dataset, param, feature_idx, grid)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string())),
            Self::NBinomial(fitted) => fitted
                .partial_dependence(dataset, param, feature_idx, grid)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string())),
            Self::Nig(fitted) => fitted
                .partial_dependence(dataset, param, feature_idx, grid)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string())),
            Self::Poisson(fitted) => fitted
                .partial_dependence(dataset, param, feature_idx, grid)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string())),
            Self::StudentT(fitted) => fitted
                .partial_dependence(dataset, param, feature_idx, grid)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string())),
            Self::Multinomial(fitted) => fitted
                .partial_dependence(dataset, param, feature_idx, grid)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string())),
            Self::Logistic(fitted) => fitted
                .partial_dependence(dataset, param, feature_idx, grid)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string())),
            Self::Tweedie(fitted) => fitted
                .partial_dependence(dataset, param, feature_idx, grid)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string())),
            Self::Zinb(fitted) => fitted
                .partial_dependence(dataset, param, feature_idx, grid)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string())),
            Self::Laplace(fitted) => fitted
                .partial_dependence(dataset, param, feature_idx, grid)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string())),
            Self::Merton(fitted) => fitted
                .partial_dependence(dataset, param, feature_idx, grid)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string())),
            Self::Shash(fitted) => fitted
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
                PyFamily::Burr12 => InternalFamily::Burr12,
                PyFamily::Gamma => InternalFamily::Gamma,
                PyFamily::Ged => InternalFamily::Ged,
                PyFamily::Gpd => InternalFamily::Gpd,
                PyFamily::InverseGaussian => InternalFamily::InverseGaussian,
                PyFamily::LogLogistic => InternalFamily::LogLogistic,
                PyFamily::NBinomial => InternalFamily::NBinomial,
                PyFamily::Nig => InternalFamily::Nig,
                PyFamily::Poisson => InternalFamily::Poisson,
                PyFamily::StudentT => InternalFamily::StudentT,
                PyFamily::Multinomial => InternalFamily::Multinomial,
            }
        } else if family.extract::<crate::family::PyLogisticLss>().is_ok() {
            InternalFamily::Logistic
        } else if let Ok(t) = family.extract::<PyTweedieLss>() {
            InternalFamily::Tweedie(t.inner.clone())
        } else if let Ok(z) = family.extract::<PyZINBLss>() {
            InternalFamily::Zinb(z.inner.clone())
        } else if let Ok(l) = family.extract::<PyLaplaceLss>() {
            InternalFamily::Laplace(l.inner.clone())
        } else if let Ok(m) = family.extract::<PyMertonJumpDiffusionLss>() {
            InternalFamily::Merton(m.inner.clone())
        } else if let Ok(s) = family.extract::<PySHASHLss>() {
            InternalFamily::Shash(s.inner.clone())
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

    #[pyo3(signature = (x, y, eval_set=None, early_stopping_rounds=None))]
    fn fit<'py>(
        &mut self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
        y: PyReadonlyArray1<'py, f64>,
        eval_set: Option<(Bound<'py, PyAny>, PyReadonlyArray1<'py, f64>)>,
        early_stopping_rounds: Option<usize>,
    ) -> PyResult<()> {
        let y_view = y.as_array();
        let y_vec =
            ndarray::Array1::from_shape_vec((y_view.len(),), y_view.to_owned().into_raw_vec())
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let dataset = crate::data::extract_dataset(py, x, Some(y_vec))?;

        self.train_data = Some(dataset.clone());

        let eval_dataset = if let Some((e_x, e_y)) = eval_set {
            let e_y_view = e_y.as_array();
            let e_y_vec = ndarray::Array1::from_shape_vec(
                (e_y_view.len(),),
                e_y_view.to_owned().into_raw_vec(),
            )
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            Some(crate::data::extract_dataset(py, &e_x, Some(e_y_vec))?)
        } else {
            None
        };
        let eval_data_ref = eval_dataset.as_ref();

        match &self.family {
            InternalFamily::Gaussian => {
                fit_family!(
                    self,
                    &dataset,
                    eval_data_ref,
                    early_stopping_rounds,
                    GaussianLss::new(),
                    FittedModel::Gaussian
                )
            }
            InternalFamily::Binomial => {
                fit_family!(
                    self,
                    &dataset,
                    eval_data_ref,
                    early_stopping_rounds,
                    BinomialLss::new(),
                    FittedModel::Binomial
                )
            }
            InternalFamily::Beta => fit_family!(
                self,
                &dataset,
                eval_data_ref,
                early_stopping_rounds,
                BetaLss::new(),
                FittedModel::Beta
            ),
            InternalFamily::Weibull => {
                fit_family!(
                    self,
                    &dataset,
                    eval_data_ref,
                    early_stopping_rounds,
                    WeibullLss::new(),
                    FittedModel::Weibull
                )
            }
            InternalFamily::LogNormal => {
                fit_family!(
                    self,
                    &dataset,
                    eval_data_ref,
                    early_stopping_rounds,
                    LogNormalLss::new(),
                    FittedModel::LogNormal
                )
            }
            InternalFamily::Zip => {
                fit_family!(
                    self,
                    &dataset,
                    eval_data_ref,
                    early_stopping_rounds,
                    ZIPLss::new(),
                    FittedModel::Zip
                )
            }
            InternalFamily::Gev => {
                fit_family!(
                    self,
                    &dataset,
                    eval_data_ref,
                    early_stopping_rounds,
                    GEVLss::new(),
                    FittedModel::Gev
                )
            }
            InternalFamily::Jsu => {
                fit_family!(
                    self,
                    &dataset,
                    eval_data_ref,
                    early_stopping_rounds,
                    JSULss::new(),
                    FittedModel::Jsu
                )
            }
            InternalFamily::Burr12 => {
                fit_family!(
                    self,
                    &dataset,
                    eval_data_ref,
                    early_stopping_rounds,
                    Burr12Lss::new(),
                    FittedModel::Burr12
                )
            }
            InternalFamily::Gamma => {
                fit_family!(
                    self,
                    &dataset,
                    eval_data_ref,
                    early_stopping_rounds,
                    GammaLss::new(),
                    FittedModel::Gamma
                )
            }
            InternalFamily::Ged => {
                fit_family!(
                    self,
                    &dataset,
                    eval_data_ref,
                    early_stopping_rounds,
                    GedLss::new(),
                    FittedModel::Ged
                )
            }
            InternalFamily::Gpd => {
                fit_family!(
                    self,
                    &dataset,
                    eval_data_ref,
                    early_stopping_rounds,
                    GpdLss::new(),
                    FittedModel::Gpd
                )
            }
            InternalFamily::InverseGaussian => fit_family!(
                self,
                &dataset,
                eval_data_ref,
                early_stopping_rounds,
                InverseGaussianLss::new(),
                FittedModel::InverseGaussian
            ),
            InternalFamily::LogLogistic => fit_family!(
                self,
                &dataset,
                eval_data_ref,
                early_stopping_rounds,
                LogLogisticLss::new(),
                FittedModel::LogLogistic
            ),
            InternalFamily::NBinomial => {
                fit_family!(
                    self,
                    &dataset,
                    eval_data_ref,
                    early_stopping_rounds,
                    NBinomialLss::new(),
                    FittedModel::NBinomial
                )
            }
            InternalFamily::Nig => {
                fit_family!(
                    self,
                    &dataset,
                    eval_data_ref,
                    early_stopping_rounds,
                    NigLss::new(),
                    FittedModel::Nig
                )
            }
            InternalFamily::Poisson => {
                fit_family!(
                    self,
                    &dataset,
                    eval_data_ref,
                    early_stopping_rounds,
                    PoissonLss::new(),
                    FittedModel::Poisson
                )
            }
            InternalFamily::StudentT => {
                fit_family!(
                    self,
                    &dataset,
                    eval_data_ref,
                    early_stopping_rounds,
                    StudentTLss::new(),
                    FittedModel::StudentT
                )
            }
            InternalFamily::Multinomial => {
                let k = dataset.response().iter().fold(0.0_f64, |m, &x| m.max(x)) as usize + 1;
                fit_family!(
                    self,
                    &dataset,
                    eval_data_ref,
                    early_stopping_rounds,
                    MultinomialLss::new(k),
                    FittedModel::Multinomial
                )
            }
            InternalFamily::Logistic => fit_family!(
                self,
                &dataset,
                eval_data_ref,
                early_stopping_rounds,
                boostlss::family::LogisticLss::new(),
                FittedModel::Logistic
            ),
            InternalFamily::Tweedie(ref t_fam) => {
                fit_family!(
                    self,
                    &dataset,
                    eval_data_ref,
                    early_stopping_rounds,
                    t_fam.clone(),
                    FittedModel::Tweedie
                )
            }
            InternalFamily::Zinb(ref z_fam) => {
                fit_family!(
                    self,
                    &dataset,
                    eval_data_ref,
                    early_stopping_rounds,
                    z_fam.clone(),
                    FittedModel::Zinb
                )
            }
            InternalFamily::Laplace(ref l_fam) => {
                fit_family!(
                    self,
                    &dataset,
                    eval_data_ref,
                    early_stopping_rounds,
                    l_fam.clone(),
                    FittedModel::Laplace
                )
            }
            InternalFamily::Merton(ref m_fam) => {
                fit_family!(
                    self,
                    &dataset,
                    eval_data_ref,
                    early_stopping_rounds,
                    m_fam.clone(),
                    FittedModel::Merton
                )
            }
            InternalFamily::Shash(ref s_fam) => {
                fit_family!(
                    self,
                    &dataset,
                    eval_data_ref,
                    early_stopping_rounds,
                    s_fam.clone(),
                    FittedModel::Shash
                )
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
                    cvrisk_family!(self, &dataset, GaussianLss::new(), folds, py)
                }
                InternalFamily::Logistic => cvrisk_family!(
                    self,
                    &dataset,
                    boostlss::family::LogisticLss::new(),
                    folds,
                    py
                ),
                InternalFamily::Tweedie(ref t_fam) => {
                    cvrisk_family!(self, &dataset, t_fam.clone(), folds, py)
                }
                InternalFamily::Zinb(ref z_fam) => {
                    cvrisk_family!(self, &dataset, z_fam.clone(), folds, py)
                }
                InternalFamily::Laplace(ref l_fam) => {
                    cvrisk_family!(self, &dataset, l_fam.clone(), folds, py)
                }
                InternalFamily::Binomial => {
                    cvrisk_family!(self, &dataset, BinomialLss::new(), folds, py)
                }
                InternalFamily::Beta => cvrisk_family!(self, &dataset, BetaLss::new(), folds, py),
                InternalFamily::Weibull => {
                    cvrisk_family!(self, &dataset, WeibullLss::new(), folds, py)
                }
                InternalFamily::LogNormal => {
                    cvrisk_family!(self, &dataset, LogNormalLss::new(), folds, py)
                }
                InternalFamily::Zip => cvrisk_family!(self, &dataset, ZIPLss::new(), folds, py),
                InternalFamily::Gev => cvrisk_family!(self, &dataset, GEVLss::new(), folds, py),
                InternalFamily::Jsu => cvrisk_family!(self, &dataset, JSULss::new(), folds, py),
                InternalFamily::Burr12 => {
                    cvrisk_family!(self, &dataset, Burr12Lss::new(), folds, py)
                }
                InternalFamily::Gamma => cvrisk_family!(self, &dataset, GammaLss::new(), folds, py),
                InternalFamily::Ged => cvrisk_family!(self, &dataset, GedLss::new(), folds, py),
                InternalFamily::Gpd => cvrisk_family!(self, &dataset, GpdLss::new(), folds, py),
                InternalFamily::InverseGaussian => {
                    cvrisk_family!(self, &dataset, InverseGaussianLss::new(), folds, py)
                }
                InternalFamily::LogLogistic => {
                    cvrisk_family!(self, &dataset, LogLogisticLss::new(), folds, py)
                }
                InternalFamily::NBinomial => {
                    cvrisk_family!(self, &dataset, NBinomialLss::new(), folds, py)
                }
                InternalFamily::Nig => cvrisk_family!(self, &dataset, NigLss::new(), folds, py),
                InternalFamily::Poisson => {
                    cvrisk_family!(self, &dataset, PoissonLss::new(), folds, py)
                }
                InternalFamily::StudentT => {
                    cvrisk_family!(self, &dataset, StudentTLss::new(), folds, py)
                }
                InternalFamily::Multinomial => {
                    let k = dataset.response().iter().fold(0.0_f64, |m, &x| m.max(x)) as usize + 1;
                    cvrisk_family!(self, &dataset, MultinomialLss::new(k), folds, py)
                }
                InternalFamily::Merton(ref m_fam) => {
                    cvrisk_family!(self, &dataset, m_fam.clone(), folds, py)
                }
                InternalFamily::Shash(ref s_fam) => {
                    cvrisk_family!(self, &dataset, s_fam.clone(), folds, py)
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

        let ds = Dataset::new(x_mat, dummy_response, None, None)
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
                stabsel_family!(self, &dataset, GaussianLss::new(), &config)
            }
            InternalFamily::Binomial => {
                stabsel_family!(self, &dataset, BinomialLss::new(), &config)
            }
            InternalFamily::Beta => stabsel_family!(self, &dataset, BetaLss::new(), &config),
            InternalFamily::Weibull => stabsel_family!(self, &dataset, WeibullLss::new(), &config),
            InternalFamily::LogNormal => {
                stabsel_family!(self, &dataset, LogNormalLss::new(), &config)
            }
            InternalFamily::Zip => stabsel_family!(self, &dataset, ZIPLss::new(), &config),
            InternalFamily::Gev => stabsel_family!(self, &dataset, GEVLss::new(), &config),
            InternalFamily::Jsu => stabsel_family!(self, &dataset, JSULss::new(), &config),
            InternalFamily::Burr12 => stabsel_family!(self, &dataset, Burr12Lss::new(), &config),
            InternalFamily::Gamma => stabsel_family!(self, &dataset, GammaLss::new(), &config),
            InternalFamily::Ged => stabsel_family!(self, &dataset, GedLss::new(), &config),
            InternalFamily::Gpd => stabsel_family!(self, &dataset, GpdLss::new(), &config),
            InternalFamily::InverseGaussian => {
                stabsel_family!(self, &dataset, InverseGaussianLss::new(), &config)
            }
            InternalFamily::LogLogistic => {
                stabsel_family!(self, &dataset, LogLogisticLss::new(), &config)
            }
            InternalFamily::NBinomial => {
                stabsel_family!(self, &dataset, NBinomialLss::new(), &config)
            }
            InternalFamily::Nig => stabsel_family!(self, &dataset, NigLss::new(), &config),
            InternalFamily::Poisson => stabsel_family!(self, &dataset, PoissonLss::new(), &config),
            InternalFamily::StudentT => {
                stabsel_family!(self, &dataset, StudentTLss::new(), &config)
            }
            InternalFamily::Multinomial => {
                let k = dataset.response().iter().fold(0.0_f64, |m, &x| m.max(x)) as usize + 1;
                stabsel_family!(self, &dataset, MultinomialLss::new(k), &config)
            }
            InternalFamily::Logistic => stabsel_family!(
                self,
                &dataset,
                boostlss::family::LogisticLss::new(),
                &config
            ),
            InternalFamily::Tweedie(ref t_fam) => {
                stabsel_family!(self, &dataset, t_fam.clone(), &config)
            }
            InternalFamily::Zinb(ref z_fam) => {
                stabsel_family!(self, &dataset, z_fam.clone(), &config)
            }
            InternalFamily::Laplace(ref l_fam) => {
                stabsel_family!(self, &dataset, l_fam.clone(), &config)
            }
            InternalFamily::Merton(ref m_fam) => {
                stabsel_family!(self, &dataset, m_fam.clone(), &config)
            }
            InternalFamily::Shash(ref s_fam) => {
                stabsel_family!(self, &dataset, s_fam.clone(), &config)
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
                InternalFamily::Logistic => crate::family::PyLogisticLss::new().into_py(py),
                InternalFamily::Tweedie(t) => {
                    crate::family::PyTweedieLss { inner: t.clone() }.into_py(py)
                }
                InternalFamily::Zinb(z) => {
                    crate::family::PyZINBLss { inner: z.clone() }.into_py(py)
                }
                InternalFamily::Laplace(l) => {
                    crate::family::PyLaplaceLss { inner: l.clone() }.into_py(py)
                }
                InternalFamily::Merton(m) => {
                    crate::family::PyMertonJumpDiffusionLss { inner: m.clone() }.into_py(py)
                }
                InternalFamily::Shash(s) => {
                    crate::family::PySHASHLss { inner: s.clone() }.into_py(py)
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
                        InternalFamily::Burr12 => PyFamily::Burr12,
                        InternalFamily::Gamma => PyFamily::Gamma,
                        InternalFamily::Ged => PyFamily::Ged,
                        InternalFamily::Gpd => PyFamily::Gpd,
                        InternalFamily::InverseGaussian => PyFamily::InverseGaussian,
                        InternalFamily::LogLogistic => PyFamily::LogLogistic,
                        InternalFamily::NBinomial => PyFamily::NBinomial,
                        InternalFamily::Nig => PyFamily::Nig,
                        InternalFamily::Poisson => PyFamily::Poisson,
                        InternalFamily::StudentT => PyFamily::StudentT,
                        InternalFamily::Multinomial => PyFamily::Multinomial,
                        InternalFamily::Logistic => unreachable!(),
                        InternalFamily::Tweedie(_) => unreachable!(),
                        InternalFamily::Zinb(_) => unreachable!(),
                        InternalFamily::Laplace(_) => unreachable!(),
                        InternalFamily::Merton(_) => unreachable!(),
                        InternalFamily::Shash(_) => unreachable!(),
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
            InternalFamily::Burr12 => "Burr12Lss",
            InternalFamily::Gamma => "GammaLss",
            InternalFamily::Ged => "GedLss",
            InternalFamily::Gpd => "GpdLss",
            InternalFamily::InverseGaussian => "InverseGaussianLss",
            InternalFamily::LogLogistic => "LogLogisticLss",
            InternalFamily::NBinomial => "NBinomialLss",
            InternalFamily::Nig => "NigLss",
            InternalFamily::Poisson => "PoissonLss",
            InternalFamily::StudentT => "StudentTLss",
            InternalFamily::Multinomial => "MultinomialLss",
            InternalFamily::Logistic => "LogisticLss",
            InternalFamily::Tweedie(t) => {
                dict.set_item("tweedie_p", t.p)?;
                "TweedieLss"
            }
            InternalFamily::Zinb(_) => "ZINBLss",
            InternalFamily::Laplace(_) => "LaplaceLss",
            InternalFamily::Merton(m) => {
                dict.set_item("merton_max_jumps", m.max_jumps)?;
                "MertonJumpDiffusionLss"
            }
            InternalFamily::Shash(_) => "SHASHLss",
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
                FittedModel::Burr12(f) => bincode::serialize(f),
                FittedModel::Gamma(f) => bincode::serialize(f),
                FittedModel::Ged(f) => bincode::serialize(f),
                FittedModel::Gpd(f) => bincode::serialize(f),
                FittedModel::InverseGaussian(f) => bincode::serialize(f),
                FittedModel::LogLogistic(f) => bincode::serialize(f),
                FittedModel::NBinomial(f) => bincode::serialize(f),
                FittedModel::Nig(f) => bincode::serialize(f),
                FittedModel::Poisson(f) => bincode::serialize(f),
                FittedModel::StudentT(f) => bincode::serialize(f),
                FittedModel::Multinomial(f) => bincode::serialize(f),
                FittedModel::Logistic(f) => bincode::serialize(f),
                FittedModel::Tweedie(f) => bincode::serialize(f),
                FittedModel::Zinb(f) => bincode::serialize(f),
                FittedModel::Laplace(f) => bincode::serialize(f),
                FittedModel::Merton(f) => bincode::serialize(f),
                FittedModel::Shash(f) => bincode::serialize(f),
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
            "Burr12Lss" => InternalFamily::Burr12,
            "GammaLss" => InternalFamily::Gamma,
            "GedLss" => InternalFamily::Ged,
            "GpdLss" => InternalFamily::Gpd,
            "InverseGaussianLss" => InternalFamily::InverseGaussian,
            "LogLogisticLss" => InternalFamily::LogLogistic,
            "NBinomialLss" => InternalFamily::NBinomial,
            "NigLss" => InternalFamily::Nig,
            "PoissonLss" => InternalFamily::Poisson,
            "StudentTLss" | "STUDENTT" => InternalFamily::StudentT,
            "MultinomialLss" | "MULTINOMIAL" => InternalFamily::Multinomial,
            "TweedieLss" => {
                let p: f64 = state
                    .get_item("tweedie_p")?
                    .ok_or_else(|| {
                        pyo3::exceptions::PyKeyError::new_err("Missing key 'tweedie_p'")
                    })?
                    .extract()?;
                InternalFamily::Tweedie(boostlss::family::TweedieLss::new(p))
            }
            "LogisticLss" => InternalFamily::Logistic,
            "ZINBLss" => InternalFamily::Zinb(boostlss::family::ZINBLss::new()),
            "LaplaceLss" => InternalFamily::Laplace(boostlss::family::LaplaceLss::new()),
            "MertonJumpDiffusionLss" => {
                let j: usize = state
                    .get_item("merton_max_jumps")?
                    .ok_or_else(|| {
                        pyo3::exceptions::PyKeyError::new_err("Missing key 'merton_max_jumps'")
                    })?
                    .extract()?;
                InternalFamily::Merton(boostlss::family::MertonJumpDiffusionLss::new(j))
            }
            "SHASHLss" => InternalFamily::Shash(boostlss::family::SHASHLss::new()),
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
                InternalFamily::Burr12 => FittedModel::Burr12(
                    bincode::deserialize(bytes)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
                ),
                InternalFamily::Gamma => FittedModel::Gamma(
                    bincode::deserialize(bytes)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
                ),
                InternalFamily::Ged => FittedModel::Ged(
                    bincode::deserialize(bytes)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
                ),
                InternalFamily::Gpd => FittedModel::Gpd(
                    bincode::deserialize(bytes)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
                ),
                InternalFamily::InverseGaussian => FittedModel::InverseGaussian(
                    bincode::deserialize(bytes)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
                ),
                InternalFamily::LogLogistic => FittedModel::LogLogistic(
                    bincode::deserialize(bytes)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
                ),
                InternalFamily::NBinomial => FittedModel::NBinomial(
                    bincode::deserialize(bytes)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
                ),
                InternalFamily::Nig => FittedModel::Nig(
                    bincode::deserialize(bytes)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
                ),
                InternalFamily::Poisson => FittedModel::Poisson(
                    bincode::deserialize(bytes)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
                ),
                InternalFamily::StudentT => FittedModel::StudentT(
                    bincode::deserialize(bytes)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
                ),
                InternalFamily::Multinomial => FittedModel::Multinomial(
                    bincode::deserialize(bytes)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
                ),
                InternalFamily::Logistic => FittedModel::Logistic(
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
                InternalFamily::Laplace(_) => FittedModel::Laplace(
                    bincode::deserialize(bytes)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
                ),
                InternalFamily::Merton(_) => FittedModel::Merton(
                    bincode::deserialize(bytes)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
                ),
                InternalFamily::Shash(_) => FittedModel::Shash(
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

    #[getter]
    pub fn evals_result_(&self, py: Python) -> PyResult<PyObject> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Model not fitted"))?;

        macro_rules! get_evals {
            ($fitted_model:expr) => {
                match $fitted_model {
                    FittedModel::Gaussian(f) => &f.eval_results,
                    FittedModel::Binomial(f) => &f.eval_results,
                    FittedModel::StudentT(f) => &f.eval_results,
                    FittedModel::Poisson(f) => &f.eval_results,
                    FittedModel::Beta(f) => &f.eval_results,
                    FittedModel::Weibull(f) => &f.eval_results,
                    FittedModel::LogNormal(f) => &f.eval_results,
                    FittedModel::Zip(f) => &f.eval_results,
                    FittedModel::Gev(f) => &f.eval_results,
                    FittedModel::Jsu(f) => &f.eval_results,
                    FittedModel::Burr12(f) => &f.eval_results,
                    FittedModel::Gamma(f) => &f.eval_results,
                    FittedModel::Ged(f) => &f.eval_results,
                    FittedModel::Gpd(f) => &f.eval_results,
                    FittedModel::InverseGaussian(f) => &f.eval_results,
                    FittedModel::LogLogistic(f) => &f.eval_results,
                    FittedModel::NBinomial(f) => &f.eval_results,
                    FittedModel::Nig(f) => &f.eval_results,
                    FittedModel::Logistic(f) => &f.eval_results,
                    FittedModel::Multinomial(f) => &f.eval_results,
                    FittedModel::Tweedie(f) => &f.eval_results,
                    FittedModel::Zinb(f) => &f.eval_results,
                    FittedModel::Laplace(f) => &f.eval_results,
                    FittedModel::Merton(f) => &f.eval_results,
                    FittedModel::Shash(f) => &f.eval_results,
                }
            };
        }
        let results = get_evals!(fitted);

        let dict = pyo3::types::PyDict::new_bound(py);
        let train_dict = pyo3::types::PyDict::new_bound(py);
        train_dict.set_item("loss", results.train_loss.clone())?;
        dict.set_item("train", train_dict)?;

        if let Some(val_loss) = &results.val_loss {
            let val_dict = pyo3::types::PyDict::new_bound(py);
            val_dict.set_item("loss", val_loss.clone())?;
            dict.set_item("valid", val_dict)?;
        }

        Ok(dict.into())
    }

    #[getter]
    pub fn best_iteration_(&self) -> PyResult<usize> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Model not fitted"))?;

        macro_rules! get_best_iter {
            ($fitted_model:expr) => {
                match $fitted_model {
                    FittedModel::Gaussian(f) => f.best_iteration,
                    FittedModel::Binomial(f) => f.best_iteration,
                    FittedModel::StudentT(f) => f.best_iteration,
                    FittedModel::Poisson(f) => f.best_iteration,
                    FittedModel::Beta(f) => f.best_iteration,
                    FittedModel::Weibull(f) => f.best_iteration,
                    FittedModel::LogNormal(f) => f.best_iteration,
                    FittedModel::Zip(f) => f.best_iteration,
                    FittedModel::Gev(f) => f.best_iteration,
                    FittedModel::Jsu(f) => f.best_iteration,
                    FittedModel::Burr12(f) => f.best_iteration,
                    FittedModel::Gamma(f) => f.best_iteration,
                    FittedModel::Ged(f) => f.best_iteration,
                    FittedModel::Gpd(f) => f.best_iteration,
                    FittedModel::InverseGaussian(f) => f.best_iteration,
                    FittedModel::LogLogistic(f) => f.best_iteration,
                    FittedModel::NBinomial(f) => f.best_iteration,
                    FittedModel::Nig(f) => f.best_iteration,
                    FittedModel::Logistic(f) => f.best_iteration,
                    FittedModel::Multinomial(f) => f.best_iteration,
                    FittedModel::Tweedie(f) => f.best_iteration,
                    FittedModel::Zinb(f) => f.best_iteration,
                    FittedModel::Laplace(f) => f.best_iteration,
                    FittedModel::Merton(f) => f.best_iteration,
                    FittedModel::Shash(f) => f.best_iteration,
                }
            };
        }
        Ok(get_best_iter!(fitted))
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
