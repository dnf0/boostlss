use crate::family::PyFamily;
use boostlss::cv::{CvRisk, Resampling};
use boostlss::data::Dataset;
use boostlss::engine::cyclical::fit_cyclical;
use boostlss::engine::Mstop;
use boostlss::family::GaussianLss;
use boostlss::learner::BaseLearner;
use boostlss::model::{BoostLss, Fitted, Scale};
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyType};

#[pyclass(module = "boostlss_py")]
pub struct BoostLssModel {
    family: PyFamily,
    mstop: usize,
    step_length: f64,
    learners: Vec<(String, BaseLearner)>,
    fitted_gaussian: Option<Fitted<GaussianLss>>,
    train_data: Option<(ndarray::Array2<f64>, ndarray::Array1<f64>)>,
}

#[pymethods]
impl BoostLssModel {
    #[new]
    #[pyo3(signature = (family, mstop=100, step_length=0.1))]
    fn new(family: PyFamily, mstop: usize, step_length: f64) -> Self {
        Self {
            family,
            mstop,
            step_length,
            learners: Vec::new(),
            fitted_gaussian: None,
            train_data: None,
        }
    }

    fn add_learner(&mut self, param: String, learner: &Bound<'_, PyAny>) -> PyResult<()> {
        let base_learner = if let Ok(l) = learner.extract::<crate::learner::PyLinearLearner>() {
            l.into()
        } else if let Ok(s) = learner.extract::<crate::learner::PyStumpLearner>() {
            s.into()
        } else if let Ok(t) = learner.extract::<crate::learner::PyTreeLearner>() {
            t.into()
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
            x_view.iter().copied().collect(),
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let y_vec =
            ndarray::Array1::from_shape_vec((y_view.len(),), y_view.iter().copied().collect())
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

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

                self.fitted_gaussian = Some(fitted);
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
            x_view.iter().copied().collect(),
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        // create dummy y for Dataset constructor requirements
        let y_dummy = ndarray::Array1::zeros(x_mat.nrows());
        let dataset = Dataset::new(x_mat, y_dummy, None)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        if let Some(fitted) = &mut self.fitted_gaussian {
            let pred = fitted
                .predict(&dataset, param, Scale::Response)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
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
            }
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Model not fitted, cannot run cvrisk without data",
            ))
        }
    }

    /// Returns the feature importance (empirical risk reduction) for each base learner.
    pub fn feature_importance(&self) -> PyResult<Vec<f64>> {
        let fitted = self.fitted_gaussian.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "Model must be fitted before calling feature_importance",
            )
        })?;
        Ok(fitted.feature_importance())
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
        let fitted = self.fitted_gaussian.as_mut().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "Model must be fitted before calling partial_dependence",
            )
        })?;

        let x_view = data.as_array();
        let x_mat = ndarray::Array2::from_shape_vec(
            (x_view.nrows(), x_view.ncols()),
            x_view.iter().copied().collect(),
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let n_samples = x_mat.nrows();
        let dummy_response = ndarray::Array1::<f64>::zeros(n_samples);

        let ds = Dataset::new(x_mat, dummy_response, None)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let pd = fitted
            .partial_dependence(&ds, param, feature_idx, &grid)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        Ok(pd)
    }

    fn __getnewargs__(&self) -> (PyFamily, usize, f64) {
        (self.family.clone(), self.mstop, self.step_length)
    }

    fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new_bound(py);

        let family_str = match self.family {
            PyFamily::GaussianLss => "GaussianLss",
        };
        dict.set_item("family", family_str)?;
        dict.set_item("mstop", self.mstop)?;
        dict.set_item("step_length", self.step_length)?;
        // Skip train_data entirely!

        if let Some(fitted) = &self.fitted_gaussian {
            let bytes = bincode::serialize(fitted)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            dict.set_item("fitted_gaussian", PyBytes::new_bound(py, &bytes))?;
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

        if let Some(bytes_any) = state.get_item("fitted_gaussian")? {
            let bytes: &[u8] = bytes_any.extract()?;
            let fitted: Fitted<GaussianLss> = bincode::deserialize(bytes)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            self.fitted_gaussian = Some(fitted);
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
