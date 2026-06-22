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
use pyo3::types::PyDict;

#[pyclass]
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
}
