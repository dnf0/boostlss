use crate::family::PyFamily;
use crate::learner::PyLinearLearner;
use boostlss::data::Dataset;
use boostlss::engine::cyclical::fit_cyclical;
use boostlss::engine::Mstop;
use boostlss::family::GaussianLss;
use boostlss::model::{BoostLss, Fitted, Scale};
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

#[pyclass]
pub struct BoostLssModel {
    family: PyFamily,
    mstop: usize,
    step_length: f64,
    learners: Vec<(String, PyLinearLearner)>,
    fitted_gaussian: Option<Fitted<GaussianLss>>,
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
        }
    }

    fn add_learner(&mut self, param: String, learner: PyLinearLearner) {
        self.learners.push((param, learner));
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

        let dataset = Dataset::new(x_mat, y_vec, None)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        match self.family {
            PyFamily::GaussianLss => {
                let mut model = BoostLss::new(GaussianLss::new())
                    .step_length(self.step_length)
                    .mstop(Mstop::Scalar(self.mstop));

                for (param, learner) in &self.learners {
                    model = model
                        .on(param.as_str(), learner.clone().into())
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
}
