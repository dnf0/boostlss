use pyo3::prelude::*;

#[pyclass(module = "boostlss_py")]
#[derive(Clone)]
pub enum PyFamily {
    GaussianLss,
}

#[pymethods]
impl PyFamily {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        match name {
            "GaussianLSS" => Ok(PyFamily::GaussianLss),
            _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown family: {}",
                name
            ))),
        }
    }

    fn __getnewargs__(&self) -> (&str,) {
        match self {
            PyFamily::GaussianLss => ("GaussianLSS",),
        }
    }
}
