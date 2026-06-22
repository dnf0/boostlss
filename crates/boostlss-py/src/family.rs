use pyo3::prelude::*;

#[pyclass(module = "boostlss_py")]
#[derive(Clone)]
pub enum PyFamily {
    GaussianLss,
    BinomialLss,
}

#[pymethods]
impl PyFamily {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        match name {
            "GaussianLSS" | "GaussianLss" => Ok(PyFamily::GaussianLss),
            "BinomialLSS" | "BinomialLss" => Ok(PyFamily::BinomialLss),
            _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown family: {}",
                name
            ))),
        }
    }

    fn __getnewargs__(&self) -> (&str,) {
        match self {
            PyFamily::GaussianLss => ("GaussianLSS",),
            PyFamily::BinomialLss => ("BinomialLSS",),
        }
    }
}
