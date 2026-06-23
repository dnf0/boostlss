use pyo3::prelude::*;

#[pyclass(module = "boostlss_py")]
#[derive(Clone)]
pub enum PyFamily {
    GaussianLss,
    BinomialLss,
    BetaLss,
    WeibullLss,
    LogNormalLss,
    ZIPLss,
    GEVLss,
}

#[pymethods]
impl PyFamily {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        match name {
            "GaussianLSS" | "GaussianLss" => Ok(PyFamily::GaussianLss),
            "BinomialLSS" | "BinomialLss" => Ok(PyFamily::BinomialLss),
            "BetaLSS" | "BetaLss" => Ok(PyFamily::BetaLss),
            "WeibullLSS" | "WeibullLss" => Ok(PyFamily::WeibullLss),
            "LogNormalLSS" | "LogNormalLss" => Ok(PyFamily::LogNormalLss),
            "ZIPLSS" | "ZIPLss" => Ok(PyFamily::ZIPLss),
            "GEVLSS" | "GEVLss" => Ok(PyFamily::GEVLss),
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
            PyFamily::BetaLss => ("BetaLSS",),
            PyFamily::WeibullLss => ("WeibullLSS",),
            PyFamily::LogNormalLss => ("LogNormalLSS",),
            PyFamily::ZIPLss => ("ZIPLSS",),
            PyFamily::GEVLss => ("GEVLSS",),
        }
    }
}
