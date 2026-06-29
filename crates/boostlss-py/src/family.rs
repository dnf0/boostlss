use pyo3::prelude::*;

#[pyclass(module = "boostlss_py")]
#[derive(Clone)]
pub enum PyFamily {
    Gaussian,
    Binomial,
    Beta,
    Weibull,
    LogNormal,
    Zip,
    Gev,
    Jsu,
}

#[pymethods]
impl PyFamily {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        match name {
            "GaussianLSS" | "GaussianLss" => Ok(PyFamily::Gaussian),
            "BinomialLSS" | "BinomialLss" => Ok(PyFamily::Binomial),
            "BetaLSS" | "BetaLss" => Ok(PyFamily::Beta),
            "WeibullLSS" | "WeibullLss" => Ok(PyFamily::Weibull),
            "LogNormalLSS" | "LogNormalLss" => Ok(PyFamily::LogNormal),
            "ZIPLSS" | "ZIPLss" | "ZipLss" => Ok(PyFamily::Zip),
            "GEVLSS" | "GEVLss" | "GevLss" => Ok(PyFamily::Gev),
            "JSULSS" | "JSULss" | "JsuLss" => Ok(PyFamily::Jsu),
            _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown family: {}",
                name
            ))),
        }
    }

    fn __getnewargs__(&self) -> (&str,) {
        match self {
            PyFamily::Gaussian => ("GaussianLSS",),
            PyFamily::Binomial => ("BinomialLSS",),
            PyFamily::Beta => ("BetaLSS",),
            PyFamily::Weibull => ("WeibullLSS",),
            PyFamily::LogNormal => ("LogNormalLSS",),
            PyFamily::Zip => ("ZIPLSS",),
            PyFamily::Gev => ("GEVLSS",),
            PyFamily::Jsu => ("JSULSS",),
        }
    }
}
