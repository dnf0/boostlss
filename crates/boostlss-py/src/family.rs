use boostlss::family::LaplaceLss;
use boostlss::family::TweedieLss;
use boostlss::family::ZINBLss;
use pyo3::prelude::*;

#[pyclass(name = "ZINBLss", module = "boostlss_py")]
#[derive(Clone)]
pub struct PyZINBLss {
    pub inner: ZINBLss,
}

#[pymethods]
impl PyZINBLss {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: ZINBLss::new(),
        }
    }

    fn __getnewargs__<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, pyo3::types::PyTuple> {
        pyo3::types::PyTuple::empty_bound(py)
    }
}

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
}

#[pymethods]
impl PyFamily {
    #[new]
    pub fn new(name: &str) -> PyResult<Self> {
        match name {
            "GaussianLSS" | "GaussianLss" => Ok(PyFamily::Gaussian),
            "BinomialLSS" | "BinomialLss" => Ok(PyFamily::Binomial),
            "BetaLSS" | "BetaLss" => Ok(PyFamily::Beta),
            "WeibullLSS" | "WeibullLss" => Ok(PyFamily::Weibull),
            "LogNormalLSS" | "LogNormalLss" => Ok(PyFamily::LogNormal),
            "ZIPLSS" | "ZIPLss" | "ZipLss" => Ok(PyFamily::Zip),
            "GEVLSS" | "GEVLss" | "GevLss" => Ok(PyFamily::Gev),
            "JSULSS" | "JSULss" | "JsuLss" => Ok(PyFamily::Jsu),
            "BURR12LSS" | "Burr12Lss" => Ok(PyFamily::Burr12),
            "GAMMALSS" | "GammaLss" => Ok(PyFamily::Gamma),
            "GEDLSS" | "GedLss" => Ok(PyFamily::Ged),
            "GPDLSS" | "GpdLss" => Ok(PyFamily::Gpd),
            "INVERSEGAUSSIANLSS" | "InverseGaussianLss" => Ok(PyFamily::InverseGaussian),
            "LOGLOGISTICLSS" | "LogLogisticLss" => Ok(PyFamily::LogLogistic),
            "NBINOMIALLSS" | "NBinomialLss" => Ok(PyFamily::NBinomial),
            "NIGLSS" | "NigLss" => Ok(PyFamily::Nig),
            "POISSONLSS" | "PoissonLss" => Ok(PyFamily::Poisson),
            "STUDENTTLSS" | "StudentTLss" => Ok(PyFamily::StudentT),
            "MULTINOMIAL" | "Multinomial" => Ok(PyFamily::Multinomial),
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
            PyFamily::Burr12 => ("Burr12Lss",),
            PyFamily::Gamma => ("GammaLss",),
            PyFamily::Ged => ("GedLss",),
            PyFamily::Gpd => ("GpdLss",),
            PyFamily::InverseGaussian => ("InverseGaussianLss",),
            PyFamily::LogLogistic => ("LogLogisticLss",),
            PyFamily::NBinomial => ("NBinomialLss",),
            PyFamily::Nig => ("NigLss",),
            PyFamily::Poisson => ("PoissonLss",),
            PyFamily::StudentT => ("StudentTLss",),
            PyFamily::Multinomial => ("MultinomialLss",),
        }
    }
}

#[pyclass(name = "TweedieLss", module = "boostlss_py")]
#[derive(Clone)]
pub struct PyTweedieLss {
    pub inner: TweedieLss,
}

#[pymethods]
impl PyTweedieLss {
    #[new]
    #[pyo3(signature = (p=1.5))]
    pub fn new(p: f64) -> Self {
        Self {
            inner: TweedieLss::new(p),
        }
    }

    fn __getnewargs__(&self) -> (f64,) {
        (self.inner.p,)
    }
}

#[pyclass(name = "LogisticLss", module = "boostlss_py")]
#[derive(Clone)]
pub struct PyLogisticLss;

#[pymethods]
impl PyLogisticLss {
    #[new]
    pub fn new() -> Self {
        Self
    }

    fn __getnewargs__<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, pyo3::types::PyTuple> {
        pyo3::types::PyTuple::empty_bound(py)
    }
}

#[pyclass(name = "LaplaceLss", module = "boostlss_py")]
#[derive(Clone)]
pub struct PyLaplaceLss {
    pub inner: LaplaceLss,
}

#[pymethods]
impl PyLaplaceLss {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: LaplaceLss::new(),
        }
    }

    fn __getnewargs__<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, pyo3::types::PyTuple> {
        pyo3::types::PyTuple::empty_bound(py)
    }
}

use boostlss::family::MertonJumpDiffusionLss;

#[pyclass(name = "MertonJumpDiffusionLss", module = "boostlss_py")]
#[derive(Clone)]
pub struct PyMertonJumpDiffusionLss {
    pub inner: MertonJumpDiffusionLss,
}

#[pymethods]
impl PyMertonJumpDiffusionLss {
    #[new]
    #[pyo3(signature = (max_jumps=10))]
    pub fn new(max_jumps: usize) -> Self {
        Self {
            inner: MertonJumpDiffusionLss::new(max_jumps),
        }
    }

    fn __getnewargs__(&self) -> (usize,) {
        (self.inner.max_jumps,)
    }
}

use boostlss::family::SHASHLss;

#[pyclass(name = "SHASHLss", module = "boostlss_py")]
#[derive(Clone)]
pub struct PySHASHLss {
    pub inner: SHASHLss,
}

#[pymethods]
impl PySHASHLss {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: SHASHLss::new(),
        }
    }

    fn __getnewargs__<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, pyo3::types::PyTuple> {
        pyo3::types::PyTuple::empty_bound(py)
    }
}
