use pyo3::prelude::*;
use std::collections::HashMap;

#[pyclass(module = "boostlss_py")]
#[derive(Clone)]
pub struct PyStabselResult {
    #[pyo3(get)]
    pub selected_joint: Vec<String>,
    #[pyo3(get)]
    pub selected_independent: HashMap<String, Vec<String>>,
    #[pyo3(get)]
    pub probabilities: HashMap<String, HashMap<String, f64>>,
    #[pyo3(get)]
    pub q: usize,
    #[pyo3(get)]
    pub pfer: f64,
    #[pyo3(get)]
    pub pi_thr: f64,
    #[pyo3(get)]
    pub b: usize,
}
