mod data;
mod family;
mod learner;
mod model;
pub mod stabsel;

use family::{PyFamily, PyTweedieLss};
use learner::{
    PyBivariatePSplineLearner, PyConstrainedPSplineLearner, PyHistTreeLearner, PyLinearLearner,
    PyPSplineLearner, PyStumpLearner, PyTreeLearner,
};
use model::BoostLssModel;
use pyo3::prelude::*;

#[pymodule]
fn boostlss_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFamily>()?;
    m.add_class::<PyTweedieLss>()?;
    m.add_class::<PyLinearLearner>()?;
    m.add_class::<PyStumpLearner>()?;
    m.add_class::<PyTreeLearner>()?;
    m.add_class::<PyHistTreeLearner>()?;
    m.add_class::<PyPSplineLearner>()?;
    m.add_class::<PyBivariatePSplineLearner>()?;
    m.add_class::<PyConstrainedPSplineLearner>()?;
    m.add_class::<BoostLssModel>()?;
    m.add_class::<model::PyRandomEffectsLearner>()?;
    m.add_class::<stabsel::PyStabselResult>()?;
    m.add_function(wrap_pyfunction!(learner::constrained_pspline, m)?)?;
    Ok(())
}
