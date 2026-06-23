mod family;
mod learner;
mod model;

use family::PyFamily;
use learner::{PyLinearLearner, PyStumpLearner, PyTreeLearner};
use model::BoostLssModel;
use pyo3::prelude::*;

#[pymodule]
fn boostlss_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFamily>()?;
    m.add_class::<PyLinearLearner>()?;
    m.add_class::<PyStumpLearner>()?;
    m.add_class::<PyTreeLearner>()?;
    m.add_class::<BoostLssModel>()?;
    m.add_class::<model::PyRandomEffectsLearner>()?;
    Ok(())
}
