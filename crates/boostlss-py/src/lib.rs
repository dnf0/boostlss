mod family;
mod learner;
mod model;

use family::PyFamily;
use learner::PyLinearLearner;
use model::BoostLssModel;
use pyo3::prelude::*;

#[pymodule]
fn boostlss_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFamily>()?;
    m.add_class::<PyLinearLearner>()?;
    m.add_class::<BoostLssModel>()?;
    Ok(())
}
