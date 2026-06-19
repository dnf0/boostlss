mod family;
mod learner;

use family::PyFamily;
use learner::PyLinearLearner;
use pyo3::prelude::*;

#[pymodule]
fn boostlss_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFamily>()?;
    m.add_class::<PyLinearLearner>()?;
    Ok(())
}
