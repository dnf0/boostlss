use crate::data::Dataset;
use crate::error::BoostlssError;
use crate::family::Family;
use crate::model::{BoostLss, Fitted};

pub fn fit_cyclical<F: Family>(
    _model: &BoostLss<F>,
    _data: &Dataset,
) -> Result<Fitted<F>, BoostlssError> {
    // 1. Initialize offsets
    // 2. Loop m = 1..max(mstop)
    // 3. For each param k:
    //      Compute ngradient
    //      Fit all learners for k
    //      Select best by RSS
    //      Update eta_k += nu * u_hat
    todo!("Cyclical fit unimplemented")
}
