//! boostlss — boosting GAMLSS (distributional regression) in Rust.

pub mod data;
pub mod error;
pub mod family;
pub mod param;
pub mod util;

pub use data::Dataset;
pub use error::BoostlssError;
pub use family::gamma::GammaLss;
pub use family::gaussian::GaussianLss;
pub use family::nbinomial::NBinomialLss;
pub use family::student_t::StudentTLss;
pub use family::Family;
pub use param::{IdentityLink, Link, LogLink, LogitLink, ParamSpec};
