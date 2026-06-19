//! Boosting algorithms and configuration.

pub mod cyclical;
pub mod stabilization;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Algorithm {
    Cyclic,
    NonCyclic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stabilization {
    None,
    Mad,
    L2,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Mstop {
    Scalar(usize),
    PerParam(Vec<usize>),
}

#[derive(Debug, Clone)]
pub struct Config {
    pub algorithm: Algorithm,
    pub step_length: f64,
    pub stabilization: Stabilization,
    pub mstop: Mstop,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            algorithm: Algorithm::Cyclic,
            step_length: 0.1,
            stabilization: Stabilization::None,
            mstop: Mstop::Scalar(100),
        }
    }
}
