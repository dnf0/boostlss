use thiserror::Error;

#[derive(Error, Debug)]
pub enum BoostlssError {
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Data validation error: {0}")]
    DataError(String),

    #[error("Model not converged: {0}")]
    NotConverged(String),

    #[error("Value out of range: {0}")]
    OutOfRange(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Invalid stability selection config: {0}")]
    InvalidStabselConfig(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err1 = BoostlssError::InvalidConfig("bad param".to_string());
        assert_eq!(err1.to_string(), "Invalid configuration: bad param");

        let err2 = BoostlssError::DataError("NaN found".to_string());
        assert_eq!(err2.to_string(), "Data validation error: NaN found");

        let err3 = BoostlssError::NotConverged("max iter reached".to_string());
        assert_eq!(err3.to_string(), "Model not converged: max iter reached");

        let err4 = BoostlssError::InvalidStabselConfig("bad".to_string());
        assert_eq!(err4.to_string(), "Invalid stability selection config: bad");
    }
}
