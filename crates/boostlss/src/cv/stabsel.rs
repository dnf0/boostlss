use crate::error::BoostlssError;

#[derive(Clone, Debug, PartialEq)]
pub enum StabselMode {
    Joint,
    Independent,
}

#[derive(Clone, Debug)]
pub struct StabselConfig {
    pub b: usize,
    pub pfer: Option<f64>,
    pub pi_thr: Option<f64>,
    pub q: Option<usize>,
    pub mode: StabselMode,
    pub p: usize,
}

impl StabselConfig {
    pub fn new(
        b: usize,
        pfer: Option<f64>,
        pi_thr: Option<f64>,
        q: Option<usize>,
        mode: StabselMode,
        p: usize,
    ) -> Result<Self, BoostlssError> {
        let provided = vec![pfer.is_some(), pi_thr.is_some(), q.is_some()]
            .into_iter()
            .filter(|&x| x)
            .count();

        if provided != 2 {
            return Err(BoostlssError::InvalidStabselConfig(
                "Exactly two of (pfer, pi_thr, q) must be provided".to_string(),
            ));
        }

        let mut config = Self {
            b,
            pfer,
            pi_thr,
            q,
            mode,
            p,
        };

        config.resolve_bounds()?;
        Ok(config)
    }

    #[allow(clippy::unnecessary_unwrap)]
    fn resolve_bounds(&mut self) -> Result<(), BoostlssError> {
        // Shah & Samworth (2013) bounds: PFER <= q^2 / ((2 * pi_thr - 1) * p)
        if self.pfer.is_none() {
            let q = self.q.unwrap() as f64;
            let pi_thr = self.pi_thr.unwrap();
            if pi_thr <= 0.5 || pi_thr >= 1.0 {
                return Err(BoostlssError::InvalidStabselConfig(
                    "pi_thr must be in (0.5, 1.0)".to_string(),
                ));
            }
            self.pfer = Some((q * q) / ((2.0 * pi_thr - 1.0) * self.p as f64));
        } else if self.pi_thr.is_none() {
            let q = self.q.unwrap() as f64;
            let pfer = self.pfer.unwrap();
            if pfer <= 0.0 {
                return Err(BoostlssError::InvalidStabselConfig(
                    "pfer must be > 0.0".to_string(),
                ));
            }
            self.pi_thr = Some(((q * q) / (pfer * self.p as f64) + 1.0) / 2.0);
            if self.pi_thr.unwrap() <= 0.5 || self.pi_thr.unwrap() >= 1.0 {
                return Err(BoostlssError::InvalidStabselConfig(
                    "Derived pi_thr must be in (0.5, 1.0). Adjust q or pfer.".to_string(),
                ));
            }
        } else if self.q.is_none() {
            let pfer = self.pfer.unwrap();
            let pi_thr = self.pi_thr.unwrap();
            if pi_thr <= 0.5 || pi_thr >= 1.0 || pfer <= 0.0 {
                return Err(BoostlssError::InvalidStabselConfig(
                    "Invalid pi_thr or pfer".to_string(),
                ));
            }
            let q_f64 = (pfer * (2.0 * pi_thr - 1.0) * self.p as f64).sqrt();
            self.q = Some(q_f64.floor() as usize);
            if self.q.unwrap() == 0 {
                return Err(BoostlssError::InvalidStabselConfig(
                    "Derived q is 0. Adjust pi_thr or pfer.".to_string(),
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invalid_config_not_two_params() {
        let err = StabselConfig::new(100, Some(1.0), Some(0.6), Some(10), StabselMode::Joint, 100)
            .unwrap_err();
        match err {
            BoostlssError::InvalidStabselConfig(msg) => {
                assert_eq!(msg, "Exactly two of (pfer, pi_thr, q) must be provided");
            }
            _ => panic!("Expected InvalidStabselConfig"),
        }
    }

    #[test]
    fn test_resolve_pfer() {
        let config =
            StabselConfig::new(100, None, Some(0.6), Some(10), StabselMode::Joint, 100).unwrap();
        assert_eq!(config.q.unwrap(), 10);
        assert_eq!(config.pi_thr.unwrap(), 0.6);
        assert!((config.pfer.unwrap() - 5.0).abs() < 1e-6); // 100 / (0.2 * 100) = 100 / 20 = 5.0
    }

    #[test]
    fn test_resolve_pi_thr() {
        let config =
            StabselConfig::new(100, Some(5.0), None, Some(10), StabselMode::Joint, 100).unwrap();
        assert_eq!(config.q.unwrap(), 10);
        assert_eq!(config.pfer.unwrap(), 5.0);
        assert!((config.pi_thr.unwrap() - 0.6).abs() < 1e-6); // (100 / (5 * 100) + 1) / 2 = (1/5 + 1)/2 = 0.6
    }

    #[test]
    fn test_resolve_q() {
        let config =
            StabselConfig::new(100, Some(5.0), Some(0.7), None, StabselMode::Joint, 100).unwrap();
        assert_eq!(config.pfer.unwrap(), 5.0);
        assert_eq!(config.pi_thr.unwrap(), 0.7);
        assert_eq!(config.q.unwrap(), 14); // sqrt(5 * 0.4 * 100) = sqrt(200) = 14.14 -> floor is 14
    }

    #[test]
    fn test_invalid_pi_thr() {
        let err = StabselConfig::new(100, None, Some(0.4), Some(10), StabselMode::Joint, 100)
            .unwrap_err();
        match err {
            BoostlssError::InvalidStabselConfig(msg) => {
                assert_eq!(msg, "pi_thr must be in (0.5, 1.0)");
            }
            _ => panic!("Expected InvalidStabselConfig"),
        }
    }
}
