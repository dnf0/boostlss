pub trait Link: std::fmt::Debug {
    /// Maps from distribution parameter (theta) to additive predictor (eta)
    fn link(&self, theta: f64) -> f64;
    /// Maps from additive predictor (eta) to distribution parameter (theta)
    fn response(&self, eta: f64) -> f64;
    /// Derivative d(eta)/d(theta)
    fn deriv(&self, theta: f64) -> f64;
}

#[derive(Debug)]
pub struct IdentityLink;
impl Link for IdentityLink {
    fn link(&self, theta: f64) -> f64 {
        theta
    }
    fn response(&self, eta: f64) -> f64 {
        eta
    }
    fn deriv(&self, _theta: f64) -> f64 {
        1.0
    }
}

#[derive(Debug)]
pub struct LogLink;
impl Link for LogLink {
    fn link(&self, theta: f64) -> f64 {
        theta.ln()
    }
    fn response(&self, eta: f64) -> f64 {
        eta.exp()
    }
    fn deriv(&self, theta: f64) -> f64 {
        1.0 / theta
    }
}

#[derive(Debug)]
pub struct LogitLink;
impl Link for LogitLink {
    fn link(&self, theta: f64) -> f64 {
        (theta / (1.0 - theta)).ln()
    }
    fn response(&self, eta: f64) -> f64 {
        1.0 / (1.0 + (-eta).exp())
    }
    fn deriv(&self, theta: f64) -> f64 {
        1.0 / (theta * (1.0 - theta))
    }
}

#[derive(Debug)]
pub struct ParamSpec {
    pub name: String,
    pub link: Box<dyn Link + Send + Sync>,
}

impl ParamSpec {
    pub fn new(name: impl Into<String>, link: impl Link + Send + Sync + 'static) -> Self {
        Self {
            name: name.into(),
            link: Box::new(link),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn identity_link_inverses() {
        let link = IdentityLink;
        assert_relative_eq!(link.response(link.link(5.0)), 5.0, epsilon = 1e-12);
        assert_relative_eq!(link.deriv(5.0), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn log_link_inverses() {
        let link = LogLink;
        assert_relative_eq!(link.response(link.link(5.0)), 5.0, epsilon = 1e-12);
        assert_relative_eq!(link.deriv(5.0), 0.2, epsilon = 1e-12);
    }

    #[test]
    fn logit_link_inverses() {
        let link = LogitLink;
        assert_relative_eq!(link.response(link.link(0.25)), 0.25, epsilon = 1e-12);
        assert_relative_eq!(link.deriv(0.25), 1.0 / (0.25 * 0.75), epsilon = 1e-12);
    }
}
