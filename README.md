# boostlss

`boostlss` is an idiomatic Rust library for boosting GAMLSS (Generalized Additive Models for Location, Scale and Shape).

This project is deeply inspired by the `gamboostlss` R package (authored by Thomas Kneib, Andreas Mayr, et al.). Our primary goal is to provide a highly performant, thread-safe core algorithmic engine in Rust, complete with native Python bindings to bring robust distributional regression to the modern Python data science ecosystem.

## Status

Currently in early development. The foundations (data structures, link functions, distribution families including Gaussian, Student-T, Gamma, and Negative Binomial) and core engine architectures have been implemented.

## Validation and Testing

To ensure strict mathematical correctness during the port, `boostlss` is continuously validated against the original R `gamboostlss` implementation. We use "golden tests"—pre-generated data fixtures output by the R package—to verify that our Rust gradient and loss calculations match the original algorithms down to tight floating-point tolerances.
