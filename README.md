# boostlss

`boostlss` is an idiomatic Rust library for boosting GAMLSS (Generalized Additive Models for Location, Scale and Shape).

This project is deeply inspired by the `gamboostlss` R implementation. Our goal is to provide a highly performant, thread-safe core engine in Rust, complete with native Python bindings to bring robust distributional regression to the modern Python data science ecosystem.

## Status

Currently in early development. The foundations (data structures, link functions, distribution families including Gaussian, Student-T, Gamma, and Negative Binomial) have been implemented.

Next steps involve building out the base learners, the core boosting engine, and validating our outputs against golden tests from the original `gamboostlss` R package.
