# boostlss

`boostlss` is an idiomatic Rust library for boosting GAMLSS (Generalized Additive Models for Location, Scale and Shape) with native Python bindings.

This project is deeply inspired by the `gamboostlss` R package (authored by Thomas Kneib, Andreas Mayr, et al.). Our primary goal is to provide a highly performant, thread-safe core algorithmic engine in Rust, complete with native Python bindings to bring robust distributional regression to the modern Python data science ecosystem.

## Features

- **High Performance**: Core engine written in Rust for maximum speed and memory safety.
- **Distributional Regression**: Model the full conditional distribution, not just the mean.
- **Flexible Algorithms**: Choose from cyclic, non-cyclic, and non-cyclic-outer boosting algorithms.
- **Rich Family Support**: Gaussian, Student-T, Gamma, Binomial, Beta, Weibull, LogNormal, ZIP, and GEV.
- **Diverse Base Learners**: Linear, P-Splines, Constrained P-Splines, Random Effects, Stumps, and Trees.
- **Sparse Matrix Support**: Native support for SciPy sparse matrices (CSR/CSC) for memory-efficient training on high-dimensional data.
- **Advanced Tooling**: Built-in cross-validation and stability selection (`stabsel`).

## Installation

You can install `boostlss` directly from PyPI:

```bash
pip install boostlss-py
```

_Note: The Python package name is `boostlss-py` but you import it as `boostlss_py`._

Please continue to the [Quickstart](quickstart.md) to learn how to use the library.
