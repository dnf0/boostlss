#!/usr/bin/env Rscript
# Requires: install.packages(c("gamboostlss", "jsonlite"))
#
# Usage: Run this script from the repo root to regenerate golden standard
# test fixtures whenever the Rust implementation needs alignment with the R package.
# Example: ./tests/generate_fixtures.R

if (!requireNamespace("gamboostlss", quietly = TRUE) || !requireNamespace("jsonlite", quietly = TRUE)) {
  stop("Please run: install.packages(c('gamboostlss', 'jsonlite'))")
}

library(gamboostlss)
library(jsonlite)

# Verify execution context
if (!dir.exists("crates/boostlss")) {
  stop("Error: This script must be run from the repository root directory.")
}

# Ensure target directory exists
fixture_dir <- "crates/boostlss/tests/fixtures"
dir.create(fixture_dir, recursive = TRUE, showWarnings = FALSE)

set.seed(42)

# 1. Generate synthetic data
n <- 100
x <- runif(n, -3, 3)
mu <- 2 * x
sigma <- exp(0.5 * x)
y <- rnorm(n, mean = mu, sd = sigma)

data <- data.frame(x = x, y = y)

# 2. Export input data
input_file <- file.path(fixture_dir, "input_data.json")
write_json(list(
  x = x,
  y = y
), input_file, auto_unbox = TRUE)

# 3. Fit GaussianLSS model (no stabilization to match Rust defaults for comparison)
model <- gamboostLSS(y ~ bols(x, intercept=TRUE), data = data, families = GaussianLSS(stabilization = "none"), control = boost_control(mstop = 2, nu = 0.1, risk = "inbag"))

# 4. Extract offsets (initial values)
offsets <- list(
  mu = model$mu$offset,
  sigma = model$sigma$offset
)

offset_file <- file.path(fixture_dir, "offsets.json")
write_json(offsets, offset_file, auto_unbox = TRUE)

# 5. Extract predictions and gradients
# For mstop = 2
predictions <- list(
  mu = predict(model, parameter = "mu"),
  sigma = predict(model, parameter = "sigma")
)
predictions_file <- file.path(fixture_dir, "predictions.json")
write_json(predictions, predictions_file, auto_unbox = TRUE)

# 6. Extract gradients (pseudo-residuals) for the first iteration (at mstop=1)
# We can set mstop to 1 to get the state after the first iteration
model[1]
gradients <- list(
  mu = model$mu$resid(),
  sigma = model$sigma$resid()
)
gradients_file <- file.path(fixture_dir, "gradients.json")
write_json(gradients, gradients_file, auto_unbox = TRUE)

cat(sprintf("Fixtures successfully generated:\n  - %s\n  - %s\n  - %s\n  - %s\n", input_file, offset_file, predictions_file, gradients_file))
