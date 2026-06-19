#!/usr/bin/env Rscript
# Requires: install.packages(c("gamboostlss", "jsonlite"))
#
# Usage: Run this script from the repo root to regenerate golden standard
# test fixtures whenever the Rust implementation needs alignment with the R package.
# Example: ./tests/generate_fixtures.R

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
model <- gamboostLSS(y ~ bols(x, intercept=TRUE), data = data, families = GaussianLSS(), control = boost_control(mstop = 2, nu = 0.1, risk = "inbag"))

# 4. Extract offsets (initial values)
offsets <- list(
  mu = model[[1]]$offset,
  sigma = model[[2]]$offset
)

offset_file <- file.path(fixture_dir, "offsets.json")
write_json(offsets, offset_file, auto_unbox = TRUE)

# 5. We could extract gradients here for early steps, but for v1
# we will just output the script as a reference point for future
# deeper integration.

cat(sprintf("Fixtures successfully generated:\n  - %s\n  - %s\n", input_file, offset_file))
