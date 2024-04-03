
bernoulli_file <- file.path(stan_folder, "bernoulli", "bernoulli.stan")
test_that("model loads", {
    suppressWarnings(model <- tinystan_model(bernoulli_file))
    expect_true(!is.null(model))
})

test_that("api_version is correct", {
    expect_equal(api_version(bernoulli_model), list(major = 0, minor = 1, patch = 0))
})
