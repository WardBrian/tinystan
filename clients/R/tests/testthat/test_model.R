library(testthat)
library(ffistan)

stan_folder <- file.path("..", "..", "..", "..", "test_models")

bernoulli_model <- file.path(stan_folder, "bernoulli", "bernoulli_model.so")

test_that("model loads", {
    model <- FFIStanModel$new(bernoulli_model)

    expect_true(!is.null(model))
})

test_that("api_version is correct", {
    model <- ffistan::FFIStanModel$new(bernoulli_model)

    expect_equal(model$api_version(), list(major = 0, minor = 1, patch = 0))
})
