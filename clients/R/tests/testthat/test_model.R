stan_folder <- file.path("..", "..", "..", "..", "test_models")

bernoulli_file <- file.path(stan_folder, "bernoulli", "bernoulli_model.so")

test_that("model loads", {
    suppressWarnings(model <- StanModel$new(bernoulli_file))

    expect_true(!is.null(model))
})

test_that("api_version is correct", {

    expect_equal(bernoulli_model$api_version(), list(major = 0, minor = 1, patch = 0))
})
