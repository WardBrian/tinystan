library(testthat)

stan_folder <- file.path("..", "..", "..", "..", "test_models")

bernoulli_model <- ffistan::FFIStanModel$new(file.path(stan_folder, "bernoulli",
    "bernoulli_model.so"))
BERNOULLI_DATA <- '{"N": 10, "y": [0, 1, 0, 0, 0, 0, 0, 0, 0, 1]}'

gaussian_model <- ffistan::FFIStanModel$new(file.path(stan_folder, "gaussian", "gaussian_model.so"))

empty_model <- ffistan::FFIStanModel$new(file.path(stan_folder, "empty", "empty_model.so"))
multimodal_model <- ffistan::FFIStanModel$new(file.path(stan_folder, "multimodal",
    "multimodal_model.so"))
simple_jacobian_model <- ffistan::FFIStanModel$new(file.path(stan_folder, "simple_jacobian",
    "simple_jacobian_model.so"))


test_that("data arguments work", {

    out1 <- bernoulli_model$sample(BERNOULLI_DATA, num_warmup = 100, num_samples = 100)

    expect_true(mean(out1$draws[, , 8]) > 0.2 && mean(out1$draws[, , 8]) < 0.3)

    data_file <- file.path(stan_folder, "bernoulli", "bernoulli.data.json")
    out2 <- bernoulli_model$sample(data = data_file, num_warmup = 100, num_samples = 100)
    expect_true(mean(out2$draws[, , 8]) > 0.2 && mean(out2$draws[, , 8]) < 0.3)

})

test_that("save_warmup works", {

    out <- bernoulli_model$sample(BERNOULLI_DATA, num_warmup = 12, num_samples = 34,
        save_warmup = FALSE)
    expect_equal(dim(out$draws)[2], 34)

    out <- bernoulli_model$sample(BERNOULLI_DATA, num_warmup = 12, num_samples = 34,
        save_warmup = TRUE)
    expect_equal(dim(out$draws)[2], 12 + 34)

})

test_that("seed works", {

    out1 <- bernoulli_model$sample(BERNOULLI_DATA, seed = 123, num_warmup = 100,
        num_samples = 100)
    out2 <- bernoulli_model$sample(BERNOULLI_DATA, seed = 123, num_warmup = 100,
        num_samples = 100)

    expect_equal(out1$draws, out2$draws)

    out3 <- bernoulli_model$sample(BERNOULLI_DATA, seed = 456, num_warmup = 100,
        num_samples = 100)

    expect_error(expect_equal(out1$draws, out3$draws))

})

test_that("save_metric works", {

    data <- '{"N": 5}'

    out_unit <- gaussian_model$sample(data, num_warmup = 100, num_samples = 10, save_metric = TRUE,
        metric = ffistan::HMCMetric$UNIT)
    expect_equal(dim(out_unit$metric), c(4, 5))
    expect_equal(out_unit$metric, matrix(1, ncol = 5, nrow = 4))

    out_diag <- gaussian_model$sample(data, num_warmup = 100, num_samples = 10, save_metric = TRUE,
        metric = ffistan::HMCMetric$DIAG)
    expect_equal(dim(out_diag$metric), c(4, 5))
    expect_equal(out_diag$metric, matrix(1, ncol = 5, nrow = 4))

    out_dense <- gaussian_model$sample(data, num_warmup = 100, num_samples = 10,
        save_metric = TRUE, metric = ffistan::HMCMetric$DENSE)
    expect_equal(dim(out_dense$metric), c(4, 5, 5))
    four_identities <- aperm(array(rep(diag(5), 4), c(5, 5, 4)), c(3, 2, 1))
    expect_equal(out_dense$metric, four_identities)

    out_nometric <- gaussian_model$sample(data, num_warmup = 10, num_samples = 10,
        save_metric = FALSE)
    expect_false(exists("metric", out_nometric))

})

test_that("multiple inits work", {

    init1 <- '{"mu": -10}'
    out1 <- multimodal_model$sample(num_chains = 2, num_warmup = 100, num_samples = 100,
        inits = init1)
    expect_true(all(out1$draws[, , 8] < 0))

    init2 <- '{"mu": 10}'
    out2 <- multimodal_model$sample(num_chains = 2, num_warmup = 100, num_samples = 100,
        inits = list(init1, init2))
    expect_true(all(out2$draws[1, , 8] < 0))
    expect_true(all(out2$draws[2, , 8] > 0))

    temp_file <- tempfile(fileext = ".json")
    write(init1, temp_file)
    out3 <- multimodal_model$sample(num_chains = 2, num_warmup = 100, num_samples = 100,
        inits = c(temp_file, init2))
    expect_true(all(out3$draws[1, , 8] < 0))
    expect_true(all(out3$draws[2, , 8] > 0))

})

test_that("bad data handled properly", {

    data <- '{"N": -1}'
    expect_error(bernoulli_model$sample(data), "greater than or equal to 0")

    data <- '{"N\": 1, \"y": [0, 1]}'
    expect_error(bernoulli_model$sample(data), "mismatch in dimension")

    expect_error(bernoulli_model$sample('{"bad"}'), "Error in JSON parsing")

    expect_error(bernoulli_model$sample("not/real/path.json"), "Could not open data file")

})

test_that("bad inits handled properly", {

    init1 <- '{"theta": 2}'
    expect_error(bernoulli_model$sample(BERNOULLI_DATA, inits = init1), "Initialization failed")

    expect_error(bernoulli_model$sample(BERNOULLI_DATA, inits = "bad/path.json"),
        "Could not open data file")

    init2 <- '{"theta": 0.5}'
    expect_error(bernoulli_model$sample(BERNOULLI_DATA, num_chains = 2, inits = c(init2,
        init1)), "Initialization failed")

    inits <- list(init2, init2)
    expect_error(bernoulli_model$sample(BERNOULLI_DATA, num_chains = 1, inits = inits),
        "match the number of chains")
    expect_error(bernoulli_model$sample(BERNOULLI_DATA, num_chains = 3, inits = inits),
        "match the number of chains")

})

test_that("bad num_warmup handled properly", {

    expect_error(bernoulli_model$sample(BERNOULLI_DATA, num_warmup = -1), "non-negative")
    expect_error(bernoulli_model$sample(BERNOULLI_DATA, save_warmup = TRUE, num_warmup = -1),
        "non-negative")

})

test_that("model with no params fails", {

    expect_error(empty_model$sample(), "Model has no parameters")

})

test_that("bad args raise errors", {

    expect_error(bernoulli_model$sample(BERNOULLI_DATA, num_chains = 0), "at least 1")
    expect_error(bernoulli_model$sample(BERNOULLI_DATA, num_samples = 0), "at least 1")
    expect_error(bernoulli_model$sample(BERNOULLI_DATA, id = 0), "positive")
    expect_error(bernoulli_model$sample(BERNOULLI_DATA, init_radius = -0.1), "non-negative")

    expect_error(bernoulli_model$sample(BERNOULLI_DATA, delta = -0.1), "between 0 and 1")
    expect_error(bernoulli_model$sample(BERNOULLI_DATA, delta = 1.1), "between 0 and 1")

    expect_error(bernoulli_model$sample(BERNOULLI_DATA, gamma = 0), "positive")
    expect_error(bernoulli_model$sample(BERNOULLI_DATA, kappa = 0), "positive")
    expect_error(bernoulli_model$sample(BERNOULLI_DATA, t0 = 0), "positive")

    expect_error(bernoulli_model$sample(BERNOULLI_DATA, stepsize = 0), "positive")
    expect_error(bernoulli_model$sample(BERNOULLI_DATA, stepsize_jitter = -0.1),
        "between 0 and 1")
    expect_error(bernoulli_model$sample(BERNOULLI_DATA, stepsize_jitter = 1.1), "between 0 and 1")

    expect_error(bernoulli_model$sample(BERNOULLI_DATA, max_depth = 0), "positive")
    expect_error(bernoulli_model$sample(BERNOULLI_DATA, num_threads = 0), "positive")

})

