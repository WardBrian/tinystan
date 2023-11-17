test_that("data arguments work", {

    out1 <- bernoulli_model$sample(BERNOULLI_DATA, num_warmup = 100, num_samples = 100)

    expect_true(mean(out1$theta) > 0.2 && mean(out1$theta) < 0.3)

    data_file <- file.path(stan_folder, "bernoulli", "bernoulli.data.json")
    out2 <- bernoulli_model$sample(data = data_file, num_warmup = 100, num_samples = 100)
    expect_true(mean(out2$theta) > 0.2 && mean(out2$theta) < 0.3)

})

test_that("save_warmup works", {

    out <- bernoulli_model$sample(BERNOULLI_DATA, num_warmup = 12, num_samples = 34,
        save_warmup = FALSE)
    expect_equal(posterior::niterations(out), 34)

    out <- bernoulli_model$sample(BERNOULLI_DATA, num_warmup = 12, num_samples = 34,
        save_warmup = TRUE)
    expect_equal(posterior::niterations(out), 12 + 34)

})

test_that("seed works", {

    out1 <- bernoulli_model$sample(BERNOULLI_DATA, seed = 123, num_warmup = 100,
        num_samples = 100)
    out2 <- bernoulli_model$sample(BERNOULLI_DATA, seed = 123, num_warmup = 100,
        num_samples = 100)

    expect_equal(out1, out2)

    out3 <- bernoulli_model$sample(BERNOULLI_DATA, seed = 456, num_warmup = 100,
        num_samples = 100)

    expect_error(expect_equal(out1, out3))

})

test_that("save_metric works", {

    data <- "{\"N\": 5}"

    out_unit <- gaussian_model$sample(data, num_warmup = 100, num_samples = 10, save_metric = TRUE,
        metric = ffistan::HMCMetric$UNIT)
    expect_equal(dim(out_unit$metric), c(4, 5))
    expect_equal(out_unit$metric, matrix(1, ncol = 5, nrow = 4))

    out_diag <- gaussian_model$sample(data, num_warmup = 100, num_samples = 10, save_metric = TRUE,
        metric = ffistan::HMCMetric$DIAGONAL)
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

test_that("init_inv_metric is used", {
    data <- "{\"N\": 3}"

    for (adapt in c(TRUE, FALSE)) {
        diag_metric <- matrix(1, 3, 2)
        # multiply only first chain
        diag_metric[, 1] <- diag_metric[, 1] * 1e+20
        out_diag <- gaussian_model$sample(data, num_chains = 2, save_warmup = TRUE,
            adapt = adapt, metric = HMCMetric$DIAGONAL, init_inv_metric = diag_metric,
            save_metric = TRUE, seed = 1234)

        divergent <- out_diag$draws$divergent__
        chain_one_divergences <- sum(posterior::subset_draws(divergent, chain = 1))
        expect_true(chain_one_divergences > ifelse(adapt, 10, 500))
        chain_two_divergences <- sum(posterior::subset_draws(divergent, chain = 2))
        expect_true(chain_two_divergences < 10)
        expect_true(chain_two_divergences < chain_one_divergences)
        expect_false(all(diag_metric == t(out_diag$metric)))

        dense_metric <- array(0, c(3, 3, 2))
        dense_metric[, , 1] <- diag(3) * 1e+20
        dense_metric[, , 2] <- diag(3)
        out_dense <- gaussian_model$sample(data, num_chains = 2, save_warmup = TRUE,
            adapt = adapt, metric = HMCMetric$DENSE, init_inv_metric = dense_metric,
            save_metric = TRUE, seed = 1234)

        divergent <- out_dense$draws$divergent__
        chain_one_divergences <- sum(posterior::subset_draws(divergent, chain = 1))
        expect_true(chain_one_divergences > ifelse(adapt, 10, 500))
        chain_two_divergences <- sum(posterior::subset_draws(divergent, chain = 2))
        expect_true(chain_two_divergences < 10)
        expect_true(chain_two_divergences < chain_one_divergences)
        expect_false(all(dense_metric == aperm(out_dense$metric, c(3, 2, 1))))
    }
})

test_that("multiple inits work", {

    init1 <- "{\"mu\": -100}"
    out1 <- multimodal_model$sample(num_chains = 2, num_warmup = 100, num_samples = 100,
        inits = init1)
    expect_true(all(out1$mu < 0))

    init2 <- "{\"mu\": 100}"
    out2 <- multimodal_model$sample(num_chains = 2, num_warmup = 100, num_samples = 100,
        inits = list(init1, init2))

    expect_true(all(posterior::subset_draws(out2$mu, chain = 1) < 0))
    expect_true(all(posterior::subset_draws(out2$mu, chain = 2) > 0))

    temp_file <- tempfile(fileext = ".json")
    write(init1, temp_file)
    out3 <- multimodal_model$sample(num_chains = 2, num_warmup = 100, num_samples = 100,
        inits = c(temp_file, init2))
    expect_true(all(posterior::subset_draws(out3$mu, chain = 1) < 0))
    expect_true(all(posterior::subset_draws(out3$mu, chain = 2) > 0))

})

test_that("bad data handled properly", {

    data <- "{\"N\": -1}"
    expect_error(bernoulli_model$sample(data), "greater than or equal to 0")

    data <- "{\"N\": 1, \"y\": [0, 1]}"
    expect_error(bernoulli_model$sample(data), "mismatch in dimension")

    expect_error(bernoulli_model$sample("{\"bad\"}"), "Error in JSON parsing")

    expect_error(bernoulli_model$sample("not/real/path.json"), "Could not open data file")

})

test_that("bad inits handled properly", {

    init1 <- "{\"theta\": 2}"
    expect_error(bernoulli_model$sample(BERNOULLI_DATA, inits = init1), "Initialization failed")

    expect_error(bernoulli_model$sample(BERNOULLI_DATA, inits = "bad/path.json"),
        "Could not open data file")

    init2 <- "{\"theta\": 0.5}"
    expect_error(bernoulli_model$sample(BERNOULLI_DATA, num_chains = 2, inits = c(init2,
        init1)), "Initialization failed")

    inits <- list(init2, init2)
    expect_error(bernoulli_model$sample(BERNOULLI_DATA, num_chains = 1, inits = inits),
        "match the number of chains")
    expect_error(bernoulli_model$sample(BERNOULLI_DATA, num_chains = 3, inits = inits),
        "match the number of chains")

})

test_that("bad initial metric shape handled properly", {
    data <- "{\"N\": 5}"

    expect_error(gaussian_model$sample(data, metric = ffistan::HMCMetric$DENSE, init_inv_metric = rep(1,
        5)), "Invalid initial metric size")

    expect_error(gaussian_model$sample(data, metric = ffistan::HMCMetric$DENSE, init_inv_metric = matrix(1,
        5, 4)), "Invalid initial metric size")

    expect_error(gaussian_model$sample(data, num_chains = 4, metric = ffistan::HMCMetric$DENSE,
        init_inv_metric = array(1, c(5, 5, 3))), "Invalid initial metric size")

    expect_error(gaussian_model$sample(data, metric = ffistan::HMCMetric$DIAGONAL,
        init_inv_metric = rep(1, 4)), "Invalid initial metric size")

    expect_error(gaussian_model$sample(data, num_chains = 4, metric = ffistan::HMCMetric$DIAGONAL,
        init_inv_metric = matrix(1, 5, 3)), "Invalid initial metric size")

    expect_error(gaussian_model$sample(data, num_chains = 4, metric = ffistan::HMCMetric$DIAGONAL,
        init_inv_metric = array(1, c(5, 5, 3))), "Invalid initial metric size")
})

test_that("bad initial metric handled properly", {
    data <- "{\"N\": 3}"

    expect_error(gaussian_model$sample(data, metric = ffistan::HMCMetric$DENSE, init_inv_metric = matrix(1e+20,
        3, 3)), "not positive definite")

    metric = array(0, c(3, 3, 2))
    metric[, , 1] <- 1e+20
    metric[, , 2] <- diag(3)
    expect_error(gaussian_model$sample(data, num_chains = 2, metric = ffistan::HMCMetric$DENSE,
        init_inv_metric = metric), "not positive definite")

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

