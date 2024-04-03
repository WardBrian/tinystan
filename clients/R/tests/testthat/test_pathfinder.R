test_that("data arguments work", {

    out1 <- pathfinder(bernoulli_model, BERNOULLI_DATA)
    expect_true(mean(out1$theta) > 0.2 && mean(out1$theta) < 0.3)

    data_file <- file.path(stan_folder, "bernoulli", "bernoulli.data.json")
    out2 <- pathfinder(bernoulli_model, data = data_file)
    expect_true(mean(out2$theta) > 0.2 && mean(out2$theta) < 0.3)

})


test_that("output sizes are correct", {

    out1 <- pathfinder(bernoulli_model, BERNOULLI_DATA, num_paths = 4, num_draws = 101,
        num_multi_draws = 99)
    expect_equal(posterior::ndraws(out1), 99)

    out2 <- pathfinder(bernoulli_model, BERNOULLI_DATA, num_paths = 1, num_draws = 101,
        num_multi_draws = 103)
    expect_equal(posterior::ndraws(out2), 103)

    out3 <- pathfinder(bernoulli_model, BERNOULLI_DATA, num_paths = 2, num_draws = 105,
        num_multi_draws = 1, calculate_lp = FALSE)
    expect_equal(posterior::ndraws(out3), 2 * 105)

    out4 <- pathfinder(bernoulli_model, BERNOULLI_DATA, num_paths = 3, num_draws = 107,
        num_multi_draws = 1, psis_resample = FALSE)
    expect_equal(posterior::ndraws(out4), 3 * 107)

    out5 <- pathfinder(bernoulli_model, BERNOULLI_DATA, num_paths = 1, num_draws = 109,
        num_multi_draws = 1, psis_resample = FALSE)
    expect_equal(posterior::ndraws(out5), 109)
})

test_that("calculate_lp works", {
    out <- pathfinder(bernoulli_model, BERNOULLI_DATA, num_paths = 2, calculate_lp = FALSE)

    expect_gt(sum(is.nan(out$lp__)), 0)
    expect_lt(sum(is.nan(out$lp__)), 2000)

    out_single <- pathfinder(bernoulli_model, BERNOULLI_DATA, num_paths = 1, calculate_lp = FALSE)

    expect_gt(sum(is.nan(out_single$lp__)), 0)
    expect_lt(sum(is.nan(out_single$lp__)), 1000)
})

test_that("seed works", {

    out1 <- pathfinder(bernoulli_model, BERNOULLI_DATA, seed = 123)
    out2 <- pathfinder(bernoulli_model, BERNOULLI_DATA, seed = 123)

    expect_equal(out1$theta, out2$theta)

    out3 <- pathfinder(bernoulli_model, BERNOULLI_DATA, seed = 456)
    expect_error(expect_equal(out1$theta, out3$theta))

})
test_that("inits work", {

    init1 <- "{\"mu\": -1000}"
    out1 <- pathfinder(multimodal_model, inits = init1)
    expect_true(all(out1$mu < 0))

    init2 <- "{\"mu\": 1000}"
    out2 <- pathfinder(multimodal_model, inits = init2)
    expect_true(all(out2$mu > 0))

    temp_file <- tempfile(fileext = ".json")
    write(init1, temp_file)
    out3 <- pathfinder(multimodal_model, num_paths = 2, inits = c(temp_file, init1))

    expect_true(all(out3$mu < 0))
})


test_that("bad data handled properly", {

    data <- "{\"N\": -1}"
    expect_error(pathfinder(bernoulli_model, data), "greater than or equal to 0")

    data <- "{\"N\": 1, \"y\": [1,2]}"
    expect_error(pathfinder(bernoulli_model, data), "mismatch in dimension")

    expect_error(pathfinder(bernoulli_model, "{\"bad\"}"), "Error in JSON parsing")

    expect_error(pathfinder(bernoulli_model, "not/real/path.json"), "Could not open data file")

})

test_that("bad inits handled properly", {
    init <- "{\"theta\": 2}"
    expect_error(pathfinder(bernoulli_model, BERNOULLI_DATA, inits = init), "Initialization failed")
    expect_error(pathfinder(bernoulli_model, BERNOULLI_DATA, num_paths = 1, inits = init),
        "Initialization failed")
    expect_error(pathfinder(bernoulli_model, BERNOULLI_DATA, inits = "bad/path.json"),
        "Could not open data file")
    inits <- c(init, init)
    expect_error(pathfinder(bernoulli_model, BERNOULLI_DATA, num_paths = 2, inits = inits),
        "Initialization failed")

    init2 <- "{\"theta\": 0.2}"

    # unlike sample, a failure of subset of inits is not fatal
    inits <- rep(list(init), 10)
    inits[[11]] <- init2
    pathfinder(bernoulli_model, BERNOULLI_DATA, num_paths = 11, num_multi_draws = 10,
        inits = inits)
    inits <- list(init2, init2)
    expect_error(pathfinder(bernoulli_model, BERNOULLI_DATA, num_paths = 1, inits = inits),
        "match the number of chains")
    expect_error(pathfinder(bernoulli_model, BERNOULLI_DATA, num_paths = 3, inits = inits),
        "match the number of chains")

})

test_that("empty model ok", {
    expect_error(pathfinder(empty_model), "no parameters")
})

test_that("bad args raise errors", {
    expect_error(pathfinder(bernoulli_model, BERNOULLI_DATA, num_paths = 0), "at least 1")
    expect_error(pathfinder(bernoulli_model, BERNOULLI_DATA, num_draws = 0), "at least 1")
    expect_error(pathfinder(bernoulli_model, BERNOULLI_DATA, id = 0), "positive")
    expect_error(pathfinder(bernoulli_model, BERNOULLI_DATA, init_radius = -0.1),
        "non-negative")
    expect_error(pathfinder(bernoulli_model, BERNOULLI_DATA, num_threads = 0), "positive")
    expect_error(pathfinder(bernoulli_model, BERNOULLI_DATA, num_iterations = 0),
        "positive")
    expect_error(pathfinder(bernoulli_model, BERNOULLI_DATA, num_elbo_draws = 0),
        "positive")
    expect_error(pathfinder(bernoulli_model, BERNOULLI_DATA, num_multi_draws = 0),
        "at least 1")
    expect_error(pathfinder(bernoulli_model, BERNOULLI_DATA, max_history_size = 0),
        "positive")
    expect_error(pathfinder(bernoulli_model, BERNOULLI_DATA, init_alpha = 0), "positive")
    expect_error(pathfinder(bernoulli_model, BERNOULLI_DATA, tol_obj = 0), "positive")
    expect_error(pathfinder(bernoulli_model, BERNOULLI_DATA, tol_rel_obj = 0), "positive")
    expect_error(pathfinder(bernoulli_model, BERNOULLI_DATA, tol_grad = 0), "positive")
    expect_error(pathfinder(bernoulli_model, BERNOULLI_DATA, tol_rel_grad = 0), "positive")
    expect_error(pathfinder(bernoulli_model, BERNOULLI_DATA, tol_param = 0), "positive")

})

