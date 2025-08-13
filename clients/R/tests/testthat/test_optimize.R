ALGORITHMS <- c(tinystan::OptimizationAlgorithm$NEWTON, tinystan::OptimizationAlgorithm$BFGS,
    tinystan::OptimizationAlgorithm$LBFGS)

test_that("data args work", {

    out1 <- optimizer(bernoulli_model, BERNOULLI_DATA)
    expect_true(mean(out1$draws$theta) > 0.19 && mean(out1$draws$theta) < 0.21)

    data_file <- file.path(stan_folder, "bernoulli", "bernoulli.data.json")
    out2 <- optimizer(bernoulli_model, data = data_file)
    expect_true(mean(out2$draws$theta) > 0.19 && mean(out2$draws$theta) < 0.21)

})

test_that("algorithm and jacobian args work", {

    for (algorithm in ALGORITHMS) {
        for (jacobian in c(TRUE, FALSE)) {

            out <- optimizer(simple_jacobian_model, algorithm = algorithm, jacobian = jacobian,
                seed = 1234)

            sigma <- posterior::extract_variable(out$draws, "sigma")

            if (jacobian) {
                expect_equal(sigma, 3.3, tolerance = 0.01, ignore_attr = TRUE)
            } else {
                expect_equal(sigma, 3, tolerance = 0.01, ignore_attr = TRUE)
            }

        }
    }

})

test_that("seed works", {

    out1 <- optimizer(bernoulli_model, BERNOULLI_DATA, seed = 123)
    out2 <- optimizer(bernoulli_model, BERNOULLI_DATA, seed = 123)

    expect_equal(out1$draws, out2$draws)

    out3 <- optimizer(bernoulli_model, BERNOULLI_DATA, seed = 456)
    expect_error(expect_equal(out1$draws, out3$draws))

})


test_that("inits work", {

    init <- "{\"mu\": -100}"
    out1 <- optimizer(multimodal_model, init = init)
    expect_true(all(out1$draws$mu < 0))

    init <- "{\"mu\": 100}"
    temp_file <- tempfile(fileext = ".json")
    write(init, temp_file)

    out2 <- optimizer(multimodal_model, init = temp_file)
    expect_true(all(out2$draws$mu > 0))

})

test_that("bad data handled properly", {

    data <- "{\"N\": -1}"
    expect_error(optimizer(bernoulli_model, data), "greater than or equal to 0")

    data <- "{\"N\": 1, \"y\": [1,2]}"
    expect_error(optimizer(bernoulli_model, data), "mismatch in dimension")

    expect_error(optimizer(bernoulli_model, "{\"bad\"}"), "Error in JSON parsing")

    expect_error(optimizer(bernoulli_model, "not/real/path.json"), "Could not open data file")

})

test_that("bad init handled properly", {

    init <- "{\"theta\": 2}"
    expect_error(optimizer(bernoulli_model, BERNOULLI_DATA, init = init), "Initialization failed")

    expect_error(optimizer(bernoulli_model, BERNOULLI_DATA, init = "bad/path.json"),
        "Could not open data file")

})

test_that("empty model ok", {

    expect_no_error(optimizer(empty_model))

})

test_that("bad args raise errors", {

    for (algorithm in ALGORITHMS) {

        expect_error(optimizer(bernoulli_model, BERNOULLI_DATA, algorithm = algorithm,
            id = 0), "positive")
        expect_error(optimizer(bernoulli_model, BERNOULLI_DATA, algorithm = algorithm,
            num_iterations = 0), "positive")
        expect_error(optimizer(bernoulli_model, BERNOULLI_DATA, algorithm = algorithm,
            init_radius = -0.1), "non-negative")

        if (algorithm != tinystan::OptimizationAlgorithm$NEWTON) {
            expected <- expect_error
        } else {
            expected <- function(e, ...) {
                expect_no_error(e)
            }
        }

        expected(optimizer(bernoulli_model, BERNOULLI_DATA, algorithm = algorithm,
            init_alpha = 0), "positive")
        expected(optimizer(bernoulli_model, BERNOULLI_DATA, algorithm = algorithm,
            tol_obj = 0), "positive")
        expected(optimizer(bernoulli_model, BERNOULLI_DATA, algorithm = algorithm,
            tol_rel_obj = 0), "positive")
        expected(optimizer(bernoulli_model, BERNOULLI_DATA, algorithm = algorithm,
            tol_grad = 0), "positive")
        expected(optimizer(bernoulli_model, BERNOULLI_DATA, algorithm = algorithm,
            tol_rel_grad = 0), "positive")
        expected(optimizer(bernoulli_model, BERNOULLI_DATA, algorithm = algorithm,
            tol_param = 0), "positive")

        if (algorithm == tinystan::OptimizationAlgorithm$LBFGS) {
            expected <- expect_error
        } else {
            expected <- function(e, ...) {
                expect_no_error(e)
            }
        }

        expected(optimizer(bernoulli_model, BERNOULLI_DATA, algorithm = algorithm,
            max_history_size = 0), "positive")


    }


})
