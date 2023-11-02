ALGORITHMS <- c(ffistan::OptimizationAlgorithm$NEWTON, ffistan::OptimizationAlgorithm$BFGS,
    ffistan::OptimizationAlgorithm$LBFGS)

test_that("data args work", {

    out1 <- bernoulli_model$optimize(BERNOULLI_DATA)
    expect_true((out1$optimum[2]) > 0.19 && (out1$optimum[2]) < 0.21)

    data_file <- file.path(stan_folder, "bernoulli", "bernoulli.data.json")
    out2 <- bernoulli_model$optimize(data = data_file)
    expect_true(mean(out2$optimum[2]) > 0.19 && mean(out2$optimum[2]) < 0.21)

})

test_that("algorithm and jacobian args work", {

    for (algorithm in ALGORITHMS) {
        for (jacobian in c(TRUE, FALSE)) {

            out <- simple_jacobian_model$optimize(BERNOULLI_DATA, algorithm = algorithm,
                jacobian = jacobian, seed = 1234)

            if (jacobian) {
                expect_equal(out$optimum[2], 3.3, tolerance = 0.01, ignore_attr = TRUE)
            } else {
                expect_equal(out$optimum[2], 3, tolerance = 0.01, ignore_attr = TRUE)
            }

        }
    }

})

test_that("seed works", {

    out1 <- bernoulli_model$optimize(BERNOULLI_DATA, seed = 123)
    out2 <- bernoulli_model$optimize(BERNOULLI_DATA, seed = 123)

    expect_equal(out1$optimum, out2$optimum)

    out3 <- bernoulli_model$optimize(BERNOULLI_DATA, seed = 456)
    expect_error(expect_equal(out1$optimum, out3$optimum))

})


test_that("inits work", {

    init <- "{\"mu\": -100}"
    out1 <- multimodal_model$optimize(init = init)
    expect_true(all(out1$mu < 0))

    init <- "{\"mu\": 100}"
    temp_file <- tempfile(fileext = ".json")
    write(init, temp_file)

    out2 <- multimodal_model$optimize(init = temp_file)
    expect_true(all(out2$mu > 0))

})

test_that("bad data handled properly", {

    data <- "{\"N\": -1}"
    expect_error(bernoulli_model$optimize(data), "greater than or equal to 0")

    data <- "{\"N\": 1, \"y\": [1,2]}"
    expect_error(bernoulli_model$optimize(data), "mismatch in dimension")

    expect_error(bernoulli_model$optimize("{\"bad\"}"), "Error in JSON parsing")

    expect_error(bernoulli_model$optimize("not/real/path.json"), "Could not open data file")

})

test_that("bad init handled properly", {

    init <- "{\"theta\": 2}"
    expect_error(bernoulli_model$optimize(BERNOULLI_DATA, init = init), "Initialization failed")

    expect_error(bernoulli_model$optimize(BERNOULLI_DATA, init = "bad/path.json"),
        "Could not open data file")

})

test_that("empty model ok", {

    expect_no_error(empty_model$optimize())

})

test_that("bad args raise errors", {

    for (algorithm in ALGORITHMS) {

        expect_error(bernoulli_model$optimize(BERNOULLI_DATA, algorithm = algorithm,
            id = 0), "positive")
        expect_error(bernoulli_model$optimize(BERNOULLI_DATA, algorithm = algorithm,
            num_iterations = 0), "positive")
        expect_error(bernoulli_model$optimize(BERNOULLI_DATA, algorithm = algorithm,
            init_radius = -0.1), "non-negative")

        if (algorithm != ffistan::OptimizationAlgorithm$NEWTON) {
            expected <- expect_error
        } else {
            expected <- function(e, ...) {
                expect_no_error(e)
            }
        }

        expected(bernoulli_model$optimize(BERNOULLI_DATA, algorithm = algorithm,
            init_alpha = 0), "positive")
        expected(bernoulli_model$optimize(BERNOULLI_DATA, algorithm = algorithm,
            tol_obj = 0), "positive")
        expected(bernoulli_model$optimize(BERNOULLI_DATA, algorithm = algorithm,
            tol_rel_obj = 0), "positive")
        expected(bernoulli_model$optimize(BERNOULLI_DATA, algorithm = algorithm,
            tol_grad = 0), "positive")
        expected(bernoulli_model$optimize(BERNOULLI_DATA, algorithm = algorithm,
            tol_rel_grad = 0), "positive")
        expected(bernoulli_model$optimize(BERNOULLI_DATA, algorithm = algorithm,
            tol_param = 0), "positive")

        if (algorithm == ffistan::OptimizationAlgorithm$LBFGS) {
            expected <- expect_error
        } else {
            expected <- function(e, ...) {
                expect_no_error(e)
            }
        }

        expected(bernoulli_model$optimize(BERNOULLI_DATA, algorithm = algorithm,
            max_history_size = 0), "positive")


    }


})
