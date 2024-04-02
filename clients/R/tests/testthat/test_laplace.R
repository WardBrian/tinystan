BERNOULLI_MODE <- "{\"theta\": 0.25}"

test_that("data arguments work", {

    out1 <- laplace_sampler(bernoulli_model, BERNOULLI_MODE, BERNOULLI_DATA)
    expect_true(mean(out1$theta) > 0.22 && mean(out1$theta) < 0.28)

    data_file <- file.path(stan_folder, "bernoulli", "bernoulli.data.json")
    out2 <- laplace_sampler(bernoulli_model, BERNOULLI_MODE, data = data_file)
    expect_true(mean(out2$theta) > 0.22 && mean(out2$theta) < 0.28)
})

test_that("output sizes are correct", {
    out1 <- laplace_sampler(bernoulli_model, BERNOULLI_MODE, BERNOULLI_DATA, num_draws = 324)
    expect_equal(posterior::ndraws(out1), 324)
})

test_that("calculate_lp works", {
    out <- laplace_sampler(bernoulli_model, BERNOULLI_MODE, BERNOULLI_DATA, num_draws = 500,
        calculate_lp = TRUE)
    expect_equal(sum(is.nan(posterior::draws_of(out$log_p__))), 0)

    out2 <- laplace_sampler(bernoulli_model, BERNOULLI_MODE, BERNOULLI_DATA, num_draws = 500,
        calculate_lp = FALSE)
    expect_equal(sum(is.nan(posterior::draws_of(out2$log_p__))), 500)
})


test_that("jacobian arg works", {
    for (jacobian in c(TRUE, FALSE)) {

        out <- optimizer(simple_jacobian_model, jacobian = jacobian, seed = 1234)
        sigma <- posterior::extract_variable(out, "sigma")

        draws <- laplace_sampler(simple_jacobian_model, c(sigma), jacobian = jacobian,
            seed = 1234)
        sigma <- mean(posterior::extract_variable(draws, "sigma"))
        if (jacobian) {
            expect_equal(sigma, 3.3, tolerance = 0.2, ignore_attr = TRUE)
        } else {
            expect_equal(sigma, 3, tolerance = 0.2, ignore_attr = TRUE)
        }

    }
})

test_that("save_hessian works", {
    data <- "{\"N\": 3}"
    mode <- "{\"alpha\": [0.1,0.2,0.3]}"

    out <- laplace_sampler(gaussian_model, mode, data, save_hessian = TRUE)
    expect_true("hessian" %in% names(out))
    expect_equal(dim(out$hessian), c(3, 3))
    expect_equal(out$hessian, matrix(c(-1, 0, 0, 0, -1, 0, 0, 0, -1), nrow = 3))
})

test_that("seed works", {

    out1 <- laplace_sampler(bernoulli_model, BERNOULLI_MODE, BERNOULLI_DATA, seed = 123)
    out2 <- laplace_sampler(bernoulli_model, BERNOULLI_MODE, BERNOULLI_DATA, seed = 123)

    expect_equal(out1$theta, out2$theta)

    out3 <- laplace_sampler(bernoulli_model, BERNOULLI_MODE, BERNOULLI_DATA, seed = 456)
    expect_error(expect_equal(out1$theta, out3$theta))

})


test_that("bad data handled properly", {

    data <- "{\"N\": -1}"
    expect_error(laplace_sampler(bernoulli_model, BERNOULLI_MODE, data), "greater than or equal to 0")

    data <- "{\"N\": 1, \"y\": [1,2]}"
    expect_error(laplace_sampler(bernoulli_model, BERNOULLI_MODE, data), "mismatch in dimension")

    expect_error(laplace_sampler(bernoulli_model, BERNOULLI_MODE, "{\"bad\"}"), "Error in JSON parsing")

    expect_error(laplace_sampler(bernoulli_model, BERNOULLI_MODE, "not/real/path.json"),
        "Could not open data file")

})

test_that("bad mode array handled properly", {
    mode1 = c(2)
    expect_error(laplace_sampler(bernoulli_model, mode1, BERNOULLI_DATA), "Bounded variable is 2")

    mode2 = c(0.5, 0.5)
    expect_error(laplace_sampler(bernoulli_model, mode2, BERNOULLI_DATA), "incorrect length")
})

test_that("bad mode json handled properly", {
    mode <- "{\"theta\": 2}"
    expect_error(laplace_sampler(bernoulli_model, mode, BERNOULLI_DATA), "Bounded variable is 2")

    mode <- "{\"theta\": [0.5, 0.5]}"
    expect_error(laplace_sampler(bernoulli_model, mode, BERNOULLI_DATA), "mismatch in number")

    expect_error(laplace_sampler(bernoulli_model, "bad/path.json", BERNOULLI_DATA),
        "Could not open data file")
})


test_that("bad num_draws raises errors", {
    expect_error(laplace_sampler(bernoulli_model, BERNOULLI_MODE, BERNOULLI_DATA, num_draws = 0),
        "at least 1")
})
