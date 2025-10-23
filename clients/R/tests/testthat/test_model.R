bernoulli_file <- file.path(stan_folder, "bernoulli", "bernoulli.stan")
test_that("model loads", {
  suppressWarnings(model <- tinystan_model(bernoulli_file))
  expect_true(!is.null(model))
})

test_that("api_version is correct", {
  expect_equal(api_version(bernoulli_model), current_version_list)
})

test_that("stan version is valid", {
  stan_version <- stan_version(bernoulli_model)
  expect_equal(stan_version$major, 2)
  expect_gte(stan_version$minor, 34)
  expect_gte(stan_version$patch, 0)
})
