

# test_that('compilation works', { name <- 'gaussian' file <-
# file.path(stan_folder, name, paste0(name, '.stan'))

# lib <- file.path(stan_folder, name, paste0(name, '_model.so')) unlink(lib,
# force = TRUE)

# out <- compile_model(file, stanc_args = c('--O1'))

# expect_true(file.exists(lib)) expect_equal(normalizePath(lib),
# normalizePath(out))

# unlink(lib, force = TRUE)

# out <- compile_model(file) })

# test_that('compilation fails on non-stan file', {
# expect_error(compile_model(file.path(stan_folder, 'bernoulli',
# 'bernoulli.data.json')), 'does not end with '.stan'') })

# test_that('compilation fails on missing file', {
# expect_error(compile_model('badpath.stan'), 'does not exist!') })

# test_that('compilation fails on bad syntax', {
# expect_error(compile_model(file.path(stan_folder, 'syntax_error',
# 'syntax_error.stan')), 'Compilation failed') })

# test_that('bad paths fail', { expect_error(set_tinystan_path('badpath'),
# 'does not exist!') expect_error(set_tinystan_path(file.path(stan_folder)),
# 'does not contain file 'Makefile'') })
