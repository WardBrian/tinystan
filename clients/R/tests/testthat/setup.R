options(warnPartialMatchDollar = TRUE)
options(warnPartialMatchArgs = TRUE)
options(warnPartialMatchAttr = TRUE)
options(warn = 0)
stan_folder <- file.path("..", "test_models")
print(getwd())
bernoulli_model <- tinystan_model(file.path(stan_folder, "bernoulli", "bernoulli.stan"))
tinystan:::api_version(bernoulli_model)
BERNOULLI_DATA <- "{\"N\": 10, \"y\": [0, 1, 0, 0, 0, 0, 0, 0, 0, 1]}"

gaussian_model <- tinystan_model(file.path(stan_folder, "gaussian", "gaussian.stan"))

empty_model <- tinystan_model(file.path(stan_folder, "empty", "empty.stan"))
multimodal_model <- tinystan_model(file.path(stan_folder, "multimodal", "multimodal.stan"))
simple_jacobian_model <- tinystan_model(file.path(stan_folder, "simple_jacobian",
  "simple_jacobian.stan"))

# hack around the fact that comparisons to NULL result in logical(0) and
# all(logical(0)) is TRUE, for some reason.
.builtin_all <- all
all <- function(l) {
  if (length(l) == 0)
    return(isTRUE(l))
  .builtin_all(l)
}
options(warn = 2)
