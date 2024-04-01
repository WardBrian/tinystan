library(tinystan)

data <- "./tests/test_models/bernoulli/bernoulli.data.json"

mod <- tinystan_model("./tests/test_models/bernoulli/bernoulli.stan")
fit = sampler(private = mod, data = ".tests/test_models/bernoulli/bernoulli.data.json")
fit
