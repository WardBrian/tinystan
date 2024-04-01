library(tinystan)

data <- "../../test_models/bernoulli/bernoulli.data.json"

mod <- tinystan_model("../../test_models/bernoulli/bernoulli.stan")
fit = sampler(private = mod,
  data = "/home/sbronder/open_source/stan/WardBrian/tinystan/test_models/bernoulli/bernoulli.data.json")
fit
