library(ffistan)

model <- FFIStanModel$new("./test_models/bernoulli/bernoulli_model.so")
data <- "./test_models/bernoulli/bernoulli.data.json"

fit <- model$sample(data)
print(colMeans(fit$draws, dims = 2)[8])

pf = model$pathfinder(data)
print(colMeans(pf$draws)[3])

o = model$optimize(data)
print(o$optimum[2])
