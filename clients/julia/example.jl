using Statistics
using FFIStan

model = FFIStanModel("./test_models/bernoulli/bernoulli.stan")
data = "./test_models/bernoulli/bernoulli.data.json"

param_names, draws = sample(model, data)
println(param_names)
println(size(draws))
println(mean(draws, dims = (1, 2))[8])

param_names, draws = pathfinder(model, data)
println(param_names)
println(size(draws))
println(mean(draws[:, 3]))

param_names, draw = optimize(model, data)
println(param_names)
println(draw)
