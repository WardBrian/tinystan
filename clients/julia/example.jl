using Statistics
using TinyStan

model = Model("./test_models/bernoulli/bernoulli.stan")
data = "./test_models/bernoulli/bernoulli.data.json"

param_names, draws = sample(model, data)
println(param_names)
println(size(draws))
println(mean(draws[:, :, param_names.=="theta"]))

param_names, draws = pathfinder(model, data)
println(param_names)
println(size(draws))
println(mean(draws[:, param_names.=="theta"]))

param_names, optimum = optimize(model, data)
println(param_names)
println(optimum[param_names.=="theta"][1])
