using Statistics
using TinyStan

model = Model("./test_models/bernoulli/bernoulli.stan")
data = "./test_models/bernoulli/bernoulli.data.json"

output = sample(model, data)
println(output.names)
println(size(output.draws))
println(mean(get_draws(output, "theta")))

output_pf = pathfinder(model, data)
println(output_pf.names)
println(size(output_pf.draws))
println(mean(get_draws(output_pf, "theta")))

output_opt = optimize(model, data)
println(output_opt.names)
println(get_draws(output_opt, "theta")[1])
