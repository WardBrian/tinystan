using Test
using FFIStan
using Statistics

const STAN_FOLDER = joinpath(@__DIR__, "..", "..", "..", "test_models")

bernoulli_model = FFIStanModel(joinpath(STAN_FOLDER, "bernoulli", "bernoulli.stan"))
BERNOULLI_DATA = "{\"N\": 10, \"y\": [0,1,0,0,0,0,0,0,0,1]}"

gaussian_model = FFIStanModel(joinpath(STAN_FOLDER, "gaussian", "gaussian.stan"))
empty_model = FFIStanModel(joinpath(STAN_FOLDER, "empty", "empty.stan"))
multimodal_model = FFIStanModel(joinpath(STAN_FOLDER, "multimodal", "multimodal.stan"))
simple_jacobian_model =
    FFIStanModel(joinpath(STAN_FOLDER, "simple_jacobian", "simple_jacobian.stan"))


@testset "FFIStan tests" verbose = true begin
    include("test_model.jl")
    include("test_sample.jl")
    include("test_pathfinder.jl")
    include("test_optimize.jl")
end
