using Test
using TinyStan
using Statistics

const STAN_FOLDER = joinpath(@__DIR__, "..", "..", "..", "test_models")

# @testset "TinyStan tests" verbose = true begin
#     include("test_compile.jl")
# end


bernoulli_model = Model(joinpath(STAN_FOLDER, "bernoulli", "bernoulli_model.so"))
BERNOULLI_DATA = "{\"N\": 10, \"y\": [0,1,0,0,0,0,0,0,0,1]}"
gaussian_model = Model(joinpath(STAN_FOLDER, "gaussian", "gaussian_model.so"))
empty_model = Model(joinpath(STAN_FOLDER, "empty", "empty_model.so"))
multimodal_model = Model(joinpath(STAN_FOLDER, "multimodal", "multimodal_model.so"))
simple_jacobian_model =
    Model(joinpath(STAN_FOLDER, "simple_jacobian", "simple_jacobian_model.so"))

@testset "TinyStan tests" verbose = true begin
    # include("test_model.jl")
    # include("test_sample.jl")
    # include("test_pathfinder.jl")
    # include("test_optimize.jl")
    include("test_laplace.jl")
end
