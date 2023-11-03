@testset "Model" verbose = true begin

    @testset "Loading model" begin
        model = FFIStanModel(joinpath(STAN_FOLDER, "bernoulli", "bernoulli.stan"))
        @test model !== nothing
    end

    @testset "API version" begin
        @test api_version(bernoulli_model) == (0, 1, 0)
    end

end
