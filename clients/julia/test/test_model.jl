@testset "Model" verbose = true begin

    @testset "Loading model" begin
        model = Model(joinpath(STAN_FOLDER, "bernoulli", "bernoulli_model.so"))
        @test model !== nothing
    end

    @testset "API version" begin
        @test api_version(bernoulli_model) == (0, 1, 0)
    end

end
