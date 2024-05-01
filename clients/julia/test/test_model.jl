@testset "Model" verbose = true begin

    @testset "Loading model" begin
        model = Model(joinpath(STAN_FOLDER, "bernoulli", "bernoulli_model.so"))
        @test model !== nothing
    end

    @testset "API version" begin
        @test VersionNumber(api_version(bernoulli_model)) == TinyStan.pkg_version
    end

    @testset "Stan version" begin
        ver = stan_version(bernoulli_model)
        @test ver[1] == 2
        @test ver[2] >= 34
        @test ver[3] >= 0
    end

end
