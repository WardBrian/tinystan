@testset "Compilation" verbose = true begin



    @testset "good" begin
        stanfile = joinpath(STAN_FOLDER, "gaussian", "gaussian.stan")
        lib = splitext(stanfile)[1] * "_model.so"
        rm(lib, force = true)
        res = TinyStan.compile_model(stanfile; stanc_args = ["--O1"])
        @test Base.samefile(lib, res)
        rm(lib)

        # test constructor triggered compilation
        model = TinyStan.Model(stanfile)
        @test isfile(lib)
    end



    @testset "bad Stan files" begin
        not_stanfile = joinpath(STAN_FOLDER, "bernoulli", "bernoulli.data.json")
        @test_throws ErrorException TinyStan.compile_model(not_stanfile)

        nonexistent = joinpath(STAN_FOLDER, "gaussian", "gaussian-notthere.stan")
        @test_throws SystemError TinyStan.compile_model(nonexistent)

        syntax_error = joinpath(STAN_FOLDER, "syntax_error", "syntax_error.stan")
        @test_throws ErrorException TinyStan.compile_model(syntax_error)
    end


    @testset "bad TinyStan paths" begin
        @test_throws ErrorException TinyStan.set_tinystan_path!("dummy")
        @test_throws ErrorException TinyStan.set_tinystan_path!(STAN_FOLDER)
    end


    @testset "download artifact" begin
        withenv("TINYSTAN" => nothing) do
            TinyStan.validate_stan_dir(TinyStan.get_tinystan_path())
        end
    end

end
