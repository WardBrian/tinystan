@testset "Compilation" verbose = true begin



    @testset "good" begin
        stanfile = joinpath(STAN_FOLDER, "gaussian", "gaussian.stan")
        lib = splitext(stanfile)[1] * "_model.so"
        rm(lib, force = true)
        res = FFIStan.compile_model(stanfile; stanc_args = ["--O1"])
        @test Base.samefile(lib, res)
        rm(lib)

        # test constructor triggered compilation
        model = FFIStan.Model(stanfile)
        @test isfile(lib)
    end



    @testset "bad Stan files" begin
        not_stanfile = joinpath(STAN_FOLDER, "bernoulli", "bernoulli.data.json")
        @test_throws ErrorException FFIStan.compile_model(not_stanfile)

        nonexistent = joinpath(STAN_FOLDER, "gaussian", "gaussian-notthere.stan")
        @test_throws SystemError FFIStan.compile_model(nonexistent)

        syntax_error = joinpath(STAN_FOLDER, "syntax_error", "syntax_error.stan")
        @test_throws ErrorException FFIStan.compile_model(syntax_error)
    end


    @testset "bad FFIStan paths" begin
        @test_throws ErrorException FFIStan.set_ffistan_path!("dummy")
        @test_throws ErrorException FFIStan.set_ffistan_path!(STAN_FOLDER)
    end


    @testset "download artifact" begin
        withenv("FFISTAN" => nothing) do
            FFIStan.validate_stan_dir(FFIStan.get_ffistan_path())
        end
    end

end
