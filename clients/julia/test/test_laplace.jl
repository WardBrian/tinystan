@testset "Laplace Sampling" verbose = true begin

    BERNOULLI_MODE = "{\"theta\": 0.25}"

    @testset "Data" begin
        out = laplace_sample(bernoulli_model, BERNOULLI_MODE, BERNOULLI_DATA)
        @test 0.22 < mean(get_draws(out, "theta")) < 0.28

        out = laplace_sample(
            bernoulli_model,
            BERNOULLI_MODE,
            joinpath(STAN_FOLDER, "bernoulli", "bernoulli.data.json"),
        )
        @test 0.22 < mean(get_draws(out, "theta")) < 0.28
    end


    @testset "Number of draws" begin
        out =
            laplace_sample(bernoulli_model, BERNOULLI_MODE, BERNOULLI_DATA; num_draws = 234)
        @test size(out.draws, 1) == 234
    end

    @testset "Calculate LP" begin
        out = laplace_sample(
            bernoulli_model,
            BERNOULLI_MODE,
            BERNOULLI_DATA;
            num_draws = 500,
            calculate_lp = true,
        )
        @test sum(isnan.(get_draws(out, "log_p__"))) == 0

        out = laplace_sample(
            bernoulli_model,
            BERNOULLI_MODE,
            BERNOULLI_DATA;
            num_draws = 500,
            calculate_lp = false,
        )
        @test sum(isnan.(get_draws(out, "log_p__"))) == 500
    end

    @testset "Jacobian" begin
        @testset for jacobian in [true, false]
            out_opt =
                optimize(simple_jacobian_model; jacobian = jacobian, seed = UInt32(12345))
            mode_array = out_opt.draws[2:end]
            out = laplace_sample(
                simple_jacobian_model,
                mode_array;
                jacobian = jacobian,
                seed = UInt32(12345),
            )

            optimum = if jacobian
                3.3
            else
                3.0
            end
            @test optimum ≈ mean(get_draws(out, "sigma")) atol = 0.2
        end
    end

    @testset "Save Hessian" begin
        data = "{\"N\": 3}"
        mode = "{\"alpha\": [0.1,0.2,0.3]}"

        out = laplace_sample(gaussian_model, mode, data; save_hessian = true)
        @test size(out.hessian) == (3, 3)
        @test out.hessian ≈ [-1.0 0.0 0.0; 0.0 -1.0 0.0; 0.0 0.0 -1.0] atol = 0.1
    end

    @testset "Seed" begin
        out1 = laplace_sample(
            bernoulli_model,
            BERNOULLI_MODE,
            BERNOULLI_DATA;
            seed = UInt32(123),
        )
        out2 = laplace_sample(
            bernoulli_model,
            BERNOULLI_MODE,
            BERNOULLI_DATA;
            seed = UInt32(123),
        )
        @test out1.draws == out2.draws

        out3 = laplace_sample(
            bernoulli_model,
            BERNOULLI_MODE,
            BERNOULLI_DATA;
            seed = UInt32(456),
        )
        @test out1.draws != out3.draws
    end

    @testset "Bad data" begin
        data1 = "{\"N\": -1}"
        @test_throws "greater than or equal to 0" laplace_sample(
            bernoulli_model,
            BERNOULLI_MODE,
            data1,
        )

        data2 = "{\"N\":1, \"y\": [0,1]}"
        @test_throws "mismatch in dimension" laplace_sample(
            bernoulli_model,
            BERNOULLI_MODE,
            data2,
        )

        @test_throws "Error in JSON parsing" laplace_sample(
            bernoulli_model,
            BERNOULLI_MODE,
            "{'bad'}",
        )
        @test_throws "Could not open data file" laplace_sample(
            bernoulli_model,
            BERNOULLI_MODE,
            "path/not/here.json",
        )
    end

    @testset "Bad mode array" begin
        mode1 = [2.0]
        @test_throws "Bounded variable is 2" laplace_sample(
            bernoulli_model,
            mode1,
            BERNOULLI_DATA,
        )

        mode2 = [1.0]
        @test_throws "incorrect length" laplace_sample(
            gaussian_model,
            mode2,
            "{\"N\": 4 }",
        )
    end

    @testset "Bad mode json" begin
        mode1 = "{\"theta\": 2.0}"
        @test_throws "Bounded variable is 2" laplace_sample(
            bernoulli_model,
            mode1,
            BERNOULLI_DATA,
        )

        mode2 = "{\"theta\": [0.5, 0.5]}"
        @test_throws "mismatch in number" laplace_sample(
            bernoulli_model,
            mode2,
            BERNOULLI_DATA,
        )

        @test_throws "Could not open data file" laplace_sample(
            bernoulli_model,
            "bad/path.json",
            BERNOULLI_DATA,
        )
    end

    @testset "Bad number of draws" begin
        @test_throws "at least 1" laplace_sample(
            bernoulli_model,
            BERNOULLI_MODE,
            BERNOULLI_DATA;
            num_draws = 0,
        )
    end
end
