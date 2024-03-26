@testset "Laplace Sampling" verbose = true begin

    BERNOULLI_MODE = "{\"theta\": 0.25}"

    @testset "Data" begin
        (names, draws) = laplace_sample(bernoulli_model, BERNOULLI_MODE, BERNOULLI_DATA)
        @test 0.22 < mean(draws[:, names.=="theta"]) < 0.28

        (names, draws) = laplace_sample(
            bernoulli_model,
            BERNOULLI_MODE,
            joinpath(STAN_FOLDER, "bernoulli", "bernoulli.data.json"),
        )
        @test 0.22 < mean(draws[:, names.=="theta"]) < 0.28
    end


    @testset "Number of draws" begin
        (names, draws) =
            laplace_sample(bernoulli_model, BERNOULLI_MODE, BERNOULLI_DATA; num_draws = 234)
        @test size(draws, 1) == 234
    end

    @testset "Calculate LP" begin
        (names, draws) = laplace_sample(
            bernoulli_model,
            BERNOULLI_MODE,
            BERNOULLI_DATA;
            num_draws = 500,
            calculate_lp = true,
        )
        @test sum(isnan.(draws[:, names.=="log_p__"])) == 0

        (names, draws) = laplace_sample(
            bernoulli_model,
            BERNOULLI_MODE,
            BERNOULLI_DATA;
            num_draws = 500,
            calculate_lp = false,
        )
        @test sum(isnan.(draws[:, names.=="log_p__"])) == 500
    end

    @testset "Jacobian" begin
        @testset for jacobian in [true, false]
            (names, mode) =
                optimize(simple_jacobian_model; jacobian = jacobian, seed = UInt32(1234))
            mode_array = mode[2:end]
            (names, draws) = laplace_sample(
                simple_jacobian_model,
                mode_array;
                jacobian = jacobian,
                seed = UInt32(1234),
            )

            optimum = if jacobian
                3.3
            else
                3.0
            end
            @test optimum ≈ mean(draws[:, names.=="sigma"]) atol = 0.2
        end
    end

    @testset "Save Hessian" begin
        data = "{\"N\": 3}"
        mode = "{\"alpha\": [0.1,0.2,0.3]}"

        (_, _, hessian) = laplace_sample(gaussian_model, mode, data; save_hessian = true)
        @test size(hessian) == (3, 3)
        @test hessian ≈ [-1.0 0.0 0.0; 0.0 -1.0 0.0; 0.0 0.0 -1.0] atol = 0.1
    end

    @testset "Seed" begin
        (_, draws1) = laplace_sample(
            bernoulli_model,
            BERNOULLI_MODE,
            BERNOULLI_DATA;
            seed = UInt32(123),
        )
        (_, draws2) = laplace_sample(
            bernoulli_model,
            BERNOULLI_MODE,
            BERNOULLI_DATA;
            seed = UInt32(123),
        )
        @test draws1 == draws2

        (_, draws3) = laplace_sample(
            bernoulli_model,
            BERNOULLI_MODE,
            BERNOULLI_DATA;
            seed = UInt32(456),
        )
        @test draws1 != draws3
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

        mode2 = [0.5, 0.5]
        @test_throws "incorrect length" laplace_sample(
            bernoulli_model,
            mode2,
            BERNOULLI_DATA,
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
