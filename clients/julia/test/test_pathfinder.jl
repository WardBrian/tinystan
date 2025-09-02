

@testset "Pathfinder" verbose = true begin

    @testset "Data" begin
        out = pathfinder(bernoulli_model, BERNOULLI_DATA)
        @test 0.2 < mean(get_draws(out, "theta")) < 0.3

        out = pathfinder(
            bernoulli_model,
            joinpath(STAN_FOLDER, "bernoulli", "bernoulli.data.json"),
        )
        @test 0.2 < mean(get_draws(out, "theta")) < 0.3
    end

    @testset "Output sizes" begin
        out1 = pathfinder(
            bernoulli_model,
            BERNOULLI_DATA,
            num_paths = 4,
            num_draws = 101,
            num_multi_draws = 99,
        )
        @test size(out1.draws, 1) == 99

        out2 = pathfinder(
            bernoulli_model,
            BERNOULLI_DATA,
            num_paths = 1,
            num_draws = 101,
            num_multi_draws = 103,
        )
        @test size(out2.draws, 1) == 103

        out3 = pathfinder(
            bernoulli_model,
            BERNOULLI_DATA,
            num_paths = 2,
            num_draws = 105,
            num_multi_draws = 1,
            calculate_lp = false,
        )
        @test size(out3.draws, 1) == 2 * 105


        out4 = pathfinder(
            bernoulli_model,
            BERNOULLI_DATA,
            num_paths = 3,
            num_draws = 107,
            num_multi_draws = 1,
            psis_resample = false,
        )
        @test size(out4.draws, 1) == 3 * 107

        out5 = pathfinder(
            bernoulli_model,
            BERNOULLI_DATA,
            num_paths = 1,
            num_draws = 109,
            num_multi_draws = 1,
            psis_resample = false,
        )
        @test size(out5.draws, 1) == 109
    end

    @testset "Calculate LP" begin
        out =
            pathfinder(bernoulli_model, BERNOULLI_DATA, num_paths = 2, calculate_lp = false)
        @test sum(isnan.(get_draws(out, "lp__"))) > 0
        @test sum(isnan.(get_draws(out, "lp__"))) < 2000

        out_single =
            pathfinder(bernoulli_model, BERNOULLI_DATA, num_paths = 1, calculate_lp = false)
        @test sum(isnan.(get_draws(out_single, "lp__"))) > 0
        @test sum(isnan.(get_draws(out_single, "lp__"))) < 1000
    end

    @testset "Seed" begin
        out1 = pathfinder(bernoulli_model, BERNOULLI_DATA; seed = UInt32(123))
        out2 = pathfinder(bernoulli_model, BERNOULLI_DATA; seed = UInt32(123))
        @test sortslices(out1.draws, dims=1) == sortslices(out2.draws, dims=1)

        out3 = pathfinder(bernoulli_model, BERNOULLI_DATA; seed = UInt32(456))
        @test sortslices(out1.draws, dims=1) != sortslices(out3.draws, dims=1)
    end

    @testset "Inits" begin
        init1 = "{\"mu\": -1000}"
        out1 = pathfinder(multimodal_model; inits = init1)
        @test all(get_draws(out1, "mu") .< 0)

        init2 = "{\"mu\": 1000}"
        out2 = pathfinder(multimodal_model; inits = init2)
        @test all(get_draws(out2, "mu") .> 0)

        init3 = tempname() * ".json"
        open(init3, "w") do io
            write(io, init1)
        end
        out3 = pathfinder(multimodal_model; inits = init3)
        @test all(get_draws(out3, "mu") .< 0)

    end


    @testset "Bad data" begin
        data1 = "{\"N\": -1}"
        @test_throws "greater than or equal to 0" pathfinder(bernoulli_model, data1)

        data2 = "{\"N\":1, \"y\": [0,1]}"
        @test_throws "mismatch in dimension" pathfinder(bernoulli_model, data2)

        @test_throws "Error in JSON parsing" pathfinder(bernoulli_model, "{\"bad\"}")

        @test_throws "Could not open data file" pathfinder(
            bernoulli_model,
            "path/not/here.json",
        )
    end

    @testset "Bad inits" begin
        init1 = "{\"theta\": 2}"
        @test_throws "Initialization failed" pathfinder(
            bernoulli_model,
            BERNOULLI_DATA;
            inits = init1,
        )
        @test_throws "Initialization failed" pathfinder(
            bernoulli_model,
            BERNOULLI_DATA;
            num_paths = 1,
            inits = init1,
        )

        @test_throws "Could not open data file" pathfinder(
            bernoulli_model,
            BERNOULLI_DATA;
            inits = "bad/path.json",
        )

        @test_throws "Initialization failed" pathfinder(
            bernoulli_model,
            BERNOULLI_DATA;
            num_paths = 2,
            inits = [init1, init1],
        )

        init2 = "{\"theta\": 0.2}"


        inits = append!(repeat([init1], 10), [init2])
        pathfinder(bernoulli_model, BERNOULLI_DATA; num_paths = 11, inits = inits)

        inits = [init2, init2]
        @test_throws "match the number of chains" pathfinder(
            bernoulli_model,
            BERNOULLI_DATA;
            num_paths = 1,
            inits = inits,
        )
        @test_throws "match the number of chains" pathfinder(
            bernoulli_model,
            BERNOULLI_DATA;
            num_paths = 3,
            inits = inits,
        )

    end

    @testset "Model without parameters" begin
        @test_throws "no parameters" pathfinder(empty_model)
    end


    @testset "Bad arguments" begin
        @testset for (name, value, match) in [
            (:num_paths, 0, "at least 1"),
            (:num_draws, 0, "at least 1"),
            (:id, 0, "positive"),
            (:init_radius, -0.1, "non-negative"),
            (:num_threads, 0, "positive"),
            (:num_iterations, 0, "positive"),
            (:num_elbo_draws, 0, "positive"),
            (:num_multi_draws, 0, "at least 1"),
            # l-bfgs sub-options:
            (:max_history_size, 0, "positive"),
            (:init_alpha, 0.0, "positive"),
            (:tol_obj, 0.0, "positive"),
            (:tol_rel_obj, 0.0, "positive"),
            (:tol_grad, 0.0, "positive"),
            (:tol_rel_grad, 0.0, "positive"),
            (:tol_param, 0.0, "positive"),
        ]
            @test_throws match pathfinder(
                bernoulli_model,
                BERNOULLI_DATA;
                Dict(name => value)...,
            )
        end
    end
end
