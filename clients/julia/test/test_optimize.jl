


@testset "Optimize" verbose = true begin

    ALL_ALGORITHMS = [TinyStan.NEWTON, TinyStan.LBFGS, TinyStan.BFGS]

    @testset "Data" begin
        out = optimize(bernoulli_model, BERNOULLI_DATA)
        @test 0.19 < get_draws(out, "theta")[1] < 0.21

        out = optimize(
            bernoulli_model,
            joinpath(STAN_FOLDER, "bernoulli", "bernoulli.data.json"),
        )
        @test 0.19 < get_draws(out, "theta")[1] < 0.21
    end

    @testset "Jacobian" begin
        @testset for algorithm in ALL_ALGORITHMS
            @testset for jacobian in [true, false]

                out = optimize(
                    simple_jacobian_model,
                    algorithm = algorithm,
                    jacobian = jacobian,
                    seed = UInt32(1234),
                )
                optimum = if jacobian
                    3.3
                else
                    3.0
                end
                @test optimum ≈ get_draws(out, "sigma")[1] atol = 0.01
            end
        end
    end


    @testset "Seed" begin
        out1 = optimize(bernoulli_model, BERNOULLI_DATA; seed = UInt32(123))
        out2 = optimize(bernoulli_model, BERNOULLI_DATA; seed = UInt32(123))
        @test out1.draws == out2.draws

        out3 = optimize(bernoulli_model, BERNOULLI_DATA; seed = UInt32(456))
        @test out1.draws != out3.draws
    end

    @testset "Inits" begin
        init1 = "{\"mu\": -1000}"
        out1 = optimize(multimodal_model; init = init1)
        @test all(get_draws(out1, "mu") .< 0)

        init2 = "{\"mu\": 1000}"
        out2 = optimize(multimodal_model; init = init2)
        @test all(get_draws(out2, "mu") .> 0)

        init3 = tempname() * ".json"
        open(init3, "w") do io
            write(io, init1)
        end
        out3 = optimize(multimodal_model; init = init3)
        @test all(get_draws(out3, "mu") .< 0)

    end


    @testset "Bad data" begin
        data1 = "{\"N\": -1}"
        @test_throws "greater than or equal to 0" optimize(bernoulli_model, data1)

        data2 = "{\"N\":1, \"y\": [0,1]}"
        @test_throws "mismatch in dimension" optimize(bernoulli_model, data2)

        @test_throws "Error in JSON parsing" optimize(bernoulli_model, "{\"bad\"}")

        @test_throws "Could not open data file" optimize(
            bernoulli_model,
            "path/not/here.json",
        )
    end

    @testset "Bad inits" begin
        init1 = "{\"theta\": 2}"
        @test_throws "Initialization failed" optimize(
            bernoulli_model,
            BERNOULLI_DATA;
            init = init1,
        )
        @test_throws "Could not open data file" optimize(
            bernoulli_model,
            BERNOULLI_DATA;
            init = "bad/path.json",
        )
    end

    @testset "Model without parameters" begin
        out = optimize(empty_model)
        @test length(out.names) == 1 # lp
    end


    @testset "Bad arguments" begin
        @testset for algorithm in ALL_ALGORITHMS
            @testset for (name, value, match) in [
                (:id, 0, "positive"),
                (:num_iterations, 0, "positive"),
                (:init_radius, -0.1, "non-negative"),
            ]

                @test_throws match optimize(
                    bernoulli_model,
                    BERNOULLI_DATA;
                    algorithm = algorithm,
                    Dict(name => value)...,
                )
            end

            # lbfgs only

            if algorithm == TinyStan.LBFGS
                @test_throws "positive" optimize(
                    bernoulli_model,
                    BERNOULLI_DATA;
                    algorithm = algorithm,
                    Dict(:max_history_size => 0)...,
                )
            else
                optimize(
                    bernoulli_model,
                    BERNOULLI_DATA;
                    algorithm = algorithm,
                    Dict(:max_history_size => 0)...,
                )
            end

            # lbfgs or bfgs

            @testset for (name, value, match) in [
                (:init_alpha, 0.0, "positive"),
                (:tol_obj, 0.0, "positive"),
                (:tol_rel_obj, 0.0, "positive"),
                (:tol_grad, 0.0, "positive"),
                (:tol_rel_grad, 0.0, "positive"),
                (:tol_param, 0.0, "positive"),
            ]
                if algorithm != TinyStan.NEWTON
                    @test_throws match optimize(
                        bernoulli_model,
                        BERNOULLI_DATA;
                        algorithm = algorithm,
                        Dict(name => value)...,
                    )
                else
                    optimize(
                        bernoulli_model,
                        BERNOULLI_DATA;
                        algorithm = algorithm,
                        Dict(name => value)...,
                    )
                end
            end

        end
    end
end
