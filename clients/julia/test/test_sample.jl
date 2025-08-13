

@testset "Sampling" verbose = true begin

    @testset "Data" begin
        output = sample(bernoulli_model, BERNOULLI_DATA)
        @test 0.2 < mean(output.draws[:, :, output.names.=="theta"]) < 0.3

        output = sample(
            bernoulli_model,
            joinpath(STAN_FOLDER, "bernoulli", "bernoulli.data.json"),
        )
        @test 0.2 < mean(output.draws[:, :, output.names.=="theta"]) < 0.3
    end

    @testset "Save warmup" begin
        output = sample(
            bernoulli_model,
            BERNOULLI_DATA;
            num_warmup = 12,
            num_samples = 34,
            save_warmup = false,
        )
        @test size(output.draws, 2) == 34

        output = sample(
            bernoulli_model,
            BERNOULLI_DATA;
            num_warmup = 12,
            num_samples = 34,
            save_warmup = true,
        )
        @test size(output.draws, 2) == 12 + 34
    end

    @testset "Seed" begin
        output1 = sample(
            bernoulli_model,
            BERNOULLI_DATA;
            num_warmup = 100,
            num_samples = 100,
            seed = UInt32(123),
        )
        output2 = sample(
            bernoulli_model,
            BERNOULLI_DATA;
            num_warmup = 100,
            num_samples = 100,
            seed = UInt32(123),
        )
        @test output1.draws == output2.draws

        output3 = sample(
            bernoulli_model,
            BERNOULLI_DATA;
            num_warmup = 100,
            num_samples = 100,
            seed = UInt32(456),
        )
        @test output1.draws != output3.draws
    end


    @testset "Save metric" begin
        data = "{\"N\": 5}"

        output = sample(
            gaussian_model,
            data;
            num_warmup = 100,
            num_samples = 10,
            save_inv_metric = true,
            metric = TinyStan.UNIT,
        )
        @test size(output.inv_metric) == (4, 5)
        @test isapprox(output.inv_metric, ones(Float64, 4, 5))

        output = sample(
            gaussian_model,
            data;
            num_warmup = 100,
            num_samples = 10,
            save_inv_metric = true,
            metric = TinyStan.DIAGONAL,
        )
        @test size(output.inv_metric) == (4, 5)

        @test isapprox(output.inv_metric, ones(Float64, 4, 5); rtol = 1e-6)

        output = sample(
            gaussian_model,
            data;
            num_warmup = 100,
            num_samples = 10,
            save_inv_metric = true,
            metric = TinyStan.DENSE,
        )
        @test size(output.inv_metric) == (4, 5, 5)
        mat = zeros(Float64, 4, 5, 5)
        for i = 1:5
            mat[:, i, i] .= 1.0
        end
        @test isapprox(output.inv_metric, mat; rtol = 1e-6)


    end

    @testset "Initial metric used" begin
        @testset for adapt in [true, false]
            data = "{\"N\": 3}"
            diag_metric = ones(3, 2)
            diag_metric[:, 1] .= 1e20
            output = sample(
                gaussian_model,
                data;
                num_chains = 2,
                save_warmup = true,
                adapt = adapt,
                metric = TinyStan.DIAGONAL,
                init_inv_metric = diag_metric,
                save_inv_metric = true,
                seed = UInt32(1234),
            )

            chain_one_divergences = sum(output.draws[1, :, output.names.=="divergent__"])
            @test chain_one_divergences > (
                if adapt
                    12
                else
                    500
                end
            )
            chain_two_divergences = sum(output.draws[2, :, output.names.=="divergent__"])
            @test chain_two_divergences < 12
            @test chain_two_divergences < chain_one_divergences
            @test diag_metric != output.inv_metric

            dense_metric = zeros(3, 3, 2)
            for i = 1:2
                for j = 1:3
                    dense_metric[j, j, i] = if i == 1
                        1e20
                    else
                        1
                    end
                end
            end
            output = sample(
                gaussian_model,
                data;
                num_chains = 2,
                save_warmup = true,
                adapt = adapt,
                metric = TinyStan.DENSE,
                init_inv_metric = dense_metric,
                save_inv_metric = true,
                seed = UInt32(1234),
            )
            chain_one_divergences = sum(output.draws[1, :, output.names.=="divergent__"])
            @test chain_one_divergences > (
                if adapt
                    12
                else
                    500
                end
            )
            chain_two_divergences = sum(output.draws[2, :, output.names.=="divergent__"])
            @test chain_two_divergences < 12
            @test chain_two_divergences < chain_one_divergences
            @test dense_metric != output.inv_metric

        end
    end

    @testset "Multiple inits" begin
        init1 = "{\"mu\": -100}"
        output1 =
            sample(multimodal_model; num_warmup = 100, num_samples = 100, inits = init1)
        @test all(output1.draws[:, :, output1.names.=="mu"] .< 0)

        init2 = "{\"mu\": 100}"
        output2 = sample(
            multimodal_model;
            num_warmup = 100,
            num_samples = 100,
            num_chains = 2,
            inits = [init1, init2],
        )
        @test all(output2.draws[1, :, output2.names.=="mu", 1] .< 0)
        @test all(output2.draws[2, :, output2.names.=="mu", 1] .> 0)

        init3 = tempname() * ".json"
        open(init3, "w") do io
            write(io, init1)
        end
        output3 = sample(
            multimodal_model;
            num_warmup = 100,
            num_samples = 100,
            num_chains = 2,
            inits = [init3, init2],
        )
        @test all(output3.draws[1, :, output3.names.=="mu", 1] .< 0)
        @test all(output3.draws[2, :, output3.names.=="mu", 1] .> 0)
    end


    @testset "Bad data" begin
        data1 = "{\"N\": -1}"
        @test_throws "greater than or equal to 0" sample(bernoulli_model, data1)

        data2 = "{\"N\":1, \"y\": [0,1]}"
        @test_throws "mismatch in dimension" sample(bernoulli_model, data2)

        @test_throws "Error in JSON parsing" sample(bernoulli_model, "{\"bad\"}")

        @test_throws "Could not open data file" sample(
            bernoulli_model,
            "path/not/here.json",
        )
    end

    @testset "Bad inits" begin
        init1 = "{\"theta\": 2}"
        @test_throws "Initialization failed" sample(
            bernoulli_model,
            BERNOULLI_DATA;
            inits = init1,
        )

        @test_throws "Could not open data file" sample(
            bernoulli_model,
            BERNOULLI_DATA;
            inits = "bad/path.json",
        )

        init2 = "{\"theta\": 0.2}"
        inits = repeat([init1, init2], 6)
        @test_throws "Initialization failed" sample(
            bernoulli_model,
            BERNOULLI_DATA;
            num_chains = 12,
            inits = inits,
        )

        inits = [init2, init2]
        @test_throws "match the number of chains" sample(
            bernoulli_model,
            BERNOULLI_DATA;
            num_chains = 1,
            inits = inits,
        )
        @test_throws "match the number of chains" sample(
            bernoulli_model,
            BERNOULLI_DATA;
            num_chains = 3,
            inits = inits,
        )

    end

    @testset "Bad initial metric size" begin
        data = "{\"N\": 5}"

        @test_throws "Invalid initial metric size" sample(
            gaussian_model,
            data;
            metric = TinyStan.DENSE,
            init_inv_metric = ones(5),
        )

        @test_throws "Invalid initial metric size" sample(
            gaussian_model,
            data;
            metric = TinyStan.DENSE,
            init_inv_metric = ones(5, 4),
        )

        @test_throws "Invalid initial metric size" sample(
            gaussian_model,
            data;
            num_chains = 2,
            metric = TinyStan.DENSE,
            init_inv_metric = ones(5, 5, 3),
        )

        @test_throws "Invalid initial metric size" sample(
            gaussian_model,
            data;
            metric = TinyStan.DIAGONAL,
            init_inv_metric = ones(4),
        )

        @test_throws "Invalid initial metric size" sample(
            gaussian_model,
            data;
            num_chains = 2,
            metric = TinyStan.DIAGONAL,
            init_inv_metric = ones(5, 3),
        )

        @test_throws "Invalid initial metric size" sample(
            gaussian_model,
            data;
            num_chains = 2,
            metric = TinyStan.DIAGONAL,
            init_inv_metric = ones(5, 5, 3),
        )
    end

    @testset "Bad initial metric" begin
        data = "{\"N\": 3}"
        @test_throws "not positive definite" sample(
            gaussian_model,
            data;
            metric = TinyStan.DENSE,
            init_inv_metric = ones(3, 3) * 1e20,
        )

        metric = zeros(3, 3, 2)
        metric[:, :, 1] .= 1e20
        for i = 1:3
            metric[i, i, 2] = 1
        end

        @test_throws "not positive definite" sample(
            gaussian_model,
            data;
            num_chains = 2,
            metric = TinyStan.DENSE,
            init_inv_metric = metric,
        )
    end

    @testset "Model without parameters" begin
        output = sample(empty_model; save_inv_metric = true)
        @test length(output.names) == 7 # HMC parameters only
        @test prod(size(output.inv_metric)) == 0
    end

    @testset "Bad num_warmup" begin
        @test_throws "non-negative" sample(
            bernoulli_model,
            BERNOULLI_DATA;
            num_warmup = -1,
            save_warmup = false,
        )
        @test_throws "non-negative" sample(
            bernoulli_model,
            BERNOULLI_DATA;
            num_warmup = -1,
            save_warmup = true,
        )
    end

    @testset "Bad arguments" begin
        @testset for (name, value, match) in [
            (:num_chains, 0, "at least 1"),
            (:num_samples, 0, "at least 1"),
            (:id, 0, "positive"),
            (:init_radius, -0.1, "non-negative"),
            (:delta, -0.1, "between 0 and 1"),
            (:delta, 1.1, "between 0 and 1"),
            (:gamma, 0.0, "positive"),
            (:kappa, 0.0, "positive"),
            (:t0, 0, "positive"),
            (:stepsize, 0.0, "positive"),
            (:stepsize_jitter, -0.1, "between 0 and 1"),
            (:stepsize_jitter, 1.1, "between 0 and 1"),
            (:max_depth, 0, "positive"),
            (:num_threads, 0, "positive"),
        ]
            @test_throws match sample(
                bernoulli_model,
                BERNOULLI_DATA;
                Dict(name => value)...,
            )
        end
    end
end
