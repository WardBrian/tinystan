import ffistan

if __name__ == "__main__":
    model = ffistan.FFIStanModel("./test_models/bernoulli/bernoulli.stan")
    data = "./test_models/bernoulli/bernoulli.data.json"
    fit = model.sample(data, num_samples=10000, num_chains=10)

    print(fit.parameters)
    print(fit["theta"].mean())
    print(fit["theta"].shape)

    pf = model.pathfinder(data)
    print(pf.parameters)
    print(pf["theta"].mean())
    print(pf["theta"].shape)

    data = {"N": 10, "y": [0, 1, 0, 0, 0, 0, 0, 0, 0, 1]}
    o = model.optimize(data, jacobian=True)
    print(o.parameters)
    print(o["theta"].mean())
    print(o["theta"].shape)
