stan_folder <- file.path("..", "..", "..", "..", "test_models")

bernoulli_model <- ffistan::FFIStanModel$new(file.path(stan_folder, "bernoulli",
    "bernoulli_model.so"))
BERNOULLI_DATA <- "{\"N\": 10, \"y\": [0, 1, 0, 0, 0, 0, 0, 0, 0, 1]}"

gaussian_model <- ffistan::FFIStanModel$new(file.path(stan_folder, "gaussian", "gaussian_model.so"))

empty_model <- ffistan::FFIStanModel$new(file.path(stan_folder, "empty", "empty_model.so"))
multimodal_model <- ffistan::FFIStanModel$new(file.path(stan_folder, "multimodal",
    "multimodal_model.so"))
simple_jacobian_model <- ffistan::FFIStanModel$new(file.path(stan_folder, "simple_jacobian",
    "simple_jacobian_model.so"))
