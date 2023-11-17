options(warnPartialMatchDollar = TRUE, warn = 2)
options(warnPartialMatchArgs = TRUE, warn = 2)
options(warnPartialMatchAttr = TRUE, warn = 2)

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

# hack around the fact that comparisons to NULL result in logical(0) and
# all(logical(0)) is TRUE, for some reason.
.builtin_all <- all
all <- function(l) {
    if (length(l) == 0)
        return(isTRUE(l))
    .builtin_all(l)
}
