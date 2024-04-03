#' @export
HMCMetric <- list(UNIT = 0, DENSE = 1, DIAGONAL = 2)

HMC_SAMPLER_VARIABLES = c("lp__", "accept_stat__", "stepsize__", "treedepth__", "n_leapfrog__",
    "divergent__", "energy__")

PATHFINDER_VARIABLES = c("lp_approx__", "lp__")

#' @export
OptimizationAlgorithm <- list(NEWTON = 0, BFGS = 1, LBFGS = 2)

OPTIMIZATION_VARIABLES = c("lp__")

LAPLACE_VARIABLES = c("log_p__", "log_g__")
