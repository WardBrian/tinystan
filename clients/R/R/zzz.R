HMC_SAMPLER_VARIABLES = c(
  "lp__",
  "accept_stat__",
  "stepsize__",
  "treedepth__",
  "n_leapfrog__",
  "divergent__",
  "energy__"
)

PATHFINDER_VARIABLES = c("lp_approx__", "lp__", "path__")

OPTIMIZATION_VARIABLES = c("lp__")

LAPLACE_VARIABLES = c("log_p__", "log_g__")

#' @title Enumeration `HMCMetric`
#' @description Choices of metric for HMC. Can select from `$UNIT`, `$DENSE`, and `$DIAGONAL`.
#' @export
HMCMetric <- list(UNIT = 0, DENSE = 1, DIAGONAL = 2)

#' @title Enumeration `OptimizationAlgorithm`
#' @description Choices of optimization algorithm. Can select from `$NEWTON`, `$BFGS`, and `$LBFGS`.
#' @export
OptimizationAlgorithm <- list(NEWTON = 0, BFGS = 1, LBFGS = 2)
