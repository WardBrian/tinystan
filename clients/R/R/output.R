
# copied from cmdstanr, definitely doesn't handle tuples, but then neither does
# posterior
repair_variable_names <- function(names) {
    names <- sub("\\.", "[", names)
    names <- gsub("\\.", ",", names)
    names[grep("\\[", names)] <- paste0(names[grep("\\[", names)], "]")
    names
}


output_as_rvars <- function(names, num_draws, num_chains, draws) {
    names <- repair_variable_names(names)
    num_params <- length(names)
    dims <- c(num_params, num_draws, num_chains)

    # all our outputs are row-major
    draws <- array(draws, dim = dims, dimnames = list(names, NULL, NULL))
    # so we need to rearrange. posterior likes draws x chains x params
    draws <- aperm(draws, c(2, 3, 1))

    posterior::as_draws_rvars(draws)
}
