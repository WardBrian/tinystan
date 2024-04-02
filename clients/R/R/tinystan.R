#' Compile a Stan model
#' @export
#' @examples
#' data_file <- system.file("bernoulli.data.json", package = "tinystan")
#' mod <- tinystan_model(system.file("bernoulli.stan", package = "tinystan"))
#' fit = sampler(private = mod, data = data_file)
#' fit
#'
tinystan_model = function(lib, stanc_args = NULL, make_args = NULL, warn = TRUE) {
    if (tools::file_ext(lib) == "stan") {
        # FIXME: This is a hack to get the code from the stan file
        stan_code <- lib
        lib <- compile_model(lib, stanc_args, make_args)
        code <- paste0(readLines(stan_code), collapse = "\n")
        built_with_so <- FALSE
    } else {
        build_with_so <- TRUE
        code <- "Built with .so object. No code available for printing"
    }
    if (.Platform$OS.type == "windows") {
        lib_old <- lib
        lib <- paste0(tools::file_path_sans_ext(lib), ".dll")
        file.copy(from = lib_old, to = lib)
        windows_dll_path_setup()
    }
    lib <- tools::file_path_as_absolute(lib)
    lib_name <- tools::file_path_sans_ext(basename(lib))
    if (warn && is.loaded("tinystan_create_model_R", PACKAGE = lib_name)) {
        warning(paste0("Loading a shared object '", lib, "' which is already loaded.\n",
            "If the file has changed since the last time it was loaded, this load may not update the library!"))
    }
    dyn.load(lib, PACKAGE = lib_name)
    sep <- .C("tinystan_separator_char_R", sep = raw(1), PACKAGE = lib_name)$sep
    sep <- rawToChar(sep)
    ret <- list(lib = lib, lib_name = lib_name, sep = sep, code = code, built_with_so = built_with_so)
    class(ret) <- c("tinystan_model", class(ret))
    return(ret)
}

#' @export
print.tinystan_model <- function(mod, ...) {
    cat(mod$code, ...)
    if (mod$built_with_so) {
        cat("Library: ", mod$lib, "\n")
    }
}

#'@export
api_version = function(stan_model) {
    .C("tinystan_api_version", major = integer(1), minor = integer(1), patch = integer(1),
        PACKAGE = stan_model$lib_name)
}

#' @noRd
with_model = function(private, data, seed, block) {
    ffi_ret <- .C("tinystan_create_model_R", model = raw(8), as.character(data),
        as.integer(seed), err = raw(8), NAOK = TRUE, PACKAGE = private$lib_name)
    handle_error(all(ffi_ret$model == 0), private$lib_name, ffi_ret$err)
    tryCatch({
        # this is the equivalent of base R's `with` function
        eval(substitute(block), ffi_ret, enclos = parent.frame())
    }, finally = {
        .C("tinystan_destroy_model_R", as.raw(ffi_ret$model), PACKAGE = private$lib_name)
    })
}

#' @noRd
get_parameter_names <- function(...) {
    UseMethod("get_parameter_names")
}

#' @noRd
get_parameter_names.tinystan_model = function(private, model) {
    param_names_raw <- .C("tinystan_model_param_names_R", as.raw(model), names = as.character(""),
        PACKAGE = private$lib_name)$names
    if (param_names_raw == "") {
        return(c())
    }
    strsplit(param_names_raw, ",")[[1]]

}

#' @noRd
get_free_params <- function(...) {
    UseMethod("get_free_params")
}


#' @noRd
get_free_params.tinystan_model = function(private, model) {
    .C("tinystan_model_num_free_params_R", as.raw(model), params = as.integer(0),
        PACKAGE = private$lib_name)$params
}

#' @noRd
encode_inits = function(private, inits) {
    if (is.null(inits)) {
        return(as.character(""))
    }
    if (is.character(inits)) {
        if (length(inits) == 1) {
            return(as.character(inits))
        } else {
            return(as.character(paste0(inits, collapse = private$sep)))
        }
    }
    if (is.list(inits)) {
        return(as.character(paste0(inits, collapse = private$sep)))
    }
    stop("inits must be a character vector or a list")
}

#' @export
sampler <- function(...) {
    UseMethod("sampler")
}

#' Run Stan's NUTS sampler
#' @export
#' @examples
#' data_file <- system.file("bernoulli.data.json", package = "tinystan")
#' mod <- tinystan_model(system.file("bernoulli.stan", package = "tinystan"))
#' fit = sampler(private = mod, data = data_file)
#' fit
sampler.tinystan_model = function(private, data = "", num_chains = 4, inits = NULL,
    seed = NULL, id = 1, init_radius = 2, num_warmup = 1000, num_samples = 1000,
    metric = HMCMetric$DIAGONAL, init_inv_metric = NULL, save_metric = FALSE, adapt = TRUE,
    delta = 0.8, gamma = 0.05, kappa = 0.75, t0 = 10, init_buffer = 75, term_buffer = 50,
    window = 25, save_warmup = FALSE, stepsize = 1, stepsize_jitter = 0, max_depth = 10,
    refresh = 0, num_threads = -1) {

    if (num_chains < 1) {
        stop("num_chains must be at least 1")
    }
    if (save_warmup && num_warmup < 0) {
        stop("num_warmup must be non-negative")
    }
    if (num_samples < 1) {
        stop("num_samples must be at least 1")
    }

    if (is.null(seed)) {
        seed <- as.integer(runif(1, min = 0, max = (2^31)))
    }

    with_model(private, data, seed, {
        free_params <- get_free_params(private, model)
        if (free_params == 0) {
            stop("Model has no parameters to sample")
        }

        params <- c(HMC_SAMPLER_VARIABLES, get_parameter_names(private, model))
        num_params <- length(params)
        num_draws <- as.integer(save_warmup) * num_warmup + num_samples
        output_size <- num_params * num_chains * num_draws

        if (metric == HMCMetric$DENSE) {
            metric_shape <- rep(free_params, 2)
        } else {
            metric_shape <- free_params
        }

        if (is.null(init_inv_metric)) {
            metric_has_init <- FALSE
            inv_metric_init <- 0
        } else {
            metric_has_init <- TRUE
            metric_dims <- dim(init_inv_metric)

            if (length(metric_dims) == length(metric_shape) && all(metric_dims ==
                metric_shape)) {
                inv_metric_init <- replicate(num_chains, init_inv_metric)
            } else if (length(metric_dims) == (length(metric_shape) + 1) && all(metric_dims ==
                c(metric_shape, num_chains))) {
                inv_metric_init <- init_inv_metric
            } else {
                stop("Invalid initial metric size. Expected a ", paste(metric_shape,
                  collapse = " x "), " or ", paste(c(metric_shape, num_chains), collapse = " x "),
                  " matrix")
            }
        }

        if (save_metric) {
            metric_size <- num_chains * prod(metric_shape)
        } else {
            metric_size <- 1
        }

        vars <- .C("tinystan_sample_R", return_code = as.integer(0), as.raw(model),
            as.integer(num_chains), encode_inits(private, inits), as.integer(seed),
            as.integer(id), as.double(init_radius), as.integer(num_warmup), as.integer(num_samples),
            as.integer(metric), as.logical(metric_has_init), as.double(inv_metric_init),
            as.logical(adapt), as.double(delta), as.double(gamma), as.double(kappa),
            as.double(t0), as.integer(init_buffer), as.integer(term_buffer), as.integer(window),
            as.logical(save_warmup), as.double(stepsize), as.double(stepsize_jitter),
            as.integer(max_depth), as.integer(refresh), as.integer(num_threads),
            out = double(output_size), as.integer(output_size), save_metric = as.logical(save_metric),
            metric = double(metric_size), err = raw(8), PACKAGE = private$lib_name)
        handle_error(vars$return_code, private$lib_name, vars$err)
        # reshape the output matrix
        out <- output_as_rvars(params, num_draws, num_chains, vars$out)

        if (save_metric) {
            if (metric == HMCMetric$DENSE) {
                metric <- aperm(array(vars$metric, dim = c(free_params, free_params,
                  num_chains)), c(3, 2, 1))
            } else {
                metric <- aperm(array(vars$metric, dim = c(free_params, num_chains)),
                  c(2, 1))
            }
            return(list(draws = out, metric = metric))
        }

        out
    })
}

#' @export
pathfinder = function(private, data = "", num_paths = 4, inits = NULL, seed = NULL,
    id = 1, init_radius = 2, num_draws = 1000, max_history_size = 5, init_alpha = 0.001,
    tol_obj = 1e-12, tol_rel_obj = 10000, tol_grad = 1e-08, tol_rel_grad = 1e+07,
    tol_param = 1e-08, num_iterations = 1000, num_elbo_draws = 100, num_multi_draws = 1000,
    calculate_lp = TRUE, psis_resample = TRUE, refresh = 0, num_threads = -1) {
    if (num_draws < 1) {
        stop("num_draws must be at least 1")
    }
    if (num_paths < 1) {
        stop("num_paths must be at least 1")
    }
    if (num_multi_draws < 1) {
        stop("num_multi_draws must be at least 1")
    }

    if (calculate_lp && psis_resample) {
        num_output <- num_multi_draws
    } else {
        num_output <- num_draws * num_paths
    }
    if (is.null(seed)) {
        seed <- as.integer(runif(1, min = 0, max = (2^31)))
    }

    with_model(private, data, seed, {
        free_params <- get_free_params(private, model)
        if (free_params == 0) {
            stop("Model has no parameters")
        }

        params <- c(PATHFINDER_VARIABLES, get_parameter_names(private, model))
        num_params <- length(params)
        output_size <- num_params * num_output

        vars <- .C("tinystan_pathfinder_R", return_code = as.integer(0), as.raw(model),
            as.integer(num_paths), encode_inits(private, inits), as.integer(seed),
            as.integer(id), as.double(init_radius), as.integer(num_draws), as.integer(max_history_size),
            as.double(init_alpha), as.double(tol_obj), as.double(tol_rel_obj), as.double(tol_grad),
            as.double(tol_rel_grad), as.double(tol_param), as.integer(num_iterations),
            as.integer(num_elbo_draws), as.integer(num_multi_draws), as.integer(calculate_lp),
            as.integer(psis_resample), as.integer(refresh), as.integer(num_threads),
            out = double(output_size), as.integer(output_size), err = raw(8), PACKAGE = private$lib_name)
        handle_error(vars$return_code, private$lib_name, vars$err)

        output_as_rvars(params, num_output, 1, vars$out)
    })
}

#' @export
optimizer = function(private, data = "", init = NULL, seed = NULL, id = 1, init_radius = 2,
    algorithm = OptimizationAlgorithm$LBFGS, jacobian = FALSE, num_iterations = 2000,
    max_history_size = 5, init_alpha = 0.001, tol_obj = 1e-12, tol_rel_obj = 10000,
    tol_grad = 1e-08, tol_rel_grad = 1e+07, tol_param = 1e-08, refresh = 0, num_threads = -1) {

    if (is.null(seed)) {
        seed <- as.integer(runif(1, min = 0, max = (2^31)))
    }

    with_model(private, data, seed, {
        params <- c(OPTIMIZATION_VARIABLES, get_parameter_names(private, model))
        num_params <- length(params)
        output_size <- num_params

        vars <- .C("tinystan_optimize_R", return_code = as.integer(0), as.raw(model),
            encode_inits(private, init), as.integer(seed), as.integer(id), as.double(init_radius),
            as.integer(algorithm), as.integer(num_iterations), as.logical(jacobian),
            as.integer(max_history_size), as.double(init_alpha), as.double(tol_obj),
            as.double(tol_rel_obj), as.double(tol_grad), as.double(tol_rel_grad),
            as.double(tol_param), as.integer(refresh), as.integer(num_threads), out = double(output_size),
            as.integer(output_size), err = raw(8), PACKAGE = private$lib_name)
        handle_error(vars$return_code, private$lib_name, vars$err)

        output_as_rvars(params, 1, 1, vars$out)
    })
}

#' @export
laplace_sampler = function(private, mode, data = "", num_draws = 1000, jacobian = TRUE,
    calculate_lp = TRUE, save_hessian = FALSE, seed = NULL, refresh = 0, num_threads = -1) {

    if (num_draws < 1) {
        stop("num_draws must be at least 1")
    }
    if (is.null(seed)) {
        seed <- as.integer(runif(1, min = 0, max = (2^31)))
    }

    with_model(private, data, seed, {
        params <- c(LAPLACE_VARIABLES, get_parameter_names(private, model))
        num_params <- length(params)
        free_params <- get_free_params(private, model)

        if (save_hessian) {
            hessian_size <- free_params * free_params
        } else {
            hessian_size <- 1
        }

        if (is.numeric(mode)) {
            if (length(mode) != num_params - length(LAPLACE_VARIABLES)) {
                stop("Mode array has incorrect length.")
            }
            mode_array <- as.double(mode)
            mode_json <- as.character("")
            use_array <- TRUE
        } else {
            mode_array <- as.double(0)
            mode_json <- as.character(mode)
            use_array <- FALSE
        }

        vars <- .C("tinystan_laplace_sample_R", return_code = as.integer(0), as.raw(model),
            as.logical(use_array), mode_array, mode_json, as.integer(seed), as.integer(num_draws),
            as.logical(jacobian), as.logical(calculate_lp), as.integer(refresh),
            as.integer(num_threads), out = double(num_params * num_draws), as.integer(num_params *
                num_draws), as.logical(save_hessian), hessian = double(hessian_size),
            err = raw(8), PACKAGE = private$lib_name)
        handle_error(vars$return_code, private$lib_name, vars$err)

        out <- output_as_rvars(params, num_draws, 1, vars$out)
        if (save_hessian) {
            hessian <- array(vars$hessian, dim = c(free_params, free_params))
            return(list(draws = out, hessian = hessian))
        }
        out
    })

}



#' Get and free the error message stored at the C++ pointer
#' @keywords internal
handle_error <- function(rc, lib_name, err_ptr) {
    if (rc != 0) {
        if (all(err_ptr == 0)) {
            stop(paste("Unknown error, function returned code", rc))
        }
        msg <- .C("tinystan_get_error_message_R", as.raw(err_ptr), err_msg = as.character(""),
            PACKAGE = lib_name)$err_msg
        type <- .C("tinystan_get_error_type_R", as.raw(err_ptr), err_type = as.integer(0),
            PACKAGE = lib_name)$err_type
        .C("tinystan_free_stan_error_R", as.raw(err_ptr), PACKAGE = lib_name)
        if (type == 3) {
            if (requireNamespace("rlang", quietly = TRUE)) {
                rlang::interrupt()
            }
            msg <- "User interrupt"
        }
        stop(msg)
    }
}
