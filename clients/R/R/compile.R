IS_WINDOWS <- isTRUE(.Platform$OS.type == "windows")
MAKE <- Sys.getenv("MAKE", "make")


verify_tinystan_path <- function(path) {
  suppressWarnings({
    folder <- normalizePath(path)
  })
  if (!dir.exists(folder)) {
    stop(paste0(
      "TinyStan folder '",
      folder,
      "' does not exist!\n",
      "If you need to set a different location, call 'set_tinystan_path()'"
    ))
  }
  makefile <- file.path(folder, "Makefile")
  if (!file.exists(makefile)) {
    stop(paste0(
      "TinyStan folder '",
      folder,
      "' does not contain file 'Makefile',",
      " please ensure it is built properly!\n",
      "If you need to set a different location, call 'set_tinystan_path()'"
    ))
  }
}

#' @title Function `set_tinystan_path()`
#' @description Set the path to TinyStan.
#' @details This should point to the top-level folder of the repository.
#' @export
set_tinystan_path <- function(path) {
  verify_tinystan_path(path)
  Sys.setenv(TINYSTAN = normalizePath(path))
}

#' Get the path to TinyStan.
#'
#' By default this is set to the value of the environment
#' variable `TINYSTAN`.
#'
#' If there is no path set, this function will download
#' a matching version of TinyStan to a folder called
#' `.tinystan` in the user's home directory.
#'
#' @seealso [set_tinystan_path]
get_tinystan_path <- function() {
  # try to get from environment
  path <- Sys.getenv("TINYSTAN", unset = "")
  if (path == "") {
    path <- CURRENT_TINYSTAN
    tryCatch(
      {
        verify_tinystan_path(path)
      },
      error = function(e) {
        print(paste0(
          "TinyStan not found at location specified by $TINYSTAN ",
          "environment variable, downloading version ",
          packageVersion("tinystan"),
          " to ",
          path
        ))
        get_tinystan_src()
      }
    )
  }

  return(path)
}


#' @title Function `compile_model()`
#' @description Compiles a Stan model.
#' @details Run TinyStan's Makefile on a `.stan` file, creating
#' the `.so` used by the StanModel class.
#' This function checks that the path to TinyStan is valid
#' and will error if not. This can be set with `set_tinystan_path`.
#'
#' @param stan_file A path to a Stan model file.
#' @param stanc_arg A vector of arguments to pass to stanc3.
#' For example, `c('--O1')` will enable compiler optimization level 1.
#' @param make_args A vector of additional arguments to pass to Make.
#' For example, `c('STAN_NO_RANGE_CHECKS=True')` will disable bounds
#' checking in the Stan Math library. If the same flags are defined
#' in `make/local`, the versions passed here will take precedent.
#' @return Path to the compiled model.
#'
#' @seealso [tinystan::set_tinystan_path()]
#' @export
compile_model <- function(stan_file, stanc_args = NULL, make_args = NULL) {
  verify_tinystan_path(get_tinystan_path())
  suppressWarnings({
    file_path <- normalizePath(stan_file)
  })
  if (tools::file_ext(file_path) != "stan") {
    stop(paste0("File '", file_path, "' does not end with '.stan'"))
  }
  if (!file.exists(file_path)) {
    stop(paste0("File '", file_path, "' does not exist!"))
  }

  output <- paste0(tools::file_path_sans_ext(file_path), "_model.so")
  stancflags <- paste("--include-paths=.", paste(stanc_args, collapse = " "))

  flags <- c(
    paste("-C", get_tinystan_path()),
    make_args,
    paste0("STANCFLAGS=\"", stancflags, "\""),
    output
  )

  suppressWarnings({
    res <- system2(MAKE, args = flags, stdout = TRUE, stderr = TRUE)
  })
  res_attrs <- attributes(res)
  if ("status" %in% names(res_attrs) && res_attrs$status != 0) {
    stop(paste0(
      "Compilation failed with error code ",
      res_attrs$status,
      "\noutput:\n",
      paste(res, collapse = "\n")
    ))
  }

  return(output)
}

tbb_found <- function() {
  suppressWarnings(
    out <- system2("where.exe", "tbb.dll", stdout = NULL, stderr = NULL)
  )
  return(out == 0)
}

WINDOWS_PATH_SET <- FALSE

windows_dll_path_setup <- function() {
  if (.Platform$OS.type == "windows" && !WINDOWS_PATH_SET) {
    if (tbb_found()) {
      assign("WINDOWS_PATH_SET", TRUE, envir = .GlobalEnv)
    } else {
      tbb_path <- file.path(
        get_tinystan_path(),
        "stan",
        "lib",
        "stan_math",
        "lib",
        "tbb"
      )
      Sys.setenv(PATH = paste(tbb_path, Sys.getenv("PATH"), sep = ";"))
      assign("WINDOWS_PATH_SET", tbb_found(), envir = .GlobalEnv)
    }
  }
}
