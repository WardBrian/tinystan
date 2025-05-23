% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/compile.R
\name{compile_model}
\alias{compile_model}
\title{Function \code{compile_model()}}
\usage{
compile_model(stan_file, stanc_args = NULL, make_args = NULL)
}
\arguments{
\item{stan_file}{A path to a Stan model file.}

\item{make_args}{A vector of additional arguments to pass to Make.
For example, \code{c('STAN_NO_RANGE_CHECKS=True')} will disable bounds
checking in the Stan Math library. If the same flags are defined
in \code{make/local}, the versions passed here will take precedent.}

\item{stanc_arg}{A vector of arguments to pass to stanc3.
For example, \code{c('--O1')} will enable compiler optimization level 1.}
}
\value{
Path to the compiled model.
}
\description{
Compiles a Stan model.
}
\details{
Run TinyStan's Makefile on a \code{.stan} file, creating
the \code{.so} used by the StanModel class.
This function checks that the path to TinyStan is valid
and will error if not. This can be set with \code{set_tinystan_path}.
}
\seealso{
\code{\link[=set_tinystan_path]{set_tinystan_path()}}
}
