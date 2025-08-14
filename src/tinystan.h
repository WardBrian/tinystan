#ifndef TINYSTAN_H
#define TINYSTAN_H

/// \file tinystan.h

#include "tinystan_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#if defined _WIN32 || defined __MINGW32__
#define TINYSTAN_ON_WINDOWS 1
#else
#define TINYSTAN_ON_WINDOWS 0
#endif

#if TINYSTAN_ON_WINDOWS
#ifdef TINYSTAN_EXPORT
#define TINYSTAN_PUBLIC __declspec(dllexport)
#else
#define TINYSTAN_PUBLIC __declspec(dllimport)
#endif
#else
#define TINYSTAN_PUBLIC __attribute__((visibility("default")))
#endif

/**
 * Get the version of the library.
 * @param[out] major The major version number.
 * @param[out] minor The minor version number.
 * @param[out] patch The patch version number.
 */
TINYSTAN_PUBLIC void tinystan_api_version(int *major, int *minor, int *patch);

/**
 * Get the version of Stan this library is built against.
 * @param[out] major The major version number.
 * @param[out] minor The minor version number.
 * @param[out] patch The patch version number
 */
TINYSTAN_PUBLIC void tinystan_stan_version(int *major, int *minor, int *patch);

/**
 * Instantiate a model from JSON-encoded data.
 *
 * Besides being an argument to the algorithm functions, the model
 * can be used to query the names of the parameters and the number
 * of free parameters. This is essential for constructing output
 * buffers of appropriate size.
 *
 * @param[in] data A path to a JSON file or a string containing JSON-encoded
 * data. Can be `NULL` or an empty string if the model does not require data.
 * @param[in] seed Random seed.
 * @param[in] user_print_callback Callback function for printing messages. Can
 * be `NULL`, in which case cout/cerr are used instead.
 * @param[out] err Error information. Can be `NULL`.
 * @return A pointer to the model. Must later be freed with
 * tinystan_destroy_model(). Returns `NULL` on error.
 */
TINYSTAN_PUBLIC TinyStanModel *tinystan_create_model(
    const char *data, unsigned int seed,
    TINYSTAN_PRINT_CALLBACK user_print_callback, TinyStanError **err);

/**
 * Deallocate a model.
 * @param[in] model The model to deallocate.
 */
TINYSTAN_PUBLIC void tinystan_destroy_model(TinyStanModel *model);

/**
 * Get the names of the parameters.
 *
 * @param[in] model The model.
 * @return A string containing the names of the parameters, comma separated.
 * Multidimensional parameters are flattened, e.g. "foo[2,3]" becomes 6 strings,
 * starting with "foo.1.1".
 */
TINYSTAN_PUBLIC const char *tinystan_model_param_names(
    const TinyStanModel *model);

/**
 * Get the number of free parameters, i.e., those declared in the parameters
 * block, not transformed parameters or generated quantities.
 *
 * @param[in] model The model.
 * @return The number of free parameters.
 */
TINYSTAN_PUBLIC size_t
tinystan_model_num_free_params(const TinyStanModel *model);

/**
 * Returns the separator character which must be used
 * to provide multiple initialization files or json strings.
 *
 * Currently, this is ASCII `0x1C`, the file separator character.
 */
TINYSTAN_PUBLIC char tinystan_separator_char();

/**
 * @brief Run Stan's No-U-Turn Sampler (NUTS) to sample from the posterior.
 *
 * A wrapper around several functions in the `stan::services::sample` namespace.
 * Same-named arguments should be interpreted as having the same meaning as in
 * the Stan documentation.
 *
 * @param[in] model The TinyStanModel to use for the sampling.
 * @param[in] num_chains The number of chains to run.
 * @param[in] inits Initial parameter values. This should be a path
 * to a JSON file or a JSON string. If `num_chains` is greater than 1,
 * this can be a list of paths or JSON strings separated by the
 * separator character returned by tinystan_separator_char().
 * @param[in] seed The seed to use for the random number generator.
 * @param[in] chain_id Chain ID for the first chain.
 * @param[in] init_radius Radius to initialize unspecified parameters within.
 * @param[in] num_warmup Number of warmup iterations to run.
 * @param[in] num_samples Number of samples to draw after warmup.
 * @param[in] metric_choice The type of inverse mass matrix to use in the sampler.
 * @param[in] init_inv_metric Initial value for the inverse mass matrix used
 * by the sampler. Depending on `metric_choice`, this should be a flattened
 * matrix for a dense metric, or a vector for a diagonal one. If `NULL`, the
 * sampler will use the identity matrix.
 * @param[in] adapt Whether the sampler should adapt the step size and metric.
 * @param[in] delta Target average acceptance probability.
 * @param[in] gamma Adaptation regularization scale.
 * @param[in] kappa Adaptation relaxation exponent.
 * @param[in] t0 Adaptation iteration offset.
 * @param[in] init_buffer Number of warmup samples to use for initial step size
 * adaptation.
 * @param[in] term_buffer Number of warmup samples to use for step size
 * adaptation after the metric is adapted.
 * @param[in] window Initial number of iterations to use for metric adaptation.
 * @param[in] save_warmup Whether to save the warmup samples.
 * @param[in] stepsize Initial step size for the sampler.
 * @param[in] stepsize_jitter Amount of random jitter to add to the step size.
 * @param[in] max_depth Maximum tree depth for the sampler.
 * @param[in] refresh Number of iterations between progress messages.
 * @param[in] num_threads Number of threads to use for sampling.
 * @param[out] out Buffer to store the samples. The buffer should be large
 * enough to store `num_chains * (num_samples  + save_warmup * num_warmup) *
 * num_params` doubles.
 * @param[in] out_size Size of the buffer in doubles. Used for bounds checking
 * unless TINYSTAN_NO_BOUNDS_CHECK is defined, in which case it is ignored.
 * @param[out] inv_metric_out Buffer to store the adapted stepsizes. Can be
 * `NULL`. If non-NULL, the buffer should be of length `num_chains`
 * @param[out] inv_metric_out Buffer to store the inverse metric. Can be `NULL`.
 * If non-NULL, the buffer should be large enough to store `num_chains` *
 * `tinystan_model_num_free_params()` doubles if using a diagonal matrix, and
 * `num_chains` * the free parameters squared if using a dense one.
 * @param[out] err Error information. Can be `NULL`.
 *
 * @return Zero on success, non-zero on error. If an error occurs, `err`
 * will be set to a non-NULL value which must be freed with
 * tinystan_destroy_error().
 */
TINYSTAN_PUBLIC int tinystan_sample(
    const TinyStanModel *model, size_t num_chains, const char *inits,
    unsigned int seed, unsigned int chain_id, double init_radius,
    int num_warmup, int num_samples, TinyStanMetric metric_choice,
    const double *init_inv_metric, bool adapt, double delta, double gamma,
    double kappa, double t0, unsigned int init_buffer, unsigned int term_buffer,
    unsigned int window, bool save_warmup, double stepsize,
    double stepsize_jitter, int max_depth, int refresh, int num_threads,
    double *out, size_t out_size, double *stepsize_out, double *inv_metric_out,
    TinyStanError **err);

/**
 * @brief Run the Pathfinder algorithm to approximate the posterior.
 *
 * A wrapper around the functions in the `stan::services::pathfinder` namespace.
 * Same-named arguments should be interpreted as having the same meaning as in
 * the Stan documentation.
 *
 * @param[in] model The TinyStanModel to use for the sampling.
 * @param[in] num_paths The number of individual runs of the algorithm to run in
 * parallel.
 * @param[in] inits Initial parameter values. This should be a path
 * to a JSON file or a JSON string. If `num_paths` is greater than 1,
 * this can be a list of paths or JSON strings separated by the
 * separator character returned by tinystan_separator_char().
 * @param[in] seed The seed to use for the random number generator.
 * @param[in] id ID for the first path.
 * @param[in] init_radius Radius to initialize unspecified parameters within.
 * @param[in] num_draws Number of approximate draws drawn from each of the
 * `num_paths` Pathfinders.
 * @param[in] max_history_size History size used by the internal L-BFGS
 * algorithm to approximate the Hessian.
 * @param[in] init_alpha Initial step size for the internal L-BFGS algorithm.
 * @param[in] tol_obj Convergence tolerance for the objective function for
 * the internal L-BFGS algorithm.
 * @param[in] tol_rel_obj Relative convergence tolerance for the objective
 * function for the internal L-BFGS algorithm.
 * @param[in] tol_grad Convergence tolerance for the gradient norm for the
 * internal L-BFGS algorithm.
 * @param[in] tol_rel_grad Relative convergence tolerance for the gradient norm
 * for the internal L-BFGS algorithm.
 * @param[in] tol_param Convergence tolerance for the changes in parameters for
 * the internal L-BFGS algorithm.
 * @param[in] num_iterations Maximum number of iterations for the internal
 * L-BFGS algorithm.
 * @param[in] num_elbo_draws Number of Monte Carlo draws used to estimate the
 * ELBO.
 * @param[in] num_multi_draws Number of draws returned by Multi-Pathfinder.
 * @param[in] calculate_lp Whether to calculate the log probability of the
 * approximate draws.
 * @param[in] psis_resample Whether to use Pareto smoothed importance sampling
 * on the approximate draws.
 * @param[in] refresh Number of iterations between progress messages.
 * @param[in] num_threads Number of threads to use for Pathfinder.
 * @param[out] out Buffer to store the samples. The buffer should be large
 * enough to store `num_multi_draws` doubles if psis_resample is true, or
 * `num_paths * num_draws` doubles otherwise.
 * @param[in] out_size Size of the buffer in doubles. Used for bounds checking
 * unless TINYSTAN_NO_BOUNDS_CHECK is defined, in which case it is ignored.
 * @param[out] err Error information. Can be `NULL`.
 *
 * @return Zero on success, non-zero on error. If an error occurs, `err`
 * will be set to a non-NULL value which must be freed with
 * tinystan_destroy_error().
 */
TINYSTAN_PUBLIC int tinystan_pathfinder(
    const TinyStanModel *model, size_t num_paths, const char *inits,
    unsigned int seed, unsigned int id, double init_radius, int num_draws,
    /* tuning params */ int max_history_size, double init_alpha, double tol_obj,
    double tol_rel_obj, double tol_grad, double tol_rel_grad, double tol_param,
    int num_iterations, int num_elbo_draws, int num_multi_draws,
    bool calculate_lp, bool psis_resample, int refresh, int num_threads,
    double *out, size_t out_size, TinyStanError **err);

/**
 * @brief Optimize the model parameters using the specified algorithm.
 *
 * A wrapper around the functions in the `stan::services::optimize` namespace.
 * Same-named arguments should be interpreted as having the same meaning as in
 * the Stan documentation.
 *
 * @param[in] model The TinyStanModel to use for the optimization.
 * @param[in] init Initial parameter values. This should be a path
 * to a JSON file or a JSON string.
 * @param[in] seed The seed to use for the random number generator.
 * @param[in] id ID used to offset the random number generator.
 * @param[in] init_radius Radius to initialize unspecified parameters within.
 * @param[in] algorithm Which optimization algorithm to use. Some of the
 * following arguments may be ignored depending on the algorithm.
 * @param[in] num_iterations Maximum number of iterations to run the
 * optimization.
 * @param[in] jacobian Whether to apply the Jacobian change of variables to the
 * log density. If False, the algorithm will find the MLE.
 * If True, the algorithm will find the MAP estimate.
 * @param[in] max_history_size History size used to approximate the Hessian.
 * @param[in] init_alpha Initial step size.
 * @param[in] tol_obj Convergence tolerance for the objective function.
 * @param[in] tol_rel_obj Relative convergence tolerance for the objective
 * function.
 * @param[in] tol_grad Convergence tolerance for the gradient norm.
 * @param[in] tol_rel_grad Relative convergence tolerance for the gradient norm.
 * @param[in] tol_param Convergence tolerance for the changes in parameters.
 * @param[in] refresh Number of iterations between progress messages.
 * @param[in] num_threads Number of threads to use for log density evaluations.
 * @param[out] out Buffer to store the samples. The buffer should be large
 * enough to store `num_params` doubles.
 * @param[in] out_size Size of the buffer in doubles. Used for bounds checking
 * unless TINYSTAN_NO_BOUNDS_CHECK is defined, in which case it is ignored.
 * @param[out] err Error information. Can be `NULL`.
 *
 * @return Zero on success, non-zero on error. If an error occurs, `err`
 * will be set to a non-NULL value which must be freed with
 * tinystan_destroy_error().
 */
TINYSTAN_PUBLIC int tinystan_optimize(
    const TinyStanModel *model, const char *init, unsigned int seed,
    unsigned int id, double init_radius,
    TinyStanOptimizationAlgorithm algorithm, int num_iterations, bool jacobian,
    /* tuning params */ int max_history_size, double init_alpha, double tol_obj,
    double tol_rel_obj, double tol_grad, double tol_rel_grad, double tol_param,
    int refresh, int num_threads, double *out, size_t out_size,
    TinyStanError **err);

/**
 * @brief Sample from the Laplace approximation of the posterior centered at the
 * provided mode.
 *
 * A wrapper around the functions in the `stan::services::laplace_sample`
 * namespace.
 *
 * @param[in] tmodel The TinyStanModel to use for the sampling.
 * @param[in] theta_hat_constr The mode to center the Laplace approximation on.
 * This should be a pointer to an array of doubles on the constrained scale,
 * like the one returned by tinystan_optimize(). At most one of
 * `theta_hat_constr` and `theta_hat_json` should be non-NULL.
 * @param[in] theta_hat_json A path to a JSON file or JSON string representing
 * the mode on the constrained scale. At most one of `theta_hat_constr` and
 * `theta_hat_json` should be non-NULL.
 * @param[in] seed The seed to use for the random number generator.
 * @param[in] num_draws Number of draws.
 * @param[in] jacobian Whether to apply the Jacobian change of variables to the
 * log density.
 * **Note:** This should match the value used when the mode was calculated.
 * @param[in] calculate_lp Whether to calculate the log probability of the
 * samples.
 * @param[in] refresh Number of iterations between progress messages.
 * @param[in] num_threads Number of threads to use for log density evaluations.
 * @param[out] out Buffer to store the samples. The buffer should be large
 * enough to store `num_draws * num_params` doubles.
 * @param[in] out_size Size of the buffer in doubles. Used for bounds checking
 * unless TINYSTAN_NO_BOUNDS_CHECK is defined, in which case it is ignored.
 * @param[out] hessian_out Buffer to store the Hessian matrix calculated at the
 * mode. Can be `NULL`.
 * @param[out] err Error information. Can be `NULL`.
 *
 * @return Zero on success, non-zero on error. If an error occurs, `err` will be
 * set to a non-NULL value which must be freed with tinystan_destroy_error().
 */
TINYSTAN_PUBLIC
int tinystan_laplace_sample(const TinyStanModel *tmodel,
                            const double *theta_hat_constr,
                            const char *theta_hat_json, unsigned int seed,
                            int num_draws, bool jacobian, bool calculate_lp,
                            int refresh, int num_threads, double *out,
                            size_t out_size, double *hessian_out,
                            TinyStanError **err);

/**
 * Get the error message from an error object.
 *
 * @param[in] err The error object.
 * @return The error message. Will be freed when the error object is freed, so
 * copy it if you need it later.
 */
TINYSTAN_PUBLIC const char *tinystan_get_error_message(
    const TinyStanError *err);

/**
 * Get the type of error.
 *
 * @param[in] err The error object.
 * @return The type of error. This is intended to provide more information about
 * the kind of error, allowing for more specific handling. For example, the
 * `interrupt` value can be used to distinguish between a user interrupt and a
 * generic error.
 */
TINYSTAN_PUBLIC TinyStanErrorType
tinystan_get_error_type(const TinyStanError *err);

/**
 * Free the error object.
 *
 * @note This will invalidate any pointers returned by
 * tinystan_get_error_message().
 *
 * @param[in] err The error object.
 */
TINYSTAN_PUBLIC void tinystan_destroy_error(TinyStanError *err);

#ifdef __cplusplus
}
#endif
#endif
