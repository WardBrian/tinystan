#ifndef TINYSTAN_H
#define TINYSTAN_H

/// \file tinystan.h

#ifdef __cplusplus
#include <cstddef>
struct TinyStanError;
struct TinyStanModel;
extern "C" {
#else
#include <stddef.h>
#include <stdbool.h>
typedef struct TinyStanError TinyStanError;  // opaque type
typedef struct TinyStanModel TinyStanModel;  // opaque type
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
 * @param[in] data JSON-encoded data.
 * @param[in] seed Random seed.
 * @param[out] err Error information.
 * @return A pointer to the model. Must later be freed with
 * tinystan_destroy_model().
 */
TINYSTAN_PUBLIC TinyStanModel *tinystan_create_model(const char *data,
                                                     unsigned int seed,
                                                     TinyStanError **err);

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
 * Currently, this is ASCII 0x1C, the file separator character.
 */
TINYSTAN_PUBLIC char tinystan_separator_char();

enum TINYSTAN_PUBLIC TinyStanMetric { unit = 0, dense = 1, diagonal = 2 };

TINYSTAN_PUBLIC int tinystan_sample(
    const TinyStanModel *model, size_t num_chains, const char *inits,
    unsigned int seed, unsigned int chain_id, double init_radius,
    int num_warmup, int num_samples, TinyStanMetric metric_choice,
    const double *init_inv_metric, bool adapt, double delta, double gamma,
    double kappa, double t0, unsigned int init_buffer, unsigned int term_buffer,
    unsigned int window, bool save_warmup, double stepsize,
    double stepsize_jitter, int max_depth, int refresh, int num_threads,
    double *out, size_t out_size, double *metric_out, TinyStanError **err);

TINYSTAN_PUBLIC int tinystan_pathfinder(
    const TinyStanModel *model, size_t num_paths, const char *inits,
    unsigned int seed, unsigned int id, double init_radius, int num_draws,
    /* tuning params */ int max_history_size, double init_alpha, double tol_obj,
    double tol_rel_obj, double tol_grad, double tol_rel_grad, double tol_param,
    int num_iterations, int num_elbo_draws, int num_multi_draws,
    bool calculate_lp, bool psis_resample, int refresh, int num_threads,
    double *out, size_t out_size, TinyStanError **err);

enum TinyStanOptimizationAlgorithm { newton = 0, bfgs = 1, lbfgs = 2 };

TINYSTAN_PUBLIC int tinystan_optimize(
    const TinyStanModel *model, const char *init, unsigned int seed,
    unsigned int id, double init_radius,
    TinyStanOptimizationAlgorithm algorithm, int num_iterations, bool jacobian,
    /* tuning params */ int max_history_size, double init_alpha, double tol_obj,
    double tol_rel_obj, double tol_grad, double tol_rel_grad, double tol_param,
    int refresh, int num_threads, double *out, size_t out_size,
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

enum TINYSTAN_PUBLIC TinyStanErrorType {
  generic = 0,
  config = 1,
  interrupt = 2
};

/**
 * Get the type of error.
 *
 * @param[in] err The error object.
 * @return The type of error.
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
TINYSTAN_PUBLIC void tinystan_free_stan_error(TinyStanError *err);

typedef void (*TINYSTAN_PRINT_CALLBACK)(const char *msg, size_t len, bool bad);

TINYSTAN_PUBLIC void tinystan_set_print_callback(TINYSTAN_PRINT_CALLBACK print);

#ifdef __cplusplus
}
#endif
#endif
