#ifndef FFISTAN_H
#define FFISTAN_H

/// \file ffistan.h

#ifdef __cplusplus
struct FFIStanError;
struct FFIStanModel;
extern "C" {
#else
#include <stddef.h>
#include <stdbool.h>
typedef struct FFIStanError FFIStanError;  // opaque type
typedef struct FFIStanModel FFIStanModel;  // opaque type
#endif

/**
 * Get the version of the library.
 * @param[out] major The major version number.
 * @param[out] minor The minor version number.
 * @param[out] patch The patch version number.
 */
void ffistan_version(int *major, int *minor, int *patch);

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
 * ffistan_destroy_model().
 */
FFIStanModel *ffistan_create_model(const char *data, unsigned int seed,
                                   FFIStanError **err);

/**
 * Deallocate a model.
 * @param[in] model The model to deallocate.
 */
void ffistan_destroy_model(FFIStanModel *model);

/**
 * Get the names of the parameters.
 *
 * @param[in] model The model.
 * @return A string containing the names of the parameters, comma separated.
 * Multidimensional parameters are flattened, e.g. "foo[2,3]" becomes 6 strings,
 * starting with "foo.1.1".
 */
const char *ffistan_model_param_names(const FFIStanModel *model);

/**
 * Get the number of free parameters, i.e., those declared in the parameters
 * block, not transformed parameters or generated quantities.
 *
 * @param[in] model The model.
 * @return The number of free parameters.
 */
size_t ffistan_model_num_free_params(const FFIStanModel *model);

/**
 * Returns the separator character which must be used
 * to provide multiple initialization files or json strings.
 *
 * Currently, this is ASCII 0x1C, the file separator character.
 */
char ffistan_separator_char();

enum FFIStanMetric { unit = 0, dense = 1, diagonal = 2 };

int ffistan_sample(const FFIStanModel *model, size_t num_chains,
                   const char *inits, unsigned int seed, unsigned int chain_id,
                   double init_radius, int num_warmup, int num_samples,
                   FFIStanMetric metric_choice, bool adapt, double delta,
                   double gamma, double kappa, double t0,
                   unsigned int init_buffer, unsigned int term_buffer,
                   unsigned int window, bool save_warmup, double stepsize,
                   double stepsize_jitter, int max_depth, int refresh,
                   int num_threads, double *out, double *metric_out,
                   FFIStanError **err);

int ffistan_pathfinder(const FFIStanModel *ffimodel, size_t num_paths,
                       const char *inits, unsigned int seed, unsigned int id,
                       double init_radius, int num_draws,
                       /* tuning params */ int max_history_size,
                       double init_alpha, double tol_obj, double tol_rel_obj,
                       double tol_grad, double tol_rel_grad, double tol_param,
                       int num_iterations, int num_elbo_draws,
                       int num_multi_draws, int refresh, int num_threads,
                       double *out, FFIStanError **err);

enum FFIStanOptimizationAlgorithm { newton = 0, bfgs = 1, lbfgs = 2 };

int ffistan_optimize(const FFIStanModel *ffimodel, const char *init,
                     unsigned int seed, unsigned int id, double init_radius,
                     FFIStanOptimizationAlgorithm algorithm, int num_iterations,
                     bool jacobian,
                     /* tuning params */ int max_history_size,
                     double init_alpha, double tol_obj, double tol_rel_obj,
                     double tol_grad, double tol_rel_grad, double tol_param,
                     int refresh, int num_threads, double *out,
                     FFIStanError **err);

/**
 * Get the error message from an error object.
 *
 * @param[in] err The error object.
 * @return The error message. Will be freed when the error object is freed, so
 * copy it if you need it later.
 */
const char *ffistan_get_error_message(const FFIStanError *err);

/**
 * Free the error object.
 *
 * @note This will invalidate any pointers returned by
 * ffistan_get_error_message().
 *
 * @param[in] err The error object.
 */
void ffistan_free_stan_error(FFIStanError *err);

#ifdef __cplusplus
}
#endif
#endif
