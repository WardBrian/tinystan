#ifndef FFISTAN_H
#define FFISTAN_H

/// \file ffistan.h

#ifdef __cplusplus
struct stan_error;
struct FFIStanModel;
extern "C" {
#else
#include <stddef.h>
#include <stdbool.h>
typedef struct stan_error stan_error;      // opaque type
typedef struct FFIStanModel FFIStanModel;  // opaque type
#endif

FFIStanModel *ffistan_create_model(const char *data, unsigned int seed,
                                   stan_error **err);

void ffistan_destroy_model(FFIStanModel *model);
const char *ffistan_model_param_names(const FFIStanModel *model);

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
                   unsigned int window, bool save_warmup, int refresh,
                   double stepsize, double stepsize_jitter, int max_depth,
                   double *out, stan_error **err);

int ffistan_pathfinder(const FFIStanModel *ffimodel, size_t num_paths,
                       const char *inits, unsigned int seed, unsigned int id,
                       double init_radius, int num_draws,
                       /* tuning params */ int max_history_size,
                       double init_alpha, double tol_obj, double tol_rel_obj,
                       double tol_grad, double tol_rel_grad, double tol_param,
                       int num_iterations, int num_elbo_draws,
                       int num_multi_draws, int refresh, double *out,
                       stan_error **err);

const char *ffistan_get_error_message(const stan_error *err);
void ffistan_free_stan_error(stan_error *err);

#ifdef __cplusplus
}
#endif
#endif
