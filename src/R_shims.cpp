#include "tinystan.h"

// R's .C function can only pass in pointers to C types, so we need to
// provide shims which indirect all our arguments and returns

extern "C" {
TINYSTAN_PUBLIC void tinystan_create_model_R(TinyStanModel** ptr_out,
                                             const char** data,
                                             unsigned int* seed,
                                             TinyStanError** err) {
  *ptr_out = tinystan_create_model(*data, *seed, err);
}

TINYSTAN_PUBLIC void tinystan_destroy_model_R(TinyStanModel** model) {
  tinystan_destroy_model(*model);
}

TINYSTAN_PUBLIC void tinystan_model_param_names_R(TinyStanModel** model,
                                                  char const** names) {
  *names = tinystan_model_param_names(*model);
}

TINYSTAN_PUBLIC void tinystan_model_num_free_params_R(TinyStanModel** model,
                                                      int* n) {
  *n = tinystan_model_num_free_params(*model);
}

TINYSTAN_PUBLIC void tinystan_separator_char_R(char* sep) {
  *sep = tinystan_separator_char();
}

TINYSTAN_PUBLIC void tinystan_sample_R(
    int* return_code, TinyStanModel** model, unsigned int* num_chains,
    char** inits, unsigned int* seed, unsigned int* chain_id,
    double* init_radius, int* num_warmup, int* num_samples, int* metric_choice,
    int* metric_has_init, const double* init_inv_metric, int* adapt,
    double* delta, double* gamma, double* kappa, double* t0,
    unsigned int* init_buffer, unsigned int* term_buffer, unsigned int* window,
    int* save_warmup, double* stepsize, double* stepsize_jitter, int* max_depth,
    int* refresh, int* num_threads, double* out, int* out_size,
    int* save_metric, double* metric_out, TinyStanError** err) {
  //  difficult to directly pass a null pointer from R
  double* metric_out_ptr = nullptr;
  if (*save_metric)
    metric_out_ptr = metric_out;

  const double* init_inv_metric_ptr = nullptr;
  if (*metric_has_init)
    init_inv_metric_ptr = init_inv_metric;

  *return_code = tinystan_sample(
      *model, *num_chains, *inits, *seed, *chain_id, *init_radius, *num_warmup,
      *num_samples, static_cast<TinyStanMetric>(*metric_choice),
      init_inv_metric_ptr, (*adapt != 0), *delta, *gamma, *kappa, *t0,
      *init_buffer, *term_buffer, *window, (*save_warmup != 0), *stepsize,
      *stepsize_jitter, *max_depth, *refresh, *num_threads, out,
      static_cast<size_t>(*out_size), metric_out_ptr, err);
}

TINYSTAN_PUBLIC void tinystan_pathfinder_R(
    int* return_code, TinyStanModel** model, unsigned int* num_paths,
    char** inits, unsigned int* seed, unsigned int* id, double* init_radius,
    int* num_draws, int* max_history_size, double* init_alpha, double* tol_obj,
    double* tol_rel_obj, double* tol_grad, double* tol_rel_grad,
    double* tol_param, int* num_iterations, int* num_elbo_draws,
    int* num_multi_draws, int* calculate_lp, int* psis_resample, int* refresh,
    int* num_threads, double* out, int* out_size, TinyStanError** err) {
  *return_code = tinystan_pathfinder(
      *model, *num_paths, *inits, *seed, *id, *init_radius, *num_draws,
      *max_history_size, *init_alpha, *tol_obj, *tol_rel_obj, *tol_grad,
      *tol_rel_grad, *tol_param, *num_iterations, *num_elbo_draws,
      *num_multi_draws, (*calculate_lp != 0), (*psis_resample != 0), *refresh,
      *num_threads, out, static_cast<size_t>(*out_size), err);
}

TINYSTAN_PUBLIC void tinystan_optimize_R(
    int* return_code, TinyStanModel** model, char** init, unsigned int* seed,
    unsigned int* id, double* init_radius, int* algorithm, int* num_iterations,
    int* jacobian, int* max_history_size, double* init_alpha, double* tol_obj,
    double* tol_rel_obj, double* tol_grad, double* tol_rel_grad,
    double* tol_param, int* refresh, int* num_threads, double* out,
    int* out_size, TinyStanError** err) {
  *return_code = tinystan_optimize(
      *model, *init, *seed, *id, *init_radius,
      static_cast<TinyStanOptimizationAlgorithm>(*algorithm), *num_iterations,
      *jacobian, *max_history_size, *init_alpha, *tol_obj, *tol_rel_obj,
      *tol_grad, *tol_rel_grad, *tol_param, *refresh, *num_threads, out,
      static_cast<size_t>(*out_size), err);
}

TINYSTAN_PUBLIC void tinystan_get_error_message_R(TinyStanError** err,
                                                  char const** err_msg) {
  *err_msg = tinystan_get_error_message(*err);
}

TINYSTAN_PUBLIC void tinystan_get_error_type_R(TinyStanError** err,
                                               int* err_type) {
  *err_type = static_cast<int>(tinystan_get_error_type(*err));
}

TINYSTAN_PUBLIC void tinystan_free_stan_error_R(TinyStanError** err) {
  tinystan_free_stan_error(*err);
}
}
