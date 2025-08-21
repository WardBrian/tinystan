#include "tinystan.h"

/**
 *  \file R_shims.cpp
 *  R's .C function can only pass in pointers to C types, so we need to
 * provide shims which indirect all our arguments and returns
 */
extern "C" {

/// see \link tinystan_create_model() \endlink for details
TINYSTAN_PUBLIC void tinystan_create_model_R(TinyStanModel** ptr_out,
                                             const char** data,
                                             unsigned int* seed,
                                             TinyStanError** err) {
  *ptr_out = tinystan_create_model(*data, *seed, nullptr, err);
}

/// see \link tinystan_destroy_model() \endlink for details
TINYSTAN_PUBLIC void tinystan_destroy_model_R(TinyStanModel** model) {
  tinystan_destroy_model(*model);
}

/// see \link tinystan_model_param_names() \endlink for details
TINYSTAN_PUBLIC void tinystan_model_param_names_R(TinyStanModel** model,
                                                  char const** names) {
  *names = tinystan_model_param_names(*model);
}

/// see \link tinystan_model_num_free_params() \endlink for details
TINYSTAN_PUBLIC void tinystan_model_num_free_params_R(TinyStanModel** model,
                                                      int* n) {
  *n = tinystan_model_num_free_params(*model);
}

/// see \link tinystan_model_num_constrained_params_for_unconstraining()
/// \endlink for details
TINYSTAN_PUBLIC void tinystan_model_num_constrained_params_for_unconstraining_R(
    TinyStanModel** model, int* n) {
  *n = tinystan_model_num_constrained_params_for_unconstraining(*model);
}

/// see \link tinystan_separator_char() \endlink for details
TINYSTAN_PUBLIC void tinystan_separator_char_R(char* sep) {
  *sep = tinystan_separator_char();
}

/// see \link tinystan_sample() \endlink for details
TINYSTAN_PUBLIC void tinystan_sample_R(
    int* return_code, TinyStanModel** model, unsigned int* num_chains,
    char** inits, unsigned int* seed, unsigned int* chain_id,
    double* init_radius, int* num_warmup, int* num_samples, int* metric_choice,
    int* metric_has_init, const double* init_inv_metric, int* adapt,
    double* delta, double* gamma, double* kappa, double* t0,
    unsigned int* init_buffer, unsigned int* term_buffer, unsigned int* window,
    int* save_warmup, double* stepsize, double* stepsize_jitter, int* max_depth,
    int* refresh, int* num_threads, double* out, int* out_size,
    int* save_stepsizes, double* stepsize_out, int* save_inv_metric,
    double* inv_metric_out, TinyStanError** err) {
  // difficult to directly pass a null pointer from R
  double* stepsize_out_ptr = nullptr;
  if (*save_stepsizes)
    stepsize_out_ptr = stepsize_out;

  double* inv_metric_out_ptr = nullptr;
  if (*save_inv_metric)
    inv_metric_out_ptr = inv_metric_out;

  const double* init_inv_metric_ptr = nullptr;
  if (*metric_has_init)
    init_inv_metric_ptr = init_inv_metric;

  *return_code = tinystan_sample(
      *model, *num_chains, *inits, *seed, *chain_id, *init_radius, *num_warmup,
      *num_samples, static_cast<TinyStanMetric>(*metric_choice),
      init_inv_metric_ptr, (*adapt != 0), *delta, *gamma, *kappa, *t0,
      *init_buffer, *term_buffer, *window, (*save_warmup != 0), *stepsize,
      *stepsize_jitter, *max_depth, *refresh, *num_threads, out,
      static_cast<size_t>(*out_size), stepsize_out_ptr, inv_metric_out_ptr,
      err);
}

/// see \link tinystan_pathfinder() \endlink for details
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

/// see \link tinystan_optimize() \endlink for details
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

/// see \link tinystan_laplace_sample() \endlink for details
TINYSTAN_PUBLIC
void tinystan_laplace_sample_R(int* return_code, const TinyStanModel** model,
                               int* use_array, const double* theta_hat_constr,
                               const char** theta_hat_json, unsigned int* seed,
                               int* num_draws, int* jacobian, int* calculate_lp,
                               int* refresh, int* num_threads, double* out,
                               int* out_size, int* save_hessian,
                               double* hessian_out, TinyStanError** err) {
  //  difficult to directly pass a null pointer from R
  double* hessian_out_ptr = nullptr;
  if (*save_hessian)
    hessian_out_ptr = hessian_out;

  const double* theta_hat_dbl_ptr = nullptr;
  const char* theta_hat_json_ptr = nullptr;
  if (*use_array) {
    theta_hat_dbl_ptr = theta_hat_constr;
  } else {
    theta_hat_json_ptr = *theta_hat_json;
  }

  *return_code = tinystan_laplace_sample(
      *model, theta_hat_dbl_ptr, theta_hat_json_ptr, *seed, *num_draws,
      (*jacobian != 0), (*calculate_lp != 0), *refresh, *num_threads, out,
      static_cast<size_t>(*out_size), hessian_out_ptr, err);
}

/// see \link tinystan_get_error_message() \endlink for details
TINYSTAN_PUBLIC void tinystan_get_error_message_R(TinyStanError** err,
                                                  char const** err_msg) {
  *err_msg = tinystan_get_error_message(*err);
}

/// see \link tinystan_get_error_type() \endlink for details
TINYSTAN_PUBLIC void tinystan_get_error_type_R(TinyStanError** err,
                                               int* err_type) {
  *err_type = static_cast<int>(tinystan_get_error_type(*err));
}

/// see \link tinystan_destroy_error() \endlink for details
TINYSTAN_PUBLIC void tinystan_destroy_error_R(TinyStanError** err) {
  tinystan_destroy_error(*err);
}
}
