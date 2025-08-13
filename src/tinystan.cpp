#include <stan/callbacks/writer.hpp>
#include <stan/callbacks/structured_writer.hpp>
#include <stan/model/model_base.hpp>
#include <stan/services/pathfinder/multi.hpp>
#include <stan/services/pathfinder/single.hpp>
#include <stan/services/optimize/bfgs.hpp>
#include <stan/services/optimize/lbfgs.hpp>
#include <stan/services/optimize/newton.hpp>
#include <stan/services/optimize/laplace_sample.hpp>
#include <stan/services/sample/hmc_nuts_diag_e.hpp>
#include <stan/services/sample/hmc_nuts_diag_e_adapt.hpp>
#include <stan/services/sample/hmc_nuts_dense_e.hpp>
#include <stan/services/sample/hmc_nuts_dense_e_adapt.hpp>
#include <stan/services/sample/hmc_nuts_unit_e.hpp>
#include <stan/services/sample/hmc_nuts_unit_e_adapt.hpp>
#include <stan/version.hpp>

#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "tinystan.h"

#include "errors.hpp"
#include "file.hpp"
#include "buffer.hpp"
#include "interrupts.hpp"
#include "util.hpp"
#include "model.hpp"
#include "version.hpp"

#include "R_shims.cpp"

using namespace tinystan;

extern "C" {

TinyStanModel *tinystan_create_model(
    const char *data, unsigned int seed,
    TINYSTAN_PRINT_CALLBACK user_print_callback, TinyStanError **err) {
  return error::catch_exceptions(err, [&]() {
    return new TinyStanModel(data, seed, user_print_callback);
  });
}

void tinystan_destroy_model(TinyStanModel *model) { delete model; }

const char *tinystan_model_param_names(const TinyStanModel *model) {
  return model->param_names.c_str();
}

size_t tinystan_model_num_free_params(const TinyStanModel *model) {
  return model->num_free_params;
}

int tinystan_sample(const TinyStanModel *tmodel, size_t num_chains,
                    const char *inits, unsigned int seed, unsigned int id,
                    double init_radius, int num_warmup, int num_samples,
                    TinyStanMetric metric_choice, const double *init_inv_metric,
                    /* adaptation params */ bool adapt, double delta,
                    double gamma, double kappa, double t0,
                    unsigned int init_buffer, unsigned int term_buffer,
                    unsigned int window, bool save_warmup, double stepsize,
                    double stepsize_jitter, int max_depth, int refresh,
                    int num_threads, double *out, size_t out_size,
                    double *stepsize_out, double *inv_metric_out,
                    TinyStanError **err) {
  return error::catch_exceptions(err, [&]() {
    error::check_positive("num_chains", num_chains);
    error::check_positive("id", id);
    error::check_nonnegative("init_radius", init_radius);
    error::check_nonnegative("num_warmup", num_warmup);
    error::check_positive("num_samples", num_samples);
    if (adapt) {
      error::check_between("delta", delta, 0, 1);
      error::check_positive("gamma", gamma);
      error::check_positive("kappa", kappa);
      error::check_positive("t0", t0);
    }
    error::check_positive("stepsize", stepsize);
    error::check_between("stepsize_jitter", stepsize_jitter, 0, 1);
    error::check_positive("max_depth", max_depth);

    util::init_threading(num_threads);

    auto json_inits = io::load_inits(num_chains, inits);

    auto &model = *tmodel->model;

    // all HMC has 7 algorithm params
    int num_params = tmodel->num_params + 7;
    int draws_offset = num_params * (num_samples + num_warmup * save_warmup);
    if (out_size < num_chains * draws_offset) {
      std::stringstream ss;
      ss << "Output buffer too small. Expected at least " << num_chains
         << " chains of " << draws_offset << " doubles, got " << out_size;
      throw std::runtime_error(ss.str());
    }

    std::vector<io::buffer_writer> sample_writers;
    sample_writers.reserve(num_chains);
    for (size_t i = 0; i < num_chains; ++i) {
      sample_writers.emplace_back(out + draws_offset * i, draws_offset);
    }

    std::vector<io::filtered_writer> inv_metric_writers(num_chains);
    int num_model_params = tmodel->num_free_params;
    int metric_offset = metric_choice == dense
                            ? num_model_params * num_model_params
                            : num_model_params;
    for (size_t i = 0; i < num_chains; ++i) {
      if (inv_metric_out != nullptr) {
        inv_metric_writers[i].add_key("inv_metric",
                                      inv_metric_out + metric_offset * i);
      }
      if (stepsize_out != nullptr) {
        inv_metric_writers[i].add_key("stepsize", stepsize_out + i);
      }
    }

    auto initial_metrics = io::make_metric_inits(
        num_chains, init_inv_metric, num_model_params, metric_choice);

    error::error_logger logger(*tmodel, refresh != 0);
    interrupt::tinystan_interrupt_handler interrupt;

    std::vector<stan::callbacks::writer> null_writers(num_chains);

    int return_code = 0;

    int thin = 1;  // no thinning

    switch (metric_choice) {
      case unit:
        if (adapt) {
          return_code = stan::services::sample::hmc_nuts_unit_e_adapt(
              model, num_chains, json_inits, seed, id, init_radius, num_warmup,
              num_samples, thin, save_warmup, refresh, stepsize,
              stepsize_jitter, max_depth, delta, gamma, kappa, t0, interrupt,
              logger, null_writers, sample_writers, null_writers,
              inv_metric_writers);
        } else {
          return_code = stan::services::sample::hmc_nuts_unit_e(
              model, num_chains, json_inits, seed, id, init_radius, num_warmup,
              num_samples, thin, save_warmup, refresh, stepsize,
              stepsize_jitter, max_depth, interrupt, logger, null_writers,
              sample_writers, null_writers);
        }
        break;
      case dense:
        if (adapt) {
          return_code = stan::services::sample::hmc_nuts_dense_e_adapt(
              model, num_chains, json_inits, initial_metrics, seed, id,
              init_radius, num_warmup, num_samples, thin, save_warmup, refresh,
              stepsize, stepsize_jitter, max_depth, delta, gamma, kappa, t0,
              init_buffer, term_buffer, window, interrupt, logger, null_writers,
              sample_writers, null_writers, inv_metric_writers);
        } else {
          return_code = stan::services::sample::hmc_nuts_dense_e(
              model, num_chains, json_inits, initial_metrics, seed, id,
              init_radius, num_warmup, num_samples, thin, save_warmup, refresh,
              stepsize, stepsize_jitter, max_depth, interrupt, logger,
              null_writers, sample_writers, null_writers);
        }
        break;
      case diagonal:
        if (adapt) {
          return_code = stan::services::sample::hmc_nuts_diag_e_adapt(
              model, num_chains, json_inits, initial_metrics, seed, id,
              init_radius, num_warmup, num_samples, thin, save_warmup, refresh,
              stepsize, stepsize_jitter, max_depth, delta, gamma, kappa, t0,
              init_buffer, term_buffer, window, interrupt, logger, null_writers,
              sample_writers, null_writers, inv_metric_writers);
        } else {
          return_code = stan::services::sample::hmc_nuts_diag_e(
              model, num_chains, json_inits, initial_metrics, seed, id,
              init_radius, num_warmup, num_samples, thin, save_warmup, refresh,
              stepsize, stepsize_jitter, max_depth, interrupt, logger,
              null_writers, sample_writers, null_writers);
        }
        break;
    }
    if (return_code != 0) {
      if (err != nullptr) {
        *err = logger.get_error();
      }
    }

    return return_code;
  });
}

int tinystan_pathfinder(const TinyStanModel *tmodel, size_t num_paths,
                        const char *inits, unsigned int seed, unsigned int id,
                        double init_radius, int num_draws,
                        /* tuning params */ int max_history_size,
                        double init_alpha, double tol_obj, double tol_rel_obj,
                        double tol_grad, double tol_rel_grad, double tol_param,
                        int num_iterations, int num_elbo_draws,
                        int num_multi_draws, bool calculate_lp,
                        bool psis_resample, int refresh, int num_threads,
                        double *out, size_t out_size, TinyStanError **err) {
  return error::catch_exceptions(err, [&]() {
    error::check_positive("num_paths", num_paths);
    error::check_positive("num_draws", num_draws);
    error::check_positive("id", id);
    error::check_nonnegative("init_radius", init_radius);
    error::check_positive("max_history_size", max_history_size);
    error::check_positive("init_alpha", init_alpha);
    error::check_positive("tol_obj", tol_obj);
    error::check_positive("tol_rel_obj", tol_rel_obj);
    error::check_positive("tol_grad", tol_grad);
    error::check_positive("tol_rel_grad", tol_rel_grad);
    error::check_positive("tol_param", tol_param);
    error::check_positive("num_iterations", num_iterations);
    error::check_positive("num_elbo_draws", num_elbo_draws);
    error::check_positive("num_multi_draws", num_multi_draws);

    util::init_threading(num_threads);

    auto json_inits = io::load_inits(num_paths, inits);

    auto &model = *tmodel->model;

    io::buffer_writer pathfinder_writer(out, out_size);
    error::error_logger logger(*tmodel, refresh != 0);

    interrupt::tinystan_interrupt_handler interrupt;
    stan::callbacks::structured_writer dummy_json_writer;

    bool save_iterations = false;

    int return_code = 0;

    if (num_paths == 1 && psis_resample == false) {
      stan::callbacks::writer null_writer;
      return_code = stan::services::pathfinder::pathfinder_lbfgs_single(
          model, *(json_inits[0]), seed, id, init_radius, max_history_size,
          init_alpha, tol_obj, tol_rel_obj, tol_grad, tol_rel_grad, tol_param,
          num_iterations, num_elbo_draws, num_draws, save_iterations, refresh,
          interrupt, logger, null_writer, pathfinder_writer, dummy_json_writer,
          calculate_lp);
    } else {
      std::vector<stan::callbacks::writer> null_writers(num_paths);
      std::vector<stan::callbacks::structured_writer> null_structured_writers(
          num_paths);
      return_code = stan::services::pathfinder::pathfinder_lbfgs_multi(
          model, json_inits, seed, id, init_radius, max_history_size,
          init_alpha, tol_obj, tol_rel_obj, tol_grad, tol_rel_grad, tol_param,
          num_iterations, num_elbo_draws, num_draws, num_multi_draws, num_paths,
          save_iterations, refresh, interrupt, logger, null_writers,
          null_writers, null_structured_writers, pathfinder_writer,
          dummy_json_writer, calculate_lp, psis_resample);
    }

    if (return_code != 0) {
      if (err != nullptr) {
        *err = logger.get_error();
      }
    }

    return return_code;
  });
}

int tinystan_optimize(const TinyStanModel *tmodel, const char *init,
                      unsigned int seed, unsigned int id, double init_radius,
                      TinyStanOptimizationAlgorithm algorithm,
                      int num_iterations, bool jacobian,
                      /* tuning params */ int max_history_size,
                      double init_alpha, double tol_obj, double tol_rel_obj,
                      double tol_grad, double tol_rel_grad, double tol_param,
                      int refresh, int num_threads, double *out,
                      size_t out_size, TinyStanError **err) {
  return error::catch_exceptions(err, [&]() {
    error::check_positive("id", id);
    error::check_positive("num_iterations", num_iterations);
    error::check_nonnegative("init_radius", init_radius);

    if (algorithm == lbfgs) {
      error::check_positive("max_history_size", max_history_size);
    }
    if (algorithm == bfgs || algorithm == lbfgs) {
      error::check_positive("init_alpha", init_alpha);
      error::check_positive("tol_obj", tol_obj);
      error::check_positive("tol_rel_obj", tol_rel_obj);
      error::check_positive("tol_grad", tol_grad);
      error::check_positive("tol_rel_grad", tol_rel_grad);
      error::check_positive("tol_param", tol_param);
    }

    util::init_threading(num_threads);

    auto json_init = io::load_data(init);
    auto &model = *tmodel->model;
    io::buffer_writer sample_writer(out, out_size);
    error::error_logger logger(*tmodel, refresh != 0);

    interrupt::tinystan_interrupt_handler interrupt;
    stan::callbacks::writer null_writer;

    bool save_iterations = false;

    int return_code = 0;
    switch (algorithm) {
      case newton:
        if (jacobian)
          return_code
              = stan::services::optimize::newton<stan::model::model_base, true>(
                  model, *json_init, seed, id, init_radius, num_iterations,
                  save_iterations, interrupt, logger, null_writer,
                  sample_writer);
        else
          return_code
              = stan::services::optimize::newton<stan::model::model_base,
                                                 false>(
                  model, *json_init, seed, id, init_radius, num_iterations,
                  save_iterations, interrupt, logger, null_writer,
                  sample_writer);
        break;
      case bfgs:
        if (jacobian)
          return_code
              = stan::services::optimize::bfgs<stan::model::model_base, true>(
                  model, *json_init, seed, id, init_radius, init_alpha, tol_obj,
                  tol_rel_obj, tol_grad, tol_rel_grad, tol_param,
                  num_iterations, save_iterations, refresh, interrupt, logger,
                  null_writer, sample_writer);
        else
          return_code
              = stan::services::optimize::bfgs<stan::model::model_base, false>(
                  model, *json_init, seed, id, init_radius, init_alpha, tol_obj,
                  tol_rel_obj, tol_grad, tol_rel_grad, tol_param,
                  num_iterations, save_iterations, refresh, interrupt, logger,
                  null_writer, sample_writer);
        break;
      case lbfgs:
        if (jacobian)
          return_code
              = stan::services::optimize::lbfgs<stan::model::model_base, true>(
                  model, *json_init, seed, id, init_radius, max_history_size,
                  init_alpha, tol_obj, tol_rel_obj, tol_grad, tol_rel_grad,
                  tol_param, num_iterations, save_iterations, refresh,
                  interrupt, logger, null_writer, sample_writer);
        else
          return_code
              = stan::services::optimize::lbfgs<stan::model::model_base, false>(
                  model, *json_init, seed, id, init_radius, max_history_size,
                  init_alpha, tol_obj, tol_rel_obj, tol_grad, tol_rel_grad,
                  tol_param, num_iterations, save_iterations, refresh,
                  interrupt, logger, null_writer, sample_writer);
        break;
    }

    if (return_code != 0) {
      if (err != nullptr) {
        *err = logger.get_error();
      }
    }

    return return_code;
  });
}

int tinystan_laplace_sample(const TinyStanModel *tmodel,
                            const double *theta_hat_constr,
                            const char *theta_hat_json, unsigned int seed,
                            int num_draws, bool jacobian, bool calculate_lp,
                            int refresh, int num_threads, double *out,
                            size_t out_size, double *hessian_out,
                            TinyStanError **err) {
  return error::catch_exceptions(err, [&]() {
    error::check_positive("num_draws", num_draws);

    util::init_threading(num_threads);

    auto &model = *tmodel->model;
    io::buffer_writer sample_writer(out, out_size);
    io::filtered_writer hessian_writer;
    hessian_writer.add_key("Hessian", hessian_out);
    error::error_logger logger(*tmodel, refresh != 0);
    interrupt::tinystan_interrupt_handler interrupt;

    Eigen::VectorXd theta_hat = model::unconstrain_parameters(
        *tmodel, theta_hat_constr, theta_hat_json);

    int return_code;
    if (jacobian) {
      return_code = stan::services::laplace_sample<true>(
          model, theta_hat, num_draws, calculate_lp, seed, refresh, interrupt,
          logger, sample_writer, hessian_writer);
    } else {
      return_code = stan::services::laplace_sample<false>(
          model, theta_hat, num_draws, calculate_lp, seed, refresh, interrupt,
          logger, sample_writer, hessian_writer);
    }

    if (return_code != 0) {
      if (err != nullptr) {
        *err = logger.get_error();
      }
    }
    return return_code;
  });
}

const char *tinystan_get_error_message(const TinyStanError *err) {
  if (err == nullptr) {
    return "Something went wrong: No error found";
  }
  return err->msg.c_str();
}

TinyStanErrorType tinystan_get_error_type(const TinyStanError *err) {
  if (err == nullptr) {
    return TinyStanErrorType::generic;
  }
  return err->type;
}

void tinystan_destroy_error(TinyStanError *err) { delete (err); }

char tinystan_separator_char() { return io::SEPARATOR; }

void tinystan_api_version(int *major, int *minor, int *patch) {
  *major = TINYSTAN_MAJOR;
  *minor = TINYSTAN_MINOR;
  *patch = TINYSTAN_PATCH;
}

void tinystan_stan_version(int *major, int *minor, int *patch) {
  *major = STAN_MAJOR;
  *minor = STAN_MINOR;
  *patch = STAN_PATCH;
}

}  // extern "C"
