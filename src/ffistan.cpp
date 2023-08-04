#include <stan/callbacks/logger.hpp>
#include <stan/callbacks/interrupt.hpp>
#include <stan/callbacks/writer.hpp>
#include <stan/callbacks/structured_writer.hpp>
#include <stan/model/model_base.hpp>
#include <stan/services/util/create_rng.hpp>
#include <stan/services/pathfinder/multi.hpp>
#include <stan/services/pathfinder/single.hpp>
#include <stan/services/sample/hmc_nuts_diag_e.hpp>
#include <stan/services/sample/hmc_nuts_diag_e_adapt.hpp>
#include <stan/services/sample/hmc_nuts_dense_e.hpp>
#include <stan/services/sample/hmc_nuts_dense_e_adapt.hpp>
#include <stan/services/sample/hmc_nuts_unit_e.hpp>
#include <stan/services/sample/hmc_nuts_unit_e_adapt.hpp>

#include <stan/math/prim/core/init_threadpool_tbb.hpp>

#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ffistan.h"
#include "errors.hpp"
#include "util.hpp"
#include "model.hpp"

// TODOs:
// - other logging?
//   - needs something like BridgeStan's print callback in general case
// - ability to output metric, hessian, etc?
//   - use diagnostic_writer, might need new service functions
// - fixed param
//   - need to know if model is 0-param excluding tp, gq
// - optimization 

extern "C" {

FFIStanModel *ffistan_create_model(const char *data, unsigned int seed,
                                   stan_error **err) {
  try {
    return new FFIStanModel(data, seed);
  } catch (const std::exception &e) {
    if (err != nullptr) {
      *err = new stan_error(strdup(e.what()));
    }
  } catch (...) {
    if (err != nullptr) {
      *err = new stan_error(strdup("Unknown error"));
    }
  }
  return nullptr;
}

void ffistan_destroy_model(FFIStanModel *model) { delete model; }

const char *ffistan_model_param_names(const FFIStanModel *model) {
  return model->param_names;
}

int ffistan_sample(const FFIStanModel *ffimodel, size_t num_chains,
                   const char *inits, unsigned int seed, unsigned int id,
                   double init_radius, int num_warmup, int num_samples,
                   FFIStanMetric metric_choice,
                   /* adaptation params */ bool adapt, double delta,
                   double gamma, double kappa, double t0,
                   unsigned int init_buffer, unsigned int term_buffer,
                   unsigned int window, bool save_warmup,
                   /* currently has no effect */ int refresh, double stepsize,
                   double stepsize_jitter, int max_depth, double *out,
                   stan_error **err) {
  try {
    // consider taking in as argument as well
    int num_threads = stan::math::internal::get_num_threads();
    stan::math::init_threadpool_tbb(num_threads);

    std::vector<std::unique_ptr<stan::io::var_context>> json_inits
        = load_inits(num_chains, inits);

    auto &model = *ffimodel->model;

    // all HMC has 7 algorithm params
    int num_params = ffimodel->num_params + 7;
    int offset = num_params * (num_samples + num_warmup * save_warmup);

    std::vector<buffer_writer> sample_writers;
    sample_writers.reserve(num_chains);
    for (size_t i = 0; i < num_chains; ++i) {
      sample_writers.emplace_back(out + offset * i);
    }

    error_logger logger;
    stan::callbacks::interrupt interrupt;

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
              logger, null_writers, sample_writers, null_writers);
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
              model, num_chains, json_inits, seed, id, init_radius, num_warmup,
              num_samples, thin, save_warmup, refresh, stepsize,
              stepsize_jitter, max_depth, delta, gamma, kappa, t0, init_buffer,
              term_buffer, window, interrupt, logger, null_writers,
              sample_writers, null_writers);
        } else {
          return_code = stan::services::sample::hmc_nuts_dense_e(
              model, num_chains, json_inits, seed, id, init_radius, num_warmup,
              num_samples, thin, save_warmup, refresh, stepsize,
              stepsize_jitter, max_depth, interrupt, logger, null_writers,
              sample_writers, null_writers);
        }
        break;
      case diagonal:
        if (adapt) {
          return_code = stan::services::sample::hmc_nuts_diag_e_adapt(
              model, num_chains, json_inits, seed, id, init_radius, num_warmup,
              num_samples, thin, save_warmup, refresh, stepsize,
              stepsize_jitter, max_depth, delta, gamma, kappa, t0, init_buffer,
              term_buffer, window, interrupt, logger, null_writers,
              sample_writers, null_writers);
        } else {
          return_code = stan::services::sample::hmc_nuts_diag_e(
              model, num_chains, json_inits, seed, id, init_radius, num_warmup,
              num_samples, thin, save_warmup, refresh, stepsize,
              stepsize_jitter, max_depth, interrupt, logger, null_writers,
              sample_writers, null_writers);
        }
        break;
    }
    if (return_code != 0) {
      if (err != nullptr) {
        *err = logger.get_error();
      }
    }

    return return_code;

  } catch (const std::exception &e) {
    if (err != nullptr) {
      *err = new stan_error(strdup(e.what()));
    }
  } catch (...) {
    if (err != nullptr) {
      *err = new stan_error(strdup("Unknown error"));
    }
  }
  return -1;
}

int ffistan_pathfinder(const FFIStanModel *ffimodel, size_t num_paths,
                       const char *inits, unsigned int seed, unsigned int id,
                       double init_radius, int num_draws,
                       /* tuning params */ int max_history_size,
                       double init_alpha, double tol_obj, double tol_rel_obj,
                       double tol_grad, double tol_rel_grad, double tol_param,
                       int num_iterations, int num_elbo_draws,
                       int num_multi_draws, int refresh, double *out,
                       stan_error **err) {
  try {
    int num_threads = stan::math::internal::get_num_threads();
    stan::math::init_threadpool_tbb(num_threads);

    std::vector<std::unique_ptr<stan::io::var_context>> json_inits
        = load_inits(num_paths, inits);

    auto &model = *ffimodel->model;

    buffer_writer pathfinder_writer(out);
    error_logger logger;

    stan::callbacks::interrupt interrupt;
    stan::callbacks::structured_writer dummy_json_writer;
    std::vector<stan::callbacks::writer> null_writers(num_paths);
    std::vector<stan::callbacks::structured_writer> null_structured_writers(
        num_paths);

    bool save_iterations = false;

    int return_code = 0;

    if (num_paths == 1) {
      return_code = stan::services::pathfinder::pathfinder_lbfgs_single(
          model, *(json_inits[0]), seed, id, init_radius, max_history_size,
          init_alpha, tol_obj, tol_rel_obj, tol_grad, tol_rel_grad, tol_param,
          num_iterations, num_elbo_draws, num_draws, save_iterations, refresh,
          interrupt, logger, null_writers[0], pathfinder_writer,
          null_structured_writers[0]);
    } else {
      return_code = stan::services::pathfinder::pathfinder_lbfgs_multi(
          model, json_inits, seed, id, init_radius, max_history_size,
          init_alpha, tol_obj, tol_rel_obj, tol_grad, tol_rel_grad, tol_param,
          num_iterations, num_elbo_draws, num_draws, num_multi_draws, num_paths,
          save_iterations, refresh, interrupt, logger, null_writers,
          null_writers, null_structured_writers, pathfinder_writer,
          dummy_json_writer);
    }

    if (return_code != 0) {
      if (err != nullptr) {
        *err = logger.get_error();
      }
    }

    return return_code;

  } catch (const std::exception &e) {
    if (err != nullptr) {
      *err = new stan_error(strdup(e.what()));
    }
  } catch (...) {
    if (err != nullptr) {
      *err = new stan_error(strdup("Unknown error"));
    }
  }
  return -1;
}

const char *ffistan_get_error_message(const stan_error *err) {
  return err->msg;
}

void ffistan_free_stan_error(stan_error *err) { delete (err); }

char ffistan_separator_char() { return SEPARATOR; }
}
