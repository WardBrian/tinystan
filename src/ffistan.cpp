#include <stan/callbacks/logger.hpp>
#include <stan/callbacks/interrupt.hpp>
#include <stan/callbacks/writer.hpp>
#include <stan/model/model_base.hpp>
#include <stan/services/util/create_rng.hpp>
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
// - multi-chain
// - other logging?
//   - question: can I get the metric out?
// - fixed param
//   - need to know if model is 0-param excluding tp, gq

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
                   const char *inits, unsigned int seed, unsigned int chain_id,
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

    // TODO (num_chains): allow multiple inits.
    // A char**?
    // Several files with the same naming structure, a-la cmdstan?
    // A char* with a separator?
    std::vector<std::unique_ptr<stan::io::var_context>> json_inits;
    json_inits.reserve(num_chains);
    for (size_t i = 0; i < num_chains; ++i) {
      json_inits.push_back(load_data(inits));
    }

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
    std::vector<stan::callbacks::writer> null_writers2(num_chains);

    int ec = 0;

    switch (metric_choice) {
      case unit:
        if (adapt) {
          ec = stan::services::sample::hmc_nuts_unit_e_adapt(
              model, num_chains, json_inits, seed, chain_id, init_radius,
              num_warmup, num_samples, /* no thinning */ 1, save_warmup,
              refresh, stepsize, stepsize_jitter, max_depth, delta, gamma,
              kappa, t0, interrupt, logger, null_writers, sample_writers,
              null_writers2);
        } else {
          ec = stan::services::sample::hmc_nuts_unit_e(
              model, num_chains, json_inits, seed, chain_id, init_radius,
              num_warmup, num_samples, /* no thinning */ 1, save_warmup,
              refresh, stepsize, stepsize_jitter, max_depth, interrupt, logger,
              null_writers, sample_writers, null_writers2);
        }
        break;
      case dense:
        if (adapt) {
          ec = stan::services::sample::hmc_nuts_dense_e_adapt(
              model, num_chains, json_inits, seed, chain_id, init_radius,
              num_warmup, num_samples, /* no thinning */ 1, save_warmup,
              refresh, stepsize, stepsize_jitter, max_depth, delta, gamma,
              kappa, t0, init_buffer, term_buffer, window, interrupt, logger,
              null_writers, sample_writers, null_writers2);
        } else {
          ec = stan::services::sample::hmc_nuts_dense_e(
              model, num_chains, json_inits, seed, chain_id, init_radius,
              num_warmup, num_samples, /* no thinning */ 1, save_warmup,
              refresh, stepsize, stepsize_jitter, max_depth, interrupt, logger,
              null_writers, sample_writers, null_writers2);
        }
        break;
      case diagonal:
        if (adapt) {
          ec = stan::services::sample::hmc_nuts_diag_e_adapt(
              model, num_chains, json_inits, seed, chain_id, init_radius,
              num_warmup, num_samples, /* no thinning */ 1, save_warmup,
              refresh, stepsize, stepsize_jitter, max_depth, delta, gamma,
              kappa, t0, init_buffer, term_buffer, window, interrupt, logger,
              null_writers, sample_writers, null_writers2);
        } else {
          ec = stan::services::sample::hmc_nuts_diag_e(
              model, num_chains, json_inits, seed, chain_id, init_radius,
              num_warmup, num_samples, /* no thinning */ 1, save_warmup,
              refresh, stepsize, stepsize_jitter, max_depth, interrupt, logger,
              null_writers, sample_writers, null_writers2);
        }
        break;
    }
    if (ec != 0) {
      if (err != nullptr) {
        *err = logger.get_error();
      }
    }

    return ec;

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
}
