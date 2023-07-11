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
// - multi-chain (requires https://github.com/stan-dev/stan/issues/3204)
// - other logging?
//   - question: can I get the metric out?

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


int ffistan_sample(const FFIStanModel *ffimodel, const char *inits,
                   unsigned int seed, unsigned int chain_id, double init_radius,
                   int num_warmup, int num_samples, FFIStanMetric metric_choice,
                   /* adaptation params */ bool adapt, double delta,
                   double gamma, double kappa, double t0,
                   unsigned int init_buffer, unsigned int term_buffer,
                   unsigned int window, bool save_warmup, int refresh,
                   double stepsize, double stepsize_jitter, int max_depth,
                   double *out, stan_error **err) {
  auto json_inits = load_data(inits);
  try {
    auto &model = *ffimodel->model;
    buffer_writer sample_writer(out);
    stan::callbacks::interrupt interrupt;
    stan::callbacks::logger logger;
    stan::callbacks::writer null_writer;

    switch (metric_choice) {
      case unit:
        if (adapt) {
          return stan::services::sample::hmc_nuts_unit_e_adapt(
              model, *json_inits, seed, chain_id, init_radius, num_warmup,
              num_samples, /* no thinning */ 1, save_warmup, refresh, stepsize,
              stepsize_jitter, max_depth, delta, gamma, kappa, t0, interrupt,
              logger, null_writer, sample_writer, null_writer);
        } else {
          return stan::services::sample::hmc_nuts_unit_e(
              model, *json_inits, seed, chain_id, init_radius, num_warmup,
              num_samples, /* no thinning */ 1, save_warmup, refresh, stepsize,
              stepsize_jitter, max_depth, interrupt, logger, null_writer,
              sample_writer, null_writer);
        }
        break;
      case dense:
        if (adapt) {
          return stan::services::sample::hmc_nuts_dense_e_adapt(
              model, *json_inits, seed, chain_id, init_radius, num_warmup,
              num_samples, /* no thinning */ 1, save_warmup, refresh, stepsize,
              stepsize_jitter, max_depth, delta, gamma, kappa, t0, init_buffer,
              term_buffer, window, interrupt, logger, null_writer,
              sample_writer, null_writer);
        } else {
          return stan::services::sample::hmc_nuts_dense_e(
              model, *json_inits, seed, chain_id, init_radius, num_warmup,
              num_samples, /* no thinning */ 1, save_warmup, refresh, stepsize,
              stepsize_jitter, max_depth, interrupt, logger, null_writer,
              sample_writer, null_writer);
        }
        break;
      case diagonal:
        if (adapt) {
          return stan::services::sample::hmc_nuts_diag_e_adapt(
              model, *json_inits, seed, chain_id, init_radius, num_warmup,
              num_samples, /* no thinning */ 1, save_warmup, refresh, stepsize,
              stepsize_jitter, max_depth, delta, gamma, kappa, t0, init_buffer,
              term_buffer, window, interrupt, logger, null_writer,
              sample_writer, null_writer);
        } else {
          return stan::services::sample::hmc_nuts_diag_e(
              model, *json_inits, seed, chain_id, init_radius, num_warmup,
              num_samples, /* no thinning */ 1, save_warmup, refresh, stepsize,
              stepsize_jitter, max_depth, interrupt, logger, null_writer,
              sample_writer, null_writer);
        }
        break;
    }

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
