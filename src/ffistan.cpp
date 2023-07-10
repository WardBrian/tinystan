#include <stan/callbacks/logger.hpp>
#include <stan/callbacks/interrupt.hpp>
#include <stan/callbacks/writer.hpp>
#include <stan/io/json/json_data.hpp>
#include <stan/io/var_context.hpp>
#include <stan/io/empty_var_context.hpp>
#include <stan/model/model_base.hpp>
#include <stan/services/util/create_rng.hpp>
#include <stan/services/sample/hmc_nuts_diag_e.hpp>
#include <stan/services/sample/hmc_nuts_diag_e_adapt.hpp>
#include <stan/services/sample/hmc_nuts_dense_e.hpp>
#include <stan/services/sample/hmc_nuts_dense_e_adapt.hpp>
#include <stan/services/sample/hmc_nuts_unit_e.hpp>
#include <stan/services/sample/hmc_nuts_unit_e_adapt.hpp>
#include <fstream>
#include <iostream>
#include <ostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// #include "err.h"
// #include "ffistan.h"

// globals for Stan model output
std::streambuf *buf = nullptr;
std::ostream *outstream = &std::cout;

// error handling

class stan_error {
 public:
  stan_error(char *msg) : msg(msg) {}

  ~stan_error() { free(this->msg); }

  char *msg;
};

extern "C" {
const char *ffistan_get_error_message(const stan_error *err) {
  return err->msg;
}

void ffistan_free_stan_error(stan_error *err) { delete (err); }
}

/**
 * Allocate and return a new model as a reference given the specified
 * data context, seed, and message stream.  This function is defined
 * in the generated model class.
 *
 * @param[in] data_context context for reading model data
 * @param[in] seed random seed for transformed data block
 * @param[in] msg_stream stream to which to send messages printed by the model
 */
stan::model::model_base &new_model(stan::io::var_context &data_context,
                                   unsigned int seed, std::ostream *msg_stream);

class buffer_writer : public stan::callbacks::writer {
 public:
  buffer_writer(double *buf) : buf(buf), pos(0){};
  ~buffer_writer(){};

  void operator()(const std::vector<double> &v) override {
    for (auto d : v) {
      buf[pos++] = d;
    }
  }

 private:
  double *buf;
  int pos;
};

std::unique_ptr<stan::io::var_context> load_data(const char *data_char) {
  if (data_char == nullptr) {
    return std::unique_ptr<stan::io::var_context>(
        new stan::io::empty_var_context());
  }
  std::string data(data_char);
  if (data.empty()) {
    return std::unique_ptr<stan::io::var_context>(
        new stan::io::empty_var_context());
  }
  std::ifstream data_stream(data);
  if (!data_stream.good()) {
    throw std::invalid_argument("Could not open data file " + data);
  }
  return std::unique_ptr<stan::io::var_context>(
      new stan::json::json_data(data_stream));
}

extern "C" {

enum Metric { unit = 0, dense = 1, diagonal = 2 };

// TODOs:
// - multi-chain (requires https://github.com/stan-dev/stan/issues/3204)
// - figure out size of `out` ahead of time
// - other logging?
//   - question: can I get the metric out?
int ffistan_sample(const char *data, const char *inits, unsigned int seed,
                   unsigned int chain_id, double init_radius, int num_warmup,
                   int num_samples, Metric metric_choice,
                   /* adaptation params */ bool adapt, double delta,
                   double gamma, double kappa, double t0,
                   unsigned int init_buffer, unsigned int term_buffer,
                   unsigned int window, bool save_warmup, int refresh,
                   double stepsize, double stepsize_jitter, int max_depth,
                   double *out, stan_error **err) {
  auto json_data = load_data(data);
  auto json_inits = load_data(inits);
  try {
    auto &model = new_model(*json_data, seed, outstream);
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
              stepsize_jitter, max_depth, delta, gamma, kappa, t0, interrupt, logger, null_writer,
              sample_writer, null_writer);
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
}
