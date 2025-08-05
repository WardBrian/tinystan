#ifndef TINYSTAN_MODEL_HPP
#define TINYSTAN_MODEL_HPP

#include <stan/model/model_base.hpp>
#include <stan/io/var_context.hpp>
#include <stan/callbacks/logger.hpp>

#include <ostream>
#include <memory>
#include <vector>

#include "tinystan_types.h"
#include "file.hpp"
#include "util.hpp"

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

/**
 * Holder for the instantiated Stan model and some extra metadata
 */
struct TinyStanModel {
 public:
  TinyStanModel(const char *data, unsigned int seed,
                TINYSTAN_PRINT_CALLBACK user_print_callback = nullptr)
      : model(&new_model(*tinystan::io::load_data(data), seed, &std::cout)),
        user_print_callback(user_print_callback),
        seed(seed),
        num_free_params(model->num_params_r()),
        param_names(),
        num_params(0) {
    std::vector<std::string> names;
    model->constrained_param_names(names, true, true);
    param_names = tinystan::util::to_csv(names);
    num_params = names.size();
  }

  /*
   * NOTE(safety): We assume the user provides a thread-safe print callback.
   * This is true of e.g. Python's ctypes.CFUNCTYPE callbacks.
   * If necessary, we could lock a mutex in the else branch of info and warn.
   */

  void info(const std::string &msg) const {
    if (user_print_callback == nullptr) {
      std::cout << msg << std::endl;
    } else {
      user_print_callback(msg.c_str(), msg.size(), false);
    }
  }

  void warn(const std::string &msg) const {
    if (user_print_callback == nullptr) {
      std::cerr << msg << std::endl;
    } else {
      user_print_callback(msg.c_str(), msg.size(), true);
    }
  }

  std::unique_ptr<stan::model::model_base> model;
  TINYSTAN_PRINT_CALLBACK user_print_callback;
  unsigned int seed;
  size_t num_free_params;
  std::string param_names;
  size_t num_params;
};

namespace tinystan {
namespace model {

/**
 * @brief Transform the constrained parameters to unconstrained space
 *
 * Can accept either a JSON string or a pointer to the constrained parameters,
 * only one of which should be provided at any given time.
 *
 * @param tmodel TinyStanModel instance
 * @param theta pointer to the constrained parameters
 * @param theta_json JSON string with the constrained parameters
 * @param logger logger instance
 */
inline Eigen::VectorXd unconstrain_parameters(const TinyStanModel &tmodel,
                                              const double *theta,
                                              const char *theta_json) {
  Eigen::VectorXd theta_unc(tmodel.num_free_params);
  auto &model = *tmodel.model;

  std::stringstream msg;
  try {
    if (theta_json != nullptr) {
      auto json_theta_hat = io::load_data(theta_json);
      model.transform_inits(*json_theta_hat, theta_unc, &msg);

    } else if (theta != nullptr) {
      Eigen::VectorXd theta_hat_constr_vec
          = Eigen::Map<const Eigen::VectorXd>(theta, tmodel.num_params);
      model.unconstrain_array(theta_hat_constr_vec, theta_unc, &msg);
    } else {
      throw std::runtime_error("No initial value provided");
    }
  } catch (...) {
    if (msg.str().length() > 0) {
      tmodel.info(msg.str());
    }
    throw;
  }
  if (msg.str().length() > 0) {
    tmodel.info(msg.str());
  }

  return theta_unc;
}

}  // namespace model
}  // namespace tinystan

#endif
