#ifndef TINYSTAN_MODEL_HPP
#define TINYSTAN_MODEL_HPP

#include <stan/model/model_base.hpp>
#include <stan/io/var_context.hpp>
#include <stan/callbacks/logger.hpp>

#include <ostream>
#include <memory>
#include <vector>

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
class TinyStanModel {
 public:
  TinyStanModel(const char *data, unsigned int seed)
      : model(&new_model(*tinystan::io::load_data(data), seed, &std::cout)),
        seed(seed),
        num_free_params(model->num_params_r()),
        param_names(nullptr),
        num_params(0),
        num_req_constrained_params(0) {
    std::vector<std::string> names;
    model->constrained_param_names(names, true, true);
    param_names = tinystan::util::to_csv(names);
    num_params = names.size();

    names.clear();
    model->constrained_param_names(names, false, false);
    num_req_constrained_params = names.size();
  }

  ~TinyStanModel() {
    delete model;
    free(param_names);
  }

  TinyStanModel(const TinyStanModel &) = delete;
  TinyStanModel(TinyStanModel &&) = delete;
  TinyStanModel operator=(const TinyStanModel &) = delete;
  TinyStanModel operator=(TinyStanModel &&) = delete;

  stan::model::model_base *model;
  unsigned int seed;
  size_t num_free_params;
  char *param_names;
  size_t num_params;
  size_t num_req_constrained_params;
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
Eigen::VectorXd unconstrain_parameters(const TinyStanModel *tmodel,
                                       const double *theta,
                                       const char *theta_json,
                                       stan::callbacks::logger &logger) {
  Eigen::VectorXd theta_unc(tmodel->num_free_params);
  auto &model = *tmodel->model;

  std::stringstream msg;
  try {
    if (theta_json != nullptr) {
      auto json_theta_hat = io::load_data(theta_json);
      model.transform_inits(*json_theta_hat, theta_unc, &msg);

    } else if (theta != nullptr) {
      Eigen::VectorXd theta_hat_constr_vec
          = Eigen::Map<const Eigen::VectorXd>(theta, tmodel->num_params);
      model.unconstrain_array(theta_hat_constr_vec, theta_unc, &msg);
    } else {
      throw std::runtime_error("No initial value provided");
    }
  } catch (...) {
    if (msg.str().length() > 0) {
      logger.info(msg.str());
    }
    throw;
  }
  if (msg.str().length() > 0) {
    logger.info(msg.str());
  }

  return theta_unc;
}

}  // namespace model
}  // namespace tinystan

#endif
