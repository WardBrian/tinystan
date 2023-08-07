#ifndef FFISTAN_MODEL_HPP
#define FFISTAN_MODEL_HPP

#include <stan/model/model_base.hpp>
#include <stan/io/var_context.hpp>

#include <iostream>
#include <ostream>

#include "util.hpp"

// globals for Stan model output
std::streambuf *buf = nullptr;
std::ostream *outstream = &std::cout;

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

class FFIStanModel {
 public:
  FFIStanModel(const char *data, unsigned int seed) : seed(seed) {
    std::unique_ptr<stan::io::var_context> data_context = load_data(data);
    model = &new_model(*data_context, seed, outstream);
    std::vector<std::string> names;
    num_free_params = model->num_params_r();
    model->constrained_param_names(names, true, true);
    param_names = to_csv(names);
    num_params = names.size();
  }

  ~FFIStanModel() {
    delete model;
    free(param_names);
  }

  stan::model::model_base *model;
  unsigned int seed;
  char *param_names;
  size_t num_params;
  size_t num_free_params;
};

#endif
