#ifndef TINYSTAN_FILE_HPP
#define TINYSTAN_FILE_HPP

#include <stan/io/ends_with.hpp>
#include <stan/io/json/json_data.hpp>
#include <stan/io/var_context.hpp>
#include <stan/io/empty_var_context.hpp>

#include <memory>
#include <stdexcept>
#include <vector>
#include <fstream>
#include <string>

namespace tinystan {
namespace io {

using var_ctx_ptr = std::unique_ptr<stan::io::var_context>;

var_ctx_ptr load_data(const char *data_char) {
  if (data_char == nullptr) {
    return var_ctx_ptr(new stan::io::empty_var_context());
  }
  std::string data(data_char);
  if (data.empty()) {
    return var_ctx_ptr(new stan::io::empty_var_context());
  }
  if (stan::io::ends_with(".json", data)) {
    std::ifstream data_stream(data);
    if (!data_stream.good()) {
      throw std::invalid_argument("Could not open data file " + data);
    }
    return var_ctx_ptr(new stan::json::json_data(data_stream));
  } else {
    std::istringstream json(data);
    return var_ctx_ptr(new stan::json::json_data(json));
  }
}

static constexpr const char SEPARATOR = '\x1C';  ///< ASCII file separator

std::vector<var_ctx_ptr> load_inits(int num_chains, const char *inits_char) {
  std::vector<var_ctx_ptr> json_inits;
  json_inits.reserve(num_chains);

  if (inits_char == nullptr
      || std::string(inits_char).find(SEPARATOR) == std::string::npos) {
    for (size_t i = 0; i < num_chains; ++i) {
      json_inits.push_back(load_data(inits_char));
    }
    return json_inits;
  }
  std::string inits(inits_char);
  std::vector<std::string> init_files;
  boost::algorithm::split(init_files, inits,
                          [](char c) { return c == SEPARATOR; });
  if (init_files.size() != num_chains) {
    throw std::invalid_argument(
        "Number of parameter initializations provided must be 0, 1, or match "
        "the number of chains");
  }
  for (auto &init_file : init_files) {
    json_inits.push_back(load_data(init_file.c_str()));
  }
  return json_inits;
}

}  // namespace io
}  // namespace tinystan
#endif
