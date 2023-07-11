#ifndef FFISTAN_UTIL_HPP
#define FFISTAN_UTIL_HPP

#include <stan/callbacks/writer.hpp>
#include <stan/io/ends_with.hpp>
#include <stan/io/json/json_data.hpp>
#include <stan/io/var_context.hpp>
#include <stan/io/empty_var_context.hpp>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <memory>

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
  if (stan::io::ends_with(".json", data)) {
    std::ifstream data_stream(data);
    if (!data_stream.good()) {
      throw std::invalid_argument("Could not open data file " + data);
    }
    return std::unique_ptr<stan::io::var_context>(
        new stan::json::json_data(data_stream));
  } else {
    std::istringstream json(data);
    return std::unique_ptr<stan::io::var_context>(
        new stan::json::json_data(json));
  }
}

char *to_csv(const std::vector<std::string> &names) {
  std::stringstream ss;
  for (size_t i = 0; i < names.size(); ++i) {
    if (i > 0)
      ss << ',';
    ss << names[i];
  }
  std::string s = ss.str();
  const char *s_c = s.c_str();
  return strdup(s_c);
}

#endif
