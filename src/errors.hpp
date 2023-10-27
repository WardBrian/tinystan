#ifndef FFISTAN_ERRORS_HPP
#define FFISTAN_ERRORS_HPP

#include <stan/callbacks/logger.hpp>

#include <sstream>
#include <string>
#include <cstdlib>
#include <stdexcept>

class FFIStanError {
 public:
  FFIStanError(const char *msg) : msg(strdup(msg)) {}

  ~FFIStanError() { free(this->msg); }

  char *msg;
};

namespace ffistan {
namespace error {

class error_logger : public stan::callbacks::logger {
 public:
  error_logger(){};
  ~error_logger(){};

  void error(const std::string &s) override {
    if (!s.empty())
      last_error += s + "\n";
  }

  void error(const std::stringstream &s) override {
    if (!s.str().empty())
      last_error += s.str() + "\n";
  }

  void fatal(const std::string &s) override {
    if (!s.empty())
      last_error += s + "\n";
  }

  void fatal(const std::stringstream &s) override {
    if (!s.str().empty())
      last_error += s.str() + "\n";
  }

  FFIStanError *get_error() {
    if (last_error.empty())
      return new FFIStanError(strdup("Unknown error"));
    last_error.pop_back();
    return new FFIStanError(strdup(last_error.c_str()));
  }

 private:
  std::string last_error;
};

template <typename T>
void check_positive(const char *name, T val) {
  if (val <= 0) {
    std::stringstream msg;
    msg << name << " must be at positive, was " << val;
    throw std::invalid_argument(msg.str());
  }
}

template <typename T>
void check_nonnegative(const char *name, T val) {
  if (val < 0) {
    std::stringstream msg;
    msg << name << " must be non-negative, was " << val;
    throw std::invalid_argument(msg.str());
  }
}

void check_between(const char *name, double val, double lb, double ub) {
  if (val < lb || val > ub) {
    std::stringstream msg;
    msg << name << " must be between " << lb << " and " << ub << ", was "
        << val;
    throw std::invalid_argument(msg.str());
  }
}

}  // namespace error
}  // namespace ffistan

#endif
