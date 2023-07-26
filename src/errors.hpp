#ifndef FFISTAN_ERRORS_HPP
#define FFISTAN_ERRORS_HPP

#include <stan/callbacks/logger.hpp>

#include <sstream>
#include <string>
#include <cstdlib>

class stan_error {
 public:
  stan_error(char *msg) : msg(msg) {}

  ~stan_error() { free(this->msg); }

  char *msg;
};

// TODO consider keeping more than just the last message
class error_logger : public stan::callbacks::logger {
 public:
  error_logger() : last_error("Unknown Error"){};
  ~error_logger(){};

  void error(const std::string &s) override {
    if (!s.empty())
      last_error = s;
  }

  void error(const std::stringstream &s) override {
    if (!s.str().empty())
      last_error = s.str();
  }

  void fatal(const std::string &s) override {
    if (!s.empty())
      last_error = s;
  }

  void fatal(const std::stringstream &s) override {
    if (!s.str().empty())
      last_error = s.str();
  }

  stan_error *get_error() { return new stan_error(strdup(last_error.c_str())); }

 private:
  std::string last_error;
};

#endif
