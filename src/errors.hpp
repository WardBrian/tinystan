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

  stan_error *get_error() {
    if (last_error.empty())
      return new stan_error(strdup("Unknown error"));
    last_error.pop_back();
    return new stan_error(strdup(last_error.c_str()));
  }

 private:
  std::string last_error;
};

#endif
