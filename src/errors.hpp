#ifndef FFISTAN_ERRORS_HPP
#define FFISTAN_ERRORS_HPP

#include <cstdlib>

class stan_error {
 public:
  stan_error(char *msg) : msg(msg) {}

  ~stan_error() { free(this->msg); }

  char *msg;
};

#endif
