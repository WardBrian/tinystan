#ifndef TINYSTAN_LOGGING_HPP
#define TINYSTAN_LOGGING_HPP

#include <stan/callbacks/logger.hpp>
#include <iostream>

namespace tinystan {
namespace io {

/*
 * NOTE(safety): We assume the user provides a thread-safe print callback.
 * This is true of e.g. Python's ctypes.CFUNCTYPE callbacks.
 * If necessary, we could lock a mutex in the else branch of info and warn.
 */

TINYSTAN_PRINT_CALLBACK user_print_callback = nullptr;

void info(const std::string& msg) {
  if (user_print_callback == nullptr) {
    std::cout << msg << std::endl;
  } else {
    user_print_callback(msg.c_str(), msg.size(), false);
  }
}

void warn(const std::string& msg) {
  if (user_print_callback == nullptr) {
    std::cerr << msg << std::endl;
  } else {
    user_print_callback(msg.c_str(), msg.size(), true);
  }
}

}  // namespace io
}  // namespace tinystan

#endif
