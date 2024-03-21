#ifndef TINYSTAN_ERRORS_HPP
#define TINYSTAN_ERRORS_HPP

#include <stan/callbacks/logger.hpp>

#include <sstream>
#include <string>
#include <cstdlib>
#include <mutex>
#include <stdexcept>

#include "logging.hpp"

class TinyStanError {
 public:
  TinyStanError(const char *msg,
                TinyStanErrorType type = TinyStanErrorType::generic)
      : msg(strdup(msg)), type(type) {}

  ~TinyStanError() { free(this->msg); }

  char *msg;
  TinyStanErrorType type;
};

/**
 * Macro for the repeated try-catch pattern used in all our wrapper functions.
 *
 * Note: __VA_ARGS__ is a special macro that expands to the arguments passed to
 * the macro. It is needed because commas can appear in the body of the code
 * passed to the macro, and the preprocessor would interpret them as argument
 * separators.
 */
#define TINYSTAN_TRY_CATCH(...)                                      \
  try {                                                              \
    __VA_ARGS__                                                      \
  } catch (const tinystan::error::interrupt_exception &e) {          \
    if (err != nullptr) {                                            \
      *err = new TinyStanError("", TinyStanErrorType::interrupt);    \
    }                                                                \
  } catch (const std::invalid_argument &e) {                         \
    if (err != nullptr) {                                            \
      *err = new TinyStanError(e.what(), TinyStanErrorType::config); \
    }                                                                \
  } catch (const std::exception &e) {                                \
    if (err != nullptr) {                                            \
      *err = new TinyStanError(e.what());                            \
    }                                                                \
  } catch (...) {                                                    \
    if (err != nullptr) {                                            \
      *err = new TinyStanError("Unknown error");                     \
    }                                                                \
  }

namespace tinystan {
namespace error {

class error_logger : public stan::callbacks::logger {
 public:
  error_logger(bool print_non_errors) : print(print_non_errors){};
  ~error_logger(){};

  void info(const std::string &s) override {
    if (print && !s.empty()) {
      io::info(s);
    }
  }

  void info(const std::stringstream &s) override {
    if (print && !s.str().empty()) {
      io::info(s.str());
    }
  }

  void warn(const std::string &s) override {
    if (print && !s.empty()) {
      io::warn(s);
    }
  }

  void warn(const std::stringstream &s) override {
    if (print && !s.str().empty()) {
      io::warn(s.str());
    }
  }

  void error(const std::string &s) override {
    if (!s.empty()) {
      std::lock_guard<std::mutex> lock(error_mutex);
      last_error << s << "\n";
    }
  }

  void error(const std::stringstream &s) override {
    if (!s.str().empty()) {
      std::lock_guard<std::mutex> lock(error_mutex);
      last_error << s.str() << "\n";
    }
  }

  void fatal(const std::string &s) override {
    if (!s.empty()) {
      std::lock_guard<std::mutex> lock(error_mutex);
      last_error << s << "\n";
    }
  }

  void fatal(const std::stringstream &s) override {
    if (!s.str().empty()) {
      std::lock_guard<std::mutex> lock(error_mutex);
      last_error << s.str() << "\n";
    }
  }

  TinyStanError *get_error() {
    std::lock_guard<std::mutex> lock(error_mutex);
    auto err = last_error.str();
    if (err.empty())
      return new TinyStanError("Unknown error");
    err.pop_back();
    return new TinyStanError(err.c_str());
  }

 private:
  std::stringstream last_error;
  std::mutex error_mutex;
  bool print;
};

template <typename T>
void check_positive(const char *name, T val) {
  if (val <= 0) {
    std::stringstream msg;
    msg << name << " must be positive, was " << val;
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

class interrupt_exception : public std::exception {};

}  // namespace error
}  // namespace tinystan

#endif
