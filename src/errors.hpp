#ifndef FFISTAN_ERRORS_HPP
#define FFISTAN_ERRORS_HPP

#include <stan/callbacks/logger.hpp>

#include <sstream>
#include <string>
#include <cstdlib>
#include <mutex>
#include <stdexcept>

class FFIStanError {
 public:
  FFIStanError(const char *msg,
               FFIStanErrorType type = FFIStanErrorType::generic)
      : msg(strdup(msg)), type(type) {}

  ~FFIStanError() { free(this->msg); }

  char *msg;
  FFIStanErrorType type;
};

/**
 * Macro for the repeated try-catch pattern used in all our wrapper functions.
 *
 * Note: __VA_ARGS__ is a special macro that expands to the arguments passed to
 * the macro. It is needed because commas can appear in the body of the code
 * passed to the macro, and the preprocessor would interpret them as argument
 * separators.
 */
#define FFISTAN_TRY_CATCH(...)                                     \
  try {                                                            \
    __VA_ARGS__                                                    \
  } catch (const ffistan::error::interrupt_exception &e) {         \
    if (err != nullptr) {                                          \
      *err = new FFIStanError("", FFIStanErrorType::interrupt);    \
    }                                                              \
  } catch (const std::invalid_argument &e) {                       \
    if (err != nullptr) {                                          \
      *err = new FFIStanError(e.what(), FFIStanErrorType::config); \
    }                                                              \
  } catch (const std::exception &e) {                              \
    if (err != nullptr) {                                          \
      *err = new FFIStanError(e.what());                           \
    }                                                              \
  } catch (...) {                                                  \
    if (err != nullptr) {                                          \
      *err = new FFIStanError("Unknown error");                    \
    }                                                              \
  }

namespace ffistan {
namespace error {

class error_logger : public stan::callbacks::logger {
 public:
  error_logger(){};
  ~error_logger(){};

  void error(const std::string &s) override {
    if (!s.empty()) {
      std::unique_lock<std::mutex> lock(error_mutex);
      last_error << s << "\n";
    }
  }

  void error(const std::stringstream &s) override {
    if (!s.str().empty()) {
      std::unique_lock<std::mutex> lock(error_mutex);
      last_error << s.str() << "\n";
    }
  }

  void fatal(const std::string &s) override {
    if (!s.empty()) {
      std::unique_lock<std::mutex> lock(error_mutex);
      last_error << s << "\n";
    }
  }

  void fatal(const std::stringstream &s) override {
    if (!s.str().empty()) {
      std::unique_lock<std::mutex> lock(error_mutex);
      last_error << s.str() << "\n";
    }
  }

  FFIStanError *get_error() {
    std::unique_lock<std::mutex> lock(error_mutex);
    auto err = last_error.str();
    if (err.empty())
      return new FFIStanError("Unknown error");
    err.pop_back();
    return new FFIStanError(err.c_str());
  }

 private:
  std::stringstream last_error;
  std::mutex error_mutex;
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
}  // namespace ffistan

#endif
