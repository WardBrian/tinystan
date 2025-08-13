#ifndef TINYSTAN_ERRORS_HPP
#define TINYSTAN_ERRORS_HPP

#include <stan/callbacks/logger.hpp>

#include <sstream>
#include <string>
#include <cstdlib>
#include <mutex>
#include <stdexcept>

#include "tinystan_types.h"
#include "model.hpp"

struct TinyStanError {
 public:
  TinyStanError(const char *msg,
                TinyStanErrorType type = TinyStanErrorType::generic)
      : msg(msg), type(type) {}

  std::string msg;
  TinyStanErrorType type;
};

namespace tinystan {
namespace error {

/**
 * Exception thrown when the user interrupts the program.
 * See tinystan::interrupt::tinystan_interrupt_handler for more details.
 */
class interrupt_exception {};

/**
 * Catches exceptions and stores them in a TinyStanError.
 *
 * This returns the result of the function if it succeeds.
 * If it fails, it returns -1 if the function returns an int,
 * nullptr if the function returns a pointer, and void otherwise.
 */
template <typename F>
inline auto catch_exceptions(TinyStanError **err, F f) {
  try {
    return f();
  } catch (const interrupt_exception &e) {
    if (err != nullptr) {
      *err = new TinyStanError("", TinyStanErrorType::interrupt);
    }
  } catch (const std::invalid_argument &e) {
    if (err != nullptr) {
      *err = new TinyStanError(e.what(), TinyStanErrorType::config);
    }
  } catch (const std::exception &e) {
    if (err != nullptr) {
      *err = new TinyStanError(e.what());
    }
  } catch (...) {
    if (err != nullptr) {
      *err = new TinyStanError("Unknown error");
    }
  }

  using Result = std::invoke_result_t<F>;
  if constexpr (std::is_same_v<Result, int>) {
    return -1;
  } else if constexpr (std::is_pointer_v<Result>) {
    return static_cast<Result>(nullptr);
  } else {
    static_assert(std::is_same_v<Result, void>, "Unexpected return type");
  }
}

/**
 * Logger which captures errors for later retrieval.
 * Optionally prints-non errors using tinystanmodel.info and
 * tinystanmodel.warn.
 */
class error_logger : public stan::callbacks::logger {
 public:
  error_logger(const TinyStanModel &model, bool print_non_errors)
      : model(model), print(print_non_errors) {};
  virtual ~error_logger() {};

  void info(const std::string &s) override {
    if (print && !s.empty()) {
      model.info(s);
    }
  }

  void info(const std::stringstream &s) override {
    if (print && !s.str().empty()) {
      model.info(s.str());
    }
  }

  void warn(const std::string &s) override {
    if (print && !s.empty()) {
      model.warn(s);
    }
  }

  void warn(const std::stringstream &s) override {
    if (print && !s.str().empty()) {
      model.warn(s.str());
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
  const TinyStanModel &model;
  bool print;
};

template <typename T>
inline void check_positive(const char *name, T val) {
  if (val <= 0) {
    std::stringstream msg;
    msg << name << " must be positive, was " << val;
    throw std::invalid_argument(msg.str());
  }
}

template <typename T>
inline void check_nonnegative(const char *name, T val) {
  if (val < 0) {
    std::stringstream msg;
    msg << name << " must be non-negative, was " << val;
    throw std::invalid_argument(msg.str());
  }
}

inline void check_between(const char *name, double val, double lb, double ub) {
  if (val < lb || val > ub) {
    std::stringstream msg;
    msg << name << " must be between " << lb << " and " << ub << ", was "
        << val;
    throw std::invalid_argument(msg.str());
  }
}

}  // namespace error
}  // namespace tinystan

#endif
