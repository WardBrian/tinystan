#ifndef TINYSTAN_UTIL_HPP
#define TINYSTAN_UTIL_HPP

#include <stan/math/prim/core/init_threadpool_tbb.hpp>

#include <vector>
#include <sstream>
#include <string>
#include <memory>
#include <stdexcept>
#include <thread>

namespace tinystan {
namespace util {

/**
 * Initialize the threading pool with the specified number of threads.
 *
 * NOTE: Repeated calls to this function may not override the number of threads
 * previously set.
 */
void init_threading(int num_threads) {
  if (num_threads == -1) {
    num_threads = std::thread::hardware_concurrency();
  }
#ifndef STAN_THREADS
  if (num_threads > 1) {
    throw std::invalid_argument(
        "Number of threads greater than 1 requested, but model not compiled "
        "with threading support.");
  }
#endif
  if (num_threads > 0) {
    stan::math::init_threadpool_tbb(num_threads);
  } else {
    throw std::invalid_argument(
        "Number of threads requested must be a positive integer or -1"
        " (for all available cores).");
  }
}

/**
 * Convert a vector of strings to a single comma-separated string.
 *
 * @param names vector of strings to convert
 * @return freshly allocated comma-separated string
 */
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
}  // namespace util
}  // namespace tinystan
#endif
