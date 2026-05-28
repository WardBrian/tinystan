#ifndef TINYSTAN_UTIL_HPP
#define TINYSTAN_UTIL_HPP

#include <stan/math/prim/core/init_threadpool_tbb.hpp>

#include <vector>
#include <sstream>
#include <string>
#include <stdexcept>
#include <thread>
#include <iomanip>

namespace tinystan {
namespace util {

/**
 * Initialize the threading pool with the specified number of threads.
 *
 * NOTE: Repeated calls to this function may not override the number of threads
 * previously set.
 */
inline void init_threading(int num_threads) {
#ifndef STAN_THREADS
  if (num_threads == -1) {
    num_threads = 1;
  }
  if (num_threads > 1) {
    throw std::invalid_argument(
        "Number of threads greater than 1 requested, but model not compiled "
        "with threading support.");
  }
#else
  if (num_threads == -1) {
    num_threads = std::max(1U, std::thread::hardware_concurrency());
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
inline std::string to_csv(const std::vector<std::string>& names) {
  std::stringstream ss;
  for (size_t i = 0; i < names.size(); ++i) {
    if (i > 0)
      ss << ',';
    ss << names[i];
  }
  return ss.str();
}

void run_progress(auto& iterator, int num, int finish, int start, int refresh,
                  int id, const char* name, auto& logger) {
  int it_print_width = std::ceil(std::log10(static_cast<double>(finish)));

  for (auto w = 0; w < num; ++w) {
    iterator();

    if (refresh > 0 && (w + 1 == finish || w == 0 || (w + 1) % refresh == 0)) {
      std::stringstream message;

      message << "Chain [" << id << "] ";
      message << "Iteration: ";
      message << std::setw(it_print_width) << w + 1 + start << " / " << finish;
      message << " [" << std::setw(3)
              << static_cast<int>((100.0 * (w + 1 + start)) / finish) << "%] ";
      message << " (" << name << ")";

      logger.info(message);
    }
  }
}

}  // namespace util
}  // namespace tinystan
#endif
