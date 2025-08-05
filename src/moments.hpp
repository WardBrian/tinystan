#ifndef TINYSTAN_MOMENTS_HPP
#define TINYSTAN_MOMENTS_HPP

#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/math/prim/fun/welford_var_estimator.hpp>
#include <stan/callbacks/writer.hpp>

#include <vector>
#include <string>

namespace tinystan {
namespace io {

class moment_writer : public stan::callbacks::writer {
 public:
  moment_writer() : est(0), cached_sample(0) {};
  virtual ~moment_writer() {};

  /**
   * Primary method used by the Stan algorithms
   */
  void operator()(const std::vector<double> &v) override {
    cached_sample = Eigen::Map<const Eigen::VectorXd>(v.data(), v.size());
    est.add_sample(cached_sample);
  }

  /**
   * Used by Pathfinder which writes draws all at once
   */
  void operator()(const std::vector<std::string>& names) override {
    est = stan::math::welford_var_estimator{static_cast<int>(names.size())};
    cached_sample.resize(names.size());
  }

  Eigen::VectorXd mean() {
    Eigen::VectorXd mean;
    est.sample_mean(mean);
    return mean;
  }

  Eigen::VectorXd variance() {
    Eigen::VectorXd var;
    est.sample_variance(var);
    return var;
  }

  using stan::callbacks::writer::operator();

 private:
  stan::math::welford_var_estimator est;
  Eigen::VectorXd cached_sample;
};

}  // namespace io
}  // namespace tinystan

#endif
