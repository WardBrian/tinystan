#ifndef FFISTAN_BUFFER_HPP
#define FFISTAN_BUFFER_HPP

#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/callbacks/writer.hpp>
#include <stan/callbacks/structured_writer.hpp>

#include <stan/services/util/create_unit_e_dense_inv_metric.hpp>
#include <stan/services/util/create_unit_e_diag_inv_metric.hpp>
#include <stan/io/var_context.hpp>
#include <stan/io/empty_var_context.hpp>

#include <vector>
#include <memory>
#include <stdexcept>
#include <string>

namespace ffistan {
namespace io {

/*
 * Shim between the Stan callbacks that are used for outputting
 * and the simple buffer interface we provide. These do _not_ perform any bounds
 * checking, it is assumed the interface code properly sized the buffer.
 */

class buffer_writer : public stan::callbacks::writer {
 public:
  buffer_writer(double *buf) : buf(buf), pos(0){};
  virtual ~buffer_writer(){};

  // primary way of writing draws
  void operator()(const std::vector<double> &v) override {
    for (auto d : v) {
      buf[pos++] = d;
    }
  }

  // needed for pathfinder - transposed order per spec
  void operator()(const Eigen::Ref<Eigen::Matrix<double, -1, -1>> &m) override {
    // copy into buffer
    Eigen::MatrixXd mT = m.transpose();
    Eigen::Map<Eigen::MatrixXd>(buf + pos, mT.rows(), mT.cols()) = mT;
    pos += mT.size();
  }

  using stan::callbacks::writer::operator();

 private:
  double *buf;
  size_t pos;
};

class metric_buffer_writer : public stan::callbacks::structured_writer {
 public:
  metric_buffer_writer(double *buf) : buf(buf), pos(0){};
  virtual ~metric_buffer_writer(){};

  void write(const std::string &key, const Eigen::MatrixXd &mat) {
    if (!pos && buf != nullptr && key == "inv_metric") {
      for (int j = 0; j < mat.cols(); ++j) {
        for (int i = 0; i < mat.rows(); ++i) {
          buf[pos++] = mat(i, j);
        }
      }
    }
  }
  void write(const std::string &key, const Eigen::VectorXd &vec) {
    if (!pos && buf != nullptr && key == "inv_metric") {
      for (int i = 0; i < vec.rows(); ++i) {
        buf[pos++] = vec(i);
      }
    }
  }

  using stan::callbacks::structured_writer::write;

 private:
  double *buf;
  size_t pos;
};

class metric_buffer_reader : public stan::io::empty_var_context {
 public:
  metric_buffer_reader(const double *buf, size_t size,
                       FFIStanMetric metric_choice)
      : buf(buf), size(size), dense(metric_choice == FFIStanMetric::dense){};
  virtual ~metric_buffer_reader(){};

  bool contains_r(const std::string &name) const override {
    return name == "inv_metric";
  }

  std::vector<double> vals_r(const std::string &name) const override {
    if (name == "inv_metric") {
      return std::vector<double>(buf, buf + size);
    }
    throw std::runtime_error("Tried to read non-metric out of metric input");
  }

  void validate_dims(const std::string &stage, const std::string &name,
                     const std::string &base_type,
                     const std::vector<size_t> &dims_declared) const override {
    if (name == "inv_metric") {
      if (dense && dims_declared.size() == 2) {
        size_t d1 = dims_declared.at(0);
        size_t d2 = dims_declared.at(1);
        if (d1 == d2 && d1 * d2 == size) {
          return;
        }
      } else if (!dense && dims_declared.size() == 1
                 && dims_declared.at(0) == size) {
        return;
      }
      throw std::runtime_error("Invalid dimensions for metric");
    }
    throw std::runtime_error("Unknown variable name");
  }

 private:
  const double *buf;
  size_t size;
  bool dense;
};

using var_ctx_ptr = std::unique_ptr<stan::io::var_context>;

var_ctx_ptr default_metric(size_t num_params, FFIStanMetric metric_choice) {
  switch (metric_choice) {
    case (FFIStanMetric::dense):
      return var_ctx_ptr(new stan::io::array_var_context(
          stan::services::util::create_unit_e_dense_inv_metric(num_params)));

    case (FFIStanMetric::diagonal):
      return var_ctx_ptr(new stan::io::array_var_context(
          stan::services::util::create_unit_e_diag_inv_metric(num_params)));

    default:
      return var_ctx_ptr(new stan::io::empty_var_context());
  }
}

std::vector<var_ctx_ptr> make_metric_inits(size_t num_chains, const double *buf,
                                           size_t num_params,
                                           FFIStanMetric metric_choice) {
  std::vector<var_ctx_ptr> metrics;
  metrics.reserve(num_chains);
  if (buf == nullptr) {
    for (size_t i = 0; i < num_chains; ++i) {
      metrics.emplace_back(
          std::move(default_metric(num_params, metric_choice)));
    }
  } else {
    int metric_size = metric_choice == FFIStanMetric::dense
                          ? num_params * num_params
                          : num_params;
    for (size_t i = 0; i < num_chains; ++i) {
      metrics.emplace_back(std::move(var_ctx_ptr(new metric_buffer_reader(
          buf + (i * metric_size), metric_size, metric_choice))));
    }
  }
  return metrics;
}

}  // namespace io
}  // namespace ffistan

#endif
