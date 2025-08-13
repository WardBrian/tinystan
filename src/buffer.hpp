#ifndef TINYSTAN_BUFFER_HPP
#define TINYSTAN_BUFFER_HPP

#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/callbacks/writer.hpp>
#include <stan/callbacks/structured_writer.hpp>

#include <stan/services/util/create_unit_e_dense_inv_metric.hpp>
#include <stan/services/util/create_unit_e_diag_inv_metric.hpp>
#include <stan/io/var_context.hpp>
#include <stan/io/array_var_context.hpp>
#include <stan/io/empty_var_context.hpp>

#include <vector>
#include <memory>
#include <stdexcept>
#include <string>

#include "tinystan_types.h"

namespace tinystan {
namespace io {

/**
 * @brief Writer for tabular data (e.g. draws)
 *
 * Adaptor for stan::callbacks::writer that writes to a C-style
 * buffer. It ignores all writes except for the primary ones used for draws.
 * Bounds checking is enabled by default, but can be disabled by defining
 * TINYSTAN_NO_BOUNDS_CHECK at compile time.
 */
class buffer_writer : public stan::callbacks::writer {
 public:
  buffer_writer(double *buf, size_t max) : buf(buf), pos(0), size(max){};
  virtual ~buffer_writer(){};

  /**
   * Primary method used by the Stan algorithms
   */
  void operator()(const std::vector<double> &v) override {
    const auto v_size = v.size();
#ifndef TINYSTAN_NO_BOUNDS_CHECK
    if (pos + v_size > size) {
      throw std::runtime_error("Buffer overflow. Please report a bug!");
    }
#endif
    std::memcpy(buf + pos, v.data(), sizeof(double) * v_size);
    pos += v_size;
  }

  /**
   * Used by Pathfinder which writes draws all at once
   */
  void operator()(const Eigen::Ref<Eigen::Matrix<double, -1, -1>> &m) override {
#ifndef TINYSTAN_NO_BOUNDS_CHECK
    if (pos + m.size() > size) {
      throw std::runtime_error("Buffer overflow. Please report a bug!");
    }
#endif
    // copy into buffer
    Eigen::Map<Eigen::MatrixXd>(buf + pos, m.cols(), m.rows()) = m.transpose();
    pos += m.size();
  }

  using stan::callbacks::writer::operator();

 private:
  double *buf;
  size_t pos;
  size_t size;
};

/**
 * @brief Writer for structured data (e.g. inv_metric) of a specific key
 *
 * Adaptor for stan::callbacks::structured_writer that writes to a C-style
 * buffer. It only writes the first key that matches to the buffer.
 */
class filtered_writer : public stan::callbacks::structured_writer {
 public:
  filtered_writer() : keys_buffers{} {};
  virtual ~filtered_writer(){};

  void add_buffer(const std::string &key_in, double *buf) {
    if (buf != nullptr) {
      keys_buffers.emplace_back(key_in, buf, 0);
    }
  }

  void write(const std::string &key_in, const Eigen::MatrixXd &mat) override {
    for (auto &[key, buf, pos] : keys_buffers) {
      if (!pos && key_in == key) {
        for (int j = 0; j < mat.cols(); ++j) {
          for (int i = 0; i < mat.rows(); ++i) {
            buf[pos++] = mat(i, j);
          }
        }
      }
    }
  }

  void write(const std::string &key_in, const Eigen::VectorXd &vec) override {
    for (auto &[key, buf, pos] : keys_buffers) {
      if (!pos && key_in == key) {
        for (int i = 0; i < vec.rows(); ++i) {
          buf[pos++] = vec(i);
        }
      }
    }
  }

  void write(const std::string &key_in, double value) override {
    for (auto &[key, buf, pos] : keys_buffers) {
      if (!pos && key_in == key) {
        buf[pos++] = value;
      }
    }
  }

  using stan::callbacks::structured_writer::write;

 private:
  std::vector<std::tuple<std::string, double *, size_t>> keys_buffers;
};

/**
 * @brief Data provider for metric initialization
 *
 * Adaptor for stan::io::var_context that reads from a C-style buffer.
 * This only supports reading the "inv_metric" key.
 */
class inv_metric_buffer_reader : public stan::io::empty_var_context {
 public:
  inv_metric_buffer_reader(const double *buf, size_t size,
                           TinyStanMetric metric_choice)
      : buf(buf), size(size), dense(metric_choice == TinyStanMetric::dense){};
  virtual ~inv_metric_buffer_reader(){};

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

inline var_ctx_ptr default_metric(size_t num_params,
                                  TinyStanMetric metric_choice) {
  switch (metric_choice) {
    case (TinyStanMetric::dense):
      return var_ctx_ptr(new stan::io::array_var_context(
          stan::services::util::create_unit_e_dense_inv_metric(num_params)));

    case (TinyStanMetric::diagonal):
      return var_ctx_ptr(new stan::io::array_var_context(
          stan::services::util::create_unit_e_diag_inv_metric(num_params)));

    default:
      return var_ctx_ptr(new stan::io::empty_var_context());
  }
}

inline std::vector<var_ctx_ptr> make_metric_inits(
    size_t num_chains, const double *buf, size_t num_params,
    TinyStanMetric metric_choice) {
  std::vector<var_ctx_ptr> metrics;
  metrics.reserve(num_chains);
  if (buf == nullptr) {
    for (size_t i = 0; i < num_chains; ++i) {
      metrics.emplace_back(default_metric(num_params, metric_choice));
    }
  } else {
    int metric_size = metric_choice == TinyStanMetric::dense
                          ? num_params * num_params
                          : num_params;
    for (size_t i = 0; i < num_chains; ++i) {
      metrics.emplace_back(var_ctx_ptr(new inv_metric_buffer_reader(
          buf + (i * metric_size), metric_size, metric_choice)));
    }
  }
  return metrics;
}

}  // namespace io
}  // namespace tinystan

#endif
