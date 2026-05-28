#ifndef TINYSTAN_BUFFER_HPP
#define TINYSTAN_BUFFER_HPP

/**
 * \file buffer.hpp
 * \brief Contains adapters for using C-style buffers with various Stan classes.
 *
 * In many ways, this is the main contribution of TinyStan as a codebase. Given
 * these implementations of the Stan `writer`, `structured_writer`, and
 * `var_context` classes, most of the rest of the code is error handling and
 * calling the Stan services functions.
 */

#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/callbacks/writer.hpp>
#include <stan/callbacks/logger.hpp>
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
#include "model.hpp"

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
  buffer_writer(double *buf, size_t max) : buf(buf), pos(0), size(max) {};
  virtual ~buffer_writer() {};

  /**
   * Primary method used by the Stan algorithms
   */
  void operator()(const std::vector<double> &v) override {
    const auto v_size = v.size();
#ifndef TINYSTAN_NO_BOUNDS_CHECK
    if (pos + v_size > size) {
      throw std::runtime_error(
          "Buffer overflow writing vector. Please report a bug!");
    }
#endif
    std::memcpy(buf + pos, v.data(), sizeof(double) * v_size);
    pos += v_size;
  }

  /**
   * Used by Pathfinder which writes draws all at once
   */
  void operator()(const Eigen::MatrixXd &m) override {
#ifndef TINYSTAN_NO_BOUNDS_CHECK
    if (pos + m.size() > size) {
      throw std::runtime_error(
          "Buffer overflow writing eigen mat. Please report a bug!");
    }
#endif
    // copy into buffer
    Eigen::Map<Eigen::MatrixXd>(buf + pos, m.cols(), m.rows()) = m.transpose();
    pos += m.size();
  }

  void operator()(const Eigen::VectorXd &v) override {
#ifndef TINYSTAN_NO_BOUNDS_CHECK
    if (pos + v.size() > size) {
      throw std::runtime_error(
          "Buffer overflow writing eigen vec. Please report a bug!");
    }
#endif
    // copy into buffer
    Eigen::Map<Eigen::RowVectorXd>(buf + pos, v.rows()) = v.transpose();
    pos += v.size();
  }

  void operator()(const Eigen::RowVectorXd &v) override {
#ifndef TINYSTAN_NO_BOUNDS_CHECK
    if (pos + v.size() > size) {
      throw std::runtime_error(
          "Buffer overflow writing eigen row vec. Please report a bug!");
    }
#endif
    // copy into buffer
    Eigen::Map<Eigen::RowVectorXd>(buf + pos, v.cols()) = v;
    pos += v.size();
  }

  bool is_valid() const noexcept override { return buf != nullptr; }

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
  filtered_writer() : filters{} {};
  virtual ~filtered_writer() {};

  void add_key(const std::string &key_in, double *buf) {
    if (buf != nullptr) {
      filters.emplace_back(key_in, buf, 0);
    }
  }

  void write(const std::string &key_in, const Eigen::MatrixXd &mat) override {
    for (auto &[key, buf, pos] : filters) {
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
    for (auto &[key, buf, pos] : filters) {
      if (!pos && key_in == key) {
        for (int i = 0; i < vec.rows(); ++i) {
          buf[pos++] = vec(i);
        }
      }
    }
  }

  void write(const std::string &key_in, double value) override {
    for (auto &[key, buf, pos] : filters) {
      if (!pos && key_in == key) {
        buf[pos++] = value;
      }
    }
  }

  // other write methods are currently unused by anything we need
  using stan::callbacks::structured_writer::write;

 private:
  std::vector<std::tuple<std::string, double *, size_t>> filters;
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
      : buf(buf), size(size), dense(metric_choice == TinyStanMetric::dense) {};
  virtual ~inv_metric_buffer_reader() {};

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
      return std::make_unique<stan::io::array_var_context>(
          stan::services::util::create_unit_e_dense_inv_metric(num_params));

    case (TinyStanMetric::diagonal):
      return std::make_unique<stan::io::array_var_context>(
          stan::services::util::create_unit_e_diag_inv_metric(num_params));

    default:
      return std::make_unique<stan::io::empty_var_context>();
  }
}

/**
 * Returns a vector containing metric initializations for each chain.
 * If the supplied buffer is null, this uses the default in Stan (identity)
 */
inline std::vector<var_ctx_ptr> make_metric_inits(
    size_t num_chains, const double *buf, size_t num_params,
    TinyStanMetric metric_choice) {
  std::vector<var_ctx_ptr> metrics;
  metrics.reserve(num_chains);
  if (buf == nullptr) {
    for (size_t i = 0; i < num_chains; ++i) {
      metrics.push_back(default_metric(num_params, metric_choice));
    }
  } else {
    int metric_size = metric_choice == TinyStanMetric::dense
                          ? num_params * num_params
                          : num_params;
    for (size_t i = 0; i < num_chains; ++i) {
      metrics.push_back(std::make_unique<inv_metric_buffer_reader>(
          buf + (i * metric_size), metric_size, metric_choice));
    }
  }
  return metrics;
}

template <typename RNG>
class BufferHandler {
 public:
  BufferHandler(const TinyStanModel *model, stan::callbacks::logger &logger,
                RNG &rng, std::size_t i, double *out, double *stepsize_out,
                double *inv_metric_out, std::size_t num_warmup,
                std::size_t num_samples, bool save_warmup)
      : model_(model),
        logger_(logger),
        rng_(rng),
        i(i),
        save_warmup_(save_warmup),
        output_index_(model->num_params
                      * (num_samples + num_warmup * save_warmup) * i),
        out(out),
        stepsize_out(stepsize_out),
        inv_metric_out(inv_metric_out) {}

  void on_sample(const Eigen::VectorXd &position, double lp) {
    constrain(position);
  }

  void on_warmup(const Eigen::VectorXd &position, double lp, double step_size,
                 const Eigen::VectorXd &diag_inv_mass) {
    if (!save_warmup_) {
      return;
    }
    constrain(position);
  }

  void on_warmup_complete(double step_size, const Eigen::VectorXd &inv_metric) {
    if (stepsize_out != nullptr) {
      stepsize_out[i] = step_size;
    }
    if (inv_metric_out != nullptr) {
      std::copy(inv_metric.data(), inv_metric.data() + inv_metric.size(),
                inv_metric_out + i * inv_metric.size());
    }
  }

 private:
  void constrain(auto &&in) {
    std::stringstream msg;
    auto output
        = Eigen::Map<Eigen::VectorXd>(out + output_index_, model_->num_params);
    try {
      Eigen::VectorXd params;
      model_->model->write_array(rng_, const_cast<Eigen::VectorXd &>(in),
                                 params, true, true, &msg);
      output = params;
      if (!msg.str().empty()) {
        logger_.info(msg.str());
      }
    } catch (...) {
      if (!msg.str().empty()) {
        logger_.info(msg.str());
      }
      logger_.error("Error in constrain_draw: exception caught");
      output.array() = std::numeric_limits<double>::quiet_NaN();
    }
    output_index_ += model_->num_params;
  }

  const TinyStanModel *model_;
  stan::callbacks::logger &logger_;
  RNG &rng_;
  std::size_t i;
  bool save_warmup_;
  std::size_t output_index_;
  double *out, *stepsize_out, *inv_metric_out;
};

}  // namespace io
}  // namespace tinystan

#endif
