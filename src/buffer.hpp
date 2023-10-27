#ifndef FFISTAN_BUFFER_HPP
#define FFISTAN_BUFFER_HPP

#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/callbacks/writer.hpp>
#include <stan/callbacks/structured_writer.hpp>
#include <vector>
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
  void operator()(const Eigen::MatrixXd &m) override {
    for (int j = 0; j < m.cols(); ++j) {
      for (int i = 0; i < m.rows(); ++i) {
        buf[pos++] = m(i, j);
      }
    }
  }

  using stan::callbacks::writer::operator();

 private:
  double *buf;
  unsigned long int pos;
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
  int pos;
};
}  // namespace io
}  // namespace ffistan

#endif
