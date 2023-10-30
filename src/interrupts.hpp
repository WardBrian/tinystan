#ifndef FFISTAN_INTERRUPT_HPP
#define FFISTAN_INTERRUPT_HPP

#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/callbacks/interrupt.hpp>
#include <csignal>

namespace ffistan {
namespace interrupt {

volatile std::sig_atomic_t interrupted = false;

class ffistan_interrupt_handler : public stan::callbacks::interrupt {
// TODO: signal handling on Windows?
// https://learn.microsoft.com/en-us/windows/console/registering-a-control-handler-function
#if !defined _WIN32 && !defined __MINGW32__
 public:
  ffistan_interrupt_handler() {
    interrupted = false;

    memset(&custom, 0, sizeof(custom));
    sigemptyset(&custom.sa_mask);
    sigaddset(&custom.sa_mask, SIGINT);
    custom.sa_flags = SA_RESETHAND;
    custom.sa_handler = &ffistan_interrupt_handler::signal_handler;
    sigaction(SIGINT, &custom, &before);
  }

  virtual ~ffistan_interrupt_handler() { sigaction(SIGINT, &before, NULL); }

  void operator()() {
    if (interrupted) {
      throw std::runtime_error("Interrupted");
    }
  }

  static void signal_handler(int signal) { interrupted = true; }

 private:
  struct sigaction before;
  struct sigaction custom;
#endif
};

}  // namespace interrupt
}  // namespace ffistan
#endif
