#ifndef TINYSTAN_INTERRUPT_HPP
#define TINYSTAN_INTERRUPT_HPP

#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/callbacks/interrupt.hpp>
#include <csignal>

#include "errors.hpp"

#if TINYSTAN_ON_WINDOWS
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace tinystan {
namespace interrupt {

volatile std::sig_atomic_t interrupted = false;

class tinystan_interrupt_handler : public stan::callbacks::interrupt {
#if !TINYSTAN_ON_WINDOWS  // POSIX signals
 public:
  tinystan_interrupt_handler() {
    interrupted = false;

    memset(&custom, 0, sizeof(custom));
    sigemptyset(&custom.sa_mask);
    sigaddset(&custom.sa_mask, SIGINT);
    custom.sa_flags = SA_RESETHAND;
    custom.sa_handler = &tinystan_interrupt_handler::signal_handler;
    sigaction(SIGINT, &custom, &before);
  }

  virtual ~tinystan_interrupt_handler() { sigaction(SIGINT, &before, NULL); }

  static void signal_handler(int signal) { interrupted = true; }

 private:
  struct sigaction before;
  struct sigaction custom;

#else  // Windows
 public:
  tinystan_interrupt_handler() {
    interrupted = false;

    SetConsoleCtrlHandler(tinystan_interrupt_handler::signal_handler, TRUE);
  }

  virtual ~tinystan_interrupt_handler() {
    SetConsoleCtrlHandler(tinystan_interrupt_handler::signal_handler, FALSE);
  }

  static BOOL WINAPI signal_handler(DWORD type) {
    switch (type) {
      case CTRL_C_EVENT:
      case CTRL_BREAK_EVENT:
        interrupted = true;
        return TRUE;
      default:
        return FALSE;
    }
  }
#endif

 public:
  void operator()() {
    if (interrupted) {
      throw tinystan::error::interrupt_exception();
    }
  }
};

}  // namespace interrupt
}  // namespace tinystan
#endif
