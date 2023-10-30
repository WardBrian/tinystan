#ifndef FFISTAN_INTERRUPT_HPP
#define FFISTAN_INTERRUPT_HPP

#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/callbacks/interrupt.hpp>
#include <csignal>

#if defined _WIN32 || defined __MINGW32__
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include "errors.hpp"

namespace ffistan {
namespace interrupt {

volatile std::sig_atomic_t interrupted = false;

class ffistan_interrupt_handler : public stan::callbacks::interrupt {
#if !defined _WIN32 && !defined __MINGW32__  // POSIX signals
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

  static void signal_handler(int signal) { interrupted = true; }

 private:
  struct sigaction before;
  struct sigaction custom;

#else  // Windows
 public:
  ffistan_interrupt_handler() {
    interrupted = false;

    SetConsoleCtrlHandler(ffistan_interrupt_handler::signal_handler, TRUE);
  }

  virtual ~ffistan_interrupt_handler() {
    SetConsoleCtrlHandler(ffistan_interrupt_handler::signal_handler, FALSE);
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
      throw ffistan::error::interrupt_exception();
    }
  }
};

}  // namespace interrupt
}  // namespace ffistan
#endif
