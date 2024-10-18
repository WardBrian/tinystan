#include "tinystan.h"
#include <stdexcept>
#include <iostream>

#ifdef _WIN32
#include <windows.h>
#define RTLD_LAZY 0
static char *dlerror_value;

static char *get_win_error(void) {
  DWORD error_code = GetLastError();
  LPSTR buffer;

  FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM
                    | FORMAT_MESSAGE_IGNORE_INSERTS,
                NULL, error_code, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                (LPSTR)&buffer, 0, NULL);
  return buffer;
}

static void *dlopen(const char *name, int flags) {
  void *result = LoadLibrary(name);

  dlerror_value = get_win_error();

  return result;
}

static char *dlerror(void) {
  /* We need to reset the error */
  char *result = dlerror_value;
  dlerror_value = NULL;
  return result;
}

static void *dlsym(void *handle, const char *name) {
  void *result = (void *)GetProcAddress((HMODULE)handle, name);
  dlerror_value = result ? NULL : (char *)"function not found";
  return result;
}

#define dlclose(handle)

#else
#include <dlfcn.h>
#endif

#define load_symbol(name)                      \
  name = (typeof(name))(dlsym(handle, #name)); \
  if (!name) {                                 \
    throw std::runtime_error(dlerror());       \
  }

struct TinyStanModule {
  explicit TinyStanModule(const char *lib) {
    handle = dlopen(lib, RTLD_LAZY);
    if (!handle) {
      throw std::runtime_error(dlerror());
    }

    load_symbol(tinystan_api_version);
    load_symbol(tinystan_stan_version);
    load_symbol(tinystan_create_model);
    load_symbol(tinystan_destroy_model);

    load_symbol(tinystan_model_param_names);
    load_symbol(tinystan_model_num_free_params);
    load_symbol(tinystan_separator_char);

    load_symbol(tinystan_sample);
    load_symbol(tinystan_pathfinder);
    load_symbol(tinystan_optimize);
    load_symbol(tinystan_laplace_sample);

    load_symbol(tinystan_get_error_message);
    load_symbol(tinystan_get_error_type);
    load_symbol(tinystan_destroy_error);
  }

  ~TinyStanModule() {
    if (handle) {
      dlclose(handle);
    }
  }

 public:
  typeof(&::tinystan_api_version) tinystan_api_version;
  typeof(&::tinystan_stan_version) tinystan_stan_version;

  typeof(&::tinystan_create_model) tinystan_create_model;
  typeof(&::tinystan_destroy_model) tinystan_destroy_model;

  typeof(&::tinystan_model_param_names) tinystan_model_param_names;
  typeof(&::tinystan_model_num_free_params) tinystan_model_num_free_params;
  typeof(&::tinystan_separator_char) tinystan_separator_char;

  typeof(&::tinystan_sample) tinystan_sample;
  typeof(&::tinystan_pathfinder) tinystan_pathfinder;
  typeof(&::tinystan_optimize) tinystan_optimize;
  typeof(&::tinystan_laplace_sample) tinystan_laplace_sample;

  typeof(&::tinystan_get_error_message) tinystan_get_error_message;
  typeof(&::tinystan_get_error_type) tinystan_get_error_type;
  typeof(&::tinystan_get_error_type) tinystan_destroy_error;

 private:
  void *handle;
};

#undef load_symbol

void run(const char *lib) {
  TinyStanModule module(lib);

  int major, minor, patch;
  module.tinystan_api_version(&major, &minor, &patch);
  std::cout << "Using TinyStan API version: " << major << "." << minor << "."
            << patch << std::endl;
}

int main(int argc, char **argv) {
  char *lib;
  char *data;

  // require at least the library name
  if (argc > 2) {
    lib = argv[1];
    data = argv[2];
  } else if (argc > 1) {
    lib = argv[1];
    data = NULL;
  } else {
    std::cerr << "Usage: " << argv[0] << " <library> [data]" << std::endl;
    return 1;
  }
  try {
    run(lib);
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
