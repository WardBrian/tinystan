#ifndef TINYSTAN_TYPES_H
#define TINYSTAN_TYPES_H

/// \file tinystan_types.h

#ifdef __cplusplus
#include <cstddef>
struct TinyStanError;
struct TinyStanModel;
#else
#include <stddef.h>
#include <stdbool.h>
/**
 * Opaque type for errors.
 *
 * Functions in TinyStan that can fail will accept a nullable `TinyStanError**`
 * argument as their last parameter.
 *
 * In addition to returning an integer status code (most functions), or NULL
 * (allocation functions), they can also set this argument to a new error object
 * if it itself was not NULL. This must be freed with tinystan_destroy_error()
 * when no longer needed. The error object contains a message describing the
 * error which can be retrieved with tinystan_get_error_message(), and and a
 * type field that can be used to distinguish between different kinds of errors
 * which can be retrieved with tinystan_get_error_type()
 */
typedef struct TinyStanError TinyStanError;
typedef struct TinyStanModel TinyStanModel;  ///< Opaque type for models
#endif

/**
 * Choice of metric for HMC.
 */
typedef enum { unit = 0, dense = 1, diagonal = 2 } TinyStanMetric;

/**
 * Choice of optimization algorithm.
 */
typedef enum { newton = 0, bfgs = 1, lbfgs = 2 } TinyStanOptimizationAlgorithm;

/**
 * An enum representing different kinds of errors TinyStan can generate.
 */
typedef enum {
  generic = 0,   ///< A generic runtime error from Stan.
  config = 1,    ///< An invalid configuration for the algorithm.
  interrupt = 2  ///< The user interrupted the algorithm with `Ctrl+C`.
} TinyStanErrorType;

/**
 * Callback used for printing.
 *
 * @param[in] msg The message to print.
 * @param[in] len The length of the message.
 * @param[in] bad Whether the message is an error message or not
 */
typedef void (*TINYSTAN_PRINT_CALLBACK)(const char *msg, size_t len, bool bad);

#endif
