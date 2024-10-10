#ifndef TORSTEN_INCLUDE
#define STAN_MATH_HPP

#include <stan/math/rev.hpp>

// removed in more recent stan_math, needed by torsten/meta/require_generics.hpp

/** \ingroup macro_helpers
 * Adds Unary require aliases.
 * @param check_type The name of the type to check, used to define
 * `require_<check_type>_t`.
 * @param checker A struct that returns holds a boolean `value`
 * @param doxygen_group The doxygen group to add this requires to.
 */
#define STAN_ADD_REQUIRE_UNARY(check_type, checker, doxygen_group)       \
  /*! \ingroup doxygen_group */                                          \
  /*! \defgroup check_type##_types check_type  */                        \
  /*! \addtogroup check_type##_types */                                  \
  /*! @{ */                                                              \
  /*! \brief Require type satisfies checker */                           \
  template <typename T>                                                  \
  using require_##check_type##_t = require_t<checker<std::decay_t<T>>>;  \
                                                                         \
  /*! \brief Require type does not satisfy checker */                    \
  template <typename T>                                                  \
  using require_not_##check_type##_t                                     \
      = require_not_t<checker<std::decay_t<T>>>;                         \
                                                                         \
  /*! \brief Require all of the types satisfy checker */                 \
  template <typename... Types>                                           \
  using require_all_##check_type##_t                                     \
      = require_all_t<checker<std::decay_t<Types>>...>;                  \
                                                                         \
  /*! \brief Require any of the types satisfy checker */                 \
  template <typename... Types>                                           \
  using require_any_##check_type##_t                                     \
      = require_any_t<checker<std::decay_t<Types>>...>;                  \
                                                                         \
  /*! \brief Require none of the types satisfy checker */                \
  template <typename... Types>                                           \
  using require_all_not_##check_type##_t                                 \
      = require_all_not_t<checker<std::decay_t<Types>>...>;              \
                                                                         \
  /*! \brief Require at least one of the types do not satisfy checker */ \
  template <typename... Types>                                           \
  using require_any_not_##check_type##_t                                 \
      = require_any_not_t<checker<std::decay_t<Types>>...>;              \
/*! @} */


// https://github.com/metrumresearchgroup/Torsten/blob/master/cmdstan/stan/lib/stan_math/stan/math.hpp#L21-L22
#include <stan/math/torsten/torsten.hpp>
using namespace torsten;
#endif
