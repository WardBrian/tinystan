cmake_minimum_required(VERSION 3.20.2)
project(
    ffistan
    VERSION 0.0.1
    LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)
# set(CMAKE_VERBOSE_MAKEFILE YES)
include(FetchContent)
# Externally provided libraries

FetchContent_Declare(
  tbb
  GIT_REPOSITORY https://github.com/oneapi-src/oneTBB
  GIT_TAG        v2021.7.0  # adjust this to the version you need
)
set(TBB_TEST OFF)

FetchContent_MakeAvailable(tbb)

FetchContent_Declare(
  Eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0)
set(EIGEN_BUILD_DOC OFF)
# note: To disable eigen tests,
# you should put this code in a add_subdirectory to avoid to change
# BUILD_TESTING for your own project too since variables are directory
# scoped
set(BUILD_TESTING OFF)
set(EIGEN_BUILD_PKGCONFIG OFF)
FetchContent_MakeAvailable(Eigen)

FetchContent_Declare(
  sundials
  GIT_REPOSITORY https://github.com/LLNL/sundials
  GIT_TAG        v6.5.1
    # adjust this to the version you need
)

FetchContent_GetProperties(sundials)
if(NOT sundials_POPULATED)
  FetchContent_Populate(sundials)
  add_subdirectory(${sundials_SOURCE_DIR} ${sundials_BINARY_DIR})
endif()

set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_STATIC_RUNTIME ON)
set(BOOST_ENABLE_CMAKE ON)
FetchContent_Declare(
  boost
  URL https://boostorg.jfrog.io/artifactory/main/release/1.78.0/source/boost_1_78_0.tar.gz
)

FetchContent_GetProperties(boost)
if(NOT boost_POPULATED)
  FetchContent_Populate(boost)
  set(boost_INCLUDE_DIR ${boost_SOURCE_DIR})
endif()

FetchContent_Declare(rapidjson
    GIT_REPOSITORY https://github.com/Tencent/rapidjson.git
    GIT_TAG        master)
FetchContent_GetProperties(rapidjson)
if(NOT rapidjson_POPULATED)
  FetchContent_Populate(rapidjson)
  set(rapidjson_INCLUDE_DIR ${rapidjson_SOURCE_DIR}/include)
endif()

FetchContent_Declare(
  stanmath
  GIT_REPOSITORY https://github.com/stan-dev/math
  GIT_TAG        develop # replace with the version you want to use
)

FetchContent_GetProperties(stanmath)
if(NOT stanmath_POPULATED)
  FetchContent_Populate(stanmath)
endif()

FetchContent_Declare(
  stan
  GIT_REPOSITORY https://github.com/stan-dev/stan
  GIT_TAG        develop # replace with the version you want to use
)

FetchContent_GetProperties(stan)
if(NOT stan_POPULATED)
  FetchContent_Populate(stan)
  set(stan_INCLUDE_DIR ${stan_SOURCE_DIR}/src)
endif()

add_compile_options(
  -fPIC
  -DNO_FPRINTF_OUTPUT
  -DBOOST_DISABLE_ASSERTS
  -DTBB_INTERFACE_NEW
  -D_REENTRANT
  -Wno-deprecated-declarations -O3)

include_directories(src)
include_directories(${boost_INCLUDE_DIR} ${stanmath_SOURCE_DIR} ${rapidjson_INCLUDE_DIR} ${stan_INCLUDE_DIR})


add_library(ffistan OBJECT src/ffistan.cpp)

target_link_libraries(ffistan PRIVATE
TBB::tbb
Eigen3::Eigen
sundials_kinsol
sundials_cvodes)

# following is specific to each model

macro(stan_model arg1)
    add_custom_command(
      OUTPUT ${arg1}.cpp
      COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/bin/stanc --o=${arg1}.cpp ${CMAKE_CURRENT_SOURCE_DIR}/${arg1}.stan
      DEPENDS ${arg1}.stan
    )

    add_library(
            ${arg1}
            SHARED
            ${CMAKE_CURRENT_BINARY_DIR}/${arg1}.cpp
            $<TARGET_OBJECTS:ffistan>
    )

    target_link_libraries(${arg1} PRIVATE
    TBB::tbb
    Eigen3::Eigen
    sundials_kinsol
    sundials_cvodes)
endmacro()


stan_model(bernoulli)
stan_model(eight_schools)
