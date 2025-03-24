# include paths
TINYSTAN_ROOT ?= .

# user customization
-include $(TINYSTAN_ROOT)/make/local

SRC ?= $(TINYSTAN_ROOT)/src/
STAN ?= $(TINYSTAN_ROOT)/stan/
STANC ?= $(TINYSTAN_ROOT)/bin/stanc$(EXE)
MATH ?= $(STAN)lib/stan_math/
RAPIDJSON ?= $(STAN)lib/rapidjson_1.1.0/

# required C++ includes
INC_FIRST ?= -I $(STAN)src -I $(RAPIDJSON)

# TinyStan wants multithreading support by default
ifndef TINYSTAN_SERIAL
	STAN_THREADS=true
	STAN_FLAG_SERIAL=
else
	STAN_FLAG_SERIAL=_serial
endif

# We can bump to C++17, even if Stan hasn't yet
STAN_HAS_CXX17 ?= true
CXXFLAGS_LANG ?= -std=c++17

# makefiles needed for math library
include $(MATH)make/compiler_flags
include $(MATH)make/libraries
include $(MATH)make/dependencies

# Set -fPIC globally since we're always building a shared library
override CXXFLAGS += -fPIC -fvisibility=hidden -fvisibility-inlines-hidden
override CXXFLAGS_SUNDIALS += -fPIC
override CPPFLAGS += -DTINYSTAN_EXPORT

ifeq ($(OS),Windows_NT)
	override CXXFLAGS += -Wa,-mbig-obj
endif

ifdef STAN_OPENCL
	# set flags for stanc compiler
	override STANCFLAGS += --use-opencl
	STAN_FLAG_OPENCL=_opencl
else
	STAN_FLAG_OPENCL=
endif
STAN_FLAGS=$(STAN_FLAG_OPENCL)$(STAN_FLAG_SERIAL)


# TORSTEN
TORSTEN ?= $(TINYSTAN_ROOT)/torsten/

SUNDIALS_ARKODE := $(patsubst %.c,%.o,\
  $(wildcard $(SUNDIALS)/src/arkode/*.c) \
  $(wildcard $(SUNDIALS)/src/sundials/*.c) \
  $(wildcard $(SUNDIALS)/src/sunmatrix/band/[^f]*.c) \
  $(wildcard $(SUNDIALS)/src/sunmatrix/dense/[^f]*.c) \
  $(wildcard $(SUNDIALS)/src/sunlinsol/band/[^f]*.c) \
  $(wildcard $(SUNDIALS)/src/sunlinsol/dense/[^f]*.c) \
  $(wildcard $(SUNDIALS)/src/sunnonlinsol/newton/[^f]*.c) \
  $(wildcard $(SUNDIALS)/src/sunnonlinsol/fixedpoint/[^f]*.c))

$(SUNDIALS)/lib/libsundials_arkode.a: $(SUNDIALS_ARKODE)
	@mkdir -p $(dir $@)
	$(AR) -rs $@ $^

$(sort $($(patsubst %.c,%.o, $(wildcard $(SUNDIALS)/src/arkode/*.c)))) : CXXFLAGS = $(CXXFLAGS_SUNDIALS) $(CXXFLAGS_OS) $(CXXFLAGS_OPTIM_SUNDIALS) -O$(O) $(INC_SUNDIALS)
$(sort $($(patsubst %.c,%.o, $(wildcard $(SUNDIALS)/src/arkode/*.c)))) : CPPFLAGS = $(CPPFLAGS_SUNDIALS) $(CPPFLAGS_OS) $(CPPFLAGS_OPTIM_SUNDIALS) -O$(O)
$(sort $($(patsubst %.c,%.o, $(wildcard $(SUNDIALS)/src/arkode/*.c)))) : %.o : %.c
	@mkdir -p $(dir $@)
	$(COMPILE.cpp) -x c -include $(SUNDIALS)/include/stan_sundials_printf_override.hpp $< $(OUTPUT_OPTION)

SUNDIALS_TARGETS += $(SUNDIALS)/lib/libsundials_arkode.a

override CPPFLAGS += -I $(TORSTEN)
# Adding Torsten functions making MPL list too long, need adjust list size
override CXXFLAGS += -DBOOST_MPL_CFG_NO_PREPROCESSED_HEADERS -DBOOST_MPL_LIMIT_LIST_SIZE=30

# END TORSTEN

TINYSTAN_O = $(patsubst %.cpp,%$(STAN_FLAGS).o,$(SRC)tinystan.cpp)
TINYSTAN_DEPS := $(SRC)tinystan.cpp $(SRC)R_shims.cpp $(wildcard $(SRC)*.hpp) $(wildcard $(SRC)*.h)
include $(SRC)tinystan.d

$(TINYSTAN_O) : $(TINYSTAN_DEPS)
	@echo '--- Compiling TinyStan C++ code ---'
	@mkdir -p $(dir $@)
	$(COMPILE.cpp) $(OUTPUT_OPTION) $(LDLIBS) $<


# support user header file for undefined functions
ifneq ($(findstring allow-undefined,$(STANCFLAGS)),)
USER_HEADER ?= $(dir $(MAKECMDGOALS))user_header.hpp
USER_INCLUDE = -include $(USER_HEADER)
# Give a better error message if the USER_HEADER is not found
$(USER_HEADER):
	@echo 'ERROR: Missing user header.'
	@echo 'Because --allow-undefined is set, we need a C++ header file to include.'
	@echo 'We tried to find the user header at:'
	@echo '  $(USER_HEADER)'
	@echo ''
	@echo 'You can also set the USER_HEADER variable to the path of your C++ file.'
	@exit 1
endif


# save compilation time by precompiling the model header
PRECOMPILED_HEADERS ?= false
ifeq ($(PRECOMPILED_HEADERS),true)
PRECOMPILED_MODEL_HEADER=$(STAN)src/stan/model/model_header.hpp.gch/model_header$(STAN_FLAGS)_$(CXX_MAJOR)_$(CXX_MINOR).hpp.gch

$(patsubst %.hpp.gch,%.d,$(PRECOMPILED_MODEL_HEADER)) : DEPTARGETS = -MT $@
$(patsubst %.hpp.gch,%.d,$(PRECOMPILED_MODEL_HEADER)) : DEPFLAGS_OS = -M -E
$(patsubst %.hpp.gch,%.d,$(PRECOMPILED_MODEL_HEADER)) : $(STAN)src/stan/model/model_header.hpp
	@mkdir -p $(dir $@)
	$(COMPILE.cpp) $(DEPFLAGS) $<

-include $(patsubst %.hpp.gch,%.d,$(PRECOMPILED_MODEL_HEADER))

$(PRECOMPILED_MODEL_HEADER): $(STAN)src/stan/model/model_header.hpp
	@echo ''
	@echo '--- Compiling pre-compiled header. ---'
	@mkdir -p $(dir $@)
	$(COMPILE.cpp)  -include $(TORSTEN)torsten_include.hpp $< $(OUTPUT_OPTION)

ifeq ($(CXX_TYPE),clang)
PRECOMPILED_HEADER_INCLUDE = -include-pch $(PRECOMPILED_MODEL_HEADER)
endif
endif

# generate .hpp file from .stan file using stanc
%.hpp : %.stan $(STANC)
	@echo ''
	@echo '--- Translating Stan model to C++ code ---'
	$(STANC) $(STANCFLAGS) --o=$(subst  \,/,$@) $(subst  \,/,$<)

%.o : %.hpp $(USER_HEADER) $(PRECOMPILED_MODEL_HEADER)
	@echo '--- Compiling C++ code ---'
	$(COMPILE.cpp) $(PRECOMPILED_HEADER_INCLUDE) $(USER_INCLUDE)  -include $(TORSTEN)torsten_include.hpp -x c++ -o $(subst  \,/,$*).o $(subst \,/,$<)

%_model.so : %.o $(TINYSTAN_O) $(SUNDIALS_TARGETS) $(MPI_TARGETS) $(TBB_TARGETS)
	@echo '--- Linking C++ code ---'
	$(LINK.cpp) -shared -lm -o $(patsubst %.o, %_model.so, $(subst \,/,$<)) $(subst \,/,$*.o) $(TINYSTAN_O) $(LDLIBS) $(SUNDIALS_TARGETS) $(MPI_TARGETS) $(TBB_TARGETS)

# build all test models at once
ALL_TEST_MODEL_NAMES = $(patsubst $(TINYSTAN_ROOT)/test_models/%/, %, $(sort $(dir $(wildcard $(TINYSTAN_ROOT)/test_models/*/))))
# these are for compilation testing in the interfaces
SKIPPED_TEST_MODEL_NAMES = syntax_error external
TEST_MODEL_NAMES := $(filter-out $(SKIPPED_TEST_MODEL_NAMES), $(ALL_TEST_MODEL_NAMES))
TEST_MODEL_LIBS = $(join $(addprefix test_models/, $(TEST_MODEL_NAMES)), $(addsuffix _model.so, $(addprefix /, $(TEST_MODEL_NAMES))))

.PHONY: test_models
test_models: $(TEST_MODEL_LIBS)


.PHONY: format
format:
	clang-format -i src/*.cpp src/*.hpp src/*.h || true
	isort clients/python || true
	black clients/python || true
	julia --project=clients/julia -e 'using JuliaFormatter; format("clients/julia/")' || true
	Rscript -e 'formatR::tidy_dir("clients/R/", recursive=TRUE)' || true

.PHONY: clean
clean:
	$(RM) $(SRC)*.o $(SRC)*.d
	$(RM) -r $(STAN)src/stan/model/model_header.hpp.gch/
	$(RM) $(TINYSTAN_ROOT)/test_models/**/*.so
	$(RM) $(join $(addprefix $(TINYSTAN_ROOT)/test_models/, $(TEST_MODEL_NAMES)), $(addsuffix .hpp, $(addprefix /, $(TEST_MODEL_NAMES))))
	$(RM) bin/stanc$(EXE)

.PHONY: stan-update stan-update-version
stan-update:
	git submodule update --init --recursive

stan-update-remote:
	git submodule update --remote --init --recursive

# print compilation command line config
.PHONY: compile_info
compile_info:
	@echo '$(LINK.cpp) $(STANC_O) $(LDLIBS) $(SUNDIALS_TARGETS) $(MPI_TARGETS) $(TBB_TARGETS)'

# print value of makefile variable (e.g., make print-TBB_TARGETS)
.PHONY: print-%
print-%  : ; @echo $* = $($*) ;

# handles downloading of stanc
STANC_DL_RETRY = 5
STANC_DL_DELAY = 10
STANC3_TEST_BIN_URL ?=
STANC3_VERSION ?= testing

ifeq ($(OS),Windows_NT)
 OS_TAG := windows
else ifeq ($(OS),Darwin)
 OS_TAG := mac
else ifeq ($(OS),Linux)
 OS_TAG := linux
 ifeq ($(shell uname -m),mips64)
  ARCH_TAG := -mips64el
 else ifeq ($(shell uname -m),ppc64le)
  ARCH_TAG := -ppc64el
 else ifeq ($(shell uname -m),s390x)
  ARCH_TAG := -s390x
 else ifeq ($(shell uname -m),aarch64)
  ARCH_TAG := -arm64
 else ifeq ($(shell uname -m),armv7l)
  ifeq ($(shell readelf -A /usr/bin/file | grep Tag_ABI_VFP_args),)
    ARCH_TAG := -armel
  else
    ARCH_TAG := -armhf
  endif
 endif
endif


#  from https://github.com/WardBrian/stanc3/tree/torsten-stanc3
ifeq ($(OS_TAG),windows)
$(STANC):
	@mkdir -p $(dir $@)
	$(shell echo "curl -L https://github.com/WardBrian/stanc3/releases/download/$(STANC3_VERSION)/$(OS_TAG)-stanc -o $(STANC) --retry $(STANC_DL_RETRY) --retry-delay $(STANC_DL_DELAY)")
else
$(STANC):
	@mkdir -p $(dir $@)
	curl -L https://github.com/WardBrian/stanc3/releases/download/$(STANC3_VERSION)/$(OS_TAG)$(ARCH_TAG)-stanc -o $(STANC) --retry $(STANC_DL_RETRY) --retry-delay $(STANC_DL_DELAY)
	chmod +x $(STANC)
endif

##
# This is only run if the `include` statements earlier fail to find a file.
# We assume that means the submodule is missing
##
$(MATH)make/% :
	@echo 'ERROR: Missing Stan submodules.'
	@echo 'We tried to find the Stan Math submodule at:'
	@echo '  $(MATH)'
	@echo ''
	@echo 'The most likely source of the problem is TinyStan was cloned without'
	@echo 'the --recursive flag.  To fix this, run the following command:'
	@echo '  git submodule update --init --recursive'
	@echo ''
	@echo 'And try building again'
	@exit 1

# EMSCRIPTEN is defined by emmake, so we can use it to detect if we're in an emscripten environment
ifneq (,$(EMSCRIPTEN))
%.js : %.o $(TINYSTAN_O) $(SUNDIALS_TARGETS) $(MPI_TARGETS) $(TBB_TARGETS)
	@echo '--- Linking C++ code ---'
	$(LINK.cpp) -lm -o $(patsubst %.o, %.js, $(subst \,/,$<)) $(subst \,/,$*.o) $(TINYSTAN_O) $(LDLIBS) $(SUNDIALS_TARGETS) $(MPI_TARGETS) $(TBB_TARGETS)
else
%.js :
	@echo 'ERROR: Emscripten is required to compile to WebAssembly.'
	@echo 'Please install Emscripten and make sure you are using `emmake`.'
	@exit 1
endif
