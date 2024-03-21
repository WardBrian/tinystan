## include paths
TINYSTAN_ROOT ?= .

# user customization
-include $(TINYSTAN_ROOT)/make/local

SRC ?= $(TINYSTAN_ROOT)/src/
STAN ?= $(TINYSTAN_ROOT)/stan/
STANC ?= $(TINYSTAN_ROOT)/bin/stanc$(EXE)
MATH ?= $(STAN)lib/stan_math/
RAPIDJSON ?= $(STAN)lib/rapidjson_1.1.0/

## required C++ includes
INC_FIRST ?= -I $(STAN)src -I $(RAPIDJSON)

# TinyStan always wants multithreading support
STAN_THREADS=true

## makefiles needed for math library
include $(MATH)make/compiler_flags
include $(MATH)make/libraries
include $(MATH)make/dependencies

## Set -fPIC globally since we're always building a shared library
CXXFLAGS += -fPIC -fvisibility=hidden -fvisibility-inlines-hidden
CXXFLAGS_SUNDIALS += -fPIC
CPPFLAGS += -DTINYSTAN_EXPORT

ifeq ($(OS),Windows_NT)
	CXXFLAGS += -Wa,-mbig-obj
endif

## set flags for stanc compiler (math calls MIGHT? set STAN_OPENCL)
ifdef STAN_OPENCL
	STANCFLAGS += --use-opencl
	STAN_FLAG_OPENCL=_opencl
else
	STAN_FLAG_OPENCL=
endif
STAN_FLAGS=$(STAN_FLAG_OPENCL)


TINYSTAN_O = $(patsubst %.cpp,%$(STAN_FLAGS).o,$(SRC)tinystan.cpp)
TINYSTAN_DEPS := $(SRC)tinystan.cpp $(SRC)R_shims.cpp $(wildcard $(SRC)*.hpp) $(wildcard $(SRC)*.h)
include $(SRC)tinystan.d

$(TINYSTAN_O) : $(TINYSTAN_DEPS)
	@echo '--- Compiling TinyStan C++ code ---'
	@mkdir -p $(dir $@)
	$(COMPILE.cpp) $(OUTPUT_OPTION) $(LDLIBS) $<



## generate .hpp file from .stan file using stanc
%.hpp : %.stan $(STANC)
	@echo ''
	@echo '--- Translating Stan model to C++ code ---'
	$(STANC) $(STANCFLAGS) --o=$(subst  \,/,$@) $(subst  \,/,$<)

%.o : %.hpp
	@echo '--- Compiling C++ code ---'
	$(COMPILE.cpp) -x c++ -o $(subst  \,/,$*).o $(subst \,/,$<)

## builds executable (suffix depends on platform)
%_model.so : %.o $(TINYSTAN_O) $(SUNDIALS_TARGETS) $(MPI_TARGETS) $(TBB_TARGETS)
	@echo '--- Linking C++ code ---'
	$(LINK.cpp) -shared -lm -o $(patsubst %.o, %_model.so, $(subst \,/,$<)) $(subst \,/,$*.o) $(TINYSTAN_O) $(LDLIBS) $(SUNDIALS_TARGETS) $(MPI_TARGETS) $(TBB_TARGETS)

# build all test models at once
TEST_MODEL_NAMES = $(patsubst $(TINYSTAN_ROOT)/test_models/%/, %, $(sort $(dir $(wildcard $(TINYSTAN_ROOT)/test_models/*/))))
TEST_MODEL_NAMES := $(filter-out syntax_error, $(TEST_MODEL_NAMES))
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
	$(RM) $(SRC)/*.o
	$(RM) bin/stanc$(EXE)
	$(RM) $(TEST_MODEL_LIBS)

.PHONY: stan-update stan-update-version
stan-update:
	git submodule update --init --recursive

stan-update-remote:
	git submodule update --remote --init --recursive

# print compilation command line config
.PHONY: compile_info
compile_info:
	@echo '$(LINK.cpp) $(STANC_O) $(LDLIBS) $(SUNDIALS_TARGETS) $(MPI_TARGETS) $(TBB_TARGETS)'

## print value of makefile variable (e.g., make print-TBB_TARGETS)
.PHONY: print-%
print-%  : ; @echo $* = $($*) ;

# handles downloading of stanc
STANC_DL_RETRY = 5
STANC_DL_DELAY = 10
STANC3_TEST_BIN_URL ?=
STANC3_VERSION ?= nightly

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

ifeq ($(OS_TAG),windows)
$(STANC):
	@mkdir -p $(dir $@)
	$(shell echo "curl -L https://github.com/stan-dev/stanc3/releases/download/$(STANC3_VERSION)/$(OS_TAG)-stanc -o $(STANC) --retry $(STANC_DL_RETRY) --retry-delay $(STANC_DL_DELAY)")
else
$(STANC):
	@mkdir -p $(dir $@)
	curl -L https://github.com/stan-dev/stanc3/releases/download/$(STANC3_VERSION)/$(OS_TAG)$(ARCH_TAG)-stanc -o $(STANC) --retry $(STANC_DL_RETRY) --retry-delay $(STANC_DL_DELAY)
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
