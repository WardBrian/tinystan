# TODO items

## Features
- [x] Add nicer ability to build models from source in the languages
    - [x] download source if needed, similar to bridgestan
    - [x] Version checking
- [-] ~Fixed param sampler for 0 dimension parameters?~
- [ ] Add wrapper around generate quantities method?
- [x] Add wraper around laplace sampling?
- [x] Pathfinder: expose the no lp/no PSIS version
  - [x] Pathfinder: now change single-path behavior to run PSIS?
- [x] Add ability to interrupt the algorithms during runs (Ctrl+C)
    - probably tricky to do in a way that works for all languages
- [x] Ability to output metric
- [x] Ability to input metric init
- [x] Other logging
- [x] Language specific outputs (rvars, stanio, not sure if Julia has a good option)

## Testing
- [-] Test with sanitizers in CI?
- [x] Add a variety of models, including:
    - [x] A model that uses SUNDIALS
    - [x] A model with no parameters
- [x] Test in each language
- [x] Test on all platforms
- [x] Set up Github Actions


## Other
- [x] Documentation
- [x] Set up visibility such that all non-API symbols are hidden
- [x] Rename
- [x] Try to compile for webassembly with emscripten
- [ ] Look into cmake/clang-cl builds

## R Package

- [ ] Add hook for R package with formatR
- [ ] Docs for all public functions
- [ ] Vignette for usage
- [ ] tinystan-tools companion package?
- [ ] Add methods for running loo etc. with tinystan model
- [ ] Verbose checking of inputs for all the algorithms
- [ ] Harden compile functions
- [ ] `tinystan-fit` as return that holds
  - [ ] return samples as rvars
  - [ ] S3 functions defined to extract rvars to the environment
  - [ ] S3 functions for calling loo etc on the fit? Since we want to keep things simple I'm not sure that's within scope
