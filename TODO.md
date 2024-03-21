# TODO items

## Features
- [ ] Add nicer ability to build models from source in the languages
    - [ ] download source if needed, similar to bridgestan
    - [ ] Version checking
- [ ] Fixed param sampler for 0 dimension parameters?
- [ ] Add wrapper around generate quantities method?
- [x] Pathfinder: expose the no lp/no PSIS version
  - [x] Pathfinder: now change single-path behavior to run PSIS?
- [x] Add ability to interrupt the algorithms during runs (Ctrl+C)
    - probably tricky to do in a way that works for all languages
- [x] Ability to output metric
- [x] Ability to input metric init
- [x] Other logging
- [x] Language specific outputs (rvars, stanio, not sure if Julia has a good option)

## Testing
- [ ] Test with sanitizers in CI?
- [x] Add a variety of models, including:
    - [x] A model that uses SUNDIALS
    - [x] A model with no parameters
- [x] Test in each language
- [x] Test on all platforms
- [x] Set up Github Actions


## Other
- [x] Set up visibility such that all non-API symbols are hidden
- [ ] Look into cmake/clang-cl builds
