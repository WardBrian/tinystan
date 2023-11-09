# TODO items

## Features
- [ ] Add nicer ability to build models from source in the languages
    - download source if needed, similar to bridgestan
- [ ] Version checking
- [ ] Language specific outputs (rvars, stanio, not sure if Julia has a good option)
- [ ] Fixed param sampler for 0 dimension parameters?
- [ ] Add wrapper around generate quantities method
- [x] Add ability to interrupt the algorithms during runs (Ctrl+C)
    - probably tricky to do in a way that works for all languages
- [x] Ability to output metric
- [x] Ability to input metric init
- [x] Other logging

## Testing
- [-] Add a variety of models, including:
    - [ ] A model that uses SUNDIALS
    - [x] A model with no parameters
- [ ] Test with sanitizers in CI?
- [x] Test in each language
- [x] Test on all platforms
- [x] Set up Github Actions


## Other
- [x] Set up visibility such that all non-API symbols are hidden
- [ ] Look into cmake/clang-cl builds
