# TODO items

## Features
- [ ] Fixed param sampler for 0 dimension parameters?
- [ ] Other logging?
    - needs something like BridgeStan's print callback in general case,
      could get ugly if need a lock to print progress?
- [ ] Add nicer ability to build models from source in the languages
    - download source if needed, similar to bridgestan
- [ ] Version checking
- [ ] Language specific outputs (rvars, stanio, not sure if Julia has a good option)
- [x] Add ability to interrupt the algorithms during runs (Ctrl+C)
    - probably tricky to do in a way that works for all languages
- [x] Ability to output metric
- [x] Ability to input metric init

## Testing
- [ ] Add a variety of models, including:
    - [ ] A model that uses SUNDIALS
    - [x] A model with no parameters
- [x] Test in each language
- [x] Test on all platforms
- [x] Set up Github Actions
