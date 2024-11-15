/**
 * @typedef {Record<string, unknown>} StanVariableInputs
 * A type holding named inputs to a Stan model,
 * e.g. the data or initial values.
 */
export type StanVariableInputs = Record<string, unknown>;

/**
 * @typedef {Function} PrintCallback
 * A callback for printing output from the Stan model.
 * @param {string} s The string to print.
 * @returns {void}
 */
export type PrintCallback = (s: string) => void;

/**
 * The metric used for the HMC sampler.
 * @enum {number}
 * @readonly
 * @property {0} UNIT - Unit metric.
 * @property {1} DENSE - Dense metric.
 * @property {2} DIAGONAL - Diagonal metric.
 */
export enum HMCMetric {
  UNIT = 0,
  DENSE = 1,
  DIAGONAL = 2,
}

/**
 * @typedef {Object} StanDraws
 * A type holding the result of a Stan sampling run.
 * @property {string[]} paramNames The names of the parameters in
 * the model
 * @property {number[][]} draws A 2D array of draws from the posterior.
 * The first dimension is the number of samples, and the second dimension
 * is the number of parameters.
 * @property {number[][] | number[][][] | undefined} metric The metric used for the
 * HMC sampler. If the metric is not saved, this field is not present.
 */
export type StanDraws = {
  paramNames: string[];
  draws: number[][];
  metric?: number[][] | number[][][];
};

/**
 * @typedef {Object} SamplerParams
 * Parameters for the HMC sampler.
 * @property {string | StanVariableInputs} [data=""] The data for the model
 * @property {number} [num_chains=4] The number of chains to run
 * @property {string | StanVariableInputs | string[] | StanVariableInputs[]} [inits=""]
 * The initial values for the sampler. If an array, must have length `num_chains`.
 * @property {number | null} [seed] The seed for the random number generator.
 * If unspecified, a random seed will be generated.
 * @property {number} [id=1] The ID for the first chain
 * @property {number} [init_radius=2.0] Radius to initialize unspecified parameters within.
 * The parameter values are drawn uniformly from the interval
 * `[-init_radius, init_radius]` on the unconstrained scale.
 * @property {number} [num_warmup=1000] The number of warmup iterations to run
 * @property {number} [num_samples=1000] The number of samples to draw after warmup
 * @property {HMCMetric} [metric=HMCMetric.DENSE] The type of mass matrix to use in the sampler
 * @property {boolean} [save_metric=false] Whether to report the final mass matrix
 * @property {number[] | number[][] | number[][][] | null} [init_inv_metric]
 * The initial inverse metric to use. Currently, this argument is unused.
 * @property {boolean} [adapt=true] Whether the sampler should adapt the step size and metric
 * @property {number} [delta=0.8] Target acceptance rate
 * @property {number} [gamma=0.05] Adaptation regularization scale
 * @property {number} [kappa=0.75] Adaptation relaxation exponent
 * @property {number} [t0=10.0] Adaptation iteration offset
 * @property {number} [init_buffer=75] Number of warmup samples to use for initial
 * step size adaptation.
 * @property {number} [term_buffer=50] Number of warmup samples to use for step size
 * adaptation after the metric is adapted
 * @property {number} [window=25] Initial number of iterations to use for metric adaptation,
 * which is doubled each time the adaptation window is hit
 * @property {boolean} [save_warmup=false] Whether to save the warmup draws
 * @property {number} [stepsize=1.0] Initial step size for the sampler
 * @property {number} [stepsize_jitter=0.0] Amount of random jitter to add to the step size
 * @property {number} [max_depth=10] Maximum tree depth for the NUTS sampler
 * @property {number} [refresh=0] Number of iterations between progress messages.
 * If 0, no output is printed.
 * @property {number} [num_threads=-1] Number of threads to use for sampling.
 * If -1, the number of threads is determined by the number of available CPU cores.
 * May not be supported in all environments, and requires extra configuration
 * during the Emscripten compilation.
 */
export interface SamplerParams {
  data: string | StanVariableInputs;
  num_chains: number;
  inits: string | StanVariableInputs | string[] | StanVariableInputs[];
  seed: number | null;
  id: number;
  init_radius: number;
  num_warmup: number;
  num_samples: number;
  metric: HMCMetric;
  save_metric: boolean;
  init_inv_metric: number[] | number[][] | number[][][] | null;
  adapt: boolean;
  delta: number;
  gamma: number;
  kappa: number;
  t0: number;
  init_buffer: number;
  term_buffer: number;
  window: number;
  save_warmup: boolean;
  stepsize: number;
  stepsize_jitter: number;
  max_depth: number;
  refresh: number;
  num_threads: number;
}

interface LBFGSConfig {
  max_history_size: number;
  init_alpha: number;
  tol_obj: number;
  tol_rel_obj: number;
  tol_grad: number;
  tol_rel_grad: number;
  tol_param: number;
  num_iterations: number;
}

interface PathfinderUniqueParams {
  data: string | StanVariableInputs;
  num_paths: number;
  inits: string | StanVariableInputs | string[] | StanVariableInputs[];
  seed: number | null;
  id: number;
  init_radius: number;
  num_draws: number;
  num_elbo_draws: number;
  num_multi_draws: number;
  calculate_lp: boolean;
  psis_resample: boolean;
  refresh: number;
  num_threads: number;
}

/**
 * @typedef {Object} PathfinderParams
 * Parameters for the Pathfinder algorithm.
 * @property {string | StanVariableInputs} [data=""] The data for the model
 * @property {number} [num_paths=4] The number of individual paths to run
 * @property {string | StanVariableInputs | string[] | StanVariableInputs[]} [inits=""]
 * The initial values for the algorithm. If an array, must have length `num_paths`.
 * @property {number | null} [seed] The seed for the random number generator.
 * If unspecified, a random seed will be generated.
 * @property {number} [id=1] The ID for the first path
 * @property {number} [init_radius=2.0] Radius to initialize unspecified parameters within.
 * The parameter values are drawn uniformly from the interval
 * `[-init_radius, init_radius]` on the unconstrained scale.
 * @property {number} [num_draws=1000] The number of draws to take for each path
 * @property {number} [max_history_size=5] History size used by the internal L-BFGS algorithm
 * to approximate the Hessian
 * @property {number} [init_alpha=0.001] Initial step size for the internal L-BFGS algorithm
 * @property {number} [tol_obj=1e-12] Convergence tolerance for the objective function
 * @property {number} [tol_rel_obj=1e4] Relative convergence tolerance for the objective function
 * @property {number} [tol_grad=1e-8] Convergence tolerance for the gradient norm
 * @property {number} [tol_rel_grad=1e7] Relative convergence tolerance for the gradient norm
 * @property {number} [tol_param=1e-8] Convergence tolerance for the changes in parameters
 * @property {number} [num_iterations=1000] Maximum number of iterations for the internal
 * L-BFGS algorithm
 * @property {number} [num_elbo_draws=25] Number of Monte Carlo draws used to estimate the ELBO
 * @property {number} [num_multi_draws=1000] Number of draws returned by Multi-Pathfinder
 * @property {boolean} [calculate_lp=true] Whether to calculate the log probability of the
 * approximate draws.
 * If false, this also implies `psis_resample=false`.
 * @property {boolean} [psis_resample=true] Whether to use Pareto smoothed importance sampling on
 * the approximate draws. If false, all `num_paths * num_draws` approximate samples will be returned.
 * @property {number} [refresh=0] Number of iterations between progress messages.
 * If 0, no output is printed.
 * @property {number} [num_threads=-1] Number of threads to use for Pathfinder.
 * If -1, the number of threads is determined by the number of available CPU cores.
 * May not be supported in all environments, and requires extra configuration
 * during the Emscripten compilation.
 */
export type PathfinderParams = LBFGSConfig & PathfinderUniqueParams;

// ------------- internal types -------------
// not exported directly, only used for WASM
// communication and therefore may not be stable

// Newtype trick to create distinct types for different kinds of pointers
const brand = Symbol("brand");
type Brand<T, U> = T & {
  [brand]: U;
};

type ptr = Brand<number, "raw pointer">;
type model_ptr = Brand<number, "model pointer">;
type error_ptr = Brand<number, "error object pointer">;
type cstr = Brand<number, "null-terminated char pointer">;

interface WasmModule {
  _malloc(n_bytes: number): ptr;
  _free(pointer: ptr | cstr): void;
  _tinystan_create_model(data: cstr, seed: number, err_ptr: ptr): model_ptr;
  _tinystan_destroy_model(model: model_ptr): void;
  _tinystan_model_param_names(model: model_ptr): cstr;
  _tinystan_model_num_free_params(model: model_ptr): number;
  _tinystan_separator_char(): number;
  // prettier-ignore
  _tinystan_sample(model: model_ptr, num_chains: number, inits: cstr, seed: number, id: number,
    init_radius: number, num_warmup: number, num_samples: number, metric: number, init_inv_metric: ptr,
    adapt: number, delta: number, gamma: number, kappa: number, t0: number, init_buffer: number,
    term_buffer: number, window: number, save_warmup: number, stepsize: number, stepsize_jitter: number,
    max_depth: number, refresh: number, num_threads: number, out: ptr, out_size: number, metric_out: ptr,
    err_ptr: ptr): number;
  // prettier-ignore
  _tinystan_pathfinder(model: model_ptr, num_paths: number, inits: cstr, seed: number, id: number,
    init_radius: number, num_draws: number, max_history_size: number, init_alpha: number, tol_obj: number,
    tol_rel_obj: number, tol_grad: number, tol_rel_grad: number, tol_param: number, num_iterations: number,
    num_elbo_draws: number, num_multi_draws: number, calculate_lp: number, psis_resample: number,
    refresh: number, num_threads: number, out: ptr, out_size: number, err_ptr: ptr): number;
  _tinystan_get_error_message(err_ptr: error_ptr): cstr;
  _tinystan_get_error_type(err_ptr: error_ptr): number;
  _tinystan_destroy_error(err_ptr: error_ptr): void;
  _tinystan_api_version(major: ptr, minor: ptr, patch: ptr): void;
  _tinystan_stan_version(major: ptr, minor: ptr, patch: ptr): void;
  lengthBytesUTF8(str: string): number;
  UTF8ToString(ptr: cstr, max?: number): string;
  stringToUTF8(str: string, outPtr: cstr, maxBytesToWrite: number): number;
  getValue(ptr: number, type: string): number;
  HEAPF64: Float64Array;
}

export type internalTypes = {
  WasmModule: WasmModule;
  ptr: ptr;
  model_ptr: model_ptr;
  error_ptr: error_ptr;
  cstr: cstr;
};
