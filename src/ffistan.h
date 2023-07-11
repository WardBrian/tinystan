#ifndef FFISTAN_H
#define FFISTAN_H
#ifdef __cplusplus
struct stan_error;
struct FFIStanModel;
extern "C" {
#else
typedef struct stan_error stan_error;      // opaque type
typedef struct FFIStanModel FFIStanModel;  // opaque type
#endif

FFIStanModel *ffistan_create_model(const char *data, unsigned int seed,
                                   stan_error **err);

void ffistan_destroy_model(FFIStanModel *model);
const char *ffistan_model_param_names(const FFIStanModel *model);

enum FFIStanMetric { unit = 0, dense = 1, diagonal = 2 };

int ffistan_sample(const FFIStanModel *model, const char *inits,
                   unsigned int seed, unsigned int chain_id, double init_radius,
                   int num_warmup, int num_samples, FFIStanMetric metric_choice,
                   bool adapt, double delta, double gamma, double kappa,
                   double t0, unsigned int init_buffer,
                   unsigned int term_buffer, unsigned int window,
                   bool save_warmup, int refresh, double stepsize,
                   double stepsize_jitter, int max_depth, double *out,
                   stan_error **err);

const char *ffistan_get_error_message(const stan_error *err);
void ffistan_free_stan_error(stan_error *err);

#ifdef __cplusplus
}
#endif
#endif
