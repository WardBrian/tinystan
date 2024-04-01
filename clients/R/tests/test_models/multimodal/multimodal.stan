parameters {
  real mu;
}
model {
  target += log_sum_exp(normal_lpdf(mu | -100, 1), normal_lpdf(mu | 100, 1));
}
