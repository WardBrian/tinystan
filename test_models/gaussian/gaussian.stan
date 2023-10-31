data {
  int<lower=0> N;
}
parameters {
  vector[N] alpha;
}
model {
  alpha ~ normal(0, 1);
}
