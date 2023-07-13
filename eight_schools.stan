data {
    int<lower=0> J;         // # of schools
    real y[J];              // estimated treatment
    real<lower=0> sigma[J]; // std err of effect
}
parameters {
    real theta[J]; // school effect
}
model {
    y ~ normal(theta, sigma);
}
