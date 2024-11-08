functions {
  vector system(vector y, vector theta){
    vector[2] z;
    z[1] = y[1] - theta[1];
    z[2] = y[1] * y[2] - theta[2];
    return z;
  }
}
data {
  vector[2] prior_mean_theta;
  vector[2] prior_sd_theta;
  vector[2] y;
  real<lower=0> sd;
  vector[2] y_guess;
  real scaling_step;
  real ftol;
  int<lower=0> max_steps;
}
parameters {
  vector[2] theta;
}
transformed parameters {
  vector[2] yhat = solve_newton_tol(system,
                                    y_guess,
                                    scaling_step,
                                    ftol,
                                    max_steps,
                                    theta);
}
model {
  theta ~ normal([3, 6]', 1);
  y ~ normal(yhat, sd);
}
