functions {
  real rmm(real s, real p, real km_s, real km_p, real vmax, real keq){
   real num = vmax * (s - p / keq) / km_s;
   real denom = 1 + s / km_s + p / km_p;
   return num / denom;
  }
  real ma(real s, real p, real kf, real keq){
   return kf * (s - p / keq);
  }
  vector dcdt(vector c, vector km, real vmax, vector keq, vector kf, vector cext){
   matrix[4, 3] S = [[-1, 1, 0, 0], [0, -1, 1, 0], [0, 0, -1, 1]]';
   vector[3] v = [
    ma(cext[1], c[1], kf[1], keq[1]),
    rmm(c[1], c[2], km[1], km[2], vmax, keq[2]),
    ma(c[1], cext[2], kf[2], keq[3])
   ]';
   vector[2] out = (S * v)[2:3];
   return out;
  }
}
data {
 vector[2] y;
 real<lower=0> sd;
 array[2] vector[2] prior_log_km;
 array[2] real prior_log_vmax;
 array[2] vector[3] prior_log_keq;
 array[2] vector[2] prior_log_kf;
 array[2] vector[2] prior_log_cext;
 vector[2] y_guess;
 real scaling_step;
 real ftol;
 int<lower=0> max_steps;
}
parameters {
 vector[2] log_km;
 real log_vmax;
 vector[3] log_keq;
 vector[2] log_kf;
 vector[2] log_cext;
}
transformed parameters {
  vector[2] km = exp(log_km);
  real vmax = exp(log_vmax);
  vector[3] keq = exp(log_keq);
  vector[2] kf = exp(log_kf);
  vector[2] cext = exp(log_cext);
  vector[2] yhat = solve_newton_tol(dcdt,
                                    y_guess,
                                    scaling_step,
                                    ftol,
                                    max_steps,
                                    km,
                                    vmax,
                                    keq,
                                    kf,
                                    cext);
}
model {
 log_km ~ normal(prior_log_km[1], prior_log_km[2]);
 log_vmax ~ normal(prior_log_vmax[1], prior_log_vmax[2]);
 log_keq ~ normal(prior_log_keq[1], prior_log_keq[2]);
 log_kf ~ normal(prior_log_kf[1], prior_log_kf[2]);
 log_cext ~ normal(prior_log_cext[1], prior_log_cext[2]);
 y ~ lognormal(log(yhat), sd);
}
