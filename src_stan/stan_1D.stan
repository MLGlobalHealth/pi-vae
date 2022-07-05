functions {
  vector layer(vector x, matrix W, vector B) {
    return(transpose(transpose(x) * W) + B);
  }
  vector generator_stan(vector input, matrix W1, matrix W2, matrix W3, vector B1, vector B2, vector B3) {
    return(layer(tanh(layer(tanh(layer(input,W1,B1)),W2,B2)),W3,B3));
  }
}
data {
  int p; // input dimensionality (latent dimension of VAE)
  int p1; // hidden layer 1 number of units
  int p2; // hidden layer 2 number of units
  int n; // output dimensionality
  matrix[p,p1] W1; // weights matrix for layer 1
  vector[p1] B1; // bias matrix for layer 1
  matrix[p1,p2] W2; // weights matrix for layer 2
  vector[p2] B2; // bias matrix for layer 2
  matrix[p2,n] W3; // weights matrix for layer output
  vector[n] B3; // bias matrix for layer output
  
  vector[n] y;
}
parameters {
  vector[p] z;
  real<lower=0> sigma2;
}
transformed parameters {
  vector[n] f;
  f = generator_stan(z,W1,W2,W3,B1,B2,B3);
}
model {
  z ~ normal(0,1);
  sigma2 ~ normal(0,10);
  y ~ normal(f,sigma2);
}
generated quantities {
  vector[n] y2;
  for (i in 1:n)
    y2[i] = normal_rng(f[i], sigma2);
}