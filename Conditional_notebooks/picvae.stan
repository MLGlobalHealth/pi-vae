functions {
  vector layer(vector x, matrix W, vector B) {
    return(transpose(transpose(x) * W) + B);
  }
  vector generator_stan(vector input1,vector input2, matrix W1, matrix W2, matrix W3, vector B1, vector B2, vector B3) {
    return(layer(tanh(layer(tanh(layer(append_col(input1,input2),W1,B1)),W2,B2)),W3,B3));
  }
}
data {
  int p; // input dimensionality (latent dimension of VAE)
  int p1; // hidden layer 1 number of units
  int p2; // hidden layer 2 number of units
  int n; // output dimensionality
  int beta_dim; //dimenasion of beta
  matrix[p+1,p1] W1; // weights matrix for layer 1
  vector[p1] B1; // bias matrix for layer 1
  matrix[p1,p2] W2; // weights matrix for layer 2
  vector[p2] B2; // bias matrix for layer 2
  matrix[p2,beta_dim] W3; // weights matrix for layer output
  vector[beta_dim] B3; // bias matrix for layer output
  
  
  matrix[n,beta_dim] phi_x; // values of loaction after going through phi
  vector[n] c;
  vector[n] y;
  int ll_len;                    // length of indices for likelihood
  int ll_idxs[ll_len];           // indices for likelihood
}
parameters {
  vector[p] z;
  real<lower=0> sigma2;
}
transformed parameters {
  vector[beta_dim] f;
  vector[n] y_hat;
  f = generator_stan(z,c,W1,W2,W3,B1,B2,B3);
  y_hat = phi_x * f;
}
model {
  z ~ normal(0,1);
  sigma2 ~ normal(0,10);
  y[ll_idxs] ~ normal(y_hat[ll_idxs],sigma2);
}
 generated quantities {
  vector[n] y2;
  for (i in 1:n)
    y2[i] = normal_rng(y_hat[i], sigma2);
}