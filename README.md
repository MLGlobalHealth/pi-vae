# πVAE
Code for utilising VAE as means of doing exact MCMC inference in complex high-dimensional space. 

Accompanying paper is [πVAE: a stochastic process prior for Bayesian deep learning with MCMC](https://arxiv.org/abs/2002.06873)

The πVAE model has 2 parts : 
1. Learning / encoding a stochastic prior via a VAE.
2. Then using the learnt basis, and decoder network , perform inference on our data to get a posterior.

To run the code :
1. Run src_py/models/pi_vae.py . To choose the the type of prior learnt, modify the training dataset, the current default is 1D GP.
2. To perform inference using stan, use the file notebooks/pivae.stan by passing the model parameters learnt in the above step. An example is given in the notebooks, notebooks/pivae.stan and notebooks/run_monotonic_mcmc.ipynb
