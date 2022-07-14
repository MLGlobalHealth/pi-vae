import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from utils import plotimp
from gen_data.gp1d import GP1D
import matplotlib.pyplot as plt 
import numpy as np
from tqdm import tqdm, trange
import cmdstanpy
import pandas as pd

class Encoder(nn.Module):
    ''' This the encoder part of VAE
    '''
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, z_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim1)
        self.linear2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.mu = nn.Linear(hidden_dim2, z_dim)
        self.sd = nn.Linear(hidden_dim2, z_dim)
    def forward(self, x):
        # x is of shape [batch_size, input_dim]
        hidden1 = torch.tanh(self.linear1(x))
        # hidden1 is of shape [batch_size, hidden_dim1]
        hidden2 = torch.tanh(self.linear2(hidden1))
        # hidden2 is of shape [batch_size, hidden_dim2]
        z_mu = self.mu(hidden2)
        # z_mu is of shape [batch_size, z_dim]
        z_sd = self.sd(hidden2)
        # z_sd is of shape [batch_size, z_dim]
        return z_mu, z_sd

class Decoder(nn.Module):
    ''' This the decoder part of VAE
    '''
    def __init__(self,z_dim, hidden_dim1, hidden_dim2, input_dim):
        super().__init__()
        self.linear1 = nn.Linear(z_dim, hidden_dim1)
        self.linear2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.out = nn.Linear(hidden_dim2, input_dim)
    def forward(self, x):
        # x is of shape [batch_size, z_dim]
        hidden1 = torch.tanh(self.linear1(x))
        # hidden1 is of shape [batch_size, hidden_dim1]
        hidden2 = torch.tanh(self.linear2(hidden1))
        # hidden2 is of shape [batch_size, hidden_dim2]
        pred = self.out(hidden2)
        # pred is of shape [batch_size, input_dim]
        return pred
class VAE(nn.Module):
    ''' This the VAE, which takes a encoder and decoder.
    '''
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, latent_dim):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim1, hidden_dim2, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim1, hidden_dim2, input_dim)

    def reparameterize(self, z_mu, z_sd):
        '''During training random sample from the learned ZDIMS-dimensional
           normal distribution; during inference its mean.
        '''
        if self.training:
            # sample from the distribution having latent parameters z_mu, z_sd
            # reparameterize
            std = torch.exp(z_sd / 2)
            eps = torch.randn_like(std)
            return (eps.mul(std).add_(z_mu))
        else:
            return z_mu


    def forward(self, x):
        # encode
        z_mu, z_sd = self.encoder(x)
        # reparameterize
        x_sample = self.reparameterize(z_mu, z_sd)
        # decode
        generated_x = self.decoder(x_sample)
        return generated_x, z_mu,z_sd

def calculate_loss(x, reconstructed_x, mean, log_sd):
    # reconstruction loss
    RCL = F.mse_loss(reconstructed_x, x, reduction='sum')
    # kl divergence loss
    KLD = -0.5 * torch.sum(1 + log_sd - mean.pow(2) - log_sd.exp())
    return RCL + KLD

if __name__ == "__main__":
    # Just showing how to use VAE to learn priors

    ###### intializing data and model parameters
    input_dim = 100
    batch_size = 500    
    hidden_dim1 = 64
    hidden_dim2 = 32
    z_dim = 20
    samples = 100000
    
    ###### creating data, model and optimizer
    train_ds = GP1D(dataPoints=input_dim, samples=samples, ls=0.1)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    model = VAE(input_dim, hidden_dim1, hidden_dim2, z_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    model = model.to(device)
    
    ###### running for 250 epochs
    t = trange(250)
    for e in t:
        # set training mode
        model.train()
        total_loss = 0
        for i,x in enumerate(train_dl):
            x = x[1].float().to(device)
            optimizer.zero_grad()   # zero the gradient buffers
            reconstructed_x, z_mu, z_sd = model(x) # fwd pass
            loss = calculate_loss(x, reconstructed_x, z_mu, z_sd) # loss cal
            loss.backward() # bck pass
            total_loss += loss.item() 
            optimizer.step() # update the weights

        t.set_description(f'Loss is {total_loss/(samples*input_dim):.3}')
    
    ###### Sampling 5 draws from learnt model
    model.eval() # model in eval mode
    z = torch.randn(5, z_dim).to(device) # random draw
    with torch.no_grad():
        sampled_y = model.decoder(z)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for no, y in enumerate(sampled_y):
        ax.plot(train_ds.evalPoints[:,0], y.cpu().numpy(), marker='o', markersize=3)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y=f(x)$')
    ax.set_title('5 different function realizations at fixed 100 points\n'
    'sampled from a VAE learned with prior as GP (RBF)')
    plt.savefig('plots/sample_prior_vae_1d_fixed.pdf')


    ###### Inference on observed data
    observed = GP1D(dataPoints=input_dim, samples=1, ls=0.1)
    model = model.to('cpu')
    decoder_dict = model.decoder.state_dict()
    f = observed[0][1]
    y = f + np.random.randn(input_dim) * 0.1
    stan_data = {'p': z_dim, 
                 'p1': hidden_dim1,
                 'p2': hidden_dim2,
                 'n': input_dim,
                 'W1': decoder_dict['linear1.weight'].T.numpy(),
                 'B1': decoder_dict['linear1.bias'].T.numpy(),
                 'W2': decoder_dict['linear2.weight'].T.numpy(),
                 'B2': decoder_dict['linear2.bias'].T.numpy(),
                 'W3': decoder_dict['out.weight'].T.numpy(),
                 'B3': decoder_dict['out.bias'].T.numpy(),
                 'y':y}
    #### stan code
    sm = cmdstanpy.CmdStanModel(stan_file='src_stan/stan_1D.stan')
    fit = sm.sample(data=stan_data, iter_sampling=2000, iter_warmup=500, chains=4)
    out = fit.stan_variables()

    df = pd.DataFrame(out['y2'])
    datapoints = observed[0][0]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(datapoints, f.reshape(-1,1), color='black', label='True')
    ax.scatter(datapoints, y.reshape(-1,1), s=46,label = 'Observations')
    ax.fill_between(datapoints, df.quantile(0.025).to_numpy(), df.quantile(0.975).to_numpy(),
                    facecolor="blue",
                    color='blue', 
                    alpha=0.2, label = '95% Credible Interval') 
    ax.plot(datapoints, df.mean().to_numpy().reshape(-1,1), color='red', alpha=0.7, label = 'Posterior mean')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y=f(x)$')
    ax.set_title('Inference fit')
    ax.legend()
    plt.savefig('plots/sample_inference_vae_1d_fixed.pdf')