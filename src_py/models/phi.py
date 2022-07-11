import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.nn.parameter import Parameter

class PHI(nn.Module):
    '''
    Implementation of feature transformation layer with RBF layer.
    We assume here that alpha is constant for all basis.
    Shape:
        - Input: (N, n_evals, in_features) N is batches
        - Output: (N, n_evals, out_dims), out_dims is a parameter
    Parameters:
        - in_features: number of input dimension for each eval point
        - alpha - trainable parameter controls width. Default is 1.0
        - n_centers - number of points to be used as centers in rbf/matern
        layers. centers are trainable, default is 100
        - hidden_dim1: hidden dimension for 1st layer. Default is 20
        - hidden_dim2: hidden dimension for 2nd layer. Default is 20
        - out_dims: output features to construct. Default is 100
    Examples:
        >>> a1 = PHI(256)
        >>> x = torch.randn(1,256)
        >>> x = a1(x)
    '''
    def __init__(self, in_features, alpha = 1.0, n_centers = 10, 
                    hidden_dim1 = 20, hidden_dim2 = 20, out_dims = 100):
        '''
        Initialization.
        INPUT:
            - in_features: number of input dimension for each eval point
            - alpha: trainable parameter
            alpha is initialized with 1.0 value by default
            - n_centers: number of points to be used as centers in rbf/matern
            layers. centers are trainable, default is 100
            - hidden_dim1: hidden dimension for 1st layer. Default is 20
            - hidden_dim2: hidden dimension for 2nd layer. Default is 20
            - out_dims: hidden dimension for 2nd layer. Default is 100
        '''
        super(PHI,self).__init__()
        self.in_features = in_features

        # initialize alpha
        self.alpha = Parameter(torch.tensor(alpha)) # create a tensor out of alpha
        self.alpha.requiresGrad = True # set requiresGrad to true!
        # centers
        self.centers = Parameter(torch.randn(n_centers, in_features)) # create a tensor out of centers
        self.centers.requiresGrad = True # set requiresGrad to true!
        # linear layers
        self.linear1 = nn.Linear(n_centers, hidden_dim1)
        self.linear2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.out = nn.Linear(hidden_dim2, out_dims)

    def forward(self, x):
        '''
        Forward pass of the function.
        Applies the function to the input elementwise.
        '''
        rbf = torch.exp(-self.alpha * torch.cdist(x, self.centers).pow(2))
        hidden1 = torch.tanh(self.linear1(rbf))
        hidden2 = torch.tanh(self.linear2(hidden1))
        out = self.out(hidden2)
        return out

if __name__ == "__main__":

    n_samples = 10
    in_features = 1
    n_evals = 20
    n_centers = 200
    input = torch.randn(n_samples, n_evals, in_features)
    phi = PHI(in_features, n_centers=n_centers)
    print(phi(input).shape)
    
    a1 = PHI(256)
    x = torch.randn(5,256)
    print(a1(x).shape)