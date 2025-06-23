# Author: Alba Garrido López
# Email: alba.garrido.lopez@upm.es

# Import libraries
import torch
import numpy as np
 

def check_nan_inf(values, log):
    if torch.isnan(values).any() or torch.isinf(values).any():
        raise RuntimeError('NAN OR INF DETECTED. ' + str(log))
    return

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0, min_epochs=200):
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs 
        self.counter = 0
        self.min_validation_loss = np.inf
        self.stop = False
        self.epoch = 0  
        
    def early_stop(self, validation_loss):
        self.epoch += 1  
        if self.epoch < self.min_epochs:
            return False

        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
        if self.counter >= self.patience:
            self.stop = True 
        return self.stop


def get_dim_from_type(feat_dists):
    return sum(d[1] for d in feat_dists)  # Returns the number of parameters needed to base_model the distributions in feat_dists
    
def get_activations_from_types(x, feat_dists, min_val=1e-3, max_std=15.0, max_alpha=2, max_k=1000.0):
    # Ancillary function that gives the correct torch activations for each data distribution type
    # Example of type list: [('bernoulli', 1), ('gaussian', 2), ('categorical', 5)]
    # (distribution, number of parameters needed for it)
    index_x = 0
    out = []
    for index_type, type in enumerate(feat_dists):
        dist, num_params = type
        if dist == 'gaussian':
            out.append(torch.tanh(x[:, index_x, np.newaxis]) * 5)  # Mean: from -inf to +inf
            out.append((torch.sigmoid(x[:, index_x + 1, np.newaxis]) * (max_std - min_val) + min_val) / (10 * max_std))
        elif dist == 'weibull':
            out.append((torch.sigmoid(x[:, index_x, np.newaxis]) * (max_std - min_val) + min_val) / max_std * max_alpha)
            out.append(
                (torch.sigmoid(x[:, index_x + 1, np.newaxis]) * (max_std - min_val) + min_val) / max_std * max_k)
            # K: (min_val, max_k + min_val)
        elif dist == 'bernoulli':
            out.append(torch.sigmoid(x[:, index_x, np.newaxis]) * (1.0 - 2 * min_val) + min_val)
            # p: (min_val, 1-min_val)
        elif dist == 'categorical':  # Softmax activation: NANs appear if values are close to 0,
            # so use min_val to prevent that
            vals = torch.tanh(x[:, index_x: index_x + num_params]) * 10.0  # Limit the max values
            out.append(torch.softmax(vals, dim=1))  # probability of each categorical value
            check_nan_inf(out[-1], 'Categorical distribution')
        else:
            raise NotImplementedError('Distribution ' + dist + ' not implemented')
        index_x += num_params
    return torch.cat(out, dim=1)