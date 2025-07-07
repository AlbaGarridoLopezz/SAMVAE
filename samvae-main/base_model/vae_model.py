# Author: Alba Garrido López
# Email: alba.garrido.lopez@upm.es

# Import libraries
import torch
import numpy as np
from .vae_utils import EarlyStopper
from .vae_modules import LatentSpaceGaussian, Encoder, Decoder, LogLikelihoodLoss

class VariationalAutoencoderMultimodal(torch.nn.Module):
    def __init__(self, params):
        super(VariationalAutoencoderMultimodal, self).__init__()

        # Hyperparameters
        self.input_dim = params['input_dim']  
        self.feat_distributions = params['feat_distributions']  
        self.modes =  params['modes']
        self.N_wsi = params['N_wsi']
        self.image_resolution = params['image_resolution']
        self.final_image_resolution = params['final_image_resolution']
        self.hyperparameter_optimization = params['hyperparameter_optimization']
        self.optimizer_with_clinical = params['optimizer_with_clinical']
        self.n = len(self.modes[:-1]) # Exclude the time-to-event-modality
        assert len(self.feat_distributions) == len(self.input_dim) 
        
        # To ensure compatibility with hyperparameter optimization
        to_list = lambda x: [x] if self.hyperparameter_optimization else x 
        
        if self.hyperparameter_optimization and self.optimizer_with_clinical:
            self.hidden_size = params['hidden_size']
            self.latent_dim = params['latent_dim']
        else:
            self.hidden_size = to_list(params['hidden_size'])
            self.latent_dim = to_list(params['latent_dim'])
         
        # Calculate the total dimension of the latent space   
        self.latent_dim_total =  sum(
            sum(ld) if isinstance(ld, list) else ld for ld in self.latent_dim
        )

        self.early_stop = params['early_stop']
        self.normalization_loss = params['normalization_loss']
        self.patience = params['patience']
        
        assert len(self.input_dim) ==  len(self.modes) == len(self.latent_dim) + 1 == len(self.hidden_size) +1 == self.n + 1

        # Standard VAE modules. Encoders and Latent Space
        self.latent_space = [LatentSpaceGaussian(l) for l in self.latent_dim]
        self.Encoders = torch.nn.ModuleList()
        self.Decoders = torch.nn.ModuleList()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
         
        # Initialize encoders and decoders for each modality
        for i in range(self.n):
            # Create an encoder for each modality
            self.Encoders.append(Encoder(input_dim= self.input_dim[i], hidden_dim= self.hidden_size[i],
                                         output_dim= self.latent_space[i].latent_params, modes= self.modes[i], N_wsi = self.N_wsi, 
                                         image_resolution = self.image_resolution, final_image_resolution = self.final_image_resolution, device= self.device)) 
            
            # Create a decoder for each modality
            self.Decoders.append(Decoder(latent_dim= self.latent_dim_total, hidden_size= self.hidden_size[i],
                                         feat_dists= self.feat_distributions[i], modes= self.modes[i], N_wsi = self.N_wsi, 
                                         image_resolution = self.image_resolution, final_image_resolution = self.final_image_resolution, device= self.device))
        
            
        # Define losses, one for each modality
        self.rec_losses = []
        for feat_dist, mode in zip(self.feat_distributions[:-1], self.modes[:-1]): # Exclude the time-to-event-modality
            self.rec_losses.append(LogLikelihoodLoss(feat_dist, mode, normalization_loss=self.normalization_loss))

        # Early stopping  
        if self.early_stop:
            self.early_stopper = EarlyStopper(patience=self.patience, min_delta=2)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()