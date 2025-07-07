# Author: Alba Garrido López
# Email: alba.garrido.lopez@upm.es

# Import libraries
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from .vae_utils import get_dim_from_type, get_activations_from_types, check_nan_inf


class LatentSpaceGaussian(object):
    def __init__(self, latent_dim):
        self.latent_dim = latent_dim
        self.latent_params = 2 * latent_dim  # Two parameters are needed for each Gaussian distribution

    def get_latent_params(self, x):
        x = x.view((x.shape[0], x.shape[1], 2, self.latent_dim))
        mu = x[:, :,  0, :]
        log_var = x[:, :,  1, :]
        return mu, log_var

    def sample_latent(self, latent_params):
        mu, log_var = latent_params
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)   # 'randn_like' as we need the same size
        z = mu + (eps * std)  # sampling as if coming from the input space
        return z
    
    def kl_loss(self, latent_params):
        total_kl = 0
        for mu, log_var in latent_params:
            kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())  # Kullback-Leibler divergence
            kl = kl / mu.shape[0]
            total_kl += kl  
        return total_kl   
 

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, modes, N_wsi, image_resolution, final_image_resolution, device, grad_clip=1000.0, latent_limit=10.0):
        super(Encoder, self).__init__()
        # Note that they are Gaussian encoders: latent_dim is the number of gaussian
        # distributions (hence, we need 2 * latent_dim parameters!)
        self.modes = modes
        self.latent_limit = latent_limit
        self.device = device
        self.N_wsi = N_wsi
        self.image_resolution = image_resolution
        self.final_image_resolution = final_image_resolution
        
        # Tabular data --> MLP
        if self.modes in ['clinical', 'omic', 'time_event']:
            self.enc1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
            self.enc2 = nn.Linear(in_features=hidden_dim, out_features=output_dim)
            self.grad_clip = grad_clip
            
        # Image data --> CNN
        elif self.modes == 'wsi':
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=hidden_dim[0], kernel_size=3, stride=1, padding=1)
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(in_channels=hidden_dim[0], out_channels=hidden_dim[1], kernel_size=3, stride=1, padding=1)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) 
            self.conv3 = nn.Conv2d(in_channels=hidden_dim[1], out_channels=hidden_dim[2], kernel_size=3, stride=1, padding=1)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  
            self.fc = nn.Linear(hidden_dim[-1] * final_image_resolution * final_image_resolution, output_dim) 
        else:
            raise RuntimeError('Encoder mode not recognized : ' + str(self.modes))
       
          
    def forward(self, inp):
        # B = Batch size
        # N = Number of images per patient 
        # C = Channels
        # H = Height (images)
        # W = Width (images)
        # L = Latent dimension (2*L)
        
        if self.modes in ['clinical', 'omic', 'time_event']:
            x = self.enc1(inp)
            if x.requires_grad:
                x.register_hook(lambda x: x.clamp(min=-self.grad_clip, max=self.grad_clip))
            x = F.relu(x)
            x = self.enc2(x)
            if x.requires_grad:
                x.register_hook(lambda x: x.clamp(min=-self.grad_clip, max=self.grad_clip))
            x = torch.tanh(x) * self.latent_limit # (B,L) 
            x = x.unsqueeze(1).repeat(1, self.N_wsi, 1)  # (B, N, L)  Project into latent space and replicate N_wsi times
            return x.to(inp.device) # Encoder output: (B, N, L)
        
        elif self.modes == 'wsi':
            batch_size, num_images, channels, height, width = inp.shape  # (B, N, C, H, W)    
            inp = inp.view(batch_size * num_images, channels, height, width)  # (B*N, C, H, W) 
            x = F.relu(self.conv1(inp))
            x = self.pool1(x)
            x = F.relu(self.conv2(x))
            x = self.pool2(x)
            x = F.relu(self.conv3(x))
            x = self.pool3(x)
            x = x.view(x.size(0), -1)  # (B*N, L)  
            x = self.fc(x)
            
            # Project into latent space and reshape back to (B, N, L)
            x = torch.tanh(x) * self.latent_limit
            x = x.view(batch_size, num_images, -1)
            return x # Encoder output: (B, N, L)


class Decoder(nn.Module):
    def __init__(self, latent_dim, feat_dists, modes, device, hidden_size, N_wsi, image_resolution= 128, final_image_resolution=16, max_k=10000.0, dropout_p=0.2):
        super(Decoder, self).__init__()
        self.feat_dists = feat_dists
        self.modes = modes
        self.out_dim = get_dim_from_type(self.feat_dists) 
        self.hidden_size = hidden_size
        self.device = device
        self.N_wsi = N_wsi
        if self.modes in ['clinical', 'omic', 'time_event']:
            self.dec1 = nn.Linear(in_features=latent_dim, out_features=hidden_size)
            self.dec2 = nn.Linear(in_features=hidden_size, out_features=self.out_dim)
            self.max_k = max_k
            self.dropout = nn.Dropout(dropout_p)
        elif self.modes == 'wsi':
            self.image_resolution = image_resolution
            self.final_image_resolution = final_image_resolution
            self.fc = nn.Linear(latent_dim, hidden_size[-1] * self.final_image_resolution * self.final_image_resolution)  
            self.deconv1 = nn.ConvTranspose2d(in_channels=hidden_size[2], out_channels=hidden_size[1], kernel_size=3, stride=2, padding=1, output_padding=1) 
            self.deconv2 = nn.ConvTranspose2d(in_channels=hidden_size[1], out_channels=hidden_size[0], kernel_size=3, stride=2, padding=1, output_padding=1) 
            self.deconv3 = nn.ConvTranspose2d(in_channels=hidden_size[0], out_channels=3, kernel_size=3, stride=2, padding=1, output_padding=1)  
        else:
            raise RuntimeError('Decoder mode not recognized: ' + str(self.modes))

    def forward(self, z):
        z = z.to(self.device) # (B, L)
        if self.modes in ['clinical', 'omic', 'time_event']:
            x = F.relu(self.dec1(z))
            x = self.dropout(x)
            x = self.dec2(x)
            x = get_activations_from_types(x, self.feat_dists, max_k=self.max_k)
            return x.to(self.device)
        
        elif self.modes == 'wsi':
            N_wsi = self.N_wsi
            batch_size_N_wsi= z.size(0)
            batch_size = int(batch_size_N_wsi//N_wsi) # (B*N, L) 
            z = self.fc(z)   
            z = z.view(batch_size_N_wsi, self.hidden_size[-1], self.final_image_resolution, self.final_image_resolution)  # (B*N, C, H, W)  
            x = F.relu(self.deconv1(z))
            x = F.relu(self.deconv2(x))
            x = torch.sigmoid(self.deconv3(x)) 
            x = x.view(batch_size, N_wsi, 3, self.image_resolution, self.image_resolution)   
            return x.to(self.device)  # Decoder output: (B, N, C, H, W)
        
        
class LogLikelihoodLoss(nn.Module):
    def __init__(self, feat_dists, modes, normalization_loss):
        super(LogLikelihoodLoss, self).__init__()
        self.feat_dists = feat_dists
        self.modes = modes
        self.normalization_loss = normalization_loss
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, inputs, targets, imp_mask):
        index_x = 0 # Index of parameters (do not confound with index_type below, which is the index of the
        # distributions!)
        loss_ll = []  # Loss for each distribution will be appended here
        
        # Covariates losses for tabular data
        if self.modes in ['clinical', 'omic']:
            for index_type, type in enumerate(self.feat_dists):
                dist, num_params = type
            
                if dist == 'gaussian':
                    mean = inputs[:, index_x]
                    std = inputs[:, index_x + 1]  
                    ll = - torch.log(torch.sqrt(torch.tensor(2 * np.pi)) * std) - 0.5 * ((targets[:, index_type] - mean) / std).pow(2)
                elif dist == 'bernoulli':
                    p = inputs[:, index_x]
                    ll = targets[:, index_type] * torch.log(p) + (1 - targets[:, index_type]) * torch.log(1 - p)
                elif dist == 'categorical':
                    p = inputs[:, index_x: index_x + num_params]
                    mask = F.one_hot(targets[:, index_type].long(), num_classes=num_params) # These are the indexes whose losses we want to compute
                    ll = torch.log(p) * mask
                else:
                    print(dist)
                    raise NotImplementedError('Unknown distribution to compute loss')
                if torch.isnan(ll).any() or torch.isinf(ll).any():
                    raise RuntimeError(f'NAN o INF detectado en la modalidad {index_type}')
                
                if 0 in imp_mask:
                    if dist == 'categorical':
                        ll *= imp_mask[:, index_type].unsqueeze(1)
                    else:
                        ll *= imp_mask[:, index_type] # Add the imputation effect: do NOT train on outputs with mask=0!

                if self.normalization_loss:
                    loss_ll.append(-torch.sum(ll) / (inputs.shape[0] * targets.shape[1])) # Normalize by the number of patients and the mode dimension
                else:
                    loss_ll.append(-torch.sum(ll) / inputs.shape[0])
                index_x += num_params
                
        # Covariates losses for images      
        elif self.modes == 'wsi': 
            if self.normalization_loss:
                mse_loss = torch.sum(torch.square(targets - inputs)) / (inputs.shape[0]*targets.shape[1]*targets.shape[2]*targets.shape[3]) # Normalize by the number of patients and the mode dimension
            else:
                mse_loss = torch.sum(torch.square(targets - inputs)) / inputs.shape[0]
            loss_ll = [mse_loss]     
        else:
            raise ValueError(f"Unsupported mode: {self.modes}") 
        return sum(loss_ll)

# Survival analysis losses
class LogLikelihoodLossWithCensoring(nn.Module):
    def __init__(self, dist_type):
        super(LogLikelihoodLossWithCensoring, self).__init__()
        self.type = dist_type

    def forward(self, inputs_list, targets_list, risk_list):
        if not isinstance(inputs_list, list):
            inputs_list = [inputs_list]
            targets_list = [targets_list]
            risk_list = [risk_list]

        loss_ll_list = []  
        for inputs, targets, risk in zip(inputs_list, targets_list, risk_list):
            targets = torch.squeeze(targets)
            # Append time losses, follow Liverani paper to account for censorship: hazard is only used when no censoring!
            risk = torch.squeeze(risk)
            if self.type[0] == 'weibull':
                alpha = inputs[:, 0]
                lam = inputs[:, 1]
                surv = - torch.pow(targets / lam, alpha) # Always present
                hazard = torch.log(alpha / lam) + (alpha - 1) * torch.log(targets / lam) # Present only when not censored data!
                loss_ll = - torch.sum(hazard[risk > 0.5]) - torch.sum(surv)
            else:
                raise NotImplementedError('Unknown time distribution to compute loss')
            check_nan_inf(loss_ll, 'Time loss')
            loss_ll_list.append(loss_ll / inputs.shape[0])
        return loss_ll_list[0] if len(loss_ll_list) == 1 else loss_ll_list


# Competing risks losses
class LogLikelihoodLossWithCensoringCompeting(nn.Module):
    def __init__(self, dist_types):
        super(LogLikelihoodLossWithCensoringCompeting, self).__init__()
        self.types = dist_types
        
    def forward(self, inputs, risk_probs, targets, risk):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if isinstance(targets, list):
            targets = targets[0] 
        if isinstance(risk, list):
            risk = risk[0] 
        targets = torch.squeeze(targets).to(self.device)
        # Append time losses, follow Liverani paper to account for censorship: hazard is only used when no censoring!
        risk = risk.squeeze().to(self.device)
        loss_ll = torch.tensor(0.0).to(self.device)

        cif_term = []

        for i in range(len(self.types)):  # For each risk
            label = i + 1  # 0 is for censoring, 1,2,... for the different risks
            if self.types[i][0] == 'weibull':
                alpha = inputs[i][:, 0]
                lam = inputs[i][:, 1]
                log_likelihood = torch.log(alpha / lam) + (alpha - 1) * torch.log(targets / lam) - torch.pow(targets / lam, alpha)
                patients_with_current_risk = (risk == label)
                loss_ll += - torch.sum(log_likelihood[patients_with_current_risk]) - torch.sum(torch.log(risk_probs[patients_with_current_risk, i]))  # Loglikelihood and risk prob term
                cdf_for_this_risk = 1.0 - torch.exp(-torch.pow(targets / lam, alpha))
                patients_censored = (risk == 0)
                cif_term.append(cdf_for_this_risk[patients_censored] * risk_probs[patients_censored, i])
            else:
                raise NotImplementedError('Unknown time distribution to compute loss')
            check_nan_inf(loss_ll, 'Time loss, risk ' + str(label))

        if torch.sum(patients_censored) > 0:
            cif_term = torch.stack(cif_term, dim=1)
            loss_ll += - torch.sum(torch.log(torch.clamp(1.0 - torch.sum(cif_term, axis=1), min=1e-10)))  # The 1e-10 is to ensure that the log is not -inf
            check_nan_inf(loss_ll, 'CIF term')
        
        return loss_ll / inputs[0].shape[0]