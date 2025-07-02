# Author: Alba Garrido López
# Email: alba.garrido.lopez@upm.es

# Packages to import
import torch 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from base_model.vae_model import VariationalAutoencoderMultimodal
from validation import bern_conf_interval, obtain_c_index
from base_model.vae_utils import check_nan_inf
from base_model.vae_modules import Decoder, LogLikelihoodLossWithCensoring

class Multimodal_SAMVAE(VariationalAutoencoderMultimodal):
    def __init__(self, params):
        # Initialize Savae parameters and modules
        super(Multimodal_SAMVAE, self).__init__(params)

        # Parameters
        self.time_dist =  params['time_dist']
        self.time_hidden_size = params['time_hidden_size']
        self.max_t = 2 * params['max_t']
        self.device = params['device']
        self.time_mode = params['time_mode']
        self.Time_Decoder = Decoder(latent_dim=self.latent_dim_total,
                                    hidden_size= self.time_hidden_size,  
                                    feat_dists=[self.time_dist],
                                    modes=self.time_mode,
                                    max_k=self.max_t,
                                    N_wsi = self.N_wsi,
                                    device=self.device)
        # Send all to device
        self.to(self.device)
        
        # Define losses
        self.time_loss = LogLikelihoodLossWithCensoring(self.time_dist)

    def forward(self, input_data):
        # Initialize lists to store the outputs of each modality
        latent_outputs = []  
        latent_params_list = [] 
        z_list = []  
        cov_params_list = []
        
        # First loop: Get concatenated z from all modalities
        for i, data in enumerate(input_data):
            # Encoder: Pass the data of modality 'i' through its respective Encoder
            latent_output = self.Encoders[i](data)  # Encoder output (B, N, L)  
            latent_outputs.append(latent_output)  
            
            # mu, logvar
            latent_params = self.latent_space[i].get_latent_params(latent_output) # (B, N, L) 
            latent_params_list.append(latent_params)

            # Sample the latent space using reparameterization
            z = self.latent_space[i].sample_latent(latent_params)   # (B, N, L)  
            check_nan_inf(z, f'Latent space modality {i}')          
            z_list.append(z)
        z_all = torch.cat(z_list, dim=-1)
        z_agg = torch.mean(z_all, axis=1)
        z_all = z_all.view((z_all.shape[0] * z_all.shape[1], z_all.shape[2]))   
        
        # Pass the corresponding z through each decoder
        for i, mode in enumerate(self.modes):  
            if mode in ['clinical', 'omic']:
                z_input = z_agg
            elif mode == 'wsi':
                z_input = z_all
            elif mode == 'time_event':
                continue   
            else:
                raise ValueError(f"Modo desconocido: {mode}")

            cov_params = self.Decoders[i](z_input)  
            check_nan_inf(cov_params, f'Decoder modality {i}')  
            cov_params_list.append(cov_params)
        
        # Time decoder
        time_params = self.Time_Decoder(z_agg)
        check_nan_inf(time_params, 'Time Decoder')
    
        return {
            'z_aggr': z_agg,  
            'z_img': z_all,    
            'cov_params': cov_params_list,  
            'latent_params': latent_params_list, 
            'time_params': [time_params] 
        }

    
    def calculate_risk(self, time_train, x_val, censor_val, time_val):
        # Calculate risk (CDF) at the end of batch training
        times = torch.unique(time_train[0]).to(self.device)
        out_data = self(x_val)
        time_params = out_data['time_params'][0]
        pred_risk = torch.zeros((time_params.shape[0], len(times)), device = self.device)
    
        for sample in range(pred_risk.shape[0]):
            if self.time_dist[0] == 'weibull':
                alpha, lam = time_params[sample, 0], time_params[sample, 1]
                pred_risk[sample, :] = 1 - torch.exp(-torch.pow(times / lam, alpha))
                
        # Convert to NumPy only for computing the C-index and IBS
        times_np = times.cpu().numpy()
        pred_risk_np = pred_risk.detach().cpu().numpy()

        # Compute C-index and IBS
        ci_list, ibs_list = [], []
        for i, (x, censor) in enumerate(zip(time_val, censor_val)):
            x_np = x.cpu().numpy().flatten()
            censor_np = censor.cpu().numpy().flatten()
            # Create DataFrame for survival analysis calculations
            surv_df = pd.DataFrame(1 - pred_risk_np.T, index=times_np)
            ci, ibs = obtain_c_index(surv_df, x_np, censor_np)
            ci_list.append(ci)
            ibs_list.append(ibs)
    
        # Compute confidence intervals for each value in ci_list and ibs_list
        ci_confident_intervals = [bern_conf_interval(len(x), ci) for x, ci in zip(x_val, ci_list)]
        ibs_confident_intervals = [bern_conf_interval(len(x), ibs, ibs=True) for x, ibs in zip(x_val, ibs_list)]

         
        return ci_confident_intervals, ibs_confident_intervals, pred_risk_np, times_np
    

    def fit_epoch(self, data, optimizer, batch_size=64):
        epoch_results = {'loss_tr': 0.0, 'loss_va': 0.0, 'kl_tr': 0.0, 'kl_va': 0.0, 'll_cov_tr': 0.0, 'll_cov_va': 0.0,
                         'll_time_tr': 0.0, 'll_time_va': 0.0, 'ci_va': 0.0, 'ibs_va': 0.0}
        cov_train, mask_train, time_train, censor_train, cov_val, mask_val, time_val, censor_val = data
        
     
        # Train epoch
        n_batches = int(np.ceil(len(cov_train[0]) / batch_size))  # Based on the first tensor
        cov_train_batch = []
        mask_train_batch = []
        for batch in range(n_batches):
            # Compute indices for the batch
            index_init = batch * batch_size
            index_end = (batch + 1) * batch_size
            cov_train_batch = [tensor[index_init:min(index_end, len(tensor))].to(self.device) for tensor in cov_train]
            mask_train_batch = [tensor[index_init:min(index_end, len(tensor))].to(self.device) for tensor in mask_train]
            time_train_batch = [tensor[index_init:min(index_end, len(tensor))].to(self.device) for tensor in time_train]
            censor_train_batch = [tensor[index_init:min(index_end, len(tensor))].to(self.device) for tensor in censor_train]
            
            self.train()
        
            # Generate output params
            out = self(cov_train_batch)

            # Compute losses
            optimizer.zero_grad()
            
            # Compute KL loss for multiple latent parameters
            loss_kl = self.latent_space[0].kl_loss(out['latent_params'])  # Do it once, as it takes into account all individual latent spaces
            
            # Now, to compute the covariates loss, apply each rec_loss function individually.
            loss_cov_total = 0.0
           
            for modality_index, rec_loss_func in enumerate(self.rec_losses):
                # Apply the loss function to the corresponding modality
                loss_cov = rec_loss_func(out['cov_params'][modality_index], cov_train_batch[modality_index], mask_train_batch[modality_index])
                loss_cov_total += loss_cov  # Sum all modality losses
            loss_time = self.time_loss(out['time_params'], time_train_batch, censor_train_batch)
            
            # Compute total loss
            loss = loss_kl + loss_cov + loss_time
            loss.backward()
            optimizer.step()

            # Save data
            epoch_results['loss_tr'] += loss.item()
            epoch_results['kl_tr'] += loss_kl.item()
            epoch_results['ll_cov_tr'] += loss_cov.item() if isinstance(loss_cov, torch.Tensor) else loss_cov
            epoch_results['ll_time_tr'] += loss_time.item() if isinstance(loss_time, torch.Tensor) else loss_time

            # Validation step
            self.eval()
       
            with torch.no_grad():
                cov_val = [tensor.to(self.device) for tensor in cov_val]
                out = self(cov_val)
                loss_kl = self.latent_space[0].kl_loss(out['latent_params'])
                loss_cov_total = 0.0
                for modality_index, rec_loss_func in enumerate(self.rec_losses):
                    loss_cov = rec_loss_func(out['cov_params'][modality_index], cov_val[modality_index], mask_val[modality_index])
                    loss_cov_total += loss_cov

                # Compute time loss and sum its components
                loss_time = self.time_loss(out['time_params'], time_val, censor_val)
                loss = loss_kl + loss_cov + loss_time
                
                # Save data
                epoch_results['loss_va'] += loss.item()
                epoch_results['kl_va'] += loss_kl.item()
                epoch_results['ll_cov_va'] += loss_cov.item() if isinstance(loss_cov, torch.Tensor) else loss_cov
                epoch_results['ll_time_va'] += loss_time.item() if isinstance(loss_time, torch.Tensor) else loss_time
            

        if self.early_stop:
            self.early_stopper.early_stop(epoch_results['loss_va'])

        return epoch_results
    
    def fit(self, data, train_params):
        training_stats = {'loss_tr': [], 'loss_va': [], 'kl_tr': [], 'kl_va': [], 'll_cov_tr': [], 'll_cov_va': [],
                          'll_time_tr': [], 'll_time_va': [], 'ci_va': [], 'ibs_va': []}

        optimizer = torch.optim.Adam(self.parameters(), lr=train_params['lr'], betas=train_params['betas'])

        epochs_ci = []
        epochs_ibs = []
        train_losses = []
        valid_losses = []
        best_model = None
        best_loss = np.inf
        best_ci = 0.0
        best_ibs = 1.0
        
        # without time or event
        cov_train  = [d[0] for d in data[:-1]]  
        mask_train_cov = [d[1] for d in data[:-1]]  
        cov_val = [d[2] for d in data[:-1]]  
        mask_val_cov = [d[3] for d in data[:-1]]  

        # Separate time and event from the last data tensor
        time_train = [data[-1][0][:, 0]] 
        censor_train = [data[-1][0][:, 1]]  
        time_val = [data[-1][2][:, 0]]  
        censor_val = [data[-1][2][:, 1]] 
        
        ep_data = (cov_train, mask_train_cov, time_train, censor_train, cov_val, mask_val_cov, time_val, censor_val)  
        
        cov_val = [tensor.to(self.device) for tensor in cov_val]

        
        for epoch in range(train_params['n_epochs']):  
            # Fit epoch
            epoch_results = self.fit_epoch(ep_data, optimizer, batch_size=train_params['batch_size'])
            
            # Compute metrics
            epoch_results['ci_va'], epoch_results['ibs_va'], _, _ = self.calculate_risk(time_train, cov_val, censor_val, time_val) 
            
            # Save training stats
            train_losses.append(epoch_results['loss_tr'])
            valid_losses.append(epoch_results['loss_va'])
            epochs_ci.append(epoch_results['ci_va'])
            epochs_ibs.append(epoch_results['ibs_va'])
            for key in epoch_results.keys():
                training_stats[key].append(epoch_results[key])

            if epoch % 50 == 0:
                print('Iteration = ', epoch,
                      '; train loss = ', '{:.2f}'.format(epoch_results['loss_tr']),
                      '; val loss = ', '{:.2f}'.format(epoch_results['loss_va']),
                      '; ll_cov_tr = ', '{:.2f}'.format(np.sum(epoch_results['ll_cov_tr'])),
                      '; ll_cov_va = ', '{:.2f}'.format(np.sum(epoch_results['ll_cov_va'])),
                      '; kl_tr = ', '{:.2f}'.format(epoch_results['kl_tr']),
                      '; kl_va = ', '{:.2f}'.format(epoch_results['kl_va']),
                      '; ll_time_tr = ', '{:.2f}'.format(np.sum(epoch_results['ll_time_tr'])),
                      '; ll_time_va = ', '{:.2f}'.format(np.sum(epoch_results['ll_time_va'])),
                      '; C-index = ', '{:.4f}'.format(np.mean(epoch_results['ci_va'])),
                      '; IBS = ', '{:.4f}'.format(np.mean(epoch_results['ibs_va'])))

            # Save best model
            if epoch_results['loss_va'] < best_loss:
                best_loss = epoch_results['loss_va']
                best_model = self.state_dict()
                best_ci = epoch_results['ci_va']
                best_ibs = epoch_results['ibs_va']

            if self.early_stop and self.early_stopper.stop:
                print('[INFO] Training early stop')
                break

        # Set the best model as the current model
        self.load_state_dict(best_model)
        training_stats['ci_va'].append(best_ci)
        training_stats['ibs_va'].append(best_ibs)
       
        return training_stats