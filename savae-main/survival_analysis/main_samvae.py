# Author: Alba Garrido López
# Email: alba.garrido.lopez@upm.es

# Packages to import
import os
import sys
import time
import torch
import json
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import pickle 
import warnings
import random 
from colorama import Fore, Style
from joblib import Parallel, delayed

warnings.filterwarnings("ignore") 
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
# Local Imports 
from multimodal_samvae import Multimodal_SAMVAE
from multimodal_samvae_cr import Multimodal_SAMVAE_cr
from data import split_cv_data_multimodal
from tables import (
    get_table, get_table_cr, get_table_no_hpo,
    get_table_cr_no_hpo, get_table_no_hpo_with_pvals,
    get_table_cr_no_hpo_with_pvals
)
from plots import (
    plot_multiple_models_vs_km, plot_model_losses, plot_survival,
    plot_model_C_index, plot_model_IBS, plot_CIF,
    plot_interactive_models_vs_km, load_existing_fold_results, find_log_names
)
from validation import (
    load_results_and_labels, get_fold_best_seed_results,
    get_fold_best_seed_results_cr
)
from utils import (
    run_args, create_output_dir, save, check_file,
    print_experiment_config, load_datasets
)

# ------------------------------------------------------------------------------------------------------
#                                   TRAINING SAMVAE
# ------------------------------------------------------------------------------------------------------
    
# Trains a multimodal survival model using the provided modalities (e.g., clinical, omic, and/or image data). 
# This function handles data loading, model training, and evaluation across multiple seeds for reproducibility.
# The training setup depends on the configuration specified in args. 

def train_multimodal(data, feat_distributions, params, seed, output_dir, args, fold):
    start_time = time.time()
    # Prepare samvae params
    data_time = data[-1]   
   
    if args['competing_risks']:  # Define max_t based on competing risks
        unique_events = torch.unique(torch.cat([data_time[0][:, 1], data_time[2][:, 1]])).tolist()  # Get unique events from both training and validation data
        unique_events = [e for e in unique_events if e != 0] # Exclude censored data (event = 0)
        labels = [str(i) for i in range(len(unique_events))]
        time_dist =  [args['time_distribution'] for _ in range(len(unique_events))]
        # Get max observed time for non-censored events
        max_t = [max(torch.max(data_time[0][data_time[0][:, 1] == event][:, 0]).item() if (data_time[0][:, 1] == event).any() else float('-inf'),
                torch.max(data_time[2][data_time[2][:, 1] == event][:, 0]).item() if (data_time[2][:, 1] == event).any() else float('-inf'))
            for event in unique_events]
        # Replace '-inf' with 0 for events not found in the dataset
        max_t = [t if t != float('-inf') else 0 for t in max_t]
        
    else: # Define max_t for survival analysis
        max_t = max([torch.max(data_time[0][:, 0]).item(), torch.max(data_time[2][:, 0]).item()]) 
        time_dist =  args['time_distribution']
      
    model_params = {'feat_distributions': feat_distributions,
                    'latent_dim': params['latent_dim'],
                    'hidden_size': params['hidden_size'],
                    'dropout_prop': params['dropout_prop'],
                    'input_dim': [len(dist) for dist in feat_distributions],
                    'max_t': max_t,
                    'time_mode': args['time_mode'],
                    'time_dist': time_dist,
                    'time_hidden_size': args['time_hidden_size'],
                    'normalization_loss': args['normalization_loss'],
                    'early_stop': args['early_stop'],
                    'modes': args['modes'],
                    'patience': args['patience'],
                    'N_wsi': args['N_wsi'],
                    'image_resolution': args['image_resolution'],
                    'final_image_resolution': args['final_image_resolution'],
                    'competing_risks': args['competing_risks'],
                    'hyperparameter_optimization': args['hyperparameter_optimization'],
                    'optimizer_with_clinical': args['optimizer_with_clinical'],
                    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')}
    
    # Competing risks
    if args['competing_risks']:
        model_params['labels'] = labels
        model = Multimodal_SAMVAE_cr(model_params)
    else:
        model = Multimodal_SAMVAE(model_params)
        
    model_path = str(model_params['latent_dim']) + '_' + str(model_params['hidden_size'])
    seed_path = model_path + '/seed_' + str(seed)
    log_name = output_dir + seed_path + '/model_fold_' + str(fold)

    # Train model
    train_params = {'n_epochs': args['n_epochs'], 'batch_size': args['batch_size'],
                    'lr': args['lr'], 'betas': args['betas'], 'path_name': log_name}
    
    training_results = model.fit(data, train_params)
    
    # Plot losses
    path_name = str(train_params['path_name'])
    for loss in ['loss_', 'll_cov_', 'kl_', 'll_time_']:
        plot_model_losses(training_results[loss + 'tr'], training_results[loss + 'va'],
                          path_name + '_' + loss + 'losses.png', 'Train and Validation ' + loss + 'losses')
 
    # Save base_model information
    model.save(log_name + '.pt')
    model_params.update(train_params)
    model_params.update(training_results)
    save(model_params, log_name + '.pickle')
    
    # Plot C-index and IBS  
    if not args['competing_risks']:  
        path_name = str(train_params['path_name'])
        c_index_path = path_name + '_' + 'c_index.png'
        ibs_path = path_name + '_' + 'ibs.png'
        plot_model_C_index(ci_val=model_params['ci_va'], fig_path=c_index_path)
        plot_model_IBS(model_params['ibs_va'], fig_path= ibs_path)
        plot_survival()
        
    # Plot CIF  
    else:
        CIF_path =  str(train_params['path_name'])  
        plot_CIF(cif_savae=training_results['final_cif'], df_train=[data[-1][0][:, 0]], df_val=[d[2] for d in data[:-1]], 
                 labels=model_params['labels'], dir_path=CIF_path)
 
    # Load validation results
    ci_results = model_params['ci_va']
    ibs_results = model_params['ibs_va']
    end_time = time.time() - start_time
    print(f"\n[INFO] Total Training time:" + str(end_time))
    
    if args['competing_risks']:
        return ci_results, ibs_results, params, seed, fold, end_time, labels  # Labels of risks
    else:
        return ci_results, ibs_results, params, seed, fold, end_time 
    
    
# ------------------------------------------------------------------------------------------------------
#                                   MAIN SAMVAE
# ------------------------------------------------------------------------------------------------------
 
def main():
    print(Fore.RED + '\n\n-------- MULTIMODAL VAE  --------' + Style.RESET_ALL)
    
    # Environment configuration
    task = 'savae_sa'
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, 'survival_analysis', 'datasets.json')
    all_results_for_plot = []
    
    # Read configurations from datasets.json  
    with open(config_path, 'r') as f:
        configurations = json.load(f)
     
    for config in configurations:
        args = run_args(task, config)
        create_output_dir(task, args)
        config_file_path =  os.path.join(args['output_dir'], "experiment_config.txt")
        print_experiment_config(args, save_path=config_file_path)
        
        
        modes_used = ", ".join(args['modes'][:-1]) # Without considering Time and Event
        datasets_used = ", ".join(args['datasets'][:-1])  # Without considering Time and Event

        if args['train']:
            print('\n---- TRAINING----')
            
            # Datasets and modes
            all_tensors = load_datasets(args)
                
            if len(all_tensors) > 1:
                print(Fore.GREEN + "Training multimodal model..." + Style.RESET_ALL)
            else:
                print(Fore.GREEN + "Training unimodal model..." + Style.RESET_ALL)
            output_dir = args['output_dir'] 

            # Prepare data
            cv_data_multimodal, feat_distributions_multimodal = split_cv_data_multimodal(
                all_tensors, args['n_folds'], time_dist=args['time_distribution'])
            
            # Train model
            train_tasks = []
            existing_models = [] 

            for params in args['param_comb']:
                for seed in range(args['n_seeds']):
                    for fold in range(args['n_folds']):
                        model_params = f"{params['latent_dim']}_{params['hidden_size']}"
                        checkpoint_path = os.path.join(output_dir, model_params, f"seed_{seed}", f"model_fold_{fold}.pt")

                        if os.path.exists(checkpoint_path):
                            print(f"Model already exists at {checkpoint_path}, skipping...")
                            existing_models.append((model_params, params, seed, fold, checkpoint_path))
                            continue
                        
                        train_tasks.append(
                            delayed(train_multimodal)(
                                [cv_data_multimodal[dataset_idx][fold] for dataset_idx in range(len(cv_data_multimodal))],
                                [feat_distributions_multimodal[dataset_idx] for dataset_idx in range(len(feat_distributions_multimodal))],
                                params, seed, output_dir, args, fold
                            )
                        )            
            models_results = Parallel(n_jobs=args['n_threads'], verbose=10)(train_tasks)
             

            # Create dictionary to save results
            results_path = os.path.join(output_dir, 'results.pkl')
            if not os.path.exists(results_path):
                results = {}
                for params in args['param_comb']:
                    model_params = str(params['latent_dim']) + '_' + str(params['hidden_size'])
                    results[model_params] = {}
                    for seed in range(args['n_seeds']):
                        results[model_params][seed] = {}
                        for fold in range(args['n_folds']):
                            results[model_params][seed][fold] = {'ci': [], 'ibs': []}
                     
                    for res in models_results:
                        params = res[2]
                        seed = res[3]
                        fold = res[4]
                        labels = res[-1]
                        model_params = str(params['latent_dim']) + '_' + str(params['hidden_size'])
                        results[model_params][seed][fold]['ci'] = res[0]
                        results[model_params][seed][fold]['ibs'] = res[1]
                        results[model_params][seed][fold]['time'] = res[5]
                        
                
                    # Create a dictionary to store results if the model was not saved correctly and needs to be saved again
                    if args['results_pkl']:
                        load_existing_fold_results(existing_models, results)
                
                        with open(results_path, 'wb') as f:
                            pickle.dump(results, f)
            
                    # Save best results
                    save(results, output_dir + 'results.pkl') 
                    
                if args['competing_risks']:
                    save(results, output_dir + 'results_cr.pkl') 
                    #save(labels, output_dir + 'labels.pkl') 
                    best_results, best_results_per_param, best_results_ibs  = get_fold_best_seed_results_cr(results, args['param_comb'], args['n_seeds'], args['n_folds'])
                    save(best_results, output_dir + 'best_results_cr.pkl') 
                    save(best_results_ibs, output_dir + 'best_results_ibs_cr.pkl')
                    save(best_results_per_param, output_dir + 'best_results_per_param.pkl')
                else:
                    #save(results, output_dir + 'results.pkl')
                    best_results, best_results_per_param, best_results_ibs  = get_fold_best_seed_results(results, args['param_comb'], args['n_seeds'], args['n_folds'])
                    save(best_results, output_dir + 'best_results.pkl') 
                    save(best_results_ibs, output_dir + 'best_results_ibs.pkl')
                    save(best_results_per_param, output_dir + 'best_results_per_param.pkl')
                    
           
        # ------------------------------------------------------------------------------------------------------
        #                                    RESULTS: TABLES  
        # ------------------------------------------------------------------------------------------------------
        
        # Obtain final tables 
        print('\n\n---- SAMVAE (SURVIVAL ANALYSIS MULTIMODAL VAE) RESULTS----')
        print(f"Modes used: {Fore.CYAN}{modes_used}{Style.RESET_ALL}")
        print(f"Datasets used: {Fore.CYAN}{datasets_used}{Style.RESET_ALL}") 
        output_dir = args['output_dir']
        
        # Load results and labels and generate intermediate tables
        if args['competing_risks']:
            results, best_results, best_results_per_param, best_results_ibs, labels = load_results_and_labels(args)   
            if args['hyperparameter_optimization']:
                get_table_cr( args, datasets_used, modes_used, best_results_per_param, labels)
            else: 
                get_table_cr_no_hpo(args, datasets_used, modes_used, best_results, best_results_ibs, labels)    
        else:
            results, best_results, best_results_per_param, best_results_ibs = load_results_and_labels(args) 
            if args['hyperparameter_optimization']:
                get_table(results, args, datasets_used, modes_used, best_results_per_param, best_results, best_results_ibs)
            else:
                get_table_no_hpo(args, datasets_used, modes_used, best_results, best_results_ibs) 
                
        # Generate final tables with p-values    
        if args['final_tables']:
            if args['competing_risks']:
                get_table_cr_no_hpo_with_pvals(args, datasets_used, modes_used, labels)
            else: 
                get_table_no_hpo_with_pvals(args, datasets_used, modes_used)
       
        # ------------------------------------------------------------------------------------------------------
        #                                PLOTS: Survival Curves & Model Comparison 
        # ------------------------------------------------------------------------------------------------------
        
        # For individual survival plots: each model vs Kaplan-Meier   
        if args['n_seeds'] == 10 and not args['final_plots']:
            
            # Get path to experiment and log name
            output_dir = args['output_dir']
            experiment_path = os.path.normpath(os.path.join(output_dir, '..'))
            log_name = find_log_names(experiment_path)
            log_name = log_name[0] 
            
            # Load model parameters and weights
            model_params = check_file(log_name + '.pickle', 'Model does not exist.')
            model_params['image_resolution'] = 128
            model_params['final_image_resolution'] = 16

            # Initialize appropriate model (with or without competing risks)
            model = Multimodal_SAMVAE_cr(model_params) if args['competing_risks'] else Multimodal_SAMVAE(model_params)
            model.load_state_dict(torch.load(log_name + '.pt'))
            model.eval()

            # Plot individual survival curves
            plot_survival(model, args, log_name, all_results_for_plot, fold=0)
    
    # For final plots (comparison of all models vs Kaplan-Meier)     
    if args['n_seeds'] == 10 and args['final_plots']:      
        multiple_survival_path = os.path.normpath(os.path.join(args['plots_output_dir'], '..', '..'))
        # Static and interactive model-vs-KM comparisons
        plot_multiple_models_vs_km(multiple_survival_path, args['competing_risks']) 
        plot_interactive_models_vs_km(multiple_survival_path)
       
         
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()