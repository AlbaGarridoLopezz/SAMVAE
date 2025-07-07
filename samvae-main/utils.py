# Author: Alba Garrido López
# Email: alba.garrido.lopez@upm.es
 
# Packages to import
import os 
import pickle  
from itertools import product 
from tabulate import tabulate
import torch
  
# Check if file exists
def check_file(path, msg):
    if os.path.exists(path):
        with open(path, 'rb') as file:
            results = pickle.load(file)
        return results
    else:
        raise RuntimeError(msg)
    
# Save dictionary to pickle file
def save(res, path):
    with open(path, 'wb') as handle:
        pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Create dir
def create_output_dir(task, args): 
    if task == 'data_preprocessing':
        output_dirs = [args['clinical_output_dir'], args['omic_output_dir'], args['wsi_output_dir'], args['time_event_output_dir']]
        for dir_path in output_dirs:
            if dir_path:  
                os.makedirs(dir_path, exist_ok=True) 
        for dataset_name in args['clinical_datasets']:
            dir_path = os.path.join(args['clinical_output_dir'], dataset_name)
            os.makedirs(dir_path, exist_ok=True)
        for dataset_name in args['omic_datasets']:
            dir_path = os.path.join(args['omic_output_dir'], dataset_name)
            os.makedirs(dir_path, exist_ok=True)
        for dataset_name in args['wsi_datasets']:
            dir_path = os.path.join(args['wsi_output_dir'], dataset_name)
            os.makedirs(dir_path, exist_ok=True)
        for dataset_name in args['time_event']:
            dir_path = os.path.join(args['time_event_output_dir'], dataset_name)
            os.makedirs(dir_path, exist_ok=True)
            
    elif task == 'samvae_sa': 
        for params in args['param_comb']:
            for seed in range(args['n_seeds']):
                model_path = f"{params['latent_dim']}_{params['hidden_size']}/seed_{seed}"
                dir = args['output_dir'] + '/' + model_path + '/'
                os.makedirs(dir, exist_ok=True)
    elif task == 'plots_samvae_sa':  
        dir = args['plots_output_dir']  
        os.makedirs(dir, exist_ok=True)            
                
# Experimental Setup
def print_experiment_config(args, save_path=None): 
    info_messages = [
        ("All Datasets: ", args.get('datasets', [])[:-1]),
        ("Modes: ", args.get('modes', [])[:-1])]
    
    config_text = []
    for label, value in info_messages:
        config_text.append(f"{label}: {value}")
        
    # Fixed Parameters
    fixed_params = {
        'Train': args['train'],
        'Eval': args['eval'],
        'Early Stop': args['early_stop'],
        'Normalization Loss': args['normalization_loss'],
        'N Folds': args['n_folds'],
        'Batch Size': args['batch_size'],
        'Epochs': args['n_epochs'],
        'Learning Rate': args['lr'],
        'Betas': args['betas'],
        'Patience': args['patience'],
        'Time Distribution': args['time_distribution'],
        'Time Mode': args['time_mode'],
        'N WSI': args['N_wsi'],
        'image_resolution' : args['image_resolution']
    }
    config_text.append("\n### Fixed Parameters ###\n")
    config_text.append(tabulate(fixed_params.items(), headers=["Parameter", "Value"], tablefmt="fancy_grid"))

    # Hyperparameter Combinations
    param_comb = args.get("param_comb", []) 
    headers = param_comb[0].keys()
    table = [list(comb.values()) for comb in param_comb]
    config_text.append("\n### Hyperparameter Combinations ###\n")
    config_text.append(tabulate(table, headers=headers, tablefmt="fancy_grid"))
    print( "\n".join(config_text))

    if save_path:
        with open(save_path, 'w') as f:
            f.write("\n".join(config_text))
 

# ------------------------------------------------------------------------------------------------------
#                                Parameter Combinations and Datasets
# ------------------------------------------------------------------------------------------------------

#  Parameter Combinations for Hyperparameter Optimization
def parameter_combination(modes):
    dropout_prop = [0.2]  
    mode_params = { 
        'clinical': {'latent_dim': [5,10], 'hidden_size': [5, 10, 25, 50, 75, 100, 500]}, 
        'omic_adn': {'latent_dim': [5, 50, 500], 'hidden_size': [5, 50, 500]},
        'omic_cnv': {'latent_dim': [5, 50, 500], 'hidden_size': [5, 50, 500]},
        'omic_miRNA': {'latent_dim': [5, 50, 500], 'hidden_size': [5, 50, 500]},
        'omic_RNAseq': {'latent_dim': [5, 50, 500], 'hidden_size': [5, 50, 500]},
        'wsi_patches': {'latent_dim': [5, 50, 500], 'hidden_size': [[8, 16, 32], [16, 32, 64], [32, 64, 128]]},
        'wsi': {'latent_dim': [5, 50, 500], 'hidden_size': [[8, 16, 32], [16, 32, 64], [32, 64, 128]]}, 
        'wsi_CLAM_mask': {'latent_dim': [5, 50, 500], 'hidden_size': [[8, 16, 32], [16, 32, 64], [32, 64, 128]]}, 
        'wsi_CLAM_heatmap': {'latent_dim': [5, 50, 500], 'hidden_size': [[8, 16, 32], [16, 32, 64], [32, 64, 128]]}
        
    }
    combined_params = []
    for mode in modes:
        params = mode_params[mode]
        latent_dim_values = params['latent_dim'] 
        hidden_size_values = params['hidden_size']
        
        if latent_dim_values is None:
            for hidden_size, dp in product(hidden_size_values, dropout_prop):
                combined_params.append({
                    'hidden_size': hidden_size,
                    'dropout_prop': dp
                })
        else:
            for latent_dim, hidden_size in product(latent_dim_values, hidden_size_values):
                for dp in dropout_prop:
                    combined_params.append({
                        'latent_dim': latent_dim,
                        'hidden_size': hidden_size,
                        'dropout_prop': dp
                    })
    return combined_params
 

# Parameters Combination for Intermediate and Final Combinations
def parameter_best_combination(modes, dataset_name, competing_risks):
    dropout_prop = [0.2]  
    if dataset_name == 'lgg':
        # Optimal parameters for competing risks in lower grade gliomas  
        if competing_risks:
            mode_params = {
                'clinical': {'latent_dim': [5], 'hidden_size': [75]}, 
                'omic_adn': {'latent_dim': [5], 'hidden_size': [50]}, 
                'omic_cnv': {'latent_dim': [5], 'hidden_size': [500]}, 
                'omic_miRNA': {'latent_dim': [5], 'hidden_size': [50]}, 
                'omic_RNAseq': {'latent_dim': [5], 'hidden_size': [50]}, 
                'wsi_patches': {'latent_dim': [5], 'hidden_size': [[16, 32,64]]},
                'time_event': {'latent_dim': None, 'hidden_size': [100]} 
            }
        # Optimal parameters for survival analysis in lower grade gliomas  
        else:
            mode_params = {
                'clinical': {'latent_dim': [10], 'hidden_size': [10]}, 
                'omic_adn': {'latent_dim': [5], 'hidden_size': [50]}, 
                'omic_cnv': {'latent_dim': [5], 'hidden_size': [500]}, 
                'omic_miRNA': {'latent_dim': [5], 'hidden_size': [5]}, 
                'omic_RNAseq': {'latent_dim': [5], 'hidden_size': [500]}, 
                'wsi_patches': {'latent_dim': [5], 'hidden_size': [[8,16, 32]]},
                'time_event': {'latent_dim': None, 'hidden_size': [100]} 
            }
             
    elif dataset_name == 'brca':
        #  Optimal parameters for competing risks in breast cancer    
        if competing_risks:
            mode_params = { 
                'clinical': {'latent_dim': [10], 'hidden_size': [500]},  
                'omic_adn': {'latent_dim': [5], 'hidden_size': [5]}, 
                'omic_cnv': {'latent_dim': [5], 'hidden_size': [50]}, 
                'omic_miRNA': {'latent_dim': [5], 'hidden_size': [5]},  
                'omic_RNAseq': {'latent_dim': [50], 'hidden_size': [50]}, 
                'wsi_patches': {'latent_dim': [50], 'hidden_size': [[16, 32, 64]]}, 
                'time_event': {'latent_dim': None, 'hidden_size': [100]}   
            }
        else:
            # Optimal parameters for survival analysis in breast cancer    
            mode_params = {
                'clinical': {'latent_dim': [10], 'hidden_size': [100]},  
                'omic_adn': {'latent_dim': [5], 'hidden_size': [5]},
                'omic_cnv': {'latent_dim': [5], 'hidden_size': [5]}, 
                'omic_miRNA': {'latent_dim': [5], 'hidden_size': [5]},
                'omic_RNAseq': {'latent_dim': [5], 'hidden_size': [500]},
                'wsi_patches': {'latent_dim': [5], 'hidden_size': [[32, 64, 128]]}, 
                'time_event': {'latent_dim': None, 'hidden_size': [100]}   
            }
    else:
        raise ValueError(f"Dataset {dataset_name} no reconocido")
    
    best_params = {'latent_dim': [], 'hidden_size': [], 'dropout_prop': []}
    for mode in modes:
        params = mode_params[mode]
        latent_dim_values = params['latent_dim'] 
        hidden_size_values = params['hidden_size']
        
        if latent_dim_values is None:
            for hidden_size in hidden_size_values:
                for dp in dropout_prop:
                    best_params['hidden_size'].append(hidden_size)
                    best_params['dropout_prop'].append(dp)
        else:
            for latent_dim in latent_dim_values:
                for hidden_size in hidden_size_values:
                    for dp in dropout_prop:
                        best_params['latent_dim'].append(latent_dim)
                        best_params['hidden_size'].append(hidden_size)
                        best_params['dropout_prop'].append(dp)                        
    return [best_params]

# Parameters Combination for Hyperparameter Optimization with Clinical Data
def parameter_combination_with_best_clinical(modes, dataset_name, use_combination=True, competing_risks=False):
    non_clinical_modes = [m for m in modes if m != 'clinical']
    param_list = []
    # Get combinations for non-clinical modes
    for mode in non_clinical_modes:
        param_list.extend(parameter_combination([mode]))
    # Get parameters for the clinical mode from the best combination function
    if 'clinical' in modes:
        clinical_params = parameter_best_combination(['clinical'], dataset_name, competing_risks)[0]
        num_combinations = len(param_list) // len(non_clinical_modes)
        # Expand clinical parameters to match in quantity
        clinical_entries = []
        for i in range(num_combinations):
            clinical_entries.append({
                'latent_dim': clinical_params['latent_dim'][0],
                'hidden_size': clinical_params['hidden_size'][0],
                'dropout_prop': clinical_params['dropout_prop'][0],
            })
    else:
        clinical_entries = []
    merged_params = []
    for i in range(num_combinations):
        merged_entry = {'latent_dim': [], 'hidden_size': [], 'dropout_prop': []}

        for mode_idx, mode in enumerate(modes):
            if mode == 'clinical':
                params = clinical_entries[i]
            else:
                idx = i + sum(1 for m in modes[:mode_idx] if m != 'clinical') * num_combinations
                params = param_list[idx]
            merged_entry['latent_dim'].append(params.get('latent_dim'))
            merged_entry['hidden_size'].append(params['hidden_size'])
            merged_entry['dropout_prop'].append(params['dropout_prop'])
        merged_params.append(merged_entry)
    return merged_params

# Parameters Combination 
def get_parameters(modes, dataset_name, use_combination, optimizer_with_clinical, competing_risks):
    if optimizer_with_clinical and use_combination:
        return parameter_combination_with_best_clinical(modes, dataset_name, use_combination, competing_risks)
    elif use_combination:
        return parameter_combination(modes)
    else:
        return parameter_best_combination(modes, dataset_name, competing_risks)

def load_datasets(args):
        clinical_datasets, omic_datasets, wsi_datasets, time_event = [], [], [], []

        for ds in args['datasets']:
            if ds in args['clinical_datasets']:
                clinical_datasets.append(ds)
            elif ds in args['omic_datasets']:
                omic_datasets.append(ds)
            elif ds in args['wsi_datasets']:
                wsi_datasets.append(ds)
            elif ds in args['time_event']:
                time_event.append(ds)

        all_tensors = []
        for ds in clinical_datasets + omic_datasets + wsi_datasets + time_event:
            if ds in clinical_datasets:
                input_dir = os.path.join(args['clinical_input_dir'], ds)
            elif ds in omic_datasets:
                input_dir = os.path.join(args['omic_input_dir'], ds)
            elif ds in wsi_datasets:
                input_dir = os.path.join(args['wsi_input_dir'], ds)
            elif ds in time_event:
                input_dir = os.path.join(args['time_event_input_dir'], ds)
            else:
                continue
            tensor = torch.load(os.path.join(input_dir, 'data.pt'))
            all_tensors.append(tensor)

        return all_tensors            
    
# ------------------------------------------------------------------------------------------------------
#                              Arguments for Environment Configuration 
# ------------------------------------------------------------------------------------------------------ 
# Function to set environment configuration
def run_args(task, config):
    args = {}

    clinical_dataset = config.get("clinical_dataset", [])
    omic_dataset = config.get("omic_dataset", None)
    wsi_dataset = config.get("wsi_dataset", None)
    time_event = config.get("time_event", [])
    
   # Modes
    modes = []
    if clinical_dataset:
        modes.append('clinical')
    if omic_dataset:
        for omic in omic_dataset:
            dataset_name, subtype = omic.split('_')  # Extract base name and subtype
            modes.append(f'omic_{subtype}')
    if wsi_dataset:
        for wsi in wsi_dataset:
            dataset_name, subtype = wsi.split('_', 1)   
            if subtype == 'wsi':
                modes.append('wsi')
            else:
                modes.append(f'wsi_{subtype}')
    modes_used = '_'.join(modes) 
      
    # Type of cancer 
    if clinical_dataset:
        dataset_name = clinical_dataset[0].split('_')[0]  
    elif omic_dataset:
        dataset_name = omic_dataset[0].split('_')[0] 
    elif wsi_dataset:
        dataset_name = wsi_dataset[0].split('_')[0]  
    else:
        raise ValueError('Error: No dataset has been loaded')  
    
    # Datasets
    args['clinical_datasets'] = clinical_dataset if clinical_dataset else []
    args['omic_datasets'] = omic_dataset if omic_dataset else []
    args['wsi_datasets'] = wsi_dataset if wsi_dataset else []
    args['time_event'] = time_event if time_event else []
    args['datasets'] = args['clinical_datasets'] + args['omic_datasets'] + args['wsi_datasets'] + args['time_event']
    args['modes'] = ['clinical'] * len(args['clinical_datasets']) + ['omic'] * len(args['omic_datasets']) + ['wsi'] * len(args['wsi_datasets']) + ['time_event'] * len(args['time_event'])


    # ------------------------------------------------------------------------------------------------------
    #                                           Configurations  
    # ------------------------------------------------------------------------------------------------------

    # Training and testing configurations  
    args['train'] =  not True 
    args['eval'] = True
    args['hyperparameter_optimization'] = not True
    args['optimizer_with_clinical'] =   True  # True when multimodal optimization (with clinical data)
    args['batch_size'] = 512  
    args['n_threads'] = 1
    args['n_seeds'] = 10 # Set 10 seeds for final combinations and 5 seeds for others 
    args['N_wsi'] = 10 # Number of WSI patches to use  
    args['final_tables'] = not True
    args['final_plots'] =  not True
    args['results_pkl']= not True # True if results did not save properly in the previous run, so it needs to be saved again (args['train'] must be True)
     
    # Fixed parameters:
    args['n_epochs'] = 3000
    args['n_folds'] = 5 
    args['early_stop'] = True
    args['normalization_loss'] = not True
    args['lr'] = 1e-4  
    args['betas'] = (0.9, 0.9) 
    args['patience'] = 50  
    args['time_distribution'] = ('weibull', 2)
    args['time_mode'] = 'time_event'
    args['time_hidden_size'] = 50
    args['image_resolution'] = 128
    args['final_image_resolution'] = 16
    args['patient_limit'] = True # For BRCA, due to the number of patients
    args['competing_risks'] = any(te.endswith("_cr") for te in time_event) # True if competing risks 
    args['param_comb'] = get_parameters(modes, dataset_name, args['hyperparameter_optimization'], args['optimizer_with_clinical'], args['competing_risks'])   
    use_pca = True  # pca data: 100-200 genes, raw data: without pca
    
    
    # Set arguments regardless of task
    abs_path = os.path.dirname(os.path.abspath(__file__)) + os.sep
    wsi_abs_path = "/hdd/alba/samvae-main/"
    
    if task == 'data_preprocessing':
        # INPUT
        args['clinical_input_dir'] = abs_path + 'data_preprocessing/raw_data/clinical_data/'
        args['omic_input_dir'] = abs_path + f"data_preprocessing/{'pca_data' if use_pca else 'raw_data'}/omic_data/"
        args['wsi_input_dir'] = wsi_abs_path + 'data_preprocessing/raw_data/wsi_data/'
        
        # OUTPUT
        args['clinical_output_dir'] = abs_path + 'data_preprocessing/data/clinical_data/'
        args['omic_output_dir'] = abs_path + 'data_preprocessing/data/omic_data/'
        args['wsi_output_dir'] = wsi_abs_path + 'data_preprocessing/data/wsi_data/'
        args['time_event_output_dir'] = abs_path + 'data_preprocessing/data/time_event/'
        
    elif task == 'samvae_sa':    
        # Data preprocessed
        args['clinical_input_dir'] = abs_path + 'data_preprocessing/data/clinical_data/'
        args['omic_input_dir'] = abs_path + 'data_preprocessing/data/omic_data/'
        args['wsi_input_dir'] = wsi_abs_path + 'data_preprocessing/data/wsi_data/'
        args['time_event_input_dir'] = abs_path + 'data_preprocessing/data/time_event/'
        
        # SAMVAE output folders
        workflow_stage = "Hyperparameter_optimization" if args['hyperparameter_optimization'] else "Intermediate_Combinations"
        problem_type = "Competing_Risks" if args['competing_risks'] else "Survival_Analysis"
        wsi_suffix = f"_{args['N_wsi']}_patch" if 'wsi_patches' in modes_used else ""
        if args['n_seeds'] == 10:
            args['output_dir'] = (abs_path + 'results/Final_Combinations/' + f'{problem_type}/{dataset_name}/{modes_used}{wsi_suffix}/' + f"{args['n_folds']}_folds_{args['batch_size']}_batch_size/")
        else:
            args['output_dir'] = (abs_path + 'results/' + f'{workflow_stage}/' + f'{problem_type}/{dataset_name}/{modes_used}{wsi_suffix}/' + f"{args['n_folds']}_folds_{args['batch_size']}_batch_size/")   
        args['plots_output_dir'] =  args['output_dir'].replace('results/', 'plots/')       
    return args