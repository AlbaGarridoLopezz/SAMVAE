# Author: Alba Garrido López
# Email: alba.garrido.lopez@upm.es

# Packages to import
import os
import sys
import torch
import json
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)
from utils import run_args, create_output_dir
from sa_datasets import preprocess_clinical_lgg, preprocess_clinical_brca, preprocess_adn,preprocess_mirna, preprocess_rnaseq, preprocess_cnv, preprocess_wsi, preprocess_CLAM, preprocess_time_event, preprocess_clinical_lgg_cr, preprocess_clinical_brca_cr

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Preprocess Data
def preprocess_data(dataset_name, args, device):
    if isinstance(dataset_name, str):
        dataset_name = [dataset_name]
    for dataset in dataset_name:
        # Clinical Datasets: lgg for lower grade glioma, brca for breast cancer
        if dataset in ['lgg_clinical']: 
            data = preprocess_clinical_lgg(dataset, args, device)
        elif dataset in ['brca_clinical']: 
            data = preprocess_clinical_brca(dataset, args, device)   
            
        # Omic Datasets
        elif dataset in ['brca_adn', 'lgg_adn']:   
            data = preprocess_adn(dataset, args, device)     # DNA-Methylation
        elif dataset in ['brca_miRNA','lgg_miRNA']:
            data = preprocess_mirna(dataset, args, device)   # MicroRNA
        elif dataset in ['brca_RNAseq','lgg_RNAseq']:
            data = preprocess_rnaseq(dataset, args, device)  # RNAseq
        elif dataset in ['brca_cnv','lgg_cnv']:
            data = preprocess_cnv(dataset, args, device)     # CNV
        
        # Whole Slide Images and Patches Datasets        
        elif dataset in ['brca_wsi', 'lgg_wsi']:   
            data = preprocess_wsi(dataset, args, device) 
        elif dataset in ['lgg_CLAM_mask','brca_CLAM_mask','lgg_CLAM_heatmap','brca_CLAM_heatmap','lgg_patches','brca_patches']:   
            data = preprocess_CLAM(dataset, args, device)     
        
        # Competing Risk Datasets
        elif dataset in ['lgg_clinical_cr']: 
            data = preprocess_clinical_lgg_cr(dataset, args, device) 
        elif dataset in ['brca_clinical_cr']: 
            data = preprocess_clinical_brca_cr(dataset, args, device) 
        elif dataset in ['melanoma_clinical_cr']: 
            data = preprocess_clinical_melanoma_cr(dataset, args, device) 
        elif dataset in ['mgus2_clinical_cr']: 
            data = preprocess_clinical_mgus2_cr(dataset, args, device) 
        elif dataset in ['ebmt2_clinical_cr']: 
            data = preprocess_clinical_ebmt2_cr(dataset, args, device)             
        else:
            raise RuntimeError(f"Dataset {dataset} no reconocido")
        
    return data


def main():
    print('\n\n-------- DATA PREPROCESSING  --------')
    start_time = time.time()
    # Environment configuration
    task = 'data_preprocessing'
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, 'survival_analysis', 'datasets.json')
    
    # Read configurations from the JSON file
    with open(config_path, 'r') as f:
        configurations = json.load(f)
    print(f"\n[INFO] Number of combinations: {len(configurations)}")    
    for config in configurations:
        print(f"\n[INFO] Processing configuration: {config}")

        # Get specific arguments from this configuration
        args = run_args(task, config)
        create_output_dir(task, args)

        # Preprocess data
        for dataset_name in args['clinical_datasets']:
            data = preprocess_data(dataset_name, args, device)  
            output_path = args['clinical_output_dir'] + dataset_name + '/' + 'data.pt'
            torch.save(data, output_path) 

        for dataset_name in args['omic_datasets']:
            data = preprocess_data(dataset_name, args, device) 
            output_path = args['omic_output_dir'] + dataset_name + '/' + 'data.pt'
            torch.save(data, output_path)  

        for dataset_name in args['wsi_datasets']:
            data = preprocess_data(dataset_name, args, device) 
            output_path = args['wsi_output_dir'] + dataset_name + '/' + 'data.pt' 
            torch.save(data, output_path)  
            
        for dataset_name in args['time_event']:
            time_event_tensor = preprocess_time_event(dataset_name, args, device)
            time_event_path = f"{args['time_event_output_dir']}{dataset_name}/data.pt"
            torch.save(time_event_tensor, time_event_path) 
            
        end_time = time.time() - start_time
        print(f"\n[INFO] Total Data Preprocessing time:" + str(end_time))
        
if __name__ == '__main__':
    main()
