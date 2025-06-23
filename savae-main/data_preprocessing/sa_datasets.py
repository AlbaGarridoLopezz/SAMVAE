# Author: Alba Garrido López
# Email: alba.garrido.lopez@upm.es
 
# Packages to import
import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tabulate import tabulate

####################################################################################################
#                                   SAMVAE DATASETS                                                #
####################################################################################################

# --------------------------------------------------------------------------------------------------
#                                   TIME AND EVENT
# --------------------------------------------------------------------------------------------------
 
def preprocess_time_event(dataset, args, device, save_path=None):
    # Define dataset paths  
    dataset_path = {
        'brca_time_event': 'brca/brca_clinical.csv', # Breast cancer for survival analysis
        'lgg_time_event': 'lgg/lgg_clinical.csv', # Lower grade glioma for survival analysis
        'brca_time_event_cr': 'brca/brca_clinical_cr.csv', # Breast cancer for competing risks
        'lgg_time_event_cr': 'lgg/lgg_clinical_cr.csv' # Lower grade glioma for competing risks
    }

    # Extract base name of the dataset (e.g., "brca", "lgg")
    dataset_base = dataset.split('_time_event')[0]
    data_filename = os.path.join(args['clinical_input_dir'], dataset_path[dataset])

    # Load clinical data
    raw_df = pd.read_csv(data_filename, sep=None, engine='python')

    # Dataset-specific preprocessing
    # Competing risks datasets (breast cancer and lower grade glioma)
    if dataset in ['brca_time_event_cr', 'lgg_time_event_cr']:
        raw_df = raw_df[['PFI.time.cr', 'PFI.cr']].copy()
        raw_df = raw_df.rename(columns={'PFI.time.cr': 'time', 'PFI.cr': 'event'})
        
    # Survival Analysis datasets (breast cancer and lower grade glioma)
    else:
        raw_df = raw_df[['time', 'event']].copy()
        raw_df['event'] = raw_df['event'].replace(['Alive', 'Dead'], [0, 1])

    # Handle missing values
    mean_time = raw_df['time'].mean()
    raw_df['time'].fillna(mean_time, inplace=True)

    mode_event = raw_df['event'].mode()[0]  # Use the first mode if multiple exist
    raw_df['event'].fillna(mode_event, inplace=True)

    # Ensure numeric types
    raw_df['time'] = pd.to_numeric(raw_df['time'], errors='coerce').astype(int)
    raw_df['event'] = pd.to_numeric(raw_df['event'], errors='coerce').astype(int)
    raw_df = raw_df[['time', 'event']].dropna()

    # Adjust time to avoid negative or zero values
    raw_df['time'] = raw_df['time'] - raw_df['time'].min() + 1

    # Display event value distribution
    print(f"\n[INFO] 'event' distribution for dataset '{dataset}':")
    value_counts = raw_df['event'].value_counts()
    value_props = raw_df['event'].value_counts(normalize=True) * 100

    for value in value_counts.index:
        count = value_counts[value]
        proportion = value_props[value]
        print(f"  - Event = {value}: {count} patients ({proportion:.2f}%)")
    clinical_tensor = torch.tensor(raw_df.values, dtype=torch.float32)

    # Optional: Subsample BRCA patients due to large dataset size (computational limitations)
    if dataset in ['brca_time_event', 'brca_time_event_cr'] and args.get('patient_limit', False):
        odd_indices = torch.arange(1, raw_df.shape[0], step=2)[:500]
        clinical_tensor = clinical_tensor[odd_indices, :]
        
    time_tensor, event_tensor = clinical_tensor[:, 0], clinical_tensor[:, 1]
    combined_tensor = torch.stack((time_tensor, event_tensor), dim=1).to(device)
    save_path = os.path.join(args['clinical_input_dir'], f"{dataset_base}.pt")
    torch.save(combined_tensor, save_path)
    return combined_tensor


def cr_time_config(df):
    # Remove all rows containing NaN values
    df = df.dropna()

    # Extract all unique event labels and remove 0 (which indicates censoring)
    labels = list(np.unique(df['event']))
    labels.remove(0)

    # Ensure there is more than one event type (competing risks scenario)
    assert len(labels) > 1  # Assumes labels are 1, 2, 3, ..., and 0 is censoring

    # Get the maximum time observed for each event type (excluding censored)
    max_t = [df[df['event'] == l]['time'].max() for l in labels]

    # Define a default time distribution configuration for each event type
    time_dist = [('weibull', 2) for _ in range(len(labels))]

    # Get the unique set of time cut points
    cuts = np.unique(df['time'])

    return df, time_dist, max_t, labels, cuts


# ------------------------------------------------------------------------------------------------------
#                                CLINICAL TABULAR DATA
# ------------------------------------------------------------------------------------------------------

# Survival Analysis Datasets: lgg & brca
def preprocess_clinical_lgg(dataset, args, device):
    # Define dataset paths
    dataset_path = {'lgg_clinical': 'lgg/lgg_clinical.csv'} 
    
    # Load the dataset
    data_filename = args['clinical_input_dir'] + dataset_path[dataset]
    raw_df = pd.read_csv(data_filename, sep=',')

    # Convert numeric columns and impute missing values with column mean
    numeric_cols = ['age_at_index', 'age_at_diagnosis', 'year_of_diagnosis']
    for col in numeric_cols:
        raw_df[col] = pd.to_numeric(raw_df[col], errors='coerce').fillna(raw_df[col].mean()).astype(int)

    # Copy DataFrame to preserve original
    df = raw_df.copy()

    # Drop unnecessary columns if they exist
    df.drop(columns=['time', 'event'], errors='ignore', inplace=True)

    # Define mapping for variables
    mapping_dict = {
        'gender': {'male': 0, 'female': 1},
        'race': {'white': 0, 'black or african american': 1, 'asian': 2, 'american indian or alaska native': 3, 'other': 4},
        'prior_treatment': {'No': 0, 'Yes': 1},
        'morphology': {'9382/3': 0, '9450/3': 1, '9401/3': 2, '9451/3': 3, '9400/3': 4},
        'primary_diagnosis': {'Mixed glioma': 0, 'Oligodendroglioma, NOS': 1, 'Astrocytoma, anaplastic': 2, 
                              'Oligodendroglioma, anaplastic': 3, 'Astrocytoma, NOS': 4},
        'prior_malignancy': {'no': 0, 'yes': 1},
        'tissue_or_organ_of_origin': {'Cerebrum': 0, 'Brain, NOS': 1, 'Temporal lobe': 2, 'Frontal lobe': 3},
        'treatment_or_therapy': {'no': 0, 'yes': 1}
    }

    # Apply mappings and handle missing values
    for col, mapping in mapping_dict.items():
        if col in df.columns:
            df[col].replace("not reported", np.nan, inplace=True)
            df[col].replace("Not Reported", np.nan, inplace=True)
            df[col] = df[col].replace(mapping)
            df[col].fillna(df[col].mode()[0], inplace=True)  # Impute missing values with mode
            df[col] = df[col].astype(int)

    # Drop the first column (patient ID)
    df = df.iloc[:, 1:]

    # Display feature names and types
    feature_info = [(i+1, col, str(dtype)) for i, (col, dtype) in enumerate(df.dtypes.items())]
    print("\n[INFO] Clinical dataset features for '{}':".format(dataset))
    print(tabulate(feature_info, headers=["#", "Feature", "Type"], tablefmt="pretty"))
    print(f"\n[INFO] Total number of features: {df.shape[1]}")
    print(f"[INFO] Total number of patients: {df.shape[0]}")

    # Convert DataFrame to tensor
    tensor = torch.tensor(df.values, dtype=torch.float32).to(device)
    
    return tensor
 
 
 
def preprocess_clinical_brca(dataset, args, device):
    # Define dataset paths
    dataset_path = {'brca_clinical': 'brca/brca_clinical.csv'} 
    
    # Load the dataset
    data_filename = args['clinical_input_dir'] + dataset_path[dataset]
    raw_df = pd.read_csv(data_filename, sep=',')
    
    # Convert numerical variables
    numeric_cols = ['age_at_index', 'age_at_diagnosis', 'year_of_diagnosis']
    for col in numeric_cols:
        raw_df[col] = pd.to_numeric(raw_df[col], errors='coerce').fillna(raw_df[col].mean()).astype(int)
    
    # Copy the DataFrame
    df = raw_df.copy()

    # Drop unnecessary columns if they exist
    df.drop(columns=['time', 'event'], errors='ignore', inplace=True)

    # Map categorical variables to numeric values
    mapping_dict = {
        'gender': {'male': 0, 'female': 1},
        'race': {'white': 0, 'black or african american': 1, 'asian': 2, 'american indian or alaska native': 3, 'other': 4},
        'prior_treatment': {'No': 0, 'Yes': 1},
        'morphology': {'8500/3': 0, '8520/3': 1, '8522/3': 2, '8523/3': 3, '8575/3': 4, '8480/3': 5, '8510/3': 6, '8524/3': 7, '8507/3': 8, '8541/3': 9, '8503/3': 10, '9020/3': 11, '8200/3': 12, '8211/3': 13, '8201/3': 14, '8502/3': 15, '8401/3': 16, '8022/3': 17, '8010/3': 18},
        'ajcc_pathologic_m': {'M0': 0, 'MX': 1, 'M1': 2, 'cM0 (i+)': 3},
        'prior_malignancy': {'no': 0, 'yes': 1},
        'tissue_or_organ_of_origin': {'Breast, NOS': 0, 'Upper-outer quadrant of breast': 1, 'Lower-inner quadrant of breast': 2, 'Upper-inner quadrant of breast': 3, 'Lower-outer quadrant of breast': 4},
        'treatment_or_therapy': {'no': 0, 'yes': 1},
        'ajcc_pathologic_n': {'N0': 0, 'N1a': 1, 'N0 (i-)': 2, 'N1': 3, 'N2a': 4, 'N3a': 5, 'N2': 6, 'N1b': 7, 'N0 (i+)': 8, 'N1mi': 9, 'N3': 10, 'NX': 11, 'N1c': 12, 'N3b': 13, 'N0 (mol+)': 14},
        'ajcc_pathologic_stage': {'Stage IIA': 0, 'Stage IIB': 1, 'Stage IIIA': 2, 'Stage I': 3, 'Stage IA': 4, 'Stage IIIC': 5, 'Stage IIIB': 6, 'Stage IV': 7, 'Stage X': 8, 'Stage IB': 9, 'Stage II': 10, 'Stage III': 11},
        'ajcc_pathologic_t': {'T2': 0, 'T1c': 1, 'T3': 2, 'T1': 3, 'T4b': 4, 'T4': 5, 'T1b': 6, 'TX': 7, 'T1a': 8, 'T4d': 9},
        'ajcc_staging_system_edition': {'6th': 0, '7th': 1, '5th': 2, '4th': 3, '3rd': 4},
        'primary_diagnosis': {
            'Infiltrating duct carcinoma, NOS': 0, 'Lobular carcinoma, NOS': 1, 'Infiltrating duct and lobular carcinoma': 2,
            'Infiltrating duct mixed with other types of carcinoma': 3, 'Metaplastic carcinoma, NOS': 4, 'Mucinous adenocarcinoma': 5,
            'Medullary carcinoma, NOS': 6, 'Infiltrating lobular mixed with other types of carcinoma': 7, 'Intraductal micropapillary carcinoma': 8,
            'Paget disease and infiltrating duct carcinoma of breast': 9, 'Intraductal papillary adenocarcinoma with invasion': 10,
            'Phyllodes tumor, malignant': 11, 'Adenoid cystic carcinoma': 12, 'Tubular adenocarcinoma': 13, 'Cribriform carcinoma, NOS': 14,
            'Secretory carcinoma of breast': 15, 'Apocrine adenocarcinoma': 16, 'Pleomorphic carcinoma': 17, 'Carcinoma, NOS': 18
        }
    }
    
    for col, mapping in mapping_dict.items():
        if col in df.columns:
            df[col].replace("not reported", np.nan, inplace=True)
            df[col].replace("Not Reported", np.nan, inplace=True)
            df[col] = df[col].replace(mapping)
            df[col].fillna(df[col].mode()[0], inplace=True)  # Impute missing values with mode
            df[col] = df[col].astype(int)

    # Drop the first column (Patient ID)
    df = df.iloc[:, 1:]

    # Display feature names and types
    feature_info = [(i+1, col, str(dtype)) for i, (col, dtype) in enumerate(df.dtypes.items())]
    print("\n[INFO] Clinical dataset features for '{}':".format(dataset))
    print(tabulate(feature_info, headers=["#", "Feature", "Type"], tablefmt="pretty"))
    print(f"\n[INFO] Total number of features: {df.shape[1]}")
    
    # Convert DataFrame to tensor
    tensor = torch.tensor(df.values, dtype=torch.float32).to(device)
    
    # Optional: Subsample BRCA patients due to large dataset size (computational limitations)
    if args.get('patient_limit', False):
        odd_indices = torch.arange(1, tensor.size(0), step=2)[:500]
        tensor = tensor[odd_indices, :] 
    
    print(f"Dimensions of the clinical tensor {dataset}: {tensor.shape}")
    
    return tensor


# Competing Risks Datasets: lgg_cr & brca_cr
def preprocess_clinical_lgg_cr(dataset, args, device):
    # Define the dataset path
    dataset_path = {'lgg_clinical_cr': 'lgg/lgg_clinical_cr.csv'}

    # Load the dataset
    data_filename = args['clinical_input_dir'] + dataset_path[dataset]
    raw_df = pd.read_csv(data_filename, sep=',')

    # Convert specified columns to numeric, filling NaNs with the column mean
    numeric_cols = ['age_at_index', 'age_at_diagnosis', 'year_of_diagnosis']
    for col in numeric_cols:
        raw_df[col] = pd.to_numeric(raw_df[col], errors='coerce').fillna(raw_df[col].mean()).astype(int)

    # Make a copy of the original DataFrame
    df = raw_df.copy()

    # Drop unnecessary or redundant columns
    df.drop(columns=['case_submitter_id', 'time', 'event', 'PFI.cr', 'PFI.time.cr'], errors='ignore', inplace=True)

    # Define categorical mappings for conversion to numerical values
    mapping_dict = {
        'gender': {'male': 0, 'female': 1},
        'race': {
            'white': 0, 'black or african american': 1, 'asian': 2,
            'american indian or alaska native': 3, 'other': 4
        },
        'prior_treatment': {'No': 0, 'Yes': 1},
        'morphology': {'9382/3': 0, '9450/3': 1, '9401/3': 2, '9451/3': 3, '9400/3': 4},
        'primary_diagnosis': {
            'Mixed glioma': 0, 'Oligodendroglioma, NOS': 1, 'Astrocytoma, anaplastic': 2,
            'Oligodendroglioma, anaplastic': 3, 'Astrocytoma, NOS': 4
        },
        'prior_malignancy': {'no': 0, 'yes': 1},
        'tissue_or_organ_of_origin': {
            'Cerebrum': 0, 'Brain, NOS': 1, 'Temporal lobe': 2, 'Frontal lobe': 3
        },
        'treatment_or_therapy': {'no': 0, 'yes': 1}
    }

    # Apply the categorical mappings and handle missing values
    for col, mapping in mapping_dict.items():
        if col in df.columns:
            df[col].replace("not reported", np.nan, inplace=True)
            df[col].replace("Not Reported", np.nan, inplace=True)
            df[col] = df[col].replace(mapping)
            df[col].fillna(df[col].mode()[0], inplace=True)  # Impute missing values with the mode
            df[col] = df[col].astype(int)

    # Drop the first column (likely an index or ID column)
    df = df.iloc[:, 1:]

    # Display feature names and types
    feature_info = [(i+1, col, str(dtype)) for i, (col, dtype) in enumerate(df.dtypes.items())]
    print("\n[INFO] Clinical dataset features for '{}':".format(dataset))
    print(tabulate(feature_info, headers=["#", "Feature", "Type"], tablefmt="pretty"))
    print(f"\n[INFO] Total number of features: {df.shape[1]}")
    print(f"[INFO] Total number of patients: {df.shape[0]}")
    
    # Convert the DataFrame to a PyTorch tensor
    tensor = torch.tensor(df.values, dtype=torch.float32).to(device)
    
    return tensor


def preprocess_clinical_brca_cr(dataset, args, device):
    # Define dataset paths for clinical data
    dataset_path = {'brca_clinical_cr': 'brca/brca_clinical_cr.csv'}

    # Load the dataset from the specified file
    data_filename = args['clinical_input_dir'] + dataset_path[dataset]
    raw_df = pd.read_csv(data_filename, sep=',')

    # Convert specified columns to numeric, replacing errors with NaN and then filling NaNs with the column mean
    numeric_cols = ['age_at_index', 'age_at_diagnosis', 'year_of_diagnosis']
    for col in numeric_cols:
        raw_df[col] = pd.to_numeric(raw_df[col], errors='coerce').fillna(raw_df[col].mean()).astype(int)

    # Make a copy of the original DataFrame for further processing
    df = raw_df.copy()

    # Remove unnecessary or redundant columns from the DataFrame
    df.drop(columns=['case_submitter_id', 'time', 'event', 'PFI.cr', 'PFI.time.cr'], errors='ignore', inplace=True)

    # Define mappings for categorical variables to numeric values
    mapping_dict = {
        'gender': {'male': 0, 'female': 1},
        'race': {'white': 0, 'black or african american': 1, 'asian': 2, 'american indian or alaska native': 3, 'other': 4},
        'prior_treatment': {'No': 0, 'Yes': 1},
        'morphology': {'8500/3': 0, '8520/3': 1, '8522/3': 2, '8523/3': 3, '8575/3': 4, '8480/3': 5, '8510/3': 6, '8524/3': 7, '8507/3': 8, '8541/3': 9, '8503/3': 10, '9020/3': 11, '8200/3': 12, '8211/3': 13, '8201/3': 14, '8502/3': 15, '8401/3': 16, '8022/3': 17, '8010/3': 18},
        'ajcc_pathologic_m': {'M0': 0, 'MX': 1, 'M1': 2, 'cM0 (i+)': 3},
        'prior_malignancy': {'no': 0, 'yes': 1},
        'tissue_or_organ_of_origin': {'Breast, NOS': 0, 'Upper-outer quadrant of breast': 1, 'Lower-inner quadrant of breast': 2, 'Upper-inner quadrant of breast': 3, 'Lower-outer quadrant of breast': 4},
        'treatment_or_therapy': {'no': 0, 'yes': 1},
        'ajcc_pathologic_n': {'N0': 0, 'N1a': 1, 'N0 (i-)': 2, 'N1': 3, 'N2a': 4, 'N3a': 5, 'N2': 6, 'N1b': 7, 'N0 (i+)': 8, 'N1mi': 9, 'N3': 10, 'NX': 11, 'N1c': 12, 'N3b': 13, 'N0 (mol+)': 14},
        'ajcc_pathologic_stage': {'Stage IIA': 0, 'Stage IIB': 1, 'Stage IIIA': 2, 'Stage I': 3, 'Stage IA': 4, 'Stage IIIC': 5, 'Stage IIIB': 6, 'Stage IV': 7, 'Stage X': 8, 'Stage IB': 9, 'Stage II': 10, 'Stage III': 11},
        'ajcc_pathologic_t': {'T2': 0, 'T1c': 1, 'T3': 2, 'T1': 3, 'T4b': 4, 'T4': 5, 'T1b': 6, 'TX': 7, 'T1a': 8, 'T4d': 9},
        'ajcc_staging_system_edition': {'6th': 0, '7th': 1, '5th': 2, '4th': 3, '3rd': 4},
        'primary_diagnosis': {'Infiltrating duct carcinoma, NOS': 0, 'Lobular carcinoma, NOS': 1, 'Infiltrating duct and lobular carcinoma': 2, 'Infiltrating duct mixed with other types of carcinoma': 3, 'Metaplastic carcinoma, NOS': 4, 'Mucinous adenocarcinoma': 5, 'Medullary carcinoma, NOS': 6, 'Infiltrating lobular mixed with other types of carcinoma': 7, 'Intraductal micropapillary carcinoma': 8, 'Paget disease and infiltrating duct carcinoma of breast': 9, 'Intraductal papillary adenocarcinoma with invasion': 10, 'Phyllodes tumor, malignant': 11, 'Adenoid cystic carcinoma': 12, 'Tubular adenocarcinoma': 13, 'Cribriform carcinoma, NOS': 14, 'Secretory carcinoma of breast': 15, 'Apocrine adenocarcinoma': 16, 'Pleomorphic carcinoma': 17, 'Carcinoma, NOS': 18}
    }

    # Apply the mappings to categorical columns and handle missing values
    for col, mapping in mapping_dict.items():
        if col in df.columns:
            df[col].replace("not reported", np.nan, inplace=True)
            df[col].replace("Not Reported", np.nan, inplace=True)
            df[col] = df[col].replace(mapping)
            df[col].fillna(df[col].mode()[0], inplace=True)  # Impute missing values with the mode
            df[col] = df[col].astype(int)

    # Optional step: Drop certain columns (commented out for now)
    # df.drop(columns=['ajcc_pathologic_m','tissue_or_organ_of_origin','primary_diagnosis','morphology','ajcc_pathologic_n','ajcc_pathologic_stage','ajcc_pathologic_t','ajcc_staging_system_edition'], errors='ignore', inplace=True)

    # Drop the first column (likely an index or ID column)
    df = df.iloc[:, 1:]

    # Display feature names and types
    feature_info = [(i+1, col, str(dtype)) for i, (col, dtype) in enumerate(df.dtypes.items())]
    print("\n[INFO] Clinical dataset features for '{}':".format(dataset))
    print(tabulate(feature_info, headers=["#", "Feature", "Type"], tablefmt="pretty"))
    print(f"\n[INFO] Total number of features: {df.shape[1]}")
    print(f"[INFO] Total number of patients: {df.shape[0]}")
    
    # Convert the DataFrame to a PyTorch tensor and move it to the appropriate device
    tensor = torch.tensor(df.values, dtype=torch.float32).to(device)

    # Optional: Subsample BRCA patients due to large dataset size (computational limitations)
    if dataset == 'brca_clinical_cr' and args.get('patient_limit', False):
        odd_indices = torch.arange(1, df.shape[0], step=2)[:500]
        tensor = tensor[odd_indices, :]

    return tensor



# ------------------------------------------------------------------------------------------------------
#                                   OMIC DATA: ADN METHYLATION
# ------------------------------------------------------------------------------------------------------

def preprocess_adn(dataset, args, device):
    # Load data from pickle file 
    dataset_path = {'brca_adn': 'adn/brca/adn_BRCA_pca_100.pkl',
                    'lgg_adn': 'adn/lgg/adn_LGG_pca_100.pkl'}
    data_filename = args['omic_input_dir'] + dataset_path[dataset]
    
    # Load pickle file
    df = pd.read_pickle(data_filename)
   
    # Convert to tensor
    tensor = torch.tensor(df.values, dtype=torch.float32).to(device)
    
    # Optional: Subsample BRCA patients due to large dataset size (computational limitations)
    if dataset == 'brca_adn' and args.get('patient_limit', False): 
        odd_indices = torch.arange(1, tensor.size(0), step=2)[:500]
        tensor = tensor[odd_indices, :]
    
    print(f"Dimensions of the ADN tensor: {tensor.shape}")
    return tensor

 
 

# ------------------------------------------------------------------------------------------------------
#                                   OMIC DATA: MicroRNA
# ------------------------------------------------------------------------------------------------------

def preprocess_mirna(dataset, args, device):
    # Load data from pickle file  
    dataset_path = {'brca_miRNA': 'miRNA/brca/miRNA_BRCA_pca_100.pkl',
                    'lgg_miRNA': 'miRNA/lgg/miRNA_LGG_pca_100.pkl'}
    data_filename = args['omic_input_dir'] + dataset_path[dataset]
    
    # Load pickle file
    df = pd.read_pickle(data_filename)
    tensor = torch.tensor(df.values, dtype=torch.float32).to(device)
    
    # Optional: Subsample BRCA patients due to large dataset size (computational limitations)
    if dataset == 'brca_miRNA' and args.get('patient_limit', False):
        odd_indices = torch.arange(1, df.shape[0], step=2)[:500]
        tensor = tensor[odd_indices, :]
    
    # Print tensor dimensions
    print(f"Dimensions of the miRNA tensor: {tensor.shape}")
    return tensor




# ------------------------------------------------------------------------------------------------------
#                                   OMIC DATA: RNA-seq 
# ------------------------------------------------------------------------------------------------------

def preprocess_rnaseq(dataset, args, device):
    # Load data from pickle file
    dataset_path = {'brca_RNAseq': 'RNAseq/brca/RNAseq_BRCA_pca_100.pkl',
                    'lgg_RNAseq': 'RNAseq/lgg/RNAseq_LGG_pca_100.pkl'} 
    data_filename = args['omic_input_dir'] + dataset_path[dataset]
    
    # Load pickle file
    df = pd.read_pickle(data_filename) 
    tensor = torch.tensor(df.values, dtype=torch.float32).to(device)
    
    # Optional: Subsample BRCA patients due to large dataset size (computational limitations)
    if dataset == 'brca_RNAseq' and args.get('patient_limit', False):
        odd_indices = torch.arange(1, df.shape[0], step=2)[:500]
        tensor = tensor[odd_indices, :]
        
    # Print tensor dimensions
    print(f"Dimensions of the RNAseq tensor: {tensor.shape}")
    return tensor


# ------------------------------------------------------------------------------------------------------
#                           OMIC DATA: Copy Number Variation (CNV) 
# ------------------------------------------------------------------------------------------------------
def preprocess_cnv(dataset, args, device):
    dataset_path = {'brca_cnv': 'cnv/brca/cnv_BRCA_pca_100.pkl',
                    'lgg_cnv': 'cnv/lgg/cnv_LGG_pca_100.pkl'} 
    data_filename = args['omic_input_dir'] + dataset_path[dataset]
    
    # Load pickle file
    df = pd.read_pickle(data_filename)
    tensor = torch.tensor(df.values, dtype=torch.float32).to(device)

    # Optional: Subsample BRCA patients due to large dataset size (computational limitations)
    if dataset == 'brca_cnv' and args.get('patient_limit', False):
        odd_indices = torch.arange(1, df.shape[0], step=2)[:500]
        tensor = tensor[odd_indices, :]
    
    # Print tensor dimensions
    print(f"Dimensions of the CNV tensor: {tensor.shape}")

    return tensor


# ------------------------------------------------------------------------------------------------------
#                                               WSI DATA
# ------------------------------------------------------------------------------------------------------

# This function preprocesses the images obtained using the existing Clustering-constrained Attention 
# Multiple Instance Learning (CLAM) model (see https://github.com/mahmoodlab/CLAM).
# Although the published work only reports results based on the top attention patches,
# additional internal experiments were conducted using other outputs such as heatmaps and masks.

def preprocess_CLAM(dataset, args, device): 
    # Set the correct manifest path based on dataset prefix
    manifest_path = ""
    if dataset.startswith('lgg_'):
        manifest_path = '../data_download/Manifest/LGG/filtered_manifest/filtered_gdc_manifest.wsi.txt'
    elif dataset.startswith('brca_'):
        manifest_path = '../data_download/Manifest/BRCA/filtered_manifest/filtered_gdc_manifest.wsi.txt'
    else:
        raise ValueError("Dataset must start with 'lgg_' or 'brca_'.")

    # Load the manifest file
    manifest_df = pd.read_csv(manifest_path, sep='\t')

    # Extract patient IDs in manifest order (remove .svs extension)
    manifest_ids = manifest_df['filename'].str.replace('.svs', '', regex=False).tolist()

    # Define input directories and file suffixes for each dataset type
    wsi_input_dirs = {
        'lgg_CLAM_mask': '/hdd/alba/CLAM-master/CLAM-master/heatmaps/heatmap_raw_results_LGG/HEATMAP_OUTPUT_LGG/LGG',
        'brca_CLAM_mask': '/hdd/alba/CLAM-master/CLAM-master/heatmaps/heatmap_raw_results_BRCA/HEATMAP_OUTPUT_BRCA/BRCA',
        'lgg_CLAM_heatmap': '/hdd/alba/CLAM-master/CLAM-master/heatmaps/heatmap_raw_results_LGG/HEATMAP_OUTPUT_LGG/LGG',
        'brca_CLAM_heatmap': '/hdd/alba/CLAM-master/CLAM-master/heatmaps/heatmap_raw_results_BRCA/HEATMAP_OUTPUT_BRCA/BRCA',
        'lgg_patches': '/hdd/alba/CLAM-master/CLAM-master/heatmaps/heatmap_production_result_LGG/HEATMAP_OUTPUT_LGG_50/sampled_patches/label_LGG_pred_1/topk_high_attention',
        'brca_patches': '/hdd/alba/CLAM-master/CLAM-master/heatmaps/heatmap_production_result_BRCA/HEATMAP_OUTPUT_BRCA_50/sampled_patches/label_BRCA_pred_0/topk_high_attention'
    }

    wsi_input_dir = wsi_input_dirs[dataset]
    file_suffix = '_mask.jpg' if 'CLAM_mask' in dataset else '.png'

    # Initialize placeholders to maintain manifest order
    all_patient_tensors = [None] * len(manifest_ids)
    patient_ids = [None] * len(manifest_ids)

    # Process each patient in the manifest
    for idx, patient_id in enumerate(manifest_ids):
        patient_folder = os.path.join(wsi_input_dir, patient_id)

        if os.path.isdir(patient_folder):
            image_tensors = []

            # List image files matching the expected suffix
            image_files = sorted(
                [image_name for image_name in os.listdir(patient_folder) if image_name.endswith(file_suffix)]
            )

            # Skip patients with ≤1 image if required (for patch datasets only)
            if args['N_wsi'] > 1 and dataset in ['lgg_patches', 'brca_patches'] and len(image_files) <= 1:
                continue

            # Select up to N_wsi images per patient
            selected_images = image_files[:args['N_wsi']]

            for image_name in selected_images:
                image_path = os.path.join(patient_folder, image_name)

                # Load and preprocess image: resize, normalize, convert to tensor
                image = Image.open(image_path).convert('RGB').resize((128, 128))
                image_tensor = torch.tensor(
                    np.array(image) / 255.0, dtype=torch.float32
                ).permute(2, 0, 1)  # Convert to (C, H, W) format
                image_tensors.append(image_tensor)

            # Stack tensors for this patient if any images were found
            if len(image_tensors) > 0: 
                patient_tensor = torch.stack(image_tensors)
                all_patient_tensors[idx] = patient_tensor
                patient_ids[idx] = patient_id

    # Filter out empty entries (patients without valid image tensors)
    all_patient_tensors = [t for t in all_patient_tensors if t is not None] 
    patient_ids = [pid for pid in patient_ids if pid is not None] 

    # Stack all patient tensors: shape (L, N, C, H, W), where L = number of patients
    if len(all_patient_tensors) > 0:
        final_tensor = torch.stack(all_patient_tensors).to(device)
        
        # Optional: Subsample BRCA patients due to large dataset size (computational limitations)
        if dataset.startswith('brca_') and args.get('patient_limit', False):
            odd_indices = torch.arange(1, final_tensor.shape[0], step=2)[:500]
            final_tensor = final_tensor[odd_indices, :]
            patient_ids = [patient_ids[i.item()] for i in odd_indices]
        
        # Summary output
        print(f"Total patients processed: {len(patient_ids)}")
        print(f"Final tensor shape: {final_tensor.shape}")
        return final_tensor
    else:
        print("No images found to process.")
        return None



# This function was used in internal experiments to preprocess whole slide images (WSIs) 
# that had not been processed using the CLAM pipeline. Unlike other inputs derived from 
# CLAM (e.g., patches, heatmaps, or masks), this function handles raw WSI representations.

def preprocess_wsi(dataset, args, device):
    # Load the appropriate manifest based on the dataset name
    if dataset == 'lgg_wsi':
        manifest_path = '../data_download/Manifest/LGG/filtered_manifest/filtered_gdc_manifest.wsi.txt'
    elif dataset == 'brca_wsi':
        manifest_path = '../data_download/Manifest/BRCA/filtered_manifest/filtered_gdc_manifest.wsi.txt'
    
    # Read the manifest file
    manifest_df = pd.read_csv(manifest_path, sep='\t')

    # Extract patient IDs in the same order as in the manifest (without .svs extension)
    manifest_ids = manifest_df['filename'].str.replace('.svs', '', regex=False).tolist()

    # Define the base directory containing WSI images
    wsi_input_dir = os.path.join(args['wsi_input_dir'], dataset)

    # Initialize placeholder lists to store tensors and patient IDs
    all_patient_tensors = [None] * len(manifest_ids)
    patient_ids = [None] * len(manifest_ids)

    # Process each patient's folder
    for idx, patient_id in enumerate(manifest_ids):
        patient_folder = os.path.join(wsi_input_dir, patient_id)
        if os.path.isdir(patient_folder):
            image_tensors = []

            # Get all .png image files within the patient's directory
            image_files = sorted([
                image_name for image_name in os.listdir(patient_folder) 
                if image_name.endswith('.png')
            ])

            # Limit the number of images to the specified N_wsi
            n_images = args['N_wsi']
            selected_images = image_files[:n_images]

            # Load and process each selected image
            for image_name in selected_images:
                image_path = os.path.join(patient_folder, image_name)
                image = Image.open(image_path).convert('RGB').resize((128, 128))
                image_tensor = torch.tensor(
                    np.array(image) / 255.0, dtype=torch.float32
                ).permute(2, 0, 1)  # Reorder to (C, H, W)
                image_tensors.append(image_tensor)

            # Stack images into a single tensor for the current patient
            if len(image_tensors) > 0:
                patient_tensor = torch.stack(image_tensors)
                all_patient_tensors[idx] = patient_tensor
                patient_ids[idx] = patient_id

    # Remove entries with no image data
    all_patient_tensors = [t for t in all_patient_tensors if t is not None]
    patient_ids = [pid for pid in patient_ids if pid is not None]

    # Stack all patient tensors into a final tensor (L, N, C, H, W)
    final_tensor = torch.stack(all_patient_tensors).to(device)

    # Print a preview of processed patient IDs
    print("\nFirst patients processed and added to tensor:")
    print(patient_ids[:min(len(patient_ids), 5)])  # Show up to 5 patient IDs
    print(f"Total number of patients processed: {len(patient_ids)}")
    print(f"Final tensor shape: {final_tensor.shape}")
    return final_tensor
