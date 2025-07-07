# Author: Alba Garrido López
# Email: alba.garrido.lopez@upm.es 

# Packages to import
import os
import pickle
import warnings
import numpy as np 
from tabulate import tabulate
from scipy.stats import ttest_ind 
from statsmodels.stats.multitest import multipletests 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# ------------------------------------------------------------------------------------------------------
#                                              TABLES
# ------------------------------------------------------------------------------------------------------

# Hyperparameter Optimization in Survival Analysis
def get_table(results, args, datasets_used, modes_used, best_results_per_param, best_results_ci, best_results_ibs):
    print('\n\n---- Hyperparameter Optimization in Survival Analysis ----')
    datasets = datasets_used.split('_')[0]   
    n_threads = args['n_threads']
    times_table = []

    for params in args['param_comb']:
        latent_dim = params.get('latent_dim', '-')
        hidden_size = params.get('hidden_size', '-') 
        model_params = f"{latent_dim}_{hidden_size}"

        # Initialize metric accumulators
        avg_ci, avg_ibs = 0.0, 0.0
        std_ci, std_ibs, std_ci_minus_ibs = 0.0, 0.0, 0.0
        pval_cindex, pval_ibs = None, None

        # Find corresponding results for this parameter combination
        for param_result in best_results_per_param:
            if model_params == param_result['param_comb']:
                avg_ci = param_result['avg_ci']
                avg_ibs = param_result['avg_ibs']

                cis_list, ibs_list = [], []

                # Collect all C-index and IBS values across folds and seeds
                for fold_idx in range(len(param_result['best_cis'])):
                    fold_cis_seeds = param_result['best_cis'][fold_idx]  
                    fold_ibs_seeds = param_result['best_ibs'][fold_idx]
                    
                    c_values = [seed[1] for seed_list in fold_cis_seeds for seed in seed_list]  
                    i_values = [seed[1] for seed_list in fold_ibs_seeds for seed in seed_list]
                    
                    cis_list.extend(c_values)
                    ibs_list.extend(i_values)

                # Compute standard deviations
                std_ci = np.std(cis_list)
                std_ibs = np.std(ibs_list)
                std_ci_minus_ibs = np.std([c + 1 - i for c, i in zip(cis_list, ibs_list)])

                # Perform statistical tests against best model
                cis_best = [seed[1] for fold in best_results_ci['best_cis'] for seed_list in fold for seed in seed_list]
                ibs_best = [seed[1] for fold in best_results_ibs['best_ibs'] for seed_list in fold for seed in seed_list]

                # One-sided t-tests
                pval_cindex = f"{ttest_ind(cis_best, cis_list, equal_var=False, alternative='greater').pvalue:.3f}"
                pval_ibs    = f"{ttest_ind(ibs_best, ibs_list, equal_var=False, alternative='less').pvalue:.3f}"
        
        # Compute CI - IBS
        ci_minus_ibs = avg_ci - avg_ibs

        # Append to table
        times_table.append([
            datasets, modes_used, latent_dim, hidden_size,
            f"{avg_ci:.3f} ± {std_ci:.3f}",
            f"{avg_ibs:.3f} ± {std_ibs:.3f}",
            f"{ci_minus_ibs:.3f} ± {std_ci_minus_ibs:.3f}",
            pval_cindex, pval_ibs, None, None
        ])
        
    # Holm-Bonferroni correction for multiple hypothesis testing
    pvals_cindex = [float(row[7]) for row in times_table if row[7] is not None]
    pvals_ibs    = [float(row[8]) for row in times_table if row[8] is not None]

    _, corrected_cindex, _, _ = multipletests(pvals_cindex, alpha=0.05, method='bonferroni')
    _, corrected_ibs, _, _    = multipletests(pvals_ibs, alpha=0.05, method='bonferroni') 

    for i, row in enumerate(times_table):
        if row[7] is not None:
            row[9] = f"{corrected_cindex[i]:.3f}"   # Corrected p-value (CI)
            row[10] = f"{corrected_ibs[i]:.3f}"     # Corrected p-value (IBS)

    # Print and save final results table
    headers = [
        'DATASETS', 'MODES', 'LATENT', 'HIDDEN', 'C-INDEX', 'IBS', 'CI - IBS',
        'p-value (CI)', 'p-value (IBS)', 'p-value-HB (CI)', 'p-value-HB (IBS)'
    ]

    table_str = tabulate(times_table, headers=headers, tablefmt='grid')
    print(table_str)

    # Save table to file
    output_dir = args['output_dir'] 
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'total_results_table.txt'), 'w') as file:
        file.write(table_str)

 

# Hyperparameter Optimization for Competing Risks  
def get_table_cr(args, datasets_used, modes_used, best_results_per_param, labels):
    print('\n\n----Hyperparameter Optimization for Competing Risks ----')
    
    datasets = datasets_used.split('_')[0]
    output_dir = args['output_dir']
    headers = ['DATASETS', 'MODES', 'LATENT', 'HIDDEN', 'RISK', 'C-INDEX', 'IBS',
               'AVG C-INDEX', 'AVG IBS', 'CI -IBS', 'p-values CI', 'p-values IBS','p-value-HB (CI)', 'p-value-HB (IBS)']
    times_table = []

    for params in args['param_comb']:
        latent_dim = params['latent_dim']
        hidden_size = params['hidden_size']
        model_params = f"{latent_dim}_{hidden_size}"
        param_result = next((r for r in best_results_per_param if r['param_comb'] == model_params), None)
        avg_cis_all_risks = []
        avg_ibs_all_risks = []
        temp_rows = []

        for risk_idx, risk_label in enumerate(labels):
            cis_list = [seed[1] for fold in param_result['best_cis'] for seed_list in fold[risk_idx] for seed in seed_list]
            ibs_list = [seed[1] for fold in param_result['best_ibs'] for seed_list in fold[risk_idx] for seed in seed_list]
            avg_ci = np.mean(cis_list)
            avg_ibs = np.mean(ibs_list)
            std_ci = np.std(cis_list)
            std_ibs = np.std(ibs_list)
            avg_cis_all_risks.append(avg_ci)
            avg_ibs_all_risks.append(avg_ibs)
            
            # Get best performing models for statistical comparison
            best_group_cis = max(best_results_per_param, key=lambda x: np.mean([seed[1] for fold in x['best_cis'] for seed_list in fold[risk_idx] for seed in seed_list]))
            best_group_ibs = min(best_results_per_param, key=lambda x: np.mean([seed[1] for fold in x['best_ibs'] for seed_list in fold[risk_idx] for seed in seed_list]))
            best_cis = [seed[1] for fold in best_group_cis['best_cis'] for seed_list in fold[risk_idx] for seed in seed_list]
            best_ibs = [seed[1] for fold in best_group_ibs['best_ibs'] for seed_list in fold[risk_idx] for seed in seed_list]

            pval_cindex = "{:.3f}".format(ttest_ind(best_cis, cis_list, equal_var=False, alternative='greater').pvalue)
            pval_ibs = "{:.3f}".format(ttest_ind(best_ibs, ibs_list, equal_var=False, alternative='less').pvalue)

            temp_rows.append([datasets, modes_used, latent_dim, hidden_size, risk_label, f"{avg_ci:.3f} ± {std_ci:.3f}",
                f"{avg_ibs:.3f} ± {std_ibs:.3f}", None, None, None, pval_cindex, pval_ibs, None, None
            ])
             
        avg_ci_per_param = np.mean(avg_cis_all_risks)
        avg_ibs_per_param = np.mean(avg_ibs_all_risks)
        avg_diff_per_param = avg_ci_per_param - avg_ibs_per_param
 
        for row in temp_rows:
            row[7] = f"{avg_ci_per_param:.3f}"
            row[8] = f"{avg_ibs_per_param:.3f}"
            row[9] = f"{avg_diff_per_param:.3f}"
            times_table.append(row)
    
    # Extract p-values and apply Holm-Bonferroni correction
    pvals_cindex = [float(row[10]) for row in times_table]
    pvals_ibs = [float(row[11]) for row in times_table]
    _, corrected_pvals_cindex, _, _ = multipletests(pvals_cindex, alpha=0.05, method='bonferroni')
    _, corrected_pvals_ibs, _, _ = multipletests(pvals_ibs, alpha=0.05, method='bonferroni')
    for i, row in enumerate(times_table):
        row[12] = f"{corrected_pvals_cindex[i]:.3f}"  # Corrected CI p-value 
        row[13] = f"{corrected_pvals_ibs[i]:.3f}"     # Corrected IBS p-value 

    table_str = tabulate(times_table, headers=headers, tablefmt='grid')
    print(table_str)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'total_results_table_cr.txt'), 'w') as file:
        file.write(table_str)
 

# Intermediate Combinations in Survival Analysis (without comparing with other models)   
def get_table_no_hpo(args, datasets_used, modes_used, best_results_ci, best_results_ibs):
    print('\n\n---- Intermediate Combinations in Survival Analysis (without comparing with other models) ----')
    datasets = datasets_used.split('_')[0]
    times_table = []
    
    avg_ci = best_results_ci['avg_ci']
    avg_ibs = best_results_ibs['avg_ibs']
    ci_minus_ibs = avg_ci - avg_ibs
    
    times_table.append([
        datasets, modes_used,
        f"{avg_ci:.3f}",
        f"{avg_ibs:.3f}",
        f"{ci_minus_ibs:.3f}"
    ])
    
    headers = ['DATASETS', 'MODES', 'C-INDEX', 'IBS', 'CI - IBS']
    table_str = tabulate(times_table, headers=headers, tablefmt='grid')
    print(table_str)
    
    output_dir = args['output_dir']
    with open(os.path.join(output_dir, 'results_table_no_hpo.txt'), 'w') as file:
        file.write(table_str)
     

# Intermediate Combinations in Competing Risks (without comparing with other models)     
def get_table_cr_no_hpo(args, datasets_used, modes_used, best_results, best_results_ibs, labels):
    print('\n\n---- Intermediate Combinations in Competing Risks (without comparing with other models) ----')
    datasets = datasets_used.split('_')[0]
    times_table = []
    
    avg_ci_total, avg_ibs_total = [], []
    risk_results = []
    
    for risk_idx, risk_label in enumerate(labels):
        avg_ci = best_results['avg_ci'][risk_idx]
        avg_ibs = best_results_ibs['avg_ibs'][risk_idx]
        avg_ci_total.append(avg_ci)
        avg_ibs_total.append(avg_ibs)
        
        risk_results.append([
            datasets, modes_used, risk_label,
            f"{avg_ci:.3f}",
            f"{avg_ibs:.3f}",
            "", "", ""
        ])
    
    if avg_ci_total and avg_ibs_total:
        mean_ci = np.mean(avg_ci_total)
        mean_ibs = np.mean(avg_ibs_total)
        ci_minus_ibs = mean_ci - mean_ibs
        
        for row in risk_results:
            row[5] = f"{mean_ci:.3f}"
            row[6] = f"{mean_ibs:.3f}"
            row[7] = f"{ci_minus_ibs:.3f}"
    
    times_table.extend(risk_results)
    headers = ['DATASETS', 'MODES', 'RISK', 'C-INDEX', 'IBS', 'AVG C-INDEX', 'AVG IBS', 'CI - IBS']
    
    table_str = tabulate(times_table, headers=headers, tablefmt='grid')
    print(table_str)
    
    output_dir = args['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'results_table_cr_no_hpo.txt'), 'w') as file:
        file.write(table_str)
     
    
# Intermediate Combinations and Final Results in Survival Analysis          
def get_table_no_hpo_with_pvals(args, datasets_used, modes_used):
    print('\n\n---- Intermediate Combinations and Final Results in Survival Analysis ----')
    output_dir = args['output_dir']
    experiment_path = os.path.normpath(os.path.join(output_dir, '..', '..'))
    
    labels = ['0'] # survival Analysis does not have risks, so we use a single label '0'
    best_paths = get_best_paths_by_group(experiment_path, labels)
 
    num_seeds =  args['n_seeds']   
    if num_seeds == 10:
        base_cis_combined, base_ibs_combined = load_reference_lists_sa(best_paths['all'][0], best_paths['all'][1])
    else:
        base_cis_wsi, base_ibs_wsi = load_reference_lists_sa(best_paths['wsi'][0], best_paths['wsi'][1])
        base_cis_list, base_ibs_list = load_reference_lists_sa(best_paths['omic'][0], best_paths['omic'][1])
    print(f"Detected number of seeds: {num_seeds}")
 
    unified_table = []
    table_rows_wsi = []
    table_rows_omic = []
    pvals_cindex_omic = []
    pvals_ibs_omic = []
    pvals_cindex_wsi = []
    pvals_ibs_wsi = []
    pvals_cindex_all = []
    pvals_ibs_all = []

    for root, dirs, files in os.walk(experiment_path):
        if 'best_results.pkl' in files and 'best_results_ibs.pkl' in files:
            with open(os.path.join(root, 'best_results.pkl'), 'rb') as f:
                ci_data = pickle.load(f)
            with open(os.path.join(root, 'best_results_ibs.pkl'), 'rb') as f:
                ibs_data = pickle.load(f)

            avg_ci = ci_data['avg_ci']
            avg_ibs = ibs_data['avg_ibs']
            ci_minus_ibs = avg_ci - avg_ibs

            cis_list = [seed[1] for fold in ci_data['best_cis'] for seed_list in fold for seed in seed_list]
            ibs_list = [seed[1] for fold in ibs_data['best_ibs'] for seed_list in fold for seed in seed_list]
            std_ci = np.std(cis_list)
            std_ibs = np.std(ibs_list)
            std_ci_minus_ibs = np.std(np.array(cis_list) - np.array(ibs_list))

            relative_path = os.path.relpath(root, experiment_path)
            mode_name = os.path.dirname(relative_path)         
            if num_seeds == 10:
                pval_ci = ttest_ind(base_cis_combined, cis_list, equal_var=False, alternative='greater').pvalue
                pval_ibs = ttest_ind(base_ibs_combined, ibs_list, equal_var=False, alternative='less').pvalue

            else:
                if 'wsi_patches' in root:
                    base_ci = base_cis_wsi
                    base_ibs = base_ibs_wsi
                else:
                    base_ci = base_cis_list
                    base_ibs = base_ibs_list
                    
                pval_ci = ttest_ind(base_ci, cis_list, equal_var=False, alternative='greater').pvalue
                pval_ibs = ttest_ind(base_ibs, ibs_list, equal_var=False, alternative='less').pvalue

            row = [
                mode_name,
                f"{avg_ci:.3f} ± {std_ci:.3f}",
                f"{avg_ibs:.3f} ± {std_ibs:.3f}",
                f"{ci_minus_ibs:.3f} ± {std_ci_minus_ibs:.3f}",
                f"{pval_ci:.3f}",
                f"{pval_ibs:.3f}",
                None,  # Holm-Bonferroni corrected p-value for CI
                None   # Holm-Bonferroni corrected p-value for IBS
            ]

            if num_seeds == 10:
                unified_table.append(row)
                pvals_cindex_all.append(pval_ci)
                pvals_ibs_all.append(pval_ibs)
            elif 'wsi_patches' in root:
                table_rows_wsi.append(row)
                pvals_cindex_wsi.append(pval_ci)
                pvals_ibs_wsi.append(pval_ibs)
            else:
                table_rows_omic.append(row)
                pvals_cindex_omic.append(pval_ci)
                pvals_ibs_omic.append(pval_ibs)

    # Holm-Bonferroni correction function
    def apply_correction(rows, pvals_ci, pvals_ibs):
        if pvals_ci and pvals_ibs:
            _, corr_ci, _, _ = multipletests(pvals_ci, alpha=0.05, method='bonferroni')
            _, corr_ibs, _, _ = multipletests(pvals_ibs, alpha=0.05, method='bonferroni')
            for i, row in enumerate(rows):
                row[6] = f"{corr_ci[i]:.3f}"
                row[7] = f"{corr_ibs[i]:.3f}"
        else:
            for row in rows:
                row[6] = "N/A"
                row[7] = "N/A"

    if num_seeds == 10:
        apply_correction(unified_table, pvals_cindex_all, pvals_ibs_all)
    else:
        apply_correction(table_rows_omic, pvals_cindex_omic, pvals_ibs_omic)
        apply_correction(table_rows_wsi, pvals_cindex_wsi, pvals_ibs_wsi)

    # Headers
    headers = ['MODES', 'C-INDEX', 'IBS', 'CI - IBS', 'PVAL CI', 'PVAL IBS', 'PVAL CI HB', 'PVAL IBS HB']
    os.makedirs(output_dir, exist_ok=True)
    if num_seeds == 10:
        print('\n\n---- RESULTS UNIFIED ----')
        print(tabulate(unified_table, headers=headers, tablefmt='grid'))
        with open(os.path.join(experiment_path, 'results_table_no_hpo_unified.txt'), 'w') as f:
            f.write(tabulate(unified_table, headers=headers, tablefmt='grid')) 
    else:
        print('\n\n---- RESULTS OMICS COMBINATIONS ----')
        print(tabulate(table_rows_omic, headers=headers, tablefmt='grid'))
        print('\n\n---- RESULTS WSI PATCHES ----')
        print(tabulate(table_rows_wsi, headers=headers, tablefmt='grid'))

        with open(os.path.join(experiment_path, 'results_wsi_patches.txt'), 'w') as f:
            f.write(tabulate(table_rows_wsi, headers=headers, tablefmt='grid'))
        with open(os.path.join(experiment_path, 'results_omic_combinations.txt'), 'w') as f:
            f.write(tabulate(table_rows_omic, headers=headers, tablefmt='grid')) 
            
            

# Intermediate Combinations and Final Results in Competing Risks  
def get_table_cr_no_hpo_with_pvals(args, datasets_used, modes_used, labels):
    print('\n\n---- Intermediate Combinations and Final Results in Competing Risks ----')
    output_dir = args['output_dir']
    experiment_path = os.path.normpath(os.path.join(output_dir, '..', '..'))
    
    best_paths = get_best_paths_by_group(experiment_path, labels)
    if args['n_seeds'] == 10: 
        base_cis, base_ibs = load_reference_lists(best_paths['all'], labels)      
        table_all = build_results_table(best_paths['roots']['all'], base_cis, base_ibs, experiment_path, labels)
    else:
        base_cis_wsi, base_ibs_wsi = load_reference_lists(best_paths['wsi'], labels)
        base_cis_omic, base_ibs_omic = load_reference_lists(best_paths['omic'], labels)
        table_wsi = build_results_table(best_paths['roots']['wsi'], base_cis_wsi, base_ibs_wsi, experiment_path, labels)
        table_omic = build_results_table(best_paths['roots']['omic'], base_cis_omic, base_ibs_omic, experiment_path, labels)

    headers = ['MODES', 'RISK', 'C-INDEX', 'IBS', 'AVG CI', 'AVG IBS', 'CI- IBS','PVAL CI', 'PVAL IBS','PVAL CI HB', 'PVAL IBS HB']
    
    if args['n_seeds'] == 10:
        print('\n\n=== TABLE FINAL COMBINATIONS ===') 
        table_str = tabulate(table_all, headers=headers, tablefmt='grid')
        print(table_str)
        with open(os.path.join(experiment_path, 'final_combinations.txt'), 'w') as f:
            f.write(table_str) 
        print('\n\n=== TABLE OMICS COMBINATIONS ===')
        table_str_other = tabulate(table_omic, headers=headers, tablefmt='grid')
        print(table_str_other)
        with open(os.path.join(experiment_path, 'results_omic_combinations.txt'), 'w') as f:
            f.write(table_str_other) 
        print('\n\n=== TABLE WSI_PATCHES ===')
        table_str_wsi = tabulate(table_wsi, headers=headers, tablefmt='grid')
        print(table_str_wsi)
        with open(os.path.join(experiment_path, 'results_wsi_patches.txt'), 'w') as f:
            f.write(table_str_wsi) 
             

# ------------------------------------------------------------------------------------------------------
#                               UTILITY FUNCTIONS FOR TABLES
# ------------------------------------------------------------------------------------------------------
                
def get_best_paths_by_group(experiment_path, labels, manual_best_all_paths=None):
    groups = {'wsi': [], 'omic': [], 'all': []}
    if labels == ['0']:
        ci_file = 'best_results.pkl'
        ibs_file = 'best_results_ibs.pkl'
    else:
        ci_file = 'best_results_cr.pkl'
        ibs_file = 'best_results_ibs_cr.pkl'
 
    for root, _, files in os.walk(experiment_path):
        if ci_file in files and ibs_file in files:
            if 'wsi_patches' in root:
                groups['wsi'].append(root)
            else:
                groups['omic'].append(root) 
            groups['all'].append(root)

    def find_best_paths(roots, labels, ci_file, ibs_file):
        # If there's only one label and it is '0', we assume no competing risks
        if labels == ['0']:
            best_ci_path = None
            best_ibs_path = None
            best_ci_value = -float('inf')
            best_ibs_value = float('inf')

            for root in roots:
                try:
                    with open(os.path.join(root, ci_file), 'rb') as f:
                        ci_data = pickle.load(f)
                    with open(os.path.join(root, ibs_file), 'rb') as f:
                        ibs_data = pickle.load(f)

                    avg_ci = ci_data['avg_ci'] if isinstance(ci_data['avg_ci'], float) else ci_data['avg_ci'][0]
                    avg_ibs = ibs_data['avg_ibs'] if isinstance(ibs_data['avg_ibs'], float) else ibs_data['avg_ibs'][0]

                    if avg_ci > best_ci_value:
                        best_ci_value = avg_ci
                        best_ci_path = os.path.join(root, ci_file)

                    if avg_ibs < best_ibs_value:
                        best_ibs_value = avg_ibs
                        best_ibs_path = os.path.join(root, ibs_file)

                except Exception as e:
                    print(f"Error reading results in {root}: {e}")

            return [best_ci_path], [best_ibs_path]   

        else:
            # Competing risks  
            best_ci_paths = [None for _ in labels]
            best_ibs_paths = [None for _ in labels]
            best_ci_values = [-float('inf') for _ in labels]
            best_ibs_values = [float('inf') for _ in labels]

            for root in roots:
                try:
                    with open(os.path.join(root, ci_file), 'rb') as f:
                        ci_data = pickle.load(f)
                    with open(os.path.join(root, ibs_file), 'rb') as f:
                        ibs_data = pickle.load(f)

                    for risk_idx in range(len(labels)):
                        avg_ci = ci_data['avg_ci'][risk_idx]
                        avg_ibs = ibs_data['avg_ibs'][risk_idx]

                        if avg_ci > best_ci_values[risk_idx]:
                            best_ci_values[risk_idx] = avg_ci
                            best_ci_paths[risk_idx] = os.path.join(root, ci_file)

                        if avg_ibs < best_ibs_values[risk_idx]:
                            best_ibs_values[risk_idx] = avg_ibs
                            best_ibs_paths[risk_idx] = os.path.join(root, ibs_file)
                except Exception as e:
                    print(f"Error reading results in {root}: {e}")

            return best_ci_paths, best_ibs_paths

    return {
        'wsi': find_best_paths(groups['wsi'], labels, ci_file, ibs_file),
        'omic': find_best_paths(groups['omic'], labels, ci_file, ibs_file),
        'all': manual_best_all_paths if manual_best_all_paths else find_best_paths(groups['all'], labels, ci_file, ibs_file),
        'roots': groups
    }


def load_reference_lists_sa(ci_paths, ibs_paths):
    ci_list = []
    ibs_list = []

    for ci_path in ci_paths:
        with open(ci_path, 'rb') as f:
            ci_data = pickle.load(f)
        # 'best_cis' structure: list of folds -> list of seeds -> (seed_num, value)
        values = [seed[1] for fold in ci_data['best_cis'] for seed_list in fold for seed in seed_list]
        ci_list.extend(values)

    # Same process for IBS
    for ibs_path in ibs_paths:
        with open(ibs_path, 'rb') as f:
            ibs_data = pickle.load(f)
        values = [seed[1] for fold in ibs_data['best_ibs'] for seed_list in fold for seed in seed_list]
        ibs_list.extend(values)

    return ci_list, ibs_list


def load_reference_lists(paths, labels):
    ci_lists = [[] for _ in labels]
    ibs_lists = [[] for _ in labels]

    for i in range(len(labels)):
        if paths[0][i]:
            with open(paths[0][i], 'rb') as f:
                ci_data = pickle.load(f)
            # Extract C-index values for each risk group
            ci_lists[i] = [seed[1] for fold in ci_data['best_cis'] for seed_list in fold[i] for seed in seed_list]

        if paths[1][i]:
            with open(paths[1][i], 'rb') as f:
                ibs_data = pickle.load(f)
            # Extract IBS values for each risk group
            ibs_lists[i] = [seed[1] for fold in ibs_data['best_ibs'] for seed_list in fold[i] for seed in seed_list]

    return ci_lists, ibs_lists


def build_results_table(group_roots, base_ci_lists, base_ibs_lists, experiment_path, labels):
    
    ci_file = 'best_results_cr.pkl'
    ibs_file = 'best_results_ibs_cr.pkl'

    all_raw_pvals_ci = []
    all_raw_pvals_ibs = []
    all_rows = []

    for root in group_roots:
        try:
            with open(os.path.join(root, ci_file), 'rb') as f:
                ci_data = pickle.load(f)
            with open(os.path.join(root, ibs_file), 'rb') as f:
                ibs_data = pickle.load(f)
        except:
            continue

        rel_path = os.path.relpath(root, experiment_path)
        modes_name = os.path.dirname(rel_path)

        avg_ci_all_risks = []
        avg_ibs_all_risks = []
        all_cis_lists = []
        all_ibs_lists = []

        for i in range(len(labels)): 
            cis = [seed[1] for fold in ci_data['best_cis'] for seed_list in fold[i] for seed in seed_list]
            ibs = [seed[1] for fold in ibs_data['best_ibs'] for seed_list in fold[i] for seed in seed_list]
            all_cis_lists.append(cis)
            all_ibs_lists.append(ibs)
            avg_ci_all_risks.append(np.mean(cis))
            avg_ibs_all_risks.append(np.mean(ibs))
 
        avg_ci_global = np.mean(avg_ci_all_risks)
        avg_ibs_global = np.mean(avg_ibs_all_risks)
        diff_global = avg_ci_global - avg_ibs_global

        for i, label in enumerate(labels):
            cis = all_cis_lists[i]
            ibs = all_ibs_lists[i]
            avg_ci = np.mean(cis)
            avg_ibs = np.mean(ibs)
            std_ci = np.std(cis)
            std_ibs = np.std(ibs)

            # Statistical tests: CI (higher is better), IBS (lower is better)
            pval_cindex = ttest_ind(base_ci_lists[i], cis, equal_var=False, alternative='greater').pvalue
            pval_ibs = ttest_ind(base_ibs_lists[i], ibs, equal_var=False, alternative='less').pvalue

            all_raw_pvals_ci.append(pval_cindex)
            all_raw_pvals_ibs.append(pval_ibs)

            row = [modes_name, label,
                   f"{avg_ci:.3f} ± {std_ci:.3f}",
                   f"{avg_ibs:.3f} ± {std_ibs:.3f}",
                   f"{avg_ci_global:.3f}",
                   f"{avg_ibs_global:.3f}",
                   f"{diff_global:.3f}",
                   pval_cindex,  
                   pval_ibs]
            all_rows.append(row)

    # Holm-Bonferroni correction for multiple comparisons
    if all_raw_pvals_ci and all_raw_pvals_ibs:
        _, pvals_ci_hb, _, _ = multipletests(all_raw_pvals_ci, alpha=0.05, method='bonferroni')
        _, pvals_ibs_hb, _, _ = multipletests(all_raw_pvals_ibs, alpha=0.05, method='bonferroni')

        # Add corrected p-values to each row
        for idx, row in enumerate(all_rows):
            row[7] = f"{all_raw_pvals_ci[idx]:.3f}"  # raw CI p-value
            row[8] = f"{all_raw_pvals_ibs[idx]:.3f}"  # raw IBS p-value
            row.append(f"{pvals_ci_hb[idx]:.3f}")    # corrected CI p-value
            row.append(f"{pvals_ibs_hb[idx]:.3f}")   # corrected IBS p-value

    return all_rows