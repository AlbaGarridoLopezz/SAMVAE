# Author: Alba Garrido López
# Email: alba.garrido.lopez@upm.es 

# Packages to import 
import os  
import numpy as np
import torch
import pandas as pd 
from pycox.evaluation import EvalSurv 
from statsmodels.stats.proportion import proportion_confint
from tabulate import tabulate 
from utils import check_file  

# This warning type is removed due to pandas future warnings
# https://github.com/havakv/pycox/issues/162. Incompatibility between pycox and pandas' new version
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# ------------------------------------------------------------------------------------------------------
#                                     UTILS FUNCTIONS
# ------------------------------------------------------------------------------------------------------

def bern_conf_interval(n, mean, ibs=False):
    # Confidence interval
    ci_bot, ci_top = proportion_confint(count=mean * n, nobs=n, alpha=0.1, method='beta')
    if mean < 0.5 and not ibs:
        ci_bot_2 = 1 - ci_top
        ci_top = 1 - ci_bot
        ci_bot = ci_bot_2
        mean = 1 - mean

    return np.round(ci_bot, 4), mean, np.round(ci_top, 4)



# ------------------------------------------------------------------------------------------------------
#                               VALIDATION FUNCTIONS
# ------------------------------------------------------------------------------------------------------
def obtain_c_index(surv_f, time, censor):
    # Evaluate using PyCox c-index
    ev = EvalSurv(surv_f, time, censor, censor_surv='km')
    ci = ev.concordance_td()

    # Obtain also IBS
    time_grid = np.linspace(time.min(), time.max(), 100)
    ibs = ev.integrated_brier_score(time_grid)
    return ci, ibs



# ------------------------------------------------------------------------------------------------------
#                               BEST RESULTS FOR EACH FOLD  
# ------------------------------------------------------------------------------------------------------
 
# Best results for each fold
def get_fold_best_seed_results(results, param_comb, n_seeds, n_folds, seeds_eval=3):
  
    best_results = {'avg_ci': 0.0, 'avg_ibs': 0.0, 'best_cis': [], 'param_comb': ''} # best cis
    best_results_ibs = {'avg_ci': 0.0, 'avg_ibs': 0.0, 'best_ibs': [], 'param_comb': ''}
    best_ci = 0.0
    best_ibs = float('inf')  
    
    # Stores the best results for each combination of parameters
    best_results_per_param = []
    for params in param_comb:
        model_params = str(params['latent_dim']) + '_' + str(params['hidden_size'])
        fold_results = []
        for fold in range(n_folds):
            # Average results from folds
            ci_per_seed = [results[model_params][seed][fold]['ci'][-1] for seed in range(n_seeds)]
            ibs_per_seed = [results[model_params][seed][fold]['ibs'][-1] for seed in range(n_seeds)]
             
            differences = []
            best_idx = []
            for i in range(len(ci_per_seed)):
                diff = ci_per_seed[i][0][1] - ibs_per_seed[i][0][1]
                
                if seeds_eval > len(differences):
                    best_idx.append(i)
                    differences.append(diff)
                else:
                    min_dif_idx = np.argmin(differences)
                    if diff > differences[min_dif_idx]:
                        differences[min_dif_idx] = diff
                        best_idx[min_dif_idx] = i
            fold_results.append((fold, np.mean(np.array([ci[0][1] for ci in ci_per_seed])[best_idx]),
                                 [ci_per_seed[idx] for idx in best_idx],
                                 np.mean(np.array([ibs[0][1] for ibs in ibs_per_seed])[best_idx]),
                                 [ibs_per_seed[idx] for idx in best_idx]))
            
        avg_ci = sum([x[1] for x in fold_results]) / n_folds
        avg_ibs = sum([x[3] for x in fold_results]) / n_folds
        
        best_results_per_param.append({
            'param_comb': model_params,
            'avg_ci': avg_ci,
            'avg_ibs': avg_ibs,
            'best_cis': [x[2] for x in fold_results],
            'best_ibs': [x[4] for x in fold_results]
        })
        
        if avg_ci > best_ci:
            best_ci = avg_ci
            best_results['avg_ci'] = avg_ci
            best_results['param_comb'] = model_params
            best_results['best_cis'] = [x[2] for x in fold_results]
            best_results['best_ibs'] = [x[4] for x in fold_results]
            best_results['avg_ibs'] = avg_ibs
            
        if avg_ibs < best_ibs:  
            best_ibs = avg_ibs
            best_results_ibs['avg_ci'] = avg_ci
            best_results_ibs['param_comb'] = model_params
            best_results_ibs['best_cis'] = [x[2] for x in fold_results]
            best_results_ibs['best_ibs'] = [x[4] for x in fold_results]
            best_results_ibs['avg_ibs'] = avg_ibs
                
    return best_results, best_results_per_param, best_results_ibs

 
def get_fold_best_seed_results_cr(results, param_comb, n_seeds, n_folds, seeds_eval=3):
    best_results = {'avg_ci': 0.0, 'avg_ibs': 0.0, 'best_cis': [], 'best_ibs': [], 'param_comb': ''}  
    best_results_ibs = {'avg_ci': 0.0, 'avg_ibs': 0.0, 'best_cis': [], 'best_ibs': [], 'param_comb': ''}
    best_ci = 0.0
    best_ibs = float('inf')   
    
    best_results_per_param = []
    
    for params in param_comb:
        model_params = str(params['latent_dim']) + '_' + str(params['hidden_size'])
        fold_results = []
        
        for fold in range(n_folds):
            ci_per_seed = [results[model_params][seed][fold]['ci'][-1] for seed in range(n_seeds)]
            ibs_per_seed = [results[model_params][seed][fold]['ibs'][-1] for seed in range(n_seeds)]
            
            best_cis_per_tensor = []
            best_ibs_per_tensor = []
            num_tensors = len(ci_per_seed[0])
            
            for tensor_idx in range(num_tensors):  
                # For each risk group, gather all values from all seeds
                cis_tensor = [np.mean(ci_per_seed[seed][tensor_idx]) for seed in range(n_seeds)]
                ibs_tensor = [np.mean(ibs_per_seed[seed][tensor_idx]) for seed in range(n_seeds)]
                
                # Get the indices of the top 3 C-indexes (highest) and top 3 IBS values (lowest)
                best_cis_idx = np.argsort(cis_tensor)[-seeds_eval:][::-1]
                best_ibs_idx = np.argsort(ibs_tensor)[:seeds_eval]
                
                # Save the best values
                best_cis_per_tensor.append([ci_per_seed[idx][tensor_idx] for idx in best_cis_idx])
                best_ibs_per_tensor.append([ibs_per_seed[idx][tensor_idx] for idx in best_ibs_idx])
            
            fold_results.append((fold, [[t[1] for t in ci] for ci in ci_per_seed[0]],  
                best_cis_per_tensor, [[t[1] for t in ibs] for ibs in ibs_per_seed[0]],
                best_ibs_per_tensor
            ))

        avg_ci = [sum(ci) / n_folds for ci in zip(*[sum(x[1], []) for x in fold_results])]
        avg_ibs = [sum(ibs) / n_folds for ibs in zip(*[sum(x[3], []) for x in fold_results])]
        
        best_results_per_param.append({
            'param_comb': model_params,
            'avg_ci': avg_ci,
            'avg_ibs': avg_ibs,
            'best_cis': [x[2] for x in fold_results],
            'best_ibs': [x[4] for x in fold_results]
        })
        
        if (isinstance(avg_ci, list) and np.mean(avg_ci) > best_ci) or (isinstance(avg_ci, float) and avg_ci > best_ci):
            best_ci = np.mean(avg_ci) if isinstance(avg_ci, list) else avg_ci
            best_results.update({
                'avg_ci': avg_ci,
                'param_comb': model_params,
                'best_cis': [x[2] for x in fold_results],
                'best_ibs': [x[4] for x in fold_results],
                'avg_ibs': avg_ibs
            })
        
        if (isinstance(avg_ibs, list) and np.mean(avg_ibs) < best_ibs) or (isinstance(avg_ibs, float) and avg_ibs < best_ibs):
            best_ibs = np.mean(avg_ibs) if isinstance(avg_ibs, list) else avg_ibs
            best_results_ibs.update({
                'avg_ci': avg_ci,
                'param_comb': model_params,
                'best_cis': [x[2] for x in fold_results],
                'best_ibs': [x[4] for x in fold_results],
                'avg_ibs': avg_ibs
            })
                
    return best_results, best_results_per_param, best_results_ibs

def load_results_and_labels(args, file_prefix='results'):
            """
            Loads results (and labels if competing_risks=True) from the output directory.
            If precomputed best results files exist, they are loaded. Otherwise, they are computed.
            """
            output_dir = args['output_dir']
            suffix = '_cr.pkl' if args['competing_risks'] else '.pkl'
            results_path = os.path.join(output_dir, file_prefix + suffix)
            best_results_path = os.path.join(output_dir, 'best_results' + suffix)
            best_results_per_param_path = os.path.join(output_dir, 'best_results_per_param' + suffix)
            best_results_ibs_path = os.path.join(output_dir, 'best_results_ibs' + suffix)
            print(f"path:{results_path}")
            results = check_file(results_path, 'Results file for model does not exist.')

            if args['competing_risks']:
                labels_path = os.path.abspath(os.path.join(output_dir, '..', '..', 'labels.pkl'))
                labels = check_file(labels_path, 'Labels file for model does not exist.')

                if (
                    os.path.exists(best_results_path) and
                    os.path.exists(best_results_per_param_path) and
                    os.path.exists(best_results_ibs_path)
                ):
                    best_results = check_file(best_results_path, 'best_results file for model does not exist.')
                    best_results_per_param = check_file(best_results_per_param_path,  'best_results_per_param file for model does not exist.')
                    best_results_ibs = check_file(best_results_ibs_path, 'best_results_ibs file for model does not exist.')
                else:
                    best_results, best_results_per_param, best_results_ibs = get_fold_best_seed_results_cr(
                        results, args['param_comb'], args['n_seeds'], args['n_folds']
                    )

                return results, best_results, best_results_per_param, best_results_ibs, labels

            else:
                if (
                    os.path.exists(best_results_path) and
                    os.path.exists(best_results_per_param_path) and
                    os.path.exists(best_results_ibs_path)
                ):
                    best_results = check_file(best_results_path, 'best_results file for model does not exist.')
                    best_results_per_param = check_file(best_results_per_param_path,  'best_results_per_param file for model does not exist.')
                    best_results_ibs = check_file(best_results_ibs_path, 'best_results_ibs file for model does not exist.')
                else:
                    best_results, best_results_per_param, best_results_ibs = get_fold_best_seed_results(
                        results, args['param_comb'], args['n_seeds'], args['n_folds']
                    )

                return results, best_results, best_results_per_param, best_results_ibs