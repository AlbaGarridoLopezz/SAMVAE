# Author: Alba Garrido López
# Email: alba.garrido.lopez@upm.es 
 
# --------------------------------------
#             IMPORT PACKAGES
# --------------------------------------

import os
import csv
import pickle
import warnings
import random 
import numpy as np
import pandas as pd
import torch 
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.colors as mcolors 
from scipy.stats import binomtest, ttest_ind, sem, wilcoxon
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint 
from lifelines import KaplanMeierFitter, AalenJohansenFitter
from pycox.evaluation import EvalSurv 
from tabulate import tabulate
from colorama import Fore, Style
  
# Custom utilities
from data import split_cv_data_multimodal
from utils import create_output_dir, load_datasets, check_file

# Ignore pandas FutureWarnings (pycox compatibility issue)
warnings.simplefilter(action='ignore', category=FutureWarning)

# --------------------------------------
#           PLOTTING FUNCTIONS
# --------------------------------------

def plot_model_losses(train_loss, val_loss, fig_path, title, x_label='Epochs'):
    """
    Plot training and validation loss over epochs.
    """
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)

    plt.figure(figsize=(15, 15))
    plt.semilogy(train_loss, label='Train')
    plt.semilogy(val_loss, label='Validation')
    plt.title(title)
    plt.xlabel(x_label)
    plt.legend(loc='upper right')
    plt.savefig(fig_path)
    plt.close()


def plot_model_C_index(ci_val, fig_path=None):
    """
    Plot validation C-index with confidence intervals over epochs.
    """
    ci_val_flat = [tup[0] for tup in ci_val if len(tup) > 0]

    ci_lower = [val[0] for val in ci_val_flat]
    ci_mean = [val[1] for val in ci_val_flat]
    ci_upper = [val[2] for val in ci_val_flat]

    plt.figure(figsize=(12, 8))
    x = np.arange(len(ci_mean))
    plt.fill_between(x, ci_lower, ci_upper, color='lightblue', alpha=0.5, label='Confidence Interval')
    plt.semilogy(ci_mean, color='blue', label='Validation C-index')
    plt.title('Validation C-index with Confidence Intervals', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('C-index', fontsize=12)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.savefig(fig_path)
    plt.close()


def plot_model_IBS(ibs_val, fig_path=None):
    """
    Plot validation IBS (Integrated Brier Score) with confidence intervals over epochs.
    """
    ibs_val_flat = [tup[0] for tup in ibs_val if len(tup) > 0]

    ibs_lower = [val[0] for val in ibs_val_flat]
    ibs_mean = [val[1] for val in ibs_val_flat]
    ibs_upper = [val[2] for val in ibs_val_flat]

    plt.figure(figsize=(12, 8))
    x = np.arange(len(ibs_mean))
    plt.fill_between(x, ibs_lower, ibs_upper, color='lightblue', alpha=0.5, label='Confidence Interval')
    plt.semilogy(ibs_mean, color='blue', label='Validation IBS')
    plt.title('Validation IBS with Confidence Intervals', fontsize=14)
    plt.ylabel('IBS', fontsize=12)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.savefig(fig_path)
    plt.close()


def plot_CIF(cif_samvae, df_train, df_val, labels, dir_path):
    """
    Plot Cumulative Incidence Functions (CIF) for a few validation patients.
    """
    os.makedirs(dir_path, exist_ok=True)
    n_pats = min(5, df_val[0].shape[0])  # Plot up to 5 patients
    colors = ['r', 'b', 'g', 'k', 'm', 'c']

    for i in range(n_pats):
        plt.figure(figsize=(8, 6))
        for j in range(len(labels)):
            plt.plot(
                np.unique(df_train[0].cpu().numpy()),
                cif_samvae[j][i, :],
                label=f'SAVAE: Risk {j + 1}',
                color=colors[j]
            )
        plt.legend(loc='best')
        plt.xlabel('Time')
        plt.ylabel('CIF')
        plt.title(f'Patient {i}')
        save_path = os.path.join(dir_path, f'cif_patient_{i}.png')
        plt.savefig(save_path)
        plt.close()

 
def plot_model_vs_km(pred_risk_np, times_np, times_days, events, plot_dir):
    """
    Plot Model vs Kaplan-Meier.
    """
    mean_surv = np.mean(1 - pred_risk_np, axis=0)
    std_surv = 1.96 * sem(1 - pred_risk_np, axis=0)

    kmf = KaplanMeierFitter()
    kmf.fit(times_days, events, label="Kaplan-Meier")

    plt.figure(figsize=(12, 6))
    kmf.plot_survival_function(ci_show=True, color='purple', linewidth=2, alpha=0.3)
    plt.plot(times_np, mean_surv, color='blue', label='SAMVAE', linewidth=2)
    plt.fill_between(times_np, mean_surv - std_surv, mean_surv + std_surv, color='blue', alpha=0.5)
    plt.xlabel("Survival Time (years)")
    plt.ylabel("Survival Probability")
    plt.title("Comparison: Kaplan-Meier vs Model")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'survival_comparison.png'))
    plt.close()


def plot_individual_samples(model, data, times_np, times_days, events, plot_dir, num_patients=2, num_samples=100):
    """
    Individual Curves for Patients.
    """
    for patient_idx in range(num_patients):
        # Data
        input_patient = [x[2][patient_idx:patient_idx+1].to(model.device) for x in data[:-1]]
        surv_curves = []

        # Sample survival curves
        with torch.no_grad():
            for _ in range(num_samples):
                out = model(input_patient)
                alpha, lam = out['time_params'][0][0, 0].item(), out['time_params'][0][0, 1].item()
                lam_years = lam / 365.25
                surv_curve = np.exp(-np.power(times_np / lam_years, alpha))
                surv_curves.append(surv_curve)

        surv_curves = np.stack(surv_curves)
        mean_surv = np.mean(surv_curves, axis=0)
        median_surv = np.median(surv_curves, axis=0)
        perc_5 = np.percentile(surv_curves, 5, axis=0)
        perc_95 = np.percentile(surv_curves, 95, axis=0)

        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(times_np, mean_surv, color='blue', linewidth=2, label='SAMVAE Mean')
        plt.plot(times_np, median_surv, color='green', linewidth=2, linestyle='--', label='SAMVAE Median')
        plt.plot(times_np, perc_5, color='orange', linewidth=1.5, linestyle=':', label='5th Percentile')
        plt.plot(times_np, perc_95, color='orange', linewidth=1.5, linestyle=':', label='95th Percentile')
        plt.fill_between(times_np, perc_5, perc_95, color='orange', alpha=0.3, label='5-95% Interval')

        # Kaplan-Meier global
        kmf = KaplanMeierFitter()
        kmf.fit(times_days, events, label="Kaplan-Meier (Global)")
        kmf.plot_survival_function(ci_show=True, color='purple', alpha=0.5)

        plt.ylim(0, 1.05)
        plt.xlabel("Time")
        plt.ylabel("Survival Probability")
        plt.title(f"Patient {patient_idx} - SAMVAE Summary vs. KM (Global)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save
        plt.savefig(os.path.join(plot_dir, f'patient_{patient_idx}_sampled_survival.png'))
        plt.close()


def plot_best_and_worst_patients(model, data, times_np, times_days, events, plot_dir, num_samples=100):
    """
    Best and Worst Prognosis Patients.
    """
    font_size = 15
    legend_font_size = 15
    title_font_size = 16

    num_total_patients = data[0][2].shape[0]
    mean_surv_all = []

    for patient_idx in range(num_total_patients):
        input_patient = [x[2][patient_idx:patient_idx+1].to(model.device) for x in data[:-1]]
        surv_curves = []

        with torch.no_grad():
            for _ in range(num_samples):
                out = model(input_patient)
                alpha, lam = out['time_params'][0][0, 0].item(), out['time_params'][0][0, 1].item()
                lam_years = lam / 365.25
                surv_curve = np.exp(-np.power(times_np / lam_years, alpha))
                surv_curves.append(surv_curve)

        surv_curves = np.stack(surv_curves)
        mean_surv = np.mean(surv_curves, axis=0)
        life_expectancy_index = np.argmax(mean_surv < 0.5)
        life_expectancy_time = times_np[life_expectancy_index] if life_expectancy_index > 0 else times_np[-1]
        mean_surv_all.append((patient_idx, life_expectancy_time, mean_surv, surv_curves))

    sorted_by_life = sorted(mean_surv_all, key=lambda x: x[1])
    worst = sorted_by_life[0]
    best = sorted_by_life[-1]

    plt.figure(figsize=(12, 6))

    for label, patient_data, color in zip(['Worst Prognosis', 'Best Prognosis'], [worst, best], ['red', 'green']):
        patient_idx, _, mean_surv, surv_curves = patient_data
        perc_5 = np.percentile(surv_curves, 5, axis=0)
        perc_95 = np.percentile(surv_curves, 95, axis=0)

        plt.plot(times_np, mean_surv, color=color, linewidth=2, label=f'{label} (Patient {patient_idx})')
        plt.fill_between(times_np, perc_5, perc_95, color=color, alpha=0.2)

    kmf = KaplanMeierFitter()
    kmf.fit(times_days, events, label="Kaplan-Meier (Global)")
    ax = kmf.plot_survival_function(ci_show=True, color='purple', alpha=0.5)
    ax.set_xlabel("Time (years)", fontsize=font_size)
    ax.set_ylabel("Survival Probability", fontsize=font_size)
    ax.set_title("Best vs Worst Prognosis Patients vs Kaplan-Meier", fontsize=title_font_size)
    ax.legend(fontsize=legend_font_size)
    ax.grid(True)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'best_vs_worst_prognosis.png'))
    plt.close()

 
    
def plot_model_vs_aj_competing_risks(pred_risk_all, times_np, times_days, events, plot_dir):
    """
    Model vs Aalen-Johansen for Competing Risks.
    """
    os.makedirs(plot_dir, exist_ok=True)

    num_risks = len(pred_risk_all)
    colors = ['blue', 'green', 'red', 'orange', 'cyan', 'magenta']

    # 1. Individual plot per risk
    for risk_idx in range(num_risks):
        pred_risk_np = pred_risk_all[risk_idx]
        model_cif = np.mean(1 - pred_risk_np, axis=0)
        model_std = 1.96 * sem(1 - pred_risk_np, axis=0)

        # Mask for the current risk events
        risk_events_mask = events == (risk_idx + 1)

        # Kaplan-Meier estimation
        kmf = KaplanMeierFitter()
        kmf.fit(durations=times_days[risk_events_mask],
                event_observed=risk_events_mask[risk_events_mask],
                label=f"KM - Risk {risk_idx}")

        # Plotting
        plt.figure(figsize=(10, 5))
        kmf.plot_survival_function(ci_show=True, color=colors[risk_idx % len(colors)],
                                   linewidth=2, alpha=0.3)
        plt.plot(times_np, model_cif, color='black', label=f'Model - Risk {risk_idx}', linewidth=2)
        plt.fill_between(times_np, model_cif - model_std, model_cif + model_std, color='black', alpha=0.3)
        plt.xlabel("Time (years)")
        plt.ylabel("Cumulative Incidence")
        plt.title(f"Risk {risk_idx}: KM vs Model")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"cif_vs_km_risk{risk_idx}.png"))
        plt.close()

    # 2. Combined plot for all risks (model vs Aalen-Johansen)
    plt.figure(figsize=(12, 6))
    for risk_idx in range(num_risks):
        pred_risk_np = pred_risk_all[risk_idx]
        model_cif = np.mean(1 - pred_risk_np, axis=0)

        # Empirical CIF using Aalen-Johansen
        ajf = AalenJohansenFitter()
        ajf.fit(durations=times_days, event_observed=events, event_of_interest=(risk_idx + 1))

        aj_times = ajf.cumulative_density_.index
        aj_cif = ajf.cumulative_density_[f"CIF_{risk_idx + 1}"]

        plt.plot(aj_times, 1 - aj_cif, label=f"AJ - Risk {risk_idx}", linestyle="--",
                 linewidth=2, color=colors[risk_idx % len(colors)])
        plt.plot(times_np, model_cif, label=f"Model - Risk {risk_idx}", linewidth=2,
                 color=colors[risk_idx % len(colors)])

    plt.xlabel("Time (years)")
    plt.ylabel("Cumulative Incidence")
    plt.title("All Risks: Aalen-Johansen vs Model CIF")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "cif_vs_aj_all_risks.png"))
    plt.close()

    
    
def plot_survival(model, args, log_name, all_results_for_plot, fold=0):
    """
    Main function for plots.
    """
    create_output_dir('plots_samvae_sa', args)
    plot_dir = args['plots_output_dir']

    # Load trained model
    model.load_state_dict(torch.load(log_name + '.pt'))
    model.eval()

    # Load and prepare dataset splits
    all_tensors = load_datasets(args)
    cv_data_multimodal, _ = split_cv_data_multimodal(all_tensors, args['n_folds'], time_dist=args['time_distribution'])
    data = [cv_data_multimodal[dataset_idx][fold] for dataset_idx in range(len(cv_data_multimodal))]

    cov_val = [d[2] for d in data[:-1]]
    time_train = [data[-1][0][:, 0]]
    time_val = [data[-1][2][:, 0]]
    censor_val = [data[-1][2][:, 1]]

    # Locate and load clinical dataset
    clinical_dataset = next((ds for ds in args['datasets'] if ds.endswith('_clinical') or ds.endswith('_clinical_cr')), None)
    clinical_df = None
    if clinical_dataset:
        clinical_csv_path = os.path.join(args['clinical_input_dir'], clinical_dataset, f"{clinical_dataset}.csv")
        clinical_df = pd.read_csv(clinical_csv_path)
        clinical_df = clinical_df.drop(columns=['time', 'event'], errors='ignore')

    # Load time/event data
    time_event_path = os.path.join(args['time_event_input_dir'], args['time_event'][0], 'data.pt')
    data_real = torch.load(time_event_path)
    times_days = data_real[:, 0].cpu().numpy()
    times_years = times_days / 365.25
    events = data_real[:, 1].cpu().numpy()

    # --- PLOTTING ---
    if not args['competing_risks']:
        # Regular survival
        _, _, pred_risk_np, times_np = model.calculate_risk(time_train, cov_val, censor_val, time_val)
        times_np = times_np / 365.25  # Convert days to years

        if clinical_df is not None:
            plot_interactive_survival(pred_risk_np, times_np, plot_dir, clinical_df=clinical_df)

        plot_model_vs_km(pred_risk_np, times_np, times_years, events, plot_dir)
        plot_individual_samples(model, data, times_np, times_years, events, plot_dir, num_patients=2)

        if clinical_df is not None:
            plot_best_and_worst_patients_interactive(model, data, times_np, times_years, events, clinical_df, plot_dir)

        plot_best_and_worst_patients(model, data, times_np, times_years, events, plot_dir)

        all_results_for_plot.append({
            'pred_risk_np': pred_risk_np,
            'times_np': times_np,
            'times_years': times_years,
            'events': events,
            'label': args['datasets']
        })

    else:
        # Competing risks
        _, _, pred_risk_all, _, times_np = model.calculate_risk(time_train, cov_val, censor_val, time_val)
        times_np = times_np / 365.25

        if clinical_df is not None:
            plot_interactive_cif(pred_risk_all, times_np, plot_dir, clinical_df=clinical_df)

        plot_model_vs_aj_competing_risks(pred_risk_all, times_np, times_years, events, plot_dir)

        all_results_for_plot.append({
            'pred_risk_np': pred_risk_all,
            'times_np': times_np,
            'times_years': times_years,
            'events': events,
            'label': args['datasets']
        })

    # Save plot data 
    label_name = '_'.join(args['datasets'][:-1]) if isinstance(args['datasets'], list) else args['datasets'][:-1]
    pkl_filename = os.path.join(plot_dir, f"results_{label_name}.pkl")
    with open(pkl_filename, 'wb') as f:
        pickle.dump(all_results_for_plot, f)

         

def find_log_names(base_path, target_subpath='seed_0/model_fold_0'):
    """
    Find full paths to model checkpoints given a base directory structure.
    """
    log_names = []
    for carpeta in os.listdir(base_path):
        path1 = os.path.join(base_path, carpeta) 
        subdirs = [d for d in os.listdir(path1) if os.path.isdir(os.path.join(path1, d))]
        for sub in subdirs:
            path2 = os.path.join(path1, sub)  
            seed_path = os.path.join(path2, target_subpath)
            log_names.append(seed_path)
    return log_names


 
def load_existing_fold_results(existing_models, results):
    """
    Load existing evaluation results (CI, IBS, etc.) from saved pickle files
    """
    for model_params, params, seed, fold, checkpoint_path in existing_models:
        try:
            fold_result_path = os.path.join(os.path.dirname(checkpoint_path), f"model_fold_{fold}.pickle")
            if os.path.exists(fold_result_path):
                with open(fold_result_path, 'rb') as f:
                    res_data = pickle.load(f)
                results[model_params][seed][fold]['ci'] = res_data['ci_va']
                results[model_params][seed][fold]['ibs'] = res_data['ibs_va']
                results[model_params][seed][fold]['time'] = res_data.get('time', None)
            else:
                print(f"[WARNING] Result file not found for existing model: {fold_result_path}")
        except Exception as e:
            print(f"[ERROR] Could not load results for {checkpoint_path}: {e}") 
 

def plot_multiple_models(plot_dir, competing_risks):
    """
    Plot survival curves of multiple models and compare them to Kaplan-Meier or Aalen-Johansen estimators. 
    """

    kmf = KaplanMeierFitter()
    colors = ['blue', 'green', 'orange', 'red', 'purple', 'cyan', 'magenta', 'brown']
    
    used_labels = set()
    processed_pkl_names = set()
    idx = 0  # For color cycling

    if competing_risks:
        fig_per_risk = []
        ax_per_risk = []
        num_risks = None

    plt.figure(figsize=(12, 6))  
    for model_dir in os.listdir(plot_dir):
        model_path = os.path.join(plot_dir, model_dir)
        if not os.path.isdir(model_path):
            continue

        for fold_dir in os.listdir(model_path):
            fold_path = os.path.join(model_path, fold_dir)
            if not os.path.isdir(fold_path):
                continue

            pkl_files = [f for f in os.listdir(fold_path) if f.endswith('.pkl') and f.startswith('results_')]

            for pkl_file in pkl_files:
                if pkl_file in processed_pkl_names:
                    continue
                processed_pkl_names.add(pkl_file)

                pkl_path = os.path.join(fold_path, pkl_file)
                with open(pkl_path, 'rb') as f:
                    results = pickle.load(f)

                for result in results:
                    label = '_'.join(result['label'][:-1]) if isinstance(result['label'], list) else str(result['label'])

                    if label in used_labels:
                        continue
                    used_labels.add(label)

                    color = colors[idx % len(colors)]
                    times_np = result['times_np']
                    times_years = result['times_years']
                    events = result['events']

                    if competing_risks:
                        if num_risks is None:
                            num_risks = len(result['pred_risk_np'])
                            fig_per_risk = [plt.figure(figsize=(12, 6)) for _ in range(num_risks)]
                            ax_per_risk = [fig.add_subplot(111) for fig in fig_per_risk]

                        for risk_idx in range(num_risks):
                            pred_risk = result['pred_risk_np'][risk_idx]
                            mean_cif = np.mean(1 - pred_risk, axis=0)
                            std_cif = 1.96 * sem(1 - pred_risk, axis=0)

                            ax = ax_per_risk[risk_idx]
                            ax.plot(times_np, mean_cif, color=color, label=label, linewidth=2)
                            ax.fill_between(times_np, mean_cif - std_cif, mean_cif + std_cif, color=color, alpha=0.1)

                            if idx == 0:
                                ajf = AalenJohansenFitter()
                                ajf.fit(durations=times_years, event_observed=events, event_of_interest=(risk_idx + 1))

                                aj_times = ajf.cumulative_density_.index
                                aj_cif = ajf.cumulative_density_[f"CIF_{risk_idx + 1}"]

                                ci_lower = ajf.confidence_interval_['AJ_estimate_lower_0.95']
                                ci_upper = ajf.confidence_interval_['AJ_estimate_upper_0.95']

                                ax.plot(aj_times, 1 - aj_cif, label=f"AJ - Risk {risk_idx + 1}", linestyle='--', color='black')
                                ax.fill_between(aj_times, 1 - ci_lower, 1 - ci_upper, color='black', alpha=0.2)

                        idx += 1
                    else:
                        pred_risk_np = result['pred_risk_np']
                        mean_surv = np.mean(1 - pred_risk_np, axis=0)
                        std_surv = 1.96 * sem(1 - pred_risk_np, axis=0)

                        plt.plot(times_np, mean_surv, color=color, label=label, linewidth=2)
                        plt.fill_between(times_np, mean_surv - std_surv, mean_surv + std_surv, color=color, alpha=0.2)

                        if idx == 0:
                            kmf.fit(times_years, events)
                            kmf.plot_survival_function(ci_show=True, color='black', linewidth=2, alpha=0.3, label='Kaplan-Meier')

                        idx += 1

    font_size = 15
    legend_font_size = 15
    title_font_size = 16

    if competing_risks:
        for risk_idx, fig in enumerate(fig_per_risk):
            ax = ax_per_risk[risk_idx]
            ax.set_xlabel("Time (years)", fontsize=font_size)
            ax.set_ylabel("Cumulative Incidence", fontsize=font_size)
            ax.set_title(f"Comparison: Aalen-Johansen vs Multiple Models (Risk {risk_idx + 1})", fontsize=title_font_size)
            ax.legend(fontsize=legend_font_size)
            ax.grid(True)
            fig.tight_layout()
            fig.savefig(os.path.join(plot_dir, f'multi_aj_comparison_risk{risk_idx + 1}.png'), dpi=300)
            plt.close(fig)

        print(f"{idx} models plotted for each of {num_risks} competing risks.")
    else:
        plt.xlabel("Survival Time (years)", fontsize=font_size)
        plt.ylabel("Survival Probability", fontsize=font_size)
        plt.title("Comparison: Kaplan-Meier vs Multiple Models", fontsize=title_font_size)
        plt.legend(fontsize=legend_font_size)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'multi_survival_comparison.png'), dpi=300)
        plt.close()
        print(f"{idx} model curves have been plotted in total.")

# --------------------------------------
#       PLOTTING INTERACTIVE HTMLs
# --------------------------------------
   
def plot_interactive_models(plot_dir, output_html="interactive_plot.html"):
    """
    Plot an interactive HTML of survival curves of multiple models compared them to Kaplan-Meier or Aalen-Johansen estimators. 
    """
    kmf = KaplanMeierFitter()
    color_list = ['blue', 'green', 'orange', 'red', 'purple', 'cyan', 'magenta', 'brown']
    fig = go.Figure()
    added_labels = set()
    processed_pkl_names = set()
    km_done = False
    color_index = 0  # To cycle through colors 

    for model_dir in os.listdir(plot_dir):
        model_path = os.path.join(plot_dir, model_dir)
        if not os.path.isdir(model_path):
            continue

        for fold_dir in os.listdir(model_path):
            fold_path = os.path.join(model_path, fold_dir)
            if not os.path.isdir(fold_path):
                continue

            pkl_files = [f for f in os.listdir(fold_path) if f.endswith('.pkl') and f.startswith('results_')]
            for pkl_file in pkl_files:
                if pkl_file in processed_pkl_names:
                    continue
                processed_pkl_names.add(pkl_file)

                with open(os.path.join(fold_path, pkl_file), 'rb') as f:
                    results = pickle.load(f)

                for result in results:
                    label = '_'.join(result['label'][:-1]) if isinstance(result['label'], list) else str(result['label'])
                    if label in added_labels:
                        continue
                    added_labels.add(label)

                    times_np = result['times_np']
                    times_years = result['times_years']
                    events = result['events']
                    pred_risk_np = result['pred_risk_np']

                    mean_surv = np.mean(1 - pred_risk_np, axis=0)
                    std_surv = 1.96 * sem(1 - pred_risk_np, axis=0)

                    base_color = color_list[color_index % len(color_list)]
                    color_index += 1

                    fig.add_trace(go.Scatter(
                        x=times_np,
                        y=mean_surv,
                        mode='lines',
                        name=label,
                        line=dict(color=base_color),
                        legendgroup=label,
                        visible=True
                    ))

                    fig.add_trace(go.Scatter(
                        x=np.concatenate([times_np, times_np[::-1]]),
                        y=np.concatenate([mean_surv - std_surv, (mean_surv + std_surv)[::-1]]),
                        fill='toself',
                        fillcolor=base_color.replace('rgb', 'rgba').replace(')', ',0.2)') if 'rgb' in base_color else f'rgba(0,0,0,0.1)',   
                        line=dict(color='rgba(255,255,255,0)'),
                        hoverinfo="skip",
                        showlegend=False,
                        legendgroup=label,
                        visible=True
                    ))

                    if not km_done:
                        kmf.fit(times_years, events)
                        fig.add_trace(go.Scatter(
                            x=kmf.survival_function_.index,
                            y=kmf.survival_function_['KM_estimate'],
                            mode='lines',
                            name='Kaplan-Meier',
                            line=dict(color='black', width=2, dash='dot'),
                            legendgroup='KM',
                            visible=True
                        ))

                        ci_lower = kmf.confidence_interval_['KM_estimate_lower_0.95']
                        ci_upper = kmf.confidence_interval_['KM_estimate_upper_0.95']

                        fig.add_trace(go.Scatter(
                            x=np.concatenate([ci_lower.index, ci_lower.index[::-1]]),
                            y=np.concatenate([ci_lower, ci_upper[::-1]]),
                            fill='toself',
                            fillcolor='rgba(0,0,0,0.2)',
                            line=dict(color='rgba(255,255,255,0)'),
                            hoverinfo="skip",
                            showlegend=False,
                            legendgroup='KM',
                            visible=True
                        ))
                        km_done = True

    fig.update_layout(
        title='Survival Curve Comparison (Click on legend to show/hide models)',
        xaxis_title='Survival Time (years)',
        yaxis_title='Survival Probability',
        legend_title="Select models to display:",
        hovermode='x unified',
        template='plotly_white',
        autosize=True,
        height=800,
        margin=dict(l=50, r=50, t=80, b=50)
    )

    output_path = os.path.join(plot_dir, output_html)
    fig.write_html(output_path, auto_open=False)
    print(f"Interactive plot saved at: {output_path}")


def plot_interactive_survival(pred_risk_np, times_np, plot_dir, clinical_df=None):
    """
    Plot an interactive HTML of survival curve of clinical patients with single risk.
    """
    fig = go.Figure()

    for patient_id in range(pred_risk_np.shape[0]):
        surv_curve = 1 - pred_risk_np[patient_id]

        if clinical_df is not None: 
            clinical_info = "<br>".join([
                f"{col}: {clinical_df.iloc[patient_id][col]}" for col in clinical_df.columns
            ])
        else:
            clinical_info = ""

        fig.add_trace(go.Scatter(
            x=times_np,
            y=surv_curve,
            mode='lines',
            name=f'Patient {patient_id}',
            line=dict(width=1),
            hovertemplate=f'Patient {patient_id}<br>Time=%{{x}}<br>Survival=%{{y:.2f}}<br>{clinical_info}'
        ))

    fig.update_layout(
        title="Estimated Survival Curves",
        xaxis_title="Time (years)",
        yaxis_title="Survival Probability",
        legend_title="Patients",
        template="plotly_white",
        height=600
    )

    fig.write_html(os.path.join(plot_dir, 'survival_interactive.html'))


def plot_interactive_cif(pred_risk_all, times_np, plot_dir, clinical_df=None): 
    """
    Plot an interactive HTML of survival curve of clinical patients with competing risks.
    """
    os.makedirs(plot_dir, exist_ok=True)

    for risk_index, pred_risk_np in enumerate(pred_risk_all):
        fig = go.Figure()
        for patient_id in range(pred_risk_np.shape[0]):
            surv_curve = pred_risk_np[patient_id]

            if clinical_df is not None:
                clinical_info = "<br>".join([
                    f"{col}: {clinical_df.iloc[patient_id][col]}" for col in clinical_df.columns
                ])
            else:
                clinical_info = ""

            fig.add_trace(go.Scatter(
                x=times_np,
                y=1 - surv_curve,
                mode='lines',
                name=f'Patient {patient_id}',
                line=dict(width=1),
                hovertemplate=f'Patient {patient_id}<br>Time=%{{x}}<br>CIF=%{{y:.2f}}<br>{clinical_info}'
            ))

        fig.update_layout(
            title=f"CIF Curves - Risk {risk_index}",
            xaxis_title="Time",
            yaxis_title="Cumulative Incidence",
            legend_title="Patients",
            template="plotly_white",
            height=600
        )
        fig.write_html(os.path.join(plot_dir, f'cif_interactive_risk{risk_index}.html'))


def plot_best_and_worst_patients_interactive(model, data, times_np, times_days, events, clinical_df, plot_dir, num_samples=100):
    """
    Generates an interactive plot showing survival curves for the best and worst prognosis patients
    based on model predictions, compared against the Kaplan-Meier estimate.
    """
    num_total_patients = data[0][2].shape[0]
    all_stats = []

    for patient_idx in range(num_total_patients):
        # Extract patient data for each modality
        input_patient = [x[2][patient_idx:patient_idx+1].to(model.device) for x in data[:-1]]
        surv_curves = []

        with torch.no_grad():
            for _ in range(num_samples):
                out = model(input_patient)
                alpha, lam = out['time_params'][0][0, 0].item(), out['time_params'][0][0, 1].item()
                lam_years = lam / 365.25
                surv_curve = np.exp(-np.power(times_np / lam_years, alpha))
                surv_curves.append(surv_curve)

        surv_curves = np.stack(surv_curves)
        mean_surv = np.mean(surv_curves, axis=0)
        perc_5 = np.percentile(surv_curves, 5, axis=0)
        perc_95 = np.percentile(surv_curves, 95, axis=0)

        life_expectancy_index = np.argmax(mean_surv < 0.5)
        life_expectancy_time = times_np[life_expectancy_index] if life_expectancy_index > 0 else times_np[-1]

        all_stats.append((patient_idx, life_expectancy_time, mean_surv, perc_5, perc_95))

    # Sort patients by life expectancy
    sorted_by_life = sorted(all_stats, key=lambda x: x[1])
    worst = sorted_by_life[0]
    best = sorted_by_life[-1]

    fig = go.Figure()

    for label, patient_data, color in zip(['Worst Prognosis', 'Best Prognosis'],
                                          [worst, best],
                                          ['red', 'green']):
        patient_idx, _, mean_surv, perc_5, perc_95 = patient_data

        # Clinical info
        if clinical_df is not None:
            clinical_row = clinical_df.iloc[patient_idx].copy()
            clinical_info = "<br>".join([f"{col}: {clinical_row[col]}" for col in clinical_df.columns])
        else:
            clinical_info = ""

        # Mean survival curve
        fig.add_trace(go.Scatter(
            x=times_np,
            y=mean_surv,
            mode='lines',
            name=f'{label} Mean (Patient {patient_idx})',
            line=dict(width=2, color=color),
            hovertemplate=f"{label}<br>Patient {patient_idx}<br>Time=%{{x}}<br>Survival=%{{y:.2f}}<br>{clinical_info}"
        ))
 
        fig.add_trace(go.Scatter(
            x=np.concatenate([times_np, times_np[::-1]]),
            y=np.concatenate([perc_5, perc_95[::-1]]),
            fill='toself',
            fillcolor='rgba' + ('(255,0,0,0.2)' if color == 'red' else '(0,255,0,0.2)'),
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        ))

    # Kaplan-Meier global + CI
    kmf = KaplanMeierFitter()
    kmf.fit(times_days, events, label="Kaplan-Meier (Global)")
    km_times = kmf.survival_function_.index.values
    km_surv = kmf.survival_function_["Kaplan-Meier (Global)"].values
    ci_upper = kmf.confidence_interval_["Kaplan-Meier (Global)_upper_0.95"].values
    ci_lower = kmf.confidence_interval_["Kaplan-Meier (Global)_lower_0.95"].values

    # Add KM curve
    fig.add_trace(go.Scatter(
        x=km_times,
        y=km_surv,
        mode='lines',
        name="Kaplan-Meier (Global)",
        line=dict(color='purple', dash='dash', width=2),
        hovertemplate="Kaplan-Meier<br>Time=%{x}<br>Survival=%{y:.2f}"
    ))

     # Add KM confidence interval
    fig.add_trace(go.Scatter(
        x=np.concatenate([km_times, km_times[::-1]]),
        y=np.concatenate([ci_lower, ci_upper[::-1]]),
        fill='toself',
        fillcolor='rgba(128,0,128,0.2)',  # púrpura claro
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False
    ))

    fig.update_layout(
        title="Best vs Worst Prognosis Patients (Interactive)",
        xaxis_title="Time (years)",
        yaxis_title="Survival Probability",
        template="plotly_white",
        height=600
    )

    fig.write_html(os.path.join(plot_dir, 'best_vs_worst_prognosis_interactive.html'))