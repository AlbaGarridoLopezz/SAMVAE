# Author: Alba Garrido López
# Email: alba.garrido.lopez@upm.es

# Packages to import
import math  
import torch
from sklearn.model_selection import train_test_split, KFold
from scipy import stats 
from sklearn.svm import SVR 
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge 
 
# -----------------------------------------------------------
#                   DATA IMPUTATION
# -----------------------------------------------------------

def zero_imputation(data):
    imp_data = data.copy()
    imp_data = imp_data.fillna(0)
    return imp_data


def mice_imputation(data, model='bayesian'):
    imp_data = data.numpy()
    if model == 'bayesian':
        clf = BayesianRidge()
    elif model == 'svr':
        clf = SVR()
    else:
        raise RuntimeError('MICE imputation base_model not recognized')
    imp = IterativeImputer(estimator=clf, verbose=2, max_iter=30, tol=1e-10, imputation_order='roman') 
    imp_data_np = imp.fit_transform(imp_data)
    imp_data = torch.tensor(imp_data_np, dtype=torch.float32)
    return imp_data


def statistics_imputation(tensor):  
    n_samp, n_feat = tensor.shape
    for i in range(n_feat):
        col_values = tensor[:, i]
        
        if torch.any(torch.isnan(col_values)): 
            no_nan_values = col_values[~torch.isnan(col_values)]
            if no_nan_values.numel() == 0:   
                continue
            if torch.is_tensor(no_nan_values) and no_nan_values.dim() == 1:
                if no_nan_values.dtype in [torch.float32, torch.float64]:
                    stats_value = no_nan_values.mean()  # Mean for ints
                else:
                    stats_value = stats.mode(no_nan_values.numpy())[0][0]  # Mode
            else:
                stats_value = no_nan_values.mean()  # Mean
 
            tensor[torch.isnan(col_values), i] = stats_value 
    return tensor


def impute_data(tensor, mode='stats'):
    # If missing data exists, impute it
    if torch.isnan(tensor).any(): 
        if mode == 'zero':
            imp_tensor = zero_imputation(tensor)
        elif mode == 'stats':
            imp_tensor = statistics_imputation(tensor)
        else:
            imp_tensor = mice_imputation(tensor)
        mask = ~torch.isnan(tensor)   
    else:
        imp_tensor = tensor.clone()   
        mask = torch.ones_like(tensor, dtype=torch.bool) # To reduce the memory
    return imp_tensor, mask


# -----------------------------------------------------------
#                   DATA NORMALIZATION
# -----------------------------------------------------------

def get_feat_distributions(tensor, force_time_dist=False, time=None):
    n_feat = tensor.shape[1]
    feat_dist = []
    
    for i in range(n_feat):
        if force_time_dist and i == n_feat - 2:  # Time column
            feat_dist.append(time)
            continue
        values = torch.unique(tensor[:, i])
        
        if len(values) == 1 and math.isnan(values[0].item()):
            values = torch.tensor([0.])
 
        no_nan_values = values[~torch.isnan(values)]   
        no_nan_values = no_nan_values.float()   
 
        no_nan_values = no_nan_values[~torch.isnan(no_nan_values)]  

        if no_nan_values.numel() <= 2:  
            feat_dist.append(('bernoulli', 1))
         
        elif torch.all(torch.remainder(no_nan_values, 1) == 0):
            no_nan_values = no_nan_values.long()   

            if torch.unique(no_nan_values).numel() < 30 and torch.amin(no_nan_values) == 0:
                feat_dist.append(('categorical', torch.max(no_nan_values) + 1))
            else:
                feat_dist.append(('gaussian', 2))
        else:
            feat_dist.append(('gaussian', 2))

    return feat_dist


def transform_data(raw_tensor, feat_distributions, norm_tensor=None):
    transf_tensor = norm_tensor.clone() if norm_tensor is not None else raw_tensor.clone()

    for i in range(raw_tensor.shape[1]):
        dist = feat_distributions[i][0]
        values = raw_tensor[:, i]
        no_nan_values = values[~torch.isnan(values)]

        if dist == 'gaussian':
            loc = torch.mean(no_nan_values)
            scale = torch.std(no_nan_values)
        elif dist == 'bernoulli':
            loc = torch.min(no_nan_values)
            scale = torch.max(no_nan_values) - torch.min(no_nan_values)
        elif dist == 'categorical':
            loc = torch.min(no_nan_values)
            scale = 1  # No escale
        elif dist == 'weibull':
            loc = 0 if not torch.any(no_nan_values == 0) else -1
            scale = 0
        else:
            raise NotImplementedError(f'Distribution {dist} not normalized!')

        if norm_tensor is not None:  # Denormalization
            transf_tensor[:, i] = (
                norm_tensor[:, i] * scale + loc if scale != 0
                else norm_tensor[:, i] + loc).to(raw_tensor.dtype)
        else:  # Normalization
            transf_tensor[:, i] = (values - loc) / scale if scale != 0 else values - loc

    return transf_tensor


# -----------------------------------------------------------
#                   DATA SPLITTING
# -----------------------------------------------------------

def append_tensor_data(train_data, test_data, train_mask, test_mask, cv_data): 
    train_data = train_data.contiguous()
    test_data = test_data.contiguous()
    train_mask = train_mask.contiguous()
    test_mask = test_mask.contiguous()
    cv_data.append((train_data, train_mask, test_data, test_mask)) # Tuples
    return cv_data


def train_test_split_tensor(data, mask, test_size=0.2, random_state=1234):
    n_samples = data.size(0)
    indices = torch.randperm(n_samples, generator=torch.manual_seed(random_state))
    test_count = int(test_size * n_samples)
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]

    return (data[train_indices], data[test_indices], mask[train_indices], mask[test_indices])


# Function that receives data and performs a split of a specific number of folds for cross-validation.
# It returns a list of tuples with the train and test data and the mask for each fold
def split_cv_data_multimodal(real_tensors, n_folds, time_dist=None):
    all_cv_data = []
    all_feat_distributions = []
   
    for idx, real_tensor in enumerate(real_tensors):
        # First, impute all together
        imp_data, mask = impute_data(real_tensor)  
        
        # Then, normalize all together
        force_time_dist = (idx == len(real_tensors) - 1) # Force Time Distribution in the time tensor (the last one)  
        feat_distributions = get_feat_distributions(imp_data, force_time_dist=force_time_dist, time=time_dist)
        all_feat_distributions.append(feat_distributions)
        norm_data = transform_data(imp_data, feat_distributions)

        # Finally, split
        cv_data = []
        if n_folds < 2:
            train_data, test_data, train_mask, test_mask = train_test_split_tensor(
                norm_data, mask, test_size=0.2, random_state=1234
            )
            cv_data = append_tensor_data(train_data, test_data, train_mask, test_mask, cv_data)
        else:
            kf = KFold(n_splits=n_folds, random_state=1234, shuffle=True)
            for train_index, test_index in kf.split(norm_data):
                train_data = norm_data[train_index]
                test_data = norm_data[test_index]
                train_mask = mask[train_index]
                test_mask = mask[test_index]
                cv_data = append_tensor_data(train_data, test_data, train_mask, test_mask, cv_data)

        all_cv_data.append(cv_data)

    return all_cv_data, all_feat_distributions