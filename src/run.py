import argparse
from joblib import Parallel, delayed
import json
import numpy as np
import os
from pytorch_models import Linreg_Torch
from train import repeat_train
from utils import mse_mean_std, mse_mean_std_scaled, load_and_plot_mse, load_and_plot_est_error

parser = argparse.ArgumentParser()
parser.add_argument("-p_in", "--p_in", default=0.8, type=float, help="within cluster edge prob.")
parser.add_argument("-p_out", "--p_out", default=0.2, type=float, help="between cluster edge prob.")
parser.add_argument("-lrate", "--lrate", default=0.01, type=float, help="learning rate of torch.optim.SGD")
parser.add_argument("-n_iters", "--n_iters", default=2000, type=int, help="Number of SGD iterations")
parser.add_argument("-reg_term", "--reg_term", default=0.05, type=float, help="GTV regularization")
parser.add_argument("-adj_matrix", "--adj_matrix", default=True, type=bool, help="if adjacency matrix given (Algo 1) or not (Algo 2 and 3)")
parser.add_argument("-n_neighbours", "--n_neighbours", default=10, type=int, help="how many edges to add if adjacency matrix is not given")
parser.add_argument("-n_neighbours_fixed", "--n_neighbours_fixed", default=False, type=bool, help="keep node degree const and equal to n_neighbours (Algo 3) or not (Algo 2), if adjacency matrix is not given")

args = parser.parse_args()
print(os.environ["SLURM_JOB_ID"],os.environ["SLURM_ARRAY_JOB_ID"], os.environ["SLURM_ARRAY_TASK_ID"])
#======================== EXP SETUP ========================#
p_in, p_out, lrate, n_iters, reg_term = args.p_in, args.p_out, args.lrate, args.n_iters, args.reg_term
adj_matrix = args.adj_matrix
if not adj_matrix: n_neighbours, n_neighbours_fixed = args.n_neighbours, args.n_neighbours_fixed
# repeat experiment number of times
repeat_times = 10
# noise scale for synthetic data
noise_val = 1.0
# data generation params
n_clusters, n_ds, n_features, m_shared = 3, 50, 10, 100

# Hyperparameters to try:
# sample size of a local dataset at each node
n_samples_list = [2, 3, 5, 10, 20]

# Create linear pytorch models
models = [Linreg_Torch(n_features, lr=lrate, bias=False) for i in range(n_clusters*n_ds)]
models_pooled = [Linreg_Torch(n_features, lr=lrate, bias=False) for i in range(n_clusters)]

# parent directory
print(os.getcwd())
p_dir = '/scratch/work/abduras1/het-FL/out/Linreg_Torch_' + os.environ["SLURM_ARRAY_JOB_ID"]
# subdirs for each reg. term value
exp_dir =  p_dir + '/reg_term_' + str(reg_term)
os.makedirs(exp_dir)

# Config
config = {
            'reg_term':           reg_term,
            'n_clusters':         n_clusters,
            'n_ds':               n_ds,
            'n_features':         n_features,
            'm_shared':           m_shared,
            'repeat_times':       repeat_times,
            'n_iters':            n_iters,
            'noise_val':          noise_val,
            'p_in':               p_in,
            'p_out':              p_out,
            'lrate':              lrate,
            'adj_matrix':         adj_matrix,
            'parametric':         True,
            'exp_dir':            exp_dir,
            'models_pooled':      models_pooled,
            'models':             models
        }

if not adj_matrix: config['n_neighbours'] = 'n_neighbours'
if not adj_matrix: config['n_neighbours_fixed'] = 'n_neighbours_fixed'

Confs = [config for i in range(len(n_samples_list))]
out = Parallel(n_jobs=-1)(delayed(repeat_train)(n_samples, config) for n_samples, config in zip(n_samples_list, Confs))

# out is the len(n_samples) list of tuples (mse_train_array, mse_val_array, mse_train_pooled_array, mse_val_pooled_array, est_error_array, est_error_pooled_array)
# where arrays' shape is (repeat_times, n_nodes, n_iters)
# re-structure results into list of len(n_samples_list) for plotting function
mse_train_list, mse_val_list = [tpl[0] for tpl in out], [tpl[1] for tpl in out]
mse_train_pooled_list, mse_val_pooled_list = [tpl[2] for tpl in out], [tpl[3] for tpl in out]
est_error_list, est_error_pooled_list = [tpl[4] for tpl in out], [tpl[5] for tpl in out]

# compute average across all runs of average across all nodes MSEs 
mse_t, mse_std_t = mse_mean_std(mse_train_list)
mse_v, mse_std_v = mse_mean_std(mse_val_list)
mse_t_scaled, mse_std_t_scaled = mse_mean_std_scaled(mse_train_list, mse_train_pooled_list)
mse_v_scaled, mse_std_v_scaled = mse_mean_std_scaled(mse_val_list, mse_val_pooled_list)

# weight vector est. error mean and std over repeatitions 
est_error_means, est_error_std = mse_mean_std(est_error_list)
est_error_means_scaled, est_error_std_scaled = mse_mean_std_scaled(est_error_list, est_error_pooled_list)

os.mkdir(exp_dir + '/stats')
with open(exp_dir + '/stats/params.json', 'w') as f:
        json.dump({
                'adj_matrix':      adj_matrix,
                'reg_term' :       reg_term,
                'n_clusters':      n_clusters,
                'n_ds':            n_ds,
                'n_features' :     n_features,
                'n_samples_list' : n_samples_list,
                'd_m_ratio' :     [n_features/ x  for x in n_samples_list]}, f, indent=2)
        
# save computed mse and std for each pair (reg.term, sample size)
np.save(exp_dir + '/stats/mse_std.npy', [(mse_t, mse_std_t), (mse_v, mse_std_v)])
np.save(exp_dir + '/stats/mse_std_scaled.npy', [(mse_t_scaled, mse_std_t_scaled), (mse_v_scaled, mse_std_v_scaled)])
np.save(exp_dir + '/stats/est_error_mean_std.npy', (est_error_means, est_error_std))
np.save(exp_dir + '/stats/est_error_mean_std_scaled.npy', (est_error_means_scaled, est_error_std_scaled))

# plotting func
load_and_plot_mse(p_dir)
load_and_plot_mse(p_dir, scaled=True)
load_and_plot_est_error(p_dir)
