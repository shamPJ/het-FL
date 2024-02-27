from pytorch_models import Linreg_Torch, MLP_Torch
from data_aug_models import Linreg_Sklearn, DTReg
import time
from train_hyperparam import iter_params
from utils import load_and_plot_mse, load_and_plot_est_error

#======================== EXP SETUP ========================#
# n.o. iterations and repeat experiment number of times
n_iters, repeat_times = 2000, 10
# noise scale for synthetic data
noise_val = 1.0
# prob of edges
p_in, p_out = 0.8, 0.2
# data generation params
n_clusters, n_ds, n_features = 3, 50, 10

# Hyperparameters to try:
# sample size of a local dataset at each node
n_samples_list = [2, 3, 5, 10, 20]
# value of regularization term lambda
reg_term_list = [0, 0.1, 0.5]

# Config
config = {
            'reg_term_list':    reg_term_list,
            'n_samples_list':   n_samples_list,
            'repeat_times':     repeat_times,
            'n_iters':          n_iters,
            'noise_val':        noise_val,
            'n_clusters':       n_clusters,
            'n_ds':             n_ds,
            'n_features':       n_features,
            'p_in':             p_in,
            'p_out':            p_out,
        }


#======================== Pytorch Linear - NO BIAS - varying regularization, sample size ========================#
#====================================== Adjacency matrix is known ======================================#

# lrates = [0.01]

# for lrate in lrates:
#     # Create linear pytorch models
#     models = [Linreg_Torch(n_features, lr=lrate, bias=False) for i in range(n_clusters*n_ds)]
#     models_pooled = [Linreg_Torch(n_features, lr=lrate, bias=False) for i in range(n_clusters)]
    
#     config['models']         = models
#     config['models_pooled']  = models_pooled
#     config['parametric']     = True
#     config['noise_val']      = 1
#     config['adj_matrix']     = True
#     config['file_name']      = 'Linreg_Torch_lrate_' + str(lrate)
    
#     # out is list len=k ({reg.term, n_samples} combinations), each el array of shape (repeat_times, n_nodes, n_iters)
#     path, mse_train, mse_val, mse_val_pooled = iter_params(config)
#     load_and_plot_mse(path.split('/')[-1])
#     load_and_plot_mse(path.split('/')[-1], scaled=True)
#     load_and_plot_est_error(path.split('/')[-1])

#======================== Pytorch Linear - NO BIAS - varying regularization, sample size ========================#
#====================================== Adjacency matrix is not known ======================================#

# lrates = [0.01]

# for lrate in lrates:
#     # Create linear pytorch models
#     models = [Linreg_Torch(n_features, lr=lrate, bias=False) for i in range(n_clusters*n_ds)]
#     models_pooled = [Linreg_Torch(n_features, lr=lrate, bias=False) for i in range(n_clusters)]
    
#     config['models']         = models
#     config['models_pooled']  = models_pooled
#     config['parametric']     = True
#     config['noise_val']      = 1
#     config['adj_matrix']     = False
#     config['n_neighbours']   = 3
#     config['file_name']      = 'Linreg_Torch_lrate_' + str(lrate)
    
#     # out is list len=k ({reg.term, n_samples} combinations), each el array of shape (repeat_times, n_nodes, n_iters)
#     path, mse_train, mse_val, mse_val_pooled = iter_params(config)
#     load_and_plot_mse(path.split('/')[-1])
#     load_and_plot_mse(path.split('/')[-1], scaled=True)
#     load_and_plot_est_error(path.split('/')[-1])

# #======================== Pytorch Linear - NO BIAS - varying regularization, sample size ========================#
# #====================================== Adjacency matrix is not known ======================================#

lrates = [0.01]

for lrate in lrates:
    # Create linear pytorch models
    models = [Linreg_Torch(n_features, lr=lrate, bias=False) for i in range(n_clusters*n_ds)]
    models_pooled = [Linreg_Torch(n_features, lr=lrate, bias=False) for i in range(n_clusters)]
    
    config['models']         = models
    config['models_pooled']  = models_pooled
    config['parametric']     = True
    config['noise_val']      = 1
    config['adj_matrix']     = False
    config['n_neighbours']   = 10
    config['file_name']      = 'Linreg_Torch_lrate_' + str(lrate)
    
    # out is list len=k ({reg.term, n_samples} combinations), each el array of shape (repeat_times, n_nodes, n_iters)
    path, mse_train, mse_val, mse_val_pooled = iter_params(config)
    load_and_plot_mse(path.split('/')[-1])
    load_and_plot_mse(path.split('/')[-1], scaled=True)
    load_and_plot_est_error(path.split('/')[-1])

#======================== Pytorch Linear - NO BIAS - varying regularization, sample size ========================#
#====================================== Adjacency matrix is not known ======================================#

# lr = 0.01

# # Create mixed models
# models = [Linreg_Sklearn() for i in range(n_ds)] + [Linreg_Sklearn() for i in range(n_ds)] + [Linreg_Sklearn() for i in range(n_ds)]
# models_pooled = [Linreg_Sklearn(), Linreg_Sklearn(), Linreg_Sklearn()]

# config['models']         = models
# config['models_pooled']  = models_pooled
# config['parametric']     = False
# config['noise_val']      = 1
# config['adj_matrix']     = False
# config['n_neighbours']   = 3
# config['file_name']      = 'Mixed_models' 

# # out is list len=k ({reg.term, n_samples} combinations), each el array of shape (repeat_times, n_nodes, n_iters)
# path, mse_train, mse_val, mse_val_pooled = iter_params(config)
# load_and_plot_mse(path.split('/')[-1])
# load_and_plot_mse(path.split('/')[-1], scaled=True)

