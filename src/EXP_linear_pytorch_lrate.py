from pytorch_models import Linreg_Torch, MLP_Torch
import time
from train_hyperparam import iter_params
from utils import mse_mean_std, mse_mean_std_scaled, save_and_plot

#======================== EXP SETUP ========================#
# n.o. iterations and repeat experiment number of times
n_iters, repeat_times = 10, 3
# noise scale for synthetic data
noise_val = 1.0
# prob of edges
p_in, p_out = 0.8, 0.2
# data generation params
n_clusters, n_ds, n_features = 3, 5, 10

# Hyperparameters to try:
# sample size of a local dataset at each node
n_samples_list = [5, 10, 50]
# value of regularization term lambda
reg_term_list = [0, 0.01, 0.1]

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
            'ds_shared_mode':   'const'
        }


#======================== Pytorch Linear - with bias - varying regularization, sample size & learning rate ========================#

lrates = [0.001, 0.01, 0.1]
mse_lrates = []
mse_lrates_scaled = []

for lrate in lrates:
    # Create linear pytorch models
    models = [Linreg_Torch(n_features, lr=lrate) for i in range(n_clusters*n_ds)]
    models_pooled = [Linreg_Torch(n_features, lr=lrate) for i in range(n_clusters)]
    
    config['models']         = models
    config['models_pooled']  = models_pooled
    config['noise_val']      = 1
    config['file_name']      = 'Linreg_Torch_lrate_' + str(lrate)
    
    # out is list len=k ({reg.term, n_samples} combinations), each el array of shape (repeat_times, n_nodes, n_iters)
    mse_train, mse_val, mse_val_pooled = iter_params(config)
    # out is list len=k, each el array of shape (n_iters)
    # mse_t, mse_std_t = mse_mean_std(mse_train)
    # mse_v, mse_std_v = mse_mean_std(mse_val)
    
    # mse_t_scaled, mse_std_t_scaled = mse_mean_std_scaled(mse_train, mse_val_pooled)
    # mse_v_scaled, mse_std_v_scaled = mse_mean_std_scaled(mse_val, mse_val_pooled)

    # # for each learning rate create a list of tuples with train/val average MSE and std
    # mse_lrates.append( [(mse_t, mse_std_t), (mse_v, mse_std_v)] )
    # mse_lrates_scaled.append( [(mse_t_scaled, mse_std_t_scaled), (mse_v_scaled, mse_std_v_scaled)] )

# Save and plot computed averages and std
timestamp = time.strftime("%Y%m%d-%H%M%S")

# save_and_plot(mse_lrates, lrates, 'lrate', reg_term_list, n_samples_list, name='Linreg_Torch_lrate_'+timestamp)
# save_and_plot(mse_lrates_scaled, lrates, 'lrate', reg_term_list, n_samples_list, name='Linreg_Torch_lrate_scaled_'+timestamp)






