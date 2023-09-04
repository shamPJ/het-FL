
from copy import deepcopy
from data_aug_models import Linreg_Torch_aug, Linreg_Sklearn, DTReg, MLP_Keras
from generate_data import get_data, get_shared_data
from generate_graph import build_edges, build_graph, build_graph_pooled
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from pytorch_models import Linreg_Torch, MLP_Torch
import time
from tqdm import tqdm
from train import train
from utils import plot_weight_dist, mse_mean_std, mse_mean_std_scaled, plot_mse

plt.style.use('ggplot')

def iter_params(config, compute_dist=False):
    
    """

    Try out different lambda and local dataset size. Each hyperparam combination is repeated `repeat_times` times and results are averaged.

    :param config            : dictionary containing necessary parameters.
    :param compute_dist      : if True returns computed output `dist_list`, otherwise returns list with zero arrays. Set True only for linear models without bias term.
    
    Number of hyperparameters' combinations is k = len(reg_term_list)*len(n_samples_list)
    
    :out nodes_preds_list    : list with k elements. Each element is an array of shape (repeat_times, m_shared, n_nodes) containing predictions on shared ds from repeat_times runs for each node in the graph. 
    :out mse_train_list      : list with k elements. Each element is an array of shape (repeat_times, n_nodes) containing training loss for repeat_times runs for each node in the graph. 
    :out mse_val_list        : list with k elements. Each element is an array of shape (repeat_times, n_nodes) containing validation loss for repeat_times runs for each node in the graph. 
    :out mse_val_pooled_list : list with k elements. Each element is an array of shape (repeat_times, n_nodes) containing validation loss for repeat_times runs for each cluster model on the local nodes val ds. 
    :out dist_list           : list with k elements. Each element is an array of shape (repeat_times, n_nodes, n_clusters) containing squared L2 distance loss between learnt weight vector and 'true' weight vector for each cluster (i.e. weight vector used for data generation). 
    
    """
    
    # Config
    n_clusters     = config['n_clusters']
    n_ds           = config['n_ds']
    n_features     = config['n_features']
    noise_val      = config['noise_val']
    
    p_in           = config['p_in']
    p_out          = config['p_out']
    
    reg_term_list  = config['reg_term_list']
    n_samples_list = config['n_samples_list']
    repeat_times   = config['repeat_times']
    n_iters        = config['n_iters']
    
    models         = config['models']
    models_pooled  = config['models_pooled']
    
    file_name      = config['file_name']

    m_shared       = 100
    n_nodes        = n_clusters * n_ds
    
    # Lists to store results of train() loop with hyperparams pairs {reg_term,n_samples} repeated `repeat_times` times
    mse_train_list      = []  # mse loss on local training ds
    mse_val_list        = []  # mse loss on local validation ds
    mse_val_pooled_list = []  # mse loss on pooled (for each cluster) validation ds
    nodes_preds_list    = []  # preds on shared ds
    dist_list           = []  # squared L2 distance between learnt and `true` weight vectors
    
    # Create dir to save results of the experiment
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    path = 'out/' + file_name + '_' + timestamp
    os.mkdir(path)

    for reg_term in reg_term_list:
        for n_samples in tqdm(n_samples_list):
            
            # Arrays to store intermediate results
            mse_train_array      = np.zeros((repeat_times, n_nodes))
            mse_val_array        = np.zeros((repeat_times, n_nodes))
            mse_val_pooled_array = np.zeros((repeat_times, n_nodes))
            nodes_preds_array    = np.zeros((repeat_times, m_shared, n_nodes))
            dist_array           = np.zeros((repeat_times, n_nodes, n_clusters))
            
            # Repeat train() with hyperparams {reg_term, n_samples} 
            for i in range(repeat_times):
                # Copy models for training in current repetition
                models        = deepcopy(models)
                models_pooled = deepcopy(models_pooled)

                # Create a dataset
                ds_train, ds_val, cluster_labels, true_weights = get_data(n_clusters, n_ds, n_samples, n_features, noise_val)
                ds_shared = get_shared_data(m_shared, n_features)
                
                # Build a graph with nodes {model, dataset local, dataset test}
                G = build_graph(ds_train, ds_val, ds_shared, models, cluster_labels)
                A = build_edges(G, cluster_labels, p_in = p_in, p_out = p_out)
                G_pooled = build_graph_pooled(ds_train, ds_val, models_pooled, n_clusters=n_clusters)

                preds_list, mse_train, mse_val, mse_val_pooled = train(G, A, G_pooled, n_iters=n_iters, regularizer_term=reg_term, verbose=False)

                # Save preds on a shared set of last iteration
                nodes_preds_array[i] = preds_list[-1] 

                # Save mse on a local ds
                mse_train_array[i]      = mse_train
                mse_val_array[i]        = mse_val
                mse_val_pooled_array[i] = mse_val_pooled

                if compute_dist:
                    # only for G with lin.models (no bias)
                    # compute dist b/learnt model's weight vector and true weight vector for each cluster 
                    dist = plot_weight_dist(G, true_weights)
                    dist_array[i] = dist
            
            data_to_save = {
                            'repeat_times':    repeat_times,
                            'n_nodes':          n_nodes,
                            'n_clusters':       n_clusters,
                            'n_ds':             n_ds,
                            'n_features':       n_features,
                            'n_samples':        n_samples, 
                            'n_samples_shared': m_shared, 
                            'noise_val':        noise_val,
                            'reg_term':         reg_term,
                            'p_in':             p_in,
                            'p_out':            p_out,
                            'models':           str(models),
                            'models_pooled':    str(models_pooled)
                            }
            
            # Create subdir to save experiment with current reg_term and n_ds values
            subdir = path + '/' + 'reg_term_' + str(reg_term) + '_n_samples_' + str(n_samples)
            os.mkdir(subdir)
            
            config_path = subdir + '/config_' + timestamp + '.json'
            with open(config_path, 'w') as f:
                json.dump(data_to_save, f, indent=2)
                
            np.save(subdir + '/mse_train'     + '_' + timestamp + '.npy', mse_train_array)
            np.save(subdir + '/mse_val'       + '_' + timestamp + '.npy', mse_val_array)
            np.save(subdir + '/mse_val_pooled'+ '_' + timestamp + '.npy', mse_val_pooled_array)
                
            mse_train_list.append(mse_train_array)
            mse_val_list.append(mse_val_array)
            mse_val_pooled_list.append(mse_val_pooled_array)
            nodes_preds_list.append(nodes_preds_array)
            dist_list.append(dist_array) 
            
    return nodes_preds_list, mse_train_list, mse_val_list, mse_val_pooled_list, dist_list


#======================== EXP SETUP ========================#

# Hyperparameters to try:

# sample size of a local dataset at each node
n_samples_list = [5, 10, 20, 30, 50, 100]
# value of regularization term lambda
reg_term_list = [0, 0.01, 0.1]
# n.o. iterations and repeat experiment number of times
n_iters, repeat_times = 1000, 10
# noise scale for synthetic data
noise_val = 1.0
# prob of edges
p_in, p_out = 0.8, 0.2
# data generation params
n_clusters, n_ds, n_features = 3, 5, 10

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
            'p_out':            p_out
        }


#======================== Pytorch Linear - with bias========================#

# Create linear pytorch models
models        = [Linreg_Torch(n_features) for i in range(n_clusters*n_ds)]
models_pooled = [Linreg_Torch(n_features) for i in range(n_clusters)]
    
config['models']        = models
config['models_pooled'] = models_pooled
config['noise_val']     = 1
config['file_name']     = 'Linreg_Torch'

nodes_preds, mse_train, mse_val, mse_val_pooled, _ = iter_params(config)

# Plot raw mse vals
mse_t, mse_std_t = mse_mean_std(mse_train)
mse_v, mse_std_v = mse_mean_std(mse_val)

mse = [ [(mse_t, mse_std_t), (mse_v, mse_std_v)] ]

titles = ['Train ds', 'Validation ds']

fig, axs = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(8,3))

for i in range(1):
    plot_list = mse[i]

    for ax, data, title in zip(axs, plot_list, titles):
        mse = data[0]
        mse_std = data[1]
        ax = plot_mse(ax, mse, mse_std, reg_term_list, n_samples_list)
        ax.set_title(title)

axs[0].set_ylabel ('MSE')
axs[0].set_xlabel ('Training ds size')
axs[1].set_xlabel ('Training ds size')

plt.legend()
fig.tight_layout()
timestamp     = time.strftime("%Y%m%d-%H%M%S")
plt.savefig('out/' +config['file_name']+ '_' + timestamp + ".png")
plt.show()





