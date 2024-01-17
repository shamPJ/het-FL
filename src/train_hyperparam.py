
from copy import deepcopy
# from data_aug_models import Linreg_Torch_aug, Linreg_Sklearn, DTReg, MLP_Keras
from generate_data import get_data
from generate_graph import build_edges, build_graph, build_graph_pooled
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from tqdm import tqdm
from train import train

plt.style.use('ggplot')

def iter_params(config, compute_dist=False):
    
    """

    Try out different reg. strength lambda and local dataset size (d/m ratio). 
    Each hyperparam combination is repeated `repeat_times` times and results are averaged.

    Args:
    : config              : dict containing necessary parameters.
    : compute_dist        : if True returns computed output `dist_list`, otherwise returns list with zero arrays. Set True only for linear models without bias term.
    
    Number of hyperparameters' combinations is k = len(reg_term_list)*len(n_samples_list)
    
    Outputs:
    : mse_train_list      : list len=k. Each el array of shape (repeat_times, n_nodes, n_iters) containing training loss for repeat_times runs for each node in the graph. 
    : mse_val_list        : list len=k. Each el array of shape (repeat_times, n_nodes, n_iters) containing validation loss for repeat_times runs for each node in the graph. 
    : mse_val_pooled_list : list len=k. Each el array of shape (repeat_times, n_nodes, n_iters) containing validation loss for repeat_times runs for each cluster model on the local nodes val ds. 
    
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
    ds_shared_mode = config['ds_shared_mode']
    
    # Store results of train() loop with hyperparams pairs {reg_term, n_samples} repeated `repeat_times` times
    mse_train_list      = []  # mse on local training ds
    mse_val_list        = []  # mse on local validation ds
    mse_val_pooled_list = []  # mse on pooled (for each cluster) validation ds
    
    # Create dir to save results of the experiment
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    path = 'out/' + file_name + '_' + timestamp
    os.mkdir(path)

    for reg_term in reg_term_list:
        for n_samples in tqdm(n_samples_list):
            
            # Arrays to store intermediate results
            mse_train_array      = np.zeros((repeat_times, n_nodes, n_iters))
            mse_val_array        = np.zeros((repeat_times, n_nodes, n_iters))
            mse_val_pooled_array = np.zeros((repeat_times, n_nodes, n_iters))
            
            # Repeat train() with hyperparams {reg_term, n_samples} 
            for i in range(repeat_times):
                # Copy models for training in current repetition
                models        = deepcopy(models)
                models_pooled = deepcopy(models_pooled)

                # Create a dataset
                ds_train, ds_val, cluster_labels, _ = get_data(n_clusters, n_ds, n_samples, n_features, noise_val)
                
                # Build a graph with nodes as dict containing local train/ val ds, model and cluster assignment
                G = build_graph(ds_train, ds_val, models, cluster_labels)
                A = build_edges(G, cluster_labels, p_in = p_in, p_out = p_out)
                G_pooled = build_graph_pooled(ds_train, ds_val, models_pooled, n_clusters=n_clusters)

                _, mse_train, mse_val, mse_val_pooled = train(G, A, G_pooled, n_iters=n_iters, regularizer_term=reg_term, ds_shared_mode=ds_shared_mode, verbose=False)

                # Save mse on a local ds
                mse_train_array[i]      = mse_train
                mse_val_array[i]        = mse_val
                mse_val_pooled_array[i] = mse_val_pooled
            
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
                            'models_pooled':    str(models_pooled),
                            'ds_shared_mode':   ds_shared_mode
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

    return mse_train_list, mse_val_list, mse_val_pooled_list


