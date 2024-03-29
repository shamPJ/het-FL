
from copy import deepcopy
# from data_aug_models import Linreg_Torch_aug, Linreg_Sklearn, DTReg, MLP_Keras
from generate_data import get_data
from generate_graph import build_edges, build_graph, build_graph_pooled
import json
import numpy as np
import os
import time
from tqdm import tqdm
from train import train, train_no_A
from utils import mse_mean_std, mse_mean_std_scaled, param_est_error, param_est_error_pooled

def iter_params(config):
    
    """

    Try out different reg. strength lambda and local dataset size. 
    Each hyperparam combination is repeated `repeat_times` times and results are averaged.

    Args:
    : config              : dict containing necessary parameters.
      
    Number of hyperparameters' combinations is k = len(reg_term_list)*len(n_samples_list)
    
    Outputs:
    : mse_train_list      : list len=k. Each el array of shape (repeat_times, n_nodes, n_iters) containing training loss for repeat_times runs for each node in the graph. 
    : mse_val_list        : list len=k. Each el array of shape (repeat_times, n_nodes, n_iters) containing validation loss for repeat_times runs for each node in the graph. 
    : mse_val_pooled_list : list len=k. Each el array of shape (repeat_times, n_nodes, n_iters) containing validation loss for repeat_times runs for each cluster model on the local nodes val ds. 
    
    """
    
    # Config
    n_clusters, n_ds, n_features, noise_val = config['n_clusters'], config['n_ds'], config['n_features'], config['noise_val']
    models, models_pooled = config['models'], config['models_pooled']
    p_in, p_out = config['p_in'], config['p_out']
    
    reg_term_list, n_samples_list, n_iters  = config['reg_term_list'], config['n_samples_list'], config['n_iters']
    repeat_times   = config['repeat_times']
    file_name      = config['file_name']
    parametric, adj_matrix     = config['parametric'], config['adj_matrix']
    if not adj_matrix: n_neighbours   = config['n_neighbours']

    m_shared       = 100
    n_nodes        = n_clusters * n_ds
     
    # Create dir to save results of the experiment
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    path = 'out/' + file_name + '_' + timestamp
    os.mkdir(path)

    for reg_term in reg_term_list:

        # Store results of train() loop with hyperparams pairs {reg_term, n_samples} repeated `repeat_times` times
        mse_train_list      = []  # mse on local training ds
        mse_val_list        = []  # mse on local validation ds
        mse_val_pooled_list = []  # mse on pooled (for each cluster) validation ds

        est_error_list         = [] # est model params for local model
        est_error_pooled_list  = [] # est model params for pooled model
   
        # Create subdir to save experiment with current reg_term and n_ds values
        subdir = path + '/' + 'reg_term_' + str(reg_term)
        os.mkdir(subdir)

        for n_samples in tqdm(n_samples_list):
            
            # Arrays to store intermediate results
            mse_train_array      = np.zeros((repeat_times, n_nodes, n_iters))
            mse_val_array        = np.zeros((repeat_times, n_nodes, n_iters))
            mse_val_pooled_array = np.zeros((repeat_times, n_nodes, n_iters))
            # Estimated error for parameter vector 
            est_error_array        = np.zeros((repeat_times, n_nodes, n_iters))
            est_error_pooled_array = np.zeros((repeat_times, n_nodes, n_iters))
            
            # Repeat train() with hyperparams {reg_term, n_samples} 
            for i in range(repeat_times):
                # Copy models for training in current repetition
                models        = deepcopy(models)
                models_pooled = deepcopy(models_pooled)

                # Create a dataset
                ds_train, ds_val, cluster_labels, true_weights = get_data(n_clusters, n_ds, n_samples, n_features, noise_val)
                
                # Build a graph with nodes as dict containing local train/ val ds, model and cluster assignment
                G = build_graph(ds_train, ds_val, models, cluster_labels)
                A = build_edges(G, cluster_labels, p_in = p_in, p_out = p_out)
                G_pooled = build_graph_pooled(ds_train, ds_val, models_pooled, n_clusters=n_clusters)

                if adj_matrix:
                    mse_train, mse_val, mse_val_pooled, est_weights, est_weights_pooled = train(G, A, G_pooled, n_iters=n_iters, regularizer_term=reg_term, m_shared=m_shared, parametric=parametric)
                else:
                    mse_train, mse_val, mse_val_pooled, est_weights, est_weights_pooled = train_no_A(G, G_pooled, n_neighbours=n_neighbours, n_iters=n_iters, regularizer_term=reg_term, m_shared=m_shared, parametric=parametric)

                # Save mse on a local ds
                mse_train_array[i]      = mse_train
                mse_val_array[i]        = mse_val
                mse_val_pooled_array[i] = mse_val_pooled

                # Save parameter vector estimation error
                est_error_array[i]        = param_est_error(true_weights, est_weights, cluster_labels)
                est_error_pooled_array[i] = param_est_error_pooled(true_weights, est_weights_pooled, cluster_labels)
            
            data_to_save = {
                            'repeat_times':     repeat_times,
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
                            'adj_matrix':       adj_matrix
                        }
            if not adj_matrix: data_to_save['n_neighbours'] = n_neighbours
            # Create subdir to save experiment with current reg_term and n_ds values
            ssubdir = subdir + '/' + 'reg_term_' + str(reg_term) + '_n_samples_' + str(n_samples)
            os.mkdir(ssubdir)
            
            config_path = ssubdir + '/config_' + timestamp + '.json'
            with open(config_path, 'w') as f:
                json.dump(data_to_save, f, indent=2)
                
            np.save(ssubdir + '/mse_train'     + '_' + timestamp + '.npy', mse_train_array)
            np.save(ssubdir + '/mse_val'       + '_' + timestamp + '.npy', mse_val_array)
            np.save(ssubdir + '/mse_val_pooled'+ '_' + timestamp + '.npy', mse_val_pooled_array)

            np.save(ssubdir + '/est_weights'       + '_' + timestamp + '.npy', est_error_array)
            np.save(ssubdir + '/est_weights_pooled'+ '_' + timestamp + '.npy', est_error_pooled_array)
    
            mse_train_list.append(mse_train_array)
            mse_val_list.append(mse_val_array)
            mse_val_pooled_list.append(mse_val_pooled_array)

            est_error_list.append(est_error_array)
            est_error_pooled_list.append(est_error_pooled_array)

        # out is list of len(n_samples), each el array of shape (n_iters) for each sample size
        mse_t, mse_std_t = mse_mean_std(mse_train_list)
        mse_v, mse_std_v = mse_mean_std(mse_val_list)
        
        mse_t_scaled, mse_std_t_scaled = mse_mean_std_scaled(mse_train_list, mse_val_pooled_list)
        mse_v_scaled, mse_std_v_scaled = mse_mean_std_scaled(mse_val_list, mse_val_pooled_list)

        # weight vector est. error mean and std over repeatitions 
        est_error_means, est_error_std = mse_mean_std(est_error_list)
        est_error_means_scaled, est_error_std_scaled = mse_mean_std_scaled(est_error_list, est_error_pooled_list)

        os.mkdir(subdir + '/stats')
        with open(subdir + '/stats/params' + '_' + timestamp + '.json', 'w') as f:
                json.dump({
                        'adj_matrix':      adj_matrix,
                        'reg_term' :       reg_term,
                        'n_clusters':      n_clusters,
                        'n_ds':            n_ds,
                        'n_features' :     n_features,
                        'n_samples_list' : n_samples_list,
                        'd_m_ratio' :     [n_features/ x  for x in n_samples_list]}, f, indent=2)
                
        # save computed mse and std for each pair (reg.term, sample size)
        np.save(subdir + '/stats/mse_std' + '_' + timestamp + '.npy', [(mse_t, mse_std_t), (mse_v, mse_std_v)])
        np.save(subdir + '/stats/mse_std_scaled' + '_' + timestamp + '.npy', [(mse_t_scaled, mse_std_t_scaled), (mse_v_scaled, mse_std_v_scaled)])
        np.save(subdir + '/stats/est_error_mean_std' + '_' + timestamp + '.npy', (est_error_means, est_error_std))
        np.save(subdir + '/stats/est_error_mean_std_scaled' + '_' + timestamp + '.npy', (est_error_means_scaled, est_error_std_scaled))

    return path, mse_train_list, mse_val_list, mse_val_pooled_list


