from copy import deepcopy
from generate_data import get_data, get_shared_data
from generate_graph import build_edges, build_graph, build_graph_pooled
import json
import numpy as np
import os
from utils import param_est_error, param_est_error_pooled

def train(G, A, G_pooled, n_iters=1000, regularizer_term=0.01, m_shared=100, parametric=True): 
    
    """

    Training loop for the case when adjacency matrix A is given.
    
    Args:
    : G                  : list of dicts [dict keys are: model, ds_train, ds_val, ds_shared, cluster_label], represents graph with n_nodes
    : A                  : numpy array of shape (n_nodes, n_nodes), adjacency matrix of the graph, created with Bernoulli distribution and probabilities p_in and p_out 
    : pooled_G           : list of dicts [dict keys are: model, ds_train, ds_val], represents graph with n_cluster nodes (local ds size = n_ds*n_samples). Data belonging to the ith-node is pooled ds of G, belonging to the ith-cluster
    : iters              : int, number of iterations or updates of the local model
    : regularizer_term   : float, scaling factor for GTV term
    : m_shared           : int, size of shared dataset
    : parametric         : boolean, if models are parametric, will store estimated models' params
    
    Output:
    : mse_train          : array (n_nodes, n_iters), local training MSE for each node for all iterations
    : mse_val            : array (n_nodes, n_iters), local validation MSE for each node for all iterations
    : mse_val_pooled     : array (n_nodes, n_iters), local validation MSE for each node incurred by corresponding "pooled" model for all iterations
    : est_weights        : array (n_iters, n_nodes, n_features), weights computed with GD updates (valid only for parametric models)
    : est_weights_pooled : array (n_iters, n_clusters, n_features), weights computed with GD updates for pooled model

    """
    n_nodes = len(G)                                # n.o. nodes in the graph
    n_clusters = len(G_pooled)                      # n.o. clusters
    n_features = G[0]["ds_train"][0].shape[1]       # n.o. features of local datasets

    # Create shared dataset
    ds_shared = get_shared_data(m_shared, n_features)

    # MSE on local ds incurred by models on each iteration
    mse_train        = np.zeros((n_nodes, n_iters))
    mse_val          = np.zeros((n_nodes, n_iters))
    mse_train_pooled = np.zeros((n_nodes, n_iters))
    mse_val_pooled   = np.zeros((n_nodes, n_iters))
    # estimated weights
    est_weights    = np.zeros((n_iters, n_nodes, n_features))
    est_weights_pooled = np.zeros((n_iters, n_clusters, n_features))

    pred_shared = np.zeros((n_nodes, m_shared))     # init predictions on a shared ds
    
    for i in range(n_iters):
        # Update pooled (oracle) models
        for n in range(n_clusters):
            ds_train = G_pooled[n]["ds_train"]
            model    = G_pooled[n]['model']
            _        = model.update_pooled(ds_train) 
            if parametric: est_weights_pooled[i, n]  = model.get_params().reshape(-1,)

        # Update local models
        for n in range(n_nodes):
            ds_train  = G[n]["ds_train"]
            ds_val    = G[n]["ds_val"]
            c         = G[n]["cluster_label"]
            model     = G[n]["model"]

            # Train model
            _ = model.update(ds_train, ds_shared, pred_shared, A[n], regularizer_term) 

            # Get predictions
            pred_train     = model.predict(ds_train[0])
            pred_val       = model.predict(ds_val[0])
            pred_shared[n] = model.predict(ds_shared).reshape(-1,)

            # Compute loss
            mse_train[n,i] = np.mean((ds_train[1] - pred_train.reshape(-1,1))**2)
            mse_val[n,i]   = np.mean((ds_val[1] - pred_val.reshape(-1,1))**2)
            
            # Save models params
            if parametric: est_weights[i, n] = model.get_params().reshape(-1,)  
        
            # Compute MSE on train/ validation data using pooled model 
            model_pooled  =  G_pooled[c]["model"]   
            pred_pooled_train, pred_pooled_val = model_pooled.predict(ds_train[0]), model_pooled.predict(ds_val[0])
            mse_train_pooled[n, i] = np.mean((ds_train[1] - pred_pooled_train.reshape(-1,1))**2)
            mse_val_pooled[n, i] = np.mean((ds_val[1] - pred_pooled_val.reshape(-1,1))**2)  # shape of ds_train/val[1] is (m, 1) 
        
    return mse_train, mse_val, mse_train_pooled, mse_val_pooled, est_weights, est_weights_pooled


def train_no_A(G, G_pooled, n_neighbours=5, n_neighbours_fixed=False, n_iters=1000, regularizer_term=0.01, m_shared=100, parametric=True): 
    
    """

    Training loop for the case when A is not given.
    
    Args:
    : G                   : list of dicts [dict keys are: model, ds_train, ds_val, ds_shared, cluster_label], represents graph with n_nodes
    : pooled_G            : list of dicts [dict keys are: model, ds_train, ds_val], represents graph with n_cluster nodes (local ds size = n_ds*n_samples). Data belonging to the ith-node is pooled ds of G, belonging to the ith-cluster
    : n_neighbours        : int, n.o. neighbours to connect to (choose p-lowest ||pred_i - pred_j||^2)
    : n_neighbours_fixed  : boolean, flag to indicate if we init adjacency matrix on each iteration (Algo 3) or not (Algo 2)
    : iters               : int, number of iterations or updates of the local model
    : regularizer_term    : float, scaling factor for GTV term
    : m_shared            : int, size of shared dataset
    : parametric          : boolean, if models are parametric, will store estimated models' params
    
    Output:
    : mse_train          : array (n_nodes, n_iters), local training MSE for each node for all iterations
    : mse_val            : array (n_nodes, n_iters), local validation MSE for each node for all iterations
    : mse_val_pooled     : array (n_nodes, n_iters), local validation MSE for each node incurred by corresponding "pooled" model for all iterations
    : est_weights        : array (n_iters, n_nodes, n_features), weights computed with GD updates (valid only for parametric models)
    : est_weights_pooled : array (n_iters, n_clusters, n_features), weights computed with GD updates for pooled model

    """

    n_nodes = len(G)                                # n.o. nodes in the graph
    n_clusters = len(G_pooled)                      # n.o. clusters
    n_features = G[0]["ds_train"][0].shape[1]       # n.o. features of local datasets

    # Create shared dataset
    ds_shared = get_shared_data(m_shared, n_features)

    # MSE on local ds incurred by models on each iteration
    mse_train        = np.zeros((n_nodes, n_iters))
    mse_val          = np.zeros((n_nodes, n_iters))
    mse_train_pooled = np.zeros((n_nodes, n_iters))
    mse_val_pooled   = np.zeros((n_nodes, n_iters))
    # estimated weights
    est_weights    = np.zeros((n_iters, n_nodes, n_features))
    est_weights_pooled = np.zeros((n_iters, n_clusters, n_features))

    pred_shared = np.zeros((n_nodes, m_shared))     # init predictions on a shared ds
    A = np.zeros((n_nodes, n_nodes))    # init adjacency matrix
    
    for i in range(n_iters):
        # Update pooled (oracle) models
        for n in range(n_clusters):
            ds_train = G_pooled[n]["ds_train"]
            model    = G_pooled[n]['model']
            _        = model.update_pooled(ds_train) 
            if parametric: est_weights_pooled[i, n]  = model.get_params().reshape(-1,)

        # 1. Train models
        for n in range(n_nodes):
            ds_train  = G[n]["ds_train"]
            ds_val    = G[n]["ds_val"]
            c         = G[n]["cluster_label"]
            model     = G[n]["model"]

            # Train model
            _ = model.update(ds_train, ds_shared, pred_shared, A[n], regularizer_term) 

            # Get predictions
            pred_train     = model.predict(ds_train[0])
            pred_val       = model.predict(ds_val[0])
            pred_shared[n] = model.predict(ds_shared).reshape(-1,)

            # Compute loss
            mse_train[n,i] = np.mean((ds_train[1] - pred_train.reshape(-1,1))**2)
            mse_val[n,i]   = np.mean((ds_val[1] - pred_val.reshape(-1,1))**2)
            
            # Save models params
            if parametric: est_weights[i, n] = model.get_params().reshape(-1,)  
        
            # Compute MSE on train/ validation data using pooled model 
            model_pooled  =  G_pooled[c]["model"]   
            pred_pooled_train, pred_pooled_val = model_pooled.predict(ds_train[0]), model_pooled.predict(ds_val[0])
            mse_train_pooled[n, i] = np.mean((ds_train[1] - pred_pooled_train.reshape(-1,1))**2)
            mse_val_pooled[n, i] = np.mean((ds_val[1] - pred_pooled_val.reshape(-1,1))**2)  # shape of ds_train/val[1] is (m, 1) 
        
        # 2. Update agjacency matrix A by choosing neighbors with lowest ||pred_i - pred_j||^2
        if n_neighbours_fixed: A = np.zeros((n_nodes, n_nodes))    # zero out adjacency matrix
        for n in range(n_nodes):
            dist_ij = np.mean((pred_shared[n].reshape(1,-1) - pred_shared)**2, axis=-1) # shape (n_nodes,)
            min_dist_idx = np.argsort(dist_ij)
            min_dist_idx = min_dist_idx[:n_neighbours+1]    # choose n_neighbours neighbors (+1 node itself)
            A[n][min_dist_idx] = 1     # set weight A_ij = 1 for k neighbors
            
        np.fill_diagonal(A, 0) # set self-connections A_ii=0
        
    return mse_train, mse_val, mse_train_pooled, mse_val_pooled, est_weights, est_weights_pooled

def repeat_train(n_samples, config):

    """
    Repeat experiment (with given reg. term and local dataset size n_samples) repeat_times to compute mean and SD.

    Args:
    : config : dict containing necessary parameters.
    
    Outputs:
    :out     : tuple of (mse_train, mse_val, mse_val_pooled, est_error, est_error_pooled). Where
                - mse_train is array of shape (repeat_times, n_nodes, n_iters) containing training loss for repeat_times runs for each node in the graph. 
                - mse_val is array of shape (repeat_times, n_nodes, n_iters) containing validation loss for repeat_times runs for each node in the graph. 
                - mse_val_pooled is array of shape (repeat_times, n_nodes, n_iters) containing validation loss for repeat_times runs for each cluster model on the local nodes val ds. 
                - est_error is array of shape (repeat_times, n_nodes, n_iters), sum of sq.L2 norm of difference vector (true cluster weights - estimated weights) for each node
                - est_error_pooled is array of shape (repeat_times, n_nodes, n_iters), sum of sq.L2 norm of difference vector (true cluster weights - estimated cluster weights) for each node in the 'pooled' model
    
    Save arrays in corresponding '/reg_term_X_n_samples_Y' subdirectory.

    """
    
    # Config
    n_clusters, n_ds, n_features, m_shared, noise_val = config['n_clusters'], config['n_ds'], config['n_features'], config['m_shared'], config['noise_val']
    models, models_pooled = config['models'], config['models_pooled']
    p_in, p_out = config['p_in'], config['p_out']
    
    reg_term, n_iters  = config['reg_term'], config['n_iters']
    repeat_times = config['repeat_times']
    exp_dir = config['exp_dir']
    parametric, adj_matrix = config['parametric'], config['adj_matrix']
    if not adj_matrix: n_neighbours, n_neighbours_fixed = config['n_neighbours'], config['n_neighbours_fixed']

    n_nodes = n_clusters * n_ds

    # Arrays to store repeatitions of experiment
    mse_train_array        = np.zeros((repeat_times, n_nodes, n_iters))
    mse_val_array          = np.zeros((repeat_times, n_nodes, n_iters))
    mse_train_pooled_array = np.zeros((repeat_times, n_nodes, n_iters))
    mse_val_pooled_array   = np.zeros((repeat_times, n_nodes, n_iters))

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
        
        # Build a graph with nodes as dict containing local train/val ds, model and cluster assignment
        G = build_graph(ds_train, ds_val, models, cluster_labels)
        if adj_matrix: A = build_edges(G, cluster_labels, p_in = p_in, p_out = p_out)
        G_pooled = build_graph_pooled(ds_train, ds_val, models_pooled, n_clusters=n_clusters)

        if adj_matrix:
            mse_train, mse_val, mse_train_pooled, mse_val_pooled, est_weights, est_weights_pooled = train(G, A, G_pooled, n_iters=n_iters, regularizer_term=reg_term, m_shared=m_shared, parametric=parametric)
        else:
            mse_train, mse_val, mse_train_pooled, mse_val_pooled, est_weights, est_weights_pooled = train_no_A(G, G_pooled, n_neighbours=n_neighbours, n_neighbours_fixed=n_neighbours_fixed, n_iters=n_iters, regularizer_term=reg_term, m_shared=m_shared, parametric=parametric)
        
        # Save mse on a local ds
        mse_train_array[i] = mse_train
        mse_val_array[i] = mse_val
        mse_train_pooled_array[i] = mse_train_pooled
        mse_val_pooled_array[i] = mse_val_pooled

        # Save parameter vector estimation error
        est_error_array[i] = param_est_error(true_weights, est_weights, cluster_labels)
        est_error_pooled_array[i] = param_est_error_pooled(true_weights, est_weights_pooled, cluster_labels)

    # update config file
    config['models_pooled'] = str(models_pooled)
    config['models'] = str(models)
    config['n_samples'] = n_samples
    if not adj_matrix: config['n_neighbours'] = n_neighbours
    if not adj_matrix: config['n_neighbours_fixed'] = n_neighbours_fixed

    # Create subdir to save experiment with current reg_term and n_ds value
    subdir = exp_dir + '/' + 'reg_term_' + str(reg_term) + '_n_samples_' + str(n_samples)
    os.mkdir(subdir)
    
    config_path = subdir + '/config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
        
    np.save(subdir + '/mse_train.npy', mse_train_array)
    np.save(subdir + '/mse_val.npy', mse_val_array)
    np.save(subdir + '/mse_train_pooled.npy', mse_train_pooled_array)
    np.save(subdir + '/mse_val_pooled.npy', mse_val_pooled_array)

    np.save(subdir + '/est_error.npy', est_error_array)
    np.save(subdir + '/est_error_pooled.npy', est_error_pooled_array)

    return mse_train_array, mse_val_array, mse_train_pooled_array, mse_val_pooled_array, est_error_array, est_error_pooled_array
