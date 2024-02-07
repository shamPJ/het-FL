from generate_data import get_shared_data
import numpy as np

def train(G, A, G_pooled, n_iters=1000, regularizer_term=0.01, m_shared=100, parametric=True): 
    
    """

    Training loop for the case when A is given.
    
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
    ds_shared = get_shared_data(1, m_shared, n_features)

    # MSE on local ds incurred by models on each iteration
    mse_train      = np.zeros((n_nodes, n_iters))
    mse_val        = np.zeros((n_nodes, n_iters))
    mse_val_pooled = np.zeros((n_nodes, n_iters))
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
            pred_shared[n] = model.predict(ds_shared[0]).reshape(-1,)

            # Compute loss
            mse_train[n,i] = np.mean((ds_train[1] - pred_train.reshape(-1,1))**2)
            mse_val[n,i]   = np.mean((ds_val[1] - pred_val.reshape(-1,1))**2)
            
            # Save models params
            if parametric: est_weights[i, n] = model.get_params().reshape(-1,)  
        
            # Compute MSE on validation data using pooled model 
            model_pooled  =  G_pooled[c]["model"]   
            pred_pooled = model_pooled.predict(ds_val[0])   
            mse_val_pooled[n, i] = np.mean((ds_val[1] - pred_pooled.reshape(-1,1))**2)  # shape of ds_train/val[1] is (m, 1) 
        
    return mse_train, mse_val, mse_val_pooled, est_weights, est_weights_pooled


def train_no_A(G, G_pooled, k=5, n_iters=1000, regularizer_term=0.01, m_shared=100, parametric=True): 
    
    """

    Training loop for the case when A is not given.
    
    Args:
    : G                  : list of dicts [dict keys are: model, ds_train, ds_val, ds_shared, cluster_label], represents graph with n_nodes
    : pooled_G           : list of dicts [dict keys are: model, ds_train, ds_val], represents graph with n_cluster nodes (local ds size = n_ds*n_samples). Data belonging to the ith-node is pooled ds of G, belonging to the ith-cluster
    : k                  : n.o. neighbours to connect to (choose k-lowest ||pred_i - pred_j||^2)
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
    ds_shared = get_shared_data(1, m_shared, n_features)

    # MSE on local ds incurred by models on each iteration
    mse_train      = np.zeros((n_nodes, n_iters))
    mse_val        = np.zeros((n_nodes, n_iters))
    mse_val_pooled = np.zeros((n_nodes, n_iters))
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
            pred_shared[n] = model.predict(ds_shared[0]).reshape(-1,)

            # Compute loss
            mse_train[n,i] = np.mean((ds_train[1] - pred_train.reshape(-1,1))**2)
            mse_val[n,i]   = np.mean((ds_val[1] - pred_val.reshape(-1,1))**2)
            
            # Save models params
            if parametric: est_weights[i, n] = model.get_params().reshape(-1,)  
        
            # Compute MSE on validation data using pooled model 
            model_pooled  =  G_pooled[c]["model"]   
            pred_pooled = model_pooled.predict(ds_val[0])   
            mse_val_pooled[n, i] = np.mean((ds_val[1] - pred_pooled.reshape(-1,1))**2)  # shape of ds_train/val[1] is (m, 1) 
        
        # 2. Update agjacency matrix A by choosing neighbors with lowest ||pred_i - pred_j||^2
        for n in range(n_nodes):
            dist_ij = np.mean((pred_shared[n] - pred_shared)**2, axis=-1) # shape (n_nodes,)
            min_dist_idx = np.argsort(dist_ij)
            min_dist_idx = min_dist_idx[:k]    # choose k neighbors (+1 node itself)
            A[n][min_dist_idx] = 1     # set weight A_ij = 1 for k neighbors
            
        np.fill_diagonal(A, 0) # set self-connections A_ii=0
        
    return mse_train, mse_val, mse_val_pooled, est_weights, est_weights_pooled