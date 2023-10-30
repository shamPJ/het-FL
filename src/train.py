from generate_data import get_shared_data
import numpy as np

def train(G, A, G_pooled, n_iters=1000, regularizer_term=0.01, m_shared=100, ds_shared_mode='const', verbose=False): 
    
    """

    Training loop.

    There are different options for shared dataset generation with a param `ds_shared_mode`:
    (1) 'const' : ds generated once, it is the same for all nodes and iterations.
    (2) 'diff'  : ds generated for each node separately
    (3) 'diff_on_iter' : ds generated for each node separately and on each iteration

    Note, that plotting using `preds_list` make sense only for ds_shared_mode='const', 
    where we can visually check if predictions on a `ds_shared` is similar among nodes belonging to the same cluster. 
    
    Args:
    : G                : list of dicts [dict keys are: model, ds_train, ds_val, ds_shared, cluster_label], represents graph with n_nodes
    : pooled_G         : list of dicts [dict keys are: model, ds_train, ds_val], represents graph with n_cluster nodes (local ds size = n_ds*n_samples). Data belonging to the ith-node is pooled ds of G, belonging to the ith-cluster
    : iters            : number of iterations or updates of the local model
    : regularizer_term : scaling factor for GTV term
    : m_shared         : int, size of shared dataset(s)
    : ds_shared_mode   : str, controls how to generate ds_shared
    
    Output:
    : preds_list         : list of arrays (n_nodes, m_shared), predictions on the shared dataset on iteration 1, n_iters/2, n_iters
    : mse_train          : array (n_nodes,), local training MSE error for each node
    : mse_val            : array (n_nodes,), local validation MSE error for each node
    : mse_val_pooled     : array (n_nodes,), local validation MSE error for each node incurred by corresponding "pooled" model

    """
    n_nodes = len(G)                                # n.o. nodes in the graph
    n_features = G[0]["ds_train"][0].shape[1]       # n.o. features of local datasets

    # Create shared dataset(s):
    if ds_shared_mode == 'const' : ds_shared = get_shared_data(1, m_shared, n_features)
    if ds_shared_mode == 'diff' : ds_shared = get_shared_data(n_nodes, m_shared, n_features)

    nodes_preds = np.zeros((n_nodes, m_shared))     # init predictions on a shared ds
    preds_list = []                                 # save predictions on a shared ds for iter 1, n_iters/2, n_iters for plotting
    
    for i in range(n_iters):
        # sample new shared datasets on each iter
        if ds_shared_mode == 'diff_on_iter' : ds_shared = get_shared_data(n_nodes, m_shared, n_features)

        # Update local models
        for n in range(n_nodes):
            ds_train  = G[n]["ds_train"]
            model     = G[n]["model"]
            _ = model.update(ds_train, ds_shared, nodes_preds, A[n], regularizer_term) 

        # Update predictions on a shared set(s)
        nodes_preds = np.zeros((n_nodes, m_shared)) 
        for n in range(n_nodes):
            ds_train  = G[n]["ds_train"]
            model     = G[n]["model"]

            # model predictions
            pred_train  = model.predict(ds_train[0])
            pred_shared = model.predict(ds_shared[0]) if ds_shared_mode == 'const' else model.predict(ds_shared[n])
            # store preds on a shared ds for plotting
            nodes_preds[n, :] = pred_shared.reshape(-1,)

            if verbose:
                print(f"Iteration {i+1}")
                print(f"Node {n+1}, Training Loss {np.mean((ds_train[1] - pred_train.reshape(-1,1))**2): .2f}")
        if (i==1) | (i==n_iters/2-1) | (i==n_iters-1):
            # store preds on a shared ds for plotting
            preds_list.append(nodes_preds)
        
        # Update pooled models
        for node in G_pooled:
            ds_train = node["ds_train"]
            model    = node['model']
            _        = model.update_pooled(ds_train) 
            
    # Compute MSE on local ds incurred by trained models
    mse_train      = np.zeros((n_nodes,))
    mse_val        = np.zeros((n_nodes,))
    mse_val_pooled = np.zeros((n_nodes,))
    
    for n in range(n_nodes):
        # node's (local) data and model
        ds_train = G[n]["ds_train"]
        ds_val   = G[n]["ds_val"]
        model    = G[n]["model"]
        c        = G[n]["cluster_label"]
        # get corresponding pooled model 
        model_pooled  =  G_pooled[c]["model"]

        # model predictions
        pred_train  = model.predict(ds_train[0])
        pred_val    = model.predict(ds_val[0])
        pred_pooled = model_pooled.predict(ds_val[0])
        # shape of ds_train/val[1] is (m, 1) 
        mse_train[n]      = np.mean((ds_train[1] - pred_train.reshape(-1,1))**2)
        mse_val[n]        = np.mean((ds_val[1] - pred_val.reshape(-1,1))**2)
        mse_val_pooled[n] = np.mean((ds_val[1] - pred_pooled.reshape(-1,1))**2)
        
    return preds_list, mse_train, mse_val, mse_val_pooled