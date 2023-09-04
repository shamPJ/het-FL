import numpy as np

def train(G, A, G_pooled, n_iters=1000, regularizer_term=0.01, verbose=False): 
    
    """

    Training loop.
    
    :param G                : list of dicts [dict keys are: model, ds_train, ds_val, ds_shared, cluster_label], represents graph with n_nodes
    :param pooled_G         : list of dicts [dict keys are: model, ds_train, ds_val], represents graph with n_cluster nodes (local ds size = n_ds*n_samples). Data belonging to the ith-node is pooled ds of G, belonging to the ith-cluster
    :param iters            : number of iterations or updates of the local model
    :param regularizer_term : scaling factor for GTV term

    :out preds_list         : list of arrays (m_shared, n_nodes), predictions on the shared dataset on iteration 1, n_iters/2, n_iters
    :out mse_train          : array (n_nodes,), local training MSE error for each node
    :out mse_val            : array (n_nodes,), local validation MSE error for each node
    :out mse_val_pooled     : array (n_nodes,), local validation MSE error for each node incurred by corresponding "pooled" model

    """
    
    n_nodes = len(G)                                # number of nodes in the graph
    m_shared = G[0]["ds_shared"][0].shape[0]        # sample size of the shared ds
    nodes_preds = np.zeros((m_shared, n_nodes))     # init predictions on a shared ds
    preds_list = []                                 # save predictions on a test set for iter 1, n_iters/2, n_iters for plotting
    
    for i in range(n_iters):
        # Update local models
        for n in range(n_nodes):
            ds_train  = G[n]["ds_train"]
            ds_shared = G[n]["ds_shared"]
            model     = G[n]["model"]
            _ = model.update(ds_train, ds_shared, nodes_preds, A[n], regularizer_term) 

        # Update predictions on a shared test set 
        nodes_preds = np.zeros((m_shared, n_nodes)) 
        for n in range(n_nodes):
            ds_train  = G[n]["ds_train"]
            ds_shared = G[n]["ds_shared"]
            model     = G[n]["model"]

            # model predictions
            pred_train  = model.predict(ds_train[0])
            pred_shared = model.predict(ds_shared[0])
            # store preds on a shared ds for plotting
            nodes_preds[:, n] = pred_shared.reshape(-1,)

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

        mse_train[n]      = np.mean((ds_train[1] - pred_train.reshape(-1,1))**2)
        mse_val[n]        = np.mean((ds_val[1] - pred_val.reshape(-1,1))**2)
        mse_val_pooled[n] = np.mean((ds_val[1] - pred_pooled.reshape(-1,1))**2)
        
    return preds_list, mse_train, mse_val, mse_val_pooled