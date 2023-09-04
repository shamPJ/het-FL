from generate_graph import build_edges, build_graph, build_graph_pooled
import matplotlib.pyplot as plt
import numpy as np

#======================== HELPER FUNCS ========================#

def plot_weight_dist(G, true_weights):

    """
    
    Compute similarity (sq. L2 distance) between learnt weight vector of a pytorch linear model and true weight vector for all clusters
    and plot results.

    Note - set bias=False for linear model

    :param G            : list of (n_clusters*n_ds) python dictionaries (graph nodes) where each dict (node)  contain local train/ val datasets, model and shared dataset
    :param true_weights : array of shape (n_clusters, n_features), true weight vector for each cluster

    :out dist           : array of shape (n_nodes, n_clusters)

    """

    n_nodes, n_clusters = len(G), len(true_weights)
    dist = np.zeros((n_nodes, n_clusters))

    for i in range(n_nodes):
        model_params = G[i]['model'].get_params()
        dist[i] = np.sum((model_params - true_weights)**2, axis=1)

    plt.title("Sq. L2 dist. b/ learnt weight vector of a node and true vectors of 3 clusters")
    plt.imshow(dist, cmap="coolwarm")
    plt.colorbar()
    plt.show()

    return dist
    
def compute_mse(preds, n_clusters):

    """

    Compute (i) MSE of predictions on a shared ds between all nodes, MSE(node_i, node_j) and (ii) average these MSE's by clusters

    :param preds      : array of shape (m_shared, n_nodes), predictions of local models on the shared dataset
    :param n_clusters : int, number of true clusters

    :out mse_node     : array of shape (n_nodes, n_nodes), MSE of predictions on a shared ds between all nodes, MSE(node_i, node_j)
    :out mse_aver     : array of shape (n_clusters, n_clusters), average MSE's (by clusters)

    """

    n_nodes  = preds.shape[1]               # number of nodes
    n_ds     = int(n_nodes / n_clusters)    # number of datasets per cluster (clusters have same number of ds)
    mse_node = np.zeros((n_nodes, n_nodes)) # array to store MSE

    for n in range(n_nodes):
        node_pred   = preds[:,n].reshape(-1,1)
        mse_node[n] = np.mean((node_pred - preds)**2, axis=0) # MSE of predictions on a shared ds of node `n` and each of other nodes

    mse_aver = np.zeros((n_clusters, n_clusters)) # Averaged MSE's (by cluster) of predictions on a shared ds 

    for i in range(n_clusters):
        for j in range(n_clusters):
            mse_aver[i,j] = np.mean(mse_node[n_ds*i:n_ds*i+n_ds, n_ds*j:n_ds*j+n_ds])
    
    return mse_node, mse_aver

def plot_preds_similarity(A, preds_list, n_clusters, n_iters):

    """

    Plot how similar (MSE) predictions of the nodes on shared ds. 
    First supblot is adjacency matrix, next matrices are MSE values on 1st, middle and last iterations. The last matrix is average of MSE's by clusters.
    Datasets in diagonal blocks are from the same cluster (thus MSE expected to be smaller).

    :param A          : np array of shape (n_nodes, n_nodes), adjacency matrix of the graph, created with Bernoulli distribution and probabilities p_in and p_out.
    :param preds_list : list of arrays (m_shared, n_nodes), predictions on the shared dataset on iteration 1, n_iters/2, n_iters
    :param n_clusters : int, number of true clusters
    :param n_iters    : int, number of iterations

    """
    
    fig, axs = plt.subplots(1, 5, figsize=(10,6))
    # Plot edges weights
    im1 = axs[0].imshow(A, cmap="RdBu")
    
    for i, preds in enumerate(preds_list):
        mse_node, _ = compute_mse(preds, n_clusters)
        # Plot matrix of MSE values on shared test dataset of (node_i, node_j) for all nodes in G
        im = axs[i+1].imshow(mse_node, cmap="coolwarm")
        fig.colorbar(im, ax=axs[i+1], shrink=0.6, location='bottom')
        
    # Same but MSE averaged by cluster for last iter
    _, mse_aver = compute_mse(preds_list[-1], n_clusters)
    im5 = axs[-1].imshow(mse_aver, cmap="coolwarm")
    
    # Set titles
    axs[0].set_title("Edge weghts (matrix A)")
    axs[1].set_title("Iteration 1")
    axs[2].set_title("Iteration " + str(int(n_iters/2)))
    axs[3].set_title("Iteration " + str(int(n_iters)))
    axs[4].set_title("Average MSE over clusters")

    fig.colorbar(im1, ax=axs[0], shrink=0.6, location='bottom')
    fig.colorbar(im5, ax=axs[4], shrink=0.6, location='bottom')

    fig.tight_layout()
    plt.show()

#=============================== MAIN FUNCS ==================================#

def iter(G, A, G_pooled, n_iters=1000, regularizer_term=0.01, verbose=False): 
    
    """
    
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


def train(config, models, models_pooled, verbose=False, plot_preds=True, plot_w_dist=False):

    """

    Build graph and run training loop.

    :param config        : dict, experiment parameters (keys: 'n_clusters', 'n_ds', 'n_samples', 'n_features', 'ds_train', 'ds_val', 'ds_shared',
                                                        'cluster_labels', 'true_weights', 'n_iters', 'regularizer_term')
    :param models        : list of len(n_nodes), pytorch, sklearn or keras local models 
    :param models_pooled : list of len(n_clusters), pytorch, sklearn or keras 'oracle' or 'pooled' models 
    :param verbose       : boolean, flag to control printing statements during training
    :param plot_preds    : boolean, flag to control plotting of MSE matrices 
    :plot_w_dist         : boolean, flag to control plotting of weight vectors sq. L2 difference matrix

    :out preds_list      : list of arrays (m_shared, n_nodes), predictions on the shared dataset on iteration 1, n_iters/2, n_iters
    :out mse_train       : array (n_nodes,), local training MSE error for each node
    :out mse_val         : array (n_nodes,), local validation MSE error for each node
    :out mse_val_pooled  : array (n_nodes,), local validation MSE error for each node incurred by corresponding "pooled" model

    """
    
    # Get data and training params
    ds_train, ds_val, ds_shared, cluster_labels, true_weights = config['ds_train'], config['ds_val'], config['ds_shared'], config['cluster_labels'], config['true_weights']
    n_clusters, n_iters, regularizer_term = config['n_clusters'], config['n_iters'], config['regularizer_term']
    
    G        = build_graph(ds_train, ds_val, ds_shared, models, cluster_labels) # Build a graph with nodes {model, dataset local, dataset test}
    G_pooled = build_graph_pooled(ds_train, ds_val, models_pooled, n_clusters=n_clusters) # Build a graph where local ds from one cluster are pooled
    A        = build_edges(G, cluster_labels)  # Build edges

    preds_list, mse_train, mse_val, mse_val_pooled = iter(G, A, G_pooled, n_iters=n_iters, regularizer_term=regularizer_term, verbose=verbose)

    # Plotting
    if plot_preds:
        plot_preds_similarity(A, preds_list, n_clusters, n_iters)

    if plot_w_dist:
        plot_weight_dist(G, true_weights)

    return preds_list, mse_train, mse_val, mse_val_pooled
