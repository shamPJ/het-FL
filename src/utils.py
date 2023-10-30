import json
import matplotlib.pyplot as plt
import numpy as np
import torch 

def fwd_check(ds, model):
    # Check manually fwd step [only linear model!]
    X, y = ds[0], ds[1]
    
    w = model.model.weight.data.numpy()
    b = model.model.bias.data.numpy()
    
    out_manual = w@X.T + b
    out = model.predict(X)
    
    np.testing.assert_allclose(out.reshape(-1,), out_manual.reshape(-1,), rtol=1e-05)
    
def bckwd_check(ds, ds_test, pred_test, A, model):
    # Check manually bckw step [only linear model!]
    
    # Data
    X, y = ds[0], ds[1]
    X_test, y_test = ds_test[0], ds_test[1]
    
    # Convert to torch tensors
    X, y = torch.FloatTensor(X), torch.FloatTensor(y)
    X_test = torch.FloatTensor(X_test)
    
    # Sample size
    m = X.shape[0]
    m_test = X_test.shape[0]
    
    lmbd = 0.01
    
    # Convert np arrays to pytorch floats
    A = torch.from_numpy(A).float().reshape(-1,1)
    pred_test = torch.from_numpy(pred_test).float()

    # Get predictions for local and shared test ds
    pred = model(X)
    pred_test_local = model(X_test)
    
    # Compute gradients manually
    dw_local = -2/m*X.T@(y - pred)
    dw_test = lmbd/m_test*(X_test.T@((pred_test_local - pred_test)@A))  # (n,m')@((m',nn)@(nn,1))-->(n,1)
    dw = (dw_local + dw_test).detach().numpy().reshape(-1,1)
    
    # Compute gradients with pytorh
    model.zero_grad()  
    criterion = torch.nn.MSELoss(reduction='mean')
    # Compute loss
    loss_local = criterion(y, pred)
    loss_GTV = torch.mean( ((pred_test_local - pred_test)**2)@A )
    
    loss = loss_local + (lmbd/2)*loss_GTV
    # bckwd pass
    loss.backward()
    # gradients
    dw_torch = model.model.weight.grad.detach().numpy().reshape(-1,1)
    np.testing.assert_allclose(dw_torch, dw, rtol=1e-05)

#==================================== PLOTTING SINGLE EXPERIMENT ======================================#

def plot_weight_dist(G, true_weights):

    """
    
    Compute similarity (sq. L2 distance) between learnt weight vector of a pytorch linear model and true weight vector for all clusters
    and plot results.

    Note - set bias=False for linear model

    Args:
    : G            : list of (n_clusters*n_ds) python dicts (graph nodes) where each dict (node)  contain local train/ val datasets, model and shared dataset
    : true_weights : array of shape (n_clusters, n_features), true weight vector for each cluster

    Output:
    : dist         : array of shape (n_nodes, n_clusters)

    """

    n_nodes, n_clusters = len(G), len(true_weights)
    dist = np.zeros((n_nodes, n_clusters))

    for i in range(n_nodes):
        model_params = G[i]['model'].get_params()
        print("model_params.shape, true_weights.shape", model_params.shape, true_weights.shape)
        dist[i] = np.sum((model_params - true_weights)**2, axis=1)

    plt.title("Sq. L2 dist. b/ learnt weight vector of a node and true vectors of 3 clusters")
    plt.imshow(dist, cmap="coolwarm")
    plt.colorbar()
    plt.show()

    return dist
    
def compute_mse(preds, n_clusters):

    """

    Compute (i) MSE of predictions on a shared ds between all nodes, MSE(node_i, node_j)
    and (ii) average these MSE's by clusters
    
    Args:
    : preds      : array of shape (n_nodes, m_shared), predictions of local models on the shared dataset
    : n_clusters : int, number of true clusters

    Output:
    : mse_node   : array of shape (n_nodes, n_nodes), MSE of predictions on a shared ds between all nodes, MSE(node_i, node_j)
    : mse_aver   : array of shape (n_clusters, n_clusters), average MSE's (by clusters)

    """

    n_nodes  = preds.shape[0]               # number of nodes
    n_ds     = int(n_nodes / n_clusters)    # number of datasets per cluster (clusters have same number of ds)
    mse_node = np.zeros((n_nodes, n_nodes)) # array to store MSE

    for n in range(n_nodes):
        node_pred   = preds[n, :].reshape(1,-1)
        mse_node[n] = np.mean((node_pred - preds)**2, axis=1) # MSE of predictions on a shared ds of node `n` and each of other nodes

    mse_aver = np.zeros((n_clusters, n_clusters)) # Averaged MSE's (by cluster) of predictions on a shared ds 

    for i in range(n_clusters):
        for j in range(n_clusters):
            mse_aver[i,j] = np.mean(mse_node[n_ds*i:n_ds*i+n_ds, n_ds*j:n_ds*j+n_ds])
    
    return mse_node, mse_aver

def plot_preds_similarity(A, preds_list, n_clusters, n_iters):

    """

    Plot how similar (MSE) predictions of the nodes on shared ds (assuming that ds_shared is the same for all nodes). 
    First supblot is adjacency matrix, next matrices are MSE values on 1st, mid and last iterations. The last matrix is average of MSE's by clusters.
    Datasets in diagonal blocks are from the same cluster (thus MSE expected to be smaller).

    Args:
    : A          : array of shape (n_nodes, n_nodes), adjacency matrix of the graph, created with Bernoulli distribution and probabilities p_in and p_out.
    : preds_list : list of arrays (n_nodes, m_shared), predictions on the shared dataset on iteration 1, n_iters/2, n_iters
    : n_clusters : int, number of true clusters
    : n_iters    : int, number of iterations

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


#==================================== PLOTTING HYPERPARAMS ======================================#

def mse_mean_std(mse_local_list):

    """

    Average MSE values (across all repetitions and nodes). STD is computed across `repeat_times`.
    `k` is the number of hyperparams combinations. 

    :param mse_local_list : list with k elements. Each element is an array of shape (repeat_times, n_nodes) containing training or validation loss for repeat_times runs for each node in the graph. 

    :out mse_local_mean   : list with k elements (floats). Each element is an average across all runs of average across all nodes MSEs for each hyperparam combination.
    :out mse_local_std    : list with k elements (floats). Corresponding standard deviation. 

    """

    mse_local_mean, mse_local_std = [], []

    for mse_local in mse_local_list:
        # compute average across all dims
        mse_local_mean.append(np.mean(mse_local))
        # first average mse across nodes, then compute std across `repeat_times` (i.e. number of runs)
        std = np.std(np.mean(mse_local, axis=1))
        mse_local_std.append(std)

    return mse_local_mean, mse_local_std

def mse_mean_std_scaled(mse_list, mse_pooled_list):
    
    """

    Scaled average MSE values (across all repetitions and nodes). STD is computed across `repeat_times`.
    `k` is the number of hyperparams combinations. 

    :param mse_list        : list with k elements. Each element is an array of shape (repeat_times, n_nodes) containing training loss for repeat_times runs for each node in the graph. 
    :param mse_pooled_list : list with k elements. Each element is an array of shape (repeat_times, n_nodes) containing validation loss for repeat_times runs for each cluster model on the corresponding local node's val ds. 

    :out mse_mean          : list with k elements (floats). Each element is an average across all runs of average across all nodes MSEs for each hyperparam combination.
    :out mse_std           : list with k elements (floats). Corresponding standard deviation. 

    """
    
    mse_mean, mse_std = [], []

    for mse, mse_pooled in zip(mse_list, mse_pooled_list):
        
        # scale local MSE vals
        mse_scaled = mse / mse_pooled
        # first average mse across nodes, then compute std across `repeat_times` (i.e. number of runs)
        std = np.std(np.mean(mse_scaled, axis=1))

        # compute average across all dims
        mse_mean.append(np.mean(mse_scaled))
        mse_std.append(std)

    return mse_mean, mse_std

def plot_mse(ax, mse_mean, mse_std, reg_term_list, n_samples_list):

    """
    
    :param ax             : matplotlib.axes object
    :param mse_mean       : list with k elements (floats). Each element is an average across all runs of average across all nodes MSEs for each hyperparam combination.
    :param mse_std        : list with k elements (floats). Corresponding standard deviation. 
    :params reg_term_list : list with k elements (floats). Penalty term values used in `iter_params` func.
    :param n_samples_list : list with k elements (ints). Sizes of the local dataset used in `iter_params` func.

    :out ax               : matplotlib.axes object with plots

    """
    
    n_regs = len(reg_term_list)
    n_sizes = len(n_samples_list)
    
    for i in range(n_regs):

        y = mse_mean[i*n_sizes:i*n_sizes+n_sizes]
        y_err = mse_std[i*n_sizes:i*n_sizes+n_sizes]
        ax.errorbar(n_samples_list, y, yerr=y_err, label='Reg. term ' + str(reg_term_list[i]), lolims=True, linestyle='--')

    ax.set_xticks(n_samples_list)
    return ax

def save_and_plot(exp_results, model_hyperparams, reg_term_list, n_samples_list, name):

    """

    Save computed average MSE's and STD's for experiment with different model hyperparams, regularization and local dataset size.
    
    :param             : 
    :param        :  
    :param        : 
    :params reg_term_list : list with k elements (floats). Penalty term values used in `iter_params` func.
    :param n_samples_list : list with k elements (ints). Sizes of the local dataset used in `iter_params` func.

    :out ax               : matplotlib.axes object with plots

    """

    with open('out/img/' + name.replace(' ', '_') + '.json', 'w') as f:
        json.dump(exp_results, f)

    titles = ['Train ds', 'Validation ds']

    fig, axes = plt.subplots(len(model_hyperparams), 2, sharey=True, sharex=True, figsize=(8,8))

    for i in range(len(model_hyperparams)):
        # get experiment results for ith value of model hyperparam
        plot_list = exp_results[i]
        axs = axes[i]

        for ax, data, title in zip(axs, plot_list, titles):
            mse = data[0]
            mse_std = data[1]
            plot_mse(ax, mse, mse_std, reg_term_list, n_samples_list)
            ax.set_title(title + ' lrate = ' + str(model_hyperparams[i]))
    
    [axs[0].set_ylabel ('Loss') for axs in axes]

    axes[-1,0].set_xlabel ('Training ds size')
    axes[-1,1].set_xlabel ('Training ds size')

    plt.legend()
    fig.tight_layout()
    plt.savefig("out/img/" + name.replace(' ', '_') + ".png")
    plt.show()