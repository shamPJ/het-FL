from glob import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import torch 

#============================================= PYTORCH MANUAL CHECK =====================================================#
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

#================================================== HELPER PLOT =========================================================#
#==================================== MSE of predictions of the nodes on shared ds ======================================#

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

#============================================== COMPUTE STATS and PLOT ==================================================#

def mean_sd(loss_list):

    """

    Average MSE (or other loss/error) values (across all repetitions and nodes for each iteration) for each sample size. SD is computed across `repeat_times`.

    Args:
    : loss_list : list with len(n_samples) elements. Each element is an array of shape (repeat_times, n_nodes, n_iters) 
                       containing MSE for repeat_times runs for each node in the graph for each iteration. 

    Output:
    : mean_list : list with len(n_samples) elements. Each element is an array (n_iters) - average across all runs of average across all nodes MSEs for each hyperparam combination.
    : sd_list   : list with len(n_samples) elements. Each element is an array (n_iters). Corresponding standard deviation across all runs (). 

    """

    mean_list, sd_list = [], []

    for loss in loss_list:
        # loss shape is (repeat_times, n_nodes, n_iters)
        # compute average across (repeat_times, n_nodes) dims
        mean_list.append(np.mean(loss, axis=(0,1)))
        # first average loss across nodes, then compute sd across `repeat_times` (i.e. number of runs)
        sd = np.std(np.mean(loss, axis=1), axis=0)
        sd_list.append(sd)

    return mean_list, sd_list

def mean_sd_scaled(loss_list, loss_list_pooled):
    
    """

    Scaled average MSE (or other loss) values (across all repetitions and nodes) for each sample size. SD is computed across `repeat_times`. 

    Args:
    : loss_list        : list with len(n_samples) elements. Each element is an array of shape (repeat_times, n_nodes, n_iters) containing training loss for repeat_times runs for each node in the graph. 
    : loss_list_pooled : list with len(n_samples) elements. Each element is an array of shape (repeat_times, n_nodes, n_iters) containing validation loss for repeat_times runs for each "oracle" model on the corresponding local node's val ds. 

    Output:
    : loss_mean        : list with len(n_samples) elements. Each element is an array (n_iters) - average across all runs of average across all nodes MSEs for each hyperparam combination.
    : loss_sd          : list with len(n_samples) elements. Each element is an array (n_iters). Corresponding standard deviation. 

    """
    
    loss_mean, loss_sd = [], []

    for loss, loss_pooled in zip(loss_list, loss_list_pooled):
        
        # scale local MSE vals
        loss_scaled = loss / loss_pooled
        # compute average across (repeat_times, n_nodes) dims
        loss_mean.append(np.mean(loss_scaled, axis=(0,1)))
        # first average loss across nodes, then compute sd across `repeat_times` (i.e. number of runs)
        sd = np.std(np.mean(loss_scaled, axis=1), axis=0)
        loss_sd.append(sd)

    return loss_mean, loss_sd

def plot_mean_sd(ax, mean_list, sd_list, d_m_ratio, reg_term, title=False):

    """
    
    Function to plot a subplot with average MSE's and error bars.
    Each line correspond to the value d/m is the number of hyperparams combinations. 

    Args:
    : ax         : matplotlib.axes object
    : mean_list  : list. Each element is an average across all runs of average across all nodes MSEs for each hyperparam combination.
    : mse_std    : list. Corresponding standard deviation. 
    : d_m_ratio  : list of corresponding d_m_ratio's.
    : reg_term   : lambda value, reg.term.

    Output:
    : ax         : matplotlib.axes object with plots

    """
    
    n_lines, iters = mean_list.shape[0], mean_list.shape[1]
    assert n_lines == len(d_m_ratio)
    
    for i in range(n_lines):
        y = mean_list[i]
        y_err = sd_list[i]
        ax.plot(range(1, iters+1), y, label='d/m ratio ' + str(d_m_ratio[i]))
        ax.fill_between(range(1, iters+1), y - y_err, y + y_err, alpha=0.2)
        
    ax.spines[['right', 'top']].set_visible(False)
    if title: ax.title.set_text('reg.term = ' + str(reg_term))
    ax.set_xticks(range(0, iters+1, int(iters / 4)))
    return ax

def load_and_plot_mse(p_dir, scaled=False):

    """
    Args:
    : p_dir : str, path to saved .npy files in /stats dir. npy file is in format [(mse_t, mse_std_t), (mse_v, mse_std_v)]
            p_dir = '/scratch/work/abduras1/het-FL/out/Linreg_Torch_' + os.environ["SLURM_ARRAY_JOB_ID"]

    Out:

    """

    # list subdirs ".../reg_term_0/" etc.
    subdirs = glob(p_dir + '/*/')
    # indexes for ordered subdirs by increasing lambda
    # subdir.split('/')[-2] gives 'reg_term_0'
    indx = np.argsort([float(subdir.split('/')[-2].split('_')[-1]) for subdir in subdirs])
    # figure with 2 rows (upper row training loss, lower - validation), n cols corresponding to different reg.term value
    fig, axes = plt.subplots(2, len(subdirs), sharey=True, figsize=(8,8))

    for i, ind in enumerate(indx):
        f = subdirs[ind]
        params = glob(f + 'stats/*.json')
        with open(params[0]) as file:
            param_dict = json.load(file)

        # get experiment settings    
        d_m_ratio = param_dict['d_m_ratio']
        reg_term = param_dict['reg_term']
        # load MSE values
        npy_files = glob(f + 'stats/*.npy')
        
        for npy_f in npy_files:
            if scaled:
                if 'mse_std_scaled' in (npy_f.split('/')[-1]):
                    (mse_t, mse_std_t), (mse_v, mse_std_v)= np.load(npy_f)
            else:
                if ('mse_std' in (npy_f.split('/')[-1])) and ('scaled' not in (npy_f.split('/')[-1])):
                    (mse_t, mse_std_t), (mse_v, mse_std_v)= np.load(npy_f)

        # plot subplot
        axes[0, i] = plot_mean_sd(axes[0, i], mse_t, mse_std_t, d_m_ratio, reg_term, title=True)
        axes[1, i] = plot_mean_sd(axes[1, i], mse_v, mse_std_v, d_m_ratio, reg_term)

    axes[1, -1].legend()
    axes[0, 0].set_ylabel ('Training loss') 
    axes[1, 0].set_ylabel ('Validation loss') 
    #axes[1, 0].set_ylim (0,30) 
    [axs.set_xlabel ('Iter') for axs in axes[1]]

    if scaled:
        file_name = '_scaled.png'
    else:
        file_name = '.png'
    
    fig.tight_layout()
    plt.savefig(p_dir + '/' + p_dir.split(sep='/')[-1] + file_name)

def param_est_error(true_weights, est_weights, cluster_labels):

    """
    Args:
    : true_weights   : array of shape (n_clusters, n_features), true weight vector for each cluster
    : est_weights    : array (n_iters, n_nodes, n_features), weights computed with GD updates (valid only for parametric models)
    : cluster_labels : list of (n_clusters*n_ds) cluster assignments for each local dataset 
    
    Out:
    : est_error      : array (n_nodes, n_iters)
    """

    n_nodes, n_features = len(cluster_labels), true_weights.shape[1]
    true_w = np.zeros((n_nodes, n_features))

    for i in range(n_nodes):
        true_w[i] = true_weights[cluster_labels[i]]
    
    true_w = true_w.reshape(1, n_nodes, n_features)
    # est error ||w-w_hat||^2
    est_error = np.sum((true_w - est_weights)**2, axis=-1)

    return est_error.T

def param_est_error_pooled(true_weights, est_weights_pooled, cluster_labels):

    """
    For each node compute est error for model parameters of corresponding pooled model.

    Args:
    : true_weights       : array of shape (n_clusters, n_features), true weight vector for each cluster
    : est_weights_pooled : array (n_iters, n_clusters, n_features), weights computed with GD updates for pooled model
    : cluster_labels     : list of (n_clusters*n_ds) cluster assignments for each local dataset 

    Out:
    : est_error_pooled   : array (n_nodes, n_iters)

    """

    n_nodes, n_features, n_iters = len(cluster_labels), true_weights.shape[1], est_weights_pooled.shape[0]
    true_w = np.zeros((1, n_nodes, n_features))
    est_w_pooled = np.zeros((n_iters, n_nodes, n_features))

    for i in range(n_nodes):
        true_w[:, i] = true_weights[cluster_labels[i]]
        # select weight vector of corresponding pooled model, out shape (n_iters, n_nodes, n_features)
        est_w_pooled[:, i] = est_weights_pooled[:, cluster_labels[i]]
    
    # est error ||w-w_hat||^2
    est_error_pooled = np.sum((true_w - est_w_pooled)**2, axis=-1)

    return est_error_pooled.T

def load_and_plot_est_error(p_dir):

    """
    Args:
    : p_dir : str, path to saved .npy files in /stats dir. npy file is in format [mse_t, mse_std_t]

    Out:

    """

    subdirs = glob(p_dir + '/*/')
    # indexes for ordered subdirs by increasing lambda
    indx = np.argsort([float(subdir.split('/')[-2].split('_')[-1]) for subdir in subdirs])
    # figure with 2 rows (upper row training loss, lower - validation), n cols corresponding to different reg.term value
    fig, axes = plt.subplots(2, len(subdirs), sharey='row', figsize=(8,8))

    for i, ind in enumerate(indx):
        f = subdirs[ind]
        params = glob(f + 'stats/*.json')
        with open(params[0]) as file:
            param_dict = json.load(file)

        # get experiment settings    
        d_m_ratio = param_dict['d_m_ratio']
        reg_term = param_dict['reg_term']
        # load MSE values
        npy_files = glob(f + 'stats/*.npy')
        for npy_f in npy_files:
            if 'est_error' in npy_f.split('/')[-1]:
                if 'scaled' in npy_f.split('/')[-1]:
                    est_error_means_scaled, est_error_std_scaled = np.load(npy_f)
                else:
                    est_error_means, est_error_std = np.load(npy_f)

        # plot subplot
        axes[0, i] = plot_mean_sd(axes[0, i], est_error_means, est_error_std, d_m_ratio, reg_term, title=True)
        axes[1, i] = plot_mean_sd(axes[1, i], est_error_means_scaled,  est_error_std_scaled, d_m_ratio, reg_term)

    axes[1, -1].legend()
    axes[0, 0].set_ylabel ('Weight vector est. error') 
    axes[1, 0].set_ylabel ('Weight vector est. error, scaled') 
    [axs.set_xlabel ('Iter') for axs in axes[1]]
    
    fig.tight_layout()
    plt.savefig(p_dir + '/' + p_dir.split(sep='/')[-1] + '_est_error.png')