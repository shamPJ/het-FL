# from data_aug_models import Linreg_Torch_aug, Linreg_Sklearn, DTReg, MLP_Keras
from generate_data import get_data, get_shared_data
from generate_graph import build_edges, build_graph, build_graph_pooled
from pytorch_models import Linreg_Torch, MLP_Torch
from train import train
from utils import plot_preds_similarity,  plot_weight_dist

def run_exp(config, models, models_pooled, verbose=False, plot_preds=True, plot_w_dist=False):

    """

    Build graph and run training loop.

    Args:
    : config        : dict, experiment params (keys: 'n_clusters', 'n_ds', 'n_samples', 'n_features', 'ds_train', 'ds_val',
                                                    'cluster_labels', 'true_weights', 'n_iters', 'regularizer_term', 'ds_shared_mode')
    : models        : list of len(n_nodes), pytorch, sklearn or keras local models 
    : models_pooled : list of len(n_clusters), pytorch, sklearn or keras 'oracle' or 'pooled' models 
    : verbose       : boolean, flag to control printing statements during training
    : plot_preds    : boolean, flag to control plotting of MSE matrices 
    : plot_w_dist   : boolean, flag to control plotting of weight vectors sq. L2 difference matrix

    Output:
    : preds_list     : list of arrays (n_nodes, m_shared), predictions on the shared dataset on iteration 1, n_iters/2, n_iters
    : mse_train      : array (n_nodes,), local training MSE error for each node
    : mse_val        : array (n_nodes,), local validation MSE error for each node
    : mse_val_pooled : array (n_nodes,), local validation MSE error for each node incurred by corresponding "pooled/ oracle" model

    """
    
    # Get data and training params
    ds_train, ds_val, cluster_labels, true_weights = config['ds_train'], config['ds_val'], config['cluster_labels'], config['true_weights']
    n_clusters, n_iters, regularizer_term = config['n_clusters'], config['n_iters'], config['regularizer_term']
    ds_shared_mode = config['ds_shared_mode']

    G        = build_graph(ds_train, ds_val, models, cluster_labels) # Build a graph with nodes as dict containing local train/ val ds, model and cluster assignment
    G_pooled = build_graph_pooled(ds_train, ds_val, models_pooled, n_clusters=n_clusters) # Build a graph where local ds from one cluster are pooled
    A        = build_edges(G, cluster_labels)  # Build edges

    preds_list, mse_train, mse_val, mse_val_pooled = train(G, A, G_pooled, n_iters=n_iters, regularizer_term=regularizer_term, ds_shared_mode=ds_shared_mode, verbose=verbose)

    # Plotting
    if plot_preds:
        plot_preds_similarity(A, preds_list, n_clusters, n_iters)

    if plot_w_dist:
        plot_weight_dist(G, true_weights)

    return preds_list, mse_train, mse_val, mse_val_pooled


# Experiment's configuration
n_clusters, n_ds, n_samples, n_features = 3, 5, 100, 10

# Create a list of n_clusters*n_ds local datasets. 
ds_train, ds_val, cluster_labels, true_weights = get_data(n_clusters, n_ds, n_samples, n_features)

# Training params
N_ITERS = 200
REG     = 0.01

config = {
        'n_clusters':       n_clusters,
        'n_ds':             n_ds,
        'n_samples':        n_samples,
        'n_features':       n_features,
        'ds_train':         ds_train, 
        'ds_val':           ds_val, 
        'cluster_labels':   cluster_labels,
        'true_weights':     true_weights, 
        'n_iters':          N_ITERS,
        'regularizer_term': REG,
        'ds_shared_mode': 'const'
        }

#========================Pytorch Linear - no bias========================#

# models = [Linreg_Torch(n_features, bias=False) for i in range(n_clusters*n_ds)]
# models_pooled = [Linreg_Torch(n_features, bias=False) for i in range(n_clusters)]

# preds_list, mse_train, mse_val, mse_val_pooled = run_exp(config, models, models_pooled, verbose=False, plot_preds=True, plot_w_dist=True)

#========================Pytorch Linear - with bias========================#

models = [Linreg_Torch(n_features) for i in range(n_clusters*n_ds)]
models_pooled = [Linreg_Torch(n_features) for i in range(n_clusters)]

preds_list, mse_train, mse_val, mse_val_pooled = run_exp(config, models, models_pooled, verbose=False, plot_preds=False)

#========================Pytorch MLP========================#

# models = [MLP_Torch(n_features) for i in range(n_clusters*n_ds)]
# models_pooled = [MLP_Torch(n_features) for i in range(n_clusters)]

# preds_list, mse_train, mse_val, mse_val_pooled = run_exp(config, models, models_pooled, verbose=False, plot_preds=True)

# #========================Pytorch Linear - with data augmentation========================#

# models = [Linreg_Torch_aug(n_features) for i in range(n_clusters*n_ds)]
# models_pooled = [Linreg_Torch_aug(n_features) for i in range(n_clusters)]

# preds_list, mse_train, mse_val, mse_val_pooled = run_exp(config, models, models_pooled, verbose=False, plot_preds=True)

# #========================Sklearn LinReg data augmentation========================#

# # Create models
# models = [Linreg_Sklearn() for i in range(n_clusters*n_ds)]
# models_pooled = [Linreg_Sklearn() for i in range(n_clusters)]

# preds_list, mse_train, mse_val, mse_val_pooled = run_exp(config, models, models_pooled, verbose=False, plot_preds=True)

# #========================Sklearn DT data augmentation========================#

# # Create models
# models = [DTReg(max_depth=5) for i in range(n_clusters*n_ds)]
# models_pooled = [DTReg(max_depth=5) for i in range(n_clusters)]

# preds_list, mse_train, mse_val, mse_val_pooled = run_exp(config, models, models_pooled, verbose=False, plot_preds=True)


