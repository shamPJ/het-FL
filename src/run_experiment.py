from data_aug_models import Linreg_Torch_aug, Linreg_Sklearn, DTReg, MLP_Keras
from generate_data import get_data, get_shared_data
from generate_graph import build_edges, build_graph, build_graph_pooled
from pytorch_models import Linreg_Torch, MLP_Torch
from train import train
from utils import plot_preds_similarity,  plot_weight_dist

def run_exp(config, models, models_pooled, verbose=False, plot_preds=True, plot_w_dist=False):

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
    ds_train, ds_val, cluster_labels, true_weights = config['ds_train'], config['ds_val'], config['cluster_labels'], config['true_weights']
    n_clusters, n_iters, regularizer_term = config['n_clusters'], config['n_iters'], config['regularizer_term']
    
    G        = build_graph(ds_train, ds_val, models, cluster_labels) # Build a graph with nodes {model, dataset local, dataset test}
    G_pooled = build_graph_pooled(ds_train, ds_val, models_pooled, n_clusters=n_clusters) # Build a graph where local ds from one cluster are pooled
    A        = build_edges(G, cluster_labels)  # Build edges

    preds_list, mse_train, mse_val, mse_val_pooled = train(G, A, G_pooled, n_iters=n_iters, regularizer_term=regularizer_term, verbose=verbose)

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
        'regularizer_term': REG
        }

#========================Pytorch Linear - no bias========================#

# models = [Linreg_Torch(n_features, bias=False) for i in range(n_clusters*n_ds)]
# models_pooled = [Linreg_Torch(n_features, bias=False) for i in range(n_clusters)]

# preds_list, mse_train, mse_val, mse_val_pooled = run_exp(config, models, models_pooled, verbose=False, plot_preds=True, plot_w_dist=True)

#========================Pytorch Linear - with bias========================#

models = [Linreg_Torch(n_features) for i in range(n_clusters*n_ds)]
models_pooled = [Linreg_Torch(n_features) for i in range(n_clusters)]

preds_list, mse_train, mse_val, mse_val_pooled = run_exp(config, models, models_pooled, verbose=False, plot_preds=True)

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


