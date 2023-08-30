
#from data_aug_models import Linreg_Torch_aug, Linreg_Sklearn, DTReg, MLP_Keras
from generate_data import get_data, get_shared_data
# import matplotlib.pyplot as plt
# import numpy as np
from pytorch_models import Linreg_Torch, MLP_Torch
from train import train, plot_preds_similarity, plot_weight_dist

n_clusters, n_ds, n_samples, n_features = 3, 5, 100, 10
ds_shared = get_shared_data(n_samples, n_features)

# Create a list of n_clusters*n_ds local datasets. 
ds_train, ds_val, cluster_labels, true_weights = get_data(n_clusters, n_ds, n_samples, n_features)

N_ITERS = 100
REG     = 0.01

config = {
        'n_clusters':       n_clusters,
        'n_ds':             n_ds,
        'n_samples':        n_samples,
        'n_features':       n_features,
        'ds_train':         ds_train, 
        'ds_val':           ds_val, 
        'ds_shared':        ds_shared, 
        'cluster_labels':   cluster_labels,
        'true_weights':     true_weights, 
        'n_iters':          N_ITERS,
        'regularizer_term': REG
        }

#========================Pytorch Linear - no bias========================#

# Create linear pytorch models
models = [Linreg_Torch(n_features, bias=False) for i in range(n_clusters*n_ds)]
models_pooled = [Linreg_Torch(n_features, bias=False) for i in range(n_clusters)]

preds_list, mse_train, mse_val, mse_val_pooled = train(config, models, models_pooled, verbose=False, plot_preds=True, plot_w_dist=True)

#========================Pytorch Linear - with bias========================#

# Create linear pytorch models
models = [Linreg_Torch(n_features) for i in range(n_clusters*n_ds)]
models_pooled = [Linreg_Torch(n_features) for i in range(n_clusters)]

preds_list, mse_train, mse_val, mse_val_pooled = train(config, models, models_pooled, verbose=False, plot_preds=True)


# #========================Pytorch MLP========================#

# # Create pytorch models
# models = [MLP_Torch(n_features) for i in range(n_clusters*n_ds)]
# models_pooled = [MLP_Torch(n_features) for i in range(n_clusters)]



# G = build_graph(ds_train, ds_val, ds_shared, models, cluster_labels)
# G_pooled = build_graph_pooled(ds_train, ds_val, models_pooled, n_clusters=n_clusters)
# A = build_edges(G, cluster_labels)

# preds_list, mse_train, mse_val, mse_val_pooled = train(G, A, G_pooled, verbose=False)

# # Plotting results
# plot_preds_similarity(A, preds_list)
# plot_weight_dist(G, true_weights)



