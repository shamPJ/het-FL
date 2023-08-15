
# from data_aug_models import Linreg_Torch_aug, Linreg_Sklearn, DTReg, MLP_Keras
from generate_data import get_shared_data
import numpy as np
import matplotlib.pyplot as plt
from pytorch_models import Linreg_Torch, MLP_Torch
 
def simulate_simpson(n_ds=3, n_samples=100, r=0.6, difference=3):

    """

    Finction to create dataset for Simpson's paradox. Re-written in Python based on https://github.com/easystats/bayestestR/blob/HEAD/R/simulate_simpson.R

    :param n_ds:        number of local datasets 
    :param n_samples:   number of samples in a local dataset
    :param r:           correlation coefficient
    :param difference:  difference between local datasets
    
    :out ds_train:       list of tuples, local training datasets (x, y)  arrays of sample size n_samples
    :out ds_plot:        list of arrays, each arrays is used for plotting corresponding local model's predictions

    """
    
    r_sign   = r / abs(r)
    ds_train = []
    ds_plot  = []
    
    for i in range(n_ds):
        mu = np.array([0, 0])
        cov = np.array([[1, r], [r, 1]])
        local_ds = np.random.multivariate_normal(mu, cov, n_samples)
        
        x = local_ds[:, 0] + difference * i
        y = local_ds[:, 1] + difference * i * -r_sign
        
        x_min, x_max = min(x), max(x)
        x_plot = np.linspace(x_min, x_max, num=100)
            
        ds_train.append((x.reshape(-1,1), y.reshape(-1,1)))
        ds_plot.append(x_plot.reshape(-1,1))

    return ds_train, ds_plot

def build_simpson_graph(ds_train, ds_plot, ds_shared, models):

    """

    Function to generate graph. We assume that all ds are in one cluster.

    :param ds_train:       list of (n_clusters*n_ds) local train datasets of sample size n_samples (see function `get_data()`)
    :param ds_shared:      tuple (X, y), shared dataset (D^(test) in the paper)
    :param models:         list of (n_clusters*n_ds) local models

    :out G:                list of n_ds python dictionaries (graph nodes) where each dict (node)  contain local train datasets, model and shared dataset
    :out A:                numpy array of shape (n_ds, n_ds), adjacency matrix of the graph

    """
    G = []
    for ds_t, ds_plt, model in zip(ds_train, ds_plot, models):
        G.append({ "model": model, "ds_train": ds_t, "ds_plot": ds_plt, "ds_shared": ds_shared}) 

    A = np.ones((len(ds_train), len(ds_train)))  
    return G, A


def train(G, A, iters=1000, regularizer_term=0.01, verbose=False): 
    
    """
    
    :param G: list of dicts [dict keys are: model, ds_train, ds_val, ds_shared, cluster_label], represents graph with n_nodes.
    :iters: number of iterations or updates of the local model.
    :regularizer_term: scaling factor for GTV term.
    
    """

    n_nodes  = len(G)                                 # number of nodes in a graph
    m_shared = G[0]["ds_shared"][0].shape[0]          # sample size of the shared ds
    m_plot   = G[0]["ds_plot"].shape[0]               # sample size of the ds for plotting
    nodes_preds = np.zeros((m_shared, n_nodes))       # predictions on a shared ds
    preds_plot  = np.zeros((iters, n_nodes, m_plot))  # predictions on a dataset for plotting

    for i in range(iters):
        print(f"Iteration {i+1}")
        # Update local models
        for n in range(n_nodes):
            ds_train  = G[n]["ds_train"]
            x_plot    = G[n]["ds_plot"]
            ds_shared = G[n]["ds_shared"]
            model     = G[n]["model"]
            model.update(ds_train, ds_shared, nodes_preds, A[n], regularizer_term) 

        # Update predictions on a shared test set 
        nodes_preds = np.zeros((m_shared, n_nodes)) 
        for n in range(n_nodes):
            ds_train  = G[n]["ds_train"]
            x_plot    = G[n]["ds_plot"]
            ds_shared = G[n]["ds_shared"]
            model     = G[n]["model"]

            # model predictions
            pred_train          = model.predict(ds_train[0])
            preds_plot[i, n, :] = model.predict(x_plot).reshape(-1,)

            pred_shared = model.predict(ds_shared[0])
            nodes_preds[:, n] = pred_shared.reshape(-1,)

            if verbose:
                print(f"Node {n+1}, Training Loss {np.mean((ds_train[1] - pred_train)**2): .2f}")
        
    return preds_plot

# Create Simpson's datasets consisting of 3 local datasets with sample size 100
ds_train, ds_plot = simulate_simpson()
print(f'Shape of ds_train dataset feature matrix {ds_train[0][0].shape}')
print(f'Shape of ds_train dataset label vector {ds_train[0][1].shape}')
print(f'Shape of ds_plot dataset {ds_plot[0].shape}')

# Create shared dataset
ds_shared = get_shared_data(n_samples=100, n_features=1)
LR = 0.01
models = [Linreg_Torch(n_features=1, lr=0.01), Linreg_Torch(n_features=1, lr=0.05), Linreg_Torch(n_features=1, lr=0.025)]
print(models)
G, A = build_simpson_graph(ds_train, ds_plot, ds_shared, models)

preds_plot = train(G, A, iters=2000, regularizer_term=0.05, verbose=True)
print(preds_plot.shape)

fig, ax = plt.subplots(1, 1, figsize=(10,8))
for ds_pred in preds_plot[-10:]:
    for ds, x_plot, y_plot in zip(ds_train, ds_plot, ds_pred):
        ax.scatter(ds[0], ds[1])
        ax.plot(x_plot, y_plot)
plt.show()

# #---------------------------------------------------#
# import numpy as np
# from sklearn.linear_model import LinearRegression

# fig, ax = plt.subplots(1, 1, figsize=(10,8))
# for ds, x_plot in zip(ds_train, ds_plot):
#     X = ds[0]
#     y = ds[1].reshape(-1,)

#     reg = LinearRegression().fit(X, y)
#     y_plot = reg.predict(x_plot)

#     print("params: ", reg.coef_, reg.intercept_)

#     ax.scatter(X, y)
#     ax.plot(x_plot, y_plot)
# plt.show()

# # animation function.  This is called sequentially
# def animate(i): 
    
#     for j in range(len(history_camel)):
#         x_scatter = history_camel[j][:i,0]
#         y_scatter = history_camel[j][:i,1]
#         z_scatter = camel(x_scatter, y_scatter)

#         ax1.plot(x_scatter,y_scatter,z_scatter,colors[j],label=labels[j])  
  
#     return ax1


# # call the animator.  blit=True means only re-draw the parts that have changed.
# anim = animation.FuncAnimation(fig, animate, frames=500, interval=10)

# anim.save('camel3D.mp4', fps=15)
# plt.show()