import numpy as np
from sklearn.utils import resample

def get_data(n_clusters, n_ds, n_samples, n_features, noise_scale=1.0):
    
    """
    
    Function to create noisy Gaussian regression datasets multivar Gaussian ~N(0,I). 
    Datasets within the cluster share the same true weight vector. 
    For each node we create training (size = n_samples) and validation (size = 100) ds.
    
    Args:
    :n_clusters : number of clusters
    :n_ds       : number of local datasets per cluster 
    :n_samples  : number of samples in a local dataset
    :n_features : number of features of a datapoint
    :noise_scale: scale of normal distribution used to generate data noise
    
    Out:
    :ds_train:       list of (n_clusters*n_ds) tuples, local train datasets  of sample size n_samples
    :ds_val:         list of (n_clusters*n_ds) tuples, local validation datasets of sample size 100
    :cluster_labels: list of (n_clusters*n_ds) cluster assignments for each local dataset 
    :true_weights:   array of shape (n_clusters, n_features), true weight vector for each cluster

    """
    
    # Lists to store and return outputs
    ds_train       = []
    ds_val         = []
    cluster_labels = []
    true_weights   = np.zeros((n_clusters, n_features))
    
    for i in range(n_clusters):
        
        # Sample true weight vector for cluster i
        w = np.random.normal(0, 1, size=(n_features,1))
        true_weights[i] = w.reshape(-1,)

        for j in range(n_ds):
            # Sample datapoints from multivar Gaussian ~N(0,I)
            X = np.random.normal(0, 1.0, size=(1000, n_features))
            
            # Sample noise 
            noise = np.random.normal(loc=0.0, scale=noise_scale, size=(1000, 1))
            
            # Noisy Gaussian regression
            y = X@w + noise
            
            X_train, y_train = resample(X, y, n_samples=n_samples)
            X_val, y_val = resample(X, y, n_samples=100)

            ds_train.append( (X_train, y_train) )
            ds_val.append( (X_val, y_val) )
            
            cluster_labels.append(i)

    return ds_train, ds_val, cluster_labels, true_weights

def get_shared_data(k, m_shared, n_features, var=1):

    """

    Create noisy Gaussian regression dataset.

    :param m_shared   : int : n.o. samples in a shared dataset(s)
    :param n_features : int : n.o. features of a datapoint
    :param k          : int : n.o. datasets (m_shared, n_features) to sample 

    :out shared_ds    : array of shape (k, m_shared, n_features), shared dataset(s)

    """

    ds_shared = np.random.normal(0, var, size=(k, m_shared, n_features))
    
    return ds_shared