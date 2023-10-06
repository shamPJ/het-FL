import numpy as np
from sklearn.utils import resample

def get_data(n_clusters, n_ds, n_samples, n_features, noise_scale=1.0):
    
    """
    
    Function to create noisy Gaussian regression datasets. 
    Datasets within the cluster share the same true weight vector. 
    For each node we create training (size = n_samples) and validation (size = 100) ds.
    
    :param n_clusters:   number of clusters
    :param n_ds:         number of local datasets per cluster 
    :param n_samples:    number of samples in a local dataset
    :param n_features:   number of features of a datapoint
    :param noise_scale:  scale of normal distribution used to generate data noise
    
    :out ds_train:       list of (n_clusters*n_ds) tuples, local train datasets  of sample size n_samples
    :out ds_val:         list of (n_clusters*n_ds) tuples, local validation datasets of sample size 100
    :out cluster_labels: list of (n_clusters*n_ds) cluster assignments for each local dataset 
    :out true_weights:   array of shape (n_clusters, n_features), true weight vector for each cluster

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

    :param m_shared   : int, =m', number of samples in a shared dataset(s)
    :param n_features : int, number of features of a datapoint
    :param k          : int, number of different X's sampled from normal distr

    :out shared_ds    : array of shape (k, m_shared, n_features), shared dataset(s)

    """

    ds_shared = np.zeros((k, m_shared, n_features))
    for i in range(k): 
        X = np.random.normal(0, var, size=(m_shared, n_features))
        ds_shared[i, :, :] = X
    
    return ds_shared