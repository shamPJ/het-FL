import numpy as np

#========================HELPER FUNCS========================#

def kl_divergence(dist1, dist2):

    """

    Function to compute the KL divergence between two multivariate normal distributions.
    See https://en.wikipedia.org/wiki/Kullback–Leibler_divergence#Multivariate_normal_distributions
    
    :param dist1: 2D numpy array, samples drawn from multivariate normal distributions
    :param dist2: 2D numpy array, samples drawn from multivariate normal distributions

    :out: computed KL divergence

    """

    mean1, mean2 = dist1.mean(axis=0), dist2.mean(axis=0)
    cov1, cov2   = np.cov(dist1, rowvar=False), np.cov(dist2, rowvar=False)

    n = dist1.shape[1]

    term_det   = np.log(np.linalg.det(cov2) / np.linalg.det(cov1)) 
    term_trace = np.trace(np.linalg.inv(cov2) @ cov1)
    term_quadr = (mean2 - mean1).T @ np.linalg.inv(cov2) @ (mean2 - mean1)

    return 0.5 * (term_det + term_trace + term_quadr - n)

def concat(ds):

    """

    Function to pool data for G_pooled.

    :param ds: list of tuples, datasets to concatenate

    :out: concatenated feature matrices and label vectors of local datasets belonging to the same cluster

    """
    Xs = []
    ys = []
    for X, y in ds:
        Xs.append(X)
        ys.append(y)
    return np.concatenate(Xs), np.concatenate(ys)  

#========================MAIN FUNCS========================#

def build_graph(ds_train, ds_val, models, cluster_labels):

    """

    Function to generate graph.

    :param ds_train:       list of (n_clusters*n_ds) local train datasets of sample size n_samples (see function `get_data()`)
    :param ds_val:         list of (n_clusters*n_ds) local validation datasets of sample size 100
    :param models:         list of (n_clusters*n_ds) local models
    :param cluster_labels: list of (n_clusters*n_ds) cluster assignments for each local dataset 

    :out G:                list of (n_clusters*n_ds) python dictionaries (graph nodes) where each dict (node)  contain local train/ val datasets, model and shared dataset

    """
    G = []
    for ds_t, ds_v, model, c in zip(ds_train, ds_val, models, cluster_labels):
        G.append({ "model": model, "ds_train": ds_t, "ds_val": ds_v, "cluster_label": c}) 
        
    return G

def build_edges(G, cluster_labels, p_in = 0.8, p_out = 0.2):

    """

    Create network edges (links) with Bernoulli distribution P(n) = p^n(1-p)^(1-n), where n={0,1}.
    Probability p_in for nodes within a cluster and probability p_out for nodes from different clusters.
    Edge weights are the same for all edges A_ij = 1; for A_ii = 0

    :param G:              list of (n_clusters*n_ds) python dictionaries (graph nodes) where each dict (node)  contain local train/ val datasets, model and shared dataset
    :param cluster_labels: list of (n_clusters*n_ds) cluster assignments for each local dataset 
    :param p_in:           probability of a link between nodes belonging to the same cluster 
    :param p_out:          probability of a link between nodes belonging to different clusters 

    :out A:                numpy array of shape (n_nodes, n_nodes), adjacency matrix of the graph, created with Bernoulli distribution and probabilities p_in and p_out.

    """
    
    nn = len(G)
    A = np.zeros((nn, nn))
    rows, cols = A.shape

    for i in range(rows):
        for j in range(cols):
            if i==j:
                A[i,j] = 0
            elif cluster_labels[i] == cluster_labels[j]:
                A[i,j] = np.random.binomial(n=1, p=p_in, size=5)[0] 
            else:
                A[i,j] = np.random.binomial(n=1, p=p_out, size=5)[0]

    # Create a symmetric matrix           
    A = np.tril(A) 
    A = A + A.T
    return A

def build_edges_KL(G, node_degree = 5):

    """

    Create network edges (links) by use of KL divergence as a measure of distance between label vectors of the local datasets. 
    For each node i we choose `node_degree` most similar dataset and assign A[i,j] = 1 for those nodes. Thus, parameter `node_degree` is the degree of the node i.


    :param G:              list of (n_clusters*n_ds) python dictionaries (graph nodes) where each dict (node)  contain local train/ val datasets, model and shared dataset
    :param node_degree:    integer, node degree
    
    :out A:                numpy array of shape (n_nodes, n_nodes), adjacency matrix of the graph, created according to KL distance (see formula below)

    """

    n_nodes = len(G)
    A = np.zeros((n_nodes, n_nodes))
    rows, cols = A.shape

    for i in range(rows):
        for j in range(cols):
            if i==j:
                A[i,j] = 0
            else:
                # Get feature matrix for nodes i,j
                y1, y2 = G[i]["ds_train"][1], G[j]["ds_train"][1]

                # Compute dist with KL divergence
                kl_dist = np.exp( -(kl_divergence(y1, y2) + kl_divergence(y2, y1)) )
                A[i,j] = kl_dist
    
    # For each row (node) select `node_degree` smallest kl_dist values and assign A[i,j] = 1, for the rest A[i,j] = 0
    for i in range(n_nodes):
        ind = np.argsort(A[i])
        A[i, ind[:-node_degree]] = 0
        A[i, ind[-node_degree:]] = 1
        A[i, i] = 0

    for i in range(n_nodes):
        
        ind = np.argsort(A[i])                           # get indices of sorted kl_dist values for the row (node) i
        ind_del = np.delete(ind, np.where(ind==i)[0][0]) # remove i from indices list (as we set A[i,i]=0)
        
        A[i, ind_del[-node_degree:]] = 1                 # set A[i,j] = 1 for `node_degree` smallest kl_dist
        A[i, ind_del[:-node_degree]] = 0                 # set A[i,j] = 0 for the rest nodes
        A[i, i] = 0                                      # set A[i,i] = 0 

    return A

def build_graph_pooled(ds_train, ds_val, models, n_clusters=3):
    
    """

    Graph with n_clusters nodes. 
    For each node (corresponding to one cluster), local training dataset is pooled train ds of all nodes belonging to that cluster. 
    
    :param ds_train:
    :param ds_val:
    :param model:
    

    """
    
    n_ds = int(len(ds_train) / n_clusters)

    G = []
    for i in range(n_clusters):
        G.append({ 
                "model": models[i], 
                "ds_train": concat(ds_train[i*n_ds: i*n_ds+n_ds]), 
                "ds_val": concat(ds_val[i*n_ds: i*n_ds+n_ds])
                 }) 
        
    return G    