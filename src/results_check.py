import numpy as np
import json
import os
from utils import load_and_plot_mse, load_and_plot_est_error

p_dir = os.getcwd() + "/out/Linreg_Torch_850816/reg_term_50.0/"

# load computed esr.errors
est_error_scaled = np.load(p_dir + 'stats/est_error_mean_std_scaled.npy')
est_error = np.load(p_dir + 'stats/est_error_mean_std.npy') # shape (2, 5, 2000)

print("est_error", est_error[0,:,-1])
print("est_error_scaled", est_error_scaled[0,:,-1])

# sdir = p_dir + 'reg_term_0.0_n_samples_20/'
# # load config
# with open(sdir + 'config.json') as file:
#     config = json.load(file)
# reg_term, n_clusters, n_ds, n_features, noise_val, n_samples = config['reg_term'], config['n_clusters'], config['n_ds'], config['n_features'], config['noise_val'], config['n_samples']
# p_in, p_out = config['p_in'], config['p_out']
# print("Config reg_term, n_clusters, n_ds, n_features, noise_val, n_samples, p_in, p_out/n", reg_term, n_clusters, n_ds, n_features, noise_val, n_samples, p_in, p_out)
# V = (noise_val*n_features) / (n_samples-n_features-1)
# V_pooled = (noise_val*n_features) / (n_samples*n_ds-n_features-1)
# print("Variance noise*d/(m-d-1) ", V)
# print("Variance pooled noise*d/(m*n_ds-d-1) ", V_pooled)
# print("Ratio Var/Var_pooled", V/V_pooled)

# # load learnt params
# # saved shapes (iters=2000, clusters=3, d=10), (2000, 150, 10), (3, 10). 
# # get last iteration, nodes in the 1st cluster and corresponding true param vector
# # sliced arrays are of shapes (10,), (50, 10), (10,)
# est_weights_pooled = np.load(sdir + 'est_weights_pooled.npy')[-1] # (3, 10)
# est_weights = np.load(sdir + 'est_weights.npy')[-1] # (150,10)
# true_weights = np.load(sdir + 'true_weights.npy')

# eep = (1/3)*np.sum((est_weights_pooled - true_weights)**2)  

# true_w = np.repeat(true_weights, [50,50,50], axis=0)
# ee = (1/150)*np.sum((est_weights - true_w)**2)
# print("error est pooled ", eep)
# print("error est ", ee)
# print("ratio ee/eep", ee/eep)

# sdir = p_dir + 'reg_term_0.0_n_samples_5/'
# # load config
# with open(sdir + 'config.json') as file:
#     config = json.load(file)
# reg_term, n_clusters, n_ds, n_features, noise_val, n_samples = config['reg_term'], config['n_clusters'], config['n_ds'], config['n_features'], config['noise_val'], config['n_samples']
# p_in, p_out = config['p_in'], config['p_out']
# print("\n")
# print("Config reg_term, n_clusters, n_ds, n_features, noise_val, n_samples, p_in, p_out/n", reg_term, n_clusters, n_ds, n_features, noise_val, n_samples, p_in, p_out)
# V = (noise_val*n_features) / (n_samples-n_features-1)
# V_pooled = (noise_val*n_features) / (n_samples*n_ds-n_features-1)
# print("Variance noise*d/(m-d-1) ", V)
# print("Variance pooled noise*d/(m*n_ds-d-1) ", V_pooled)
# print("Ratio Var/Var_pooled", V/V_pooled)

# # load learnt params
# # saved shapes (2000, 3, 10), (2000, 150, 10), (3, 10)
# # get last iteration, nodes in the 1st cluster and corresponding true param vector
# # sliced arrays are of shapes (10,), (50, 10), (10,)
# est_weights_pooled = np.load(sdir + 'est_weights_pooled.npy')[-1,0,:]
# est_weights = np.load(sdir + 'est_weights.npy')[-1,:50,:]
# true_weights = np.load(sdir + 'true_weights.npy')[0,:]

# eep = np.sum((est_weights_pooled.reshape(1,10) - true_weights.reshape(1,10))**2)  
# ee = (1/50)*np.sum((est_weights - true_weights.reshape(1,10))**2)
# print("error est pooled ", eep)
# print("error est ", ee)
# print("ratio ee/eep", ee/eep)