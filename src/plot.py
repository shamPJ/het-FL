import numpy as np
import os
from utils import load_and_plot_mse, load_and_plot_est_error

p_dir = os.getcwd() + "/out/Linreg_Torch_594391"
# load_and_plot_mse(p_dir)
# load_and_plot_mse(p_dir, scaled=True)
# load_and_plot_est_error(p_dir)
est_error_scaled = np.load(p_dir + '/reg_term_10.0/stats/est_error_mean_std_scaled.npy')
est_error = np.load(p_dir + '/reg_term_10.0/stats/est_error_mean_std.npy')

print("est_error", est_error[0,:,-5:])
print("est_error_scaled", est_error_scaled[0,:,-5:])

# shapes (2000, 3, 10), (2000, 150, 10), (3, 10)
est_weights_pooled = np.load(p_dir + '/reg_term_10.0/reg_term_10.0_n_samples_20/est_weights_pooled.npy')
est_weights = np.load(p_dir + '/reg_term_10.0/reg_term_10.0_n_samples_20/est_weights.npy')
true_weights = np.load(p_dir + '/reg_term_10.0/reg_term_10.0_n_samples_20/true_weights.npy')
# print(est_weights_pooled[-1,0,:])
# print(est_weights[-1,:10,:])
# print(true_weights[0,:])
print("sample size 20")
print("error est pooled")
eep = np.mean((est_weights_pooled[-1,0,:] - true_weights[0].reshape(1,1,10))**2)  
ee = np.mean((est_weights[-1,:50,:] - true_weights[0].reshape(1,1,10))**2)
print(eep)

print("error est")
print(ee)
print("ratio", ee/eep)

est_weights_pooled = np.load(p_dir + '/reg_term_10.0/reg_term_10.0_n_samples_5/est_weights_pooled.npy')
est_weights = np.load(p_dir + '/reg_term_10.0/reg_term_10.0_n_samples_5/est_weights.npy')
true_weights = np.load(p_dir + '/reg_term_10.0/reg_term_10.0_n_samples_5/true_weights.npy')
# print(est_weights_pooled[-1,0,:])
# print(est_weights[-1,:10,:])
# print(true_weights[0,:])

print("sample size 5")
print("error est pooled")
eep = np.mean((est_weights_pooled[-1,0,:] - true_weights[0].reshape(1,1,10))**2)  
ee = np.mean((est_weights[-1,:50,:] - true_weights[0].reshape(1,1,10))**2)
print(eep)

print("error est")
print(ee)
print("ratio", ee/eep)
