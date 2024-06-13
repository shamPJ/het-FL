import numpy as np
import os
from utils import load_and_plot_mse, load_and_plot_est_error

p_dir = os.getcwd() + "/out/Linreg_Torch_570915"
# load_and_plot_mse(p_dir)
# load_and_plot_mse(p_dir, scaled=True)
# load_and_plot_est_error(p_dir)
# est_error = np.load(p_dir + '/reg_term_0.0/reg_term_0.0_n_samples_20/est_weights.npy')
# est_error_scaled = np.load(p_dir + '/reg_term_0.0/reg_term_0.0_n_samples_20/est_weights_pooled.npy')

# print((est_error[:,0,-1]/est_error_scaled[:,0,-1]))
# print(np.sum(est_error[:,0,-1]/est_error_scaled[:,0,-1])**2)
