import os
from utils import load_and_plot_mse, load_and_plot_est_error

p_dir = os.getcwd() + "/out/Linreg_Torch_568783"
load_and_plot_mse(p_dir)
load_and_plot_mse(p_dir, scaled=True)
load_and_plot_est_error(p_dir)