import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def simulate_simpson(n_ds=3, n_samples=100, r=0.6, difference=3):
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

    return ds_train

class Linreg_Torch(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Linear(1, 1)

        
    def forward(self, x):
        if type(x) != torch.Tensor:
            x = torch.FloatTensor(x) 
        pred = self.model(x)
        return pred

def train(X, y, model, lr=0.01):
        
        X, y = torch.FloatTensor(X), torch.FloatTensor(y)

        # Define Loss and Optimizer
        criterion = torch.nn.MSELoss(reduction='mean')
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        pred = model(X)
        
        model.zero_grad()  
        loss = criterion(y, pred)
        loss.backward()
        # print("Loss: ", loss.item())
        optimizer.step()

ds_train = simulate_simpson()

models = [Linreg_Torch(), Linreg_Torch(), Linreg_Torch()]
scalers = [StandardScaler(), StandardScaler(), StandardScaler()]

ds_scaled = []
for data, scaler in zip(ds_train, scalers):
    x_scaled = scaler.fit_transform(data[0])
    ds_scaled.append((x_scaled, data[1]))
    print(scaler.get_params())

epochs = 1000
lrs = [0.01, 0.01, 0.01]

for i in range(epochs):
     print("Epoch: ", i)
     for data, model, lr in zip(ds_scaled, models, lrs):
        train(data[0], data[1].reshape(-1,1), model, lr)

fig, ax = plt.subplots(1, 1, figsize=(10,8))

for data, scaler, model in zip(ds_train, scalers, models):

    x, y = data[0], data[1]
    x_min, x_max = min(x), max(x)
    x_plot = np.linspace(x_min, x_max, num=100)

    x_plot_scaled = scaler.transform(x_plot)
    y_plot        = model(x_plot_scaled).detach().numpy()

    ax.scatter(x, y)
    ax.plot(x_plot, y_plot)

plt.show()           