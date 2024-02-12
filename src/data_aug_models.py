
import math
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from tensorflow import keras
from tensorflow.keras import layers
import torch

class Data_augmentation():
    
    @staticmethod
    def data_aug(ds_train, ds_shared, pred_shared, A, regularizer_term):
        
        """
        nn - number of nodes
        m' - sample size of shared test data
        
        Args:
        : ds_train         :         list of (n_clusters*n_ds) local train datasets of sample size n_samples
        : ds_shared        :         shared dataset, X (m_shared, n_features)
        : pred_shared      :      predictions for ds_shared of all nodes, array of size (n_nodes, m_shared)
        : A                :                row of a symmetric matrix A (weights of edges), array of size (nn,); A_ii=0 (zero diagonal)
        : regularizer_term : lambda, float number

        :out X_aug: stacked features X (local train ds), X_shared (stacked n.o. nodes times)
        :out y_aug: stacked labels y (local train ds), nodes_preds (predictions for X_shared at each node)
        :out sample_weight: sample_weight - 1 for (X, y) and regularizer_term/2*A for other samples.
        
        """
        
        X, y = ds_train[0], ds_train[1].reshape(-1,)
        
        # Construct augmented dataset
        nn = pred_shared.shape[0]
        
        X_shared_repeat = np.tile(ds_shared, (nn,1))
        X_aug = np.concatenate((X, X_shared_repeat), axis=0)
        y_aug = np.concatenate((y, pred_shared.reshape(-1,)), axis=0)
        
        # Format edges' weights A and compute sample weight
        m = y.shape[0] 
        m_shared = ds_shared.shape[0]    
    
        A_repeat = np.repeat(A, m_shared, axis=0)
        sample_weight = np.concatenate((np.ones((m,)), np.ones((nn*m_shared,))*(regularizer_term/2)*A_repeat))   

        return X_aug, y_aug.reshape(-1,1), sample_weight

#-----------------------PYTORCH MODELS-------------------------------#

class Optimize_aug(torch.nn.Module, Data_augmentation):
    def __init__(self):
        torch.nn.Module.__init__(self)
        
    # Model prediction with tracking gradients
    def forward(self, x):
        if type(x) != torch.Tensor:
            x = torch.FloatTensor(x)
            
        pred = self.model(x)
        return pred
    
    # Model prediction without tracking gradients
    def predict(self, x):
        if type(x) != torch.Tensor:
            x = torch.FloatTensor(x)
            
        with torch.no_grad():
            pred = self.model(x)
        return pred.detach().numpy() 

    def update(self, ds_train, ds_shared, nodes_preds, A, regularizer_term):
        
        # Create augmented dataset
        X_aug, y_aug, sample_weight = self.data_aug(ds_train, ds_shared, nodes_preds, A, regularizer_term)
        
        # Convert numpy arrays to torch tensors
        X_aug, y_aug = torch.FloatTensor(X_aug), torch.FloatTensor(y_aug)
        sample_weight =  torch.FloatTensor(sample_weight)
        
        # Set all gradient values to zeros
        self.model.zero_grad() 
        
        # Divide ds into smaller batches and do gradient accumulation
        bs = 100
        iters = math.ceil(len(X_aug)/bs)
        for i in range(iters):
                X_b, y_b = X_aug[i*bs:i*bs+bs], y_aug[i*bs:i*bs+bs]
                sw_b = sample_weight[i*bs:i*bs+bs]
                pred = self.model(X_b)
                loss = sw_b@((pred - y_b)**2) / bs
                loss.backward()

        # Update parameters of the model using the chosen optimizer
        self.optimizer.step()
        return loss.item()
    
    def update_pooled(self, ds_train):
        
        X, y = ds_train[0], ds_train[1]
        X, y = torch.FloatTensor(X), torch.FloatTensor(y)

        # Get predictions for local and shared test ds
        pred = self.model(X)
        
        # Set all gradient values to zeros
        self.model.zero_grad()  
        
        # Compute loss
        loss = self.criterion(y, pred)
        
        # Backpropagate the gradients
        loss.backward()

        # Update parameters of the model using the chosen optimizer
        self.optimizer.step()

        return loss.item()

class Linreg_Torch_aug(Optimize_aug):
    def __init__(self, n_features, lr=0.001):
        Optimize_aug.__init__(self)
        
        # Define model
        self.model = torch.nn.Linear(n_features, 1)
        
        # Define Loss and Optimizer
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=lr)


#-----------------------SKLEARN MODELS-------------------------------#

class Linreg_Sklearn(LinearRegression, Data_augmentation):
    def __init__(self, **kwargs):
        LinearRegression.__init__(self, **kwargs)

    def update(self, ds_train, ds_shared, nodes_preds, A, regularizer_term):
        
        # Get augmented dataset
        X_aug, y_aug, sample_weight = self.data_aug(ds_train, ds_shared, nodes_preds, A, regularizer_term)
        # Fit data
        self.fit(X_aug, y_aug, sample_weight=sample_weight)
        
        # Compute MSE
        preds = self.predict(X_aug)
        loss = np.mean((y_aug - preds.reshape(-1,1))**2)

        return loss
    
    def update_pooled(self, ds_train):
    
        X, y = ds_train[0], ds_train[1]

        # Fit data
        self.fit(X, y)
        
        # Compute MSE
        preds = self.predict(X)
        loss = np.mean((y - preds.reshape(-1,1))**2)

        return loss



class DTReg(DecisionTreeRegressor, Data_augmentation):
    def __init__(self, **kwargs):
        DecisionTreeRegressor.__init__(self, **kwargs)

    def update(self, ds_local, ds_test,  pred_test, A, regularizer_term):
         
        X_aug, y_aug, sample_weight = self.data_aug(ds_local, ds_test,  pred_test, A, regularizer_term)
   
        self.fit(X_aug, y_aug.reshape(-1,), sample_weight=sample_weight)
        preds = self.predict(X_aug)
        loss = np.mean((y_aug - preds.reshape(-1,1))**2)

        return loss
    
    def update_pooled(self, ds_train):
        X, y = ds_train[0], ds_train[1]
        self.fit(X, y)
        preds = self.predict(X)
        loss = np.mean((y - preds.reshape(-1,1))**2)
        
        return loss

#-----------------------KERAS MODELS-------------------------------#

class MLP_Keras(Data_augmentation):
    def __init__(self, n_features):
        
        # Define model
        self.model = model = keras.Sequential(
            [
                layers.Dense(10, activation="relu", input_shape=(n_features,)),
                layers.Dense(1)
            ]
        )
        
        # Define Loss and Optimizer
        optimizer = keras.optimizers.RMSprop(learning_rate=0.01)
        self.model.compile(loss="mean_squared_error", optimizer=optimizer)
        
    # Model prediction 
    def predict(self, x):
        return self.model(x).numpy()

    def update(self, ds_local, ds_test,  pred_test, A, regularizer_term):
        
        # Get augmented dataset
        X_aug, y_aug, sample_weight = self.data_aug(ds_local, ds_test,  pred_test, A, regularizer_term)
        # Run a single gradient update on a single batch of data
        self.model.train_on_batch(X_aug, y_aug, sample_weight=sample_weight)
        
        # Compute MSE
        preds = self.predict(X_aug) 
        loss = np.mean((y_aug - preds.reshape(-1,1))**2)

        return loss

    def update_pooled(self, ds_train):
        X, y = ds_train[0], ds_train[1]
        self.model.train_on_batch(X, y)
        preds = self.predict(X)
        loss = np.mean((y - preds.reshape(-1,1))**2)
        
        return loss