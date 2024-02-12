import numpy as np
import torch

class Optimize(torch.nn.Module):
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
        
        """

        nn - number of nodes
        m' - sample size of shared test data
        
        Args:
        : ds_train         : list of (n_clusters*n_ds) local train ds of sample size n_samples
        : ds_shared        : array of shape (m_shared, n_features),  shared test dataset; 
        : nodes_preds      : array of size (nn, m_shared), predictions for ds_shared of all nodes
        : A                : array of size (nn,), row of a symmetric adjacency matrix A; A_ii=0 (zero diagonal)
        : regularizer_term : float number, lambda, reg.param

        Output:
        : loss             : loss value
        
        """
        
        X, y = ds_train[0], ds_train[1]    # local ds, feature matrix X and label vector y of the node
        m_shared = ds_shared.shape[0]      # n.o. data pts in shared ds

        # Convert numpy arrays to torch tensors
        X, y = torch.Tensor(X), torch.Tensor(y)
        ds_shared = torch.Tensor(ds_shared)
        
        A = torch.Tensor(A).reshape(-1,1)   # reshape to (nn, 1)
        nodes_preds = torch.Tensor(nodes_preds)   # shape (nn, m_shared)

        # Get predictions for local and shared ds
        pred = self.model(X)
        pred_shared = self.model(ds_shared)  # out shape (m_shared,1)
        pred_shared = pred_shared.reshape(1, m_shared)
        
        # Set all gradient values to zeros
        self.model.zero_grad()  
        
        # Compute loss
        loss_local = self.criterion(y, pred)
        loss_per_node  = torch.mean((pred_shared - nodes_preds)**2, axis=1) # out shape (nn,)
        # loss_per_node shape is ([nn])
        loss_GTV = torch.mean(loss_per_node.reshape(-1,1)*A)
        loss = loss_local + (regularizer_term/2)*loss_GTV

        # Backpropagate the gradients
        loss.backward()

        # Update parameters of the model using the chosen optimizer
        self.optimizer.step()

        return loss.item()
    
    def update_pooled(self, ds_train):
        
        """

        Simple Gradient Step without penalty term (for pooled node and "oracle" model).

        """
        
        X, y = ds_train[0], ds_train[1]
        X, y = torch.Tensor(X), torch.Tensor(y)

        # Get predictions on local train ds
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

class Linreg_Torch(Optimize):
    def __init__(self, n_features, lr=0.001, bias=True):
        Optimize.__init__(self)
        
        # Define model
        if bias==False:
            self.model = torch.nn.Linear(n_features, 1, bias=False)
        else:
            self.model = torch.nn.Linear(n_features, 1)
            
        # Define Loss and Optimizer
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        
    def get_params(self):
        params = [layer.data for layer in self.model.parameters()][0].detach().numpy() 
        return params

class MLP_Torch(Optimize):
    def __init__(self, n_features, n_units = 15, lr=0.001):
        Optimize.__init__(self)
        
        # Define model
        self.model = torch.nn.Sequential(
                    torch.nn.Linear(n_features,  n_units),
                    torch.nn.ReLU(),
                    torch.nn.Linear(n_units, 1)
        )

        # Define Loss and Optimizer
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=lr)