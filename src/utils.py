import numpy as np
import torch 

def fwd_check(ds, model):
    # Check manually fwd step [only linear model!]
    X, y = ds[0], ds[1]
    
    w = model.model.weight.data.numpy()
    b = model.model.bias.data.numpy()
    
    out_manual = w@X.T + b
    out = model.predict(X)
    
    np.testing.assert_allclose(out.reshape(-1,), out_manual.reshape(-1,), rtol=1e-05)
    
    
def bckwd_check(ds, ds_test, pred_test, A, model):
    # Check manually bckw step [only linear model!]
    
    # Data
    X, y = ds[0], ds[1]
    X_test, y_test = ds_test[0], ds_test[1]
    
    # Convert to torch tensors
    X, y = torch.FloatTensor(X), torch.FloatTensor(y)
    X_test = torch.FloatTensor(X_test)
    
    # Sample size
    m = X.shape[0]
    m_test = X_test.shape[0]
    
    lmbd = 0.01
    
    # Convert np arrays to pytorch floats
    A = torch.from_numpy(A).float().reshape(-1,1)
    pred_test = torch.from_numpy(pred_test).float()

    # Get predictions for local and shared test ds
    pred = model(X)
    pred_test_local = model(X_test)
    
    # Compute gradients manually
    dw_local = -2/m*X.T@(y - pred)
    dw_test = lmbd/m_test*(X_test.T@((pred_test_local - pred_test)@A))  # (n,m')@((m',nn)@(nn,1))-->(n,1)
    dw = (dw_local + dw_test).detach().numpy().reshape(-1,1)
    
    # Compute gradients with pytorh
    model.zero_grad()  
    criterion = torch.nn.MSELoss(reduction='mean')
    # Compute loss
    loss_local = criterion(y, pred)
    loss_GTV = torch.mean( ((pred_test_local - pred_test)**2)@A )
    
    loss = loss_local + (lmbd/2)*loss_GTV
    # bckwd pass
    loss.backward()
    # gradients
    dw_torch = model.model.weight.grad.detach().numpy().reshape(-1,1)
    np.testing.assert_allclose(dw_torch, dw, rtol=1e-05)