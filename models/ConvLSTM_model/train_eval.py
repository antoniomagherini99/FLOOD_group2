import torch
import torch.nn as nn
import numpy as np

def train_epoch_conv_lstm(model, loader, optimizer, device):
    model.to(device)
    model.train() # specifies that the model is in training mode

    losses = []

    for batch in loader:
        x = batch[0].to(device)
        y = batch[1].to(device)

        # Model prediction
        preds, _ = model(x)
        # concat over the dimension 1 (time steps)
        list_preds = torch.cat(preds, dim=1)
        # MSE loss function
        loss = nn.MSELoss()(list_preds, y)
        
        losses.append(loss.cpu().detach())
        
        # Backpropagate and update weights
        loss.backward()   # compute the gradients using backpropagation
        optimizer.step()  # update the weights with the optimizer
        optimizer.zero_grad(set_to_none=True)   # reset the computed gradients

    losses = np.array(losses).mean()

    return losses

def evaluation_conv_lstm(model, loader, device):
    model.to(device)
    model.eval() # specifies that the model is in evaluation mode 

    losses = []

    for batch in loader:
        x = batch[0].to(device)
        y = batch[1].to(device)

        # Model prediction
        preds, _ = model(x)
        list_preds = torch.cat(preds, dim=1)

        # MSE loss function
        loss = nn.MSELoss()(list_preds, y)
        
        losses.append(loss.cpu().detach())
        
        # Backpropagate and update weights
        loss.backward()   # compute the gradients using backpropagation

    losses = np.array(losses).mean()

    return losses