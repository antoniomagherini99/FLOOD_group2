import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import TwoSlopeNorm
from matplotlib.colors import Normalize
import os
import imageio
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data.dataset import random_split
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
from torch.utils.data import DataLoader
from matplotlib.colors import TwoSlopeNorm
from IPython.display import Image, display

# Training and evaluation for the autoregressive model

def train_epoch(model, loader, optimizer, device='cpu'):
    model.to(device)
    model.train() # specifies that the model is in training mode

    losses = []

    for batch in loader:
        x = batch[0]
        y = batch[1]
        x, y = x.float().to(device), y.float().to(device)

        # Model prediction
        preds = model(x)

        # MSE loss function
        loss1 = nn.MSELoss()(preds, y[:,0:1,:,:])

        # Prepare the input for the next prediction. The last channel of the input tensor
        # (which contains the water depth information) is replaced by the predictions
        # from the first time step. This step is key for the autoregressive nature of the model.
        x_updated = torch.cat((x[:, :3, :, :], preds), dim=1)
        
        # Forward pass: Compute predictions with the updated input tensor
        preds2 = model(x_updated)

        # Calculate the loss for the second predicted time step
        loss2 = nn.MSELoss()(preds2, y[:,1:,:,:])

        loss = loss1 + loss2

        losses.append(loss.cpu().detach())

        # Backpropagate and update weights
        loss.backward()   # compute the gradients using backpropagation
        optimizer.step()  # update the weights with the optimizer
        optimizer.zero_grad(set_to_none=True)   # reset the computed gradients

    losses = np.array(losses).mean()

    return losses

def evaluation(model, loader, device='cpu'):
    model.to(device)
    model.eval()  # specifies that the model is in evaluation mode

    losses = []

    with torch.no_grad():
        for batch in loader:
            x, y = batch
            x, y = x.float().to(device), y.float().to(device)

            input_tensor = x

            for time_step in range(1, y.shape[1]):
           
                preds = model(input_tensor)

                loss = nn.MSELoss()(y[:,time_step:time_step+1,:,:], preds)

                input_tensor = torch.cat((input_tensor[:, :3, :, :], preds), dim=1)

                losses.append(loss.cpu().detach())

    average_loss = np.mean(losses)

    return average_losses

# The class below is used to 
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Access the actual sample from the Subset
        sample = self.dataset[idx]
        # Extract 'inputs' and 'targets' from the sample
        inputs = sample['inputs']
        targets = sample['targets']
        return inputs, targets