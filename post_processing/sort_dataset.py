# .py file containing the functions used for plotting results from CNN/ConvLSTM models for DSAIE FLOOD project.

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from models.ConvLSTM_model.train_eval import *
from post_processing.metrics import confusion_mat
from pre_processing.normalization import *
from post_processing.cool_animation import *

def get_indexes(dataset, model, train_val_test, scaler_x, scaler_y, 
                device, thresholds = torch.tensor([0.1, 0]).reshape(1, -1), 
                loss_f = 'MSE', loss_recall='loss'):
    '''
    Function for plotting the DEMs variation sorted in increasing order 
    of average loss (of Water Depth and Discharge), the relative Water Depth and 
    Discharge loss and the relative Recall

    Input: dataset = tensor, normalized dataset
           model = trained AI model
           train_val_test: key for specifying what we are using the model for
                            'train_val' = train and validate the model
                            'test1' = test the model with dataset 1
                            'test2' = test the model with dataset 2
                            'test3' = test the model with dataset 3
           scaler_x, scaler_y = scalers for inputs (x) and targets (water depth and discharge), created 
                                            with the scaler function 
           device = device on which running the simulations, accepts 'cpu' or 'cuda'
           thresholds: torch.tensor, Denormalized thresholds for each feature. Expects a tensor that 
                        has shape: (1 x num_features). 
                        default = 0.1 m
           loss_f = str, key that specifies the function for computing the loss, 
                    accepts 'MSE' and 'MAE'. If other arguments are set it raises an Exception
                    default = 'MSE'
           loss_recall : srt,
                  specifies if samples are sorted based on average loss or recall. 
                  Expects 'None', 'loss' or 'recall'.
                  default = 'loss'
                    
    Output: None (plot)
    '''

    # get inputs and outputs
    # 1st sample, 2nd input(0)/target(1), 3rd time step, 4th features, 5th/6th pixels
    n_samples = len(dataset)
    n_features = dataset[0][1].shape[1]
    n_pixels = dataset[0][1].shape[-1]
    time_steps = dataset[0][1].shape[0]
    
    # initialize inputs and outputs
    inputs = []
    targets = []
    
    for i in range(n_samples):
        inputs.append(dataset[i][0])
        targets.append(dataset[i][1])

    # initialize denormalization of dataset
    elevation = np.zeros((n_samples, n_pixels, n_pixels))

    # initialize losses
    losses = torch.zeros((n_samples, n_features))

    for i in range(n_samples):
        # denormalize elevation
        elevation[i], _, _ = denormalize_dataset(inputs[i], targets[i], train_val_test,
                                                            scaler_x, scaler_y)
        # make predictions
        preds = obtain_predictions(model, inputs[i], device, time_steps)

    # compute MSE losses
    for feature in range(n_features):
            for i in range(n_samples):
                 losses[i, feature] = choose_loss(loss_f, preds[:][feature], targets[i][:][feature])

    recall, _, _ = confusion_mat(dataset, model, scaler_y, device, thresholds)

    # sorting dataset
    if loss_recall == 'loss':
         # average over columns = features
         avg_loss = torch.mean(losses, dim=1)
         _, sorted_indexes = torch.sort(avg_loss)
    elif loss_recall == 'recall':
         _, sorted_indexes = torch.sort(recall, descending=True)

    return sorted_indexes, elevation, losses, recall
