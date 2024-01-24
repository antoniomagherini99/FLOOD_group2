# .py file containing the functions used for plotting results from CNN/ConvLSTM models for DSAIE FLOOD project.

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from pre_processing.normalization import *

def plot_losses(train_losses, validation_losses, model):
    '''
    Function for plotting the training and validation losses of CNN and ConvLSTM models.
    y-axis is in log-scale. 
    
    Inputs: train_losses = list containing training losses throughout epochs
            validation_losses =  list containing validation losses throughout epochs
            model = 'CNN' or 'ConvLSTM', key that specifies model used 

    Outputs: None 
    '''

    plt.figure() 
    plt.plot(train_losses, color='blue', ls='--', label='Training')
    plt.plot(validation_losses, color='red', ls='--', label='Validation')
    plt.yscale('log')
    plt.title(model + 'training and validation losses')
    plt.xlabel('Epochs [-]')
    plt.ylabel('Nromalized Loss [-]')
    plt.gca().yaxis.set_label_coords(-0.01, 0.5)
    plt.legend()
    plt.show()

    return None

def plot_test_loss(dataset, model, train_val, device = 'cuda'):
    '''
    Plot test loss for each sample of the requested dataset and model

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Normalized dataset of train_val or test (1, 2, or 3).
    model : class of model
        Model to predict on the sample.
    train_val : str
        key for specifying what we are using the model for
            'train_val' = train and validate the model
            'test1' = test the model with dataset 1
            'test2' = test the model with dataset 2
            'test3' = test the model with dataset 3
        Used in the title of the animation.
    device : str
        Device on which to perform the computations; 'cuda' is the default.

    Returns
    -------
    None.

    '''
    model_who = str(model.__class__.__name__)
    
    num_samples = len(dataset)
    sample_array = np.arange(0, num_samples)
    loss_sample = np.zeros((num_samples))
    
    for sample in range(num_samples):
        target = dataset[sample][1]
        
        if model_who == 'ConvLSTM':
            sample_list, _ = model(dataset[sample][0].unsqueeze(0).to(device))  # create a batch of 1?
            preds = torch.cat(sample_list, dim=1).detach().cpu()[0]  # remove batch
        elif model_who == 'UNet':
            preds = model(dataset[sample][0]).to(device).detach().cpu()
            
        loss_sample[sample] = nn.MSELoss()(preds, target)
    
    plt.figure()
    plt.scatter(sample_array, loss_sample)
    plt.xticks(range(num_samples))
    plt.xlabel('Sample Number')
    plt.ylabel('Normalized Loss [-]')
    plt.title(model_who + ': ' + train_val + ' losses for each sample')
    plt.grid()
    plt.show()
    
    return None

def plot_sorted(dataset, train_val, scaler_x, scaler_wd, scaler_q, loss_wd, loss_q, recall):
    '''
    Function for plotting the DEMs variation sorted in increasing order 
    of average loss (of Water Depth and Discharge)

    Input: dataset = tensor, normalized dataset
           train_val_test : str, Identifier of dictionary. Expects: 'train_val', 'test1', 'test2', 'test3'.
           scaler_x, scaler_wd, scaler_q = scalers for inputs (x) and targets (water depth and discharge), created 
                                            with the scaler function 
    Output: None (plot)
    '''
    
    # get inputs and outputs
    input = dataset[:][0]
    target = dataset[:][1]
    boundary_condition = input[0, 3]

    # denormalize dataset
    elevation, water_depth, discharge = denormalize_dataset(input, target, train_val, 
                                                            scaler_x, scaler_wd, scaler_q)
    
    # compute average loss for sorting dataset
    avg_loss = np.mean(loss_wd, loss_q)

    # sorting dataset

    elevation_sorted = sorted(elevation, avg_loss)
    sorted_indexes = [index for index, _ in elevation_sorted]
    wd_sorted, q_sorted = [water_depth[i] for i in sorted_indexes], [discharge[i] for i in sorted_indexes]
    sorted_recall = [recall[i] for i in sorted_indexes]

    elevation_var = np.std(elevation_sorted)
    
    # plot 
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    fig.subplots_adjust(wspace=0.5)

    # create second y-axis for discharge scale
    ax1_2 = axes[1].twinx()

    axes[0].plot(sorted_indexes, elevation_var)
    axes[1].plot(sorted_indexes, wd_sorted)
    ax1_2.plot(sorted_indexes, q_sorted)
    axes[2].scatter(sorted_indexes, sorted_recall)

    for ax in axes:
        ax.set_xlabel('Sample ID')
    
    axes[0].set_ylabel('Normalized variation [-]')
    axes[1].set_ylabel('Normalized Water Depth loss [-]')
    ax1_2.set_ylabel('Normalized Discharge loss [-]')
    axes[2].set_ylabel('Recall [-]')

    axes[0].set_title('Normalized DEM variation [-]')
    axes[1].set_title('Normalized MSE loss [-]')
    axes[2].set_title('Recall [-]')

    plt.legend()
    plt.show()

    return None