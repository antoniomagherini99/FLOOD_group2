# .py file containing the functions used for plotting results from CNN/ConvLSTM models for DSAIE FLOOD project.

import matplotlib.pyplot as plt
import numpy as np
#import torch
import torch.nn as nn

from models.ConvLSTM_model.train_eval import obtain_predictions
from post_processing.metrics import confusion_mat
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

def plot_test_loss(dataset, model, train_val, device):
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
        Device on which to perform the computations

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
        time_steps = target.shape[0]
        preds = obtain_predictions(model, dataset[sample][0], device, time_steps)
        
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

def plot_sorted(dataset, train_val, scaler_x, scaler_wd, scaler_q, model, device):
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
    # input = dataset[:][0]
    # target = dataset[:][1]
    
    inputs = []
    targets = []
    for i in range(len(dataset)):
        inputs[i] = dataset[i][0]
        targets[i] = dataset[i][1]
    # boundary_condition = input[0, 3]

    # denormalize dataset
    elevation, water_depth, discharge = denormalize_dataset(input, target, train_val, 
                                                            scaler_x, scaler_wd, scaler_q)
    
    # Compute losses uses MSELoss
    # model_who = str(model.__class__.__name__)
    preds = obtain_predictions(model, input, device)

    features = target.shape[1]
    time_steps = target.shape[0]
    losses = np.zeros((features, time_steps)) # initialize empty array
    time_step_array = np.arange(1, time_steps + 1)

    for step in range(time_steps):
        for feature in range(features):
            losses[feature, step] = nn.MSELoss()(preds[step][feature], target[step][feature])
    
    # compute average loss for sorting dataset
    avg_loss = np.mean(losses[0, :], losses[1, :])

    # compute recall
    recall, _, _ = confusion_mat(dataset, model, device)

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

    axes[0].boxplot(sorted_indexes, elevation_var, label='DEM')
    axes[1].plot(sorted_indexes, wd_sorted, color='blue', label='water depth')
    ax1_2.plot(sorted_indexes, q_sorted, color='red', label='discharge')
    axes[2].scatter(sorted_indexes, sorted_recall, color='green', label='recall')

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

def plot_metrics(dataset, model, train_val, device):
    model_who = str(model.__class__.__name__)
    recall, accuracy, f1 = confusion_mat(dataset, model, device)
    
    num_samples = len(dataset)
    sample_array = np.arange(0, num_samples)
    
    plt.figure()
    plt.scatter(sample_array, recall, label = 'Recall')
    plt.scatter(sample_array, accuracy, label = 'Accuracy')
    plt.scatter(sample_array, f1, label = 'F1')
    plt.xticks(range(num_samples))
    plt.xlabel('Sample Number')
    plt.ylabel('Score [-]')
    plt.title(model_who + ': ' + train_val + ' metrics for each sample (WD only)')
    plt.legend()
    plt.grid()
    plt.show()
    return None
        
