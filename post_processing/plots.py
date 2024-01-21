# .py file containing the functions used for plotting results from CNN/ConvLSTM models for DSAIE FLOOD project.

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

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