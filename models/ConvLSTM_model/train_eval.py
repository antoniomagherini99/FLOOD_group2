import torch
import torch.nn as nn
import numpy as np

def obtain_predictions(model, input, device, steps = 0):
    '''
    Obtain Predictions based on model class

    Parameters
    ----------
    model : trained AI model.
    input : tensor, inputs tensor with shape (time_stpes, fetures, height, width).
    device : str
        Device on which to perform the computations; 'cuda' is the default.
    steps : int
        Number of time steps, used by multistep convlstm model.
        Default is 48

    Raises
    ------
    Exception: if model specified is not 'ConvLSTM' nor 'MultiStepConvLSTM'
    it raises the following message
        'Need to check if statements to see if model is ' +
                        'implemented correctly'.

    Returns
    -------
    predictions : list
        Predictions from model

    '''
    model_who = str(model.__class__.__name__)
    if model_who == 'ConvLSTM' or model_who == 'MultiStepConvLSTM':
        if len(input.shape) == 4: # no batch
            input = input.unsqueeze(0) # create a batch of 1, model requires it to run
            unbatch = True
        else:
            unbatch = False
            
        if model_who == 'ConvLSTM':
            sample_list, _ = model(input.to(device))
            predictions = torch.cat(sample_list, dim=1)
            
        elif model_who == 'MultiStepConvLSTM':
            input = input.repeat(1, steps, 1, 1, 1) # Requires input at all time steps
            predictions = model(input.to(device)).to(device)
            
        if unbatch:
            predictions = predictions[0].detach().cpu() # remove batch
        
    elif model_who == 'UNet':
        predictions = model(input).to(device).detach().cpu()
    else:
        raise Exception('Need to check if statements to see if model is ' +
                        'implemented correctly')
    return predictions

def train_epoch_conv_lstm(model, loader, optimizer, device='cuda', loss_f='MSE'):
    '''
    Function for training and validating the trained model.
    It uses MSE loss.

    Inputs: model : trained AI model.
            loader : dataloader to feed data to the model in batches
            optimizer : chosen optimizer for SGD 
            device : str
                     Device on which to perform the computations; 'cuda' is the default.
            loss_f = str, key that specifies the function for computing the loss, 
                     accepts 'MSE' and 'MAE'. If other arguments are set it raises an Exception
                     default = 'MSE'
    
    Outputs: losses : MSE training loss between predictions and targets 
    '''
    model.to(device)
    model.train() # specifies that the model is in training mode

    losses = []
    for batch in loader:
        sequence_length = batch[1].shape[1]
        x = batch[0]
        y = batch[1].to(device)

        predictions = obtain_predictions(model, x, device, sequence_length)
        
        # compute loss
        loss = choose_loss(loss_f, predictions, y)
        
        losses.append(loss.cpu().detach())
        
        # Backpropagate and update weights
        loss.backward()   # compute the gradients using backpropagation
        optimizer.step()  # update the weights with the optimizer
        optimizer.zero_grad(set_to_none=True)   # reset the computed gradients

    losses = np.array(losses).mean()

    return losses

def evaluation_conv_lstm(model, loader, device='cuda', loss_f='MSE'):
    '''
    Function for validating the trained model.
    It uses MSE loss.

    Inputs: model : trained AI model.
            loader : dataloader to feed data to the model in batches
            device : str
                     Device on which to perform the computations; 'cuda' is the default.
            loss_f = str, key that specifies the function for computing the loss, 
                     accepts 'MSE' and 'MAE'. If other arguments are set it raises an Exception
                     default = 'MSE'
    
    Outputs: Outputs: losses : MSE validating loss between predictions and targets 
    '''
    model.to(device)
    model.eval() # specifies that the model is in evaluation mode 

    losses = []
    
    with torch.no_grad():
        for batch in loader:
            sequence_length = batch[1].shape[1]
            x = batch[0]
            y = batch[1].to(device)
    
            predictions = obtain_predictions(model, x, device, sequence_length)
            
            # compute loss
            loss = choose_loss(loss_f, predictions, y)
            
            losses.append(loss.cpu().detach())

    losses = np.array(losses).mean()

    return losses

def choose_loss(loss_f, preds, targets):
    '''
    Function to specify the function to be used for computing the loss

    Inputs: loss_f = str, key that specifies the function for computing the loss, 
                     accepts 'MSE' and 'MAE'. If other arguments are set it raises an Exception
                     default = 'MSE'
            preds = torch.tensor, contains the predictions computed with the function "obtain_predictions"
            targets = torch.tensor, contains the targets of the dataset

    Output: loss = computed loss with the specified function between predictions and targets 
    '''

    if loss_f == 'MSE':
        loss = nn.MSELoss()(preds, targets)
    elif loss_f == 'MAE':
        loss = nn.L1Loss()(preds, targets)
    else: raise Exception('The specified loss function is not MSE nor MAE or you spelled it wrongly.\n\
Set loss_f="MSE" for Mean Squared Error or loss_f="MAE" for Mean Absolut Error.')
    
    return loss