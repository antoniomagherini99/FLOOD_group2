# .py file containing the functions used for plotting results from CNN/ConvLSTM models for DSAIE FLOOD project.

import matplotlib.pyplot as plt

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
    plt.title(f'{model} training and validation losses')
    plt.xlabel('Epochs [-]')
    plt.ylabel('Loss [-]')
    plt.gca().yaxis.set_label_coords(-0.01, 0.5)
    plt.legend()
    plt.show()

    return None