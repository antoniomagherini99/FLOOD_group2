import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import animation
import torch
import numpy as np

from pre_processing.normalization import denormalize_dataset

def definitions(index):
    feature_dic = {
        0: 'Water Depth',
        1: 'Discharge per Meter Width'
    }

    feature_dic_units = {
        0: r'$m$',
        1: r'$m^3 s^{-1} m^{-1}$'
    }
    var = feature_dic[index]
    unit = feature_dic_units[index]
    return var, unit

def find_axes(axis):
    div = make_axes_locatable(axis)
    cax = div.append_axes('right', '5%', '5%')
    return cax

def animated_plot(figure, animated_tensor, axis, variable, diff = False, prediction = False):
    
    # set color maps, units and titles
    
    if variable == 'water_depth':
        cmap = 'Blues'
        title, label = definitions(0)
    elif variable == 'discharge':
        cmap = 'Greens'
        title, label = definitions(1)
    else:
        raise('Check variable input')  
    
    if diff == False:
        if prediction == False:
            title = 'Targets: ' + title
        elif prediction == True:
            title = 'Predictions: ' + title
    elif diff == True:
        cmap = 'seismic' 
        title = 'Differences: ' + title
        
    cax = find_axes(axis)
    image = axis.imshow(animated_tensor[0], cmap=cmap, origin='lower')
    cb = figure.colorbar(image, cax=cax)
    cb.set_label(label)
    axis.set_title(title)
    # set color bar limits
    min_val = animated_tensor.min()
    max_val = animated_tensor.max()
    if diff == True: # Only the differences will have a minimum that is negative (!check for CNN)
        absolute_max = np.max(np.array([abs(min_val), max_val]))
        min_val = -1 * absolute_max
        max_val = absolute_max
    image.set_clim(min_val, max_val)
    return image


def plot_animation(sample, dataset, model, title_anim, scaler_x,
                   scaler_wd, scaler_q, device='cuda', save=False):
    '''
    Plot animation to visualize the evolution of certain variables over time.

    Parameters
    ----------
    sample : int
        Choose a sample to animate.
    dataset : torch.utils.data.Dataset
        Normalized dataset of train_val or test (1, 2, or 3).
    model : class of model
        Model to predict on the sample.
    title_anim : str
        Title of the animation.
    scaler_x : instance of normalizer
        Used on inputs.
    scaler_wd : instance of normalizer
        Used on the water depth targets.
    scaler_q : instance of normalizer
        Used on the discharge target.
    device : str
        Device on which to perform the computations; 'cuda' is the default.
    save : bool
        Default is False. If True, will save the animation in the
        post_processing folder with the title in the format:
        'title_anim + model_who + sample'.

    Returns
    -------
    None.
    '''

    # Extracting information from the dataset
    input = dataset[sample][0]
    output = dataset[sample][1]
    boundary_condition = input[0, 3]

    # Denormalizing the data for plotting
    elevation, water_depth, discharge = denormalize_dataset(input, output, title_anim, scaler_x, scaler_wd, scaler_q, sample)
    # need to try and find a more generic way to do this
    model_who = str(model.__class__)[-10:-2]
    if model_who == 'ConvLSTM':
        sample_list, _ = model(dataset[sample][0].unsqueeze(0).to(device))  # create a batch of 1?
        preds = torch.cat(sample_list, dim=1).detach().cpu()[0]  # remove batch
    
    _, wd_pred, q_pred = denormalize_dataset(input, preds, title_anim, scaler_x, scaler_wd, scaler_q, sample)


    # Creating subplots
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, figsize=(10, 10))
    fig.subplots_adjust(wspace=0.5)  # Adjust the width space between subplots

    # Subplot 1
    cax1 = find_axes(ax1)
    im1 = ax1.imshow(elevation, cmap='terrain', origin='lower')
    cb1 = fig.colorbar(im1, cax=cax1)
    cb1.set_label(r'$m$')
    ax1.set_title('Elevation')

    # Subplot 4
    ax4.imshow(boundary_condition, cmap='binary', origin='lower') # should edit to make more clear
    ax4.set_title('Breach Location')

    # Subplot 2
    im2 = animated_plot(fig, water_depth, ax2, 'water_depth')

    # Subplot 3
    im3 = animated_plot(fig, discharge, ax3, 'discharge')

    # Subplot 5
    im5 = animated_plot(fig, wd_pred, ax5, 'water_depth', False, True)

    # Subplot 6
    im6 = animated_plot(fig, q_pred, ax6, 'discharge', False, True)

    # Subplot 8
    diff_wd = wd_pred - water_depth
    im8 = animated_plot(fig, diff_wd, ax8, 'water_depth', True)

    # Subplot 9
    diff_q = q_pred - discharge
    im9 = animated_plot(fig, diff_q, ax9, 'discharge', True)

    title_con = title_anim + f' for sample {sample} using model: ' + model_who
    over_title = fig.suptitle('Hour 1: ' + title_con, fontsize=16) # try and update this to show the hour

    def animate(i):
        over_title.set_text(f'Hour {i + 1}: ' + title_con)
        # Subplot 2
        im2.set_data(water_depth[i])

        # Subplot 3
        im3.set_data(discharge[i])

        # Subplot 5
        im5.set_data(wd_pred[i])

        # Subplot 6
        im6.set_data(q_pred[i])

        # Subplot 8
        im8.set_data(diff_wd[i])

        # Subplot 9
        im9.set_data(diff_q[i])

    # Set up the animation
    ani = animation.FuncAnimation(fig, animate, frames=water_depth.shape[0])

    # Display the animation
    plt.show()

    # Save the animation if specified
    if save:
        ani.save('post_processing/' + title_anim + '_' + model_who + '_' +
                 str(sample) + '.gif', writer='Pillow', fps=5)

    return None
