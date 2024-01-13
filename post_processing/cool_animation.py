import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import animation
import torch
import numpy as np

from pre_processing.normalization import denormalize_dataset

def find_axes(axis):
    div = make_axes_locatable(axis)
    cax = div.append_axes('right', '5%', '5%')
    return cax

def set_colorbar_limits(animated_tensor, image, diff = False):
    min_val = animated_tensor.min()
    max_val = animated_tensor.max()
    if diff == True:
        absolute_max = np.max(np.array([abs(min_val), max_val]))
        min_val = -1 * absolute_max
        max_val = absolute_max
    image.set_clim(min_val, max_val)
    return None



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

    feature1 = 0
    feature2 = 1

    feature_dic = {
        0: 'Water Depth',
        1: 'Discharge per meter width'
    }

    feature_dic_units = {
        0: r'$m$',
        1: r'$m^2 s^{-1}$'
    }

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
    cax2 = find_axes(ax2)
    im2 = ax2.imshow(water_depth[0], cmap='Blues', origin='lower')
    cb2 = fig.colorbar(im2, cax=cax2)
    cb2.set_label(f'{feature_dic_units[feature1]}')
    tx2 = ax2.set_title(f'Hour 0: {feature_dic[feature1]}')
    set_colorbar_limits(water_depth, im2)

    # Subplot 3
    cax3 = find_axes(ax3)
    im3 = ax3.imshow(discharge[0], cmap='Greens', origin='lower')
    cb3 = fig.colorbar(im3, cax=cax3)
    cb3.set_label(f'{feature_dic_units[feature2]}')
    tx3 = ax3.set_title(f'Hour 0: {feature_dic[feature2]}')
    set_colorbar_limits(discharge, im3)

    # Subplot 5
    cax5 = find_axes(ax5)
    im5 = ax5.imshow(wd_pred[0], cmap='Blues', origin='lower')
    cb5 = fig.colorbar(im5, cax=cax5)
    cb5.set_label(f'{feature_dic_units[feature1]}')
    tx5 = ax5.set_title(f'Prediction: {feature_dic[feature1]}')
    set_colorbar_limits(wd_pred, im5)

    # Subplot 6
    cax6 = find_axes(ax6)
    im6 = ax6.imshow(q_pred[0], cmap='Greens', origin='lower')
    cb6 = fig.colorbar(im6, cax=cax6)
    cb6.set_label(f'{feature_dic_units[feature2]}')
    tx6 = ax6.set_title(f'Prediction: {feature_dic[feature2]}')
    set_colorbar_limits(q_pred, im6)

    # Subplot 8
    cax8 = find_axes(ax8)

    diff_wd = wd_pred - water_depth
    im8 = ax8.imshow(diff_wd[0], cmap='seismic', origin='lower')
    cb8 = fig.colorbar(im8, cax=cax8)
    cb8.set_label(f'{feature_dic_units[feature1]}')
    tx8 = ax8.set_title(f'Diff: {feature_dic[feature1]}')
    set_colorbar_limits(diff_wd, im8, True)

    # Subplot 9
    cax9 = find_axes(ax9)
    diff_q = q_pred - discharge
    im9 = ax9.imshow(diff_q[0], cmap='seismic', origin='lower')
    cb9 = fig.colorbar(im9, cax=cax9)
    cb9.set_label(f'{feature_dic_units[feature2]}')
    tx9 = ax9.set_title(f'Diff: {feature_dic[feature2]}')
    set_colorbar_limits(diff_q, im9, True)

    fig.suptitle(title_anim + f' for sample {sample} using model: ' +
                 model_who, fontsize=16)

    def animate(i):
        # Subplot 2
        im2.set_data(water_depth[i])
        tx2.set_text(f'Hour {i}: {feature_dic[feature1]}')

        # Subplot 3
        im3.set_data(discharge[i])
        tx3.set_text(f'Hour {i}: {feature_dic[feature2]}')

        # Subplot 5
        im5.set_data(wd_pred[i])
        tx5.set_text(f'Prediction: {feature_dic[feature1]}')

        # Subplot 6
        im6.set_data(q_pred[i])
        tx6.set_text(f'Prediction: {feature_dic[feature2]}')

        # Subplot 8
        im8.set_data(diff_wd[i])
        tx8.set_text(f'Diff: {feature_dic[feature1]}')

        # Subplot 9
        im9.set_data(diff_q[i])
        tx9.set_text(f'Diff: {feature_dic[feature2]}')

    # Set up the animation
    ani = animation.FuncAnimation(fig, animate, frames=water_depth.shape[0])

    # Display the animation
    plt.show()

    # Save the animation if specified
    if save:
        ani.save('post_processing/' + title_anim + '_' + model_who + '_' +
                 str(sample) + '.gif', writer='Pillow', fps=5)

    return None
