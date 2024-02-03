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

# Data Processing

def process_elevation_data(file_id):
    """
    Processes elevation data from a DEM file.

    Args:
    file_id (str): Identifier of the DEM file to be processed.

    Returns:
    torch.Tensor: A tensor combining the original elevation data and its slope in x and y directions.
    """

    # Construct the file path from the given file identifier
    file_path = f'DEM_{file_id}.txt'

    # Load the elevation data from the file
    elevation_data = np.loadtxt(file_path)

    # Reshape the elevation data into a 64x64 grid
    elevation_grid = elevation_data[:, 2].reshape(64, 64)

    # Convert the elevation grid to a PyTorch tensor
    elevation_tensor = torch.tensor(elevation_grid)

    # Compute the slope in the x and y directions
    slope_x, slope_y = torch.gradient(elevation_tensor)

    # Combine the elevation tensor with the slope tensors
    elevation_slope_tensor = torch.stack((elevation_tensor, slope_x, slope_y), dim=0)

    return elevation_slope_tensor

def process_water_depth(file_id, time_step=0):
    """
    Processes water depth data from a specific time step in a file.

    Args:
    file_id (str): Identifier of the water depth file to be processed.
    time_step (int): Time step to extract from the file. Default is the first time step. Default is the first time step.

    Returns:
    torch.Tensor or None: A 64x64 tensor representing water depth at the given time step, or None if the data is invalid.
    """
    file_path = f'WD_{file_id}.txt'

    # Read the file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    try:
        # Extract the specified row and convert string elements to floats
        selected_row = lines[time_step].split()
        depth_values = [float(val) for val in selected_row]

        # Validate and reshape the data into a 64x64 tensor
        if len(depth_values) == 64 * 64:
            depth_tensor = torch.tensor(depth_values).view(64, 64)
            return depth_tensor
        else:
            raise ValueError(f"The number of elements in {file_path} at time step {time_step} doesn't match a 64x64 matrix.")
    except IndexError:
        raise IndexError(f"Time step {time_step} is out of range for the file {file_path}.")

# Create plot/GIF of an example dataset

def plot_and_save_tensors(tensors, file_number, titles, filename):
    num_tensors = len(tensors)
    fig, axes = plt.subplots(1, num_tensors, figsize=(12, 5))

    if num_tensors == 1:
        axes = [axes]

    for ax, tensor, title in zip(axes, tensors, titles):
        # Define 'cmap' and 'vmin', 'vmax' for each type of plot
        if title.startswith('Elevation'):
            cmap = 'terrain'
            vmin, vmax = None, None  # or set specific min/max values for elevation
        elif 'Water Depth' in title:
            cmap = 'Blues'
            vmin, vmax = 0, 1  # Fixed scale for water depth
        else:
            cmap = 'coolwarm'
            vmin, vmax = None, None  # or set specific min/max values for other types of plots

        # Use 'imshow' with the specified 'cmap', 'vmin', and 'vmax'
        im = ax.imshow(tensor, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)

        fig.colorbar(im, ax=ax)
        ax.set_title(title)

    plt.suptitle(f'Dataset ID: {file_number}', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename)
    plt.close(fig)

def create_water_depth_gif(file_id, start_time_step, end_time_step, fps=10, gif_filename='water_depth_animation.gif', loop=0):
    images = []

    for WD_time_step in range(start_time_step, end_time_step + 1):
        elevation_slope_tensor = process_elevation_data(file_id)
        water_depth_tensor = process_water_depth(file_id, time_step=WD_time_step)
        combined_tensors = list(elevation_slope_tensor) + [water_depth_tensor]

        water_depth_title = f'Water Depth after\n{WD_time_step*30} min'
        titles = ['Elevation', 'Slope X', 'Slope Y', water_depth_title]

        # Filename for the saved image
        filename = f"plot_{WD_time_step}.png"
        plot_and_save_tensors(combined_tensors, file_id, titles, filename)

        images.append(imageio.imread(filename))

    # Save the GIF with the specified loop parameter
    imageio.mimsave(gif_filename, images, fps=fps, loop=loop)

    # Display the GIF in the notebook
    display(Image(filename=gif_filename))

# Create training and test datasets

def create_training_dataset(t0, t):
    train_dataset = []

    for i in range(1, 81):  # Looping through file IDs from DEM_1 to DEM_80
        file_id = i
        
        # Input Tensor (input_tensor.shape will be [4, 64, 64])
        elevation_slope_tensor = process_elevation_data(file_id)
        water_depth_input_tensor = process_water_depth(file_id, time_step=t0)  # Time Step is t0
        # Add an extra dimension to make water_depth_tensor [1, 64, 64]
        water_depth_input_tensor = torch.unsqueeze(water_depth_input_tensor, 0)
        
        # Concatenate to create the input tensor
        input_tensor = torch.cat((elevation_slope_tensor, water_depth_input_tensor), dim=0)
        input_tensor = input_tensor.double()
        
        # Output Tensor (output_tensor.shape will be [1, 64, 64])
        water_depth_output_tensor = process_water_depth(file_id, time_step=t0+t)  # Time Step is now t0+t
        # Add an extra dimension to make water_depth_tensor [1, 64, 64]
        output_tensor = torch.unsqueeze(water_depth_output_tensor, 0)
        output_tensor = output_tensor.double()
        
        # Create a tuple and append to the train_dataset
        train_dataset_sample = (input_tensor, output_tensor)
        train_dataset.append(train_dataset_sample)

    return train_dataset

def create_test_dataset(t0, t):
    test_dataset = []

    for i in range(500, 520):  # Assuming file IDs are numbered from 1 to 80
        file_id = i
        
        # Input Tensor (input_tensor.shape will be [4, 64, 64])
        
        elevation_slope_tensor = process_elevation_data(file_id)
        water_depth_input_tensor = process_water_depth(file_id, time_step=t0) # Time Step is t0
        # Add an extra dimension to make water_depth_tensor [1, 64, 64]
        water_depth_input_tensor = torch.unsqueeze(water_depth_input_tensor, 0)
        
        # elevation_slope_tensor.shape --> [3, 64, 64]
        # water_depth_tensor.shape --> [1,64, 64]

        # Concatenate to create the input tensor for this file ID
        input_tensor = torch.cat((elevation_slope_tensor, water_depth_input_tensor), dim=0)
        input_tensor = input_tensor.double()
        
        # Output Tensor (output_tensor.shape will be [1, 64, 64])
        water_depth_output_tensor = process_water_depth(file_id, time_step=t0+t) # Time Step is now t0+t
        # Add an extra dimension to make water_depth_tensor [1, 64, 64]
        output_tensor = torch.unsqueeze(water_depth_output_tensor, 0)
        output_tensor = output_tensor.double()
        
        # Create a tuple
        test_dataset_sample = (input_tensor, output_tensor)
        
        # Append the sample to the test_dataset list
        test_dataset.append(test_dataset_sample)

    return test_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Normalization

def normalize_dataset(dataset, scaler_x, scaler_y):
    min_x, max_x = scaler_x.data_min_[0], scaler_x.data_max_[0]
    min_y, max_y = scaler_y.data_min_[0], scaler_y.data_max_[0]
    normalized_dataset = []
    for idx in range(len(dataset)):
        x = dataset[idx][0]
        y = dataset[idx][1]
        norm_x = (x - min_x) / (max_x - min_x)
        norm_y = (y - min_y) / (max_y - min_y)
        normalized_dataset.append((norm_x, norm_y))
    return normalized_dataset

# Create you own CNN model

# Define the CNN architecture

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels=1, out_channels=1, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UniDepth_UNet(nn.Module):
    def __init__(self, n_channels=4, n_classes=1, bilinear=False):
        super(UniDepth_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 8))
        self.down1 = (Down(8, 16))
        self.down2 = (Down(16, 32))
        self.down3 = (Down(32, 64))
        factor = 2 if bilinear else 1
        self.down4 = (Down(64, 128 // factor))
        self.up1 = (Up(128, 64 // factor, bilinear))
        self.up2 = (Up(64, 32 // factor, bilinear))
        self.up3 = (Up(32, 16 // factor, bilinear))
        self.up4 = (Up(16, 8, bilinear))
        self.outc = (OutConv(8, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
        
# Training and Evaluation

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
        loss = nn.MSELoss()(preds, y)

        losses.append(loss.cpu().detach())

        # Backpropagate and update weights
        loss.backward()   # compute the gradients using backpropagation
        optimizer.step()  # update the weights with the optimizer
        optimizer.zero_grad(set_to_none=True)   # reset the computed gradients

    losses = np.array(losses).mean()

    return losses

def evaluation(model, loader, device='cpu'):
    model.to(device)
    model.eval() # specifies that the model is in evaluation mode

    losses = []

    with torch.no_grad():
        for batch in loader:
            x = batch[0]
            y = batch[1]
            x, y = x.float().to(device), y.float().to(device)

            # Model prediction
            preds = model(x)

            # MSE loss function
            loss = nn.MSELoss()(preds, y)
            losses.append(loss.cpu().detach())

    losses = np.array(losses).mean()

    return losses

# Predict water depth and retrieve DEM and real water depth (based on a test data sample)

def predict_water_depth_and_retrieve_data(test_dataset, model, data_id, scaler_x, scaler_y, device=device):
    """
    Selects a sample from the test dataset, makes a prediction using the provided model,
    and returns the DEM, real water depth, and predicted water depth.

    Args:
    test_dataset: The test dataset containing the data.
    model: The trained model used for prediction.
    data_id (int): The ID of the test data sample to be retrieved and predicted.
    scaler_x: Scaler used to normalize the input features.
    scaler_y: Scaler used to normalize the output features.
    device: The device on which the model is running (e.g., 'cpu' or 'cuda').

    Returns:
    tuple: DEM, real water depth, and predicted water depth arrays.
    """
    # Select one sample
    x = test_dataset[data_id][0].unsqueeze(0)
    x = x.float().to(device)
    real_WD = test_dataset[data_id][1]

    # Predict the water depth
    with torch.no_grad():
        pred_WD = model(x).detach()

    # Inverse transform the data
    DEM = scaler_x.inverse_transform(x[0].reshape(4, -1).T.cpu())[:, 0].reshape(64, 64)
    real_WD = scaler_y.inverse_transform(real_WD.reshape(-1, 1).cpu()).reshape(64, 64)
    pred_WD = scaler_y.inverse_transform(pred_WD.reshape(-1, 1).cpu()).reshape(64, 64)

    return DEM, real_WD, pred_WD

# Plot comparison between predicted and real water depth

def plot_water_depth_comparison(DEM, real_WD, pred_WD):
    """
    Plots comparison between real and predicted water depth along with DEM and difference.

    Args:
    DEM (array-like): Digital Elevation Model data.
    real_WD (array-like): Real water depth data.
    pred_WD (array-like): Predicted water depth data.
    """
    fig, axs = plt.subplots(1, 4, figsize=(17, 5))

    # Calculate the difference and maximum values
    diff_WD = real_WD - pred_WD
    max_WD = max(pred_WD.max(), real_WD.max())
    max_diff_WD = max(diff_WD.max(), -diff_WD.min())

    # Custom colormap that starts with white and then uses 'Blues'
    cmap = plt.cm.Blues
    cmap.set_under('white', alpha=0)

    axs[0].imshow(DEM.squeeze(), cmap='terrain', origin='lower')
    axs[1].imshow(real_WD.squeeze(), vmin=1e-6, vmax=max_WD, cmap=cmap, origin='lower')
    axs[2].imshow(pred_WD.squeeze(), vmin=1e-6, vmax=max_WD, cmap=cmap, origin='lower')
    axs[3].imshow(diff_WD.squeeze(), vmin=-max_diff_WD, vmax=max_diff_WD, cmap='RdBu', origin='lower')

    # Adding colorbars
    plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin=DEM.min(), vmax=DEM.max()), cmap='terrain'),
                 fraction=0.05, shrink=0.9, ax=axs[0])
    plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin=1e-6, vmax=max_WD), cmap=cmap),
                 fraction=0.05, shrink=0.9, ax=axs[1])
    plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin=1e-6, vmax=max_WD), cmap=cmap),
                 fraction=0.05, shrink=0.9, ax=axs[2])
    plt.colorbar(plt.cm.ScalarMappable(norm=TwoSlopeNorm(vmin=-max_diff_WD, vmax=max_diff_WD, vcenter=0),
                 cmap='RdBu'), fraction=0.05, shrink=0.9, ax=axs[3])

    for ax in axs:
        ax.axis('off')

    axs[0].set_title('DEM')
    axs[1].set_title('Real Water Depth (m)')
    axs[2].set_title('Predicted Water Depth (m)')
    axs[3].set_title('Difference (m)')

    plt.show()
    
# Train and evaluate a model for a specific time step t for prediction
    
def train_and_evaluate_model_for_timestep(t, device, test_data_id, num_epochs=300, learning_rate=0.001, batch_size=64):
    # Create datasets
    train_dataset = create_training_dataset(t0=0, t=t)
    val_dataset = create_test_dataset(t0=0, t=t)
    test_dataset = create_test_dataset(t0=0, t=t)

    # Initialize scalers
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    # Fit scalers to training data
    for x, y in train_dataset:
        scaler_x.partial_fit(x.reshape(-1, 4))
        scaler_y.partial_fit(y.reshape(-1, 1))

    # Normalize datasets
    normalized_train_dataset = normalize_dataset(train_dataset, scaler_x, scaler_y)
    normalized_val_dataset = normalize_dataset(val_dataset, scaler_x, scaler_y)
    normalized_test_dataset = normalize_dataset(test_dataset, scaler_x, scaler_y)

    # Instantiate the model
    model = UniDepth_UNet().to(device)

    # Training setup
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    train_loader = DataLoader(normalized_train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(normalized_val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(normalized_test_dataset, batch_size=batch_size, shuffle=False)

    train_losses, validation_losses = [], []

    # Training loop
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device=device)
        validation_loss = evaluation(model, val_loader, device=device)
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)
    
    # Evaluate the model
    test_loss = evaluation(model, test_loader, device=device)
    print(f'Test Loss for t={t}: {test_loss}')

    # Plot training and validation losses
    plt.plot(train_losses, label='Training')
    plt.plot(validation_losses, label='Validation')
    plt.yscale('log')
    plt.title(f'Losses for t={t}')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()

    # Predict and plot water depth comparison
    DEM, real_WD, pred_WD = predict_water_depth_and_retrieve_data(normalized_test_dataset, model, test_data_id, scaler_x, scaler_y)
    plot_water_depth_comparison(DEM, real_WD, pred_WD)

    return model, train_losses, validation_losses, test_loss, pred_WD

# Generate a GIF that compares predicted and actual flood progession over time

def save_water_depth_plot_comparison(DEM, real_WD, pred_WD, timestep, save_path="plots"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    
    cmap_real_pred = plt.cm.Blues.copy()
    cmap_real_pred.set_under('white', alpha=0)  # Set areas with no water depth to white for real and predicted WD plots
    
    norm_diff = Normalize(vmin=-2, vmax=2)
    
    # DEM Plot
    axs[0].imshow(DEM, cmap='terrain', origin='lower')
    axs[0].set_title('DEM')
    axs[0].axis('off')

    # Real Water Depth Plot
    im_real = axs[1].imshow(real_WD, cmap=cmap_real_pred, vmin=0, vmax=2, origin='lower')
    fig.colorbar(im_real, ax=axs[1], fraction=0.046, pad=0.04)
    axs[1].set_title('Real Water Depth')
    axs[1].axis('off')

    # Predicted Water Depth Plot
    im_pred = axs[2].imshow(pred_WD, cmap=cmap_real_pred, vmin=0, vmax=2, origin='lower')
    fig.colorbar(im_pred, ax=axs[2], fraction=0.046, pad=0.04)
    axs[2].set_title('Predicted Water Depth')
    axs[2].axis('off')

    # Difference in Water Depth Plot
    im_diff = axs[3].imshow(pred_WD - real_WD, cmap='RdBu', norm=norm_diff, origin='lower')
    fig.colorbar(im_diff, ax=axs[3], fraction=0.046, pad=0.04)
    axs[3].set_title('Difference (Pred - Real)')
    axs[3].axis('off')

    plt.tight_layout()
    plt.savefig(f"{save_path}/plot_{timestep}.png")
    plt.close(fig)

def create_predictions_gif(image_folder, gif_name, fps=2):
    images = []
    for file_name in sorted(os.listdir(image_folder), key=lambda x: int(x.split('_')[1].split('.')[0])):
        if file_name.endswith('.png'):
            file_path = os.path.join(image_folder, file_name)
            images.append(imageio.imread(file_path))
    imageio.mimsave(gif_name, images, fps=fps, loop=0)
    
 
