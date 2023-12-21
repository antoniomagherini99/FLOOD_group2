# file for storing functions used for loading datasets
# 1st version - Antonio

# The following paths access the main folder (i.e., dataset_train_val, dataset1 and so on). 
# The path of the specific type of data (DEM, VX and so on) is to be specified after.
path_train = f'./dataset_train_val/' 
path_test1 = f'./dataset1/'
path_test2 = f'./dataset2/'
path_test3 = f'./dataset3/'

# ------------- #

# The following lines create variables to more easily specify what we use the model for 
# (i.e., train and validate, test with dataset 1 and so on) in the following functions.

train_val = 'train_val'
test1 = 'test1'
test2 = 'test2'
test3 = 'test3'

# ------------- #

def process_elevation_data(file_id, train_val_test):
    """
    Processes elevation data from a DEM file.

    Input:
    file_id (str): Identifier of the DEM file to be processed.
    train_val_test: key for specifying what we are using the model for
                   'train_val' = train and validate the model
                   'test1' = test the model with dataset 1
                   'test2' = test the model with dataset 2
                   'test3' = test the model with dataset 3

    Output:
    torch.Tensor: A tensor combining the original elevation data and its slope in x and y directions.
    """
    # specify what we use the model for -- so far works for only one specified input (i.e., file_id), 
    # will need to be improved to work for all inputs regardless of the number of the file
    if train_val_test == 'train_val':
        file_path = path_train + f'DEM_{file_id}.txt'
    elif train_val_test == 'test1':
        file_path = path_test1 + f'DEM_{file_id}.txt'
    elif train_val_test == 'test2':
        file_path = path_test2 + f'DEM_{file_id}.txt'
    elif train_val_test == 'test3':
        file_path = path_test3 + f'DEM_{file_id}.txt'

    # # Construct the file path from the given file identifier
    # file_path = f'DEM_{file_id}.txt'

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

# ------------- #

    def process_water_depth(file_id, train_val_test, time_step=0):
    """
    Processes water depth data from a specific time step in a file.

    Args:
    file_id (str): Identifier of the water depth file to be processed.
    train_val_test: key for specifying what we are using the model for
                   'train_val' = train and validate the model
                   'test1' = test the model with dataset 1
                   'test2' = test the model with dataset 2
                   'test3' = test the model with dataset 3
    time_step (int): Time step to extract from the file. Default is the first time step. Default is the first time step.

    Returns:
    torch.Tensor or None: A 64x64 tensor representing water depth at the given time step, or None if the data is invalid.
    """
    # specify what we use the model for -- so far works for only one specified input (i.e., file_id), 
    # will need to be improved to work for all inputs regardless of the number of the file
    if train_val_test == 'train_val':
        file_path = path_train + f'WD_{file_id}.txt'
    elif train_val_test == 'test1':
        file_path = path_test1 + f'WD_{file_id}.txt'
    elif train_val_test == 'test2':
        file_path = path_test2 + f'WD_{file_id}.txt'
    elif train_val_test == 'test3':
        file_path = path_test3 + f'WD_{file_id}.txt'
    
    # file_path = f'WD_{file_id}.txt'

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
    
    return None

# ------------- #
