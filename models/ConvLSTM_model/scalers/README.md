This folder contains the scalers of the ConvLSTM models that were created during the training process. These are used to normalize the input and output data of the test datasets.
These only need to be created once unless the seed/augmentation affects the random split of training and validation datasets.

# scaler_x and scaler_y
Used to scale the test datasets of the more complex model which has 16 hidden dimensions

# scaler_x_simple and scaler_y_simples
Used to scale the test datasets for the less complex model which has 8 hidden dimensions
