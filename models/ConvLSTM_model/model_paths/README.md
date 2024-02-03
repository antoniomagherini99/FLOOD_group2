This folder contains the file paths of the ConvLSTM models that were created during the training process.
# 4 hidden_dim
The model with 4 hidden dim was trained on non augmented data and performs badly on all testing datasets.
Hyperparameters:
  hidden_dimension = 4
  hidden_layers = 8
  kernel = 5 x 5
  batch_size = 8
# 8 hidden_dim
The model with 8 hidden dim was trained on augmented data (which had random flipping/rotation applied once to the entire training dataset).
Hyperparameters:
  hidden_dimension = 8
  hidden_layers = 8
  kernel = 5 x 5
  batch_size = 8
  scalers:
    scaler_x_simple
    scaler_y_simple
Metrics:
  If f1 is used, this is the best model on almost all datasets
# 16 hidden dim
The largest model was trained on 560 samples (a large amount of augmentation). Extremely expensive to train.
Hyperparameters:
  hidden_dimension = 16
  hidden_layers = 8
  kernel = 5 x 5
  batch_size = 56
  Scalers:
    scaler_x
    scaler_y
  Metrics:
    Lowest loss and recall for test dataset 2 specifically.
