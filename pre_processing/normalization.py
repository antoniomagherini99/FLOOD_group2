# .py file containing the functions used for normalizing datasets in ConvLSTM model for DSAIE FLOOD project.

from sklearn.preprocessing import MinMaxScaler

def scaler(train_dataset, scaler_x=MinMaxScaler(), scaler_y=MinMaxScaler()):
    '''
    Scaler function for the train dataset.

    Inputs: train_datset = training and validation dataset
            scaler_x = input scaler, default = 'MinMaxScaler()'
            scaler_y = targets scaler, default = 'MinMaxScaler()'
    
    Outputs: scaler_x, scaler_y = scalers after partial fit with train dataset 
    '''
    for idx in range(len(train_dataset)):
        scaler_x.partial_fit(train_dataset[idx][0].reshape(train_dataset[0][0].shape[1], -1).T.cpu())
        scaler_y.partial_fit(train_dataset[idx][1].reshape(-1, 1).cpu())

    
    return scaler_x, scaler_y

def normalize_dataset(dataset, scaler_x, scaler_y):
    '''
    Function for normalizing every dataset. 

    Inputs: dataset = dataset to be normalized
            scaler_x, scaler_y = scalers for inputs and targets, created 
                                 with the scaler function

    Outputs: normalized_dataset = dataset after normalization 
    '''
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