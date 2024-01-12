# .py file containing the functions used for normalizing datasets in ConvLSTM model for DSAIE FLOOD project.

from sklearn.preprocessing import MinMaxScaler
import torch

def scaler(train_dataset, scaler_x=MinMaxScaler(), scaler_wd=MinMaxScaler(), scaler_q=MinMaxScaler()):
    '''
    Scaler function for the train dataset.

    Inputs: train_datset = training and validation dataset
            scaler_x = input scaler, default = 'MinMaxScaler()'
            scaler_wd = water depth scaler, default = 'MinMaxScaler()'
            scaler_q = discharge scaler, default = 'MinMaxScaler()'
    
    Outputs: scaler_x, scaler_wd, scaler_q = scalers after partial fit with train dataset 
    '''
    for idx in range(len(train_dataset)):
        scaler_x.partial_fit(train_dataset[idx][0].reshape(train_dataset[0][0].shape[1], -1).T.cpu())
        scaler_wd.partial_fit(train_dataset[idx][1][:, 0].reshape(-1, 1).cpu())
        scaler_q.partial_fit(train_dataset[idx][1][:, 1].reshape(-1, 1).cpu())

    
    return scaler_x, scaler_wd, scaler_q

def normalize_dataset(dataset, scaler_x, scaler_wd, scaler_q):
    '''
    Function for normalizing every dataset. 

    Inputs: dataset = dataset to be normalized
            scaler_x, scaler_y = scalers for inputs and targets, created 
                                 with the scaler function

    Outputs: normalized_dataset = dataset after normalization 
    '''
    min_x, max_x = scaler_x.data_min_[0], scaler_x.data_max_[0]
    min_wd, max_wd = scaler_wd.data_min_[0], scaler_wd.data_max_[0]
    min_q, max_q = scaler_q.data_min_[0], scaler_q.data_max_[0]
    normalized_dataset = []
    for idx in range(len(dataset)):
        x = dataset[idx][0]
        wd = dataset[idx][1][:, 0]
        q = dataset[idx][1][:, 1]
        norm_x = (x - min_x) / (max_x - min_x)
        norm_wd = (wd - min_wd) / (max_wd - min_wd)
        norm_q = (q - min_q) / (max_q - min_q)
        norm_y = torch.cat((norm_wd, norm_q), dim = 1)
        normalized_dataset.append((norm_x, norm_y))
    
    return normalized_dataset