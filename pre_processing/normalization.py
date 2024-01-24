# .py file containing the functions used for normalizing datasets in ConvLSTM model for DSAIE FLOOD project.

import torch

from sklearn.preprocessing import MinMaxScaler

from pre_processing.load_datasets import count_pixels 

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

def normalize_dataset(dataset, scaler_x, scaler_wd, scaler_q, train_val):
    '''
    Function for normalizing every dataset. 

    Inputs: dataset = dataset to be normalized
            scaler_x, scaler_wd, scaler_q = scalers for inputs and targets (water depth and discharge), created 
                                            with the scaler function
            train_val = str, 

    Outputs: normalized_dataset = dataset after normalization 
    '''
    len_time = dataset[0][1].shape[0]
    pixels = count_pixels(train_val)
    input_features = dataset[0][0].shape[1]
    normalized_dataset = []
    for idx in range(len(dataset)):
        norm_x = torch.FloatTensor(scaler_x.transform(dataset[idx][0][0].reshape(input_features, -1).T).T.reshape((1, input_features, pixels, pixels)))
        norm_wd = torch.FloatTensor(scaler_wd.transform(dataset[idx][1][:, 0].reshape(1, -1).T).reshape(len_time, 1, pixels, pixels))
        norm_q = torch.FloatTensor(scaler_q.transform(dataset[idx][1][:, 1].reshape(1, -1).T).reshape(len_time, 1, pixels, pixels))
        
        norm_y = torch.cat((norm_wd, norm_q), dim = 1)
        normalized_dataset.append((norm_x, norm_y))
    return normalized_dataset

def denormalize_dataset(inputs, outputs, train_val, scaler_x, scaler_wd, scaler_q, sample):
    '''
    Function for denormalizing every dataset. 

    Inputs: dataset = dataset to be normalized
            train_val_test : str, Identifier of dictionary. Expects: 'train_val', 'test1', 'test2', 'test3'.
            scaler_x, scaler_wd, scaler_q = scalers for inputs and targets (water depth and discharge), created 
                                            with the scaler function

    Outputs: normalized_dataset = dataset after normalization 
    '''
    x = inputs #inputs 
    wd = outputs[:, 0] #.permute(1, 0, 2, 3)
    q = outputs[:, 1] #.permute(1, 0, 2, 3)

    pixels = count_pixels(train_val)

    # denormalize inputs and targets 
    elevation = scaler_x.inverse_transform(x.reshape(4, -1).T.cpu())[:, 0].reshape(pixels, pixels)

    water_depth = scaler_wd.inverse_transform(wd.reshape(1, -1).T.cpu()).reshape(48, pixels, pixels)
    discharge = scaler_q.inverse_transform(q.reshape(1, -1).T.cpu()).reshape(48, pixels, pixels)

    return elevation, water_depth, discharge