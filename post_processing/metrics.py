import torch

def threshold_function(tensor):
     '''
     Use a tensor to return a binary tensor which has ones at non zero elements
     and zeros at zero elements

     Parameters
     ----------
     tensor : TYPE
         DESCRIPTION.

     Returns
     -------
     TYPE
         DESCRIPTION.

     '''
     return torch.where(tensor > 0, torch.tensor(1), torch.tensor(0))

def binary_recall(pred_binary, target_binary):
    true_positives = torch.sum(pred_binary * target_binary)
    actual_positives = torch.sum(target_binary)
    return true_positives / actual_positives if actual_positives > 0 else 0.0

def binary_accuracy(pred_binary, target_binary):
    correct_predictions = torch.sum(pred_binary == target_binary)
    total_elements = pred_binary.numel()
    return correct_predictions / total_elements

def binary_f1_score(pred_binary, target_binary):
    true_positives = torch.sum(pred_binary * target_binary)
    predicted_positives = torch.sum(pred_binary)
    actual_positives = torch.sum(target_binary)
    
    precision = true_positives / predicted_positives if predicted_positives > 0 else 0.0
    recall = true_positives / actual_positives if actual_positives > 0 else 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1

def confusion_mat_sample(sample, dataset, model, device):
    input = dataset[sample][0]
    target = dataset[sample][1]
    model_who = str(model.__class__.__name__)
    if model_who == 'ConvLSTM':
        sample_list, _ = model(input.unsqueeze(0).to(device))  # create a batch of 1?
        preds = torch.cat(sample_list, dim=1).detach().cpu()[0]  # remove batch
    elif model_who == 'UNet':
        preds = model(dataset[sample][0]).to(device).detach().cpu()
    else:
        raise Exception('Need to check if statements to see if model is ' +
                        'implemented correctly')
        
    target_binary = threshold_function(target)
    pred_binary = threshold_function(preds)
    
    fet = target.shape[1]
    time_steps = target.shape[0]
    
    recall_arr = torch.zeros((fet, time_steps))
    accuracy_arr = torch.zeros((fet, time_steps))
    f1_arr = torch.zeros((fet, time_steps))
    for feature in range(fet):
        for step in range(time_steps):
            recall_arr[feature][step] = binary_recall(pred_binary[step][feature], target_binary[step][feature])
            accuracy_arr[feature][step] = binary_accuracy(pred_binary[step][feature], target_binary[step][feature])
            f1_arr[feature][step] = binary_f1_score(pred_binary[step][feature], target_binary[step][feature])
    return recall_arr, accuracy_arr, f1_arr
