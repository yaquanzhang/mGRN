import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either &quot;0&quot; or &quot;1&quot;
gpu_index = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
print(gpu_index)
import get_model as get_model
import numpy as np
import torch
import check_accuracy


def validation_single_model(result_dir, folder_identifier, X_test, Y_test, data_type):
    result_dir_root = os.path.join(result_dir, folder_identifier)
    print(result_dir_root)
    model_saving_dir = os.path.join(result_dir_root, "model")
    model_file_list = [x for x in os.listdir(model_saving_dir) if '.pth' in x]
    model_param_dict = np.load(os.path.join(model_saving_dir, 'model_param_dict.npy'), allow_pickle=True).item()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_param_dict['device'] = device
    model_list = []
    N_trial = len(model_file_list)
    for i in range(N_trial):
        model = get_model.get_model(**model_param_dict)
        model_file = 'model_weights' + str(i) + '.pth'
        model_weights = torch.load(os.path.join(model_saving_dir, model_file))
        model.load_state_dict(model_weights)
        model_list.append(model)
    accuray_result = check_accuracy.check_accuracy_multiple_model_simulation(model_list, X_test, Y_test)
    accuray_result["data_type"] = data_type
    accuray_result["model_name"] = folder_identifier
    accuray_result.to_excel(os.path.join(result_dir_root, "accuracy_" + data_type + ".xlsx"))
    #### assure files are identical
    # for pretrained models, you may include the following lines to compare with results provided.
    import pandas as pd
    pretrained_acc_df = pd.read_excel(os.path.join(result_dir_root, "pretrained_model_accuracy_" + data_type + ".xlsx"))
    assert (pretrained_acc_df['loss']-accuray_result['loss']).abs().sum() < 1e-5
    assert (pretrained_acc_df['model_name'] == accuray_result['model_name']).all()


def rolling_window_stupid(combined, window):
    axis = 0
    output_list = []
    for i in range(combined.shape[axis]-window):
        output_list.append(combined[i:(i+window),:])
    output_array = np.array(output_list)
    return output_array

def main(simulation_file_root, simulation_index, result_dir_root):
    ####### read data
    simulation_file_dir = os.path.join(simulation_file_root, simulation_index)
    simulation = np.genfromtxt(simulation_file_dir, delimiter=',')
    print('time series shape is', simulation.shape)
    return_series = simulation[:, :2].copy()
    ####### data preparation
    # swap the feature positions to group y_i with its parameters.
    temp = simulation[:, 1]
    simulation[:, 1] = simulation[:, 8]
    simulation[:, 8] = temp
    result_dir = os.path.join(result_dir_root, str(simulation_index))
    Y = return_series[:, 0] * return_series[:, 1] * 100

    if len(Y.shape) == 1:
        Y = Y.reshape((Y.shape[0], 1))
    Y = np.roll(Y, shift=-1, axis=0)
    ###### no need to change
    validation_start = 70000
    test_start = 85000
    for i in range(simulation.shape[1]):
        train_std = simulation[:validation_start, i].std()
        train_mean = simulation[:validation_start, i].mean()
        if train_std != 0:
            simulation[:, i] = (simulation[:, i] - train_mean) / train_std
        else:
            # constant
            simulation[:, i] = 0
    ## combine X and Y
    combined = np.concatenate((simulation, Y), axis=1)
    # discard the last value as there is no prediction
    combined = combined[:-1, :]
    # reshape the input matrix.
    # past_interval = 50  # number of past steps
    past_interval = 5
    # past_interval = 10
    print("number of past steps is", past_interval)
    axis = 0
    window = past_interval
    # output = rolling_window(combined, window, axis)
    output = rolling_window_stupid(combined, window)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output = torch.tensor(output).to(device)
    validation_set = output[validation_start:test_start]
    print("validation set shape is", validation_set.shape, ", (X+Y)")
    test_set = output[test_start:]
    print("test set shape is", test_set.shape, ", (X+Y)")
    n_feature = simulation.shape[1]
    ## validation
    X_validation = validation_set[:, :, 0:n_feature].float()
    X_validation = X_validation.permute(1, 0, 2)
    Y_validation = validation_set[:, -1, n_feature:].float()
    if Y_validation.shape[-1] == 1:
        Y_validation = Y_validation.view(Y_validation.shape[0])
    model_name_list = os.listdir(result_dir)
    data_type = "out_of_sample"
    # model_name_list = ['mGRN_two_groups']
    for model_name in model_name_list:
        validation_single_model(result_dir, model_name, X_validation, Y_validation, data_type)
    # test
    X_test = test_set[:, :, 0:n_feature].float()
    X_test = X_test.permute(1, 0, 2)
    Y_test = test_set[:, -1, n_feature:].float()
    if Y_test.shape[-1] == 1:
        Y_test = Y_test.view(Y_test.shape[0])
    data_type = "test"
    for model_name in model_name_list:
        validation_single_model(result_dir, model_name, X_test, Y_test, data_type)
    return None

if __name__ == '__main__':
    ### parameters
    dir_path = os.path.dirname(os.path.realpath(__file__))
    result_dir = os.path.join(dir_path, 'simulation_pretrained_models')
    ###
    dir_path = os.path.dirname(os.path.realpath(__file__))
    simulation_file_root = os.path.join(dir_path, 'simulation_data', 'data_generation_matlab', 'simulation_data')
    simulation_index_list = os.listdir(simulation_file_root)
    for simulation_index in simulation_index_list:
        main(simulation_file_root, simulation_index, result_dir)




