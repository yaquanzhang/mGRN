import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either &quot;0&quot; or &quot;1&quot;
gpu_index = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
print(gpu_index)
import get_model as get_model
import train_model
import numpy as np
import torch
import check_accuracy
import pandas as pd


def get_model_suffix(model_name, input_size_list, size_of):
    if model_name in ['LSTM', 'GRU']:
        # no suffix
        suffix = ''
    else:
        if len(input_size_list) == 2:
            suffix = '_two_groups'
        else:
            suffix = '_total_split'
        if size_of!=1:
            suffix = suffix + '_sizeof_' + str(size_of)
    return suffix


# NN main simulation regression
def single_model(result_dir, datasets, learning_rate_list, model_name, n_feature, n_rnn_units, input_size_list, size_of,
                 n_layers, batch_first, n_Y, batch_size, early_stopping_delta,
                 early_stopping_patience, num_epochs):
    # check is processed
    print(result_dir)
    model_suffix = get_model_suffix(model_name, input_size_list, size_of)
    result_dir_root = os.path.join(result_dir, model_name + model_suffix)
    N_trial = len(learning_rate_list)
    ###########################
    # model training
    model_saving_dir = os.path.join(result_dir_root, 'model')
    if not os.path.exists(model_saving_dir):
        os.makedirs(model_saving_dir)
    train_set, validation_set, test_set = datasets
    X_array_list = []
    Y_array_list = []
    for this_array in [validation_set, test_set]:
        # split
        this_X = this_array[:, :, 0:n_feature].float()
        this_X = this_X.permute(1, 0, 2)
        this_Y = this_array[:, -1, -n_Y:].float()  # regression
        if this_Y.shape[-1] == 1:
            this_Y = this_Y.view(this_Y.shape[0])
        num_classes = n_Y
        X_array_list.append(this_X)
        Y_array_list.append(this_Y)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_param_dict = {"model_name": model_name, "n_feature": n_feature, "n_rnn_units": n_rnn_units,
                        "n_layers": n_layers, "num_classes": num_classes,
                        "batch_first": batch_first, "device": device,  "input_size_list": input_size_list,
                        "size_of": size_of
                        }
    print(model_param_dict)
    np.save(os.path.join(model_saving_dir, 'model_param_dict.npy'), model_param_dict)
    learning_rate_dict = {}

    model_list = []
    validation_X = X_array_list[0]
    validation_Y = Y_array_list[0]
    for i in range(N_trial):
        print(i)
        learning_rate = learning_rate_list[i]
        learning_rate_dict[str(i)] = learning_rate
        model = get_model.get_model(**model_param_dict)
        model = train_model.train_simulation(model, train_set, validation_X, validation_Y, batch_size, n_Y,
                                                      num_epochs, batch_first, learning_rate,
                                                      early_stopping_delta, early_stopping_patience)

        torch.save(model.state_dict(), os.path.join(model_saving_dir, 'model_weights' + str(i) + '.pth'))
        model_list.append(model)

    learning_rate_df = pd.DataFrame(learning_rate_dict, index=[0])
    learning_rate_df.to_excel(os.path.join(model_saving_dir , "learning_rate.xlsx"))

    ######## validation
    for idx, data_type in enumerate(['out_of_sample', 'test']):
        accuray_result = check_accuracy.check_accuracy_multiple_model_simulation(model_list, X_array_list[idx],
                                                                                 Y_array_list[idx])
        accuray_result["data_type"] = data_type
        accuray_result["model_name"] = model_name + model_suffix
        accuray_result.to_excel(os.path.join(result_dir_root, "accuracy_" + data_type + ".xlsx"))


def rolling_window_stupid(combined, window):
    axis = 0
    output_list = []
    for i in range(combined.shape[axis]-window):
        output_list.append(combined[i:(i+window),:])
    output_array = np.array(output_list)
    return output_array


def main(simulation_file_root, simulation_index, result_dir_root, past_interval,
         learning_rate_list, model_name_list,
         n_rnn_units_list, size_of_list, input_size_list, n_layers,
         batch_size, early_stopping_delta, early_stopping_patience, num_epochs):
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
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    Y = return_series[:, 0] * return_series[:, 1] * 100

    if len(Y.shape) == 1:
        Y = Y.reshape((Y.shape[0], 1))
    Y = np.roll(Y, shift=-1, axis=0)
    #######
    validation_start = 70000
    test_start = 85000
    for i in range(simulation.shape[1]):
        train_std = simulation[:validation_start, i].std()
        train_mean = simulation[:validation_start, i].mean()
        simulation[:, i] = (simulation[:, i] - train_mean) / train_std
    ## combine X and Y
    combined = np.concatenate((simulation, Y), axis=1)
    # discard the last value as there is no prediction
    combined = combined[:-1, :]
    print("number of past steps is", past_interval)
    window = past_interval
    output = rolling_window_stupid(combined, window)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output = torch.tensor(output).to(device)
    # split data sets
    train_set = output[:validation_start]
    print("train set shape is", train_set.shape, ", (X+Y)")
    validation_set = output[validation_start:test_start]
    print("validation set shape is", validation_set.shape, ", (X+Y)")
    test_set = output[test_start:]
    print("validation set shape is", test_set.shape, ", (X+Y)")
    n_feature = simulation.shape[1]
    n_Y = Y.shape[1]
    datasets = (train_set, validation_set, test_set)
    batch_first = False
    n_model = len(model_name_list)
    for i in range(n_model):
        single_model(result_dir, datasets, learning_rate_list,
                     model_name_list[i], n_feature, n_rnn_units_list[i], input_size_list[i], size_of_list[i],
                     n_layers, batch_first, n_Y, batch_size, early_stopping_delta,
                     early_stopping_patience, num_epochs)
    return None


if __name__ == '__main__':
    ### parameters
    model_name_list = ['LSTM', 'GRU'] + ['ChannelwiseLSTM'] * 8 + ['mGRN'] * 8
    # dimension of marginal components. In the case of LSTM and GRU, this is the unit dimension.
    n_rnn_units_list = [14, 17, 5, 4, 3, 2, 3, 2, 2, 2, 10, 8, 6, 3, 4, 4, 3, 2]
    size_of_list = [None] * 2 + [1, 2, 4, 8]*4
    n_feature = 16
    half_n_feature = int(n_feature / 2)
    input_size_list = [None] * 2 + [[half_n_feature, half_n_feature]] * 4 + [[1] * n_feature] * 4 + [
        [half_n_feature, half_n_feature]] * 4 + [[1] * n_feature] * 4
    learning_rate_list = [1e-3, 5e-4, 1e-4]
    batch_size = 128
    early_stopping_delta = 0.00005  # relative sense
    early_stopping_patience = 10
    num_epochs = 300  # max number of epochs to run.
    n_layers = 1
    past_interval = 5 # past 5 steps are included as inputs.
    ###
    dir_path = os.path.dirname(os.path.realpath(__file__))
    simulation_file_root = os.path.join(dir_path, 'simulation_data', 'data_generation_matlab',
                                        'simulation_data')
    simulation_index_list = os.listdir(simulation_file_root)
    _, simulation_file_name = os.path.split(simulation_file_root)
    result_dir_root = os.path.join(dir_path, 'simulation')
    if not os.path.exists(result_dir_root):
        os.makedirs(result_dir_root)
    for simulation_index in simulation_index_list[0:1]:
        main(simulation_file_root, simulation_index, result_dir_root, past_interval,
             learning_rate_list, model_name_list,
         n_rnn_units_list, size_of_list, input_size_list,n_layers,
         batch_size, early_stopping_delta, early_stopping_patience, num_epochs)


