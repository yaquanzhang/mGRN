# train models with datasets from TapNet (Zhang et al., 2020)
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either &quot;0&quot; or &quot;1&quot;
gpu_index = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
print(gpu_index)
import uea_utils.utils as utils
import uea_utils.param_dict as param_dict
import shutil
import time
import torch
import get_model
import train_model
import pandas as pd
import numpy as np
import pathlib

def single_model(result_dir_root, model_param_dict, X_train, y_train, X_test, y_test,
                 training_param_dict, N_trial = 1):
    print(result_dir_root)
    ########################### model training
    print("training...")
    model_saving_dir = os.path.join(result_dir_root , 'model')
    if os.path.exists(model_saving_dir):
        shutil.rmtree(model_saving_dir)
    time.sleep(1)
    os.makedirs(model_saving_dir)
    # data preparation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train = torch.tensor(X_train.copy()).to(device)
    y_train = torch.tensor(y_train.copy()).to(device)
    X_test = torch.tensor(X_test.copy()).to(device)
    y_test = torch.tensor(y_test.copy()).to(device)
    model_param_dict['device'] = device
    batch_first = model_param_dict['batch_first']
    num_classes = y_train.max().cpu().numpy() + 1
    n_feature = X_train.shape[2]
    model_param_dict['num_classes'] = num_classes
    model_param_dict['n_feature'] = n_feature
    if model_param_dict['input_size_list'] == 'total_split':
        model_param_dict['input_size_list'] = [1]*n_feature
    if not batch_first:
        X_train = X_train.permute(1, 0, 2)
        X_test = X_test.permute(1, 0, 2)
    y_train = y_train.long().squeeze()
    y_test = y_test.long().squeeze()
    X_train = X_train.float()
    X_test = X_test.float()
    test_acc_list = []
    print('print(model_param_dict)', model_param_dict)
    print('print(training_param_dict)', training_param_dict)
    for i in range(N_trial):
        print(i)
        model = get_model.get_model(**model_param_dict)
        print('The number of trainable parameters is', model.param_num)
        this_model_saving_dir = os.path.join(model_saving_dir, 'model_weights' + str(i) + '.pth')
        # X_train and X_test are permuted tensors
        test_acc = train_model.train_tapnet(model, X_train, y_train, X_test, y_test,
                                                 this_model_saving_dir, batch_first, **training_param_dict)
        print('This accuracy is', test_acc)
        test_acc_list.append(test_acc)

    accuray_result = pd.DataFrame({"accuracy": test_acc_list})
    accuray_result.to_excel(os.path.join(result_dir_root, "accuracy_test.xlsx"))


if __name__ == '__main__':
    ##################### hyper parameters
    dataset = 'CharacterTrajectories'
    # dataset = 'FaceDetection'
    # dataset = 'LSST'
    # dataset = 'PenDigits'
    # dataset = 'PhonemeSpectra'
    # dataset = 'SpokenArabicDigits'
    data_root_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), 'uea_utils', 'uea_datasets') # data folder
    model_param_dict = param_dict.model_param_dict[dataset]
    training_param_dict = param_dict.training_param_dict[dataset]
    result_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), 'uea') # results are saved in this folder
    N_trial = 3  # repeat the experiments for N_trial times.
    #####################
    root_dir = os.path.join(data_root_dir, dataset)
    data_dir = os.path.join(root_dir, 'train.npz')
    data = np.load(data_dir)
    X_train = np.nan_to_num(data['arr_0'], nan=0.0)
    Y_train = data['arr_1']
    data_dir = os.path.join(root_dir, 'test.npz')
    data = np.load(data_dir)
    X_test = np.nan_to_num(data['arr_0'], nan=0.0)
    Y_test = data['arr_1']
    X_train, X_test = utils.normalization(X_train, X_test)
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
    this_result_dir = os.path.join(result_dir, dataset)
    if not os.path.isdir(this_result_dir):
        os.makedirs(this_result_dir)
    single_model(this_result_dir, model_param_dict, X_train, Y_train, X_test, Y_test, training_param_dict, N_trial)


